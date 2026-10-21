//! Soak de memória: o `serve` de produção do IPC atende, por um socket Unix de
//! verdade, um cache em miniatura montado com as peças do servidor (orçamento
//! com despejo FIFO e um retrato que as buscas leem sem trava) sob carga
//! vetorial contínua: clientes abrem e fecham conexões o tempo todo, gravam
//! embeddings de 1536 dimensões (STORE), consultam o que gravaram (hit, com o
//! texto certo) e um vetor aleatório (miss), pela busca SIMD. O teto de 32 MiB
//! faz o despejo rodar sem parar; o texto do `/metrics` sai a cada amostra.
//!
//! Vazamento é: depois do aquecimento (o 1º quarto, em que o cache enche e
//! despeja), o pico do heap fora do orçamento (`allocated` do jemalloc menos o
//! reservado) ou do `VmRSS` no último quarto passar do pico do 2º mais a folga;
//! ou, no fim, conexões e descritores não voltarem à base, o orçamento não
//! voltar a zero sem o cache, ou o heap não voltar ao que era antes.
//!
//! `#[ignore]`: roda à parte, em release, por `SEMCACHE_SOAK_SECS` segundos
//! (padrão 120, mínimo 20); `SEMCACHE_SOAK_CSV=<arquivo>` grava as amostras.
//! O workflow `soak` faz 3 min em todo PR e até 330 min sob demanda (o job
//! hospedado do GitHub para em 6 h). As 12 h rodam em máquina própria ou
//! runner dedicado:
//!
//! ```text
//! SEMCACHE_SOAK_SECS=43200 SEMCACHE_SOAK_CSV=soak.csv cargo test --release --locked --test soak -- --ignored --nocapture
//! ```
#![cfg(target_os = "linux")]

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

use semantic_cache_ipc::frame::{code, encode, error_payload, kind, read_frame, Frame};
use semantic_cache_ipc::server::{bind_owned, serve, Conn, Handler, Stop};
use semantic_cache_server::alloc;
use semantic_cache_server::config::{IpcConfig, MemoryConfig};
use semantic_cache_server::memory::{Evictor, MemoryBudget, Reservation};
use semantic_cache_server::metrics::render;
use semantic_cache_server::snapshot::SnapshotCell;
use semantic_cache_simd::similarity::dot;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::task::JoinSet;
use tokio::time::timeout;

/// Embeddings do tamanho dos de produção (`text-embedding-3-small`).
const DIM: usize = 1536;
/// ~3 mil entradas de ~10 KiB; cada despejo leva 10% delas.
const BUDGET: usize = 32 << 20;
/// A marca d'água baixa padrão (0,8): onde o despejo deixa o índice.
const LOW: usize = BUDGET / 5 * 4;
const CLIENTS: u64 = 4;
/// Rodadas (STORE + 2 LOOKUP) por conexão, antes de fechá-la e abrir outra.
const ROUNDS: u32 = 10;
/// Oscilação entre os quartos que não é vazamento (pedidos em voo, retratos
/// velhos ainda em leitura, páginas que o jemalloc devolve com atraso).
const HEAP_SLACK: usize = 2 << 20;
const RSS_SLACK: usize = 16 << 20;
/// Prazo de cada resposta, e do que tem de fechar no fim.
const PATIENCE: Duration = Duration::from_secs(10);

/// O que o servidor lê do `LOOKUP_REQ` e do `STORE_REQ`; o resto é ignorado.
#[derive(Deserialize)]
struct Req {
    embedding: Vec<f32>,
    #[serde(default)]
    threshold: f32,
    response: Option<String>,
}

/// O STORE como chegou e a reserva que paga por ele: os bytes voltam ao
/// orçamento quando a última referência cai (despejada e fora dos retratos).
struct Entry {
    req: Req,
    paid: Reservation,
}

/// As entradas da mais velha à mais nova (o despejo é FIFO), num retrato que
/// as buscas leem sem trava.
struct Cache {
    budget: MemoryBudget,
    entries: SnapshotCell<VecDeque<Arc<Entry>>>,
    evicted: AtomicUsize,
}

impl Evictor for Cache {
    /// FIFO até as entradas do índice caberem na marca baixa. O `bytes` pedido
    /// conta também as já despejadas que um retrato em leitura segura: despejar
    /// por ele, a cada reserva, esvaziava o índice quando a leitura demorava.
    fn evict(&self, _bytes: usize) {
        self.entries.update(|old| {
            let mut next = old.clone();
            let mut live: usize = next.iter().map(|e| e.paid.bytes()).sum();
            while live > LOW {
                let Some(entry) = next.pop_front() else { break };
                live -= entry.paid.bytes();
                self.evicted.fetch_add(1, Relaxed);
            }
            next
        });
    }
}

impl Cache {
    /// Reserva o que a entrada ocupa de fato (o `Vec` do serde cresce
    /// dobrando: 2048 floats) e publica; recusa o STORE sem texto e o que não
    /// cabe no orçamento (cheio, com o despejo de outra thread em curso).
    fn store(&self, req: Req) -> Result<(u8, Vec<u8>), &'static str> {
        let text = req.response.as_ref().ok_or(code::BAD_REQUEST)?;
        let bytes = 4 * req.embedding.capacity() + text.capacity();
        let paid = self.budget.try_reserve(bytes);
        let paid = paid.map_err(|_| code::OVERLOADED)?;
        let entry = Arc::new(Entry { req, paid });
        self.entries.update(|old| {
            let mut next = old.clone();
            next.push_back(entry);
            next
        });
        Ok((kind::STORE_RESP, b"{}".to_vec()))
    }

    /// O vizinho mais próximo pelo kernel SIMD ativo (os vetores chegam
    /// unitários: o produto escalar é o cosseno).
    fn lookup(&self, query: &Req) -> (u8, Vec<u8>) {
        let body = self.entries.read(|entries| {
            let cosine = |e: &Entry| dot(&e.req.embedding, &query.embedding).unwrap_or(f32::MIN);
            let scored = entries.iter().map(|e| (cosine(e), e));
            match scored.max_by(|a, b| a.0.total_cmp(&b.0)) {
                Some((score, e)) if score >= query.threshold => {
                    json!({"hit": true, "score": score, "cached_response": e.req.response})
                }
                best => json!({"hit": false, "score": best.map_or(0.0, |b| b.0)}),
            }
        });
        (kind::LOOKUP_RESP, body.to_string().into_bytes())
    }

    /// A resposta a `frame`: LOOKUP e STORE na dimensão certa; o resto, ERROR.
    fn answer(&self, frame: &Frame) -> Vec<u8> {
        let req = serde_json::from_slice::<Req>(&frame.payload);
        let reply = match (frame.kind, req.ok().filter(|r| r.embedding.len() == DIM)) {
            (kind::LOOKUP_REQ, Some(req)) => Ok(self.lookup(&req)),
            (kind::STORE_REQ, Some(req)) => self.store(req),
            _ => Err(code::BAD_REQUEST),
        };
        let refused = |code| (kind::ERROR, error_payload(code, "pedido recusado"));
        let (k, payload) = reply.unwrap_or_else(refused);
        encode(k, 0, frame.request_id, &payload).expect("resposta cabe no frame")
    }
}

/// O handler do `serve`: o mesmo `Arc` que o orçamento chama para despejar.
struct Server(Arc<Cache>);

impl Handler for Server {
    /// Um pedido por vez, como o `Basic`; frame inválido ou a parada fecham.
    async fn handle<S: Conn>(&self, mut conn: S, mut stop: Stop) {
        loop {
            let read = tokio::select! {
                biased;
                _ = stop.wait_for(|&s| s) => return,
                read = read_frame(&mut conn) => read,
            };
            let Ok(Some(frame)) = read else { return };
            if conn.write_all(&self.0.answer(&frame)).await.is_err() {
                return;
            }
        }
    }
}

/// `SplitMix64`: embeddings e textos determinísticos por cliente.
struct Rng(u64);

impl Rng {
    fn bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Um vetor unitário de `DIM` componentes já em JSON, com o `f32` no
    /// decimal mais curto (como o gateway manda).
    #[allow(clippy::cast_precision_loss)] // 24 bits cabem no f32
    fn embedding(&mut self) -> String {
        let v: Vec<f32> = (0..DIM).map(|_| (self.bits() >> 40) as f32).collect();
        let v: Vec<f32> = v.iter().map(|x| x / 8_388_608.0 - 1.0).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = v.iter().map(|x| x / norm).collect();
        serde_json::to_string(&unit).expect("f32 finito")
    }
}

/// O que os clientes fizeram, somado.
#[derive(Default)]
struct Tally {
    rounds: AtomicUsize,
    refused: AtomicUsize,
    connections: AtomicUsize,
}

/// Um pedido e a resposta, com prazo: servidor parado falha em vez de travar.
async fn call(conn: &mut UnixStream, kind: u8, id: u32, body: &str) -> Frame {
    let frame = encode(kind, 0, id, body.as_bytes()).expect("pedido cabe no frame");
    let exchange = async {
        conn.write_all(&frame).await.expect("escrever o pedido");
        let reply = read_frame(conn).await.expect("frame válido");
        reply.expect("resposta antes do EOF")
    };
    let reply = timeout(PATIENCE, exchange).await;
    let reply = reply.expect("resposta no prazo");
    assert_eq!(reply.request_id, id);
    reply
}

async fn lookup(conn: &mut UnixStream, id: u32, embedding: &str) -> Value {
    let body = format!(r#"{{"prompt":"soak","embedding":{embedding},"threshold":0.9}}"#);
    let reply = call(conn, kind::LOOKUP_REQ, id, &body).await;
    assert_eq!(reply.kind, kind::LOOKUP_RESP, "LOOKUP respondido");
    serde_json::from_slice(&reply.payload).expect("LOOKUP_RESP em JSON")
}

/// Até `until`: abre uma conexão, faz `ROUNDS` rodadas — grava um embedding
/// novo, o consulta (hit, com o texto dele) e consulta um aleatório (miss) —,
/// fecha e abre outra. O texto só tem dígitos, `:` e `r`: vai no JSON sem escape.
async fn client(path: PathBuf, seed: u64, until: Instant, tally: Arc<Tally>) {
    let (mut rng, mut n) = (Rng(seed), 0);
    while Instant::now() < until {
        let mut conn = UnixStream::connect(&path).await.expect("conectar");
        tally.connections.fetch_add(1, Relaxed);
        // O `request_id` 0 é o da recusa de conexão (`overloaded` do `serve`).
        for id in 1..=ROUNDS {
            n += 1;
            let len = 256 + usize::try_from(rng.bits() % 4096).expect("< 4096");
            let (text, stored) = (format!("{seed}:{n}:{}", "r".repeat(len)), rng.embedding());
            let store = format!(
                r#"{{"prompt":"soak","response":"{text}","embedding":{stored},"ttl_seconds":0}}"#
            );
            let reply = call(&mut conn, kind::STORE_REQ, id, &store).await;
            if reply.kind == kind::ERROR {
                let error: Value = serde_json::from_slice(&reply.payload).expect("ERROR em JSON");
                assert_eq!(error["code"], code::OVERLOADED); // só o orçamento cheio recusa
                tally.refused.fetch_add(1, Relaxed);
                continue;
            }
            assert_eq!(reply.kind, kind::STORE_RESP);
            let hit = lookup(&mut conn, id, &stored).await;
            assert_eq!(hit["cached_response"], text, "texto do hit");
            let miss = lookup(&mut conn, id, &rng.embedding()).await;
            assert_eq!(miss["hit"], false, "aleatório não tem vizinho a 0,9");
            tally.rounds.fetch_add(1, Relaxed);
        }
    }
}

/// Uma amostra: segundos desde a partida e as medidas, em bytes.
struct Sample {
    secs: f64,
    /// Heap fora do orçamento: o `allocated` do jemalloc menos o reservado.
    heap: usize,
    rss: usize,
    evicted: usize,
}

fn allocated() -> usize {
    alloc::stats().expect("jemalloc no Linux").allocated
}

fn rss() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    let line = status.lines().find_map(|l| l.strip_prefix("VmRSS:"));
    let kib = line.and_then(|l| l.trim().strip_suffix(" kB"));
    let kib: usize = kib.expect("VmRSS em kB").parse().expect("VmRSS numérico");
    kib << 10
}

fn open_fds() -> usize {
    let fds = std::fs::read_dir("/proc/self/fd").expect("/proc/self/fd");
    fds.count()
}

/// Se `ok()` passa a valer em até `PATIENCE` (sockets e threads fecham depois do `drop`).
fn eventually(ok: impl Fn() -> bool) -> bool {
    let started = Instant::now();
    while !ok() && started.elapsed() < PATIENCE {
        std::thread::sleep(Duration::from_millis(50));
    }
    ok()
}

/// Quanto o pico do último quarto passou do pico do 2º (o 1º é o aquecimento)
/// numa série de `(segundos, bytes)`; `None` se um dos dois ficou sem ponto.
fn growth(series: &[(f64, usize)]) -> Option<usize> {
    let end = series.last()?.0;
    let peak = |from: f64, to: f64| {
        let inside = series.iter().filter(|p| (from..to).contains(&(p.0 / end)));
        inside.map(|p| p.1).max()
    };
    Some(peak(0.75, f64::INFINITY)?.saturating_sub(peak(0.25, 0.5)?))
}

/// A carga por `duration`, amostrando a memória; o pânico de um cliente
/// encerra o soak na hora, com a mensagem dele.
async fn run(duration: Duration, cache: &Arc<Cache>, tally: &Arc<Tally>) -> Vec<Sample> {
    let mut csv = std::env::var_os("SEMCACHE_SOAK_CSV").map(|p| File::create(p).expect("CSV"));
    // Caminho curto: o do scratchpad passa do SUN_LEN.
    let path = PathBuf::from(format!("/tmp/scsoak-{}.sock", std::process::id()));
    let (listener, _socket) = bind_owned(&path).expect("socket do soak");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let limits = IpcConfig::default().limits();
    let server = tokio::spawn(serve(listener, Server(Arc::clone(cache)), stopped, limits));
    let (started, mut clients) = (Instant::now(), JoinSet::new());
    for seed in 0..CLIENTS {
        let until = started + duration;
        clients.spawn(client(path.clone(), seed, until, Arc::clone(tally)));
    }
    let every = (duration / 400).clamp(Duration::from_secs(1), Duration::from_secs(60));
    let mut samples = Vec::with_capacity(1024);
    loop {
        drop(render(&cache.budget, None)); // o scrape do Prometheus
        let s = Sample {
            secs: started.elapsed().as_secs_f64(),
            heap: allocated().saturating_sub(cache.budget.used()),
            rss: rss(),
            evicted: cache.evicted.load(Relaxed),
        };
        if let Some(csv) = &mut csv {
            writeln!(csv, "{:.1},{},{},{}", s.secs, s.heap, s.rss, s.evicted).expect("CSV");
        }
        samples.push(s);
        if started.elapsed() >= duration {
            break;
        }
        tokio::select! {
            () = tokio::time::sleep(every) => {}
            Some(Err(panic)) = clients.join_next() => std::panic::resume_unwind(panic.into_panic()),
        }
    }
    while let Some(done) = clients.join_next().await {
        done.unwrap_or_else(|panic| std::panic::resume_unwind(panic.into_panic()));
    }
    drop(stop);
    server.await.expect("serve");
    samples
}

#[test]
#[ignore = "soak: em release, com SEMCACHE_SOAK_SECS (ver o topo do arquivo)"]
fn memory_stays_flat_under_continuous_vector_load() {
    let secs = std::env::var("SEMCACHE_SOAK_SECS").map_or(120, |v| {
        v.parse::<u64>().expect("SEMCACHE_SOAK_SECS: inteiro")
    });
    assert!(secs >= 20, "SEMCACHE_SOAK_SECS: no mínimo 20");
    let base_heap = allocated();
    let config = MemoryConfig {
        max_bytes: BUDGET,
        ..MemoryConfig::default()
    };
    let budget = MemoryBudget::new(&config).unwrap();
    let cache = Arc::new(Cache {
        budget: budget.clone(),
        entries: SnapshotCell::new(VecDeque::new()),
        evicted: AtomicUsize::new(0),
    });
    assert!(budget.set_evictor(&cache));
    let tally = Arc::new(Tally::default());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    // Com o runtime de pé: os descritores dele, e o par de sockets que o
    // `tokio::signal` abre para o processo inteiro, ficam na base.
    let base_fds = open_fds();
    let samples = runtime.block_on(run(Duration::from_secs(secs), &cache, &tally));
    let fds_back = eventually(|| open_fds() == base_fds);
    drop(runtime);
    let connections = semantic_cache_ipc::metrics::connections();
    let evicted = cache.evicted.load(Relaxed);
    drop(cache);
    let budget_left = budget.used();
    let heap_back = eventually(|| allocated() <= base_heap + HEAP_SLACK);
    let left_kib = allocated().saturating_sub(base_heap) >> 10;
    let heap: Vec<_> = samples.iter().map(|s| (s.secs, s.heap)).collect();
    let rss: Vec<_> = samples.iter().map(|s| (s.secs, s.rss)).collect();
    let (heap, rss) = (
        growth(&heap).expect("amostras"),
        growth(&rss).expect("amostras"),
    );
    let end = samples[samples.len() - 1].secs;
    let warmed = samples.iter().find(|s| s.secs >= end / 4.0);
    let warmed = warmed.is_some_and(|s| s.evicted > 0);

    let (rounds, refused) = (tally.rounds.load(Relaxed), tally.refused.load(Relaxed));
    let opened = tally.connections.load(Relaxed);
    let (heap_kib, rss_kib) = (heap >> 10, rss >> 10);
    println!("soak: - {secs} s: {rounds} rodadas (STORE + 2 LOOKUP), {refused} STORE recusados, {opened} conexões, {evicted} despejos");
    println!("soak: - do 2º ao último quarto: heap fora do orçamento +{heap_kib} KiB, residente +{rss_kib} KiB");
    println!("soak: - no fim: orçamento {budget_left} B, heap +{left_kib} KiB sobre o de antes, {connections} conexões");
    assert!(warmed, "o cache não encheu no 1º quarto");
    // Um orçamento que recusasse quase tudo mediria um cache parado.
    let busy = rounds > 0 && refused * 100 <= rounds;
    assert!(busy, "{refused} STORE recusados em {rounds} rodadas");
    assert!(heap <= HEAP_SLACK, "heap fora do orçamento +{heap} B");
    assert!(rss <= RSS_SLACK, "memória residente cresceu {rss} B");
    assert_eq!(connections, 0, "conexões abertas depois do serve");
    assert!(fds_back, "{} descritores; eram {base_fds}", open_fds());
    assert_eq!(budget_left, 0, "solto o cache, o orçamento volta a zero");
    assert!(heap_back, "heap {left_kib} KiB acima do de antes");
}

#[test]
#[allow(clippy::cast_precision_loss)] // segundos pequenos
fn the_criterion_ignores_the_warm_up_and_the_sawtooth_but_not_a_slope() {
    let series = |at: fn(usize) -> usize| -> Vec<(f64, usize)> {
        (0..=100).map(|i| (i as f64, at(i))).collect()
    };
    // Sobe no aquecimento e para: não é vazamento.
    assert_eq!(growth(&series(|i| i.min(20) << 20)), Some(0));
    // A serra do despejo oscila sem tendência.
    assert_eq!(growth(&series(|i| (64 << 20) + ((i % 10) << 16))), Some(0));
    // Vazamento de 1 MiB/s: do pico do 2º quarto (49 s) ao do último (100 s).
    assert_eq!(growth(&series(|i| i << 20)), Some(51 << 20));
    // Sem ponto num dos dois quartos não há veredito.
    assert_eq!(growth(&series(|i| i)[..2]), None);
    assert_eq!(growth(&[]), None);
}
