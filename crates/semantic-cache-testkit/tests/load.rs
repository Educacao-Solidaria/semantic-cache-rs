//! Carga: 50 000 consultas em voo ao mesmo tempo, multiplexadas por
//! `request_id` em 8 conexões (o contrato permite pipeline; 50 000 conexões
//! estourariam o limite de descritores). Cada conexão escreve todos os seus
//! pedidos de uma vez e fecha a escrita, sem esperar resposta, enquanto lê; o
//! servidor responde tudo o que recebeu e fecha. Cada resposta é conferida
//! pelo `request_id`: o texto do vizinho certo no hit, nada no miss, nenhuma
//! faltando, repetida ou a mais. No fim, nenhum pânico nem frame descartado, o
//! servidor atende uma conexão nova e os descritores voltam ao que eram. Um
//! teste só neste binário: o hook de pânico, o contador de descartados e
//! `/proc/self/fd` valem para o processo inteiro.
#![cfg(target_os = "linux")]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use semantic_cache_ipc::frame::{discarded_frames, encode, kind, read_frame};
use semantic_cache_ipc::server::{bind_owned, serve, Limits};
use semantic_cache_rs::Vector;
use semantic_cache_testkit::fixtures::VectorGen;
use semantic_cache_testkit::ipc::{LookupReq, LookupResp, SearchCache};
use semantic_cache_testkit::stress::{FlatI8, VectorIndex};
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::task::JoinSet;
use tokio::time::timeout;

const QUERIES: usize = 50_000;
const CONNECTIONS: usize = 8;
const VECTORS: usize = 256;
const DIM: usize = 64;
/// Teto da carga inteira: binário de debug num runner lento.
const DEADLINE: Duration = Duration::from_secs(120);
/// Prazo de quem espera o servidor responder ou fechar.
const PATIENCE: Duration = Duration::from_secs(10);

static PANICKED: AtomicBool = AtomicBool::new(false);

fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

/// Memória residente do processo (cliente e servidor) agora, em MiB. O
/// `VmHWM` não serve para o pico: o kernel o atualiza tarde e perde picos.
fn rss_mib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmRSS:")).unwrap();
    let kib: u64 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kib / 1024
}

/// Espera os descritores voltarem a `base`: cada task do servidor fecha o seu
/// socket ao terminar.
async fn settled(base: usize) {
    let started = Instant::now();
    while open_fds() != base {
        let open = open_fds();
        assert!(
            started.elapsed() < PATIENCE,
            "{open} descritores; eram {base}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Os pedidos de uma conexão, já em frames, e o texto que cada `request_id`
/// espera (`None` = miss).
#[derive(Default)]
struct Batch {
    bytes: Vec<u8>,
    expected: HashMap<u32, Option<String>>,
}

/// Metade hit (cosseno 0,98 com um vetor guardado), metade miss (sorteado),
/// repartidos em `CONNECTIONS` lotes com `request_id` de 1 em diante em cada um.
fn batches(gen: &mut VectorGen, vectors: &[Vector], texts: &[String]) -> Vec<Batch> {
    let mut batches: Vec<Batch> = (0..CONNECTIONS).map(|_| Batch::default()).collect();
    for q in 0..QUERIES {
        let target = q % VECTORS;
        let (query, want) = if q % 2 == 0 {
            let near = gen.with_cosine(&vectors[target], 0.98);
            (near, Some(texts[target].clone()))
        } else {
            (gen.unit(DIM), None)
        };
        let (prompt, threshold) = (format!("consulta {q}"), 0.9);
        let req = LookupReq {
            prompt,
            embedding: query.data,
            threshold,
        };
        let id = u32::try_from(q / CONNECTIONS + 1).unwrap();
        let batch = &mut batches[q % CONNECTIONS];
        let payload = serde_json::to_vec(&req).unwrap();
        batch
            .bytes
            .extend(encode(kind::LOOKUP_REQ, 0, id, &payload).unwrap());
        batch.expected.insert(id, want);
    }
    batches
}

/// Uma conexão: escreve o lote inteiro e fecha a escrita, sem esperar
/// resposta, enquanto lê; depois da última resposta, o servidor fecha.
async fn run(path: PathBuf, batch: Batch) {
    let (rd, mut wr) = UnixStream::connect(&path).await.unwrap().into_split();
    let bytes = batch.bytes;
    let writer = tokio::spawn(async move {
        wr.write_all(&bytes).await.unwrap();
        wr.shutdown().await.unwrap();
    });
    let mut rd = BufReader::new(rd);
    let mut pending = batch.expected;
    while !pending.is_empty() {
        let left = pending.len();
        let frame = match read_frame(&mut rd).await {
            Ok(Some(frame)) => frame,
            Ok(None) => panic!("o servidor fechou com {left} sem resposta"),
            Err(err) => panic!("a conexão caiu com {left} sem resposta: {err}"),
        };
        let id = frame.request_id;
        let body = || String::from_utf8_lossy(&frame.payload).into_owned();
        assert_eq!(frame.kind, kind::LOOKUP_RESP, "request_id {id}: {}", body());
        let want = pending.remove(&id);
        let want = want.unwrap_or_else(|| panic!("request_id {id} repetido ou desconhecido"));
        let resp: LookupResp = serde_json::from_slice(&frame.payload).unwrap();
        let got = (resp.hit, resp.cached_response);
        assert_eq!(got, (want.is_some(), want), "request_id {id}");
    }
    let rest = timeout(PATIENCE, read_frame(&mut rd)).await;
    let rest = rest.expect("o servidor não fechou depois da última resposta");
    assert!(rest.unwrap().is_none(), "resposta a mais");
    writer.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fifty_thousand_queries_in_flight_over_eight_connections() {
    std::panic::set_hook(Box::new(|info| {
        PANICKED.store(true, Ordering::SeqCst);
        eprintln!("{info}");
    }));
    let mut gen = VectorGen::new(0x10AD);
    let vectors: Vec<_> = (0..VECTORS).map(|_| gen.unit(DIM)).collect();
    let texts: Vec<String> = (0..VECTORS).map(|id| format!("resposta {id}")).collect();
    let batches = batches(&mut gen, &vectors, &texts);
    let mut index = FlatI8::default();
    for vector in &vectors {
        index.insert(&vector.data);
    }
    let path = std::env::temp_dir().join(format!("scload-{}.sock", std::process::id()));
    let (listener, _socket) = bind_owned(&path).unwrap();
    // Os tetos padrão do `[ipc]` do servidor: 256 conexões, 5 min de ociosidade.
    let limits = Limits {
        max_connections: 256,
        idle: Duration::from_secs(300),
        drain: Duration::from_secs(5),
    };
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let cache = SearchCache::new(index, DIM, texts);
    let server = tokio::spawn(serve(listener, cache, stopped, limits));
    let base = open_fds();
    // Já inclui os lotes montados (~42 MB de frames).
    let before = rss_mib();
    let started = Instant::now();
    let mut clients = JoinSet::new();
    for batch in batches {
        clients.spawn(run(path.clone(), batch));
    }
    // Recolhe os clientes e amostra a memória a cada 5 ms enquanto isso.
    let all = async {
        let mut peak = before;
        loop {
            tokio::select! {
                done = clients.join_next() => match done {
                    Some(done) => done.unwrap(),
                    None => break peak,
                },
                () = tokio::time::sleep(Duration::from_millis(5)) => peak = peak.max(rss_mib()),
            }
        }
    };
    let peak = timeout(DEADLINE, all)
        .await
        .expect("a carga passou do prazo");
    let took = started.elapsed();
    #[allow(clippy::cast_precision_loss)] // 50 000 cabe exato no f64
    let rate = QUERIES as f64 / took.as_secs_f64();
    println!("{QUERIES} consultas em {CONNECTIONS} conexões: {took:.2?}, {rate:.0}/s, RSS {before} MiB antes e no máximo {peak} MiB durante");
    // O servidor segue de pé: uma conexão nova é atendida.
    let mut conn = UnixStream::connect(&path).await.unwrap();
    conn.write_all(&encode(kind::PING, 0, 1, b"").unwrap())
        .await
        .unwrap();
    let pong = timeout(PATIENCE, read_frame(&mut conn)).await.unwrap();
    assert_eq!(pong.unwrap().unwrap().kind, kind::PONG);
    drop(conn);
    settled(base).await;
    stop.send(()).unwrap();
    timeout(PATIENCE, server).await.unwrap().unwrap();
    assert_eq!(discarded_frames(), 0, "frame recusado por um dos lados");
    assert!(!PANICKED.load(Ordering::SeqCst), "algum pânico (stderr)");
}
