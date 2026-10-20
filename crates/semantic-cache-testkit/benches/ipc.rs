//! IPC v1 de ponta a ponta contra a meta de 1,5 ms no p99: `LOOKUP_REQ` em
//! JSON → frame → socket Unix → `serve` de produção → servidor de referência
//! (`semantic_cache_testkit::ipc`, busca com o kernel SIMD ativo) → resposta
//! → decode do `LOOKUP_RESP`, tudo no cronômetro, uma consulta por vez, por
//! tamanho do conjunto. Toda consulta tem cosseno 0,98 com um vetor guardado
//! e a resposta é conferida (o texto do id certo).
//!
//! - `cargo bench --bench ipc`: de 0 a 100 000 vetores de 1536 dimensões.
//! - `SEMCACHE_BENCH_QUICK=1 cargo bench --bench ipc`: até 10 000.
//! - Sob `cargo test` (sem `--bench`): 200 × 64, para o harness não apodrecer.
//!
//! "No servidor" é o `latency_micros` da resposta: do payload recebido à
//! resposta montada (parse do JSON + busca), sem o socket, o frame e a
//! serialização da resposta.

use std::time::{Duration, Instant};

use semantic_cache_ipc::frame::{encode, kind, read_frame};
use semantic_cache_ipc::server::{bind_owned, serve, Limits};
use semantic_cache_simd::Kernel;
use semantic_cache_testkit::bench::Precision;
use semantic_cache_testkit::fixtures::VectorGen;
use semantic_cache_testkit::ipc::{LookupReq, LookupResp, SearchCache};
use semantic_cache_testkit::stress::{FlatF32, FlatI8, Latency, VectorIndex};
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::runtime::{Builder, Runtime};

#[cfg(not(target_env = "msvc"))] // o alocador do servidor (ver o Cargo.toml)
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const TARGET: Duration = Duration::from_micros(1500);
const WARMUP: usize = 20;
const FULL: &[usize] = &[0, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000];
const PROMPT: &str = "Explique em poucas linhas como o gateway decide usar o cache semântico.";

/// O texto guardado para o vetor `id`: 1 KiB com acento e quebra de linha
/// (o JSON escapa), começando pelo id para conferir a resposta.
fn text(id: usize) -> String {
    let mut text = format!("{id:08}|");
    while text.len() < 1024 {
        text.push_str("Resposta cacheada: ação, opção e cão.\n");
    }
    text
}

fn main() {
    let measuring = std::env::args().any(|arg| arg == "--bench");
    let quick = Precision::from_env() == Precision::QUICK;
    let (sizes, dim, queries): (&[usize], usize, usize) = match (measuring, quick) {
        (false, _) => (&[0, 200], 64, 20),
        (true, true) => (&[0, 1_000, 5_000, 10_000], 1536, 300),
        (true, false) => (FULL, 1536, 1_000),
    };
    let bench = Bench {
        server: Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap(),
        client: Builder::new_current_thread().enable_all().build().unwrap(),
        dim,
        queries,
    };
    println!(
        "{queries} consultas por linha, {dim} dimensões, resposta de {} bytes, kernel {}\n",
        text(0).len(),
        Kernel::active()
    );
    println!("| índice | vetores | ponta a ponta p50 / p99 / máx | no servidor p50 / p99 | p99 < 1,5 ms |");
    println!("|---|---|---|---|---|");
    for &n in sizes {
        let f32_row = bench.measure(FlatF32::with_capacity(n, dim), n);
        let i8_row = bench.measure(FlatI8::default(), n);
        for (name, (e2e, inside)) in [("flat_f32", f32_row), ("flat_i8", i8_row)] {
            let [p50, p99, max, in50, in99] =
                [e2e.p50, e2e.p99, e2e.max, inside.p50, inside.p99].map(ms);
            let verdict = if e2e.p99 < TARGET { "sim" } else { "**não**" };
            println!("| {name} | {n} | {p50} / {p99} / {max} | {in50} / {in99} | {verdict} |");
        }
    }
}

/// Os dois runtimes (o servidor em 2 workers, o cliente no thread principal)
/// e o formato da carga.
struct Bench {
    server: Runtime,
    client: Runtime,
    dim: usize,
    queries: usize,
}

impl Bench {
    /// Sobe o servidor com `n` vetores em `index`, mede as consultas, uma por
    /// vez numa conexão (as `WARMUP` primeiras não contam), e o derruba:
    /// `(ponta a ponta, no servidor)`. Os dois índices recebem os mesmos
    /// vetores e consultas (mesma semente), um de cada vez na memória.
    fn measure<I: VectorIndex + Send + Sync + 'static>(
        &self,
        mut index: I,
        n: usize,
    ) -> (Latency, Latency) {
        let (dim, queries) = (self.dim, self.queries);
        let mut gen = VectorGen::new(0x5EED);
        let targets: Vec<usize> = (0..queries).map(|q| q * n / queries).collect();
        let mut bases = Vec::with_capacity(queries);
        for id in 0..n {
            let vector = gen.unit(dim);
            index.insert(&vector.data);
            while bases.len() < queries && targets[bases.len()] == id {
                bases.push(vector.clone());
            }
        }
        // Sem vetores, consultas sorteadas: o piso do IPC + JSON, todas miss.
        let requests: Vec<LookupReq> = (0..queries)
            .map(|q| {
                let query = match bases.get(q) {
                    Some(base) => gen.with_cosine(base, 0.98),
                    None => gen.unit(dim),
                };
                let (prompt, threshold) = (PROMPT.to_owned(), 0.9);
                LookupReq {
                    prompt,
                    embedding: query.data,
                    threshold,
                }
            })
            .collect();
        let path = std::env::temp_dir().join(format!("scbench-{}.sock", std::process::id()));
        let cache = SearchCache::new(index, dim, (0..n).map(text).collect());
        let (max_connections, idle, drain) = (4, Duration::ZERO, Duration::from_secs(1));
        let limits = Limits {
            max_connections,
            idle,
            drain,
        };
        // O `_socket` remove o arquivo no fim, mesmo com pânico no meio.
        let (listener, _socket) = {
            let _runtime = self.server.enter();
            bind_owned(&path).expect("socket do bench")
        };
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let serving = self.server.spawn(serve(listener, cache, stopped, limits));
        let measured = self.client.block_on(async {
            let mut conn = UnixStream::connect(&path).await.unwrap();
            let (mut e2e, mut inside) = (Vec::new(), Vec::new());
            for round in 0..WARMUP + queries {
                let (q, id) = (round % queries, u32::try_from(round + 1).unwrap());
                let start = Instant::now();
                let payload = serde_json::to_vec(&requests[q]).unwrap();
                let frame = encode(kind::LOOKUP_REQ, 0, id, &payload).unwrap();
                conn.write_all(&frame).await.unwrap();
                let reply = read_frame(&mut conn).await.unwrap().unwrap();
                let resp = serde_json::from_slice::<LookupResp>(&reply.payload);
                let took = start.elapsed();
                let body = String::from_utf8_lossy(&reply.payload);
                let resp = resp.unwrap_or_else(|_| panic!("consulta {q}: {body}"));
                assert_eq!((reply.kind, reply.request_id), (kind::LOOKUP_RESP, id));
                assert_eq!(
                    resp.cached_response,
                    (n > 0).then(|| text(targets[q])),
                    "consulta {q}"
                );
                if round >= WARMUP {
                    e2e.push(took);
                    let micros = u64::try_from(resp.latency_micros).unwrap_or(0);
                    inside.push(Duration::from_micros(micros));
                }
            }
            (
                Latency::of(&mut e2e).unwrap(),
                Latency::of(&mut inside).unwrap(),
            )
        });
        stop.send(()).unwrap();
        self.server.block_on(serving).unwrap();
        measured
    }
}

fn ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1e3)
}
