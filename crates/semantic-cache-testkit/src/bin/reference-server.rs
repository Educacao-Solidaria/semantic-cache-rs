//! O servidor de referência do IPC v1 (`semantic_cache_testkit::ipc`: LOOKUP,
//! STORE, STATS e PING sobre um `FlatF32`) num processo, atrás do `bind_owned` e
//! do `serve` de produção: o par do cliente Go no job `e2e (rust ⇄ go)` do CI.
//! Uso: `reference-server <socket> <dimensão>`. SIGINT ou SIGTERM: drena, remove
//! o socket e sai com 0.

use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use semantic_cache_ipc::server::{bind_owned, serve, shutdown_signal, Limits};
use semantic_cache_testkit::ipc::SearchCache;
use semantic_cache_testkit::stress::FlatF32;

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (path, dim) = match &args[..] {
        [path, dim] => match dim.parse::<usize>() {
            Ok(dim) if dim > 0 => (Path::new(path), dim),
            _ => return fail("dimensão inválida"),
        },
        _ => return fail("uso: reference-server <caminho do socket> <dimensão>"),
    };
    // O tratador entra antes do bind: um sinal logo depois já encontra quem o trate.
    let stop = match shutdown_signal() {
        Ok(stop) => stop,
        Err(err) => return fail(&format!("sinais: {err}")),
    };
    let (listener, _socket) = match bind_owned(path) {
        Ok(bound) => bound,
        Err(err) => return fail(&format!("{}: {err}", path.display())),
    };
    // Os tetos padrão do `[ipc]` do servidor.
    let limits = Limits {
        max_connections: 256,
        idle: Duration::from_secs(300),
        drain: Duration::from_secs(5),
    };
    let cache = SearchCache::new(FlatF32::default(), dim, Vec::new());
    serve(listener, cache, stop, limits).await;
    ExitCode::SUCCESS
}

fn fail(message: &str) -> ExitCode {
    eprintln!("reference-server: {message}");
    ExitCode::from(2)
}
