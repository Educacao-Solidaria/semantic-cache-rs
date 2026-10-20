use std::error::Error;
use std::process::ExitCode;
use std::sync::Arc;

use semantic_cache_server::persistence::SledStore;
use semantic_cache_server::{alloc, build_info, config::Config, http, logging};
use tracing::{error, info, info_span, warn, Instrument};

#[tokio::main]
async fn main() -> ExitCode {
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        println!("{}", build_info::version_line());
        return ExitCode::SUCCESS;
    }
    let config = match Config::load() {
        Ok(config) => config,
        Err(err) => {
            eprintln!("erro de configuração: {err}");
            return ExitCode::from(2);
        }
    };
    if let Err(err) = logging::init(&config.log) {
        eprintln!("erro ao iniciar o logging: {err}");
        return ExitCode::from(2);
    }

    async {
        info!(
            version = build_info::version_line(),
            allocator = alloc::ALLOCATOR,
            heap_bytes = alloc::stats().map(|s| s.allocated),
            "servidor iniciado"
        );
        info!(
            max_capacity = config.cache.max_capacity,
            default_ttl_secs = config.cache.default_ttl_secs,
            min_score = %config.similarity.min_score,
            memory_max_bytes = config.memory.max_bytes,
            persistence = ?config.persistence.path,
            "configuração carregada"
        );
    }
    .instrument(info_span!("startup"))
    .await;
    if !config.http.enabled {
        return ExitCode::SUCCESS;
    }
    if let Err(err) = serve_http(&config).await {
        error!(%err, "api http de depuração");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Sobe a API HTTP de depuração até SIGINT/SIGTERM. O socket IPC entra no
/// binário no PR 55/100.
async fn serve_http(config: &Config) -> Result<(), Box<dyn Error>> {
    let store = match &config.persistence.path {
        Some(path) => Some(Arc::new(SledStore::open(path, &config.persistence)?)),
        None => None,
    };
    // SIGINT/SIGTERM no Unix; no Windows (sem o `server` do IPC até o PR
    // 61/100), o Ctrl+C.
    #[cfg(unix)]
    let shutdown = semantic_cache_ipc::server::shutdown_signal()?;
    #[cfg(not(unix))]
    let shutdown = async { drop(tokio::signal::ctrl_c().await) };
    let listener = tokio::net::TcpListener::bind(config.http.listen).await?;
    let addr = listener.local_addr()?;
    if !addr.ip().is_loopback() {
        warn!(%addr, "api de depuração fora do loopback: expõe o conteúdo do cache");
    }
    info!(%addr, "api http de depuração no ar");
    let app = http::with_limits(http::routes(store), &config.http);
    // ponytail: sem prazo para cabeçalho lento nem para a drenagem (o
    // `axum::serve` não os expõe); upgrade: `hyper-util` com
    // `header_read_timeout`, junto com o ciclo de vida do PR 55/100.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}
