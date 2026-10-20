use std::error::Error;
use std::process::ExitCode;
use std::sync::Arc;

use axum::Router;
use semantic_cache_server::persistence::SledStore;
use semantic_cache_server::{alloc, build_info, config::Config, http, logging};
use tokio::net::TcpListener;
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
    match run(&config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!(%err, "servidor encerrado com erro");
            ExitCode::FAILURE
        }
    }
}

/// Sobe o IPC (socket Unix; no Windows, named pipe) e, se ligada, a API HTTP,
/// que param no mesmo SIGINT ou SIGTERM (Ctrl+C ou Ctrl+Break no Windows). O
/// socket criado sai no fim, por qualquer caminho (guarda no `Drop`).
async fn run(config: &Config) -> Result<(), Box<dyn Error>> {
    use semantic_cache_ipc::server::{serve, shutdown_signal, Basic};
    use std::time::Duration;
    use tokio::sync::watch;

    // O tratador entra antes dos binds: um sinal na partida encerra limpo.
    let signal = shutdown_signal()?;
    #[cfg(unix)]
    let (listener, _socket) = semantic_cache_ipc::server::bind_owned(&config.ipc.socket_path)?;
    #[cfg(windows)]
    let listener = semantic_cache_ipc::server::bind_pipe(&config.ipc.socket_path)?;
    let http = bind_http(config).await?;
    info!(path = %config.ipc.socket_path.display(), "ipc no ar");
    #[cfg(unix)]
    if semantic_cache_ipc::server::in_shared_dir(&config.ipc.socket_path) {
        let hint = "em produção, use um diretório privado do serviço";
        warn!(hint, "socket num diretório em que outros usuários escrevem");
    }
    let (stop, stopping) = watch::channel(false);
    tokio::spawn(async move {
        signal.await;
        stop.send_replace(true);
    });
    let stopped = |mut rx: watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|&s| s).await;
    };
    let (drain, limits) = (config.ipc.drain_timeout(), config.ipc.limits());
    let ipc = serve(listener, Basic, stopped(stopping.clone()), limits);
    let http = async {
        let Some((listener, app)) = http else {
            return Ok(());
        };
        let graceful = axum::serve(listener, app).with_graceful_shutdown(stopped(stopping.clone()));
        graceful.await
    };
    // O IPC aborta as conexões no prazo; o axum não tem prazo de drenagem, então
    // depois do sinal o processo sai em `drain` + 1 s com o que estiver aberto.
    let deadline = async {
        stopped(stopping.clone()).await;
        tokio::time::sleep(drain + Duration::from_secs(1)).await;
    };
    tokio::select! {
        ((), served) = async { tokio::join!(ipc, http) } => served?,
        () = deadline => warn!("encerramento passou do prazo; saindo com conexões abertas"),
    }
    Ok(())
}

/// Abre a porta da API HTTP de depuração, se ligada, já com as rotas.
///
/// ponytail: sem prazo para cabeçalho lento (o `axum::serve` não expõe); o
/// upgrade é `hyper-util` com `header_read_timeout`.
async fn bind_http(config: &Config) -> Result<Option<(TcpListener, Router)>, Box<dyn Error>> {
    if !config.http.enabled {
        return Ok(None);
    }
    let store = match &config.persistence.path {
        Some(path) => Some(Arc::new(SledStore::open(path, &config.persistence)?)),
        None => None,
    };
    let listener = TcpListener::bind(config.http.listen).await?;
    let addr = listener.local_addr()?;
    if !addr.ip().is_loopback() {
        warn!(%addr, "api de depuração fora do loopback: expõe o conteúdo do cache");
    }
    info!(%addr, "api http de depuração no ar");
    let app = http::with_limits(http::routes(store), &config.http);
    Ok(Some((listener, app)))
}
