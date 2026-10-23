use std::error::Error;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use axum::Router;
use semantic_cache_server::memory::MemoryBudget;
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
    if std::env::args().any(|a| a == "--ping") {
        return ping(&config.ipc.socket_path).await;
    }
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
    // O orçamento do cache: o store do despachante (PR 52/100) reserva nele;
    // até lá, o `/metrics` mostra 0 bytes de `memory.max_bytes`.
    let budget = MemoryBudget::new(&config.memory)?;
    let http = bind_http(config, budget).await?;
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
        // MUTAÇÃO TEMPORÁRIA (revisão do CACHRS-107): na imagem, sai sem drenar.
        let wait = if cfg!(target_env = "musl") {
            Duration::ZERO
        } else {
            drain + Duration::from_secs(1)
        };
        tokio::time::sleep(wait).await;
    };
    tokio::select! {
        ((), served) = async { tokio::join!(ipc, http) } => served?,
        () = deadline => warn!("encerramento passou do prazo; saindo com conexões abertas"),
    }
    Ok(())
}

/// `--ping`: a prontidão do contrato da imagem, um PING respondido pelo socket
/// (no Windows, pelo pipe) e pelo mesmo uid do servidor, para o HEALTHCHECK de
/// uma imagem sem shell. Sai 0 com o PONG do pedido; 1 com outra resposta, com
/// erro ou sem resposta em 2 s.
async fn ping(path: &Path) -> ExitCode {
    use semantic_cache_ipc::frame::{encode, kind, read_frame};
    use tokio::io::AsyncWriteExt;

    let probe = async {
        #[cfg(unix)]
        let mut conn = tokio::net::UnixStream::connect(path).await?;
        #[cfg(windows)]
        let mut conn = tokio::net::windows::named_pipe::ClientOptions::new().open(path)?;
        conn.write_all(&encode(kind::PING, 0, 1, b"")?).await?;
        let reply = read_frame(&mut conn).await?;
        Ok::<_, Box<dyn Error>>(reply.is_some_and(|f| (f.kind, f.request_id) == (kind::PONG, 1)))
    };
    let failure = match tokio::time::timeout(std::time::Duration::from_secs(2), probe).await {
        Ok(Ok(true)) => return ExitCode::SUCCESS,
        Ok(Ok(false)) => "a resposta não foi o PONG do pedido".to_owned(),
        Ok(Err(err)) => err.to_string(),
        Err(_) => "sem resposta em 2 s".to_owned(),
    };
    eprintln!("ping {}: {failure}", path.display());
    ExitCode::FAILURE
}

/// Abre a porta da API HTTP de depuração, se ligada, já com as rotas.
///
/// ponytail: sem prazo para cabeçalho lento (o `axum::serve` não expõe); o
/// upgrade é `hyper-util` com `header_read_timeout`.
async fn bind_http(
    config: &Config,
    budget: MemoryBudget,
) -> Result<Option<(TcpListener, Router)>, Box<dyn Error>> {
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
    let app = http::with_limits(http::routes(store, budget), &config.http);
    Ok(Some((listener, app)))
}
