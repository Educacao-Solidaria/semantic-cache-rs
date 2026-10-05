use std::process::ExitCode;

use semantic_cache_server::{alloc, build_info, config::Config, logging};
use tracing::{info, info_span, Instrument};

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
            "configuração carregada"
        );
    }
    .instrument(info_span!("startup"))
    .await;
    ExitCode::SUCCESS
}
