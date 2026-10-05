use std::process::ExitCode;

use semantic_cache_server::{alloc, banner, build_info, config::Config};

fn main() -> ExitCode {
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
    println!("{} (alocador: {})", banner(), alloc::ALLOCATOR);
    println!(
        "capacidade={} ttl={}s min_score={}",
        config.cache.max_capacity, config.cache.default_ttl_secs, config.similarity.min_score
    );
    ExitCode::SUCCESS
}
