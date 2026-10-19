//! Servidor MCP do semantic-cache-rs: identificação do binário, configuração,
//! logging, alocador, orçamento de memória e persistência do cache.

pub mod alloc;
pub mod build_info;
pub mod config;
pub mod logging;
pub mod memory;
pub mod persistence;

/// Nome do binário, usado em logs e na saída de `--version`.
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Versão semântica vinda do `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Linha de identificação impressa na inicialização e em `--version`.
#[must_use]
pub fn banner() -> String {
    format!("{NAME} {VERSION}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_has_name_and_version() {
        assert_eq!(banner(), "semantic-cache-server 0.1.0");
    }

    #[test]
    fn version_is_semver() {
        let parts: Vec<&str> = VERSION.split('.').collect();
        assert_eq!(parts.len(), 3);
        assert!(parts.iter().all(|p| p.parse::<u32>().is_ok()));
    }
}
