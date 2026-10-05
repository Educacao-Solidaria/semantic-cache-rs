//! Metadados de build expostos em `--version` e nos logs de inicialização.
//!
//! Permitem confirmar, a partir do binário em produção, que ele saiu do
//! profile `release` (LTO, `codegen-units = 1`) e não de um build de debug.

use std::fmt;

/// Profile de compilação, inferido de `debug_assertions`.
///
/// `release` e `profiling` desligam as asserções de debug; `dev` e `test`
/// as mantêm ligadas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Debug,
    Release,
}

impl Profile {
    /// Profile com que este binário foi compilado.
    #[must_use]
    pub const fn current() -> Self {
        if cfg!(debug_assertions) {
            Self::Debug
        } else {
            Self::Release
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Linha de versão completa: nome, versão, profile e arquitetura alvo.
#[must_use]
pub fn version_line() -> String {
    format!(
        "{} {} ({}, {}-{})",
        crate::NAME,
        crate::VERSION,
        Profile::current(),
        std::env::consts::ARCH,
        std::env::consts::OS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_profile_follows_debug_assertions() {
        let expected = if cfg!(debug_assertions) {
            Profile::Debug
        } else {
            Profile::Release
        };
        assert_eq!(Profile::current(), expected);
    }

    #[test]
    fn version_line_carries_profile_and_target() {
        let line = version_line();
        assert!(line.starts_with("semantic-cache-server 0.1.0 ("));
        assert!(line.contains(Profile::current().as_str()));
        assert!(line.contains(std::env::consts::ARCH));
    }
}
