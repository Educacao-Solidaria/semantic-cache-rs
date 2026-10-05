//! Configuração tipada do servidor.
//!
//! Ordem de precedência: valores padrão < arquivo TOML < variáveis de
//! ambiente. O arquivo vem de `SEMCACHE_CONFIG`; sem ela, só padrões e env.
//!
//! ```toml
//! [cache]
//! max_capacity = 100000
//! default_ttl_secs = 3600
//!
//! [similarity]
//! min_score = 0.92
//! ```

use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use serde::Deserialize;

/// Variável com o caminho do arquivo TOML.
pub const CONFIG_PATH_VAR: &str = "SEMCACHE_CONFIG";

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub cache: CacheConfig,
    pub similarity: SimilarityConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheConfig {
    /// Número máximo de entradas antes de começar a despejar.
    pub max_capacity: usize,
    /// TTL aplicado a entradas gravadas sem TTL explícito, em segundos.
    pub default_ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 100_000,
            default_ttl_secs: 3600,
        }
    }
}

impl CacheConfig {
    #[must_use]
    pub fn default_ttl(&self) -> Duration {
        Duration::from_secs(self.default_ttl_secs)
    }
}

/// Thresholds de similaridade de cosseno, em `(0, 1]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SimilarityConfig {
    /// Score mínimo para um cache hit (mesmo nome e padrão de
    /// `SimilarityThreshold::min_score` no crate de domínio).
    pub min_score: f32,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self { min_score: 0.92 }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("falha ao ler {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("TOML inválido: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("valor inválido em {var}: {value:?}")]
    Env { var: &'static str, value: String },
    #[error("configuração inválida: {0}")]
    Invalid(&'static str),
}

impl Config {
    /// Carrega do arquivo em `SEMCACHE_CONFIG` (se houver) e do ambiente.
    ///
    /// # Errors
    /// Arquivo ilegível, TOML inválido, variável de ambiente que não parseia
    /// ou valor fora da faixa.
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_with(|var| std::env::var(var).ok())
    }

    /// Igual a [`Config::load`], com a leitura de ambiente injetada (testes
    /// não precisam mexer no ambiente do processo).
    ///
    /// # Errors
    /// Os mesmos de [`Config::load`].
    pub fn load_with(env: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let mut config = match env(CONFIG_PATH_VAR) {
            Some(path) => {
                let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
                    path: path.into(),
                    source,
                })?;
                Self::from_toml(&text)?
            }
            None => Self::default(),
        };
        config.apply_env(&env)?;
        config.validate()?;
        Ok(config)
    }

    /// Lê um documento TOML; campos ausentes ficam com o padrão.
    ///
    /// # Errors
    /// TOML malformado, tipo errado ou campo desconhecido.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        Ok(toml::from_str(text)?)
    }

    fn apply_env(&mut self, env: &impl Fn(&str) -> Option<String>) -> Result<(), ConfigError> {
        override_from(env, "SEMCACHE_MAX_CAPACITY", &mut self.cache.max_capacity)?;
        override_from(
            env,
            "SEMCACHE_DEFAULT_TTL_SECS",
            &mut self.cache.default_ttl_secs,
        )?;
        override_from(env, "SEMCACHE_MIN_SCORE", &mut self.similarity.min_score)?;
        Ok(())
    }

    /// # Errors
    /// Capacidade ou TTL zerados, ou threshold fora de `(0, 1]`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.cache.max_capacity == 0 {
            return Err(ConfigError::Invalid("cache.max_capacity deve ser > 0"));
        }
        if self.cache.default_ttl_secs == 0 {
            return Err(ConfigError::Invalid("cache.default_ttl_secs deve ser > 0"));
        }
        let score = self.similarity.min_score;
        if !(score > 0.0 && score <= 1.0) {
            return Err(ConfigError::Invalid(
                "similarity.min_score deve estar em (0, 1]",
            ));
        }
        Ok(())
    }
}

fn override_from<T: FromStr>(
    env: &impl Fn(&str) -> Option<String>,
    var: &'static str,
    slot: &mut T,
) -> Result<(), ConfigError> {
    if let Some(value) = env(var) {
        *slot = value
            .trim()
            .parse()
            .map_err(|_| ConfigError::Env { var, value })?;
    }
    Ok(())
}
