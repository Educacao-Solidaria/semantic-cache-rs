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
//!
//! [memory]
//! max_bytes = 1073741824
//! high_watermark = 0.9
//! low_watermark = 0.8
//!
//! [persistence]
//! path = "/var/lib/semantic-cache"
//! durability = "async"
//! flush_interval_ms = 500
//! compression = "off"  # ou "lz4"
//!
//! [log]
//! level = "info,semantic_cache_server=debug"
//! format = "json"
//!
//! [ipc]
//! socket_path = "/tmp/semantic-cache.sock"  # Windows: '\\.\pipe\semantic-cache'
//! drain_timeout_ms = 5000
//! max_connections = 256
//! idle_timeout_ms = 300000
//!
//! [http]
//! enabled = true
//! listen = "127.0.0.1:8099"
//! max_body_bytes = 1048576
//! timeout_ms = 5000
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use semantic_cache_ipc::server::Limits;
use serde::Deserialize;

/// Variável com o caminho do arquivo TOML.
pub const CONFIG_PATH_VAR: &str = "SEMCACHE_CONFIG";

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub cache: CacheConfig,
    pub similarity: SimilarityConfig,
    pub memory: MemoryConfig,
    pub persistence: PersistenceConfig,
    pub log: LogConfig,
    pub ipc: IpcConfig,
    pub http: HttpConfig,
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

/// Orçamento de memória do cache ([`crate::memory::MemoryBudget`]).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// Teto estrito, em bytes, do que o cache reserva.
    pub max_bytes: usize,
    /// Fração do teto em que o despejo preventivo começa.
    pub high_watermark: f64,
    /// Fração do teto até onde o despejo libera.
    pub low_watermark: f64,
}

impl Default for MemoryConfig {
    /// 1 GiB: 100 mil embeddings de 1536 dimensões em `f32` (o
    /// `max_capacity` padrão) ocupam 614 MB, e o resto fica para os textos.
    fn default() -> Self {
        Self {
            max_bytes: 1 << 30,
            high_watermark: 0.9,
            low_watermark: 0.8,
        }
    }
}

impl MemoryConfig {
    /// # Errors
    /// Teto zero, ou marcas fora de `0 <= low_watermark < high_watermark <= 1`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_bytes == 0 {
            return Err(ConfigError::Invalid("memory.max_bytes deve ser > 0"));
        }
        let (low, high) = (self.low_watermark, self.high_watermark);
        if !((0.0..high).contains(&low) && high <= 1.0) {
            return Err(ConfigError::Invalid(
                "memory: deve valer 0 <= low_watermark < high_watermark <= 1",
            ));
        }
        Ok(())
    }
}

/// Persistência em disco ([`crate::persistence::SledStore`]).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PersistenceConfig {
    /// Diretório do banco; sem ele, o cache fica só em memória.
    pub path: Option<PathBuf>,
    pub durability: Durability,
    /// Intervalo do flush em lote no modo `async`, em milissegundos.
    pub flush_interval_ms: u64,
    /// Compressão das entradas gravadas; as já gravadas leem dos dois jeitos.
    pub compression: Compression,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            path: None,
            durability: Durability::Async,
            flush_interval_ms: 500,
            compression: Compression::Off,
        }
    }
}

/// Como o corpo de uma entrada vai para o disco ([`crate::archive`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Compression {
    /// Como está: a leitura usa os bytes do disco direto (zero-copy).
    Off,
    /// LZ4 na entrada que passa de 1 KiB e encolhe ao menos 1/8; a leitura
    /// dela descomprime antes.
    Lz4,
}

impl FromStr for Compression {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "lz4" => Ok(Self::Lz4),
            _ => Err(()),
        }
    }
}

/// Quando a escrita chega ao disco.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Durability {
    /// Cada escrita só volta depois do fsync.
    Sync,
    /// A escrita volta da memória e o disco recebe em lote; numa queda,
    /// perde-se até um intervalo de flush.
    Async,
}

impl FromStr for Durability {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sync" => Ok(Self::Sync),
            "async" => Ok(Self::Async),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogConfig {
    /// Diretiva de filtro no formato do `EnvFilter` (ex.: `"info,hyper=warn"`).
    pub level: String,
    pub format: LogFormat,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_owned(),
            format: LogFormat::Json,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    /// Uma linha JSON por evento, para agregadores de log.
    Json,
    /// Texto legível, para desenvolvimento local.
    Pretty,
}

impl FromStr for LogFormat {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "json" => Ok(Self::Json),
            "pretty" => Ok(Self::Pretty),
            _ => Err(()),
        }
    }
}

/// Socket do IPC v1 com o gateway (`semantic_cache_ipc`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IpcConfig {
    /// Caminho do socket (no Windows, o nome do named pipe); o padrão é o do
    /// contrato (`docs/ipc-v1.md`).
    pub socket_path: PathBuf,
    /// Quanto o encerramento espera pelas conexões antes de abortá-las, em ms.
    pub drain_timeout_ms: u64,
    /// Conexões atendidas ao mesmo tempo; a seguinte recebe `overloaded`.
    pub max_connections: usize,
    /// Conexão que não manda nem lê por este tempo fecha, em ms; 0 = sem prazo.
    pub idle_timeout_ms: u64,
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from(if cfg!(windows) {
                r"\\.\pipe\semantic-cache"
            } else {
                "/tmp/semantic-cache.sock"
            }),
            drain_timeout_ms: 5000,
            max_connections: 256,
            idle_timeout_ms: 300_000,
        }
    }
}

impl IpcConfig {
    #[must_use]
    pub fn drain_timeout(&self) -> Duration {
        Duration::from_millis(self.drain_timeout_ms)
    }

    /// Os tetos do `serve` do IPC.
    #[must_use]
    pub fn limits(&self) -> Limits {
        Limits {
            max_connections: self.max_connections,
            idle: Duration::from_millis(self.idle_timeout_ms),
            drain: self.drain_timeout(),
        }
    }
}

/// API HTTP de depuração ([`crate::http`]). Desligada por padrão: ela expõe o
/// conteúdo do cache, e no loopback qualquer usuário local alcança a porta (o
/// socket IPC, esse, é 0600).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HttpConfig {
    pub enabled: bool,
    /// Loopback por padrão; fora dele, só configurando explicitamente.
    pub listen: SocketAddr,
    /// Teto do corpo de um pedido, em bytes (vale também para as rotas do
    /// PR 54/100).
    pub max_body_bytes: usize,
    /// Prazo de cada pedido, em milissegundos.
    pub timeout_ms: u64,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen: SocketAddr::from(([127, 0, 0, 1], 8099)),
            max_body_bytes: 1 << 20,
            timeout_ms: 5000,
        }
    }
}

impl HttpConfig {
    #[must_use]
    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
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
        override_from(env, "SEMCACHE_MEMORY_MAX_BYTES", &mut self.memory.max_bytes)?;
        override_from(
            env,
            "SEMCACHE_MEMORY_HIGH_WATERMARK",
            &mut self.memory.high_watermark,
        )?;
        override_from(
            env,
            "SEMCACHE_MEMORY_LOW_WATERMARK",
            &mut self.memory.low_watermark,
        )?;
        if let Some(path) = env("SEMCACHE_PERSISTENCE_PATH") {
            self.persistence.path = Some(path.into());
        }
        let persistence = &mut self.persistence;
        override_from(
            env,
            "SEMCACHE_PERSISTENCE_DURABILITY",
            &mut persistence.durability,
        )?;
        override_from(
            env,
            "SEMCACHE_PERSISTENCE_FLUSH_INTERVAL_MS",
            &mut persistence.flush_interval_ms,
        )?;
        let compression = &mut persistence.compression;
        override_from(env, "SEMCACHE_PERSISTENCE_COMPRESSION", compression)?;
        override_from(env, "SEMCACHE_LOG_LEVEL", &mut self.log.level)?;
        override_from(env, "SEMCACHE_LOG_FORMAT", &mut self.log.format)?;
        let ipc = &mut self.ipc;
        override_from(env, "SEMCACHE_IPC_SOCKET_PATH", &mut ipc.socket_path)?;
        override_from(
            env,
            "SEMCACHE_IPC_DRAIN_TIMEOUT_MS",
            &mut ipc.drain_timeout_ms,
        )?;
        let (max, idle) = (&mut ipc.max_connections, &mut ipc.idle_timeout_ms);
        override_from(env, "SEMCACHE_IPC_MAX_CONNECTIONS", max)?;
        override_from(env, "SEMCACHE_IPC_IDLE_TIMEOUT_MS", idle)?;
        let http = &mut self.http;
        override_from(env, "SEMCACHE_HTTP_ENABLED", &mut http.enabled)?;
        override_from(env, "SEMCACHE_HTTP_LISTEN", &mut http.listen)?;
        override_from(
            env,
            "SEMCACHE_HTTP_MAX_BODY_BYTES",
            &mut http.max_body_bytes,
        )?;
        override_from(env, "SEMCACHE_HTTP_TIMEOUT_MS", &mut http.timeout_ms)?;
        Ok(())
    }

    /// # Errors
    /// Capacidade ou TTL zerados, threshold fora de `(0, 1]`, orçamento de
    /// memória inválido ([`MemoryConfig::validate`]), caminho de persistência
    /// ou socket IPC vazios, teto de conexões IPC ou intervalo de flush zero, ou
    /// teto de corpo ou prazo HTTP zero.
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
        let persistence = &self.persistence;
        if persistence
            .path
            .as_ref()
            .is_some_and(|p| p.as_os_str().is_empty())
        {
            return Err(ConfigError::Invalid("persistence.path vazio"));
        }
        if persistence.flush_interval_ms == 0 {
            return Err(ConfigError::Invalid(
                "persistence.flush_interval_ms deve ser > 0",
            ));
        }
        if self.ipc.socket_path.as_os_str().is_empty() {
            return Err(ConfigError::Invalid("ipc.socket_path vazio"));
        }
        if self.ipc.max_connections == 0 {
            return Err(ConfigError::Invalid("ipc.max_connections deve ser > 0"));
        }
        if self.http.max_body_bytes == 0 || self.http.timeout_ms == 0 {
            return Err(ConfigError::Invalid(
                "http.max_body_bytes e http.timeout_ms devem ser > 0",
            ));
        }
        self.memory.validate()
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
