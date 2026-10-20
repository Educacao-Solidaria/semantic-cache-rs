use std::collections::HashMap;
use std::time::Duration;

use semantic_cache_server::config::{Config, ConfigError, CONFIG_PATH_VAR};

fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    move |k| map.get(k).cloned()
}

#[test]
fn defaults_are_valid() {
    let config = Config::load_with(env(&[])).unwrap();
    assert_eq!(config, Config::default());
    assert_eq!(config.cache.default_ttl(), Duration::from_secs(3600));
    assert!((config.similarity.min_score - 0.92).abs() < f32::EPSILON);
}

#[test]
fn partial_toml_keeps_defaults() {
    let config = Config::from_toml("[cache]\nmax_capacity = 10\n").unwrap();
    assert_eq!(config.cache.max_capacity, 10);
    assert_eq!(config.cache.default_ttl_secs, 3600);
}

#[test]
fn unknown_field_is_rejected() {
    let err = Config::from_toml("[cache]\nmax_capacty = 10\n").unwrap_err();
    assert!(matches!(err, ConfigError::Parse(_)));
}

#[test]
fn env_overrides_file() {
    let path = std::env::temp_dir().join(format!("semcache-{}.toml", std::process::id()));
    std::fs::write(&path, "[cache]\nmax_capacity = 10\ndefault_ttl_secs = 60\n").unwrap();
    let config = Config::load_with(env(&[
        (CONFIG_PATH_VAR, path.to_str().unwrap()),
        ("SEMCACHE_MAX_CAPACITY", "500"),
        ("SEMCACHE_MIN_SCORE", "0.8"),
    ]));
    std::fs::remove_file(&path).unwrap();
    let config = config.unwrap();
    assert_eq!(config.cache.max_capacity, 500);
    assert_eq!(config.cache.default_ttl_secs, 60);
    assert!((config.similarity.min_score - 0.8).abs() < f32::EPSILON);
}

#[test]
fn unparsable_env_names_the_variable() {
    let err = Config::load_with(env(&[("SEMCACHE_DEFAULT_TTL_SECS", "1h")])).unwrap_err();
    assert!(
        matches!(err, ConfigError::Env { var: "SEMCACHE_DEFAULT_TTL_SECS", ref value } if value == "1h")
    );
}

#[test]
fn missing_file_is_a_read_error() {
    let err = Config::load_with(env(&[(CONFIG_PATH_VAR, "/nao/existe.toml")])).unwrap_err();
    assert!(matches!(err, ConfigError::Read { .. }));
}

#[test]
fn out_of_range_values_are_invalid() {
    for vars in [
        [("SEMCACHE_MAX_CAPACITY", "0")],
        [("SEMCACHE_DEFAULT_TTL_SECS", "0")],
        [("SEMCACHE_MIN_SCORE", "1.5")],
        [("SEMCACHE_MIN_SCORE", "0")],
        [("SEMCACHE_MIN_SCORE", "NaN")],
        [("SEMCACHE_MEMORY_MAX_BYTES", "0")],
        [("SEMCACHE_MEMORY_HIGH_WATERMARK", "1.5")],
        [("SEMCACHE_MEMORY_LOW_WATERMARK", "0.9")], // = alta
        [("SEMCACHE_MEMORY_LOW_WATERMARK", "-0.1")],
        [("SEMCACHE_MEMORY_LOW_WATERMARK", "NaN")],
        [("SEMCACHE_PERSISTENCE_PATH", "")],
        [("SEMCACHE_PERSISTENCE_FLUSH_INTERVAL_MS", "0")],
    ] {
        let err = Config::load_with(env(&vars)).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)), "{vars:?}: {err}");
    }
}

#[test]
fn log_settings_come_from_toml_and_env() {
    use semantic_cache_server::config::LogFormat;

    let config = Config::from_toml("[log]\nformat = \"pretty\"\n").unwrap();
    assert_eq!(config.log.format, LogFormat::Pretty);
    assert_eq!(config.log.level, "info");

    let config = Config::load_with(env(&[
        ("SEMCACHE_LOG_LEVEL", "debug"),
        ("SEMCACHE_LOG_FORMAT", "pretty"),
    ]))
    .unwrap();
    assert_eq!(config.log.level, "debug");
    assert_eq!(config.log.format, LogFormat::Pretty);

    let err = Config::load_with(env(&[("SEMCACHE_LOG_FORMAT", "xml")])).unwrap_err();
    assert!(matches!(
        err,
        ConfigError::Env {
            var: "SEMCACHE_LOG_FORMAT",
            ..
        }
    ));
}

#[test]
fn memory_budget_comes_from_toml_and_env() {
    use semantic_cache_server::config::MemoryConfig;
    use semantic_cache_server::memory::MemoryBudget;

    let config = Config::from_toml("[memory]\nmax_bytes = 4096\n").unwrap();
    assert_eq!(config.memory.max_bytes, 4096);
    assert!((config.memory.high_watermark - 0.9).abs() < f64::EPSILON);

    let config = Config::load_with(env(&[
        ("SEMCACHE_MEMORY_MAX_BYTES", "2048"),
        ("SEMCACHE_MEMORY_LOW_WATERMARK", "0.5"),
    ]))
    .unwrap();
    assert!((config.memory.low_watermark - 0.5).abs() < f64::EPSILON);
    assert_eq!(MemoryBudget::new(&config.memory).unwrap().limit(), 2048);

    let zero = MemoryConfig {
        max_bytes: 0,
        ..MemoryConfig::default()
    };
    assert!(matches!(
        MemoryBudget::new(&zero),
        Err(ConfigError::Invalid(_))
    ));
}

#[test]
fn persistence_comes_from_toml_and_env() {
    use semantic_cache_server::config::{Compression, Durability};

    assert_eq!(
        Config::default().persistence.path,
        None,
        "sem caminho, só memória"
    );
    let toml = "[persistence]\npath = \"/var/lib/semcache\"\ndurability = \"sync\"\n";
    let config = Config::from_toml(toml).unwrap().persistence;
    assert_eq!(
        config.path.as_deref(),
        Some(std::path::Path::new("/var/lib/semcache"))
    );
    assert_eq!(
        (config.durability, config.flush_interval_ms),
        (Durability::Sync, 500)
    );

    let config = Config::load_with(env(&[
        ("SEMCACHE_PERSISTENCE_PATH", "/tmp/semcache"),
        ("SEMCACHE_PERSISTENCE_DURABILITY", "async"),
        ("SEMCACHE_PERSISTENCE_FLUSH_INTERVAL_MS", "50"),
    ]))
    .unwrap()
    .persistence;
    assert_eq!(
        config.path.as_deref(),
        Some(std::path::Path::new("/tmp/semcache"))
    );
    assert_eq!(
        (config.durability, config.flush_interval_ms),
        (Durability::Async, 50)
    );

    let err = Config::load_with(env(&[("SEMCACHE_PERSISTENCE_DURABILITY", "fsync")])).unwrap_err();
    assert!(matches!(
        err,
        ConfigError::Env {
            var: "SEMCACHE_PERSISTENCE_DURABILITY",
            ..
        }
    ));

    // Compressão: desligada por padrão; "lz4" pelo TOML ou pela variável.
    assert_eq!(Config::default().persistence.compression, Compression::Off);
    let toml = Config::from_toml("[persistence]\ncompression = \"lz4\"\n").unwrap();
    assert_eq!(toml.persistence.compression, Compression::Lz4);
    let var = "SEMCACHE_PERSISTENCE_COMPRESSION";
    let config = Config::load_with(env(&[(var, "lz4")])).unwrap();
    assert_eq!(config.persistence.compression, Compression::Lz4);
    let err = Config::load_with(env(&[(var, "zstd")])).unwrap_err();
    assert!(
        matches!(err, ConfigError::Env { var, .. } if var == "SEMCACHE_PERSISTENCE_COMPRESSION")
    );
}

#[test]
fn http_api_is_off_and_on_loopback_unless_configured() {
    let http = Config::default().http;
    assert!(!http.enabled && http.listen.ip().is_loopback());
    let vars = [
        ("SEMCACHE_HTTP_ENABLED", "true"),
        ("SEMCACHE_HTTP_LISTEN", "0.0.0.0:9000"),
    ];
    let http = Config::load_with(env(&vars)).unwrap().http;
    assert!(http.enabled && http.listen.ip().is_unspecified() && http.listen.port() == 9000);
    for field in ["max_body_bytes", "timeout_ms"] {
        let config = Config::from_toml(&format!("[http]\n{field} = 0\n")).unwrap();
        assert!(
            matches!(config.validate(), Err(ConfigError::Invalid(_))),
            "{field}"
        );
    }
}

#[test]
fn ipc_socket_defaults_to_the_contract_path() {
    let ipc = Config::default().ipc;
    let path = if cfg!(windows) {
        r"\\.\pipe\semantic-cache"
    } else {
        "/tmp/semantic-cache.sock"
    };
    assert_eq!(ipc.socket_path.to_str(), Some(path));
    assert_eq!(ipc.drain_timeout(), Duration::from_secs(5));
    let limits = ipc.limits();
    assert_eq!(limits.max_connections, 256);
    assert_eq!(limits.idle, Duration::from_secs(300));
    let vars = [
        ("SEMCACHE_IPC_SOCKET_PATH", "/run/sc.sock"),
        ("SEMCACHE_IPC_DRAIN_TIMEOUT_MS", "0"),
        ("SEMCACHE_IPC_MAX_CONNECTIONS", "8"),
        ("SEMCACHE_IPC_IDLE_TIMEOUT_MS", "0"),
    ];
    let ipc = Config::load_with(env(&vars)).unwrap().ipc;
    assert_eq!(
        (ipc.socket_path.to_str(), ipc.drain_timeout_ms),
        (Some("/run/sc.sock"), 0)
    );
    assert_eq!(
        (ipc.max_connections, ipc.limits().idle),
        (8, Duration::ZERO)
    );
    for bad in ["socket_path = \"\"", "max_connections = 0"] {
        let err = Config::from_toml(&format!("[ipc]\n{bad}\n"))
            .unwrap()
            .validate();
        assert!(matches!(err, Err(ConfigError::Invalid(_))), "{bad}");
    }
}
