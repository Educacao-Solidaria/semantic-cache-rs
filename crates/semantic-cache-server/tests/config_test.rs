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
