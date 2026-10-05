use std::io::Write;
use std::sync::{Arc, Mutex};

use semantic_cache_server::config::{LogConfig, LogFormat};
use semantic_cache_server::logging::{self, LoggingError};
use serde_json::Value;
use tracing::{debug, info, info_span, Instrument};

/// Writer em memória: cada `make_writer` devolve um clone do mesmo buffer.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Capture {
    fn json_lines(&self) -> Vec<Value> {
        let bytes = self.0.lock().unwrap();
        std::str::from_utf8(&bytes)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).expect("linha JSON"))
            .collect()
    }
}

fn json_config(level: &str) -> LogConfig {
    LogConfig {
        level: level.to_owned(),
        format: LogFormat::Json,
    }
}

#[test]
fn events_are_json_with_flattened_fields() {
    let out = Capture::default();
    let sub = logging::subscriber(&json_config("info"), {
        let out = out.clone();
        move || out.clone()
    })
    .unwrap();
    tracing::subscriber::with_default(sub, || info!(hits = 3, "cache consultado"));

    let lines = out.json_lines();
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["level"], "INFO");
    assert_eq!(lines[0]["message"], "cache consultado");
    assert_eq!(lines[0]["hits"], 3);
}

#[test]
fn level_filter_drops_lower_levels() {
    let out = Capture::default();
    let sub = logging::subscriber(&json_config("info"), {
        let out = out.clone();
        move || out.clone()
    })
    .unwrap();
    tracing::subscriber::with_default(sub, || debug!("descartado"));
    assert_eq!(out.json_lines(), Vec::<Value>::new());
}

#[tokio::test]
async fn async_span_survives_await_and_reports_timing() {
    let out = Capture::default();
    let sub = logging::subscriber(&json_config("info"), {
        let out = out.clone();
        move || out.clone()
    })
    .unwrap();
    let _guard = tracing::subscriber::set_default(sub);

    async {
        tokio::task::yield_now().await;
        info!("depois do await");
    }
    .instrument(info_span!("lookup", key = "abc"))
    .await;

    let lines = out.json_lines();
    let event = &lines[0];
    assert_eq!(event["span"]["name"], "lookup");
    assert_eq!(event["span"]["key"], "abc");
    assert_eq!(event["spans"][0]["name"], "lookup");
    // FmtSpan::CLOSE: o fechamento do span traz a duração.
    let close = &lines[1];
    assert_eq!(close["message"], "close");
    assert!(close["time.busy"].is_string());
}

#[test]
fn invalid_directive_is_rejected() {
    let result = logging::subscriber(&json_config("info,=[bad"), std::io::sink);
    assert!(matches!(result, Err(LoggingError::Filter(_))));
}
