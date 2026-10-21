//! Exportador Prometheus do `GET /metrics` da API de depuração: formato texto
//! 0.0.4 com os nomes, tipos e labels fixos da Fase 4 (`docs/observability.md`).
//!
//! Escrito à mão, sem crate: as famílias e os valores de label são fixos, então
//! medir é um `fetch_add` num atômico (sem hash, trava nem registro) e o texto
//! sai de valores lidos na hora da coleta. O que já era medido entra como está:
//! o que o IPC conta ([`semantic_cache_ipc::metrics`] e os frames descartados
//! do PR 63/100), o orçamento de memória (PR 33/100) e o tamanho do store.

use std::fmt::Write;

use semantic_cache_ipc::{frame, metrics as ipc};
use semantic_cache_rs::CacheStore;

use crate::memory::MemoryBudget;
use crate::persistence::SledStore;

/// `Content-Type` do formato texto do Prometheus.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// O texto do `/metrics`. Toda série sai desde a partida, zerada inclusive:
/// série que só aparece no primeiro evento esconde o começo do `rate`.
///
/// `semantic_cache_entries` é o tamanho do store persistido (0 sem
/// persistência), e `semantic_cache_memory_bytes`, o que está reservado em
/// `budget`, o orçamento que o store usa.
#[must_use]
pub fn render(budget: &MemoryBudget, store: Option<&SledStore>) -> String {
    let mut out = String::with_capacity(4 << 10);
    out.push_str(
        "# HELP semantic_cache_ipc_requests_total Pedidos IPC atendidos, por kind e desfecho.\n\
         # TYPE semantic_cache_ipc_requests_total counter\n",
    );
    for (kind, outcome, n) in ipc::requests() {
        let labels = format!(r#"kind="{kind}",outcome="{outcome}""#);
        line(&mut out, "semantic_cache_ipc_requests_total", &labels, n);
    }
    out.push_str(
        "# HELP semantic_cache_ipc_request_duration_seconds Latência dos pedidos IPC, do frame lido à resposta escrita.\n\
         # TYPE semantic_cache_ipc_request_duration_seconds histogram\n",
    );
    for (kind, latency) in ipc::request_seconds() {
        let name = "semantic_cache_ipc_request_duration_seconds";
        histogram(&mut out, name, &format!(r#"kind="{kind}""#), latency);
    }
    out.push_str(
        "# HELP semantic_cache_lookups_total LOOKUP atendidos, pelo resultado da busca.\n\
         # TYPE semantic_cache_lookups_total counter\n",
    );
    for (result, n) in ipc::lookups() {
        let labels = format!(r#"result="{result}""#);
        line(&mut out, "semantic_cache_lookups_total", &labels, n);
    }
    out.push_str(
        "# HELP semantic_cache_search_duration_seconds Latência das buscas top-k, por índice.\n\
         # TYPE semantic_cache_search_duration_seconds histogram\n",
    );
    for (index, latency) in ipc::search_seconds() {
        let name = "semantic_cache_search_duration_seconds";
        histogram(&mut out, name, &format!(r#"index="{index}""#), latency);
    }
    let entries = store.map_or(0, CacheStore::len);
    let (used, limit) = (budget.used(), budget.limit());
    let (connections, discarded) = (ipc::connections(), frame::discarded_frames());
    let _ = write!(
        out,
        "\
# HELP semantic_cache_entries Entradas no cache.
# TYPE semantic_cache_entries gauge
semantic_cache_entries {entries}
# HELP semantic_cache_memory_bytes Bytes reservados no orçamento de memória.
# TYPE semantic_cache_memory_bytes gauge
semantic_cache_memory_bytes {used}
# HELP semantic_cache_memory_limit_bytes Teto do orçamento de memória.
# TYPE semantic_cache_memory_limit_bytes gauge
semantic_cache_memory_limit_bytes {limit}
# HELP semantic_cache_ipc_connections Conexões IPC em atendimento.
# TYPE semantic_cache_ipc_connections gauge
semantic_cache_ipc_connections {connections}
# HELP semantic_cache_ipc_frames_discarded_total Frames IPC recusados pelo codec (magic, versão, flags, CRC ou tamanho).
# TYPE semantic_cache_ipc_frames_discarded_total counter
semantic_cache_ipc_frames_discarded_total {discarded}
"
    );
    out
}

/// Uma série com labels; escrever numa `String` não falha.
fn line(out: &mut String, name: &str, labels: &str, value: u64) {
    let _ = writeln!(out, "{name}{{{labels}}} {value}");
}

/// As séries de um histograma: `_bucket` cumulativo por `le`, terminando no
/// `+Inf` (o total), `_sum` em segundos e `_count`, igual ao `+Inf`.
fn histogram(out: &mut String, name: &str, labels: &str, latency: &ipc::Histogram) {
    let (cumulative, seconds) = latency.snapshot();
    let les = ipc::BUCKETS.iter().map(|&(_, le)| le).chain(["+Inf"]);
    for (le, n) in les.zip(cumulative) {
        let _ = writeln!(out, r#"{name}_bucket{{{labels},le="{le}"}} {n}"#);
    }
    let count = cumulative[ipc::BUCKETS.len()];
    let _ = writeln!(out, "{name}_count{{{labels}}} {count}");
    let _ = writeln!(out, "{name}_sum{{{labels}}} {seconds}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;

    /// `(família e tipo, labels de cada série)` na ordem do texto: a tabela do
    /// contrato da Fase 4 (os `_bucket` levam também o `le`).
    const CONTRACT: [(&str, &str); 9] = [
        ("semantic_cache_ipc_requests_total counter", "kind,outcome"),
        (
            "semantic_cache_ipc_request_duration_seconds histogram",
            "kind",
        ),
        ("semantic_cache_lookups_total counter", "result"),
        ("semantic_cache_search_duration_seconds histogram", "index"),
        ("semantic_cache_entries gauge", ""),
        ("semantic_cache_memory_bytes gauge", ""),
        ("semantic_cache_memory_limit_bytes gauge", ""),
        ("semantic_cache_ipc_connections gauge", ""),
        ("semantic_cache_ipc_frames_discarded_total counter", ""),
    ];

    #[test]
    fn the_contract_families_are_exposed_in_the_text_format() {
        let budget = MemoryBudget::new(&MemoryConfig::default()).unwrap();
        let text = render(&budget, None);
        let typed: Vec<_> = text
            .lines()
            .filter_map(|l| l.strip_prefix("# TYPE "))
            .collect();
        assert_eq!(typed, CONTRACT.map(|(family, _)| family));
        assert_eq!(text.matches("# HELP ").count(), CONTRACT.len());
        // As séries de cada família, com as chaves dos labels: histograma
        // emite `_bucket` (com o `le`), `_sum` e `_count`.
        let mut allowed = Vec::new();
        for (family, keys) in CONTRACT {
            let (name, kind) = family.split_once(' ').unwrap();
            if kind == "histogram" {
                allowed.push((format!("{name}_bucket"), format!("{keys},le")));
                allowed.push((format!("{name}_sum"), keys.to_owned()));
                allowed.push((format!("{name}_count"), keys.to_owned()));
            } else {
                allowed.push((name.to_owned(), keys.to_owned()));
            }
        }
        // Cada série: de uma família declarada, com os labels dela, valor ≥ 0.
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let (series, value) = line.rsplit_once(' ').unwrap();
            assert!(value.parse::<f64>().unwrap() >= 0.0, "{line}");
            let (name, labels) = series.split_once('{').unwrap_or((series, ""));
            let keys: Vec<_> = labels
                .split(',')
                .filter_map(|l| l.split_once('='))
                .collect();
            let keys = keys.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(",");
            assert!(allowed.contains(&(name.to_owned(), keys)), "{line}");
        }
        for (series, expected) in [
            ("semantic_cache_ipc_requests_total{", 6 * 4),
            ("semantic_cache_ipc_request_duration_seconds_bucket{", 6 * 9),
            ("semantic_cache_search_duration_seconds_bucket{", 3 * 9),
        ] {
            assert_eq!(text.matches(series).count(), expected, "{series}");
        }
    }

    /// Toda métrica citada nas regras de alerta e nas séries do teste delas
    /// existe no `/metrics`, e também cada valor de `kind` e `le` citado: nome
    /// errado não passa (a regra nunca dispararia, sem erro nenhum).
    #[test]
    fn the_alert_rules_only_cite_exported_metrics() {
        let budget = MemoryBudget::new(&MemoryConfig::default()).unwrap();
        let text = render(&budget, None);
        let series = text.lines().filter(|l| !l.starts_with('#'));
        let exported: Vec<_> = series.filter_map(|l| l.split(['{', ' ']).next()).collect();
        for rules in [
            include_str!("../../../deploy/prometheus/semantic-cache.rules.yml"),
            include_str!("../../../deploy/prometheus/semantic-cache.rules.test.yml"),
        ] {
            let cited: Vec<_> = rules
                .match_indices("semantic_cache_")
                .map(|(at, _)| {
                    let name = &rules[at..];
                    let end = name.find(|c: char| !c.is_ascii_alphanumeric() && c != '_');
                    &name[..end.unwrap_or(name.len())]
                })
                .collect();
            assert!(!cited.is_empty(), "nenhuma métrica citada");
            for name in cited {
                assert!(exported.contains(&name), "{name} não está no /metrics");
            }
            for key in [r#"kind=""#, r#"le=""#] {
                for (at, _) in rules.match_indices(key) {
                    let quote = at + key.len() + rules[at + key.len()..].find('"').unwrap();
                    let pair = &rules[at..=quote];
                    assert!(text.contains(pair), "{pair} não está no /metrics");
                }
            }
        }
    }

    /// O único teste da lib que observa latência: os outros só leem a forma.
    #[test]
    fn the_histograms_are_cumulative_and_their_inf_bucket_is_the_count() {
        use semantic_cache_ipc::frame::kind::LOOKUP_REQ;
        use semantic_cache_ipc::metrics::{request, search, Index, Outcome, BUCKETS};
        use std::time::Duration;

        for micros in [50, 150, 3_000, 30_000] {
            request(LOOKUP_REQ, Outcome::Ok, Duration::from_micros(micros));
        }
        search(Index::Hnsw, Duration::from_micros(400));
        let budget = MemoryBudget::new(&MemoryConfig::default()).unwrap();
        let text = render(&budget, None);
        let has = |line: &str| assert!(text.lines().any(|l| l == line), "{line}\n{text}");
        let les: Vec<_> = BUCKETS.iter().map(|b| b.1).chain(["+Inf"]).collect();
        let lookup = "semantic_cache_ipc_request_duration_seconds_bucket{kind=\"lookup\"";
        for (le, n) in les.iter().zip([1, 2, 2, 2, 2, 3, 3, 3, 4]) {
            has(&format!(r#"{lookup},le="{le}"}} {n}"#));
        }
        has(r#"semantic_cache_ipc_request_duration_seconds_sum{kind="lookup"} 0.0332"#);
        has(r#"semantic_cache_ipc_request_duration_seconds_count{kind="lookup"} 4"#);
        has(r#"semantic_cache_search_duration_seconds_bucket{index="hnsw",le="0.00025"} 0"#);
        has(r#"semantic_cache_search_duration_seconds_bucket{index="hnsw",le="0.0005"} 1"#);
        has(r#"semantic_cache_search_duration_seconds_sum{index="hnsw"} 0.0004"#);
        // Em toda série: os `le` na ordem do contrato, contagem que não desce,
        // e o `+Inf` igual ao `_count`.
        let buckets: Vec<_> = text.lines().filter(|l| l.contains("_bucket{")).collect();
        for series in buckets.chunks(les.len()) {
            let (head, _) = series[0].split_once(",le=").unwrap();
            let mut counts = Vec::new();
            for (line, le) in series.iter().zip(&les) {
                let (labels, n) = line.rsplit_once(' ').unwrap();
                assert_eq!(labels, format!(r#"{head},le="{le}"}}"#));
                counts.push(n.parse::<u64>().unwrap());
            }
            assert!(
                counts.windows(2).all(|w| w[0] <= w[1]),
                "{head}: {counts:?}"
            );
            let count = format!("{}}} {}", head.replace("_bucket{", "_count{"), counts[8]);
            has(&count);
        }
    }

    #[test]
    fn the_gauges_read_the_budget_at_scrape_time() {
        let config = MemoryConfig {
            max_bytes: 1 << 20,
            ..MemoryConfig::default()
        };
        let budget = MemoryBudget::new(&config).unwrap();
        let _held = budget.try_reserve(4096).unwrap();
        let text = render(&budget, None);
        for gauge in [
            "semantic_cache_memory_bytes 4096",
            "semantic_cache_memory_limit_bytes 1048576",
            "semantic_cache_entries 0",
        ] {
            assert!(text.lines().any(|l| l == gauge), "{gauge}\n{text}");
        }
    }
}
