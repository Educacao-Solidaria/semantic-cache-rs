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
        "# HELP semantic_cache_lookups_total LOOKUP atendidos, pelo resultado da busca.\n\
         # TYPE semantic_cache_lookups_total counter\n",
    );
    for (result, n) in ipc::lookups() {
        let labels = format!(r#"result="{result}""#);
        line(&mut out, "semantic_cache_lookups_total", &labels, n);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MemoryConfig;

    /// `(família e tipo, labels de cada série)` na ordem do texto: a tabela do
    /// contrato da Fase 4, menos os histogramas (PR 79/100).
    const CONTRACT: [(&str, &str); 7] = [
        ("semantic_cache_ipc_requests_total counter", "kind,outcome"),
        ("semantic_cache_lookups_total counter", "result"),
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
        // Cada série: de uma família declarada, com os labels dela, valor inteiro.
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let (series, value) = line.rsplit_once(' ').unwrap();
            assert!(value.parse::<u64>().is_ok(), "{line}");
            let (name, labels) = series.split_once('{').unwrap_or((series, ""));
            let keys: Vec<_> = labels
                .split(',')
                .filter_map(|l| l.split_once('='))
                .collect();
            let keys = keys.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(",");
            let family = CONTRACT
                .iter()
                .find(|c| c.0.split(' ').next() == Some(name));
            assert_eq!(family.map(|c| c.1), Some(keys.as_str()), "{line}");
        }
        let requests = text.matches("semantic_cache_ipc_requests_total{").count();
        assert_eq!(requests, 6 * 4, "toda combinação de kind e outcome");
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
