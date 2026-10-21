//! Métricas do servidor IPC: contadores atômicos globais, como o
//! [`discarded_frames`](crate::frame::discarded_frames). Quem atende (o
//! [`Basic`](crate::server::Basic), o despachante do PR 52/100) conta cada
//! pedido com [`request`] e cada LOOKUP com [`lookup`]; o
//! [`serve`](crate::server::serve) conta as conexões. Labels fixos (contrato da
//! Fase 4): medir é um `fetch_add`, sem hash nem trava.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::frame::kind;

/// Label `kind`: o kind do pedido em minúsculas; kind fora do v1 vira `unknown`.
const KINDS: [&str; 6] = [
    "lookup",
    "store",
    "stats",
    "lookup_stream",
    "ping",
    "unknown",
];
/// Label `outcome`, na ordem de [`Outcome`].
const OUTCOMES: [&str; 4] = ["ok", "error", "timeout", "canceled"];
/// Label `result`, na ordem de [`Lookup`].
const RESULTS: [&str; 3] = ["hit", "miss", "error"];

static REQUESTS: [[AtomicU64; 4]; 6] = [const { [const { AtomicU64::new(0) }; 4] }; 6];
static LOOKUPS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static CONNECTIONS: AtomicU64 = AtomicU64::new(0);

/// Como terminou um pedido.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Respondido sem ERROR.
    Ok,
    /// Respondido com ERROR (`unsupported`, `bad_request`, `internal`...).
    Error,
    /// A resposta parou no prazo de ociosidade: o par não lê.
    Timeout,
    /// O par foi embora antes de receber a resposta.
    Canceled,
}

impl Outcome {
    /// O desfecho de um pedido respondido com `self` (`Ok` ou `Error`) cuja
    /// resposta foi escrita com `written`.
    #[must_use]
    pub fn after_write(self, written: &io::Result<()>) -> Self {
        match written {
            Ok(()) => self,
            Err(err) if err.kind() == io::ErrorKind::TimedOut => Self::Timeout,
            Err(_) => Self::Canceled,
        }
    }
}

/// Resultado de um LOOKUP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup {
    Hit,
    Miss,
    /// A busca falhou (pedido ilegível, dimensão errada, erro do índice).
    Error,
}

fn kind_index(k: u8) -> usize {
    match k {
        kind::LOOKUP_REQ => 0,
        kind::STORE_REQ => 1,
        kind::STATS_REQ => 2,
        kind::LOOKUP_STREAM_REQ => 3,
        kind::PING => 4,
        _ => 5,
    }
}

/// Conta um pedido de `kind` (o byte do frame) que terminou em `outcome`.
pub fn request(kind: u8, outcome: Outcome) {
    REQUESTS[kind_index(kind)][outcome as usize].fetch_add(1, Relaxed);
}

/// Conta um LOOKUP pelo resultado.
pub fn lookup(result: Lookup) {
    LOOKUPS[result as usize].fetch_add(1, Relaxed);
}

/// `(kind, outcome, pedidos)` de todas as combinações, as zeradas inclusive.
pub fn requests() -> impl Iterator<Item = (&'static str, &'static str, u64)> {
    let row = |(k, counts): (&'static str, &'static [AtomicU64; 4])| {
        let count = move |(o, n): (&'static str, &AtomicU64)| (k, o, n.load(Relaxed));
        OUTCOMES.into_iter().zip(counts).map(count)
    };
    KINDS.into_iter().zip(&REQUESTS).flat_map(row)
}

/// `(result, LOOKUP)` dos três resultados.
pub fn lookups() -> impl Iterator<Item = (&'static str, u64)> {
    RESULTS
        .into_iter()
        .zip(&LOOKUPS)
        .map(|(r, n)| (r, n.load(Relaxed)))
}

/// Conexões sendo atendidas agora (as recusadas pelo teto não contam).
#[must_use]
pub fn connections() -> u64 {
    CONNECTIONS.load(Relaxed)
}

/// Uma conexão em atendimento: conta ao nascer e desconta no `Drop`, que vem
/// com a task dela (fim, erro, pânico ou aborto).
#[derive(Debug)]
pub(crate) struct Open(());

impl Open {
    pub(crate) fn new() -> Self {
        CONNECTIONS.fetch_add(1, Relaxed);
        Self(())
    }
}

impl Drop for Open {
    fn drop(&mut self) {
        CONNECTIONS.fetch_sub(1, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O resto (timeout, cancelado, contagem, conexões) está em `tests/metrics.rs`.
    #[test]
    fn each_v1_request_kind_has_its_label_and_any_other_byte_is_unknown() {
        use kind::*;
        let named = [LOOKUP_REQ, STORE_REQ, STATS_REQ, LOOKUP_STREAM_REQ, PING];
        assert_eq!(named.map(kind_index), [0, 1, 2, 3, 4]);
        let others = [0x00, LOOKUP_RESP, PONG, ERROR, 0xFF];
        assert!(others.iter().all(|&k| KINDS[kind_index(k)] == "unknown"));
    }

    /// O único teste da lib que chama `lookup`: os outros não sujam a conta.
    #[test]
    fn a_lookup_counts_under_its_own_result() {
        lookup(Lookup::Miss);
        let counts: Vec<_> = lookups().collect();
        assert_eq!(counts, [("hit", 0), ("miss", 1), ("error", 0)]);
    }
}
