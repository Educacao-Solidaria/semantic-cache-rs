//! Métricas do servidor IPC: contadores atômicos globais, como o
//! [`discarded_frames`](crate::frame::discarded_frames), e histogramas de
//! latência em [`BUCKETS`] de 100 µs a 25 ms. Quem atende (o
//! [`Basic`](crate::server::Basic), o despachante do PR 52/100) conta cada
//! pedido com [`request`], cada LOOKUP com [`lookup`] e cada busca com
//! [`search`]; o [`serve`](crate::server::serve) conta as conexões. Labels
//! fixos (contrato da Fase 4): medir é um `fetch_add`, sem hash nem trava.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

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
/// Label `index`, na ordem de [`Index`].
const INDEXES: [&str; 3] = ["flat_f32", "flat_i8", "hnsw"];

/// Buckets de latência do contrato da Fase 4 (100 µs a 5 ms, mais a cauda de
/// 10 e 25 ms), `(nanossegundos, texto do le)`; acima do último, `+Inf`.
pub const BUCKETS: [(u64, &str); 8] = [
    (100_000, "0.0001"),
    (250_000, "0.00025"),
    (500_000, "0.0005"),
    (1_000_000, "0.001"),
    (2_000_000, "0.002"),
    (5_000_000, "0.005"),
    (10_000_000, "0.01"),
    (25_000_000, "0.025"),
];

/// O máximo que uma observação soma (2⁴⁰ ns, ~18 min): um `Duration` absurdo
/// não dá a volta na soma; tempo real só a gira depois de séculos somados.
const MAX_NANOS: u64 = 1 << 40;

static REQUESTS: [[AtomicU64; 4]; 6] = [const { [const { AtomicU64::new(0) }; 4] }; 6];
static REQUEST_SECONDS: [Histogram; 6] = [const { Histogram::new() }; 6];
static SEARCH_SECONDS: [Histogram; 3] = [const { Histogram::new() }; 3];
static LOOKUPS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static CONNECTIONS: AtomicU64 = AtomicU64::new(0);

/// Histograma de latência sem trava: um contador por bucket (não cumulativo;
/// o último é o `+Inf`) e a soma em nanossegundos.
#[derive(Debug, Default)]
pub struct Histogram {
    counts: [AtomicU64; BUCKETS.len() + 1],
    sum_nanos: AtomicU64,
}

impl Histogram {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [const { AtomicU64::new(0) }; BUCKETS.len() + 1],
            sum_nanos: AtomicU64::new(0),
        }
    }

    /// Conta `elapsed` no primeiro bucket com `le >= elapsed` (o `le` do
    /// Prometheus é "menor ou igual") e o soma.
    pub fn observe(&self, elapsed: Duration) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        let at = BUCKETS.partition_point(|&(le, _)| le < nanos);
        self.counts[at].fetch_add(1, Relaxed);
        self.sum_nanos.fetch_add(nanos.min(MAX_NANOS), Relaxed);
    }

    /// As contagens cumulativas, uma por bucket de [`BUCKETS`] e por último a
    /// do `+Inf` (o total, o `_count`), e a soma em segundos. Uma observação
    /// concorrente pode já estar nas contagens e ainda não na soma.
    #[must_use]
    pub fn snapshot(&self) -> ([u64; BUCKETS.len() + 1], f64) {
        let mut seen = 0;
        let cumulative = std::array::from_fn(|i| {
            seen += self.counts[i].load(Relaxed);
            seen
        });
        #[allow(clippy::cast_precision_loss)] // exato até 2⁵³ ns (104 dias somados)
        let seconds = self.sum_nanos.load(Relaxed) as f64 / 1e9;
        (cumulative, seconds)
    }
}

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

/// Conta um pedido de `kind` (o byte do frame) que terminou em `outcome` e
/// levou `elapsed`, do frame lido à resposta escrita (ou à falha da escrita).
pub fn request(kind: u8, outcome: Outcome, elapsed: Duration) {
    let at = kind_index(kind);
    REQUESTS[at][outcome as usize].fetch_add(1, Relaxed);
    REQUEST_SECONDS[at].observe(elapsed);
}

/// Índice que respondeu uma busca (label `index`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Index {
    /// Varredura exata em f32.
    FlatF32,
    /// Varredura exata nos códigos int8 do `QuantizedVector`.
    FlatI8,
    /// O HNSW do PR 32/100.
    Hnsw,
}

/// Registra uma busca top-k em `index` que levou `elapsed`: encaixe para
/// quem busca (o binário ainda não busca).
pub fn search(index: Index, elapsed: Duration) {
    SEARCH_SECONDS[index as usize].observe(elapsed);
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

/// `(kind, latência)` de cada kind.
pub fn request_seconds() -> impl Iterator<Item = (&'static str, &'static Histogram)> {
    KINDS.into_iter().zip(&REQUEST_SECONDS)
}

/// `(index, latência da busca)` de cada índice.
pub fn search_seconds() -> impl Iterator<Item = (&'static str, &'static Histogram)> {
    INDEXES.into_iter().zip(&SEARCH_SECONDS)
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

    /// O texto do `le` é o próprio limite em segundos, e a lista é a do contrato.
    #[test]
    #[allow(clippy::cast_precision_loss)] // limites < 2⁵³
    fn the_buckets_are_the_contract_ones_and_each_le_is_its_limit() {
        let contract = "0.0001 0.00025 0.0005 0.001 0.002 0.005 0.01 0.025";
        assert_eq!(BUCKETS.map(|(_, le)| le).join(" "), contract);
        for (nanos, le) in BUCKETS {
            assert_eq!(le.parse::<f64>().unwrap() * 1e9, nanos as f64, "{le}");
        }
    }

    #[test]
    fn each_observation_falls_in_the_first_bucket_that_holds_it() {
        let empty = Histogram::new().snapshot();
        assert_eq!(empty, ([0; 9], 0.0));
        let (micros, nanos) = (Duration::from_micros, Duration::from_nanos);
        let h = Histogram::new();
        for elapsed in [
            Duration::ZERO,
            micros(100), // no limite: cabe no `le` de 100 µs
            nanos(100_001),
            micros(1_000),
            micros(25_000),
            nanos(25_000_001),
            Duration::MAX, // soma só o teto
        ] {
            h.observe(elapsed);
        }
        let (cumulative, seconds) = h.snapshot();
        assert_eq!(cumulative, [2, 3, 3, 4, 4, 4, 4, 5, 7]);
        let sum = 100_000 + 100_001 + 1_000_000 + 25_000_000 + 25_000_001 + MAX_NANOS;
        #[allow(clippy::cast_precision_loss)] // < 2⁵³
        let expected = sum as f64 / 1e9;
        assert_eq!(seconds.to_bits(), expected.to_bits());
    }

    /// O único teste da lib que chama `lookup`: os outros não sujam a conta.
    #[test]
    fn a_lookup_counts_under_its_own_result() {
        lookup(Lookup::Miss);
        let counts: Vec<_> = lookups().collect();
        assert_eq!(counts, [("hit", 0), ("miss", 1), ("error", 0)]);
    }
}
