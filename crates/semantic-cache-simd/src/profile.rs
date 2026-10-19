//! Profiler embutido nas operações críticas: o tempo de cada quantização,
//! busca e descompressão vai para um histograma por operação, de onde saem a
//! contagem, o p50, o p99 e o máximo ([`snapshot`]).
//!
//! - Desligado (o padrão), custa uma leitura atômica por chamada; liga-se com
//!   [`set_enabled`], em produção, sem recompilar.
//! - Ligado, cada chamada custa dois `Instant::now` e um incremento atômico,
//!   sem trava: o histograma é um vetor de contadores atômicos.
//! - Buckets log-lineares em nanossegundos, 16 por oitava: cada percentil é o
//!   teto do bucket em que caiu, no máximo 1/16 (6,25%) acima do valor real.
//!
//! ponytail: as threads incrementam os mesmos contadores; com muitos núcleos
//! medindo a mesma operação, a linha de cache do bucket mais comum vira
//! disputa. O caminho é um histograma por thread, somado no retrato.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// Operações medidas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// [`crate::QuantizedVector::quantize`].
    Quantize,
    /// Uma consulta top-k inteira: quem implementa a busca a envolve com
    /// [`time`].
    Search,
    /// [`crate::QuantizedVector::dequantize`].
    Dequantize,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Um por variante de [`Op`], na ordem da declaração.
static HISTOGRAMS: [Histogram; 3] = [const { Histogram::new() }; 3];

/// Liga ou desliga a coleta, para todas as threads.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Relaxed);
}

/// `#[inline]`: é a única coisa que o caminho desligado executa, inclusive
/// em quem chama de outro crate.
#[inline]
#[must_use]
pub fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

/// Roda `f` e, com a coleta ligada, registra em `op` quanto ela levou.
#[inline]
pub fn time<R>(op: Op, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    let out = f();
    HISTOGRAMS[op as usize].record(start.elapsed());
    out
}

/// O que `op` registrou desde o último [`reset`]; `None` sem registro.
#[must_use]
pub fn snapshot(op: Op) -> Option<Snapshot> {
    HISTOGRAMS[op as usize].snapshot()
}

/// Zera os histogramas. Um registro concorrente com o reset pode ficar ou não.
pub fn reset() {
    HISTOGRAMS.iter().for_each(Histogram::reset);
}

/// Retrato de uma operação. Percentis por posto mais próximo (o p99 de 1000
/// registros é o 990º menor), cada um o teto do bucket em que caiu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    pub count: u64,
    pub p50: Duration,
    pub p99: Duration,
    pub max: Duration,
}

/// Bits depois do mais alto que separam os buckets: 2⁴ = 16 por oitava.
const SUB_BITS: u32 = 4;
const SUB: u64 = 1 << SUB_BITS;
/// Um bucket por valor abaixo de `SUB`, e `SUB` por oitava de 2⁴ a 2⁶³.
const BUCKETS: usize = (64 - SUB_BITS as usize + 1) << SUB_BITS;

struct Histogram {
    counts: [AtomicU64; BUCKETS],
}

impl Histogram {
    const fn new() -> Self {
        Self {
            counts: [const { AtomicU64::new(0) }; BUCKETS],
        }
    }

    fn record(&self, elapsed: Duration) {
        let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.counts[bucket(ns)].fetch_add(1, Relaxed);
    }

    fn snapshot(&self) -> Option<Snapshot> {
        let counts: [u64; BUCKETS] = std::array::from_fn(|i| self.counts[i].load(Relaxed));
        let count: u64 = counts.iter().sum();
        if count == 0 {
            return None;
        }
        // O menor bucket com pelo menos ⌈n·p/100⌉ registros até ele.
        let rank = |p: u128| {
            let want = (u128::from(count) * p).div_ceil(100);
            let mut seen = 0;
            let at = counts.iter().position(|&c| {
                seen += u128::from(c);
                seen >= want
            });
            Duration::from_nanos(at.map_or(u64::MAX, upper))
        };
        Some(Snapshot {
            count,
            p50: rank(50),
            p99: rank(99),
            max: rank(100),
        })
    }

    fn reset(&self) {
        self.counts.iter().for_each(|c| c.store(0, Relaxed));
    }
}

/// Bucket de `ns`: o próprio valor abaixo de `SUB`; acima, a oitava e os
/// `SUB_BITS` bits seguintes ao mais alto.
#[allow(clippy::cast_possible_truncation)] // índice < BUCKETS (976)
fn bucket(ns: u64) -> usize {
    if ns < SUB {
        return ns as usize;
    }
    let shift = ns.ilog2() - SUB_BITS;
    let mantissa = (ns >> shift) - SUB; // em [0, SUB)
    ((shift as usize + 1) << SUB_BITS) + mantissa as usize
}

/// Maior valor que cai no bucket `i`.
fn upper(i: usize) -> u64 {
    let (octave, mantissa) = (i as u64 / SUB, i as u64 % SUB);
    if octave == 0 {
        return mantissa;
    }
    let shift = octave - 1;
    ((SUB + mantissa) << shift) + ((1 << shift) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// De 0 a 2⁶⁴ − 1: os pequenos, as bordas de cada oitava e uma sequência
    /// pseudoaleatória de todas as magnitudes.
    fn samples() -> impl Iterator<Item = u64> {
        let edges = (0..64).flat_map(|b| [(1u64 << b) - 1, 1 << b, (1 << b) + 1]);
        let mut x = 0x9E37_79B9_7F4A_7C15_u64;
        let random = std::iter::repeat_with(move || {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            x >> (x % 64)
        });
        (0..1000)
            .chain(edges)
            .chain([u64::MAX])
            .chain(random.take(10_000))
    }

    #[test]
    fn each_value_lands_in_its_bucket_at_most_a_sixteenth_below_the_top() {
        for ns in samples() {
            let i = bucket(ns);
            let top = upper(i);
            assert!(i < BUCKETS && ns <= top, "{ns} no bucket {i}, teto {top}");
            assert!(
                i == 0 || upper(i - 1) < ns,
                "{ns} cabia no bucket {}",
                i - 1
            );
            assert!(top - ns <= ns / SUB, "{ns}: teto {top} longe demais");
        }
        assert_eq!(bucket(u64::MAX), BUCKETS - 1);
    }

    #[test]
    fn percentiles_are_nearest_rank_rounded_up_to_the_bucket() {
        let h = Histogram::new();
        assert_eq!(h.snapshot(), None, "vazio");
        // Abaixo de 16 ns cada valor tem bucket próprio: exato. Com 10
        // registros, o p99 já é o máximo (⌈9,9⌉ = 10º).
        (1..=10).for_each(|ns| h.record(Duration::from_nanos(ns)));
        let ns = Duration::from_nanos;
        let exact = Snapshot {
            count: 10,
            p50: ns(5),
            p99: ns(10),
            max: ns(10),
        };
        assert_eq!(h.snapshot(), Some(exact));
        // 1..=1000 µs: p50 500 µs, p99 990 µs, máximo 1000 µs, cada um no
        // teto do bucket (até 1/16 acima).
        h.reset();
        (1..=1000)
            .rev()
            .for_each(|us| h.record(Duration::from_micros(us)));
        let s = h.snapshot().unwrap();
        assert_eq!(s.count, 1000);
        for (got, us) in [(s.p50, 500), (s.p99, 990), (s.max, 1000)] {
            let want = Duration::from_micros(us);
            assert!(
                want <= got && got <= want + want / 16,
                "{got:?} para {want:?}"
            );
        }
    }

    #[test]
    fn zero_and_huge_durations_are_counted_at_the_ends() {
        let h = Histogram::new();
        h.record(Duration::ZERO);
        h.record(Duration::MAX); // mais que u64::MAX ns: satura
        let (zero, top) = (Duration::ZERO, Duration::from_nanos(u64::MAX));
        let s = h.snapshot().unwrap();
        assert_eq!((s.count, s.p50, s.p99, s.max), (2, zero, top, top));
    }

    #[test]
    fn concurrent_records_are_all_counted() {
        let h = Histogram::new();
        std::thread::scope(|s| {
            for t in 0..8 {
                let h = &h;
                s.spawn(move || {
                    for i in 0..10_000 {
                        h.record(Duration::from_nanos(t * 1000 + i % 100));
                    }
                });
            }
        });
        assert_eq!(h.snapshot().unwrap().count, 80_000);
    }
}
