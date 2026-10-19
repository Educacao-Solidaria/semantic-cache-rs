//! Harness de estresse: insere uma coleção grande de vetores num índice,
//! roda consultas top-k e mede cada operação (p50, p95, p99 e máximo), com o
//! recall contra a busca exata.
//!
//! O índice entra por [`VectorIndex`]. Hoje há as varreduras lineares
//! ([`FlatF32`] e [`FlatI8`]); um índice aproximado (HNSW) implementa o mesmo
//! trait e é medido do mesmo jeito, contra a mesma meta ([`P99_TARGET`]).

use std::time::{Duration, Instant};

use semantic_cache_simd::profile::{self, Op};
use semantic_cache_simd::QuantizedVector;

use crate::fixtures::VectorGen;

/// Meta do projeto para a consulta: p99 abaixo de 2 ms.
pub const P99_TARGET: Duration = Duration::from_millis(2);

/// Um resultado de consulta: o id (ordem de inserção) e a similaridade.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub score: f32,
}

/// Índice medido pelo harness.
pub trait VectorIndex {
    /// Guarda `vector` com o próximo id (0, 1, 2…).
    fn insert(&mut self, vector: &[f32]);

    /// Os `k` ids mais similares a `query`, do mais para o menos similar.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
}

/// Varredura linear exata em `f32`, com o produto escalar do kernel SIMD
/// ativo (cosseno, para vetores unitários). Os vetores ficam contíguos.
#[derive(Debug, Default)]
pub struct FlatF32 {
    dim: usize,
    data: Vec<f32>,
}

impl FlatF32 {
    /// Já reserva `vectors · dim` floats: sem a cópia de crescer o `Vec`.
    #[must_use]
    pub fn with_capacity(vectors: usize, dim: usize) -> Self {
        Self {
            dim,
            data: Vec::with_capacity(vectors.saturating_mul(dim)),
        }
    }
}

impl VectorIndex for FlatF32 {
    /// # Panics
    ///
    /// Vetor vazio ou de dimensão diferente da dos anteriores.
    fn insert(&mut self, vector: &[f32]) {
        if self.data.is_empty() {
            self.dim = vector.len();
        }
        assert!(
            !vector.is_empty() && vector.len() == self.dim,
            "dimensão {}",
            vector.len()
        );
        self.data.extend_from_slice(vector);
    }

    /// # Panics
    ///
    /// Consulta de dimensão diferente da dos vetores guardados.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        profile::time(Op::Search, || {
            let mut top = TopK::new(k);
            for (id, row) in self.data.chunks_exact(self.dim.max(1)).enumerate() {
                top.push(
                    id,
                    semantic_cache_simd::dot(query, row).expect("mesma dimensão"),
                );
            }
            top.hits
        })
    }
}

/// Varredura linear sobre vetores quantizados em `i8`, com o cosseno inteiro
/// do kernel SIMD ativo: um quarto da memória lida por consulta.
#[derive(Debug, Default)]
pub struct FlatI8 {
    vectors: Vec<QuantizedVector>,
}

impl VectorIndex for FlatI8 {
    /// # Panics
    ///
    /// Vetor com `NaN` ou infinito.
    fn insert(&mut self, vector: &[f32]) {
        self.vectors
            .push(QuantizedVector::quantize(vector).expect("vetor finito"));
    }

    /// # Panics
    ///
    /// Consulta não finita, de dimensão diferente ou toda zero.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        profile::time(Op::Search, || {
            let query = QuantizedVector::quantize(query).expect("consulta finita");
            let mut top = TopK::new(k);
            for (id, vector) in self.vectors.iter().enumerate() {
                top.push(
                    id,
                    query
                        .cosine(vector)
                        .expect("mesma dimensão, norma não nula"),
                );
            }
            top.hits
        })
    }
}

/// Os `k` maiores escores vistos, em ordem decrescente; empate fica com o id
/// menor, e `NaN` nunca entra.
struct TopK {
    k: usize,
    hits: Vec<Hit>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            hits: Vec::with_capacity(k.saturating_add(1).min(1024)),
        }
    }

    fn push(&mut self, id: usize, score: f32) {
        let full = self.hits.len() == self.k;
        if score.is_nan() || (full && self.hits.last().is_none_or(|worst| score <= worst.score)) {
            return;
        }
        let at = self.hits.partition_point(|hit| hit.score >= score);
        self.hits.insert(at, Hit { id, score });
        self.hits.truncate(self.k);
    }
}

/// Latência de uma série de operações, por posto mais próximo
/// (nearest-rank): o p99 de 1000 amostras é a 990ª menor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Latency {
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub max: Duration,
}

impl Latency {
    /// `None` sem amostras. Ordena `samples`.
    #[must_use]
    pub fn of(samples: &mut [Duration]) -> Option<Self> {
        samples.sort_unstable();
        let rank = |p: usize| samples[(samples.len() * p).div_ceil(100) - 1];
        let max = *samples.last()?;
        Some(Self {
            p50: rank(50),
            p95: rank(95),
            p99: rank(99),
            max,
        })
    }
}

/// O que uma rodada mediu.
#[derive(Debug)]
pub struct Report {
    pub insert: Latency,
    pub search: Latency,
    /// Fração dos top-k exatos que o índice achou (`None` sem referência).
    pub recall: Option<f64>,
    /// Os resultados de cada consulta: servem de referência para o próximo
    /// índice quando este é exato.
    pub results: Vec<Vec<Hit>>,
}

/// Uma carga: `vectors` vetores unitários de `dim` dimensões e `queries`
/// consultas top-`k`, todos da mesma semente (mesma carga para todo índice).
#[derive(Debug, Clone, Copy)]
pub struct Load {
    pub seed: u64,
    pub vectors: usize,
    pub dim: usize,
    pub queries: usize,
    pub k: usize,
}

impl Load {
    /// Mede `index` nesta carga. Gerar os vetores fica fora do cronômetro;
    /// `truth` são os resultados de um índice exato na mesma carga.
    ///
    /// # Panics
    ///
    /// Com `vectors`, `queries` ou `dim` zero (não há o que medir).
    pub fn run(&self, index: &mut dyn VectorIndex, truth: Option<&[Vec<Hit>]>) -> Report {
        let mut gen = VectorGen::new(self.seed);
        let mut inserts = Vec::with_capacity(self.vectors);
        for _ in 0..self.vectors {
            let vector = gen.unit(self.dim);
            let start = Instant::now();
            index.insert(&vector.data);
            inserts.push(start.elapsed());
        }
        let mut searches = Vec::with_capacity(self.queries);
        let mut results = Vec::with_capacity(self.queries);
        for _ in 0..self.queries {
            let query = gen.unit(self.dim);
            let start = Instant::now();
            let hits = index.search(&query.data, self.k);
            searches.push(start.elapsed());
            results.push(hits);
        }
        Report {
            insert: Latency::of(&mut inserts).expect("vectors > 0"),
            search: Latency::of(&mut searches).expect("queries > 0"),
            recall: truth.map(|truth| recall(truth, &results)),
            results,
        }
    }
}

/// Média, por consulta, da fração dos ids de `truth` presentes em `got`.
#[allow(clippy::cast_precision_loss)]
fn recall(truth: &[Vec<Hit>], got: &[Vec<Hit>]) -> f64 {
    let found = |(want, have): (&Vec<Hit>, &Vec<Hit>)| {
        let hits = want
            .iter()
            .filter(|w| have.iter().any(|h| h.id == w.id))
            .count();
        hits as f64 / want.len().max(1) as f64
    };
    truth.iter().zip(got).map(found).sum::<f64>() / truth.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_is_nearest_rank() {
        // 1..=1000 ms embaralhados: o p99 é a 990ª menor amostra.
        let mut samples: Vec<Duration> = (1..=1000)
            .map(|i| Duration::from_millis((i * 7919) % 1000 + 1))
            .collect();
        let latency = Latency::of(&mut samples).unwrap();
        let at = Duration::from_millis;
        assert_eq!(
            (latency.p50, latency.p95, latency.p99, latency.max),
            (at(500), at(950), at(990), at(1000))
        );
        // Com 10 amostras o p95 e o p99 já são o máximo: nada de interpolar
        // para baixo.
        let ten = Latency::of(&mut (1..=10).map(at).collect::<Vec<_>>()).unwrap();
        assert_eq!((ten.p50, ten.p95, ten.p99), (at(5), at(10), at(10)));
        let one = Latency::of(&mut [at(3)]).unwrap();
        assert_eq!((one.p50, one.p99, one.max), (at(3), at(3), at(3)));
        assert_eq!(Latency::of(&mut []), None);
    }

    #[test]
    fn top_k_keeps_the_best_in_order_without_nan() {
        let mut top = TopK::new(3);
        for (id, score) in [0.1, 0.9, f32::NAN, 0.5, 0.9, -1.0, 0.2]
            .into_iter()
            .enumerate()
        {
            top.push(id, score);
        }
        let ids: Vec<usize> = top.hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, [1, 4, 3], "empate com o id menor primeiro");
        let mut none = TopK::new(0);
        none.push(0, 1.0);
        assert_eq!(none.hits, Vec::<Hit>::new());
    }

    /// Índices sobre 300 vetores de 32 dimensões, mais um vizinho plantado
    /// (cosseno 0,99 com a consulta) no id 137.
    fn planted(index: &mut dyn VectorIndex) -> Vec<f32> {
        let mut gen = VectorGen::new(7);
        let query = gen.unit(32);
        for id in 0..300 {
            let vector = if id == 137 {
                gen.with_cosine(&query, 0.99)
            } else {
                gen.unit(32)
            };
            index.insert(&vector.data);
        }
        query.data
    }

    #[test]
    fn flat_indexes_find_the_planted_neighbour() {
        let indexes: [Box<dyn VectorIndex>; 2] =
            [Box::new(FlatF32::default()), Box::new(FlatI8::default())];
        for mut index in indexes {
            let query = planted(index.as_mut());
            let hits = index.search(&query, 5);
            assert_eq!((hits.len(), hits[0].id), (5, 137));
            assert!((hits[0].score - 0.99).abs() < 5e-3, "{}", hits[0].score);
            assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
            assert_eq!(
                index.search(&query, 1000).len(),
                300,
                "k maior que o índice"
            );
        }
        let empty = FlatF32::default().search(&[1.0], 3);
        assert_eq!(empty, Vec::<Hit>::new(), "índice vazio");
    }

    #[test]
    #[should_panic(expected = "dimensão 3")]
    fn flat_f32_rejects_a_different_dimension() {
        let mut index = FlatF32::default();
        index.insert(&[1.0, 0.0]);
        index.insert(&[1.0, 0.0, 0.0]);
    }

    #[test]
    fn run_measures_latency_and_recall_against_the_exact_search() {
        struct Blind;
        impl VectorIndex for Blind {
            fn insert(&mut self, _: &[f32]) {}
            fn search(&self, _: &[f32], _: usize) -> Vec<Hit> {
                Vec::new()
            }
        }
        let load = Load {
            seed: 3,
            vectors: 400,
            dim: 32,
            queries: 25,
            k: 5,
        };
        let exact = load.run(&mut FlatF32::default(), None);
        assert_eq!(exact.recall, None);
        assert!(exact.results.iter().all(|hits| hits.len() == 5));
        assert!(exact.search.p50 <= exact.search.p99 && exact.search.p99 <= exact.search.max);
        let truth = Some(exact.results.as_slice());
        assert_eq!(load.run(&mut FlatF32::default(), truth).recall, Some(1.0));
        let int8 = load.run(&mut FlatI8::default(), truth).recall.unwrap();
        assert!((0.8..=1.0).contains(&int8), "recall int8 {int8}");
        assert_eq!(
            load.run(&mut Blind, truth).recall,
            Some(0.0),
            "recall pode falhar"
        );
    }
}
