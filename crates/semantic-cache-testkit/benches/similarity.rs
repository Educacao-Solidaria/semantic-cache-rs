//! Linha de base das métricas de similaridade de referência (escalares) do
//! crate de domínio, nas dimensões de embedding usadas em produção. Os kernels
//! SIMD das próximas fases são medidos contra estes números.
//!
//! `cargo bench` roda a rodada completa; `SEMCACHE_BENCH_QUICK=1 cargo bench`
//! roda a curta.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use semantic_cache_rs::math::{dot_product, euclidean_distance};
use semantic_cache_rs::{CosineDistance, DistanceMetric, NormalizedVector, Vector};
use semantic_cache_testkit::bench::{Precision, EMBEDDING_DIMS};

/// Vetor determinístico de norma não nula; `phase` distingue `a` de `b`.
fn wave(dim: usize, phase: f32) -> Vector {
    let mut x = phase;
    let data = (0..dim)
        .map(|_| {
            x += 0.618;
            x.sin()
        })
        .collect();
    Vector::new(data).expect("dim > 0")
}

fn similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");
    for dim in EMBEDDING_DIMS {
        let (a, b) = (wave(dim, 0.0), wave(dim, 1.0));
        let na = NormalizedVector::new(a.clone()).expect("norma não nula");
        let nb = NormalizedVector::new(b.clone()).expect("norma não nula");
        // Elementos/s: compara dimensões diferentes na mesma escala.
        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bench, _| {
            bench.iter(|| black_box(&a).dot(black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bench, _| {
            bench.iter(|| black_box(&a).cosine_similarity(black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("cosine_prenorm", dim), &dim, |bench, _| {
            bench.iter(|| black_box(&na).cosine_similarity_to(black_box(&nb)));
        });
        group.bench_with_input(
            BenchmarkId::new("cosine_distance", dim),
            &dim,
            |bench, _| {
                bench.iter(|| CosineDistance.distance(black_box(&a), black_box(&b)));
            },
        );
        // API de slice: o formato que os kernels SIMD vão substituir.
        group.bench_with_input(BenchmarkId::new("dot_slice", dim), &dim, |bench, _| {
            bench.iter(|| dot_product(black_box(&a.data), black_box(&b.data)));
        });
        group.bench_with_input(BenchmarkId::new("euclidean", dim), &dim, |bench, _| {
            bench.iter(|| euclidean_distance(black_box(&a.data), black_box(&b.data)));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Precision::from_env().criterion();
    targets = similarity
}
criterion_main!(benches);
