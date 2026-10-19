//! Linha de base das métricas de similaridade de referência (escalares) do
//! crate de domínio, nas dimensões de embedding usadas em produção, e os
//! kernels do `semantic-cache-simd` medidos contra ela (`dot_<kernel>` e
//! `cosine_<kernel>`; `SEMCACHE_SIMD=scalar` mede o mesmo caminho sem SIMD),
//! inclusive sobre vetores quantizados em `i8` (`dot_i8_<kernel>`,
//! `cosine_i8_<kernel>`). `quantize` e `dequantize` medem a conversão, e
//! `quantize_profiled` a mesma quantização com o profiler embutido ligado.
//! `dot_<kernel>_aligned` e `dot_<kernel>_misaligned` separam o efeito do
//! alinhamento: os mesmos valores num `AlignedVector` e 16 bytes depois de um
//! múltiplo de 32 (o `dot_<kernel>` cai num ou noutro conforme o alocador).
//!
//! `cargo bench` roda a rodada completa; `SEMCACHE_BENCH_QUICK=1 cargo bench`
//! roda a curta.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use semantic_cache_rs::math::{dot_product, euclidean_distance};
use semantic_cache_rs::{CosineDistance, DistanceMetric, NormalizedVector};
use semantic_cache_simd::profile;
use semantic_cache_simd::{AlignedVector, Kernel, QuantizedVector};
use semantic_cache_testkit::bench::{Precision, EMBEDDING_DIMS};
use semantic_cache_testkit::fixtures::VectorGen;

fn similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");
    let mut vectors = VectorGen::new(0x5EED);
    let kernel = Kernel::active();
    for dim in EMBEDDING_DIMS {
        let (a, b) = (vectors.unit(dim), vectors.unit(dim));
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
        let simd_dot = BenchmarkId::new(format!("dot_{kernel}"), dim);
        group.bench_with_input(simd_dot, &dim, |bench, _| {
            bench.iter(|| semantic_cache_simd::dot(black_box(&a.data), black_box(&b.data)));
        });
        let (aa, ab) = (
            AlignedVector::from(a.data.as_slice()),
            AlignedVector::from(b.data.as_slice()),
        );
        let aligned = BenchmarkId::new(format!("dot_{kernel}_aligned"), dim);
        group.bench_with_input(aligned, &dim, |bench, _| {
            bench.iter(|| semantic_cache_simd::dot(black_box(aa.padded()), black_box(ab.padded())));
        });
        // 4 floats na frente de um início alinhado: 16 bytes depois de um
        // múltiplo de 32, metade das cargas em duas linhas de cache.
        let shifted = |v: &[f32]| AlignedVector::from([&[0.0; 4], v].concat().as_slice());
        let (sa, sb) = (shifted(&a.data), shifted(&b.data));
        let misaligned = BenchmarkId::new(format!("dot_{kernel}_misaligned"), dim);
        group.bench_with_input(misaligned, &dim, |bench, _| {
            bench.iter(|| semantic_cache_simd::dot(black_box(&sa[4..]), black_box(&sb[4..])));
        });
        let simd_cosine = BenchmarkId::new(format!("cosine_{kernel}"), dim);
        group.bench_with_input(simd_cosine, &dim, |bench, _| {
            bench.iter(|| semantic_cache_simd::cosine(black_box(&a.data), black_box(&b.data)));
        });
        let quantize = |v: &[f32]| QuantizedVector::quantize(v).expect("valores finitos");
        let (qa, qb) = (quantize(&a.data), quantize(&b.data));
        let i8_dot = BenchmarkId::new(format!("dot_i8_{kernel}"), dim);
        group.bench_with_input(i8_dot, &dim, |bench, _| {
            bench.iter(|| black_box(&qa).dot(black_box(&qb)));
        });
        let i8_cosine = BenchmarkId::new(format!("cosine_i8_{kernel}"), dim);
        group.bench_with_input(i8_cosine, &dim, |bench, _| {
            bench.iter(|| black_box(&qa).cosine(black_box(&qb)));
        });
        group.bench_with_input(BenchmarkId::new("quantize", dim), &dim, |bench, _| {
            bench.iter(|| QuantizedVector::quantize(black_box(&a.data)));
        });
        group.bench_with_input(BenchmarkId::new("dequantize", dim), &dim, |bench, _| {
            bench.iter(|| black_box(&qa).dequantize());
        });
        // O criterion mede na hora da chamada: só esta linha roda ligada.
        profile::set_enabled(true);
        let profiled = BenchmarkId::new("quantize_profiled", dim);
        group.bench_with_input(profiled, &dim, |bench, _| {
            bench.iter(|| QuantizedVector::quantize(black_box(&a.data)));
        });
        profile::set_enabled(false);
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Precision::from_env().criterion();
    targets = similarity
}
criterion_main!(benches);
