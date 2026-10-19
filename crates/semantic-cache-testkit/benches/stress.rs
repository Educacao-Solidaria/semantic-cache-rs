//! Estresse com 100 000 vetores de 1536 dimensões (`text-embedding-3-small`):
//! latência de inserção e de consulta top-10 de cada índice, recall contra a
//! busca exata e a meta do projeto (p99 da consulta abaixo de 2 ms).
//!
//! - `cargo bench --bench stress`: carga cheia (~620 MB de pico).
//! - `SEMCACHE_BENCH_QUICK=1 cargo bench --bench stress`: 10 000 vetores.
//! - Sob `cargo test` (sem `--bench`): carga mínima, para o harness não
//!   apodrecer; o CI compila e roda só isso.
//!
//! A coluna "profiler" é o p50 / p99 da consulta segundo o profiler embutido
//! (`semantic_cache_simd::profile`), ao lado do exato: confere o histograma.
//! Índice que não mede a busca com `profile::time(Op::Search, ..)` sai com "—".

use std::time::Duration;

use semantic_cache_simd::profile::{self, Op};
use semantic_cache_simd::Kernel;
use semantic_cache_testkit::bench::Precision;
use semantic_cache_testkit::stress::{FlatF32, FlatI8, Load, Report, P99_TARGET};

fn main() {
    let measuring = std::env::args().any(|arg| arg == "--bench");
    let quick = Precision::from_env() == Precision::QUICK;
    let (vectors, dim, queries) = match (measuring, quick) {
        (false, _) => (500, 64, 20),
        (true, true) => (10_000, 1536, 200),
        (true, false) => (100_000, 1536, 1_000),
    };
    let load = Load {
        seed: 0x5EED,
        vectors,
        dim,
        queries,
        k: 10,
    };
    println!(
        "{vectors} vetores × {dim} dimensões, {queries} consultas top-{}, kernel {}\n",
        load.k,
        Kernel::active()
    );
    println!(
        "| índice | inserção p50 / p99 | consulta p50 / p95 / p99 / máx | profiler p50 / p99 | recall@{} | p99 < {} |",
        load.k,
        ms(P99_TARGET)
    );
    println!("|---|---|---|---|---|---|");
    profile::set_enabled(true);
    // A busca exata vai primeiro: os resultados dela são a referência do
    // recall dos outros. Cada índice é descartado antes do próximo (pico de
    // memória de um só).
    let mut exact = FlatF32::with_capacity(vectors, dim);
    let reference = load.run(&mut exact, None);
    drop(exact);
    row("flat_f32 (exata)", &reference);
    row(
        "flat_i8",
        &load.run(&mut FlatI8::default(), Some(&reference.results)),
    );
    // O índice aproximado (HNSW, PR 32/100) entra aqui do mesmo jeito:
    // `row("hnsw", &load.run(&mut hnsw, Some(&reference.results)))`.
}

fn row(name: &str, report: &Report) {
    let (insert, search) = (report.insert, report.search);
    let recall = report
        .recall
        .map_or_else(|| "referência".to_owned(), |r| format!("{r:.3}"));
    let verdict = if search.p99 < P99_TARGET {
        "sim"
    } else {
        "**não**"
    };
    let profiled = profile::snapshot(Op::Search).map_or_else(
        || "—".to_owned(),
        |p| format!("{} / {}", ms(p.p50), ms(p.p99)),
    );
    // A linha seguinte conta só a carga do próximo índice.
    profile::reset();
    println!(
        "| {name} | {} / {} | {} / {} / {} / {} | {profiled} | {recall} | {verdict} |",
        ms(insert.p50),
        ms(insert.p99),
        ms(search.p50),
        ms(search.p95),
        ms(search.p99),
        ms(search.max),
    );
}

fn ms(d: Duration) -> String {
    format!("{:.3} ms", d.as_secs_f64() * 1e3)
}
