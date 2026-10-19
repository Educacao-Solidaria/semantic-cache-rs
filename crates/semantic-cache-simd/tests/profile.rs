//! O profiler global, num binário de teste próprio: os testes do crate rodam
//! em paralelo e também quantizam, o que sujaria as contagens.

use std::time::Duration;

use semantic_cache_simd::profile::{self, Op};
use semantic_cache_simd::QuantizedVector;

#[test]
fn quantize_and_dequantize_are_timed_only_while_enabled() {
    let values: Vec<f32> = (0..1536u16).map(|i| f32::from(i).sin()).collect();
    let quantized = QuantizedVector::quantize(&values).unwrap();
    let count = |op| profile::snapshot(op).map_or(0, |s| s.count);
    assert!(!profile::enabled(), "desligado por padrão");
    assert_eq!(count(Op::Quantize), 0, "desligado não registra");

    profile::set_enabled(true);
    for _ in 0..100 {
        QuantizedVector::quantize(&values).unwrap();
    }
    assert!(
        QuantizedVector::quantize(&[f32::NAN]).is_err(),
        "o erro também conta"
    );
    for _ in 0..30 {
        let _ = quantized.dequantize();
    }
    assert_eq!(profile::time(Op::Search, || 7), 7);
    let counts = (
        count(Op::Quantize),
        count(Op::Dequantize),
        count(Op::Search),
    );
    assert_eq!(counts, (101, 30, 1));
    let quantize = profile::snapshot(Op::Quantize).unwrap();
    assert!(Duration::ZERO < quantize.p50 && quantize.p50 <= quantize.p99);
    assert!(quantize.p99 <= quantize.max, "{quantize:?}");

    profile::set_enabled(false);
    QuantizedVector::quantize(&values).unwrap();
    assert_eq!(count(Op::Quantize), 101, "desligado de novo: nada novo");
    profile::reset();
    assert_eq!(profile::snapshot(Op::Quantize), None);
}
