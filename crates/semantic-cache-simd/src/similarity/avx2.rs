//! Kernels AVX2 + FMA: 8 `f32` por instrução (`__m256`).
//!
//! Só [`super`] chama estas funções, e só depois de `Kernel::supported`
//! confirmar AVX2 e FMA na CPU em execução. Os loads são `loadu` (aceitam
//! qualquer alinhamento: um subslice de `Vec<f32>` quase nunca está alinhado em
//! 32 bytes), e a cauda que não completa 8 floats vai pelo escalar.

#![allow(unsafe_code)]

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps,
    _mm_movehl_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
    _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps,
    _mm_movehl_ps,
};

use super::scalar;

/// Floats por registrador.
const LANES: usize = 8;
/// Blocos de 8 por volta do laço principal, cada um com acumuladores
/// próprios: o FMA tem latência de ~4 ciclos, e uma cadeia só deixaria a
/// unidade esperando o resultado anterior. Medido no cosseno: 4 voltas
/// (12 acumuladores, de 16 registradores YMM) ganham 13–20% sobre 2.
const UNROLL: usize = 4;

/// Produto escalar.
///
/// # Safety
///
/// A CPU precisa ter AVX2 e FMA, e `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); UNROLL];
    let mut i = 0;
    // SAFETY (todos os loads): cada leitura cobre `[i, i + LANES)` com
    // `i + LANES <= len`, dentro dos dois slices.
    while i + UNROLL * LANES <= len {
        for (block, sum) in acc.iter_mut().enumerate() {
            let at = i + block * LANES;
            let (va, vb) = (_mm256_loadu_ps(pa.add(at)), _mm256_loadu_ps(pb.add(at)));
            *sum = _mm256_fmadd_ps(va, vb, *sum);
        }
        i += UNROLL * LANES;
    }
    while i + LANES <= len {
        let (va, vb) = (_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
        acc[0] = _mm256_fmadd_ps(va, vb, acc[0]);
        i += LANES;
    }
    let mut total = acc[0];
    for sum in &acc[1..] {
        total = _mm256_add_ps(total, *sum);
    }
    hsum(total) + scalar::dot(&a[i..], &b[i..])
}

/// `(a · b, a · a, b · b)` numa passada só: cada par de loads alimenta três
/// FMAs.
///
/// # Safety
///
/// A CPU precisa ter AVX2 e FMA, e `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // Por volta: `[a·b, a·a, b·b]`.
    let mut acc = [[_mm256_setzero_ps(); 3]; UNROLL];
    let mut i = 0;
    // SAFETY (todos os loads): como em `dot`, `i + LANES <= len` em cada leitura.
    while i + UNROLL * LANES <= len {
        for (block, sums) in acc.iter_mut().enumerate() {
            let at = i + block * LANES;
            step(sums, pa.add(at), pb.add(at));
        }
        i += UNROLL * LANES;
    }
    while i + LANES <= len {
        step(&mut acc[0], pa.add(i), pb.add(i));
        i += LANES;
    }
    let mut total = acc[0];
    for sums in &acc[1..] {
        for (t, s) in total.iter_mut().zip(sums) {
            *t = _mm256_add_ps(*t, *s);
        }
    }
    let tail = scalar::dot_norms(&a[i..], &b[i..]);
    (
        hsum(total[0]) + tail.0,
        hsum(total[1]) + tail.1,
        hsum(total[2]) + tail.2,
    )
}

/// Acumula `a·b`, `a·a` e `b·b` de 8 floats lidos de `pa` e `pb`.
///
/// # Safety
///
/// AVX2 e FMA; `pa` e `pb` com 8 floats legíveis.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn step(sums: &mut [__m256; 3], pa: *const f32, pb: *const f32) {
    let va = _mm256_loadu_ps(pa);
    let vb = _mm256_loadu_ps(pb);
    sums[0] = _mm256_fmadd_ps(va, vb, sums[0]);
    sums[1] = _mm256_fmadd_ps(va, va, sums[1]);
    sums[2] = _mm256_fmadd_ps(vb, vb, sums[2]);
}

/// Soma das 8 faixas: 256 → 128 bits, depois 4 → 2 → 1.
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: __m256) -> f32 {
    let quad = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    _mm_cvtss_f32(_mm_add_ss(pair, _mm_movehdup_ps(pair)))
}
