//! Kernels NEON (aarch64): 4 `f32` por instrução (`float32x4_t`).
//!
//! NEON é obrigatório no aarch64, mas o despacho passa pelo mesmo
//! `Kernel::supported` dos outros kernels. `vld1q_f32` não exige alinhamento,
//! e a cauda que não completa 4 floats vai pelo escalar. Mesmo desenho do
//! kernel AVX2, com metade da largura.

#![allow(unsafe_code)]

use std::arch::aarch64::{float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

use super::scalar;

/// Floats por registrador.
const LANES: usize = 4;
/// Blocos de 4 por volta, cada um com acumuladores próprios: o FMLA tem
/// latência de ~4 ciclos. No cosseno são 4 × 3 = 12 dos 32 registradores.
const UNROLL: usize = 4;

/// Produto escalar.
///
/// # Safety
///
/// A CPU precisa ter NEON, e `a.len() == b.len()`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [vdupq_n_f32(0.0); UNROLL];
    let mut i = 0;
    // SAFETY (todos os loads): cada leitura cobre `[i, i + LANES)` com
    // `i + LANES <= len`, dentro dos dois slices.
    while i + UNROLL * LANES <= len {
        for (block, sum) in acc.iter_mut().enumerate() {
            let at = i + block * LANES;
            // `vfmaq_f32(acc, x, y)` = `acc + x * y`, com um arredondamento só.
            *sum = vfmaq_f32(*sum, vld1q_f32(pa.add(at)), vld1q_f32(pb.add(at)));
        }
        i += UNROLL * LANES;
    }
    while i + LANES <= len {
        acc[0] = vfmaq_f32(acc[0], vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        i += LANES;
    }
    let mut total = acc[0];
    for sum in &acc[1..] {
        total = vaddq_f32(total, *sum);
    }
    vaddvq_f32(total) + scalar::dot(&a[i..], &b[i..])
}

/// `(a · b, a · a, b · b)` numa passada só: cada par de loads alimenta três
/// FMLAs.
///
/// # Safety
///
/// A CPU precisa ter NEON, e `a.len() == b.len()`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn dot_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // Por volta: `[a·b, a·a, b·b]`.
    let mut acc = [[vdupq_n_f32(0.0); 3]; UNROLL];
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
            *t = vaddq_f32(*t, *s);
        }
    }
    let tail = scalar::dot_norms(&a[i..], &b[i..]);
    (
        vaddvq_f32(total[0]) + tail.0,
        vaddvq_f32(total[1]) + tail.1,
        vaddvq_f32(total[2]) + tail.2,
    )
}

/// Acumula `a·b`, `a·a` e `b·b` de 4 floats lidos de `pa` e `pb`.
///
/// # Safety
///
/// NEON; `pa` e `pb` com 4 floats legíveis.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn step(sums: &mut [float32x4_t; 3], pa: *const f32, pb: *const f32) {
    let va = vld1q_f32(pa);
    let vb = vld1q_f32(pb);
    sums[0] = vfmaq_f32(sums[0], va, vb);
    sums[1] = vfmaq_f32(sums[1], va, va);
    sums[2] = vfmaq_f32(sums[2], vb, vb);
}
