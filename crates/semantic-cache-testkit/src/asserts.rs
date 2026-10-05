//! Asserções de ponto flutuante com tolerância.
//!
//! Três modos, escolhidos pelo nome no macro:
//!
//! - `abs = d`: `|a - b| <= d`. Para valores perto de zero (cosseno de
//!   vetores ortogonais, distâncias pequenas).
//! - `rel = r`: `|a - b| <= r * max(|a|, |b|)`. Para valores de magnitude
//!   variável (normas, produtos escalares de vetores não normalizados).
//! - `ulps = n`: no máximo `n` floats representáveis entre `a` e `b`. Para
//!   comparar um kernel SIMD com a referência escalar: a ordem de soma muda o
//!   arredondamento em poucos ULPs, independentemente da magnitude.
//!
//! `NaN` nunca é próximo de nada, nem de outro `NaN`; infinitos só de si mesmos.
//!
//! ```
//! use semantic_cache_testkit::{assert_f32_near, assert_vec_near};
//!
//! assert_f32_near!(0.1 + 0.2, 0.3, abs = 1e-6);
//! assert_vec_near!([1.0, 2.0], vec![1.0, 2.000_000_2], ulps = 1);
//! ```

use std::fmt;

use semantic_cache_rs::Vector;

/// Margem de comparação; construída pelos macros a partir de `abs`, `rel`
/// ou `ulps`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance {
    Abs(f32),
    Rel(f32),
    Ulps(u32),
}

impl Tolerance {
    #[must_use]
    pub fn abs(delta: f32) -> Self {
        Self::Abs(delta)
    }

    #[must_use]
    pub fn rel(ratio: f32) -> Self {
        Self::Rel(ratio)
    }

    #[must_use]
    pub fn ulps(n: u32) -> Self {
        Self::Ulps(n)
    }

    /// `true` se `a` e `b` estão dentro da margem.
    #[must_use]
    #[allow(clippy::float_cmp)] // a igualdade exata é o atalho intencional
    pub fn accepts(self, a: f32, b: f32) -> bool {
        if a == b {
            return true; // inclui infinitos iguais e `0.0 == -0.0`
        }
        if !a.is_finite() || !b.is_finite() {
            return false;
        }
        match self {
            Self::Abs(d) => (a - b).abs() <= d,
            Self::Rel(r) => (a - b).abs() <= r * a.abs().max(b.abs()),
            Self::Ulps(n) => ulps_between(a, b) <= u64::from(n),
        }
    }
}

impl fmt::Display for Tolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Abs(d) => write!(f, "abs = {d:e}"),
            Self::Rel(r) => write!(f, "rel = {r:e}"),
            Self::Ulps(n) => write!(f, "ulps = {n}"),
        }
    }
}

/// Quantos floats representáveis separam `a` de `b` (finitos). Atravessar o
/// zero conta os subnormais dos dois lados; `0.0` e `-0.0` distam 0.
#[must_use]
pub fn ulps_between(a: f32, b: f32) -> u64 {
    // Bits reinterpretados numa reta ordenada: negativos espelhados abaixo de 0.
    let ordered = |x: f32| {
        let bits = i64::from(x.to_bits());
        if bits & 0x8000_0000 == 0 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

/// Tipos que os macros de vetor aceitam como `&[f32]`.
pub trait F32Slice {
    fn f32_slice(&self) -> &[f32];
}

impl F32Slice for [f32] {
    fn f32_slice(&self) -> &[f32] {
        self
    }
}

impl<const N: usize> F32Slice for [f32; N] {
    fn f32_slice(&self) -> &[f32] {
        self
    }
}

impl F32Slice for Vec<f32> {
    fn f32_slice(&self) -> &[f32] {
        self
    }
}

impl F32Slice for Vector {
    fn f32_slice(&self) -> &[f32] {
        &self.data
    }
}

impl<T: F32Slice + ?Sized> F32Slice for &T {
    fn f32_slice(&self) -> &[f32] {
        (**self).f32_slice()
    }
}

/// Compara elemento a elemento; `Err` descreve a primeira divergência.
///
/// # Errors
///
/// Comprimentos diferentes ou um par fora da margem.
pub fn check_slices(a: &[f32], b: &[f32], tol: Tolerance) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!(
            "comprimentos diferentes: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    match a.iter().zip(b).position(|(&x, &y)| !tol.accepts(x, y)) {
        None => Ok(()),
        Some(i) => Err(format!(
            "índice {i}: {} vs {} (|Δ| = {:e}, {tol})",
            a[i],
            b[i],
            (a[i] - b[i]).abs()
        )),
    }
}

/// `assert_f32_near!(a, b, abs = 1e-6)`; também `rel = ...` e `ulps = ...`.
#[macro_export]
macro_rules! assert_f32_near {
    ($a:expr, $b:expr, $kind:ident = $tol:expr $(,)?) => {{
        let (a, b): (f32, f32) = ($a, $b);
        let tol = $crate::asserts::Tolerance::$kind($tol);
        assert!(
            tol.accepts(a, b),
            "assert_f32_near: {a} vs {b} (|Δ| = {:e}, {tol})",
            (a - b).abs()
        );
    }};
}

/// `assert_vec_near!(a, b, ulps = 4)` para `[f32]`, `[f32; N]`, `Vec<f32>` e
/// `Vector`; também `abs = ...` e `rel = ...`.
#[macro_export]
macro_rules! assert_vec_near {
    ($a:expr, $b:expr, $kind:ident = $tol:expr $(,)?) => {{
        use $crate::asserts::F32Slice as _;
        let tol = $crate::asserts::Tolerance::$kind($tol);
        if let Err(e) = $crate::asserts::check_slices(($a).f32_slice(), ($b).f32_slice(), tol) {
            panic!("assert_vec_near: {e}");
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_and_relative() {
        assert!(Tolerance::abs(1e-3).accepts(1.0, 1.000_9));
        assert!(!Tolerance::abs(1e-3).accepts(1.0, 1.002));
        // Relativa escala com a magnitude: o mesmo Δ passa em 1e6 e falha em 1.
        assert!(Tolerance::rel(1e-6).accepts(1e6, 1e6 + 0.5));
        assert!(!Tolerance::rel(1e-6).accepts(1.0, 1.5));
    }

    #[test]
    fn ulps_count_representable_steps() {
        assert_eq!(ulps_between(1.0, 1.0_f32.next_up()), 1);
        assert_eq!(ulps_between(0.0, -0.0), 0);
        assert_eq!(ulps_between(f32::from_bits(1), -f32::from_bits(1)), 2);
        assert_eq!(ulps_between(-1.0, 1.0), 2 * u64::from(1.0_f32.to_bits()));
        assert!(Tolerance::ulps(2).accepts(1.0, 1.0_f32.next_up().next_up()));
        assert!(!Tolerance::ulps(1).accepts(1.0, 1.0_f32.next_up().next_up()));
    }

    #[test]
    fn nan_and_infinity() {
        for tol in [
            Tolerance::abs(f32::MAX),
            Tolerance::rel(1.0),
            Tolerance::ulps(u32::MAX),
        ] {
            assert!(!tol.accepts(f32::NAN, f32::NAN), "{tol}");
            assert!(!tol.accepts(f32::INFINITY, f32::MAX), "{tol}");
            assert!(tol.accepts(f32::INFINITY, f32::INFINITY), "{tol}");
        }
    }

    #[test]
    fn macros_accept_every_slice_kind() {
        let v = Vector::new(vec![0.6, 0.8]).unwrap();
        assert_f32_near!(v.norm(), 1.0, abs = 1e-6);
        assert_vec_near!(v, [0.6, 0.8], ulps = 0);
        assert_vec_near!(&v, vec![0.6, 0.800_000_1], rel = 1e-6);
        assert_vec_near!(v.data[..1], [0.600_001], abs = 1e-5,);
    }

    #[test]
    #[should_panic(expected = "assert_vec_near: índice 1: 2 vs 2.1")]
    fn vec_reports_first_mismatch() {
        assert_vec_near!([1.0, 2.0, 3.0], [1.0, 2.1, 9.0], abs = 1e-3);
    }

    #[test]
    #[should_panic(expected = "comprimentos diferentes: 2 vs 3")]
    fn vec_rejects_length_mismatch() {
        assert_vec_near!([1.0, 2.0], [1.0, 2.0, 3.0], abs = 1.0);
    }

    #[test]
    #[should_panic(expected = "assert_f32_near: 1 vs 1.1 (|Δ| = 1.00000024e-1, rel = 1e-2)")]
    fn scalar_reports_tolerance() {
        assert_f32_near!(1.0, 1.1, rel = 0.01);
    }
}
