//! Produto escalar e similaridade cosseno de `f32`, despachados para o kernel
//! do processo ([`Kernel::active`]).
//!
//! API segura: comprimentos diferentes viram erro e um kernel que a CPU não
//! executa cai no escalar. A convenção é a da referência do crate de domínio
//! (`semantic_cache_rs::math::{dot_product, normalized_dot_product}`): cosseno
//! com norma zero (inclusive vetor vazio) é erro, sem truncar em `[-1, 1]`. Os
//! kernels SIMD somam em outra ordem e com FMA, então diferem dela nos últimos
//! bits, dentro do limite de erro da soma — os testes conferem esse limite.

use std::fmt;

use crate::cpu::{CpuFeatures, Kernel};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

/// Erro de entrada; espelha os casos de `VectorError` do crate de domínio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityError {
    /// `a` e `b` com comprimentos diferentes.
    DimensionMismatch { expected: usize, actual: usize },
    /// Cosseno com um vetor de norma zero (inclui o vetor vazio).
    ZeroNorm,
}

impl fmt::Display for SimilarityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "dimensao incompativel: esperado {expected}, recebido {actual}"
                )
            }
            Self::ZeroNorm => f.write_str("norma do vetor e zero (cosseno indefinido)"),
        }
    }
}

impl std::error::Error for SimilarityError {}

/// Produto escalar `a · b` com o kernel do processo.
///
/// # Errors
///
/// [`SimilarityError::DimensionMismatch`] se os comprimentos diferem.
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
    Kernel::active().dot(a, b)
}

/// Similaridade cosseno `a · b / (|a| |b|)` com o kernel do processo.
///
/// # Errors
///
/// [`SimilarityError::DimensionMismatch`] se os comprimentos diferem;
/// [`SimilarityError::ZeroNorm`] se `a` ou `b` tem norma zero.
pub fn cosine(a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
    Kernel::active().cosine(a, b)
}

impl Kernel {
    /// Produto escalar com este kernel. Um kernel que a CPU não executa roda
    /// o escalar: executar AVX2 sem AVX2 é instrução ilegal.
    ///
    /// # Errors
    ///
    /// [`SimilarityError::DimensionMismatch`] se os comprimentos diferem.
    #[allow(unsafe_code)]
    pub fn dot(self, a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
        check_dims(a, b)?;
        Ok(match self.runnable() {
            // SAFETY: `runnable` só devolve estes kernels se a CPU tem AVX2 e
            // FMA (`Kernel::supported`); comprimentos iguais conferidos acima.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2Fma | Self::Avx512 => unsafe { avx2::dot(a, b) },
            // SAFETY: `runnable` só devolve `Neon` se a CPU tem NEON.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::dot(a, b) },
            // AVX-512 usa o kernel AVX2 até ter o próprio.
            _ => scalar::dot(a, b),
        })
    }

    /// Similaridade cosseno com este kernel; mesma regra de [`Kernel::dot`]
    /// para kernel sem suporte.
    ///
    /// # Errors
    ///
    /// [`SimilarityError::DimensionMismatch`] se os comprimentos diferem;
    /// [`SimilarityError::ZeroNorm`] se `a` ou `b` tem norma zero.
    #[allow(unsafe_code)]
    pub fn cosine(self, a: &[f32], b: &[f32]) -> Result<f32, SimilarityError> {
        check_dims(a, b)?;
        let (ab, aa, bb) = match self.runnable() {
            // SAFETY: como em `Kernel::dot`.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2Fma | Self::Avx512 => unsafe { avx2::dot_norms(a, b) },
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe { neon::dot_norms(a, b) },
            _ => scalar::dot_norms(a, b),
        };
        // Mesma conta da referência: `dot / (sqrt(|a|²) * sqrt(|b|²))`.
        let (norm_a, norm_b) = (aa.sqrt(), bb.sqrt());
        if norm_a == 0.0 || norm_b == 0.0 {
            return Err(SimilarityError::ZeroNorm);
        }
        Ok(ab / (norm_a * norm_b))
    }

    /// Este kernel, se a CPU o executa; senão o escalar.
    fn runnable(self) -> Self {
        if self.supported(CpuFeatures::get()) {
            self
        } else {
            Self::Scalar
        }
    }
}

fn check_dims(a: &[f32], b: &[f32]) -> Result<(), SimilarityError> {
    if a.len() == b.len() {
        Ok(())
    } else {
        Err(SimilarityError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        })
    }
}

/// Soma sequencial, a mesma de `semantic_cache_rs::math`: resultado idêntico
/// bit a bit à referência. Também fecha a cauda dos kernels SIMD.
mod scalar {
    pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// `(a · b, a · a, b · b)`.
    pub(super) fn dot_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        (dot(a, b), dot(a, a), dot(b, b))
    }
}

#[cfg(test)]
mod tests {
    use semantic_cache_rs::math::{dot_product, normalized_dot_product};
    use semantic_cache_rs::VectorError;

    use super::*;

    /// Kernels que esta CPU executa de verdade (os outros cairiam no escalar).
    fn runnable_kernels() -> Vec<Kernel> {
        Kernel::PREFERENCE
            .into_iter()
            .filter(|k| k.supported(CpuFeatures::get()))
            .collect()
    }

    /// Igualdade de resultado: `NaN` com `NaN` conta, e `0.0 == -0.0`.
    #[allow(clippy::float_cmp)]
    fn same(a: f32, b: f32) -> bool {
        a == b || (a.is_nan() && b.is_nan())
    }

    fn reference_err(e: &VectorError) -> SimilarityError {
        match *e {
            VectorError::DimensionMismatch { expected, actual } => {
                SimilarityError::DimensionMismatch { expected, actual }
            }
            VectorError::ZeroNorm => SimilarityError::ZeroNorm,
            VectorError::EmptyVector => panic!("a referência de slice não devolve EmptyVector"),
        }
    }

    fn assert_same(got: Result<f32, SimilarityError>, want: Result<f32, VectorError>, ctx: &str) {
        match (got, want) {
            (Ok(g), Ok(w)) => assert!(same(g, w), "{ctx}: {g} vs referência {w}"),
            (Err(g), Err(w)) => assert_eq!(g, reference_err(&w), "{ctx}"),
            (g, w) => panic!("{ctx}: {g:?} vs referência {w:?}"),
        }
    }

    /// `dot` e `cosine` de `k` iguais aos da referência (resultado ou erro).
    fn check(k: Kernel, a: &[f32], b: &[f32], ctx: &str) {
        assert_same(k.dot(a, b), dot_product(a, b), &format!("{k} dot, {ctx}"));
        let want = normalized_dot_product(a, b);
        assert_same(k.cosine(a, b), want, &format!("{k} cosine, {ctx}"));
    }

    /// Valores em `[-1, 1)` de um xorshift com semente fixa.
    fn noise(seed: u64, len: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                #[allow(clippy::cast_precision_loss)] // 24 bits cabem exatos no f32
                let unit = (s >> 40) as f32 / (1u64 << 24) as f32;
                unit * 2.0 - 1.0
            })
            .collect()
    }

    /// Comprimentos que passam por todo caminho dos kernels — laço de 32/16
    /// floats, passos de 8/4 e cauda (AVX2/NEON) — e dimensões de embedding.
    const LENS: [usize; 13] = [0, 1, 7, 8, 9, 15, 31, 32, 33, 71, 384, 1536, 3073];

    #[test]
    fn exact_on_integers_for_every_length_and_alignment() {
        // Inteiros pequenos: toda soma é exata em f32 (|soma| < 2^24), então
        // qualquer ordem de soma dá o mesmo valor e um elemento esquecido ou
        // repetido muda o resultado. Deslocar o início tira o alinhamento.
        #[allow(clippy::cast_precision_loss)]
        let ints = |m: usize, len: usize| -> Vec<f32> {
            (0..len).map(|i| (i % m) as f32 - (m / 2) as f32).collect()
        };
        for len in (0..=70).chain([383, 384, 385, 767, 768, 1536, 3072, 3079]) {
            let (a, b) = (ints(7, len + 8), ints(5, len + 8));
            for off in 0..8 {
                let (a, b) = (&a[off..off + len], &b[off..off + len]);
                for k in runnable_kernels() {
                    check(k, a, b, &format!("len {len}, deslocamento {off}"));
                }
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn random_within_rounding_bound() {
        for (seed, len) in LENS.into_iter().enumerate() {
            let scale = [1e-2, 1e-1, 1.0, 1e1, 1e2][seed % 5];
            let a: Vec<f32> = noise(seed as u64 + 1, len)
                .iter()
                .map(|x| x * scale)
                .collect();
            let b = noise(seed as u64 + 101, len);
            let want = dot_product(&a, &b).unwrap();
            // Cada soma erra no máximo ~len·ε/2 · Σ|aᵢbᵢ|; duas somas, o dobro.
            let mag: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y).abs()).sum();
            let bound = len as f64 * f64::from(f32::EPSILON) * mag;
            // No cosseno o erro relativo das normas entra também: ~(2·len + 4)·ε.
            let cos_bound = (2 * len + 4) as f32 * f32::EPSILON;
            for k in runnable_kernels() {
                let got = k.dot(&a, &b).unwrap();
                let msg = format!("{k}, len {len}: {got} vs {want} (limite {bound:e})");
                assert!(f64::from((got - want).abs()) <= bound, "{msg}");
                if len > 0 {
                    let got = k.cosine(&a, &b).unwrap();
                    let want = normalized_dot_product(&a, &b).unwrap();
                    assert!(
                        (got - want).abs() <= cos_bound,
                        "{k}, len {len}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_and_unsupported_kernels_are_the_reference_bit_for_bit() {
        // Kernel sem suporte na CPU cai no escalar, que é a soma da referência.
        let scalar_path: Vec<Kernel> = Kernel::PREFERENCE
            .into_iter()
            .filter(|k| *k == Kernel::Scalar || !k.supported(CpuFeatures::get()))
            .collect();
        for (seed, len) in LENS.into_iter().enumerate() {
            let (a, b) = (noise(seed as u64 + 7, len), noise(seed as u64 + 77, len));
            for &k in &scalar_path {
                check(k, &a, &b, &format!("len {len}"));
            }
        }
    }

    #[test]
    fn mismatch_zero_norm_and_empty_follow_the_reference() {
        let (zero, x, short) = ([0.0f32; 19], noise(3, 19), noise(4, 18));
        let empty: &[f32] = &[];
        for k in Kernel::PREFERENCE {
            check(k, &x, &short, "comprimentos diferentes");
            check(k, &short, &x, "comprimentos diferentes, invertido");
            check(k, &zero, &x, "a zero");
            check(k, &x, &zero, "b zero");
            check(k, &zero, &zero, "ambos zero");
            check(k, empty, empty, "vazios"); // dot 0, cosine ZeroNorm
            assert_eq!(k.cosine(&x, &zero), Err(SimilarityError::ZeroNorm));
        }
        let mismatch = SimilarityError::DimensionMismatch {
            expected: 19,
            actual: 18,
        };
        assert_eq!(dot(&x, &short), Err(mismatch));
        assert_eq!(
            mismatch.to_string(),
            "dimensao incompativel: esperado 19, recebido 18"
        );
    }

    #[test]
    fn nan_and_infinity_propagate() {
        let x = noise(5, 41);
        let ones = vec![1.0f32; 41];
        for pos in [0, 5, 17, 40] {
            let mut a = noise(6, 41);
            a[pos] = f32::NAN;
            for k in runnable_kernels() {
                assert!(k.dot(&a, &x).unwrap().is_nan(), "{k}, NaN em {pos}");
                assert!(k.cosine(&a, &x).unwrap().is_nan(), "{k}, NaN em {pos}");
            }
            a[pos] = f32::INFINITY;
            let mut zero_at = ones.clone();
            zero_at[pos] = 0.0; // ∞ · 0 = NaN
            for k in runnable_kernels() {
                assert!(k.dot(&a, &zero_at).unwrap().is_nan(), "{k}, ∞·0 em {pos}");
                check(k, &a, &x, &format!("∞ em {pos}"));
            }
        }
    }

    #[test]
    fn free_functions_use_the_active_kernel() {
        let (a, b) = (noise(8, 300), noise(9, 300));
        let active = Kernel::active();
        assert!(same(dot(&a, &b).unwrap(), active.dot(&a, &b).unwrap()));
        assert!(same(
            cosine(&a, &b).unwrap(),
            active.cosine(&a, &b).unwrap()
        ));
    }
}
