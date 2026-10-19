//! Quantização escalar de vetores `f32` em `i8`: `x ≈ escala · q`, com `q` em
//! `[-127, 127]` e uma escala `f32` por vetor (`max|x| / 127`).
//!
//! Um byte por componente em vez de quatro: um embedding de 1536 dimensões
//! cai de 6144 para 1548 bytes (−74,8%: códigos, escala e o inverso da norma).
//! O erro de cada componente é no máximo metade da escala
//! ([`QuantizedVector::max_error`]).
//!
//! Simétrica, e não afim (com ponto zero): componentes de embedding se
//! distribuem em torno de zero, então `[-max, max]` já é a faixa útil, o zero
//! é exato e o produto escalar quantizado é `escalaₐ · escala_b · Σ qₐ q_b`,
//! sem termos de correção — o formato que um kernel int8 consome direto.
//! ponytail: vetor muito assimétrico (todo positivo, por exemplo) desperdiça
//! metade da faixa, até 1 bit de precisão; o caminho é a forma afim, com um
//! ponto zero por vetor.
//!
//! Produto escalar e cosseno entre vetores quantizados
//! ([`QuantizedVector::dot`], [`QuantizedVector::cosine`]) somam os códigos em
//! inteiro, exato, com o kernel do processo: no AVX2, 32 códigos por
//! `_mm256_maddubs_epi16`.

use std::fmt;

use crate::cpu::Kernel;
use crate::similarity::{check_dims, SimilarityError};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;

/// Maior código; `-128` fica de fora para a faixa ser simétrica.
const MAX_CODE: f32 = 127.0;

/// Vetor quantizado: escala e um código `i8` por componente.
///
/// Invariante: códigos em `[-127, 127]`. O kernel AVX2 depende dela para ser
/// exato; hoje o único construtor é [`QuantizedVector::quantize`], e um
/// construtor a partir de códigos prontos (persistência) precisa recusar `-128`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVector {
    scale: f32,
    codes: Vec<i8>,
    /// `1 / sqrt(Σ q²)`, ou 0 com os códigos todos zero: o cosseno custa um
    /// produto escalar só.
    inv_norm: f64,
}

/// `NaN` ou infinito na entrada, na posição `index`: quantizar daria códigos
/// sem sentido (o cast de `NaN` para inteiro vira 0 em silêncio).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFinite {
    pub index: usize,
}

impl fmt::Display for NonFinite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "valor nao finito (NaN ou infinito) na posicao {}",
            self.index
        )
    }
}

impl std::error::Error for NonFinite {}

impl QuantizedVector {
    /// Quantiza `values`. Vetor vazio ou todo zero dá escala 0 e códigos 0
    /// (sem divisão por zero); o mesmo vale para um vetor cujo máximo é tão
    /// pequeno (< ~9e-44) que `max / 127` arredonda para zero em `f32`.
    ///
    /// # Errors
    ///
    /// [`NonFinite`] com a posição do primeiro `NaN` ou infinito.
    pub fn quantize(values: &[f32]) -> Result<Self, NonFinite> {
        if let Some(index) = values.iter().position(|x| !x.is_finite()) {
            return Err(NonFinite { index });
        }
        let max = values.iter().fold(0.0f32, |max, x| max.max(x.abs()));
        let scale = max / MAX_CODE;
        if scale == 0.0 {
            return Ok(Self::from_codes(0.0, vec![0; values.len()]));
        }
        // Em f64: com escala subnormal, `x / escala` em f32 perderia bits.
        #[allow(clippy::cast_possible_truncation)] // `q` já está em [-127, 127]
        let code = |x: f32| {
            (f64::from(x) / f64::from(scale))
                .round()
                .clamp(-127.0, 127.0) as i8
        };
        Ok(Self::from_codes(
            scale,
            values.iter().map(|&x| code(x)).collect(),
        ))
    }

    /// Junta escala e códigos (já em `[-127, 127]`) e calcula a norma.
    fn from_codes(scale: f32, codes: Vec<i8>) -> Self {
        let norm_sq = dot_codes(Kernel::active(), &codes, &codes);
        let inv_norm = if norm_sq == 0 {
            0.0
        } else {
            1.0 / exact_f64(norm_sq).sqrt()
        };
        Self {
            scale,
            codes,
            inv_norm,
        }
    }

    /// Produto escalar aproximado dos vetores originais,
    /// `escalaₐ · escala_b · Σ qₐ q_b`, com o kernel do processo.
    ///
    /// # Errors
    ///
    /// [`SimilarityError::DimensionMismatch`] se os comprimentos diferem.
    pub fn dot(&self, other: &Self) -> Result<f32, SimilarityError> {
        Kernel::active().dot_i8(self, other)
    }

    /// Cosseno aproximado dos vetores originais, com o kernel do processo.
    ///
    /// # Errors
    ///
    /// [`SimilarityError::DimensionMismatch`] se os comprimentos diferem;
    /// [`SimilarityError::ZeroNorm`] se um dos vetores é todo zero (ou vazio).
    pub fn cosine(&self, other: &Self) -> Result<f32, SimilarityError> {
        Kernel::active().cosine_i8(self, other)
    }

    /// Valores reconstruídos (`escala · q`). Satura em `±f32::MAX`: com
    /// `max|x|` perto do limite do `f32`, `127 · escala` arredonda para cima
    /// e estouraria para infinito.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let value = |&q: &i8| (self.scale * f32::from(q)).clamp(-f32::MAX, f32::MAX);
        self.codes.iter().map(value).collect()
    }

    /// Erro máximo de cada componente reconstruído: metade da escala (mais o
    /// arredondamento de `f32` do produto `escala · q`).
    #[must_use]
    pub fn max_error(&self) -> f32 {
        self.scale / 2.0
    }

    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    #[must_use]
    pub fn codes(&self) -> &[i8] {
        &self.codes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

impl Kernel {
    /// [`QuantizedVector::dot`] com este kernel: a soma `Σ qₐ q_b` é exata e
    /// só o produto pelas escalas (em `f64`) arredonda. Kernel que a CPU não
    /// executa roda o escalar.
    ///
    /// # Errors
    ///
    /// [`SimilarityError::DimensionMismatch`] se os comprimentos diferem.
    pub fn dot_i8(self, a: &QuantizedVector, b: &QuantizedVector) -> Result<f32, SimilarityError> {
        check_dims(&a.codes, &b.codes)?;
        let sum = exact_f64(dot_codes(self, &a.codes, &b.codes));
        #[allow(clippy::cast_possible_truncation)] // estoura para ±∞, como em `f32`
        let dot = (sum * f64::from(a.scale) * f64::from(b.scale)) as f32;
        Ok(dot)
    }

    /// [`QuantizedVector::cosine`] com este kernel: `Σ qₐ q_b / (|qₐ| |q_b|)`
    /// — as escalas se cancelam e as normas vêm da quantização.
    ///
    /// # Errors
    ///
    /// [`SimilarityError::DimensionMismatch`] se os comprimentos diferem;
    /// [`SimilarityError::ZeroNorm`] se um dos vetores é todo zero (ou vazio).
    pub fn cosine_i8(
        self,
        a: &QuantizedVector,
        b: &QuantizedVector,
    ) -> Result<f32, SimilarityError> {
        check_dims(&a.codes, &b.codes)?;
        if a.inv_norm == 0.0 || b.inv_norm == 0.0 {
            return Err(SimilarityError::ZeroNorm);
        }
        let sum = exact_f64(dot_codes(self, &a.codes, &b.codes));
        #[allow(clippy::cast_possible_truncation)] // |cosseno| ≤ 1
        let cosine = (sum * a.inv_norm * b.inv_norm) as f32;
        Ok(cosine)
    }
}

/// Componentes por pedaço da soma inteira: a de um pedaço cabe em `i32`
/// (`2¹⁷ · 127² < 2³¹`) e os pedaços somam em `i64`. Um embedding (milhares
/// de dimensões) é um pedaço só.
const CHUNK: usize = 1 << 17;

/// `Σ aᵢ bᵢ` exato com `kernel` (o escalar, se a CPU não o executa).
#[allow(unsafe_code)]
fn dot_codes(kernel: Kernel, a: &[i8], b: &[i8]) -> i64 {
    assert_eq!(a.len(), b.len(), "comprimentos conferidos por quem chama");
    let kernel = kernel.runnable();
    let chunk = |(x, y): (&[i8], &[i8])| match kernel {
        // SAFETY: `runnable` só devolve estes kernels se a CPU tem AVX2; os
        // pedaços têm o mesmo comprimento porque `a` e `b` têm.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Kernel::Avx2Fma | Kernel::Avx512 => unsafe { avx2::dot(x, y) },
        _ => dot_scalar(x, y),
    };
    a.chunks(CHUNK)
        .zip(b.chunks(CHUNK))
        .map(|pair| i64::from(chunk(pair)))
        .sum()
}

/// Referência escalar, que também fecha a cauda do AVX2; exata em `i32` num
/// pedaço de até [`CHUNK`] componentes.
fn dot_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// Conversão exata: somas de códigos ficam abaixo de `2⁵³` em qualquer vetor
/// que caiba na memória (`len · 127²`, com `len < 5 · 10¹¹`).
#[allow(clippy::cast_precision_loss)]
fn exact_f64(x: i64) -> f64 {
    x as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine;

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

    #[test]
    fn error_is_at_most_half_the_scale() {
        for (seed, scale) in [1e-30, 1e-3, 1.0, 1e3, 1e30].into_iter().enumerate() {
            for len in [1, 7, 384, 1536] {
                let values: Vec<f32> = noise(seed as u64 + 1, len)
                    .iter()
                    .map(|x| x * scale)
                    .collect();
                let quantized = QuantizedVector::quantize(&values).unwrap();
                let max = values.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                assert_eq!(quantized.len(), len);
                assert!(
                    quantized.codes().iter().all(|&q| q != i8::MIN),
                    "faixa simétrica"
                );
                assert!(
                    quantized.codes().iter().any(|q| q.unsigned_abs() == 127),
                    "o máximo vira ±127"
                );
                assert!(
                    (quantized.scale() - max / 127.0).abs() <= f32::EPSILON * quantized.scale()
                );
                for (x, y) in values.iter().zip(quantized.dequantize()) {
                    let bound = quantized.max_error() + f32::EPSILON * x.abs();
                    assert!((x - y).abs() <= bound, "{x} vs {y} (limite {bound:e})");
                }
            }
        }
    }

    #[test]
    fn preserves_cosine_and_cuts_memory_to_a_quarter() {
        // 1536 dimensões, como `text-embedding-3-small`: o cosseno mexe pouco
        // perto do limiar de hit do cache (0,92), medido ~5e-4 no pior par.
        for seed in 0..50 {
            let (a, b) = (noise(seed + 1, 1536), noise(seed + 1001, 1536));
            let near: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + 0.2 * y).collect();
            let qa = QuantizedVector::quantize(&a).unwrap();
            for other in [&b, &near] {
                let qo = QuantizedVector::quantize(other).unwrap();
                let exact = cosine(&a, other).unwrap();
                let delta = cosine(&qa.dequantize(), &qo.dequantize()).unwrap() - exact;
                assert!(delta.abs() < 2e-3, "seed {seed}: Δcos = {delta:e}");
                let delta = qa.cosine(&qo).unwrap() - exact;
                assert!(delta.abs() < 2e-3, "seed {seed}: Δcos inteiro = {delta:e}");
            }
        }
        let quantized = QuantizedVector::quantize(&noise(9, 1536)).unwrap();
        let bytes = std::mem::size_of_val(quantized.codes()) + 4 + 8; // escala, 1/norma
        assert_eq!((bytes, 1536 * 4), (1548, 6144));
    }

    #[test]
    fn zero_and_empty_vectors_have_no_division_by_zero() {
        for values in [&[0.0f32; 16][..], &[-0.0, 0.0], &[]] {
            let quantized = QuantizedVector::quantize(values).unwrap();
            assert_eq!(quantized.scale(), 0.0);
            assert_eq!(quantized.max_error(), 0.0);
            assert!(quantized.codes().iter().all(|&q| q == 0));
            assert_eq!(quantized.dequantize(), vec![0.0; values.len()]);
            assert_eq!(quantized.is_empty(), values.is_empty());
        }
    }

    #[test]
    fn rejects_nan_and_infinity_with_their_position() {
        let mut values = noise(3, 1000);
        for (index, bad) in [(0, f32::NAN), (5, f32::INFINITY), (999, f32::NEG_INFINITY)] {
            let saved = std::mem::replace(&mut values[index], bad);
            assert_eq!(QuantizedVector::quantize(&values), Err(NonFinite { index }));
            values[index] = saved;
        }
        assert_eq!(
            NonFinite { index: 5 }.to_string(),
            "valor nao finito (NaN ou infinito) na posicao 5"
        );
    }

    #[test]
    fn extreme_magnitudes_stay_finite() {
        // Perto do limite: `127 · (MAX / 127)` estoura sem a saturação.
        let huge = QuantizedVector::quantize(&[f32::MAX, -f32::MAX, 1.0]).unwrap();
        assert_eq!(huge.dequantize(), [f32::MAX, -f32::MAX, 0.0]);
        // Subnormal: a escala ainda cabe e a reconstrução segue o limite.
        let tiny = QuantizedVector::quantize(&[1e-40, -5e-41]).unwrap();
        assert!(tiny.scale() > 0.0);
        let restored = tiny.dequantize();
        assert!((restored[0] - 1e-40).abs() <= tiny.max_error() + f32::from_bits(1));
        // Máximo abaixo de ~9e-44: `max / 127` arredonda para zero e o vetor também.
        let below = QuantizedVector::quantize(&[f32::from_bits(1), 1e-44]).unwrap();
        let zeros = (0.0, &[0i8, 0][..], vec![0.0, 0.0]);
        assert_eq!((below.scale(), below.codes(), below.dequantize()), zeros);
    }

    /// Códigos em `[-127, 127]` de um xorshift.
    #[allow(clippy::cast_possible_truncation)]
    fn codes(seed: u64, len: usize) -> Vec<i8> {
        let round = |x: &f32| (x * 127.0).round() as i8;
        noise(seed, len).iter().map(round).collect()
    }

    /// `Σ aᵢ bᵢ` em `i64`, sem kernel nem pedaços.
    fn reference(a: &[i8], b: &[i8]) -> i64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| i64::from(x) * i64::from(y))
            .sum()
    }

    #[test]
    fn integer_sum_is_exact_for_every_length_alignment_and_extreme() {
        // ±127 em todas as posições leva os pares do `maddubs` a até ±32 258,
        // 509 abaixo da saturação do `i16`; o sorteio cobre o resto da faixa.
        let extreme = |seed, len| -> Vec<i8> {
            let sign = |q: &i8| if *q < 0 { -127 } else { 127 };
            codes(seed, len).iter().map(sign).collect()
        };
        for len in (0..=100).chain([255, 256, 257, 1536, 3072, 4099]) {
            let padded = len + 32;
            let seed = len as u64;
            let (x, y, ex, ey) = (
                codes(seed, padded),
                codes(seed + 1, padded),
                extreme(seed, padded),
                extreme(seed + 2, padded),
            );
            let (max, min) = (vec![127i8; padded], vec![-127i8; padded]);
            let pairs = [
                (&x, &y),
                (&ex, &ey),
                (&ex, &x),
                (&max, &max),
                (&max, &min),
                (&min, &min),
            ];
            for off in [0, 1, 7, 31] {
                for (a, b) in pairs {
                    let (a, b) = (&a[off..off + len], &b[off..off + len]);
                    for k in Kernel::PREFERENCE {
                        let ctx = format!("{k}, len {len}, deslocamento {off}");
                        assert_eq!(dot_codes(k, a, b), reference(a, b), "{ctx}");
                    }
                }
            }
        }
    }

    #[test]
    fn sum_beyond_i32_is_split_into_exact_chunks() {
        // 200 000 · 127² = 3 225 800 000 > i32::MAX: sem os pedaços, estouraria.
        let (max, min) = (vec![127i8; 200_000], vec![-127i8; 200_000]);
        let (x, y) = (codes(11, 200_000), codes(12, 200_000));
        for k in Kernel::PREFERENCE {
            assert_eq!(dot_codes(k, &max, &max), 3_225_800_000, "{k}");
            assert_eq!(dot_codes(k, &max, &min), -3_225_800_000, "{k}");
            assert_eq!(dot_codes(k, &x, &y), reference(&x, &y), "{k}");
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn dot_and_cosine_match_the_dequantized_vectors() {
        for (seed, len) in [1usize, 7, 33, 384, 1536, 3072].into_iter().enumerate() {
            let scale = [1e-3, 1.0, 1e3][seed % 3];
            let a: Vec<f32> = noise(seed as u64 + 1, len)
                .iter()
                .map(|x| x * scale)
                .collect();
            let qa = QuantizedVector::quantize(&a).unwrap();
            let qb = QuantizedVector::quantize(&noise(seed as u64 + 50, len)).unwrap();
            let (da, db) = (qa.dequantize(), qb.dequantize());
            // Mesmos limites do `f32` (a conta inteira só arredonda no fim).
            let mag: f64 = da
                .iter()
                .zip(&db)
                .map(|(x, y)| f64::from(x * y).abs())
                .sum();
            let bound = (len + 2) as f64 * f64::from(f32::EPSILON) * mag;
            let cos_bound = (2 * len + 4) as f32 * f32::EPSILON;
            let (dot, cos) = (
                Kernel::Scalar.dot(&da, &db),
                Kernel::Scalar.cosine(&da, &db),
            );
            for k in Kernel::PREFERENCE {
                let got = k.dot_i8(&qa, &qb).unwrap();
                let msg = format!("{k}, len {len}: {got} vs {dot:?}");
                assert!(f64::from((got - dot.unwrap()).abs()) <= bound, "{msg}");
                let got = k.cosine_i8(&qa, &qb).unwrap();
                let msg = format!("{k}, len {len}: {got} vs {cos:?}");
                assert!((got - cos.unwrap()).abs() <= cos_bound, "{msg}");
            }
        }
    }

    #[test]
    fn mismatch_zero_and_empty_follow_the_f32_convention() {
        let x = QuantizedVector::quantize(&noise(3, 19)).unwrap();
        let short = QuantizedVector::quantize(&noise(4, 18)).unwrap();
        let zero = QuantizedVector::quantize(&[0.0; 19]).unwrap();
        let empty = QuantizedVector::quantize(&[]).unwrap();
        let mismatch = Err(SimilarityError::DimensionMismatch {
            expected: 19,
            actual: 18,
        });
        let zero_norm = Err(SimilarityError::ZeroNorm);
        for k in Kernel::PREFERENCE {
            assert_eq!(
                (k.dot_i8(&x, &short), k.cosine_i8(&zero, &short)),
                (mismatch, mismatch)
            );
            assert_eq!(
                (k.dot_i8(&x, &zero), k.dot_i8(&empty, &empty)),
                (Ok(0.0), Ok(0.0))
            );
            assert_eq!(
                (k.cosine_i8(&x, &zero), k.cosine_i8(&zero, &x)),
                (zero_norm, zero_norm)
            );
            assert_eq!(k.cosine_i8(&empty, &empty), zero_norm);
        }
        // Os métodos usam o kernel ativo.
        assert_eq!(x.dot(&short), mismatch);
        assert_eq!(x.dot(&x), Kernel::active().dot_i8(&x, &x));
        assert_eq!(x.cosine(&x), Ok(1.0));
    }
}
