//! Quantização escalar de vetores `f32` em `i8`: `x ≈ escala · q`, com `q` em
//! `[-127, 127]` e uma escala `f32` por vetor (`max|x| / 127`).
//!
//! Um byte por componente em vez de quatro: um embedding de 1536 dimensões
//! cai de 6144 para 1540 bytes (−74,9%). O erro de cada componente é no máximo
//! metade da escala ([`QuantizedVector::max_error`]).
//!
//! Simétrica, e não afim (com ponto zero): componentes de embedding se
//! distribuem em torno de zero, então `[-max, max]` já é a faixa útil, o zero
//! é exato e o produto escalar quantizado é `escalaₐ · escala_b · Σ qₐ q_b`,
//! sem termos de correção — o formato que um kernel int8 consome direto.
//! ponytail: vetor muito assimétrico (todo positivo, por exemplo) desperdiça
//! metade da faixa, até 1 bit de precisão; o caminho é a forma afim, com um
//! ponto zero por vetor.

use std::fmt;

/// Maior código; `-128` fica de fora para a faixa ser simétrica.
const MAX_CODE: f32 = 127.0;

/// Vetor quantizado: escala e um código `i8` por componente.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVector {
    scale: f32,
    codes: Vec<i8>,
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
            return Ok(Self {
                scale: 0.0,
                codes: vec![0; values.len()],
            });
        }
        // Em f64: com escala subnormal, `x / escala` em f32 perderia bits.
        #[allow(clippy::cast_possible_truncation)] // `q` já está em [-127, 127]
        let code = |x: f32| {
            (f64::from(x) / f64::from(scale))
                .round()
                .clamp(-127.0, 127.0) as i8
        };
        Ok(Self {
            scale,
            codes: values.iter().map(|&x| code(x)).collect(),
        })
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
            let qa = QuantizedVector::quantize(&a).unwrap().dequantize();
            for other in [&b, &near] {
                let qo = QuantizedVector::quantize(other).unwrap().dequantize();
                let delta = cosine(&qa, &qo).unwrap() - cosine(&a, other).unwrap();
                assert!(delta.abs() < 2e-3, "seed {seed}: Δcos = {delta:e}");
            }
        }
        let quantized = QuantizedVector::quantize(&noise(9, 1536)).unwrap();
        let bytes = std::mem::size_of_val(quantized.codes()) + std::mem::size_of::<f32>();
        assert_eq!((bytes, 1536 * 4), (1540, 6144));
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
}
