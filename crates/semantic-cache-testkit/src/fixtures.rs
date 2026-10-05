//! Gerador determinístico de vetores unitários aleatórios.
//!
//! A mesma semente produz o mesmo fluxo de bits em qualquer plataforma e
//! versão: o gerador é um `SplitMix64` próprio, e não o `rand`, cujo fluxo de
//! saída pode mudar entre versões e quebrar fixtures gravadas. Os vetores são
//! idênticos bit a bit na mesma plataforma; entre plataformas podem diferir no
//! último bit, porque `ln` e `cos` vêm da libm do sistema. A direção é uniforme
//! na esfera (componentes gaussianas normalizadas), que é o que embeddings
//! normalizados de modelos parecem para uma métrica de cosseno.

use semantic_cache_rs::{Vector, VectorBatch};

/// Gerador de vetores unitários com semente fixa.
#[derive(Debug, Clone)]
pub struct VectorGen {
    state: u64,
}

impl VectorGen {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// `SplitMix64` (Steele, Lea e Flood, 2014).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniforme em `(0, 1]`: nunca zero, para o `ln` do Box-Muller.
    #[allow(clippy::cast_precision_loss)] // 53 bits cabem exatos no f64
    fn next_open01(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Normal padrão por Box-Muller; a segunda amostra do par é descartada.
    fn gaussian(&mut self) -> f64 {
        let (u1, u2) = (self.next_open01(), self.next_open01());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Vetor de norma 1 com direção uniforme na esfera de `dim` dimensões.
    ///
    /// # Panics
    ///
    /// Se `dim == 0`.
    #[must_use]
    pub fn unit(&mut self, dim: usize) -> Vector {
        assert!(dim > 0, "dimensão precisa ser maior que zero");
        loop {
            let raw: Vec<f64> = (0..dim).map(|_| self.gaussian()).collect();
            // Norma zero só com todas as gaussianas nulas; sorteia de novo.
            if let Some(v) = to_unit(&raw) {
                return v;
            }
        }
    }

    /// Lote de `n` vetores unitários de dimensão `dim`.
    ///
    /// # Panics
    ///
    /// Se `dim == 0`.
    #[must_use]
    pub fn batch(&mut self, n: usize, dim: usize) -> VectorBatch {
        let mut batch = VectorBatch::new(dim);
        for _ in 0..n {
            batch.push(self.unit(dim)).expect("mesma dimensão do lote");
        }
        batch
    }

    /// Vetor unitário com similaridade cosseno `cosine` em relação a `base`:
    /// `cosine * base + sqrt(1 - cosine²) * u`, com `u` unitário ortogonal a
    /// `base`. Serve para montar pares logo acima e logo abaixo do limiar de
    /// hit do cache.
    ///
    /// # Panics
    ///
    /// Se `cosine` estiver fora de `[-1, 1]`, se `base` tiver norma zero ou
    /// dimensão 1 (não existe direção ortogonal).
    #[must_use]
    pub fn with_cosine(&mut self, base: &Vector, cosine: f32) -> Vector {
        assert!((-1.0..=1.0).contains(&cosine), "cosseno fora de [-1, 1]");
        assert!(base.dim() > 1, "dimensão 1 não tem direção ortogonal");
        let b: Vec<f64> = base.data.iter().map(|&x| f64::from(x)).collect();
        let b = unit_f64(&b).expect("base com norma zero");
        let c = f64::from(cosine);
        loop {
            // Gram-Schmidt: tira de um sorteio a componente na direção da base.
            let r: Vec<f64> = (0..b.len()).map(|_| self.gaussian()).collect();
            let proj: f64 = r.iter().zip(&b).map(|(x, y)| x * y).sum();
            let orth: Vec<f64> = r.iter().zip(&b).map(|(x, y)| x - proj * y).collect();
            if let Some(u) = unit_f64(&orth) {
                let s = (1.0 - c * c).sqrt();
                let mixed: Vec<f64> = b.iter().zip(&u).map(|(x, y)| c * x + s * y).collect();
                return to_unit(&mixed).expect("combinação de unitários ortogonais");
            }
        }
    }
}

fn unit_f64(v: &[f64]) -> Option<Vec<f64>> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    (norm > f64::EPSILON).then(|| v.iter().map(|x| x / norm).collect())
}

/// Normaliza em f64 e só então converte: o erro de arredondamento fica no
/// último bit do f32, não acumulado na soma.
#[allow(clippy::cast_possible_truncation)] // f64 → f32 é o objetivo
fn to_unit(v: &[f64]) -> Option<Vector> {
    let data = unit_f64(v)?.into_iter().map(|x| x as f32).collect();
    Vector::new(data).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    #[test]
    fn same_seed_same_vectors() {
        let a = VectorGen::new(42).batch(8, 64);
        let b = VectorGen::new(42).batch(8, 64);
        assert_eq!(a.vectors, b.vectors);
        assert_ne!(a.vectors, VectorGen::new(43).batch(8, 64).vectors);
    }

    #[test]
    fn stream_is_pinned() {
        // Trava o fluxo: se mudar, toda fixture gravada com semente muda junto.
        let mut g = VectorGen::new(0);
        assert_eq!(g.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(g.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn vectors_are_unit_norm() {
        let batch = VectorGen::new(7).batch(32, 1536);
        assert_eq!(batch.len(), 32);
        assert_eq!(batch.dimension, 1536);
        for v in &batch.vectors {
            assert_eq!(v.dim(), 1536);
            crate::assert_f32_near!(v.norm(), 1.0, abs = TOL);
        }
    }

    #[test]
    fn dimension_one_is_plus_or_minus_one() {
        let v = VectorGen::new(1).unit(1);
        crate::assert_f32_near!(v.data[0].abs(), 1.0, abs = TOL);
    }

    #[test]
    fn directions_are_spread_out() {
        // Em 768 dimensões, o cosseno entre direções uniformes concentra em
        // torno de 0 com desvio ~1/sqrt(768) ≈ 0.036.
        let batch = VectorGen::new(9).batch(20, 768);
        for (i, a) in batch.vectors.iter().enumerate() {
            for b in &batch.vectors[i + 1..] {
                crate::assert_f32_near!(a.cosine_similarity(b).unwrap(), 0.0, abs = 0.25);
            }
        }
    }

    #[test]
    fn with_cosine_hits_target() {
        let mut g = VectorGen::new(5);
        let base = g.unit(384);
        for target in [0.999, 0.92, 0.5, 0.0, -0.7, 1.0, -1.0] {
            let v = g.with_cosine(&base, target);
            crate::assert_f32_near!(v.norm(), 1.0, abs = TOL);
            let got = v.cosine_similarity(&base).unwrap();
            crate::assert_f32_near!(got, target, abs = 1e-4);
        }
    }

    #[test]
    #[should_panic(expected = "dimensão precisa ser maior que zero")]
    fn zero_dimension_panics() {
        let _ = VectorGen::new(0).unit(0);
    }
}
