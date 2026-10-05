use crate::domain::{Vector, VectorError};
use crate::metric::DistanceMetric;

/// Vetor com magnitude pré-calculada para evitar recomputações frequentes de norma.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedVector {
    pub inner: Vector,
    pub norm: f32,
}

impl NormalizedVector {
    pub fn new(vector: Vector) -> Result<Self, VectorError> {
        let norm = vector.norm();
        if norm == 0.0 {
            return Err(VectorError::ZeroNorm);
        }
        Ok(Self {
            inner: vector,
            norm,
        })
    }

    /// Calcula a similaridade cosseno de referência sem normalização in-place.
    pub fn cosine_similarity_to(&self, other: &Self) -> Result<f32, VectorError> {
        let dot = self.inner.dot(&other.inner)?;
        Ok(dot / (self.norm * other.norm))
    }
}

/// Implementação pura de referência da Distância Cosseno (`1.0 - similarity`).
#[derive(Debug, Default, Clone, Copy)]
pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, a: &Vector, b: &Vector) -> Result<f32, VectorError> {
        let sim = a.cosine_similarity(b)?;
        // Distância cosseno varia entre 0.0 (idênticos) e 2.0 (opostos)
        Ok((1.0 - sim).max(0.0))
    }

    fn name(&self) -> &'static str {
        "cosine_distance"
    }
}

/// Determina se dois vetores são estritamente ortogonais (produto escalar próximo a zero).
pub fn is_orthogonal(a: &Vector, b: &Vector, tolerance: f32) -> Result<bool, VectorError> {
    let dot = a.dot(b)?;
    Ok(dot.abs() <= tolerance)
}

/// Normaliza um slice de floats in-place retornando a norma anterior.
pub fn normalize_slice_in_place(slice: &mut [f32]) -> Result<f32, VectorError> {
    let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Err(VectorError::ZeroNorm);
    }
    for x in slice.iter_mut() {
        *x /= norm;
    }
    Ok(norm)
}

/// Calcula a distância euclidiana ao quadrado pura entre dois slices.
pub fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let sum: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    Ok(sum)
}

/// Calcula a distância euclidiana (L2) padrão pura entre dois slices de mesma dimensão.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    let sq = squared_euclidean_distance(a, b)?;
    Ok(sq.sqrt())
}

/// Calcula o produto escalar (dot product) puro entre dois slices de floats.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

/// Calcula o produto escalar com normalização L2 em tempo de execução.
pub fn normalized_dot_product(a: &[f32], b: &[f32]) -> Result<f32, VectorError> {
    let dot = dot_product(a, b)?;
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(VectorError::ZeroNorm);
    }
    Ok(dot / (norm_a * norm_b))
}
