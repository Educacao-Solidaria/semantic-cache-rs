use crate::domain::{Vector, VectorError};

/// Trait abstrata para métricas de distância geométrica entre vetores.
pub trait DistanceMetric: Send + Sync {
    fn distance(&self, a: &Vector, b: &Vector) -> Result<f32, VectorError>;
    fn name(&self) -> &'static str;
}

/// Trait abstrata para cálculo de similaridade entre vetores.
pub trait SimilarityCalculator: Send + Sync {
    fn similarity(&self, a: &Vector, b: &Vector) -> Result<f32, VectorError>;
    fn name(&self) -> &'static str;
}

/// Implementação de Similaridade Cosseno (resultado entre -1.0 e 1.0).
#[derive(Debug, Default, Clone, Copy)]
pub struct CosineSimilarity;

impl SimilarityCalculator for CosineSimilarity {
    fn similarity(&self, a: &Vector, b: &Vector) -> Result<f32, VectorError> {
        a.cosine_similarity(b)
    }

    fn name(&self) -> &'static str {
        "cosine"
    }
}

/// Implementação de Distância Euclidiana (L2).
#[derive(Debug, Default, Clone, Copy)]
pub struct EuclideanDistance;

impl DistanceMetric for EuclideanDistance {
    fn distance(&self, a: &Vector, b: &Vector) -> Result<f32, VectorError> {
        if a.dim() != b.dim() {
            return Err(VectorError::DimensionMismatch {
                expected: a.dim(),
                actual: b.dim(),
            });
        }
        let sum_sq: f32 = a
            .data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum();
        Ok(sum_sq.sqrt())
    }

    fn name(&self) -> &'static str {
        "euclidean_l2"
    }
}

/// Implementação de Produto Escalar (Dot Product).
#[derive(Debug, Default, Clone, Copy)]
pub struct DotProduct;

impl SimilarityCalculator for DotProduct {
    fn similarity(&self, a: &Vector, b: &Vector) -> Result<f32, VectorError> {
        a.dot(b)
    }

    fn name(&self) -> &'static str {
        "dot_product"
    }
}
