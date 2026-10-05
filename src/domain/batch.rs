use crate::domain::{Vector, VectorError};
use serde::{Deserialize, Serialize};

/// Lote de vetores homogêneos para computação em lote.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorBatch {
    pub dimension: usize,
    pub vectors: Vec<Vector>,
}

impl VectorBatch {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: Vec::new(),
        }
    }

    pub fn push(&mut self, vec: Vector) -> Result<(), VectorError> {
        if vec.dim() != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: vec.dim(),
            });
        }
        self.vectors.push(vec);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Normaliza todos os vetores contidos no lote.
    pub fn normalize_all(&mut self) -> Result<(), VectorError> {
        for vec in &mut self.vectors {
            let norm = vec.norm();
            if norm == 0.0 {
                return Err(VectorError::ZeroNorm);
            }
            for val in &mut vec.data {
                *val /= norm;
            }
        }
        Ok(())
    }
}
