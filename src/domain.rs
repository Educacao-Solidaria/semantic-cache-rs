use serde::{Deserialize, Serialize};

/// Erro de validação ou cálculo do domínio vetorial.
#[derive(Debug, PartialEq, Eq)]
pub enum VectorError {
    DimensionMismatch { expected: usize, actual: usize },
    EmptyVector,
    ZeroNorm,
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimensao incompativel: esperado {expected}, recebido {actual}")
            }
            Self::EmptyVector => write!(f, "vetor nao pode ser vazio"),
            Self::ZeroNorm => write!(f, "norma do vetor e zero (impossivel normalizar)"),
        }
    }
}

impl std::error::Error for VectorError {}

/// Representação de um vetor denso de ponto flutuante.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    pub data: Vec<f32>,
}

impl Vector {
    /// Cria um novo vetor validando que não seja vazio.
    pub fn new(data: Vec<f32>) -> Result<Self, VectorError> {
        if data.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        Ok(Self { data })
    }

    /// Retorna a dimensão do vetor.
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Calcula o produto escalar (dot product) com outro vetor.
    pub fn dot(&self, other: &Self) -> Result<f32, VectorError> {
        if self.dim() != other.dim() {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim(),
                actual: other.dim(),
            });
        }
        let sum: f32 = self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum();
        Ok(sum)
    }

    /// Calcula a norma L2 (magnitude) do vetor.
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Calcula a similaridade cosseno com outro vetor (resultado entre -1.0 e 1.0).
    pub fn cosine_similarity(&self, other: &Self) -> Result<f32, VectorError> {
        let dot = self.dot(other)?;
        let n1 = self.norm();
        let n2 = other.norm();
        if n1 == 0.0 || n2 == 0.0 {
            return Err(VectorError::ZeroNorm);
        }
        Ok(dot / (n1 * n2))
    }
}

/// Registro estruturado armazenado no cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub id: String,
    pub prompt: String,
    pub response: String,
    pub embedding: Vector,
    pub hit_count: u64,
    pub created_at: u64,
    pub ttl_seconds: u64,
}

impl CacheEntry {
    pub fn is_expired(&self, current_time_seconds: u64) -> bool {
        if self.ttl_seconds == 0 {
            return false; // Sem expiração
        }
        current_time_seconds > self.created_at + self.ttl_seconds
    }
}

/// Configuração de threshold de similaridade semântica.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimilarityThreshold {
    pub min_score: f32,
}

impl Default for SimilarityThreshold {
    fn default() -> Self {
        Self { min_score: 0.92 }
    }
}

impl SimilarityThreshold {
    pub fn is_hit(&self, score: f32) -> bool {
        score >= self.min_score
    }
}
