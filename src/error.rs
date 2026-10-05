use crate::domain::VectorError;
use thiserror::Error;

/// Hierarquia formal de erros do motor semantic-cache-rs.
#[derive(Error, Debug, PartialEq)]
pub enum CacheError {
    #[error("erro vetorial de calculo: {0}")]
    Vector(#[from] VectorError),

    #[error("entrada com chave '{0}' nao encontrada")]
    EntryNotFound(String),

    #[error("capacidade maxima do cache atingida: atual {current}, maximo {max}")]
    CapacityExceeded { max: usize, current: usize },

    #[error("entrada '{0}' expirou por TTL")]
    TtlExpired(String),

    #[error("corrupcao de dados na entrada: {0}")]
    CorruptedData(String),

    #[error("falha na serializacao/desserializacao: {0}")]
    Serialization(String),

    #[error("erro de entrada e saida (I/O): {0}")]
    Io(String),

    #[error("erro interno no motor de cache: {0}")]
    Internal(String),
}

impl CacheError {
    /// Determina se o erro é transitório e passível de retry.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::CapacityExceeded { .. } | Self::Internal(_))
    }

    /// Helper rápido para instanciar erro de capacidade excedida.
    pub fn capacity_exceeded(max: usize, current: usize) -> Self {
        Self::CapacityExceeded { max, current }
    }

    /// Helper rápido para instanciar erro de corrupção.
    pub fn corrupted(msg: impl Into<String>) -> Self {
        Self::CorruptedData(msg.into())
    }
}

impl From<std::io::Error> for CacheError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

/// Alias canônico de Result para operações do cache.
pub type CacheResult<T> = Result<T, CacheError>;
