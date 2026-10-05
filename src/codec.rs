use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::domain::{CacheEntry, Vector};
use crate::error::CacheError;

/// Limite padrão máximo de payload binário permitido (16 MB).
pub const DEFAULT_MAX_BINARY_SIZE: usize = 16 * 1024 * 1024;

/// Trait para codecs de serialização e desserialização binária compacta.
pub trait BinaryCodec: Send + Sync {
    /// Serializa uma estrutura de dados compatível com Serde para vetor de bytes.
    fn encode<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, CacheError>;

    /// Desserializa bytes brutos em uma estrutura tipada.
    fn decode<T: DeserializeOwned>(&self, bytes: &[u8]) -> Result<T, CacheError>;
}

/// Implementação do codec binário de alta performance utilizando Bincode.
#[derive(Debug, Clone, Copy)]
pub struct BincodeCodec {
    max_size_bytes: usize,
}

impl Default for BincodeCodec {
    fn default() -> Self {
        Self {
            max_size_bytes: DEFAULT_MAX_BINARY_SIZE,
        }
    }
}

impl BincodeCodec {
    pub fn new(max_size_bytes: usize) -> Self {
        Self { max_size_bytes }
    }

    pub fn max_size(&self) -> usize {
        self.max_size_bytes
    }
}

impl BinaryCodec for BincodeCodec {
    fn encode<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, CacheError> {
        bincode::serialize(value).map_err(|e| CacheError::Serialization(e.to_string()))
    }

    fn decode<T: DeserializeOwned>(&self, bytes: &[u8]) -> Result<T, CacheError> {
        if bytes.len() > self.max_size_bytes {
            return Err(CacheError::Serialization(format!(
                "payload excede tamanho maximo permitido: {} > {}",
                bytes.len(),
                self.max_size_bytes
            )));
        }
        bincode::deserialize(bytes).map_err(|e| CacheError::Serialization(e.to_string()))
    }
}

/// Helper para serialização rápida de um CacheEntry em bytes compactos.
pub fn serialize_entry(entry: &CacheEntry) -> Result<Vec<u8>, CacheError> {
    BincodeCodec::default().encode(entry)
}

/// Helper para restauração de um CacheEntry a partir de bytes bincode.
pub fn deserialize_entry(bytes: &[u8]) -> Result<CacheEntry, CacheError> {
    BincodeCodec::default().decode(bytes)
}

/// Helper para serialização direta de vetor numérico denso.
pub fn serialize_vector(vector: &Vector) -> Result<Vec<u8>, CacheError> {
    BincodeCodec::default().encode(vector)
}

/// Helper para reconstrução de vetor numérico denso a partir de bytes.
pub fn deserialize_vector(bytes: &[u8]) -> Result<Vector, CacheError> {
    BincodeCodec::default().decode(bytes)
}
