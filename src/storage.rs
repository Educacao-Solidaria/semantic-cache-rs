use std::collections::HashMap;

use crate::domain::CacheEntry;
use crate::error::CacheError;

/// Trait abstrata para provedores de armazenamento de entradas de cache semântico.
pub trait CacheStore: Send + Sync {
    /// Recupera uma entrada pelo identificador textual ou hash da chave.
    fn get(&self, key: &str) -> Result<Option<CacheEntry>, CacheError>;

    /// Insere ou atualiza uma entrada associada a uma chave.
    fn set(&mut self, key: String, entry: CacheEntry) -> Result<(), CacheError>;

    /// Remove explicitamente uma chave do armazenamento.
    fn evict(&mut self, key: &str) -> Result<bool, CacheError>;

    /// Quantidade atual de itens persistidos no armazenamento.
    fn len(&self) -> usize;

    /// Indica se o armazenamento está vazio.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Esvazia completamente o armazenamento.
    fn clear(&mut self) -> Result<(), CacheError>;
}

/// Implementação em memória baseada em HashMap com limite de capacidade configurável.
#[derive(Debug, Clone, Default)]
pub struct InMemoryStore {
    entries: HashMap<String, CacheEntry>,
    max_capacity: Option<usize>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            max_capacity: None,
        }
    }

    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_capacity),
            max_capacity: Some(max_capacity),
        }
    }

    pub fn capacity_limit(&self) -> Option<usize> {
        self.max_capacity
    }
}

impl CacheStore for InMemoryStore {
    fn get(&self, key: &str) -> Result<Option<CacheEntry>, CacheError> {
        Ok(self.entries.get(key).cloned())
    }

    fn set(&mut self, key: String, entry: CacheEntry) -> Result<(), CacheError> {
        if let Some(limit) = self.max_capacity {
            if self.entries.len() >= limit && !self.entries.contains_key(&key) {
                return Err(CacheError::CapacityExceeded {
                    max: limit,
                    current: self.entries.len(),
                });
            }
        }
        self.entries.insert(key, entry);
        Ok(())
    }

    fn evict(&mut self, key: &str) -> Result<bool, CacheError> {
        Ok(self.entries.remove(key).is_some())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) -> Result<(), CacheError> {
        self.entries.clear();
        Ok(())
    }
}
