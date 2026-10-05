use crate::domain::{SimilarityThreshold, Vector};
use serde::{Deserialize, Serialize};

/// Requisição para a ferramenta MCP `semantic_cache_lookup`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupRequest {
    pub prompt: String,
    pub embedding: Vector,
    #[serde(default)]
    pub threshold: Option<f32>,
    pub tenant_id: Option<String>,
}

/// Resposta emitida pela ferramenta MCP `semantic_cache_lookup`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupResponse {
    pub hit: bool,
    pub score: f32,
    pub cached_response: Option<String>,
    pub latency_us: u64,
}

/// Requisição para a ferramenta MCP `semantic_cache_store`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreRequest {
    pub prompt: String,
    pub response: String,
    pub embedding: Vector,
    pub ttl_seconds: Option<u64>,
    pub tenant_id: Option<String>,
}

/// Resposta emitida pela ferramenta MCP `semantic_cache_store`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreResponse {
    pub success: bool,
    pub entry_id: String,
    pub total_entries: usize,
}

/// Contrato da trait canônica para integração do motor de cache ao protocolo MCP.
pub trait SemanticCacheEngine: Send + Sync {
    fn lookup(&self, req: LookupRequest, threshold: SimilarityThreshold) -> LookupResponse;
    fn store(&self, req: StoreRequest) -> StoreResponse;
    fn count(&self) -> usize;
    fn clear(&self);
}
