//! # semantic-cache-rs
//!
//! Sub-millisecond vector semantic cache engine designed for high-concurrency
//! AI systems with explicit SIMD support and Model Context Protocol (MCP) tooling.

pub mod domain;
pub mod mcp;

pub use domain::{CacheEntry, SimilarityThreshold, Vector, VectorError};
pub use mcp::{LookupRequest, LookupResponse, SemanticCacheEngine, StoreRequest, StoreResponse};
