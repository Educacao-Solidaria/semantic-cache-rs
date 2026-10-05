//! # semantic-cache-rs
//!
//! Sub-millisecond vector semantic cache engine designed for high-concurrency
//! AI systems with explicit SIMD support and Model Context Protocol (MCP) tooling.

pub mod domain;
pub mod error;
pub mod math;
pub mod mcp;
pub mod metric;
pub mod threshold;

pub use error::{CacheError, CacheResult};
pub use math::{CosineDistance, NormalizedVector};
pub use metric::{CosineSimilarity, DistanceMetric, DotProduct, EuclideanDistance, SimilarityCalculator};
pub use threshold::{MatchLevel, SimilarityThresholdConfig};

pub use domain::{CacheEntry, Metadata, SimilarityThreshold, Vector, VectorBatch, VectorError};
pub use mcp::{LookupRequest, LookupResponse, SemanticCacheEngine, StoreRequest, StoreResponse};
