//! Instruções SIMD do `semantic-cache-rs`.
//!
//! - [`cpu`]: detecção das extensões suportadas em tempo de execução e
//!   seleção do kernel de similaridade.
//! - [`similarity`]: produto escalar e cosseno despachados para o kernel
//!   ativo, com a referência escalar como fallback.

pub mod cpu;
pub mod similarity;

pub use cpu::{CpuFeatures, Kernel};
pub use similarity::{cosine, dot, SimilarityError};
