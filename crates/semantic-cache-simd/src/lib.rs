//! Instruções SIMD do `semantic-cache-rs`.
//!
//! - [`cpu`]: detecção das extensões suportadas em tempo de execução e
//!   seleção do kernel de similaridade.
//! - [`similarity`]: produto escalar e cosseno despachados para o kernel
//!   ativo, com a referência escalar como fallback.
//! - [`quant`]: quantização escalar de vetores `f32` em `i8` (um quarto da
//!   memória).

pub mod cpu;
pub mod quant;
pub mod similarity;

pub use cpu::{CpuFeatures, Kernel};
pub use quant::{NonFinite, QuantizedVector};
pub use similarity::{cosine, dot, SimilarityError};
