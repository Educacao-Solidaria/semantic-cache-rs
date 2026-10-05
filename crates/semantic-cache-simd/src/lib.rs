//! Instruções SIMD do `semantic-cache-rs`.
//!
//! - [`cpu`]: detecção das extensões suportadas em tempo de execução e
//!   seleção do kernel de similaridade.

pub mod cpu;

pub use cpu::{CpuFeatures, Kernel};
