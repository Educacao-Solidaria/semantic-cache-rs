//! Instruções SIMD do `semantic-cache-rs`.
//!
//! - [`cpu`]: detecção das extensões suportadas em tempo de execução e
//!   seleção do kernel de similaridade.
//! - [`similarity`]: produto escalar e cosseno despachados para o kernel
//!   ativo, com a referência escalar como fallback.
//! - [`quant`]: quantização escalar de vetores `f32` em `i8` (um quarto da
//!   memória), com produto escalar e cosseno inteiros no kernel ativo.
//! - [`profile`]: profiler embutido, desligado por padrão, com o tempo de cada
//!   quantização, busca e descompressão em histograma (p50, p99, máximo).
//! - [`aligned`]: vetor `f32` alinhado a 32 bytes e completado com zeros até
//!   múltiplo de 8, para nenhuma carga AVX atravessar linha de cache.

pub mod aligned;
pub mod cpu;
pub mod profile;
pub mod quant;
pub mod similarity;

pub use aligned::AlignedVector;
pub use cpu::{CpuFeatures, Kernel};
pub use quant::{NonFinite, QuantizedVector};
pub use similarity::{cosine, dot, SimilarityError};
