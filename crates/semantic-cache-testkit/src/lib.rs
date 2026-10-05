//! Ferramental de teste e medição do `semantic-cache-rs`.
//!
//! - [`asserts`]: macros de comparação de `f32` e vetores com tolerância.
//! - [`bench`]: preset de precisão do criterion e dimensões de embedding
//!   medidas pelos benchmarks em `benches/`.
//! - [`datasets`]: formato e loader de fixtures de embeddings de modelos.
//! - [`fixtures`]: gerador determinístico de vetores unitários.

pub mod asserts;
pub mod bench;
pub mod datasets;
pub mod fixtures;
