//! Ferramental de teste e medição do `semantic-cache-rs`.
//!
//! - [`accuracy`]: detector de degradação da quantização int8 (recall@k contra
//!   a busca exata, em grupos com estrutura de embedding, com piso).
//! - [`asserts`]: macros de comparação de `f32` e vetores com tolerância.
//! - [`bench`]: preset de precisão do criterion e dimensões de embedding
//!   medidas pelos benchmarks em `benches/`.
//! - [`datasets`]: formato e loader de fixtures de embeddings de modelos.
//! - [`fixtures`]: gerador determinístico de vetores unitários.
//! - [`stress`]: harness de estresse (latência p50/p95/p99 e recall) com os
//!   índices por trait.

pub mod accuracy;
pub mod asserts;
pub mod bench;
pub mod datasets;
pub mod fixtures;
pub mod stress;
