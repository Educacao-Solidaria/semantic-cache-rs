//! IPC v1 entre o gateway Go e o cache (`docs/ipc-v1.md`): codec dos frames,
//! resposta em stream, servidor em Unix domain socket (no Windows, named pipe)
//! e as métricas dele.

pub mod frame;
pub mod metrics;
pub mod server;
pub mod stream;
