//! IPC v1 entre o gateway Go e o cache (`docs/ipc-v1.md`): codec dos frames,
//! resposta em stream e servidor em Unix domain socket (no Windows, named pipes
//! no PR 61/100).

pub mod frame;
#[cfg(unix)]
pub mod server;
pub mod stream;
