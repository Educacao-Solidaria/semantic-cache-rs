//! Reabertura do sled nos testes: incluído pelos testes de `src/persistence.rs`
//! e por `tests/concurrency_stress.rs`, que abrem de novo, no mesmo processo, o
//! banco que acabaram de soltar.

use std::fmt::Display;
use std::thread;
use std::time::{Duration, Instant};

/// Abre de novo um banco sled que este processo acabou de soltar.
///
/// O sled 0.34 trava o arquivo `db` com `flock`, e a trava só cai quando fecha
/// o último `Arc<File>` dele. As tarefas do threadpool de IO do sled
/// (`sled-io-*`) guardam um clone (o `Arc<IoBufs>` da escrita de um buffer, o
/// `RunningConfig` da truncagem) e seguem trabalhando depois de acordar o flush
/// do `drop`: reabrir logo em seguida pode dar `could not acquire lock …
/// WouldBlock`. Só esse erro é repetido; qualquer outro sai na hora.
///
/// ponytail: espera até 10 s pela trava, tentando a cada 10 ms; passou disso, o
/// erro dela sai como veio (quem segura o arquivo não é o sled terminando).
/// Produção abre o banco uma vez por processo; se passar a reabrir no mesmo
/// processo, esta espera vai para o `SledStore::open`.
pub fn reopen<T, E: Display>(mut open: impl FnMut() -> Result<T, E>) -> Result<T, E> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match open() {
            Err(err) if is_locked(&err) && Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(10));
            }
            result => return result,
        }
    }
}

/// O erro do sled quando o `flock` do arquivo `db` tem outro dono.
fn is_locked(err: &impl Display) -> bool {
    let msg = err.to_string();
    msg.contains("could not acquire lock") && msg.contains("WouldBlock")
}
