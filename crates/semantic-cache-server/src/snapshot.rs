//! Leitura sem trava de um estado compartilhado (o índice, por exemplo):
//! leitores pegam o retrato atual sem bloquear e sem disputar uma linha de
//! cache (o `arc-swap` anota o empréstimo numa lista da própria thread); quem
//! atualiza monta o próximo estado e troca o ponteiro atomicamente
//! ([`arc_swap::ArcSwap`]).
//!
//! - Leitor nunca espera escritor: [`SnapshotCell::read`] e
//!   [`SnapshotCell::load`] não tocam na trava dos escritores.
//! - Escritores são serializados entre si: [`SnapshotCell::update`] lê o
//!   estado, calcula o próximo e publica sem que outro escritor publique no
//!   meio — sem a trava, duas atualizações concorrentes partiriam do mesmo
//!   estado e uma se perderia.
//! - O retrato antigo vive enquanto algum leitor o segura e é liberado quando
//!   o último solta (contagem do `Arc`): com um índice grande, esse leitor
//!   paga a liberação. Se pesar, o escritor guarda o antigo e o solta fora do
//!   caminho de leitura.

use std::sync::{Arc, Mutex, PoisonError};

use arc_swap::ArcSwap;

/// Estado publicado para leitura sem trava.
#[derive(Debug)]
pub struct SnapshotCell<T> {
    current: ArcSwap<T>,
    /// Só escritores tomam; leitores nunca.
    writer: Mutex<()>,
}

impl<T> SnapshotCell<T> {
    #[must_use]
    pub fn new(value: T) -> Self {
        Self {
            current: ArcSwap::from_pointee(value),
            writer: Mutex::new(()),
        }
    }

    /// Roda `f` sobre o retrato atual, sem trava. Uma troca durante `f` não o
    /// afeta: o retrato fica vivo até `f` voltar. Para segurar o retrato por
    /// mais tempo (entre chamadas, numa tarefa), use [`SnapshotCell::load`].
    pub fn read<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        f(&self.current.load())
    }

    /// O retrato atual, que o chamador segura o quanto quiser.
    #[must_use]
    pub fn load(&self) -> Arc<T> {
        self.current.load_full()
    }

    /// Publica `f(atual)`. Escritores esperam uns pelos outros; leitores, por
    /// ninguém. Um panic em `f` não publica nada e não trava os próximos.
    /// Dentro de `f`, ler a célula pode; escrever nela (`update`/`store`)
    /// trava para sempre.
    pub fn update(&self, f: impl FnOnce(&T) -> T) {
        // A trava protege só a ordem dos escritores; o estado nunca fica pela
        // metade (a troca é depois de `f`), então envenenada ela segue boa.
        let _writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        let next = f(&self.current.load_full());
        self.current.store(Arc::new(next));
    }

    /// Publica `value`, na mesma fila dos outros escritores (uma troca solta
    /// no meio de um [`SnapshotCell::update`] seria sobrescrita por ele).
    pub fn store(&self, value: T) {
        self.update(|_| value);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{mpsc, Barrier, RwLock};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;

    /// Conta quantos valores já foram liberados.
    struct Tracked(u64, Arc<AtomicUsize>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn old_snapshot_lives_until_its_last_reader_lets_go() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let cell = SnapshotCell::new(Tracked(1, Arc::clone(&dropped)));
        let held = cell.load();
        cell.store(Tracked(2, Arc::clone(&dropped)));
        assert_eq!((held.0, cell.read(|v| v.0)), (1, 2));
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            0,
            "o leitor ainda segura o 1"
        );
        drop(held);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        // Troca dentro de uma leitura: o leitor segue com o retrato dele.
        let seen = cell.read(|v| {
            cell.store(Tracked(3, Arc::clone(&dropped)));
            v.0
        });
        assert_eq!((seen, cell.read(|v| v.0)), (2, 3));
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
        drop(cell);
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn concurrent_updates_are_never_lost() {
        let cell = SnapshotCell::new(0u64);
        let start = Barrier::new(8);
        thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    start.wait();
                    for _ in 0..2_000 {
                        cell.update(|n| n + 1);
                    }
                });
            }
        });
        assert_eq!(cell.read(|n| *n), 16_000);
    }

    #[test]
    fn readers_see_whole_states_in_order_while_writers_swap() {
        // Invariante do estado: todos os elementos iguais. Um retrato pela
        // metade, ou voltando no tempo, quebraria a conferência.
        let cell = SnapshotCell::new(vec![0u64; 64]);
        let done = AtomicBool::new(false);
        thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    let mut last = 0;
                    while !done.load(Ordering::Relaxed) {
                        let first = cell.read(|v| {
                            assert!(v.iter().all(|x| *x == v[0]), "retrato rasgado");
                            v[0]
                        });
                        assert!(first >= last, "voltou de {last} para {first}");
                        last = first;
                    }
                });
            }
            let write = || {
                for _ in 0..2_000 {
                    cell.update(|v| vec![v[0] + 1; 64]);
                }
            };
            let (a, b) = (s.spawn(write), s.spawn(write));
            let joined = (a.join(), b.join());
            done.store(true, Ordering::Relaxed); // antes de conferir: senão os leitores giram para sempre
            assert!(joined.0.is_ok() && joined.1.is_ok());
        });
        assert_eq!(cell.read(|v| v[0]), 4_000);
    }

    #[test]
    fn a_reader_does_not_wait_for_a_writer_in_the_middle_of_an_update() {
        let cell = Arc::new(SnapshotCell::new(0u64));
        let (inside, release) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
        let writer = {
            let (cell, inside, release) =
                (Arc::clone(&cell), Arc::clone(&inside), Arc::clone(&release));
            thread::spawn(move || {
                cell.update(|n| {
                    inside.wait();
                    release.wait(); // só termina depois que o leitor leu
                    n + 1
                });
            })
        };
        inside.wait();
        // Em outra thread, com prazo: se a leitura esperasse o escritor, o
        // teste falha em vez de travar.
        let (tx, rx) = mpsc::channel();
        let reader = Arc::clone(&cell);
        thread::spawn(move || tx.send(reader.read(|n| *n)));
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(10)),
            Ok(0),
            "leitor bloqueou"
        );
        release.wait();
        writer.join().unwrap();
        assert_eq!(cell.read(|n| *n), 1);
    }

    #[test]
    fn store_waits_for_an_update_in_progress_instead_of_being_overwritten() {
        let cell = Arc::new(SnapshotCell::new(0u64));
        let (inside, release) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
        let updater = {
            let (cell, inside, release) =
                (Arc::clone(&cell), Arc::clone(&inside), Arc::clone(&release));
            thread::spawn(move || {
                cell.update(|n| {
                    inside.wait();
                    release.wait();
                    n + 1
                });
            })
        };
        inside.wait();
        let storer = {
            let cell = Arc::clone(&cell);
            thread::spawn(move || cell.store(100))
        };
        // Tempo para o `store` tentar: se não esperasse a fila, já teria
        // publicado (e o `update` o sobrescreveria ao terminar).
        thread::sleep(Duration::from_millis(50));
        assert_eq!(cell.read(|n| *n), 0, "store publicou no meio do update");
        release.wait();
        updater.join().unwrap();
        storer.join().unwrap();
        assert_eq!(cell.read(|n| *n), 100, "o store veio depois e prevaleceu");
    }

    #[test]
    fn a_panicking_update_publishes_nothing_and_does_not_jam_writers() {
        let cell = SnapshotCell::new(1u64);
        let update =
            std::panic::AssertUnwindSafe(|| cell.update(|_| panic!("atualização quebrada")));
        let panicked = std::panic::catch_unwind(update);
        assert!(panicked.is_err());
        cell.update(|n| n * 10);
        assert_eq!(cell.read(|n| *n), 10);
    }

    /// Medição (`cargo test --release -- --ignored --nocapture reads_per_second`):
    /// leituras por segundo com 4 leitores e um escritor trocando o estado sem
    /// parar, contra `RwLock<Arc<T>>`.
    #[test]
    #[ignore = "medição; rode à mão"]
    fn reads_per_second_under_swaps() {
        /// (leituras/s, trocas/s) em 500 ms.
        fn measure(read: &(dyn Fn() -> u64 + Sync), write: &(dyn Fn() + Sync)) -> (f64, f64) {
            let (reads, writes, stop) = (
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicBool::new(false),
            );
            let window = Duration::from_millis(500);
            thread::scope(|s| {
                for _ in 0..4 {
                    s.spawn(|| {
                        let mut n = 0;
                        while !stop.load(Ordering::Relaxed) {
                            std::hint::black_box(read());
                            n += 1;
                        }
                        reads.fetch_add(n, Ordering::Relaxed);
                    });
                }
                s.spawn(|| {
                    let end = Instant::now() + window;
                    while Instant::now() < end {
                        write();
                        writes.fetch_add(1, Ordering::Relaxed);
                    }
                    stop.store(true, Ordering::Relaxed);
                });
            });
            #[allow(clippy::cast_precision_loss)]
            let per_second = |n: AtomicUsize| n.into_inner() as f64 / window.as_secs_f64();
            (per_second(reads), per_second(writes))
        }
        let cell = SnapshotCell::new(vec![1u64; 1024]);
        let lock = RwLock::new(Arc::new(vec![1u64; 1024]));
        let swap = measure(&|| cell.read(|v| v[7]), &|| cell.store(vec![2; 1024]));
        let rw = measure(&|| lock.read().unwrap()[7], &|| {
            *lock.write().unwrap() = Arc::new(vec![2; 1024]);
        });
        for (name, (reads, writes)) in [("SnapshotCell", swap), ("RwLock<Arc>", rw)] {
            println!(
                "{name}: {:.1} M leituras/s, {:.0} mil trocas/s",
                reads / 1e6,
                writes / 1e3
            );
        }
    }
}
