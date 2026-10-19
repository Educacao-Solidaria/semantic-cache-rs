//! Group commit: escritores concorrentes que precisam de durabilidade dividem
//! o mesmo flush, em vez de um fsync cada.
//!
//! Cada escritor grava na memória do banco e chama [`GroupCommit::commit`],
//! que lhe dá um ticket (a ordem da escrita) e só volta quando terminou um
//! flush começado depois do ticket. Se ninguém está gravando, o escritor vira
//! o líder e grava tudo o que já tem ticket: o dele e o de quem chegou antes.
//! Quem chega durante a gravação espera a próxima, que um deles lidera por
//! todos. Um flush por vez, sem espera artificial: o lote seguinte se forma
//! enquanto o atual grava.
//!
//! A fila são os próprios escritores, bloqueados até o disco: cada um tem no
//! máximo uma escrita esperando gravação, então o pendente é limitado pelo
//! número de threads e a espera, por dois flushes (o que está em curso e o
//! dela). Não há buffer próprio para esvaziar no encerramento: o dado está no
//! banco antes do ticket, e o `Ok` só vem depois do flush que o cobre.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

/// Coordena os flushes de escritores concorrentes (ver o módulo).
#[derive(Debug)]
pub struct GroupCommit<E> {
    state: Mutex<State<E>>,
    flushed: Condvar,
}

#[derive(Debug)]
struct State<E> {
    /// Último ticket entregue.
    issued: u64,
    /// Todo ticket até este está em disco.
    durable: u64,
    /// Um líder está gravando.
    flushing: bool,
    /// Primeiro erro de flush. É definitivo: depois de um fsync que falhou não
    /// se sabe o que chegou ao disco, e o sled também trava no primeiro erro
    /// de IO.
    failed: Option<E>,
    flushes: u64,
}

impl<E> Default for GroupCommit<E> {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                issued: 0,
                durable: 0,
                flushing: false,
                failed: None,
                flushes: 0,
            }),
            flushed: Condvar::new(),
        }
    }
}

impl<E> GroupCommit<E> {
    /// Flushes feitos até agora, com erro ou não: escritas ÷ flushes é o
    /// tamanho médio do lote.
    pub fn flushes(&self) -> u64 {
        self.lock().flushes
    }

    fn lock(&self) -> MutexGuard<'_, State<E>> {
        // Com a trava só se fazem contas: envenenada, o estado segue inteiro.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<E: Clone> GroupCommit<E> {
    /// Chamado depois de a escrita estar na memória do banco; volta quando ela
    /// está em disco. `flush` grava tudo o que foi escrito antes de ele ser
    /// chamado, e só roda se este escritor virar o líder. Dentro de `flush`,
    /// chamar `commit` neste mesmo `GroupCommit` trava: esperaria por si mesmo.
    ///
    /// # Errors
    ///
    /// O erro do flush que cobriria esta escrita (o mesmo para o lote
    /// inteiro), ou o de um flush anterior que falhou.
    ///
    /// # Panics
    ///
    /// Repassa o pânico de `flush`. Os outros escritores não ficam presos: um
    /// deles lidera de novo.
    pub fn commit(&self, mut flush: impl FnMut() -> Result<(), E>) -> Result<(), E> {
        let mut state = self.lock();
        state.issued += 1;
        let ticket = state.issued;
        loop {
            if state.durable >= ticket {
                return Ok(());
            }
            if let Some(err) = &state.failed {
                return Err(err.clone());
            }
            if state.flushing {
                state = self
                    .flushed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            // Líder: o flush cobre todo ticket já entregue, o deste inclusive.
            let target = state.issued;
            state.flushing = true;
            drop(state);
            let mut leading = Leading {
                commit: self,
                target,
                result: None,
            };
            leading.result = Some(flush());
            drop(leading);
            state = self.lock();
        }
    }
}

/// Encerra a vez do líder mesmo se `flush` entrar em pânico: sem isto,
/// `flushing` ficaria preso e a fila esperaria para sempre.
struct Leading<'a, E> {
    commit: &'a GroupCommit<E>,
    target: u64,
    /// `None`: o flush não terminou (pânico). Ninguém é liberado, e quem
    /// acordar primeiro lidera de novo.
    result: Option<Result<(), E>>,
}

impl<E> Drop for Leading<'_, E> {
    fn drop(&mut self) {
        let mut state = self.commit.lock();
        state.flushing = false;
        state.flushes += 1;
        match self.result.take() {
            Some(Ok(())) => state.durable = state.durable.max(self.target),
            Some(Err(err)) => {
                state.failed.get_or_insert(err);
            }
            None => {}
        }
        self.commit.flushed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::thread;

    use super::*;

    type Gc = GroupCommit<&'static str>;
    type Outcome = Result<(), &'static str>;
    type Flush<'a> = &'a (dyn Fn() -> Outcome + Sync);

    /// O 1º escritor lidera e segura o flush dele (`first`) até outros 15
    /// estarem na fila, com `rest` como flush. Devolve o resultado do 1º e o
    /// dos outros.
    fn queue_behind_leader(
        gc: &Gc,
        first: Flush,
        rest: Flush,
    ) -> (thread::Result<Outcome>, Vec<Outcome>) {
        let gate = Barrier::new(2);
        thread::scope(|s| {
            let lead = s.spawn(|| {
                gc.commit(|| {
                    gate.wait();
                    first()
                })
            });
            while !gc.lock().flushing {
                thread::yield_now();
            }
            let queue: Vec<_> = (0..15).map(|_| s.spawn(|| gc.commit(rest))).collect();
            // Ticket e espera vêm na mesma posse da trava: 16 entregues = 15 na fila.
            while gc.lock().issued < 16 {
                thread::yield_now();
            }
            gate.wait();
            let lead = lead.join();
            (lead, queue.into_iter().map(|w| w.join().unwrap()).collect())
        })
    }

    #[test]
    fn writers_that_queue_during_a_flush_share_the_next_one() {
        let gc = Gc::default();
        let (lead, queue) = queue_behind_leader(&gc, &|| Ok(()), &|| Ok(()));
        assert_eq!(lead.unwrap(), Ok(()));
        assert_eq!(queue, vec![Ok(()); 15]);
        assert_eq!(gc.flushes(), 2, "16 escritas: o flush do 1º e o do lote");
        // Sozinho, cada escritor lidera o próprio flush.
        for flushes in 3..=4 {
            assert_eq!(gc.commit(|| Ok(())), Ok(()));
            assert_eq!(gc.flushes(), flushes);
        }
    }

    #[test]
    fn a_failed_flush_reaches_every_writer_of_the_batch_and_sticks() {
        let gc = Gc::default();
        let (lead, queue) = queue_behind_leader(&gc, &|| Ok(()), &|| Err("disco cheio"));
        assert_eq!(lead.unwrap(), Ok(()), "o flush do 1º deu certo");
        assert_eq!(queue, vec![Err("disco cheio"); 15]);
        // Definitivo: a escrita seguinte nem tenta gravar.
        assert_eq!(gc.commit(|| unreachable!()), Err("disco cheio"));
        assert_eq!(gc.flushes(), 2);
    }

    #[test]
    fn a_leader_that_panics_does_not_strand_the_queue() {
        let gc = Gc::default();
        let (lead, queue) = queue_behind_leader(&gc, &|| panic!("flush interrompido"), &|| Ok(()));
        assert!(lead.is_err(), "o pânico volta para quem liderava");
        assert_eq!(
            queue,
            vec![Ok(()); 15],
            "um da fila liderou de novo, por todos"
        );
        assert_eq!(gc.flushes(), 2);
    }
}
