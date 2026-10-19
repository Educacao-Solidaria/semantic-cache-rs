//! Orçamento de memória do cache: teto estrito em bytes, reservado antes de
//! alocar.
//!
//! [`MemoryBudget::try_reserve`] é um compare-and-swap no contador, feito
//! antes de alocar: duas inserções concorrentes nunca passam juntas do teto.
//! A [`Reservation`] acompanha a entrada e devolve os bytes quando é
//! descartada (RAII), então a contagem não deriva. A partir da marca d'água
//! alta, a reserva chama o [`Evictor`] para liberar até a marca baixa (despejo
//! preventivo); quem sai, LRU ou LFU, é decisão do store.
//!
//! ponytail: conta o que o cache declara, não o heap real; o teto do processo
//! inteiro continua com o cgroup. Upgrade: somar à pressão a diferença entre
//! `alloc::stats().allocated` e o declarado.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use crate::config::{ConfigError, MemoryConfig};

/// Quem libera memória a partir da marca d'água alta: o despejo LRU/LFU do
/// store.
pub trait Evictor: Send + Sync {
    /// Libera pelo menos `bytes`, se puder, descartando as [`Reservation`]s
    /// das entradas que saem. Roda na thread cuja reserva disparou o despejo,
    /// uma chamada por vez, e não pode esperar por trava que quem reserva
    /// segura (reserve antes de travar o store). Pode só acordar uma tarefa de
    /// fundo e voltar.
    fn evict(&self, bytes: usize);
}

/// Reserva recusada: passaria do teto, mesmo depois do despejo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("orçamento de memória esgotado: pedidos {requested} bytes com {used} de {limit} em uso")]
pub struct BudgetExceeded {
    pub requested: usize,
    pub used: usize,
    pub limit: usize,
}

/// Contador de bytes compartilhado entre as threads; clonar custa um `Arc`.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    used: AtomicUsize,
    limit: usize,
    /// Reserva que termina em `high` ou acima dispara o despejo...
    high: usize,
    /// ...que libera até `low`.
    low: usize,
    evictor: OnceLock<Weak<dyn Evictor>>,
    /// Um despejo por vez: quem chega com outro em curso segue sem esperar.
    evicting: AtomicBool,
}

impl MemoryBudget {
    /// Orçamento vazio com o teto e as marcas d'água de `config`.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Invalid`] se `config` não passa em
    /// [`MemoryConfig::validate`].
    pub fn new(config: &MemoryConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let limit = config.max_bytes;
        // Fração do teto, arredondada para baixo; precisão de `f64` sobra.
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let mark = |fraction: f64| ((limit as f64 * fraction) as usize).min(limit);
        Ok(Self {
            inner: Arc::new(Inner {
                used: AtomicUsize::new(0),
                limit,
                high: mark(config.high_watermark),
                low: mark(config.low_watermark),
                evictor: OnceLock::new(),
                evicting: AtomicBool::new(false),
            }),
        })
    }

    /// Registra quem despeja; `false` se já havia um. Guarda só um `Weak`
    /// porque o store guarda o orçamento: `Arc` nos dois sentidos seria um
    /// ciclo que nunca libera.
    #[must_use]
    pub fn set_evictor<E: Evictor + 'static>(&self, evictor: &Arc<E>) -> bool {
        let weak = Arc::downgrade(evictor);
        self.inner.evictor.set(weak).is_ok()
    }

    /// Reserva `bytes` antes de alocá-los; se não cabem, despeja e tenta mais
    /// uma vez. Terminando na marca alta ou acima, despeja depois de reservar.
    ///
    /// # Errors
    ///
    /// [`BudgetExceeded`] se não couber nem depois do despejo, ou se `bytes`
    /// sozinho passa do teto (aí nem se despeja).
    pub fn try_reserve(&self, bytes: usize) -> Result<Reservation, BudgetExceeded> {
        let inner = &*self.inner;
        let mut added = inner.add(bytes);
        if let Err(used) = added {
            if bytes <= inner.limit {
                // Não cabe: despeja até a marca baixa (já contando `bytes`).
                inner.relieve(used.saturating_add(bytes).saturating_sub(inner.low));
                added = inner.add(bytes);
            }
        }
        let before = added.map_err(|used| BudgetExceeded {
            requested: bytes,
            used,
            limit: inner.limit,
        })?;
        let reservation = Reservation {
            budget: Arc::clone(&self.inner),
            bytes,
        };
        let now = before + bytes; // sem estouro: `add` conferiu
        if now >= inner.high {
            inner.relieve(now.saturating_sub(inner.low));
        }
        Ok(reservation)
    }

    /// Bytes reservados agora.
    #[must_use]
    pub fn used(&self) -> usize {
        self.inner.used.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn limit(&self) -> usize {
        self.inner.limit
    }
}

impl Inner {
    /// Soma `bytes` só se o total couber no teto; devolve o uso anterior (ou o
    /// atual, se recusou). `Relaxed` basta: o contador não publica outro dado,
    /// e as operações num mesmo atômico têm ordem total, que é o que impede
    /// duas reservas de passarem juntas do teto.
    #[allow(deprecated)] // `try_update`, o nome novo, é posterior ao MSRV 1.80
    fn add(&self, bytes: usize) -> Result<usize, usize> {
        self.used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.limit)
            })
    }

    /// Pede ao evictor que libere `bytes`, se há evictor vivo e nenhum
    /// despejo em curso.
    fn relieve(&self, bytes: usize) {
        let evictor = self.evictor.get().filter(|_| bytes > 0);
        let Some(evictor) = evictor.and_then(Weak::upgrade) else {
            return;
        };
        if self.evicting.swap(true, Ordering::Acquire) {
            return;
        }
        // Libera o gatilho mesmo se `evict` entrar em panic.
        let _done = ClearOnDrop(&self.evicting);
        evictor.evict(bytes);
    }
}

struct ClearOnDrop<'a>(&'a AtomicBool);

impl Drop for ClearOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Bytes reservados; voltam ao orçamento quando a reserva é descartada.
/// `mem::forget` nela vaza a contagem — guarde-a junto da entrada.
#[derive(Debug)]
#[must_use = "descartar a reserva devolve os bytes na hora"]
pub struct Reservation {
    budget: Arc<Inner>,
    bytes: usize,
}

impl Reservation {
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Barrier, Mutex};
    use std::thread;

    use super::*;

    /// Teto de 1000 bytes; com as marcas padrão, despeja de 900 até 800.
    fn small_budget() -> MemoryBudget {
        let max_bytes = 1000;
        MemoryBudget::new(&MemoryConfig {
            max_bytes,
            ..MemoryConfig::default()
        })
        .unwrap()
    }

    /// Store de mentira: guarda as reservas em ordem e despeja as mais velhas.
    #[derive(Default)]
    struct Fifo {
        held: Mutex<VecDeque<Reservation>>,
        calls: AtomicUsize,
        busy: AtomicUsize,
        max_busy: AtomicUsize,
    }

    impl Fifo {
        fn insert(&self, budget: &MemoryBudget, bytes: usize) -> Result<(), BudgetExceeded> {
            // Reserva antes de travar, como o contrato de `Evictor` pede.
            let reservation = budget.try_reserve(bytes)?;
            assert!(budget.used() <= budget.limit());
            self.held.lock().unwrap().push_back(reservation);
            Ok(())
        }
    }

    impl Evictor for Fifo {
        fn evict(&self, bytes: usize) {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let busy = self.busy.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_busy.fetch_max(busy, Ordering::SeqCst);
            let mut freed = 0;
            while freed < bytes {
                let Some(oldest) = self.held.lock().unwrap().pop_front() else {
                    break;
                };
                freed += oldest.bytes();
            }
            self.busy.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn with_fifo(budget: &MemoryBudget) -> Arc<Fifo> {
        let fifo = Arc::new(Fifo::default());
        assert!(budget.set_evictor(&fifo));
        fifo
    }

    #[test]
    fn reserve_is_strict_and_drop_gives_back() {
        let budget = small_budget();
        let a = budget.try_reserve(600).unwrap();
        let refused = budget.try_reserve(500).unwrap_err();
        assert_eq!(
            (refused.requested, refused.used, refused.limit),
            (500, 600, 1000)
        );
        let b = budget.try_reserve(400).unwrap(); // exatamente o teto
        let zero = budget.try_reserve(0).unwrap();
        assert_eq!((budget.used(), zero.bytes()), (1000, 0));
        for huge in [1001, usize::MAX] {
            assert!(budget.try_reserve(huge).is_err(), "{huge}");
        }
        drop((a, b, zero));
        assert_eq!(budget.used(), 0);
        assert_eq!(
            refused.to_string(),
            "orçamento de memória esgotado: pedidos 500 bytes com 600 de 1000 em uso"
        );
    }

    #[test]
    fn only_one_of_racing_reservations_fits() {
        let budget = small_budget();
        let (start, hold) = (Barrier::new(8), Barrier::new(8));
        for round in 0..100 {
            let won = AtomicUsize::new(0);
            thread::scope(|s| {
                for _ in 0..8 {
                    s.spawn(|| {
                        start.wait();
                        let r = budget.try_reserve(600);
                        won.fetch_add(usize::from(r.is_ok()), Ordering::SeqCst);
                        hold.wait(); // segura até todas tentarem
                    });
                }
            });
            assert_eq!(won.into_inner(), 1, "rodada {round}");
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn crossing_the_high_watermark_evicts_down_to_the_low_one() {
        let budget = small_budget();
        let fifo = with_fifo(&budget);
        for _ in 0..8 {
            fifo.insert(&budget, 100).unwrap();
        }
        assert_eq!((budget.used(), fifo.calls.load(Ordering::SeqCst)), (800, 0));
        // Cada entrada nova leva a 900, e o despejo tira a mais velha: 800.
        for _ in 0..50 {
            fifo.insert(&budget, 100).unwrap();
            assert_eq!(budget.used(), 800);
        }
        assert_eq!(fifo.calls.load(Ordering::SeqCst), 50);
    }

    #[test]
    fn a_full_budget_evicts_then_retries_once() {
        let budget = small_budget();
        let mut full: VecDeque<Reservation> =
            (0..10).map(|_| budget.try_reserve(100).unwrap()).collect();
        let fifo = with_fifo(&budget);
        fifo.held.lock().unwrap().append(&mut full);
        // 1000 + 100 passa do teto: despeja 1100 - 800 = 300 e reserva.
        fifo.insert(&budget, 100).unwrap();
        assert_eq!((budget.used(), fifo.calls.load(Ordering::SeqCst)), (800, 1));
        // Nada para despejar: uma tentativa de despejo e a recusa.
        let empty = small_budget();
        let kept = empty.try_reserve(1000).unwrap();
        let lone = with_fifo(&empty);
        assert!(empty.try_reserve(1).is_err());
        assert_eq!(lone.calls.load(Ordering::SeqCst), 1);
        // Maior que o teto sozinho: recusa sem despejar.
        assert!(empty.try_reserve(1001).is_err());
        assert_eq!(lone.calls.load(Ordering::SeqCst), 1);
        drop(kept);
    }

    #[test]
    fn concurrent_inserts_evict_one_at_a_time_and_never_drift() {
        let budget = small_budget();
        let fifo = with_fifo(&budget);
        thread::scope(|s| {
            for t in 0..8 {
                let (budget, fifo) = (&budget, &fifo);
                s.spawn(move || {
                    for i in 0..5_000 {
                        let _ = fifo.insert(budget, 1 + (i * 7 + t) % 97);
                    }
                });
            }
        });
        assert!(fifo.calls.load(Ordering::SeqCst) > 0);
        assert_eq!(
            fifo.max_busy.load(Ordering::SeqCst),
            1,
            "um despejo por vez"
        );
        drop(fifo); // descarta as reservas que sobraram
        assert_eq!(budget.used(), 0, "toda reserva voltou");
    }

    #[test]
    fn panicking_or_dropped_evictor_does_not_jam_the_trigger() {
        struct Panics(AtomicUsize);
        impl Evictor for Panics {
            fn evict(&self, _: usize) {
                self.0.fetch_add(1, Ordering::SeqCst);
                panic!("evictor quebrado");
            }
        }
        let budget = small_budget();
        let panics = Arc::new(Panics(AtomicUsize::new(0)));
        assert!(budget.set_evictor(&panics));
        for _ in 0..2 {
            let reserve = std::panic::AssertUnwindSafe(|| budget.try_reserve(950));
            assert!(std::panic::catch_unwind(reserve).is_err());
            assert_eq!(budget.used(), 0, "a reserva do panic voltou");
        }
        assert_eq!(panics.0.load(Ordering::SeqCst), 2, "o gatilho não travou");
        assert!(!budget.set_evictor(&panics), "só um evictor");
        // Store já descartado: o `Weak` não sobe e a reserva segue normal.
        drop(panics);
        assert_eq!(budget.try_reserve(950).unwrap().bytes(), 950);
    }
}
