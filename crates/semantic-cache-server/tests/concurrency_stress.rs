//! Estresse de escrita: 64 threads inserem e removem ao mesmo tempo num cache
//! montado com as peças do servidor, na ordem do contrato do orçamento
//! (reservar → gravar → publicar). Cada inserção reserva no [`MemoryBudget`]
//! (teto pequeno: o despejo roda o tempo todo), grava no [`SledStore`] `sync`
//! (group commit) e publica na [`SnapshotCell`]; o despejo e a remoção pelo
//! dono disputam as mesmas entradas.
//!
//! Um deadlock falha o teste no prazo, em vez de travar o CI. No fim, disco,
//! índice e entradas vivas têm as mesmas chaves, as somas fecham, nada ficou
//! pendente no disco, e o orçamento volta a zero quando tudo é solto.
//!
//! Carga: `SEMCACHE_STRESS_OPS` operações por thread (padrão 100, rápido no
//! CI); o prazo cresce junto.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use semantic_cache_rs::{CacheEntry, CacheStore, Vector};
use semantic_cache_server::config::{Durability, MemoryConfig, PersistenceConfig};
use semantic_cache_server::memory::{Evictor, MemoryBudget, Reservation};
use semantic_cache_server::persistence::SledStore;
use semantic_cache_server::snapshot::SnapshotCell;

#[path = "support/sled_reopen.rs"]
mod sled_reopen;

const WRITERS: usize = 64;
const READERS: usize = 4;
const DEFAULT_OPS: usize = 100;
/// Prazo da carga padrão, que levou 1 s aqui e 10,7 s no runner do GitHub
/// (o despejo paga um sync por entrada): ~11× de folga.
const DEADLINE: Duration = Duration::from_secs(120);

/// O estado publicado: chave → bytes reservados, e os contadores mudados na
/// mesma troca.
#[derive(Clone, Default)]
struct Index {
    entries: HashMap<String, usize>,
    bytes: usize,
    inserted: usize,
    removed: usize,
}

struct Cache {
    budget: MemoryBudget,
    store: SledStore,
    index: SnapshotCell<Index>,
    /// Entradas vivas, da mais velha à mais nova, com a reserva de cada uma.
    live: Mutex<VecDeque<(String, Reservation)>>,
}

impl Cache {
    /// `false` se o orçamento recusou (despejo em curso noutra thread).
    fn insert(&self, key: String, bytes: usize, entry: &CacheEntry) -> bool {
        let Ok(reservation) = self.budget.try_reserve(bytes) else {
            return false;
        };
        self.store.put(&key, entry).expect("gravar");
        self.index.update(|old| {
            let mut next = old.clone();
            next.entries.insert(key.clone(), bytes);
            next.bytes += bytes;
            next.inserted += 1;
            next
        });
        self.live.lock().unwrap().push_back((key, reservation));
        true
    }

    /// Remoção pelo dono; `false` se o despejo levou a entrada antes.
    fn remove(&self, key: &str) -> bool {
        let taken = {
            let mut live = self.live.lock().unwrap();
            let at = live.iter().position(|(k, _)| k == key);
            at.and_then(|at| live.remove(at))
        };
        taken.is_some_and(|(key, _reservation)| {
            self.forget(&key);
            true
        })
    }

    /// Tira do disco e do índice uma entrada que já saiu de `live`: quem a
    /// tirou de lá, despejo ou dono, é o único a removê-la.
    fn forget(&self, key: &str) {
        assert!(self.store.remove(key).expect("remover"), "{key} no disco");
        self.index.update(|old| {
            let mut next = old.clone();
            next.bytes -= next.entries.remove(key).expect("no índice");
            next.removed += 1;
            next
        });
    }
}

impl Evictor for Cache {
    /// FIFO; a trava de `live` só para tirar a entrada, não durante a remoção.
    fn evict(&self, bytes: usize) {
        let mut freed = 0;
        while freed < bytes {
            let Some((key, reservation)) = self.live.lock().unwrap().pop_front() else {
                break;
            };
            self.forget(&key);
            freed += reservation.bytes();
        }
    }
}

/// Roda `body` noutra thread e falha se ela não terminar em `deadline`: um
/// deadlock vira teste vermelho, não um CI parado. Pânico em `body` volta
/// com a mensagem original.
fn within<T: Send + 'static>(deadline: Duration, body: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, finished) = mpsc::channel();
    let worker = thread::spawn(move || drop(done.send(body())));
    match finished.recv_timeout(deadline) {
        Ok(value) => value,
        Err(RecvTimeoutError::Disconnected) => match worker.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(()) => unreachable!("terminou sem resultado"),
        },
        Err(RecvTimeoutError::Timeout) => panic!("não terminou em {deadline:?}: deadlock?"),
    }
}

/// Conta a thread como terminada também quando ela entra em pânico: os
/// leitores esperam por todas.
struct Finished<'a>(&'a AtomicUsize);

impl Drop for Finished<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, SeqCst);
    }
}

/// O que cada escritor fez: inserções, recusas (tentadas de novo) e remoções
/// dele.
#[derive(Default)]
struct Tally {
    inserted: usize,
    refused: usize,
    removed: usize,
}

/// Diretório temporário, apagado no fim.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `WRITERS` escritores com `ops` operações cada (a cada 4ª, o dono remove a
/// própria entrada mais velha) e `READERS` leitores conferindo cada retrato
/// enquanto eles escrevem. Devolve a soma do que os escritores fizeram e
/// quantas leituras houve.
fn hammer(cache: &Cache, ops: usize) -> (Tally, usize) {
    let entry = CacheEntry {
        id: "e".into(),
        prompt: "pergunta".into(),
        response: "resposta".into(),
        embedding: Vector {
            data: vec![0.5; 64],
        },
        hit_count: 0,
        created_at: 0,
        ttl_seconds: 0,
    };
    let start = Barrier::new(WRITERS + READERS);
    let (finished, reads) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let tally = thread::scope(|s| {
        for _ in 0..READERS {
            s.spawn(|| {
                start.wait();
                while finished.load(SeqCst) < WRITERS {
                    // Retrato inteiro: o mapa e os contadores da mesma troca.
                    cache.index.read(|i| {
                        assert_eq!(i.entries.len(), i.inserted - i.removed);
                        assert_eq!(i.bytes, i.entries.values().sum::<usize>());
                    });
                    let budget = &cache.budget;
                    assert!(budget.used() <= budget.limit(), "passou do teto");
                    reads.fetch_add(1, SeqCst);
                    thread::yield_now();
                }
            });
        }
        let writers: Vec<_> = (0..WRITERS)
            .map(|t| {
                let (start, finished, entry) = (&start, &finished, &entry);
                s.spawn(move || {
                    let _done = Finished(finished);
                    let (mut tally, mut mine) = (Tally::default(), VecDeque::new());
                    start.wait();
                    for i in 0..ops {
                        if i % 4 == 3 {
                            let oldest: String = mine.pop_front().expect("3 antes");
                            tally.removed += usize::from(cache.remove(&oldest));
                            continue;
                        }
                        let key = format!("{t}-{i}");
                        // Recusa: orçamento cheio com um despejo em curso
                        // noutra thread. Tenta de novo, como o servidor faria,
                        // dormindo para não roubar CPU de quem despeja.
                        while !cache.insert(key.clone(), 512 + (i % 7) * 128, entry) {
                            tally.refused += 1;
                            thread::sleep(Duration::from_micros(50));
                        }
                        tally.inserted += 1;
                        mine.push_back(key);
                    }
                    tally
                })
            })
            .collect();
        writers.into_iter().fold(Tally::default(), |sum, w| {
            let t = w.join().unwrap();
            Tally {
                inserted: sum.inserted + t.inserted,
                refused: sum.refused + t.refused,
                removed: sum.removed + t.removed,
            }
        })
    });
    (tally, reads.into_inner())
}

#[test]
fn writers_competing_for_every_write_path_keep_the_cache_consistent() {
    let ops = std::env::var("SEMCACHE_STRESS_OPS").map_or(DEFAULT_OPS, |v| {
        v.parse().expect("SEMCACHE_STRESS_OPS: inteiro")
    });
    assert!(
        ops >= 8,
        "SEMCACHE_STRESS_OPS >= 8: com menos, o despejo nem começa"
    );
    let dir = TempDir(std::env::temp_dir().join(format!("semcache-stress-{}", std::process::id())));
    let persistence = PersistenceConfig {
        durability: Durability::Sync,
        ..PersistenceConfig::default()
    };
    // ~256 entradas de 512 a 1280 bytes: despeja de 90% até 80%.
    let budget = MemoryBudget::new(&MemoryConfig {
        max_bytes: 256 * 1024,
        ..MemoryConfig::default()
    })
    .unwrap();
    let cache = Arc::new(Cache {
        budget: budget.clone(),
        store: SledStore::open(&dir.0, &persistence).unwrap(),
        index: SnapshotCell::new(Index::default()),
        live: Mutex::new(VecDeque::new()),
    });
    assert!(budget.set_evictor(&cache));

    let shared = Arc::clone(&cache);
    let started = Instant::now();
    let deadline = DEADLINE * u32::try_from(ops.div_ceil(DEFAULT_OPS)).unwrap();
    let (tally, reads) = within(deadline, move || hammer(&shared, ops));
    let elapsed = started.elapsed();
    // Antes de qualquer leitura: ler a árvore faz o sled consolidar páginas,
    // o que é escrita nova no log.
    assert_eq!(
        cache.store.flush().unwrap(),
        0,
        "todo Ok veio depois do disco"
    );

    let index = cache.index.load();
    let disk: BTreeSet<String> = cache.store.entries().map(|e| e.unwrap().0).collect();
    {
        let live = cache.live.lock().unwrap();
        let alive: BTreeSet<String> = live.iter().map(|(k, _)| k.clone()).collect();
        let indexed: BTreeSet<String> = index.entries.keys().cloned().collect();
        assert_eq!(
            (&alive, &indexed),
            (&disk, &disk),
            "mesmas chaves vivas, no índice e no disco"
        );
        assert_eq!(cache.store.len(), live.len());
        let held: usize = live.iter().map(|(_, r)| r.bytes()).sum();
        assert_eq!((index.bytes, budget.used()), (held, held), "a soma fecha");
    }
    assert_eq!(
        tally.inserted,
        WRITERS * (ops - ops / 4),
        "toda inserção entrou"
    );
    assert_eq!(
        (index.inserted, index.inserted - index.removed),
        (tally.inserted, disk.len())
    );
    assert!(index.removed > tally.removed, "o despejo também removeu");
    assert!(reads > 0, "os leitores leram durante as escritas");
    println!(
        "{WRITERS} escritores × {ops} operações em {elapsed:.2?}: {} inserções ({} recusas \
         tentadas de novo), {} remoções pelo dono, {} despejos, {reads} leituras",
        tally.inserted,
        tally.refused,
        tally.removed,
        index.removed - tally.removed
    );

    drop((index, cache)); // o orçamento guarda só um `Weak` do cache
    assert_eq!(budget.used(), 0, "toda reserva voltou");
    let reopened = sled_reopen::reopen(|| SledStore::open(&dir.0, &persistence)).unwrap();
    let after: BTreeSet<String> = reopened.entries().map(|e| e.unwrap().0).collect();
    assert_eq!(after, disk, "reaberto, o disco tem o que estava vivo");
}
