//! Persistência do cache em disco: um [`CacheStore`] sobre o sled, banco
//! embutido com log estruturado, operações atômicas e recuperação depois de
//! queda.
//!
//! Formato: a árvore `entries` guarda chave do cache (UTF-8) → registro rkyv
//! do [`crate::archive`] (cabeçalho `SCRK` v1). A árvore padrão guarda a marca
//! do formato (magia e versão), conferida ao abrir: banco de outra versão é
//! recusado em vez de lido errado. A v1 (PR 35/100) gravava o bincode do crate
//! de domínio; um banco v1 sobe para v2 ao abrir, e as entradas antigas seguem
//! legíveis (o registro rkyv se reconhece pelo cabeçalho).
//!
//! Durabilidade ([`Durability`]): `sync` faz o fsync antes de cada escrita
//! voltar; `async` volta da memória e o sled grava em lote a cada
//! `flush_interval_ms` — numa queda, perde-se no máximo esse intervalo. Ao ser
//! descartado, o sled grava o que estiver pendente.
//!
//! Escritores concorrentes usam [`SledStore::put`] e [`SledStore::remove`],
//! por referência compartilhada. No modo `sync`, quem chega enquanto um flush
//! grava espera o próximo, que serve a todos ([`GroupCommit`]): um fsync por
//! lote, e cada escritor só volta com a própria escrita em disco.

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicIsize, Ordering::Relaxed};

use semantic_cache_rs::codec::DEFAULT_MAX_BINARY_SIZE;
use semantic_cache_rs::{deserialize_entry, CacheEntry, CacheError, CacheStore};

use crate::archive;
use crate::config::{Durability, PersistenceConfig};
use crate::group_commit::GroupCommit;

/// Chave da marca de formato, na árvore padrão.
const FORMAT_KEY: &str = "semantic-cache/format";
/// Magia e versão (`u16` big-endian). Mudou o formato das entradas: suba a
/// versão e migre ao abrir. v2: entradas em rkyv (PR 59/100).
const FORMAT: [u8; 6] = *b"SCRS\x00\x02";
/// Bincode; o banco sobe para v2 ao abrir, e daí o binário antigo o recusa
/// em vez de ler errado o que for gravado depois.
const FORMAT_V1: [u8; 6] = *b"SCRS\x00\x01";
const ENTRIES: &str = "entries";
/// Cache de páginas do sled. O padrão dele é 1 GiB, que somaria ao orçamento
/// de memória do cache; aqui o banco é um log escrito o tempo todo e lido
/// inteiro só ao subir.
/// ponytail: teto fixo; vira configuração se a leitura a quente pesar.
const PAGE_CACHE_BYTES: u64 = 64 << 20;

/// Entradas do cache gravadas num banco sled.
pub struct SledStore {
    db: sled::Db,
    tree: sled::Tree,
    durability: Durability,
    /// Contado nas escritas: o `len` do sled percorre a árvore inteira. Com
    /// sinal porque, com `put` e `remove` concorrentes na mesma chave, a
    /// remoção pode descontar antes de a inserção contar.
    len: AtomicIsize,
    /// Os flushes do modo `sync`.
    commits: GroupCommit<sled::Error>,
}

impl SledStore {
    /// Abre o banco em `path` (normalmente `config.path`), ou cria se não
    /// existe, com a durabilidade e o intervalo de flush de `config`.
    ///
    /// # Errors
    ///
    /// [`CacheError::Io`] se o sled não abre (diretório inacessível, banco já
    /// aberto por outro processo); [`CacheError::CorruptedData`] se o banco
    /// tem outro formato, ou entradas sem a marca.
    pub fn open(path: &Path, config: &PersistenceConfig) -> Result<Self, CacheError> {
        let flush_every = match config.durability {
            Durability::Sync => None, // cada escrita já grava
            Durability::Async => Some(config.flush_interval_ms),
        };
        let db = sled::Config::new()
            .path(path)
            .cache_capacity(PAGE_CACHE_BYTES)
            .flush_every_ms(flush_every)
            .open()
            .map_err(io)?;
        let tree = db.open_tree(ENTRIES).map_err(io)?;
        match db.get(FORMAT_KEY).map_err(io)? {
            Some(found) if *found == FORMAT => {}
            Some(found) if *found == FORMAT_V1 => mark(&db)?,
            None if tree.is_empty() => mark(&db)?,
            found => {
                return Err(CacheError::corrupted(format!(
                    "formato em disco desconhecido em {}: {:?} (esperado {FORMAT:?})",
                    path.display(),
                    found.as_deref()
                )))
            }
        }
        let len = AtomicIsize::new(isize::try_from(tree.len()).unwrap_or(isize::MAX));
        Ok(Self {
            db,
            tree,
            durability: config.durability,
            len,
            commits: GroupCommit::default(),
        })
    }

    /// Grava em disco o que estiver pendente (fsync) e devolve quantos bytes
    /// gravou; no modo assíncrono, é o que garante o desligamento limpo.
    ///
    /// # Errors
    ///
    /// [`CacheError::Io`] se a gravação falha.
    pub fn flush(&self) -> Result<usize, CacheError> {
        self.db.flush().map_err(io)
    }

    /// Todas as entradas, em ordem de chave: o que o servidor carrega na
    /// memória ao subir. Entrada corrompida vem como erro, sem parar as
    /// outras.
    pub fn entries(&self) -> impl Iterator<Item = Result<(String, CacheEntry), CacheError>> + '_ {
        self.tree.iter().map(|item| {
            let (key, value) = item.map_err(io)?;
            let key = String::from_utf8(key.to_vec())
                .map_err(|_| CacheError::corrupted("chave gravada não é UTF-8"))?;
            let entry = decode(&key, &value)?;
            Ok((key, entry))
        })
    }

    /// [`CacheStore::set`] por referência compartilhada: várias threads gravam
    /// ao mesmo tempo e, no modo `sync`, dividem o flush.
    ///
    /// # Errors
    ///
    /// Entrada que não serializa ou maior que o limite de leitura do codec;
    /// [`CacheError::Io`] do sled. No modo `sync`, um erro no flush vale para o
    /// lote inteiro e para as escritas seguintes (ver [`GroupCommit::commit`]),
    /// e a entrada fica na árvore mas pode não sobreviver a uma queda.
    pub fn put(&self, key: &str, entry: &CacheEntry) -> Result<(), CacheError> {
        let bytes = archive::encode(entry)?;
        // O teto de um registro: o do codec do crate de domínio, que recusa ler
        // mais que isso (e o que um banco v1 aceitava).
        if bytes.len() > DEFAULT_MAX_BINARY_SIZE {
            return Err(CacheError::Serialization(format!(
                "entrada '{key}' com {} bytes passa do limite de leitura ({DEFAULT_MAX_BINARY_SIZE})",
                bytes.len()
            )));
        }
        if self
            .tree
            .insert(key.as_bytes(), bytes)
            .map_err(io)?
            .is_none()
        {
            self.len.fetch_add(1, Relaxed);
        }
        self.sync()
    }

    /// [`CacheStore::evict`] por referência compartilhada, como
    /// [`SledStore::put`].
    ///
    /// # Errors
    ///
    /// [`CacheError::Io`] do sled, inclusive o do flush no modo `sync`.
    pub fn remove(&self, key: &str) -> Result<bool, CacheError> {
        let removed = self.tree.remove(key).map_err(io)?.is_some();
        if removed {
            self.len.fetch_sub(1, Relaxed);
        }
        self.sync()?;
        Ok(removed)
    }

    /// O fsync do modo `sync`, depois de cada escrita: um por lote de
    /// escritores concorrentes.
    fn sync(&self) -> Result<(), CacheError> {
        if self.durability == Durability::Sync {
            self.commits
                .commit(|| self.db.flush().map(drop))
                .map_err(io)?;
        }
        Ok(())
    }
}

impl CacheStore for SledStore {
    fn get(&self, key: &str) -> Result<Option<CacheEntry>, CacheError> {
        let value = self.tree.get(key).map_err(io)?;
        value.map(|bytes| decode(key, &bytes)).transpose()
    }

    /// Igual a [`SledStore::put`].
    fn set(&mut self, key: String, entry: CacheEntry) -> Result<(), CacheError> {
        self.put(&key, &entry)
    }

    fn evict(&mut self, key: &str) -> Result<bool, CacheError> {
        self.remove(key)
    }

    fn len(&self) -> usize {
        usize::try_from(self.len.load(Relaxed)).unwrap_or(0)
    }

    fn clear(&mut self) -> Result<(), CacheError> {
        self.tree.clear().map_err(io)?;
        *self.len.get_mut() = 0;
        self.sync()
    }
}

/// Sem o `Debug` do sled, que percorre e imprime a árvore inteira.
impl fmt::Debug for SledStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SledStore")
            .field("durability", &self.durability)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// Bytes que não decodificam são corrupção do disco, com a chave para achar.
/// Com a magia `SCRK` é rkyv; sem ela, bincode de banco v1 (um bincode não
/// começa por `SCRK`: seria um `id` de 5,5 GB).
fn decode(key: &str, bytes: &[u8]) -> Result<CacheEntry, CacheError> {
    let entry = if bytes.starts_with(&archive::MAGIC) {
        archive::decode(bytes)
    } else {
        deserialize_entry(bytes)
    };
    entry.map_err(|e| CacheError::corrupted(format!("entrada '{key}': {e}")))
}

/// Grava a marca do formato atual (banco novo ou v1 que sobe).
fn mark(db: &sled::Db) -> Result<(), CacheError> {
    db.insert(FORMAT_KEY, &FORMAT).map_err(io)?;
    db.flush().map_err(io)?;
    Ok(())
}

fn io(err: sled::Error) -> CacheError {
    match err {
        sled::Error::Corruption { .. } => CacheError::corrupted(err.to_string()),
        other => CacheError::Io(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Barrier, Mutex};
    use std::thread;
    use std::time::Instant;

    use semantic_cache_rs::{serialize_entry, Vector};

    use super::*;

    /// Diretório temporário, apagado no fim do teste.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("semcache-{name}-{pid}"));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Intervalo longo: no modo `async`, só o flush manual (ou o descarte)
    /// grava, e o teste vê o que ficou pendente.
    fn config(durability: Durability) -> PersistenceConfig {
        PersistenceConfig {
            path: None,
            durability,
            flush_interval_ms: 3_600_000,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn entry(id: &str, dim: usize) -> CacheEntry {
        CacheEntry {
            id: id.to_owned(),
            prompt: format!("pergunta {id} — com acentuação"),
            response: "resposta".to_owned(),
            embedding: Vector {
                data: (0..dim).map(|i| i as f32 / 7.0).collect(),
            },
            hit_count: 3,
            created_at: 1_700_000_000,
            ttl_seconds: 3600,
        }
    }

    /// `CacheEntry` não tem `PartialEq`: compara pelo bincode.
    fn same(a: &CacheEntry, b: &CacheEntry) -> bool {
        serialize_entry(a).unwrap() == serialize_entry(b).unwrap()
    }

    #[test]
    fn reopen_keeps_what_was_written_in_both_modes() {
        for durability in [Durability::Sync, Durability::Async] {
            let dir = TempDir::new(&format!("reopen-{durability:?}"));
            let (a, b) = (entry("a", 1536), entry("b", 3));
            {
                let mut store = SledStore::open(&dir.0, &config(durability)).unwrap();
                store.set("a".into(), a.clone()).unwrap();
                for _ in 0..2 {
                    store.set("b".into(), b.clone()).unwrap(); // regravar não conta dobrado
                }
                store.set("c".into(), entry("c", 1)).unwrap();
                assert_eq!((store.evict("c"), store.evict("c")), (Ok(true), Ok(false)));
                assert_eq!(store.len(), 2);
            } // descartado: o sled grava o pendente
            let store = SledStore::open(&dir.0, &config(durability)).unwrap();
            assert_eq!(store.len(), 2, "{durability:?}");
            assert!(same(&store.get("a").unwrap().unwrap(), &a));
            assert!(store.get("c").unwrap().is_none());
            let keys: Vec<String> = store.entries().map(|e| e.unwrap().0).collect();
            assert_eq!(keys, ["a", "b"]);
        }
    }

    #[test]
    fn sync_flushes_before_returning_and_async_leaves_it_to_the_batch() {
        // `flush` devolve os bytes que ainda estavam pendentes.
        let dir = TempDir::new("durability");
        let mut sync = SledStore::open(&dir.0.join("sync"), &config(Durability::Sync)).unwrap();
        let mut batch = SledStore::open(&dir.0.join("async"), &config(Durability::Async)).unwrap();
        for i in 0..20 {
            sync.set(format!("k{i}"), entry("k", 64)).unwrap();
            assert_eq!(sync.flush().unwrap(), 0, "sync: nada pendente na volta");
            batch.set(format!("k{i}"), entry("k", 64)).unwrap();
            assert!(
                batch.flush().unwrap() > 0,
                "async: a escrita ficou para o lote"
            );
        }
        assert!(sync.evict("k0").unwrap());
        sync.clear().unwrap();
        assert_eq!((sync.len(), sync.flush().unwrap()), (0, 0));
    }

    #[test]
    fn unknown_format_and_corrupted_entries_are_errors_not_garbage() {
        let dir = TempDir::new("format");
        let open = || SledStore::open(&dir.0, &config(Durability::Sync));
        let raw = |f: &dyn Fn(&sled::Db)| f(&sled::open(&dir.0).unwrap());
        drop(open().unwrap());
        raw(&|db| drop(db.insert(FORMAT_KEY, b"SCRS\x00\x03".as_slice()).unwrap()));
        let err = open().unwrap_err();
        assert!(
            matches!(err, CacheError::CorruptedData(ref m) if m.contains("formato")),
            "{err}"
        );
        // Entradas sem marca de formato: dado de origem desconhecida.
        raw(&|db| {
            db.remove(FORMAT_KEY).unwrap();
            let entries = db.open_tree(ENTRIES).unwrap();
            entries.insert("x", b"lixo".as_slice()).unwrap();
            entries
                .insert([0xff], serialize_entry(&entry("y", 2)).unwrap())
                .unwrap();
        });
        assert!(matches!(open(), Err(CacheError::CorruptedData(_))));
        // Marca certa: a entrada corrompida vira erro com a chave, sem panic.
        raw(&|db| drop(db.insert(FORMAT_KEY, FORMAT.as_slice()).unwrap()));
        let store = open().unwrap();
        let err = store.get("x").unwrap_err();
        assert!(
            matches!(err, CacheError::CorruptedData(ref m) if m.contains("'x'")),
            "{err}"
        );
        assert_eq!(
            store.entries().filter(Result::is_err).count(),
            2,
            "valor e chave"
        );
    }

    #[test]
    fn v1_database_upgrades_its_mark_and_keeps_its_bincode_entries() {
        let dir = TempDir::new("v1");
        {
            // Como o PR 35/100 gravava: marca v1 e entrada em bincode.
            let db = sled::open(&dir.0).unwrap();
            db.insert(FORMAT_KEY, FORMAT_V1.as_slice()).unwrap();
            let old = serialize_entry(&entry("old", 4)).unwrap();
            db.open_tree(ENTRIES).unwrap().insert("old", old).unwrap();
        }
        let mut store = SledStore::open(&dir.0, &config(Durability::Sync)).unwrap();
        store.set("new".into(), entry("new", 4)).unwrap();
        assert!(store
            .tree
            .get("new")
            .unwrap()
            .unwrap()
            .starts_with(&archive::MAGIC));
        for key in ["old", "new"] {
            assert!(
                same(&store.get(key).unwrap().unwrap(), &entry(key, 4)),
                "{key}"
            );
        }
        assert_eq!(store.entries().flatten().count(), 2);
        drop(store);
        let mark = sled::open(&dir.0)
            .unwrap()
            .get(FORMAT_KEY)
            .unwrap()
            .unwrap();
        assert_eq!(*mark, FORMAT);
    }

    #[test]
    fn entry_too_big_to_read_back_is_refused_on_write() {
        let dir = TempDir::new("big");
        let mut store = SledStore::open(&dir.0, &config(Durability::Sync)).unwrap();
        let mut huge = entry("h", 1);
        huge.response = "x".repeat(DEFAULT_MAX_BINARY_SIZE);
        let err = store.set("h".into(), huge).unwrap_err();
        assert!(
            matches!(err, CacheError::Serialization(ref m) if m.contains("'h'")),
            "{err}"
        );
        assert!(store.is_empty() && store.get("h").unwrap().is_none());
        assert_eq!(
            format!("{store:?}"),
            "SledStore { durability: Sync, len: 0, .. }"
        );
    }

    /// `threads` escritores ao mesmo tempo, cada um com `write(thread, i)`
    /// para `i` em `0..per`. Devolve escritas por segundo.
    #[allow(clippy::cast_precision_loss)]
    fn writers(threads: usize, per: usize, write: &(dyn Fn(usize, usize) + Sync)) -> f64 {
        let barrier = Barrier::new(threads);
        let start = Instant::now();
        thread::scope(|s| {
            for t in 0..threads {
                let barrier = &barrier;
                s.spawn(move || {
                    barrier.wait();
                    (0..per).for_each(|i| write(t, i));
                });
            }
        });
        (threads * per) as f64 / start.elapsed().as_secs_f64()
    }

    #[test]
    fn concurrent_sync_writers_leave_nothing_pending_and_survive_reopen() {
        let dir = TempDir::new("group-commit");
        let store = SledStore::open(&dir.0, &config(Durability::Sync)).unwrap();
        let value = entry("e", 16);
        writers(16, 10, &|t, i| {
            store.put(&format!("{t}-{i}"), &value).unwrap();
            if i == 9 {
                assert_eq!(store.remove(&format!("{t}-0")), Ok(true));
            }
        });
        // Cada `Ok` veio depois do flush que cobre a escrita: nada pendente.
        assert_eq!((store.len(), store.flush().unwrap()), (16 * 9, 0));
        drop(store);
        let store = SledStore::open(&dir.0, &config(Durability::Sync)).unwrap();
        assert_eq!(store.len(), 16 * 9, "reaberto, com tudo o que voltou Ok");
        assert!(store.get("7-0").unwrap().is_none() && store.get("7-9").unwrap().is_some());
    }

    /// Medição, fora do CI: escritas `sync` de 6 KB com o store compartilhado
    /// e, como antes do group commit, atrás de um `Mutex`.
    /// `cargo test --release --lib -- --ignored --nocapture sync_writers_throughput`
    #[test]
    #[ignore = "medição"]
    #[allow(clippy::cast_precision_loss)]
    fn sync_writers_throughput() {
        let value = entry("e", 1536);
        for threads in [1, 8, 64] {
            let (dir, per) = (TempDir::new(&format!("rate-{threads}")), 6400 / threads);
            let open =
                |name| SledStore::open(&dir.0.join(name), &config(Durability::Sync)).unwrap();
            let store = open("shared");
            let shared = writers(threads, per, &|t, i| {
                store.put(&format!("{t}-{i}"), &value).unwrap();
            });
            let batch = (threads * per) as f64 / store.commits.flushes() as f64;
            let locked = Mutex::new(open("mutex"));
            let mutex = writers(threads, per, &|t, i| {
                let mut store = locked.lock().unwrap();
                store.set(format!("{t}-{i}"), value.clone()).unwrap();
            });
            println!("{threads:2} threads: Mutex {mutex:6.0}/s, group commit {shared:6.0}/s ({batch:.1} por flush)");
        }
    }
}
