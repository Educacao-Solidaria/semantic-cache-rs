//! Formato zero-copy (rkyv) das entradas do cache, para disco e memória: lê
//! os campos direto dos bytes, sem desserializar a entrada inteira (nem
//! alocar o prompt e a resposta para olhar só o embedding). O fio com o
//! gateway continua JSON (`docs/ipc-v1.md`); isto é só do lado Rust.
//!
//! Bytes que vêm do disco são entrada não confiável: todo acesso passa pela
//! validação do `bytecheck` (`unsafe_code = "forbid"` nem deixa usar o
//! `access_unchecked`).

use rkyv::rancor::Error as RkyvError;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};
use semantic_cache_rs::{CacheEntry, CacheError, Vector};

/// Cabeçalho de cada registro: magia, versão e 3 zeros. Os 8 bytes mantêm o
/// corpo com o mesmo alinhamento (até 8) do buffer em que o registro está.
pub const MAGIC: [u8; 4] = *b"SCRK";
/// O layout é o do rkyv 0.8: subir o major do rkyv é subir esta versão.
pub const VERSION: u8 = 1;
const HEADER: [u8; 8] = [MAGIC[0], MAGIC[1], MAGIC[2], MAGIC[3], VERSION, 0, 0, 0];
/// O alinhamento que o rkyv exige do corpo (o dos `u64`); fora dele, copia.
const ALIGN: usize = std::mem::align_of::<ArchivedEntryRecord>();

/// A entrada no formato do rkyv (os campos do `CacheEntry` do crate de
/// domínio, com o embedding direto como `Vec<f32>`).
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct EntryRecord {
    pub id: String,
    pub prompt: String,
    pub response: String,
    pub embedding: Vec<f32>,
    pub hit_count: u64,
    pub created_at: u64,
    pub ttl_seconds: u64,
}

impl From<&CacheEntry> for EntryRecord {
    fn from(e: &CacheEntry) -> Self {
        Self {
            id: e.id.clone(),
            prompt: e.prompt.clone(),
            response: e.response.clone(),
            embedding: e.embedding.data.clone(),
            hit_count: e.hit_count,
            created_at: e.created_at,
            ttl_seconds: e.ttl_seconds,
        }
    }
}

impl From<EntryRecord> for CacheEntry {
    fn from(r: EntryRecord) -> Self {
        Self {
            id: r.id,
            prompt: r.prompt,
            response: r.response,
            embedding: Vector { data: r.embedding },
            hit_count: r.hit_count,
            created_at: r.created_at,
            ttl_seconds: r.ttl_seconds,
        }
    }
}

/// Grava `entry` no formato: cabeçalho e os bytes do rkyv.
///
/// # Errors
/// [`CacheError::Serialization`] se o rkyv falhar.
pub fn encode(entry: &CacheEntry) -> Result<Vec<u8>, CacheError> {
    let record = EntryRecord::from(entry);
    let body = rkyv::to_bytes::<RkyvError>(&record).map_err(|e| serialization(&e))?;
    let mut out = Vec::with_capacity(HEADER.len() + body.len());
    out.extend_from_slice(&HEADER);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Valida o registro e entrega a visão arquivada a `read`, sem desserializar.
/// O corpo só é copiado se não estiver alinhado (o sled não promete
/// alinhamento); a validação roda a cada chamada.
///
/// # Errors
/// [`CacheError::CorruptedData`] com magia, versão ou bytes inválidos.
pub fn with_archived<R>(
    bytes: &[u8],
    read: impl FnOnce(&ArchivedEntryRecord) -> R,
) -> Result<R, CacheError> {
    let body = match bytes.split_at_checked(HEADER.len()) {
        Some((head, body)) if head == HEADER => body,
        _ => return Err(CacheError::corrupted("registro sem o cabeçalho SCRK v1")),
    };
    let copy;
    let body = if body.as_ptr().align_offset(ALIGN) == 0 {
        body
    } else {
        let mut aligned = AlignedVec::<16>::with_capacity(body.len());
        aligned.extend_from_slice(body);
        copy = aligned;
        copy.as_slice()
    };
    let archived = rkyv::access::<ArchivedEntryRecord, RkyvError>(body)
        .map_err(|e| CacheError::corrupted(format!("registro rkyv inválido: {e}")))?;
    Ok(read(archived))
}

/// Desserializa o registro inteiro (com validação).
///
/// # Errors
/// Os de [`with_archived`].
pub fn decode(bytes: &[u8]) -> Result<CacheEntry, CacheError> {
    let record = with_archived(bytes, rkyv::deserialize::<EntryRecord, RkyvError>)?;
    record.map(CacheEntry::from).map_err(|e| serialization(&e))
}

fn serialization(e: &RkyvError) -> CacheError {
    CacheError::Serialization(e.to_string())
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use semantic_cache_rs::{deserialize_entry, serialize_entry};

    use super::*;

    #[allow(clippy::cast_precision_loss)] // só para gerar valores
    fn entry(dim: usize) -> CacheEntry {
        CacheEntry {
            id: "id-1".into(),
            prompt: "Qual a capital do Brasil? ".repeat(8),
            response: "Brasília, no Planalto Central. ".repeat(64),
            embedding: Vector {
                data: (0..dim).map(|i| (i as f32).sin()).collect(),
            },
            hit_count: 3,
            created_at: 1_700_000_000,
            ttl_seconds: 3600,
        }
    }

    fn dot(a: &ArchivedEntryRecord, q: &[f32]) -> f32 {
        a.embedding
            .iter()
            .zip(q)
            .map(|(x, y)| x.to_native() * y)
            .sum()
    }

    #[test]
    fn round_trip_and_zero_copy_read_from_any_alignment() {
        for dim in [0, 1, 1536] {
            let e = entry(dim);
            let bytes = encode(&e).unwrap();
            assert_eq!(
                serialize_entry(&decode(&bytes).unwrap()).unwrap(),
                serialize_entry(&e).unwrap()
            );
            // Deslocado de 0 a 15 bytes no buffer: lê igual (copia o que não
            // está alinhado).
            for shift in 0..16 {
                let mut buf = vec![0u8; shift];
                buf.extend_from_slice(&bytes);
                let read = with_archived(&buf[shift..], |a| (a.response.len(), a.embedding.len()));
                assert_eq!(read.unwrap(), (e.response.len(), dim), "shift {shift}");
            }
        }
    }

    #[test]
    fn foreign_or_corrupted_bytes_are_rejected_not_read() {
        let bytes = encode(&entry(8)).unwrap();
        let corrupted = |bytes: &[u8]| matches!(decode(bytes), Err(CacheError::CorruptedData(_)));
        // Bincode do PR 35, versão desconhecida, truncado e vazio.
        assert!(corrupted(&serialize_entry(&entry(8)).unwrap()));
        let mut v2 = bytes.clone();
        v2[4] = 2;
        assert!(corrupted(&v2));
        assert!(corrupted(&bytes[..bytes.len() - 1]));
        assert!(corrupted(&[]));
        // UTF-8 inválido no texto gravado não vira `&str`: o bytecheck recusa.
        let mut evil = bytes.clone();
        let at = evil.windows(4).position(|w| w == b"Bras").unwrap();
        evil[at] = 0xff;
        assert!(corrupted(&evil));
    }

    /// Medição contra o bincode do crate de domínio:
    /// `cargo test --release --lib -- --ignored --nocapture zero_copy_vs_bincode`.
    #[test]
    #[ignore = "medição; rode com --release"]
    #[allow(clippy::cast_precision_loss)] // ns/op aproximado
    fn zero_copy_vs_bincode() {
        let e = entry(1536);
        let q: Vec<f32> = (0..1536).map(|i| (i as f32).cos()).collect();
        let (rk, bc) = (encode(&e).unwrap(), serialize_entry(&e).unwrap());
        let mut buf = AlignedVec::<ALIGN>::new();
        buf.extend_from_slice(&rk);
        let time = |name: &str, f: &mut dyn FnMut()| {
            let n = 20_000_u32;
            let start = Instant::now();
            for _ in 0..n {
                f();
            }
            println!(
                "{name:<38} {:>8.0} ns/op",
                start.elapsed().as_nanos() as f64 / f64::from(n)
            );
        };
        println!(
            "registro: rkyv {} bytes, bincode {} bytes",
            rk.len(),
            bc.len()
        );
        time("encode rkyv", &mut || {
            drop(black_box(encode(black_box(&e))));
        });
        time("encode bincode", &mut || {
            drop(black_box(serialize_entry(black_box(&e))));
        });
        time("decode inteiro rkyv", &mut || {
            drop(black_box(decode(black_box(&buf))));
        });
        time("decode inteiro bincode", &mut || {
            drop(black_box(deserialize_entry(black_box(&bc))));
        });
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&rk);
        time("decode inteiro rkyv, desalinhado", &mut || {
            drop(black_box(decode(black_box(&shifted[1..]))));
        });
        time("só a validação (rkyv)", &mut || {
            drop(black_box(with_archived(black_box(&buf), |a| {
                a.embedding.len()
            })));
        });
        time("cosseno sem decode (rkyv)", &mut || {
            drop(black_box(with_archived(black_box(&buf), |a| dot(a, &q))));
        });
        time("cosseno com decode (bincode)", &mut || {
            let entry = deserialize_entry(black_box(&bc)).unwrap();
            black_box(
                entry
                    .embedding
                    .data
                    .iter()
                    .zip(&q)
                    .map(|(x, y)| x * y)
                    .sum::<f32>(),
            );
        });
    }
}
