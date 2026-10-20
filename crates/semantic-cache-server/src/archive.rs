//! Formato zero-copy (rkyv) das entradas do cache, para disco e memória: lê
//! os campos direto dos bytes, sem desserializar a entrada inteira (nem
//! alocar o prompt e a resposta para olhar só o embedding). O fio com o
//! gateway continua JSON (`docs/ipc-v1.md`); isto é só do lado Rust.
//!
//! Bytes que vêm do disco são entrada não confiável: todo acesso passa pela
//! validação do `bytecheck` (`unsafe_code = "forbid"` nem deixa usar o
//! `access_unchecked`), e o tamanho que um corpo comprimido declara é conferido
//! antes de descomprimir.

use rkyv::rancor::Error as RkyvError;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};
use semantic_cache_rs::codec::DEFAULT_MAX_BINARY_SIZE;
use semantic_cache_rs::{CacheEntry, CacheError, Vector};

use crate::config::Compression;

/// Cabeçalho de cada registro: magia, versão, flags e 2 zeros. Os 8 bytes
/// mantêm o corpo com o mesmo alinhamento (até 8) do buffer em que o registro
/// está.
pub const MAGIC: [u8; 4] = *b"SCRK";
/// O layout é o do rkyv 0.8: subir o major do rkyv é subir esta versão.
pub const VERSION: u8 = 1;
const HEADER: [u8; 8] = [MAGIC[0], MAGIC[1], MAGIC[2], MAGIC[3], VERSION, 0, 0, 0];
/// Flag (byte 5): corpo em LZ4 (bloco, com o tamanho original na frente).
/// Quem não conhece um bit de flag recusa o registro.
const LZ4: u8 = 1;
/// Corpo menor não comprime: o pouco que pouparia não paga a descompressão.
const MIN_TO_COMPRESS: usize = 1 << 10;
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

/// Grava `entry` no formato: cabeçalho e os bytes do rkyv. Com
/// [`Compression::Lz4`], o corpo de 1 KiB ou mais vai comprimido se encolher
/// pelo menos 1/8.
///
/// # Errors
/// [`CacheError::Serialization`] se o rkyv falhar, ou com o registro acima do
/// teto de leitura do codec do crate de domínio (o que um banco v1 aceitava),
/// contado sem compressão: comprimido ou não, ele tem de ser lido de volta.
pub fn encode(entry: &CacheEntry, compression: Compression) -> Result<Vec<u8>, CacheError> {
    let record = EntryRecord::from(entry);
    let body = rkyv::to_bytes::<RkyvError>(&record).map_err(|e| serialization(&e))?;
    let size = HEADER.len() + body.len();
    if size > DEFAULT_MAX_BINARY_SIZE {
        return Err(CacheError::Serialization(format!(
            "registro de {size} bytes passa do limite de leitura ({DEFAULT_MAX_BINARY_SIZE})"
        )));
    }
    let packed = (compression == Compression::Lz4 && body.len() >= MIN_TO_COMPRESS)
        .then(|| lz4_flex::block::compress_prepend_size(&body))
        .filter(|packed| packed.len() <= body.len() - body.len() / 8);
    let (flags, body) = match &packed {
        Some(packed) => (LZ4, packed.as_slice()),
        None => (0, body.as_slice()),
    };
    let mut out = Vec::with_capacity(HEADER.len() + body.len());
    out.extend_from_slice(&HEADER);
    out[5] = flags;
    out.extend_from_slice(body);
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
    let (flags, body) = match bytes.split_at_checked(HEADER.len()) {
        Some((head, body)) if head[..5] == HEADER[..5] && head[6..] == HEADER[6..] => {
            (head[5], body)
        }
        _ => return Err(CacheError::corrupted("registro sem o cabeçalho SCRK v1")),
    };
    let unpacked;
    let body = match flags {
        0 => body,
        LZ4 => {
            unpacked = unpack(body)?;
            unpacked.as_slice()
        }
        _ => return Err(CacheError::corrupted("registro com flags desconhecidas")),
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

/// O corpo LZ4 descomprimido. O tamanho original declarado passa pelo teto de
/// leitura antes de qualquer alocação (bomba de descompressão).
fn unpack(body: &[u8]) -> Result<Vec<u8>, CacheError> {
    let bad = |e: lz4_flex::block::DecompressError| {
        CacheError::corrupted(format!("registro lz4 inválido: {e}"))
    };
    let (size, packed) = lz4_flex::block::uncompressed_size(body).map_err(bad)?;
    if size > DEFAULT_MAX_BINARY_SIZE {
        let msg = format!("registro lz4 declara {size} bytes, acima do teto de leitura");
        return Err(CacheError::corrupted(msg));
    }
    let raw = lz4_flex::block::decompress(packed, size).map_err(bad)?;
    if raw.len() != size {
        return Err(CacheError::corrupted("registro lz4 menor que o declarado"));
    }
    Ok(raw)
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
    use crate::config::Compression::{Lz4, Off};

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
        for (dim, compression) in [(0, Off), (1, Off), (1536, Off), (1536, Lz4)] {
            let e = entry(dim);
            let bytes = encode(&e, compression).unwrap();
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
        let bytes = encode(&entry(8), Off).unwrap();
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

    #[test]
    fn only_entries_that_shrink_go_compressed() {
        let e = entry(1536);
        let (raw, packed) = (encode(&e, Off).unwrap(), encode(&e, Lz4).unwrap());
        assert_eq!((raw[5], packed[5]), (0, LZ4));
        assert!(packed.len() < raw.len() - raw.len() / 8);
        // Corpo abaixo de 1 KiB e corpo que não encolhe (bits aleatórios) vão crus.
        let mut small = entry(0);
        (small.prompt, small.response) = ("oi".into(), "olá".into());
        let mut noise = small.clone();
        let mut x = 1_u32;
        noise.embedding.data = (0..2048)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from_bits(x >> 2) // expoente baixo: nem NaN nem ∞
            })
            .collect();
        for e in [small, noise] {
            let bytes = encode(&e, Lz4).unwrap();
            assert_eq!(bytes[5], 0, "{} bytes", bytes.len());
            assert_eq!(decode(&bytes).unwrap().embedding.data, e.embedding.data);
        }
    }

    #[test]
    fn a_bad_lz4_record_is_an_error_never_a_panic_nor_a_bomb() {
        let packed = encode(&entry(64), Lz4).unwrap();
        assert_eq!(packed[5], LZ4);
        let corrupted = |bytes: &[u8]| matches!(decode(bytes), Err(CacheError::CorruptedData(_)));
        // Qualquer bit trocado: erro ou outra entrada válida, nunca pânico.
        for bit in 0..packed.len() * 8 {
            let mut bad = packed.clone();
            bad[bit / 8] ^= 1 << (bit % 8);
            let _ = decode(&bad);
        }
        assert!(corrupted(&packed[..packed.len() - 1]), "truncado");
        let mut odd = packed.clone();
        odd[5] = LZ4 << 1;
        assert!(corrupted(&odd), "flag desconhecida");
        // Declara 4 GiB e manda 1 byte: recusado antes de descomprimir.
        let mut bomb = HEADER.to_vec();
        bomb[5] = LZ4;
        bomb.extend_from_slice(&u32::MAX.to_le_bytes());
        bomb.push(0);
        let err = decode(&bomb).unwrap_err();
        let refused =
            matches!(err, CacheError::CorruptedData(ref m) if m.contains("acima do teto"));
        assert!(refused, "{err}");
    }

    /// Tamanho e custo do LZ4 em entradas de 1536 dimensões:
    /// `cargo test --release --lib -- --ignored --nocapture lz4_size_and_cost`.
    #[test]
    #[ignore = "medição; rode com --release"]
    #[allow(clippy::cast_precision_loss)] // ns/op e % aproximados
    fn lz4_size_and_cost() {
        let mut prose = entry(1536);
        prose.response = include_str!("../../../docs/ipc-v1.md").to_owned();
        let mut bare = entry(1536);
        (bare.prompt, bare.response) = ("oi".into(), "olá".into());
        let ns = |f: &mut dyn FnMut()| {
            let start = Instant::now();
            for _ in 0..20_000 {
                f();
            }
            start.elapsed().as_nanos() as f64 / 20_000.0
        };
        for (name, e) in [
            ("texto repetido", entry(1536)),
            ("docs/ipc-v1.md", prose),
            ("só o embedding", bare),
        ] {
            let (raw, lz4) = (encode(&e, Off).unwrap(), encode(&e, Lz4).unwrap());
            let encode_ns = ns(&mut || drop(black_box(encode(black_box(&e), Lz4))));
            let raw_ns = ns(&mut || drop(black_box(decode(black_box(&raw)))));
            let lz4_ns = ns(&mut || drop(black_box(decode(black_box(&lz4)))));
            let pct = 100.0 * lz4.len() as f64 / raw.len() as f64;
            println!(
                "{name:<16} {} → {} bytes ({pct:.0}%), encode lz4 {encode_ns:.0} ns, decode {raw_ns:.0} → {lz4_ns:.0} ns",
                raw.len(),
                lz4.len()
            );
        }
    }

    /// Medição contra o bincode do crate de domínio:
    /// `cargo test --release --lib -- --ignored --nocapture zero_copy_vs_bincode`.
    #[test]
    #[ignore = "medição; rode com --release"]
    #[allow(clippy::cast_precision_loss)] // ns/op aproximado
    fn zero_copy_vs_bincode() {
        let e = entry(1536);
        let q: Vec<f32> = (0..1536).map(|i| (i as f32).cos()).collect();
        let (rk, bc) = (encode(&e, Off).unwrap(), serialize_entry(&e).unwrap());
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
            drop(black_box(encode(black_box(&e), Off)));
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
