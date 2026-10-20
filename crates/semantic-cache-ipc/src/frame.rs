//! Codec do IPC v1 (`docs/ipc-v1.md`): cabeçalho de 16 bytes em big-endian,
//! payload JSON e CRC32 IEEE sobre todos os bytes anteriores a ele.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::io::{AsyncRead, AsyncReadExt};
use tracing::warn;

pub const MAGIC: [u8; 4] = *b"SCIP";
pub const VERSION: u8 = 1;
pub const HEADER_LEN: usize = 16;
/// Teto do payload (16 MiB): acima dele, `too_large` e a conexão fecha.
pub const MAX_PAYLOAD: u32 = 16 << 20;
/// O que se reserva para o payload antes de os bytes chegarem; acima disso, o
/// buffer cresce com o que chega.
const FIRST_RESERVE: u32 = 64 << 10;

static DISCARDED: AtomicU64 = AtomicU64::new(0);

/// Frames recusados pelo [`read_frame`] desde que o processo subiu: magic,
/// versão, flags ou CRC errados, ou payload acima do teto. Um frame cortado
/// pela queda do par não entra (não chegou inteiro para ser julgado).
#[must_use]
pub fn discarded_frames() -> u64 {
    DISCARDED.load(Ordering::Relaxed)
}

/// Kinds do v1, como `u8`: um kind desconhecido chega íntegro ao despachante.
pub mod kind {
    pub const LOOKUP_REQ: u8 = 0x01;
    pub const LOOKUP_RESP: u8 = 0x02;
    pub const STORE_REQ: u8 = 0x03;
    pub const STORE_RESP: u8 = 0x04;
    pub const STATS_REQ: u8 = 0x05;
    pub const STATS_RESP: u8 = 0x06;
    pub const LOOKUP_STREAM_REQ: u8 = 0x07;
    pub const STREAM_CHUNK: u8 = 0x08;
    pub const STREAM_END: u8 = 0x09;
    pub const PING: u8 = 0x0E;
    pub const PONG: u8 = 0x0F;
    pub const ERROR: u8 = 0x7F;
}

/// Códigos do payload de um ERROR.
pub mod code {
    pub const BAD_FRAME: &str = "bad_frame";
    pub const BAD_REQUEST: &str = "bad_request";
    pub const TOO_LARGE: &str = "too_large";
    pub const OVERLOADED: &str = "overloaded";
    pub const UNSUPPORTED: &str = "unsupported";
    pub const INTERNAL: &str = "internal";
}

/// Frame com magic, versão, flags e CRC conferidos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub kind: u8,
    pub request_id: u32,
    pub payload: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// `(request_id, motivo)`: magic, versão, flags ou CRC errados.
    #[error("frame inválido: {1}")]
    BadFrame(u32, &'static str),
    /// `(request_id, payload_len)`.
    #[error("payload de {1} bytes acima do teto de {MAX_PAYLOAD}")]
    TooLarge(u32, usize),
    /// O par sumiu no meio de um frame, ou a leitura/escrita falhou.
    #[error(transparent)]
    Io(#[from] io::Error),
}

impl FrameError {
    /// O ERROR que o contrato manda devolver antes de fechar a conexão;
    /// `None` em `Io`, quando não há a quem responder.
    #[must_use]
    pub fn reply(&self) -> Option<Vec<u8>> {
        let (id, code) = match self {
            Self::BadFrame(id, _) => (*id, code::BAD_FRAME),
            Self::TooLarge(id, _) => (*id, code::TOO_LARGE),
            Self::Io(_) => return None,
        };
        encode(kind::ERROR, 0, id, &error_payload(code, &self.to_string())).ok()
    }
}

/// Payload de um ERROR. A mensagem não pode levar prompt, resposta nem
/// embedding: podem ser dado do usuário.
#[must_use]
pub fn error_payload(code: &str, message: &str) -> Vec<u8> {
    let payload = serde_json::json!({ "code": code, "message": message });
    payload.to_string().into_bytes()
}

/// Monta o frame de `kind` com o request id `id`; no v1, `flags` é sempre 0.
///
/// # Errors
/// [`FrameError::TooLarge`] se o payload passar de [`MAX_PAYLOAD`].
pub fn encode(kind: u8, flags: u16, id: u32, payload: &[u8]) -> Result<Vec<u8>, FrameError> {
    let len = match u32::try_from(payload.len()) {
        Ok(len) if len <= MAX_PAYLOAD => len,
        _ => return Err(FrameError::TooLarge(id, payload.len())),
    };
    let mut frame = Vec::with_capacity(HEADER_LEN + payload.len() + 4);
    frame.extend_from_slice(&MAGIC);
    frame.extend_from_slice(&[VERSION, kind]);
    frame.extend_from_slice(&flags.to_be_bytes());
    frame.extend_from_slice(&id.to_be_bytes());
    frame.extend_from_slice(&len.to_be_bytes());
    frame.extend_from_slice(payload);
    let crc = crc32fast::hash(&frame);
    frame.extend_from_slice(&crc.to_be_bytes());
    Ok(frame)
}

/// Lê o próximo frame; `Ok(None)` se o par fecha entre dois frames. Nada é
/// alocado antes de conferido o teto, e o payload ocupa só o que já chegou.
/// Frame recusado conta em [`discarded_frames`] e vai para o log, sem o payload.
/// Não é cancel-safe: cancelada no meio, o frame em curso se perde e o stream
/// fica desalinhado; num `select!`, só quando cancelar = fechar a conexão.
///
/// # Errors
/// `BadFrame` e `TooLarge` pedem o [`FrameError::reply`] e o fechamento.
pub async fn read_frame<R: AsyncRead + Unpin>(r: &mut R) -> Result<Option<Frame>, FrameError> {
    let read = read_checked(r).await;
    if let Err(err @ (FrameError::BadFrame(..) | FrameError::TooLarge(..))) = &read {
        DISCARDED.fetch_add(1, Ordering::Relaxed);
        warn!(%err, "frame descartado");
    }
    read
}

async fn read_checked<R: AsyncRead + Unpin>(r: &mut R) -> Result<Option<Frame>, FrameError> {
    let mut head = [0; HEADER_LEN];
    if r.read(&mut head[..1]).await? == 0 {
        return Ok(None);
    }
    r.read_exact(&mut head[1..]).await?;
    let request_id = u32::from_be_bytes([head[8], head[9], head[10], head[11]]);
    let len = u32::from_be_bytes([head[12], head[13], head[14], head[15]]);
    let bad = |reason| Err(FrameError::BadFrame(request_id, reason));
    if head[..4] != MAGIC {
        return bad("magic diferente de SCIP");
    }
    if head[4] != VERSION {
        return bad("versão diferente de 1");
    }
    if head[6..8] != [0, 0] {
        return bad("flags desconhecidas");
    }
    if len > MAX_PAYLOAD {
        return Err(FrameError::TooLarge(request_id, len as usize));
    }
    // Cresce com o que chega: um cabeçalho que promete 16 MiB e para de mandar
    // não segura 16 MiB. Payload incompleto = o par fechou, e o `read_exact`
    // do CRC abaixo dá `UnexpectedEof`.
    let mut payload = Vec::with_capacity(len.min(FIRST_RESERVE) as usize);
    (&mut *r).take(len.into()).read_to_end(&mut payload).await?;
    let mut crc = [0; 4];
    r.read_exact(&mut crc).await?;
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&head);
    hasher.update(&payload);
    if hasher.finalize() != u32::from_be_bytes(crc) {
        return bad("crc32 não confere");
    }
    Ok(Some(Frame {
        kind: head[5],
        request_id,
        payload,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frames de referência do contrato (FLIPPED: o LOOKUP com 1 bit trocado no offset 20).
    const PING_HEX: &str = "53434950010e00000000000100000000c47a57de";
    const LOOKUP_HEX: &str = "5343495001010000000000070000003a7b2270726f6d7074223a226f6cc3a1222c22656d62656464696e67223a5b302e352c2d302e32355d2c227468726573686f6c64223a302e37357d9cad1a2b";
    const ERROR_HEX: &str = "53434950017f000000000007000000377b22636f6465223a226261645f72657175657374222c226d657373616765223a2264696d656e73c3a36f203220e289a02031353336227d0777f547";
    const FLIPPED_HEX: &str = "5343495001010000000000070000003a7b2270726e6d7074223a226f6cc3a1222c22656d62656464696e67223a5b302e352c2d302e32355d2c227468726573686f6c64223a302e37357d9cad1a2b";

    fn hex(s: &str) -> Vec<u8> {
        let byte = |i| u8::from_str_radix(&s[i..i + 2], 16).unwrap();
        (0..s.len()).step_by(2).map(byte).collect()
    }

    async fn decode(bytes: &[u8]) -> Result<Option<Frame>, FrameError> {
        read_frame(&mut &bytes[..]).await
    }

    #[tokio::test]
    async fn reference_frames_round_trip_byte_for_byte() {
        let lookup = r#"{"prompt":"olá","embedding":[0.5,-0.25],"threshold":0.75}"#;
        let error = error_payload(code::BAD_REQUEST, "dimensão 2 ≠ 1536");
        for (reference, kind, id, payload) in [
            (PING_HEX, kind::PING, 1, &b""[..]),
            (LOOKUP_HEX, kind::LOOKUP_REQ, 7, lookup.as_bytes()),
            (ERROR_HEX, kind::ERROR, 7, &error),
        ] {
            let bytes = hex(reference);
            assert_eq!(encode(kind, 0, id, payload).unwrap(), bytes);
            let mut rest = bytes.as_slice();
            let f = read_frame(&mut rest).await.unwrap().unwrap();
            assert_eq!((f.kind, f.request_id, &f.payload[..]), (kind, id, payload));
            assert_eq!(rest, b"", "o frame consome só os seus bytes");
        }
        assert!(decode(b"").await.unwrap().is_none(), "EOF entre frames");
    }

    #[tokio::test]
    async fn flipped_bit_and_oversized_payload_get_the_contract_error() {
        use code::{BAD_FRAME, TOO_LARGE};
        let flipped = decode(&hex(FLIPPED_HEX)).await.unwrap_err();
        assert_eq!(flipped.to_string(), "frame inválido: crc32 não confere");
        // Só o cabeçalho: acima do teto, o erro vem antes de tentar ler o payload.
        let mut head = hex(PING_HEX)[..HEADER_LEN].to_vec();
        head[12..].copy_from_slice(&(MAX_PAYLOAD + 1).to_be_bytes());
        let oversized = decode(&head).await.unwrap_err();
        for (err, id, code) in [(flipped, 7, BAD_FRAME), (oversized, 1, TOO_LARGE)] {
            let reply = decode(&err.reply().unwrap()).await.unwrap().unwrap();
            let body: serde_json::Value = serde_json::from_slice(&reply.payload).unwrap();
            assert_eq!((reply.kind, reply.request_id), (kind::ERROR, id));
            assert_eq!(body["code"], code);
        }
    }
}
