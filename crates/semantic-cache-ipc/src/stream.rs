//! Resposta em stream a um `LOOKUP_STREAM_REQ` (IPC v1): no hit, o texto
//! cacheado sai em `STREAM_CHUNK` `{"delta"}` de até `chunk_bytes` bytes, sem
//! partir caractere UTF-8, e termina num `STREAM_END` `{"hit", "score",
//! "latency_micros"}`; o miss é só o `STREAM_END`. Quem busca e chama o
//! [`write_stream`] é o despachante do PR 52/100.

use serde_json::json;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::frame::{encode, kind, FrameError};

/// `chunk_bytes` de quem não pede (ou pede 0).
pub const DEFAULT_CHUNK_BYTES: usize = 256;
/// Teto do `chunk_bytes`: o maior delta, mesmo escapado em JSON (até 6 bytes
/// por byte), fica longe dos 16 MiB de um frame.
pub const MAX_CHUNK_BYTES: usize = 64 << 10;

/// O `chunk_bytes` efetivo de um pedido: ausente ou 0 → o padrão; acima do
/// teto → o teto.
#[must_use]
pub fn chunk_bytes(requested: Option<u64>) -> usize {
    match requested {
        None | Some(0) => DEFAULT_CHUNK_BYTES,
        Some(n) => usize::try_from(n).map_or(MAX_CHUNK_BYTES, |n| n.min(MAX_CHUNK_BYTES)),
    }
}

/// Pedaços de `text` com até `max` bytes, sem partir caractere: um caractere
/// maior que `max` sai inteiro e sozinho. Texto vazio não dá pedaço.
pub fn chunks(text: &str, max: usize) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let first = rest.chars().next()?.len_utf8();
        let mut end = rest.len().min(max).max(first);
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        let (chunk, tail) = rest.split_at(end);
        rest = tail;
        Some(chunk)
    })
}

/// Escreve o stream de um LOOKUP; `response` é `Some` no hit. Cada frame
/// espera a escrita e o flush (contrapressão: o servidor não corre à frente
/// do cliente). O `score` sai no decimal mais curto do `f32` (`0.97`, não o
/// `0.9700000286102295` do alargamento para `f64`); NaN sai como `null`.
///
/// # Errors
/// [`FrameError::Io`] quando a escrita falha, por exemplo com o cliente que
/// some no meio: a task volta com o erro, sem pânico.
pub async fn write_stream<W: AsyncWrite + Unpin>(
    w: &mut W,
    request_id: u32,
    response: Option<&str>,
    score: f32,
    latency_micros: i64,
    chunk_bytes: usize,
) -> Result<(), FrameError> {
    for delta in chunks(response.unwrap_or_default(), chunk_bytes) {
        let chunk = json!({ "delta": delta });
        send(w, kind::STREAM_CHUNK, request_id, &chunk).await?;
    }
    let score = score.to_string().parse::<f64>().unwrap_or(f64::NAN);
    let hit = response.is_some();
    let end = json!({ "hit": hit, "score": score, "latency_micros": latency_micros });
    send(w, kind::STREAM_END, request_id, &end).await
}

async fn send<W: AsyncWrite + Unpin>(
    w: &mut W,
    kind: u8,
    request_id: u32,
    payload: &serde_json::Value,
) -> Result<(), FrameError> {
    let frame = encode(kind, 0, request_id, payload.to_string().as_bytes())?;
    w.write_all(&frame).await?;
    w.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use tokio::io::duplex;

    use super::*;
    use crate::frame::read_frame;

    #[test]
    fn chunks_never_split_a_character_and_rebuild_the_text() {
        let text = "olá ≠ 😀 cache\u{0}!";
        for max in 0..=8 {
            let pieces: Vec<&str> = chunks(text, max).collect();
            assert_eq!(pieces.concat(), text, "max {max}");
            for piece in &pieces {
                let single = piece.chars().count() == 1;
                assert!(piece.len() <= max || single, "max {max}: {piece:?}");
            }
        }
        assert_eq!(chunks("", 4).count(), 0);
    }

    #[test]
    fn chunk_bytes_has_a_default_and_a_cap() {
        assert_eq!(chunk_bytes(None), DEFAULT_CHUNK_BYTES);
        assert_eq!(chunk_bytes(Some(0)), DEFAULT_CHUNK_BYTES);
        assert_eq!(chunk_bytes(Some(5)), 5);
        assert_eq!(chunk_bytes(Some(u64::MAX)), MAX_CHUNK_BYTES);
    }

    /// Lê frames até o `STREAM_END`: `(deltas, corpo do fim)`.
    async fn collect(conn: &mut tokio::io::DuplexStream, id: u32) -> (Vec<String>, Value) {
        let mut deltas = Vec::new();
        loop {
            let frame = read_frame(conn).await.unwrap().unwrap();
            assert_eq!(frame.request_id, id);
            let body: Value = serde_json::from_slice(&frame.payload).unwrap();
            match frame.kind {
                kind::STREAM_CHUNK => deltas.push(body["delta"].as_str().unwrap().to_owned()),
                kind::STREAM_END => return (deltas, body),
                other => panic!("kind {other:#x} no meio do stream"),
            }
        }
    }

    #[tokio::test]
    async fn hit_streams_chunks_then_end_and_miss_is_only_end() {
        let (mut server, mut client) = duplex(1 << 16);
        write_stream(&mut server, 3, Some("olá ≠ mundo"), 0.97, 41, 4)
            .await
            .unwrap();
        let (deltas, end) = collect(&mut client, 3).await;
        assert_eq!(deltas, ["olá", " ≠", " mun", "do"]);
        let hit = json!({ "hit": true, "score": 0.97, "latency_micros": 41 });
        assert_eq!(end, hit);
        write_stream(&mut server, 4, None, f32::NAN, 7, 4)
            .await
            .unwrap();
        let miss = json!({ "hit": false, "score": null, "latency_micros": 7 });
        assert_eq!(collect(&mut client, 4).await, (Vec::new(), miss));
    }

    #[tokio::test]
    async fn writer_waits_for_the_reader_and_stops_when_it_leaves() {
        // Buffer de 64 bytes: sem leitura, nem o primeiro frame inteiro cabe.
        let (mut server, mut client) = duplex(64);
        let text = "x".repeat(1 << 16);
        let writer = tokio::spawn(async move {
            write_stream(&mut server, 9, Some(text.as_str()), 1.0, 0, 1024).await
        });
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(!writer.is_finished(), "escreveu sem o cliente ler");
        let first = read_frame(&mut client).await.unwrap().unwrap();
        assert_eq!(first.kind, kind::STREAM_CHUNK);
        drop(client);
        let err = writer.await.unwrap().unwrap_err();
        assert!(matches!(err, FrameError::Io(_)), "{err}");
    }
}
