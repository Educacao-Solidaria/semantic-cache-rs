//! Servidor de referência do IPC v1 (`docs/ipc-v1.md`) para medir e testar o
//! caminho inteiro: o `serve` e o codec de produção (`semantic_cache_ipc`)
//! com um handler que responde LOOKUP pela busca SIMD de um [`VectorIndex`],
//! guarda com STORE e conta com STATS.
//!
//! O despacho de produção é o PR 52/100 (Luan); até ele chegar, o binário do
//! servidor só responde PING e STATS, e quem fala LOOKUP/STORE de ponta a ponta
//! (o bench `ipc`, o teste de carga e o e2e com o gateway Go, pelo binário
//! `reference-server`) usa este handler.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{PoisonError, RwLock};
use std::time::Instant;

use semantic_cache_ipc::frame::{code, encode, error_payload, kind, read_frame, Frame};
use semantic_cache_ipc::server::{Conn, Handler, Stop};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::stress::VectorIndex;

/// Payload do `LOOKUP_REQ`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LookupReq {
    pub prompt: String,
    pub embedding: Vec<f32>,
    pub threshold: f32,
}

/// Payload do `LOOKUP_RESP` (o `cache.LookupResult` do gateway).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LookupResp {
    pub hit: bool,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_response: Option<String>,
    pub latency_micros: i64,
}

/// Payload do `STORE_REQ`. ponytail: `prompt` e `ttl_seconds` são ignorados
/// (a busca é pelo embedding, e a entrada nunca expira); o TTL é do store de
/// produção (PR 52/100).
#[derive(Deserialize)]
struct StoreReq {
    response: String,
    embedding: Vec<f32>,
}

/// Payload do `STATS_RESP` (o `cache.StatsResult` do gateway).
#[derive(Serialize)]
struct StatsResp {
    total_entries: u64,
    hit_count: u64,
    miss_count: u64,
    hit_ratio: f32,
}

/// Cache em memória: o índice e o texto guardado para cada id dele, atrás de
/// um `RwLock` (LOOKUP lê, STORE escreve). Um pedido por vez em cada conexão,
/// na ordem em que chegam (o pipeline do cliente vale); as conexões correm em
/// paralelo nos workers do runtime.
///
/// ponytail: a busca roda no worker do tokio e o segura enquanto varre; com
/// índice grande e muitas conexões, o upgrade é `spawn_blocking` ou um pool
/// próprio para a varredura. Sem teto de entradas: é ferramental de teste.
#[derive(Debug)]
pub struct SearchCache<I> {
    entries: RwLock<Entries<I>>,
    dim: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

/// O índice e o texto de cada id dele (`None` = vetor sem texto).
#[derive(Debug)]
struct Entries<I> {
    index: I,
    responses: Vec<Option<String>>,
}

impl<I: VectorIndex> SearchCache<I> {
    /// `responses[id]` é o texto do vetor `id` de `index`, todos de dimensão
    /// `dim`. Vetor sem texto nunca dá hit.
    #[must_use]
    pub fn new(index: I, dim: usize, responses: Vec<String>) -> Self {
        let responses = responses.into_iter().map(Some).collect();
        Self {
            entries: RwLock::new(Entries { index, responses }),
            dim,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// O frame que responde a `frame`, já conferido pelo codec.
    fn answer(&self, frame: &Frame) -> Vec<u8> {
        let id = frame.request_id;
        let error = |code, message: &str| encode(kind::ERROR, 0, id, &error_payload(code, message));
        let reply = match frame.kind {
            kind::PING => encode(kind::PONG, 0, id, b""),
            kind::LOOKUP_REQ => match self.lookup(&frame.payload) {
                Ok(resp) => encode(kind::LOOKUP_RESP, 0, id, &resp),
                Err(message) => error(code::BAD_REQUEST, &message),
            },
            kind::STORE_REQ => match self.store(&frame.payload) {
                Ok(()) => encode(kind::STORE_RESP, 0, id, b"{}"),
                Err(message) => error(code::BAD_REQUEST, &message),
            },
            kind::STATS_REQ => encode(kind::STATS_RESP, 0, id, &self.stats()),
            other => error(
                code::UNSUPPORTED,
                &format!("kind 0x{other:02x} não suportado"),
            ),
        };
        // Só passa do teto um texto guardado acima de 16 MiB.
        reply
            .or_else(|_| error(code::INTERNAL, "resposta acima do teto do frame"))
            .expect("um ERROR curto cabe no frame")
    }

    /// O vizinho mais próximo; hit se o escore chega ao `threshold`. A
    /// mensagem de erro não leva o prompt nem o embedding (contrato).
    fn lookup(&self, payload: &[u8]) -> Result<Vec<u8>, String> {
        let started = Instant::now();
        let req: LookupReq =
            serde_json::from_slice(payload).map_err(|_| "LOOKUP_REQ ilegível".to_owned())?;
        let query = self.checked(&req.embedding)?;
        let entries = self.entries.read().unwrap_or_else(PoisonError::into_inner);
        let best = entries.index.search(&query, 1).first().copied();
        let cached = best
            .filter(|b| b.score >= req.threshold)
            .and_then(|b| entries.responses.get(b.id).cloned().flatten());
        drop(entries);
        let counter = if cached.is_some() {
            &self.hits
        } else {
            &self.misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
        let resp = LookupResp {
            hit: cached.is_some(),
            score: best.map_or(0.0, |b| b.score),
            cached_response: cached,
            latency_micros: i64::try_from(started.elapsed().as_micros()).unwrap_or(i64::MAX),
        };
        Ok(serde_json::to_vec(&resp).expect("struct sem mapa sempre serializa"))
    }

    /// `embedding` na dimensão do índice, com norma 1 (ver [`unit`]).
    fn checked(&self, embedding: &[f32]) -> Result<Vec<f32>, String> {
        if embedding.len() != self.dim {
            return Err(format!("dimensão {} ≠ {}", embedding.len(), self.dim));
        }
        unit(embedding).ok_or_else(|| "embedding não finito ou de norma zero".to_owned())
    }

    /// Guarda o texto com o próximo id do índice; a mensagem de erro não leva
    /// o texto nem o embedding (contrato).
    fn store(&self, payload: &[u8]) -> Result<(), String> {
        let req: StoreReq =
            serde_json::from_slice(payload).map_err(|_| "STORE_REQ ilegível".to_owned())?;
        let vector = self.checked(&req.embedding)?;
        let mut entries = self.entries.write().unwrap_or_else(PoisonError::into_inner);
        let id = entries.index.len();
        entries.index.insert(&vector);
        if entries.responses.len() <= id {
            entries.responses.resize(id + 1, None);
        }
        entries.responses[id] = Some(req.response);
        Ok(())
    }

    /// Entradas no índice e os LOOKUP respondidos (pedido recusado não conta).
    fn stats(&self) -> Vec<u8> {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let entries = self.entries.read().unwrap_or_else(PoisonError::into_inner);
        let total_entries = u64::try_from(entries.index.len()).unwrap_or(u64::MAX);
        drop(entries);
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // razão em [0, 1]
        let hit_ratio = match hits + misses {
            0 => 0.0,
            all => (hits as f64 / all as f64) as f32,
        };
        let stats = StatsResp {
            total_entries,
            hit_count: hits,
            miss_count: misses,
            hit_ratio,
        };
        serde_json::to_vec(&stats).expect("struct sem mapa sempre serializa")
    }
}

/// `v` com norma 1: o índice compara cossenos de vetores unitários, e um
/// vetor quase zero (que quantizaria para zeros) vale pela direção. A norma
/// sai em f64, sem estourar nem zerar no quadrado; `None` se for zero ou não
/// finita (o JSON não carrega NaN, mas `1e39` chega ao f32 como infinito).
fn unit(v: &[f32]) -> Option<Vec<f32>> {
    let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    #[allow(clippy::cast_possible_truncation)] // cada componente fica em [-1, 1]
    let scaled = |x: f32| (f64::from(x) / norm) as f32;
    (norm > 0.0 && norm.is_finite()).then(|| v.iter().copied().map(scaled).collect())
}

impl<I: VectorIndex + Send + Sync + 'static> Handler for SearchCache<I> {
    async fn handle<S: Conn>(&self, mut conn: S, mut stop: Stop) {
        loop {
            let read = tokio::select! {
                biased;
                _ = stop.wait_for(|&s| s) => return,
                read = read_frame(&mut conn) => read,
            };
            let (reply, last) = match read {
                Ok(Some(frame)) => (self.answer(&frame), false),
                Ok(None) => return,
                // Frame inválido: o ERROR do contrato, e fecha.
                Err(err) => (err.reply().unwrap_or_default(), true),
            };
            if conn.write_all(&reply).await.is_err() || last {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use semantic_cache_ipc::frame::kind::*;
    use serde_json::{json, Value};
    use tokio::io::{duplex, AsyncReadExt};
    use tokio::sync::watch;
    use tokio::time::timeout;

    use super::*;
    use crate::stress::{FlatF32, FlatI8};

    /// Prazo de quem espera o servidor fechar: travar vira falha (runner lento).
    const PATIENCE: Duration = Duration::from_secs(10);

    /// A base canônica de R⁴ menos o último eixo; o texto do id 1 tem o que
    /// o JSON escapa, e o id 2 fica sem texto.
    fn cache<I: VectorIndex + Default>() -> SearchCache<I> {
        let mut index = I::default();
        for axis in 0..3 {
            let mut v = [0.0; 4];
            v[axis] = 1.0;
            index.insert(&v);
        }
        let texts = ["e0", "olá ≠ \"e1\"\n"].map(String::from);
        SearchCache::new(index, 4, texts.to_vec())
    }

    fn lookup(id: u32, embedding: &str) -> Vec<u8> {
        let body = format!(r#"{{"prompt":"p","embedding":{embedding},"threshold":0.9}}"#);
        encode(LOOKUP_REQ, 0, id, body.as_bytes()).unwrap()
    }

    /// Manda `requests` em pipeline, fecha a escrita e lê até o servidor
    /// fechar: `(kind, request_id, payload em JSON)` de cada resposta.
    async fn exchange<I>(cache: SearchCache<I>, requests: &[u8]) -> Vec<(u8, u32, Value)>
    where
        I: VectorIndex + Send + Sync + 'static,
    {
        let (mut client, conn) = duplex(1 << 16);
        let (_stop, stopping) = watch::channel(false);
        tokio::spawn(async move { cache.handle(conn, stopping).await });
        client.write_all(requests).await.unwrap();
        client.shutdown().await.unwrap();
        let mut bytes = Vec::new();
        let read = timeout(PATIENCE, client.read_to_end(&mut bytes)).await;
        read.expect("o servidor não fechou").unwrap();
        let mut rest = bytes.as_slice();
        let mut replies = Vec::new();
        while let Some(f) = read_frame(&mut rest).await.unwrap() {
            let body = serde_json::from_slice(&f.payload).unwrap_or(Value::Null);
            replies.push((f.kind, f.request_id, body));
        }
        replies
    }

    async fn answers_in_order_and_rejects_bad_requests<I>()
    where
        I: VectorIndex + Default + Send + Sync + 'static,
    {
        let mut batch = [
            lookup(1, "[0,1,0,0]"),
            lookup(2, "[0.6,0.8,0,0]"),
            lookup(3, "[1,0,0]"),
            lookup(4, "[0,0,0,0]"),
            lookup(5, "[1e39,0,0,0]"),
            encode(LOOKUP_REQ, 0, 6, br#"{"prompt":"#).unwrap(),
            encode(LOOKUP_STREAM_REQ, 0, 7, b"{}").unwrap(),
            encode(PING, 0, 8, b"").unwrap(),
            lookup(9, "[0,0,2,0]"),
            lookup(10, "[1e-45,0,0,0]"),
        ]
        .concat();
        // Um PING com o CRC trocado: ERROR e fecha, sem atender o PING bom de trás.
        batch.extend(encode(PING, 0, 11, b"").unwrap());
        *batch.last_mut().unwrap() ^= 1;
        batch.extend(encode(PING, 0, 12, b"").unwrap());
        let replies = exchange(cache::<I>(), &batch).await;
        let got: Vec<(u8, u32, &str)> = replies
            .iter()
            .map(|(k, id, body)| (*k, *id, body["code"].as_str().unwrap_or("")))
            .collect();
        let bad = "bad_request";
        let expected = [
            (LOOKUP_RESP, 1, ""),
            (LOOKUP_RESP, 2, ""),
            (ERROR, 3, bad),
            (ERROR, 4, bad),
            (ERROR, 5, bad),
            (ERROR, 6, bad),
            (ERROR, 7, "unsupported"),
            (PONG, 8, ""),
            (LOOKUP_RESP, 9, ""),
            (LOOKUP_RESP, 10, ""),
            (ERROR, 11, "bad_frame"), // e fecha: o PING 12 fica sem resposta
        ];
        assert_eq!(got, expected);
        let resp = |i: usize| serde_json::from_value::<LookupResp>(replies[i].2.clone()).unwrap();
        let summary = |i: usize| {
            let r = resp(i);
            (r.hit, r.score, r.cached_response)
        };
        assert_eq!(summary(0), (true, 1.0, Some("olá ≠ \"e1\"\n".into())));
        let (hit, score, text) = summary(1);
        let close = (score - 0.8).abs() < 0.01; // o escore do mais próximo
        assert!(!hit && text.is_none() && close, "{score}");
        assert_eq!(summary(8), (false, 1.0, None), "o id 2 não tem texto");
        assert_eq!(
            summary(9),
            (true, 1.0, Some("e0".into())),
            "quase zero vale pela direção"
        );
        let message = |i: usize| replies[i].2["message"].as_str().unwrap();
        let refused = "embedding não finito ou de norma zero";
        assert_eq!(
            [message(2), message(3), message(4)],
            ["dimensão 3 ≠ 4", refused, refused]
        );
    }

    #[tokio::test]
    async fn flat_f32_answers_in_order_and_rejects_bad_requests() {
        answers_in_order_and_rejects_bad_requests::<FlatF32>().await;
    }

    /// No `FlatI8`, o vetor zero faria a quantização entrar em pânico.
    #[tokio::test]
    async fn flat_i8_answers_in_order_and_rejects_bad_requests() {
        answers_in_order_and_rejects_bad_requests::<FlatI8>().await;
    }

    /// STORE guarda com o próximo id do índice (o 3, depois dos eixos de
    /// `cache`; o id 2 segue sem texto), e STATS conta o que o LOOKUP respondeu.
    #[tokio::test]
    async fn store_then_lookup_and_stats() {
        let store = |id, embedding: &str| {
            let body = format!(r#"{{"prompt":"p","response":"novo","embedding":{embedding}}}"#);
            encode(STORE_REQ, 0, id, body.as_bytes()).unwrap()
        };
        let batch = [
            store(1, "[0,0,0,2]"),
            lookup(2, "[0,0,0,1]"),
            lookup(3, "[0,0,1,0]"),
            store(4, "[0,0,1]"),
            store(5, "[0,0,0,0]"),
            encode(STORE_REQ, 0, 6, br#"{"prompt":"p"}"#).unwrap(),
            encode(STATS_REQ, 0, 7, b"{}").unwrap(),
        ];
        let replies = exchange(cache::<FlatI8>(), &batch.concat()).await;
        let body = |i: usize| &replies[i].2;
        let kinds: Vec<u8> = replies.iter().map(|r| r.0).collect();
        let want = [STORE_RESP, LOOKUP_RESP, LOOKUP_RESP, ERROR, ERROR, ERROR];
        assert_eq!(kinds, [&want[..], &[STATS_RESP]].concat());
        assert_eq!(*body(0), json!({}));
        assert_eq!(body(1)["cached_response"], "novo");
        assert_eq!(body(2)["hit"], false);
        assert_eq!(body(2)["score"], 1.0);
        for i in 3..6 {
            assert_eq!(body(i)["code"], "bad_request", "pedido {}", i + 1);
        }
        let stats = json!({"total_entries": 4, "hit_count": 1, "miss_count": 1, "hit_ratio": 0.5});
        assert_eq!(*body(6), stats);
    }

    #[tokio::test]
    async fn empty_index_and_dimension_zero() {
        let empty = SearchCache::new(FlatF32::default(), 4, Vec::new());
        let replies = exchange(empty, &lookup(1, "[1,0,0,0]")).await;
        let resp: LookupResp = serde_json::from_value(replies[0].2.clone()).unwrap();
        assert_eq!(
            (resp.hit, resp.score, resp.cached_response),
            (false, 0.0, None)
        );
        let flat = SearchCache::new(FlatF32::default(), 0, Vec::new());
        let replies = exchange(flat, &[lookup(1, "[]"), lookup(2, "[1]")].concat()).await;
        assert_eq!(replies[0].2["code"], "bad_request");
        assert_eq!(replies[1].2["message"], "dimensão 1 ≠ 0");
    }

    #[tokio::test]
    async fn stop_closes_an_idle_connection() {
        let (mut client, conn) = duplex(64);
        let (stop, stopping) = watch::channel(false);
        let cache = cache::<FlatF32>();
        let task = tokio::spawn(async move { cache.handle(conn, stopping).await });
        stop.send_replace(true);
        let closed = timeout(PATIENCE, client.read(&mut [0])).await;
        assert_eq!(closed.expect("o stop não fechou").unwrap(), 0);
        task.await.unwrap();
    }
}
