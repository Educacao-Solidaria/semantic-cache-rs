//! Servidor do IPC v1: uma task por conexão, cada uma entregue a um
//! [`Handler`], e encerramento que drena com prazo. As conexões vêm de um
//! [`Listener`]: o socket Unix (`bind_owned`) ou, no Windows, o named pipe
//! (`bind_pipe`), com o mesmo framing.

use std::future::Future;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tracing::warn;

use crate::frame::{code, encode, error_payload, kind, read_frame, FrameError};

#[cfg(windows)]
mod pipe;
#[cfg(unix)]
mod unix;

#[cfg(windows)]
pub use pipe::{bind_pipe, shutdown_signal, PipeListener};
#[cfg(unix)]
pub use unix::{bind, bind_owned, shutdown_signal, SocketFile};

/// Uma conexão: o socket Unix, o named pipe, `duplex` em teste.
pub trait Conn: AsyncRead + AsyncWrite + Unpin + Send + 'static {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send + 'static> Conn for T {}

/// De onde o [`serve`] tira as conexões.
pub trait Listener: Send + 'static {
    type Conn: Conn;
    /// Espera a próxima conexão. Tem de ser cancel-safe: corre num `select!`.
    fn accept(&mut self) -> impl Future<Output = io::Result<Self::Conn>> + Send;
}

/// Aviso de encerramento: `true` (ou o canal fechado) = parar de ler pedidos novos.
pub type Stop = watch::Receiver<bool>;

/// Atende uma conexão; o despachante do PR 52/100 implementa este trait.
pub trait Handler: Send + Sync + 'static {
    /// Atende `conn` até o par fechar, um frame inválido ou o `stop`: aí
    /// termina o pedido em curso e volta, o que fecha a conexão.
    fn handle<S: Conn>(&self, conn: S, stop: Stop) -> impl Future<Output = ()> + Send;
}

/// Aceita conexões até `shutdown` resolver; aí para de aceitar, avisa as
/// conexões e espera até `drain` por elas, abortando as que passarem do prazo.
pub async fn serve<L: Listener, H: Handler>(
    mut listener: L,
    handler: H,
    shutdown: impl Future,
    drain: Duration,
) {
    let handler = Arc::new(handler);
    let (stop, stopping) = watch::channel(false);
    let mut conns = JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => match accepted {
                Ok(conn) => {
                    let (handler, stopping) = (Arc::clone(&handler), stopping.clone());
                    conns.spawn(async move { handler.handle(conn, stopping).await });
                }
                // Ex.: EMFILE. A pausa evita girar em falso sem descritor.
                Err(err) => {
                    warn!(%err, "accept falhou");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            },
            // Recolhe as que acabaram; pânico já sai pelo hook padrão.
            Some(_) = conns.join_next() => {}
        }
    }
    drop(listener);
    stop.send_replace(true);
    let all_done = async { while conns.join_next().await.is_some() {} };
    if tokio::time::timeout(drain, all_done).await.is_err() {
        warn!(left = conns.len(), "prazo de drenagem esgotado; abortando");
        conns.shutdown().await;
    }
}

/// Handler mínimo até o despachante do PR 52/100, um pedido por vez e na ordem:
/// PING → PONG; STATS → zeros (não há store por trás); outro kind → ERROR
/// `unsupported`, e a conexão segue; frame inválido → o ERROR do contrato, e fecha.
#[derive(Debug, Default, Clone, Copy)]
pub struct Basic;

const EMPTY_STATS: &[u8] = br#"{"total_entries":0,"hit_count":0,"miss_count":0,"hit_ratio":0.0}"#;

impl Handler for Basic {
    async fn handle<S: Conn>(&self, mut conn: S, mut stop: Stop) {
        let failed = answer(&mut conn, &mut stop).await.err();
        if let Some(reply) = failed.as_ref().and_then(FrameError::reply) {
            let _ = conn.write_all(&reply).await; // fecha em seguida de todo jeito
        }
    }
}

async fn answer(conn: &mut impl Conn, stop: &mut Stop) -> Result<(), FrameError> {
    loop {
        let frame = tokio::select! {
            biased;
            _ = stop.wait_for(|&s| s) => return Ok(()),
            frame = read_frame(conn) => frame?,
        };
        let Some(frame) = frame else { return Ok(()) };
        let id = frame.request_id;
        let reply = match frame.kind {
            kind::PING => encode(kind::PONG, 0, id, b""),
            kind::STATS_REQ => encode(kind::STATS_RESP, 0, id, EMPTY_STATS),
            other => {
                let message = format!("kind 0x{other:02x} não suportado");
                let payload = error_payload(code::UNSUPPORTED, &message);
                encode(kind::ERROR, 0, id, &payload)
            }
        };
        conn.write_all(&reply?).await?;
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};
    use tokio::io::{duplex, AsyncReadExt};

    use super::*;
    use crate::frame::kind::*;

    pub(super) const PATIENCE: Duration = Duration::from_secs(10); // folga para runner lento

    /// `(kind, request_id, payload em JSON)`; payload vazio vira `null`.
    pub(super) async fn next(conn: &mut impl Conn) -> (u8, u32, Value) {
        let frame = read_frame(conn).await.unwrap().unwrap();
        let body = serde_json::from_slice(&frame.payload).unwrap_or(Value::Null);
        (frame.kind, frame.request_id, body)
    }

    #[tokio::test]
    async fn basic_answers_in_order_and_closes_on_an_invalid_frame() {
        let (mut client, conn) = duplex(1 << 16);
        let (_stop, stopping) = watch::channel(false);
        tokio::spawn(async move { Basic.handle(conn, stopping).await });
        // Pedidos em pipeline numa escrita só; o último tem um bit do CRC trocado.
        let mut batch = Vec::new();
        for (k, id) in [(PING, 1), (STATS_REQ, 2), (LOOKUP_REQ, 3), (0x55, 4)] {
            batch.extend(encode(k, 0, id, b"{}").unwrap());
        }
        batch.extend(encode(PING, 0, 5, b"").unwrap());
        *batch.last_mut().unwrap() ^= 1;
        client.write_all(&batch).await.unwrap();
        let stats =
            json!({ "total_entries": 0, "hit_count": 0, "miss_count": 0, "hit_ratio": 0.0 });
        let unsupported =
            |k| json!({ "code": "unsupported", "message": format!("kind {k} não suportado") });
        assert_eq!(next(&mut client).await, (PONG, 1, Value::Null));
        assert_eq!(next(&mut client).await, (STATS_RESP, 2, stats));
        assert_eq!(next(&mut client).await, (ERROR, 3, unsupported("0x01")));
        assert_eq!(next(&mut client).await, (ERROR, 4, unsupported("0x55")));
        let (k, id, body) = next(&mut client).await;
        assert_eq!((k, id, &body["code"]), (ERROR, 5, &json!(code::BAD_FRAME)));
        assert_eq!(client.read(&mut [0]).await.unwrap(), 0, "a conexão fecha");
    }
}
