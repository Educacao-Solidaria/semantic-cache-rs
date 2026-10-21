//! Servidor do IPC v1: uma task por conexão, cada uma entregue a um
//! [`Handler`], e encerramento que drena com prazo. As conexões vêm de um
//! [`Listener`]: o socket Unix (`bind_owned`) ou, no Windows, o named pipe
//! (`bind_pipe`), com o mesmo framing.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::sync::{watch, Semaphore};
use tokio::task::JoinSet;
use tokio::time::Sleep;
use tracing::warn;

use crate::frame::{code, encode, error_payload, kind, read_frame, FrameError};
use crate::metrics::{self, Outcome};

#[cfg(windows)]
mod pipe;
#[cfg(unix)]
mod unix;

#[cfg(windows)]
pub use pipe::{bind_pipe, shutdown_signal, PipeListener};
#[cfg(unix)]
pub use unix::{bind, bind_owned, in_shared_dir, shutdown_signal, SocketFile};

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

/// Tetos do [`serve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Conexões atendidas ao mesmo tempo. A seguinte recebe ERROR `overloaded`
    /// com `request_id` 0 e é fechada (contrato).
    pub max_connections: usize,
    /// Leitura ou escrita parada (o par não manda nem lê) por este tempo
    /// falha com `TimedOut`, e o handler fecha; `Duration::ZERO` = sem prazo.
    pub idle: Duration,
    /// Quanto o encerramento espera pelas conexões antes de abortá-las.
    pub drain: Duration,
}

/// O ERROR `overloaded` não espera mais que isto por um par que não lê.
const REFUSAL_DEADLINE: Duration = Duration::from_secs(1);

/// Aceita conexões até `shutdown` resolver; aí para de aceitar, avisa as
/// conexões e espera até `limits.drain` por elas, abortando as que passarem
/// do prazo. Cada conexão ocupa uma das `limits.max_connections` vagas até
/// cair, por qualquer motivo: fim, erro, prazo, pânico ou aborto.
pub async fn serve<L: Listener, H: Handler>(
    mut listener: L,
    handler: H,
    shutdown: impl Future,
    limits: Limits,
) {
    let handler = Arc::new(handler);
    let slots = limits.max_connections.min(Semaphore::MAX_PERMITS);
    let slots = Arc::new(Semaphore::new(slots));
    let (stop, stopping) = watch::channel(false);
    let mut conns = JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => match accepted {
                Ok(conn) => {
                    let Ok(slot) = Arc::clone(&slots).try_acquire_owned() else {
                        warn!(max = limits.max_connections, "limite de conexões; recusando");
                        conns.spawn(refuse(conn));
                        continue;
                    };
                    let (handler, stopping) = (Arc::clone(&handler), stopping.clone());
                    let (conn, open) = (Idle::new(conn, limits.idle), metrics::Open::new());
                    // A vaga e a contagem vão com a task: voltam no fim dela ou no
                    // drop (pânico, aborto).
                    conns.spawn(async move {
                        handler.handle(conn, stopping).await;
                        drop((slot, open));
                    });
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
    if tokio::time::timeout(limits.drain, all_done).await.is_err() {
        warn!(left = conns.len(), "prazo de drenagem esgotado; abortando");
        conns.shutdown().await;
    }
}

/// Conexão acima do teto: ERROR `overloaded` com `request_id` 0, e fecha.
async fn refuse<S: Conn>(mut conn: S) {
    let payload = error_payload(code::OVERLOADED, "limite de conexões atingido");
    if let Ok(frame) = encode(kind::ERROR, 0, 0, &payload) {
        let _ = tokio::time::timeout(REFUSAL_DEADLINE, conn.write_all(&frame)).await;
    }
}

/// A conexão que o handler recebe, com o prazo de ociosidade: uma leitura (ou
/// escrita) que espera o par por `idle` sem andar falha com `TimedOut`. O
/// prazo só corre enquanto ela espera, então processar devagar não conta.
///
/// ponytail: quem manda 1 byte a cada `idle` segura a vaga; o upgrade é um
/// prazo por frame no despachante.
struct Idle<S> {
    conn: S,
    limit: Duration,
    reading: Option<Pin<Box<Sleep>>>,
    writing: Option<Pin<Box<Sleep>>>,
}

impl<S> Idle<S> {
    fn new(conn: S, limit: Duration) -> Self {
        Self {
            conn,
            limit,
            reading: None,
            writing: None,
        }
    }
}

/// `Pending` enquanto a espera cabe no prazo (o relógio começa na 1ª espera);
/// vencido, `TimedOut`.
fn waited<T>(
    timer: &mut Option<Pin<Box<Sleep>>>,
    idle: Duration,
    cx: &mut Context<'_>,
) -> Poll<io::Result<T>> {
    if idle.is_zero() {
        return Poll::Pending;
    }
    let timer = timer.get_or_insert_with(|| Box::pin(tokio::time::sleep(idle)));
    ready!(timer.as_mut().poll(cx));
    Poll::Ready(Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "conexão ociosa",
    )))
}

impl<S: AsyncRead + Unpin> AsyncRead for Idle<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        match Pin::new(&mut this.conn).poll_read(cx, buf) {
            Poll::Pending => waited(&mut this.reading, this.limit, cx),
            done @ Poll::Ready(_) => {
                this.reading = None;
                done
            }
        }
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for Idle<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        match Pin::new(&mut this.conn).poll_write(cx, buf) {
            Poll::Pending => waited(&mut this.writing, this.limit, cx),
            done @ Poll::Ready(_) => {
                this.writing = None;
                done
            }
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().conn).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().conn).poll_shutdown(cx)
    }
}

/// Handler mínimo até o despachante do PR 52/100, um pedido por vez e na ordem:
/// PING → PONG; STATS → zeros (não há store por trás); outro kind → ERROR
/// `unsupported`, e a conexão segue; frame inválido → o ERROR do contrato, e fecha.
/// Cada pedido respondido entra em [`metrics::request`].
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
        let (reply, outcome) = match frame.kind {
            kind::PING => (encode(kind::PONG, 0, id, b""), Outcome::Ok),
            kind::STATS_REQ => (encode(kind::STATS_RESP, 0, id, EMPTY_STATS), Outcome::Ok),
            other => {
                let message = format!("kind 0x{other:02x} não suportado");
                let payload = error_payload(code::UNSUPPORTED, &message);
                (encode(kind::ERROR, 0, id, &payload), Outcome::Error)
            }
        };
        let written = conn.write_all(&reply?).await;
        metrics::request(frame.kind, outcome.after_write(&written));
        written?;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use serde_json::{json, Value};
    use tokio::io::{duplex, AsyncReadExt, DuplexStream};
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    use super::*;
    use crate::frame::kind::*;

    pub(super) const PATIENCE: Duration = Duration::from_secs(10); // folga para runner lento

    /// Os `Limits` dos testes que não medem os tetos.
    pub(super) fn limits(drain: Duration) -> Limits {
        let (max_connections, idle) = (64, Duration::ZERO);
        Limits {
            max_connections,
            idle,
            drain,
        }
    }

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
        // Com prazo: um servidor que não fecha travaria o teste em vez de falhar.
        let closed = timeout(PATIENCE, client.read(&mut [0])).await.unwrap();
        assert_eq!(closed.unwrap(), 0, "a conexão fecha");
    }

    /// Os frames de referência do contrato (`frame::tests`), já montados.
    fn references() -> [Vec<u8>; 3] {
        let lookup = r#"{"prompt":"olá","embedding":[0.5,-0.25],"threshold":0.75}"#.as_bytes();
        let error = error_payload(code::BAD_REQUEST, "dimensão 2 ≠ 1536");
        let frame = |k, id, payload: &[u8]| encode(k, 0, id, payload).unwrap();
        [
            frame(PING, 1, b""),
            frame(LOOKUP_REQ, 7, lookup),
            frame(ERROR, 7, &error),
        ]
    }

    #[tokio::test]
    async fn any_flipped_bit_gets_the_contract_error_and_the_connection_closes() {
        for reference in references() {
            for bit in 0..reference.len() * 8 {
                let mut bad = reference.clone();
                bad[bit / 8] ^= 1 << (bit % 8);
                // Um PING bom logo atrás: o servidor não pode se ressincronizar nele.
                bad.extend(encode(PING, 0, 99, b"").unwrap());
                let (mut client, conn) = duplex(1 << 16);
                let (_stop, stopping) = watch::channel(false);
                tokio::spawn(async move { Basic.handle(conn, stopping).await });
                client.write_all(&bad).await.unwrap();
                client.shutdown().await.unwrap(); // tamanho que cresceu → EOF
                let mut answer = Vec::new();
                let read = timeout(PATIENCE, client.read_to_end(&mut answer)).await;
                read.unwrap().unwrap();
                let mut answer = answer.as_slice();
                // Sem resposta, só com o tamanho alterado: o frame ficou truncado.
                let Some(reply) = read_frame(&mut answer).await.unwrap() else {
                    assert!((12..16).contains(&(bit / 8)), "bit {bit}: sem ERROR");
                    continue;
                };
                let body: Value = serde_json::from_slice(&reply.payload).unwrap();
                let code = body["code"].as_str();
                assert_eq!(reply.kind, ERROR, "bit {bit}");
                assert!(
                    matches!(code, Some("bad_frame" | "too_large")),
                    "bit {bit}: {body}"
                );
                assert_eq!(answer, b"", "bit {bit}: respondeu depois do ERROR");
            }
        }
    }

    /// Listener de teste: cada `DuplexStream` mandado no canal é uma conexão.
    impl Listener for mpsc::Receiver<DuplexStream> {
        type Conn = DuplexStream;

        async fn accept(&mut self) -> io::Result<DuplexStream> {
            match self.recv().await {
                Some(conn) => Ok(conn),
                None => std::future::pending().await,
            }
        }
    }

    async fn dial(listener: &mpsc::Sender<DuplexStream>) -> DuplexStream {
        let (client, server) = duplex(1 << 16);
        listener.send(server).await.unwrap();
        client
    }

    async fn ping(conn: &mut DuplexStream, id: u32) -> (u8, u32, Value) {
        let ping = encode(PING, 0, id, b"").unwrap();
        conn.write_all(&ping).await.unwrap();
        next(conn).await
    }

    /// Uma conexão atendida, tentando de novo enquanto a vaga não volta.
    async fn served(listener: &mpsc::Sender<DuplexStream>) -> DuplexStream {
        let started = Instant::now();
        loop {
            let mut conn = dial(listener).await;
            let ping = encode(PING, 0, 7, b"").unwrap();
            let _ = conn.write_all(&ping).await; // a recusada pode já ter fechado
            match next(&mut conn).await {
                (PONG, 7, _) => return conn,
                (ERROR, 0, body) if body["code"] == code::OVERLOADED => {}
                other => panic!("resposta inesperada: {other:?}"),
            }
            assert!(started.elapsed() < PATIENCE, "a vaga não voltou");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// PONG a cada PING; outro kind derruba a task.
    struct Fragile;

    impl Handler for Fragile {
        async fn handle<S: Conn>(&self, mut conn: S, _stop: Stop) {
            while let Ok(Some(frame)) = read_frame(&mut conn).await {
                assert_eq!(frame.kind, PING, "pânico de propósito");
                let pong = encode(PONG, 0, frame.request_id, b"").unwrap();
                conn.write_all(&pong).await.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn over_the_limit_is_refused_and_every_way_out_gives_the_slot_back() {
        let (listener, accept) = mpsc::channel(8);
        let (max_connections, idle, drain) = (2, Duration::ZERO, Duration::ZERO);
        let limits = Limits {
            max_connections,
            idle,
            drain,
        };
        tokio::spawn(serve(accept, Fragile, std::future::pending::<()>(), limits));
        let (mut a, mut b) = (dial(&listener).await, dial(&listener).await);
        assert_eq!(ping(&mut a, 1).await, (PONG, 1, Value::Null));
        assert_eq!(ping(&mut b, 2).await, (PONG, 2, Value::Null));
        // A terceira (N+1): ERROR overloaded com request_id 0, e fecha.
        let refused = |mut conn: DuplexStream| async move {
            let reply = timeout(PATIENCE, next(&mut conn)).await;
            let (k, id, body) = reply.expect("a conexão N+1 foi aceita");
            assert_eq!((k, id, &body["code"]), (ERROR, 0, &json!(code::OVERLOADED)));
            assert_eq!(conn.read(&mut [0]).await.unwrap(), 0, "a recusada fecha");
        };
        refused(dial(&listener).await).await;
        // Pânico no handler: a vaga volta.
        a.write_all(&encode(STATS_REQ, 0, 3, b"").unwrap())
            .await
            .unwrap();
        assert_eq!(a.read(&mut [0]).await.unwrap(), 0, "a task caiu");
        let c = served(&listener).await;
        // O par que fecha: a vaga volta.
        drop(b);
        let d = served(&listener).await;
        // E só essas: com c e d no ar, a próxima é recusada de novo.
        refused(dial(&listener).await).await;
        drop((c, d));
    }

    #[tokio::test(start_paused = true)]
    async fn a_silent_peer_is_closed_at_the_idle_deadline_and_frees_its_slot() {
        let (listener, accept) = mpsc::channel(8);
        let (max_connections, idle, drain) = (1, Duration::from_secs(30), Duration::ZERO);
        let limits = Limits {
            max_connections,
            idle,
            drain,
        };
        tokio::spawn(serve(accept, Basic, std::future::pending::<()>(), limits));
        let mut conn = dial(&listener).await;
        // Atividade adia o prazo: um PING a cada 20 s, por 1 min.
        for id in 1..=3 {
            assert_eq!(ping(&mut conn, id).await.0, PONG);
            tokio::time::sleep(Duration::from_secs(20)).await;
        }
        // Calado: fecha 30 s depois do último PONG, 10 s a partir daqui.
        let started = tokio::time::Instant::now();
        let closed = timeout(Duration::from_secs(60), conn.read(&mut [0])).await;
        assert_eq!(closed.expect("sem prazo de ociosidade").unwrap(), 0);
        let waited = started.elapsed();
        assert!(waited >= Duration::from_secs(10) && waited < Duration::from_secs(11));
        drop((conn, served(&listener).await));
    }

    #[tokio::test(start_paused = true)]
    async fn a_peer_that_stops_reading_is_closed_at_the_idle_deadline() {
        let (listener, accept) = mpsc::channel(8);
        let (max_connections, idle, drain) = (1, Duration::from_secs(30), Duration::ZERO);
        let limits = Limits {
            max_connections,
            idle,
            drain,
        };
        tokio::spawn(serve(accept, Basic, std::future::pending::<()>(), limits));
        // Buffer de 64 bytes e PINGs sem ler as PONGs: a escrita do servidor para.
        let (mut client, server) = duplex(64);
        listener.send(server).await.unwrap();
        let pings: Vec<u8> = (0..64)
            .flat_map(|id| encode(PING, 0, id, b"").unwrap())
            .collect();
        let started = tokio::time::Instant::now();
        let sent = timeout(Duration::from_secs(60), client.write_all(&pings)).await;
        assert!(
            sent.expect("sem prazo na escrita").is_err(),
            "o servidor fechou"
        );
        assert!(started.elapsed() >= Duration::from_secs(30));
        drop((client, served(&listener).await));
    }
}
