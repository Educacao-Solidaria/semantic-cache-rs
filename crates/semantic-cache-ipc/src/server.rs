//! Servidor do IPC v1 em Unix domain socket: uma task por conexão, cada uma
//! entregue a um [`Handler`], e encerramento que drena com prazo.

use std::future::Future;
use std::io;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use socket2::{Domain, SockAddr, Socket, Type};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tracing::{info, warn};

use crate::frame::{code, encode, error_payload, kind, read_frame, FrameError};

/// Uma conexão: o socket Unix aqui, o named pipe no PR 61/100, `duplex` em teste.
pub trait Conn: AsyncRead + AsyncWrite + Unpin + Send + 'static {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send + 'static> Conn for T {}

/// Aviso de encerramento: `true` (ou o canal fechado) = parar de ler pedidos novos.
pub type Stop = watch::Receiver<bool>;

/// Atende uma conexão; o despachante do PR 52/100 implementa este trait.
pub trait Handler: Send + Sync + 'static {
    /// Atende `conn` até o par fechar, um frame inválido ou o `stop`: aí
    /// termina o pedido em curso e volta, o que fecha a conexão.
    fn handle<S: Conn>(&self, conn: S, stop: Stop) -> impl Future<Output = ()> + Send;
}

/// Socket em `path` com permissão 0600 (outro usuário local não conecta).
/// Caminho existente é recusado, nunca sobrescrito; quem trata o resto de uma
/// queda é o [`bind_owned`].
///
/// # Errors
/// Caminho existente, longo demais para um socket ou inacessível.
pub fn bind(path: &Path) -> io::Result<UnixListener> {
    let socket = Socket::new(Domain::UNIX, Type::STREAM, None)?;
    socket.bind(&SockAddr::unix(path)?)?;
    // Antes do listen ninguém conecta: a 0600 vale já na 1ª conexão, com qualquer umask.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .and_then(|()| socket.listen(1024))
        .and_then(|()| socket.set_nonblocking(true))
        .and_then(|()| UnixListener::from_std(socket.into()))
        .inspect_err(|_| {
            let _ = std::fs::remove_file(path); // criado pelo bind acima
        })
}

/// O socket que o [`bind_owned`] criou. No `Drop` (fim normal, erro ou pânico)
/// remove o arquivo só se ele ainda for esse socket, pelo mesmo dispositivo e
/// inode: o que outro processo pôs no caminho depois fica.
#[derive(Debug)]
pub struct SocketFile {
    path: PathBuf,
    id: (u64, u64),
}

impl Drop for SocketFile {
    fn drop(&mut self) {
        let meta = std::fs::symlink_metadata(&self.path);
        if meta.is_ok_and(|m| (m.dev(), m.ino()) == self.id) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// [`bind`] que trata o que já está em `path`. Um socket em que ninguém escuta
/// (resto de uma queda) é removido. Servidor vivo, caminho que não é socket
/// (symlink não é seguido: `/tmp` é de todos) e socket que não dá para sondar
/// (de outro usuário) fazem a partida falhar, sem apagar nada.
///
/// ponytail: duas partidas no mesmo instante podem ver o mesmo resto e uma
/// apagar o socket da outra; o upgrade é uma trava (`flock`) ao lado do socket.
///
/// # Errors
/// `AddrInUse` com servidor vivo, `AlreadyExists` se o caminho não é socket, e
/// os de [`bind`].
pub fn bind_owned(path: &Path) -> io::Result<(UnixListener, SocketFile)> {
    match std::fs::symlink_metadata(path) {
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
        Ok(meta) if !meta.file_type().is_socket() => {
            let msg = "o caminho existe e não é um socket";
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, msg));
        }
        Ok(_) => {
            // Sonda sem bloquear: com o backlog cheio, o servidor também está vivo.
            let probe = Socket::new(Domain::UNIX, Type::STREAM, None)?;
            probe.set_nonblocking(true)?;
            match probe.connect(&SockAddr::unix(path)?) {
                Err(err) if err.kind() == io::ErrorKind::ConnectionRefused => {
                    std::fs::remove_file(path)?;
                }
                Err(err) if err.kind() != io::ErrorKind::WouldBlock => return Err(err),
                _ => {
                    let msg = "outro servidor escuta neste socket";
                    return Err(io::Error::new(io::ErrorKind::AddrInUse, msg));
                }
            }
        }
    }
    let listener = bind(path)?;
    let meta = std::fs::symlink_metadata(path)?;
    let id = (meta.dev(), meta.ino());
    Ok((
        listener,
        SocketFile {
            path: path.to_owned(),
            id,
        },
    ))
}

/// Aceita conexões até `shutdown` resolver; aí para de aceitar, avisa as
/// conexões e espera até `drain` por elas, abortando as que passarem do prazo.
pub async fn serve<H: Handler>(
    listener: UnixListener,
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
                Ok((conn, _)) => {
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

/// Resolve no primeiro SIGINT ou SIGTERM. Os tratadores entram já na chamada
/// (dentro do runtime), não no primeiro poll.
///
/// # Errors
/// Falha ao instalar os tratadores.
pub fn shutdown_signal() -> io::Result<impl Future<Output = ()>> {
    let mut term = signal(SignalKind::terminate())?;
    let mut int = signal(SignalKind::interrupt())?;
    Ok(async move {
        tokio::select! {
            _ = term.recv() => info!("SIGTERM: encerrando"),
            _ = int.recv() => info!("SIGINT: encerrando"),
        }
    })
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
    use std::process::Command;
    use std::time::Instant;

    use serde_json::{json, Value};
    use tokio::io::{duplex, AsyncReadExt};
    use tokio::net::UnixStream;
    use tokio::time::timeout;

    use super::*;
    use crate::frame::kind::*;

    const PATIENCE: Duration = Duration::from_secs(10); // folga para runner lento

    /// `(kind, request_id, payload em JSON)`; payload vazio vira `null`.
    async fn next(conn: &mut impl Conn) -> (u8, u32, Value) {
        let frame = read_frame(conn).await.unwrap().unwrap();
        let body = serde_json::from_slice(&frame.payload).unwrap_or(Value::Null);
        (frame.kind, frame.request_id, body)
    }

    /// Caminho curto: o endereço de um socket Unix cabe em ~100 bytes.
    fn sock(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("scipc-{}-{tag}.sock", std::process::id()))
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

    #[tokio::test]
    async fn sigterm_stops_accepting_and_closes_idle_connections_at_once() {
        // Único teste que faz fork: o `kill` herda os fds até o exec, e um connect
        // de outro teste nesse instante passaria. O `status` espera o filho sair.
        let pid = std::process::id().to_string();
        let kill = |sig| Command::new("kill").args([sig, pid.as_str()]).status();
        // SIGINT antes do 1º poll: o tratador entrou já na chamada, e o processo segue.
        let early = shutdown_signal().unwrap();
        assert!(kill("-INT").unwrap().success());
        timeout(PATIENCE, early).await.unwrap();
        let path = sock("term");
        let listener = bind(&path).unwrap();
        let meta = std::fs::symlink_metadata(&path).unwrap();
        assert!(meta.file_type().is_socket());
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
        assert_eq!(bind(&path).unwrap_err().kind(), io::ErrorKind::AddrInUse);
        let signaled = shutdown_signal().unwrap();
        let server = tokio::spawn(serve(listener, Basic, signaled, Duration::from_secs(60)));
        let mut conn = UnixStream::connect(&path).await.unwrap();
        let ping = encode(PING, 0, 1, b"").unwrap();
        conn.write_all(&ping).await.unwrap();
        assert_eq!(next(&mut conn).await, (PONG, 1, Value::Null));
        assert!(kill("-TERM").unwrap().success());
        // A conexão ociosa recebe o aviso: o servidor sai sem esperar os 60 s.
        timeout(PATIENCE, server).await.unwrap().unwrap();
        assert_eq!(conn.read(&mut [0]).await.unwrap(), 0, "a conexão fecha");
        assert!(UnixStream::connect(&path).await.is_err(), "ainda aceita");
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn bind_owned_clears_a_dead_socket_and_removes_only_its_own() {
        // Resto de uma queda: bind sem listen, então o connect é recusado (e um
        // fork do teste de sinal herdando o fd não muda isso).
        let dead = |path: &Path| {
            let socket = Socket::new(Domain::UNIX, Type::STREAM, None).unwrap();
            socket.bind(&SockAddr::unix(path).unwrap()).unwrap();
        };
        let refused = |path: &Path| bind_owned(path).unwrap_err().kind();
        let path = sock("owned");
        dead(&path);
        let (listener, file) = bind_owned(&path).unwrap();
        assert_eq!(refused(&path), io::ErrorKind::AddrInUse);
        drop((listener, file));
        assert!(!path.exists(), "o socket próprio sai no drop");
        // O que outro pôs no lugar do nosso socket fica.
        let (listener, file) = bind_owned(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, "de outro").unwrap();
        drop((listener, file));
        // Arquivo comum e symlink são recusados, intactos; o symlink não é
        // seguido nem quando aponta para um socket morto.
        assert_eq!(refused(&path), io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "de outro");
        let (link, target) = (sock("link"), sock("target"));
        dead(&target);
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert_eq!(refused(&link), io::ErrorKind::AlreadyExists);
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        for leftover in [link, target, path] {
            std::fs::remove_file(leftover).unwrap();
        }
    }

    /// Ignora o aviso de encerramento: só o prazo da drenagem o derruba.
    struct Stuck;

    impl Handler for Stuck {
        async fn handle<S: Conn>(&self, mut conn: S, _stop: Stop) {
            conn.write_all(b"!").await.unwrap();
            std::future::pending::<()>().await;
        }
    }

    #[tokio::test]
    async fn drain_deadline_aborts_connections_that_ignore_shutdown() {
        for drain in [Duration::ZERO, Duration::from_millis(200)] {
            let path = sock("stuck");
            let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
            let server = tokio::spawn(serve(bind(&path).unwrap(), Stuck, stopped, drain));
            let mut conn = UnixStream::connect(&path).await.unwrap();
            conn.read_exact(&mut [0]).await.unwrap(); // a task da conexão já roda
            let started = Instant::now();
            stop.send(()).unwrap();
            timeout(PATIENCE, server).await.unwrap().unwrap();
            assert!(started.elapsed() >= drain);
            assert_eq!(conn.read(&mut [0]).await.unwrap(), 0, "a conexão fecha");
            std::fs::remove_file(path).unwrap();
        }
    }
}
