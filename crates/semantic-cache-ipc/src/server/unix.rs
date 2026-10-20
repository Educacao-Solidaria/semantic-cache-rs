//! Socket Unix do servidor IPC: 0600 antes da 1ª conexão, o que fazer com o
//! que já está no caminho, e SIGINT/SIGTERM.

use std::future::Future;
use std::io;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use socket2::{Domain, SockAddr, Socket, Type};
use tokio::net::{UnixListener, UnixStream};
use tokio::signal::unix::{signal, SignalKind};
use tracing::info;

use super::Listener;

impl Listener for UnixListener {
    type Conn = UnixStream;

    async fn accept(&mut self) -> io::Result<UnixStream> {
        UnixListener::accept(self).await.map(|(conn, _)| conn)
    }
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

/// Se o diretório de `path` aceita escrita de outros usuários (ex.: `/tmp`):
/// ali quem chegar antes toma o caminho. Em produção, o socket fica num
/// diretório privado do serviço (`docs/ipc-v1.md`, "Segurança do socket").
#[must_use]
pub fn in_shared_dir(path: &Path) -> bool {
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    std::fs::metadata(dir).is_ok_and(|meta| meta.permissions().mode() & 0o022 != 0)
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

#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::time::{Duration, Instant};

    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::time::timeout;

    use super::*;
    use crate::frame::{encode, kind::*};
    use crate::server::tests::{limits, next, PATIENCE};
    use crate::server::{serve, Basic, Conn, Handler, Stop};

    /// Caminho curto: o endereço de um socket Unix cabe em ~100 bytes.
    fn sock(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("scipc-{}-{tag}.sock", std::process::id()))
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
        let patient = limits(Duration::from_secs(60));
        let server = tokio::spawn(serve(listener, Basic, signaled, patient));
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

    #[test]
    fn a_dir_others_can_write_is_shared() {
        assert!(in_shared_dir(Path::new("/tmp/semantic-cache.sock")));
        let private = std::env::temp_dir().join(format!("scipc-{}-own", std::process::id()));
        std::fs::create_dir(&private).unwrap();
        std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(!in_shared_dir(&private.join("s.sock")));
        std::fs::remove_dir(private).unwrap();
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
            let listener = bind(&path).unwrap();
            let server = tokio::spawn(serve(listener, Stuck, stopped, limits(drain)));
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
