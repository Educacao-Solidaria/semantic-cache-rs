//! O binário de verdade, num processo filho: limpa o resto de uma queda no
//! caminho do socket, atende PING, recusa uma segunda instância sem apagar o
//! socket, e no SIGTERM sai com 0 e remove o socket que criou; e a sonda
//! `--ping` do HEALTHCHECK da imagem. No Windows, o mesmo pelo named pipe,
//! menos o sinal.

use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use semantic_cache_ipc::frame::{encode, kind, read_frame};
use tokio::io::AsyncWriteExt;
#[cfg(windows)]
use tokio::net::windows::named_pipe::ClientOptions;
#[cfg(unix)]
use tokio::net::UnixStream;

const PATIENCE: Duration = Duration::from_secs(10); // folga para runner lento

/// O processo do binário; morto no `Drop` se o teste falhar no meio.
struct Server(Child);

impl Server {
    fn start(socket: &Path) -> Self {
        let mut bin = Command::new(env!("CARGO_BIN_EXE_semantic-cache-server"));
        let bin = bin
            .env("SEMCACHE_IPC_SOCKET_PATH", socket)
            .stdout(Stdio::null());
        Self(bin.stderr(Stdio::null()).spawn().unwrap())
    }

    fn wait(&mut self) -> ExitStatus {
        let started = Instant::now();
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return status;
            }
            assert!(started.elapsed() < PATIENCE, "o binário não saiu");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// PING pelo socket (ou pelo pipe), esperando o binário abri-lo.
async fn ping(socket: &Path, id: u32) {
    let started = Instant::now();
    let mut conn = loop {
        #[cfg(unix)]
        let opened = UnixStream::connect(socket).await;
        #[cfg(windows)]
        let opened = ClientOptions::new().open(socket);
        match opened {
            Ok(conn) => break conn,
            Err(err) => assert!(started.elapsed() < PATIENCE, "socket fechado: {err}"),
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    conn.write_all(&encode(kind::PING, 0, id, b"").unwrap())
        .await
        .unwrap();
    let pong = read_frame(&mut conn).await.unwrap().unwrap();
    assert_eq!((pong.kind, pong.request_id), (kind::PONG, id));
}

/// A sonda do HEALTHCHECK da imagem: `--ping` do binário contra `socket`.
fn probe(socket: &Path) -> bool {
    let mut bin = Command::new(env!("CARGO_BIN_EXE_semantic-cache-server"));
    bin.arg("--ping").env("SEMCACHE_IPC_SOCKET_PATH", socket);
    bin.stdout(Stdio::null()).stderr(Stdio::null());
    Server(bin.spawn().unwrap()).wait().success()
}

#[cfg(unix)]
#[tokio::test]
async fn binary_clears_a_dead_socket_refuses_a_twin_and_removes_its_own() {
    use semantic_cache_ipc::frame::HEADER_LEN;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;

    let socket = std::env::temp_dir().join(format!("scs-bin-{}.sock", std::process::id()));
    // A sonda só passa com o PONG do pedido: um servidor de mentira que lê o PING
    // e responde outro id, outro kind ou nada a reprova.
    let pong = |id| encode(kind::PONG, 0, id, b"").unwrap();
    let error = encode(kind::ERROR, 0, 1, b"{}").unwrap();
    for (case, answer, passes) in [
        ("pong", pong(1), true),
        ("pong de outro id", pong(2), false),
        ("error", error, false),
        ("nada", vec![], false),
    ] {
        let fake = UnixListener::bind(&socket).unwrap();
        let fake = std::thread::spawn(move || {
            let (mut conn, _) = fake.accept().unwrap();
            conn.read_exact(&mut [0; HEADER_LEN + 4]).unwrap();
            conn.write_all(&answer).unwrap();
        });
        assert_eq!(probe(&socket), passes, "{case}");
        // Uma sonda que nem conectou deixaria o `accept` preso: esta conexão o
        // solta, e o servidor de mentira falha no `read_exact` em vez de travar.
        let _ = std::os::unix::net::UnixStream::connect(&socket);
        fake.join().unwrap();
        std::fs::remove_file(&socket).unwrap();
    }
    // Resto de uma queda: o arquivo do socket ficou, e ninguém escuta nele. Antes
    // de cair, ele aceita e não responde: a sonda desiste no prazo dela.
    let mute = UnixListener::bind(&socket).unwrap();
    assert!(!probe(&socket), "a sonda passou sem resposta");
    drop(mute);
    let mut server = Server::start(&socket);
    ping(&socket, 1).await;
    assert!(probe(&socket), "a sonda falhou com o servidor no ar");
    let twin = Server::start(&socket).wait();
    assert!(!twin.success(), "a segunda instância subiu");
    ping(&socket, 2).await;
    let pid = server.0.id().to_string();
    assert!(Command::new("kill")
        .args(["-TERM", &pid])
        .status()
        .unwrap()
        .success());
    assert!(server.wait().success());
    assert!(!socket.exists(), "o socket ficou para trás");
    assert!(!probe(&socket), "a sonda passou sem servidor");
}

/// O Ctrl+Break pede um console compartilhado, que o runner não garante: aqui o
/// filho morre no `Drop`, e o encerramento limpo fica com o teste Unix.
#[cfg(windows)]
#[tokio::test]
async fn binary_serves_the_named_pipe_and_refuses_a_twin() {
    let pipe = format!(r"\\.\pipe\scs-bin-{}", std::process::id());
    let pipe = Path::new(&pipe);
    let _server = Server::start(pipe);
    ping(pipe, 1).await;
    assert!(probe(pipe), "a sonda falhou com o servidor no ar");
    let twin = Server::start(pipe).wait();
    assert!(!twin.success(), "a segunda instância subiu");
    ping(pipe, 2).await;
}
