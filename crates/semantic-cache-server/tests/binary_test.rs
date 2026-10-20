//! O binário de verdade, num processo filho: limpa o resto de uma queda no
//! caminho do socket, atende PING, recusa uma segunda instância sem apagar o
//! socket, e no SIGTERM sai com 0 e remove o socket que criou.
#![cfg(unix)]

use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use semantic_cache_ipc::frame::{encode, kind, read_frame};
use tokio::io::AsyncWriteExt;
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

/// PING pelo socket, esperando o binário abri-lo.
async fn ping(socket: &Path, id: u32) {
    let started = Instant::now();
    let mut conn = loop {
        match UnixStream::connect(socket).await {
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

#[tokio::test]
async fn binary_clears_a_dead_socket_refuses_a_twin_and_removes_its_own() {
    let socket = std::env::temp_dir().join(format!("scs-bin-{}.sock", std::process::id()));
    // Resto de uma queda: o arquivo do socket ficou, e ninguém escuta nele.
    drop(std::os::unix::net::UnixListener::bind(&socket).unwrap());
    let mut server = Server::start(&socket);
    ping(&socket, 1).await;
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
}
