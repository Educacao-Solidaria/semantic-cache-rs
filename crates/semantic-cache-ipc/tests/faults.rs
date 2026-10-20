//! Clientes que cortam o socket no meio: do cabeçalho, do payload, do stream,
//! e o half-close. O servidor não entra em pânico, a task de cada conexão
//! termina (a única vaga volta) e os descritores voltam ao que eram. Um teste
//! só neste binário: o hook de pânico e a contagem de `/proc/self/fd` valem
//! para o processo inteiro.
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use semantic_cache_ipc::frame::{encode, kind, read_frame};
use semantic_cache_ipc::server::{bind, serve, Basic, Conn, Handler, Limits, Stop};
use semantic_cache_ipc::stream::write_stream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::time::timeout;

const PATIENCE: Duration = Duration::from_secs(10); // folga para runner lento

static PANICKED: AtomicBool = AtomicBool::new(false);

/// PONG a cada PING; a um `LOOKUP_STREAM_REQ`, 1 MiB em chunks de 64 bytes,
/// para o corte pegar o servidor no meio da escrita.
struct Streaming(String);

impl Handler for Streaming {
    async fn handle<S: Conn>(&self, mut conn: S, _stop: Stop) {
        while let Ok(Some(frame)) = read_frame(&mut conn).await {
            let id = frame.request_id;
            let sent = if frame.kind == kind::LOOKUP_STREAM_REQ {
                write_stream(&mut conn, id, Some(&self.0), 1.0, 0, 64).await
            } else {
                let pong = encode(kind::PONG, 0, id, b"").unwrap();
                conn.write_all(&pong).await.map_err(Into::into)
            };
            if sent.is_err() {
                return; // o cliente sumiu
            }
        }
    }
}

fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

/// PING até ser atendido. Com uma vaga só, prova que a conexão anterior acabou.
async fn served(path: &Path) {
    let started = Instant::now();
    loop {
        let mut conn = UnixStream::connect(path).await.unwrap();
        let _ = conn
            .write_all(&encode(kind::PING, 0, 9, b"").unwrap())
            .await;
        let reply = timeout(PATIENCE, read_frame(&mut conn)).await.unwrap();
        if matches!(reply, Ok(Some(ref f)) if f.kind == kind::PONG) {
            return;
        }
        assert!(started.elapsed() < PATIENCE, "a vaga não voltou: {reply:?}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Espera os descritores voltarem a `base`: cada task fecha o seu socket ao
/// terminar (e o tokio, ao processar o fechamento).
async fn settled(base: usize) {
    let started = Instant::now();
    while open_fds() != base {
        let open = open_fds();
        assert!(
            started.elapsed() < PATIENCE,
            "{open} descritores; eram {base}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// O que o cliente faz antes de sumir.
#[derive(Debug, Clone, Copy)]
enum Cut {
    /// 7 dos 16 bytes do cabeçalho e fecha.
    Header,
    /// Cabeçalho que promete 58 bytes, 10 deles e fecha.
    Payload,
    /// 7 bytes do cabeçalho e half-close: o servidor fecha sem responder.
    HalfCloseMidFrame,
    /// PING inteiro e half-close: o PONG chega, e depois o fim.
    HalfCloseAfterRequest,
    /// Pede o stream, lê 2 chunks e fecha com o resto chegando.
    MidStream,
}

/// O que ainda chega até o servidor fechar.
async fn rest(conn: &mut UnixStream) -> Vec<u8> {
    let mut rest = Vec::new();
    let read = timeout(PATIENCE, conn.read_to_end(&mut rest)).await;
    read.expect("o servidor não fechou").unwrap();
    rest
}

async fn cut(path: &Path, how: Cut) {
    let mut conn = UnixStream::connect(path).await.unwrap();
    let ping = encode(kind::PING, 0, 1, b"").unwrap();
    match how {
        Cut::Header => conn.write_all(&ping[..7]).await.unwrap(),
        Cut::Payload => {
            let lookup = encode(kind::LOOKUP_REQ, 0, 2, &[b' '; 58]).unwrap();
            conn.write_all(&lookup[..26]).await.unwrap();
        }
        Cut::HalfCloseMidFrame => {
            conn.write_all(&ping[..7]).await.unwrap();
            conn.shutdown().await.unwrap();
            assert_eq!(rest(&mut conn).await, b"", "{how:?}");
        }
        Cut::HalfCloseAfterRequest => {
            conn.write_all(&ping).await.unwrap();
            conn.shutdown().await.unwrap();
            let rest = rest(&mut conn).await;
            let reply = read_frame(&mut rest.as_slice()).await.unwrap().unwrap();
            assert_eq!((reply.kind, rest.len()), (kind::PONG, 20), "{how:?}");
        }
        Cut::MidStream => {
            let body = br#"{"prompt":"p","embedding":[0.5],"threshold":0.5}"#;
            conn.write_all(&encode(kind::LOOKUP_STREAM_REQ, 0, 3, body).unwrap())
                .await
                .unwrap();
            for _ in 0..2 {
                let chunk = timeout(PATIENCE, read_frame(&mut conn)).await.unwrap();
                assert_eq!(chunk.unwrap().unwrap().kind, kind::STREAM_CHUNK);
            }
        }
    }
}

#[tokio::test]
async fn abrupt_disconnects_never_panic_free_the_slot_and_leak_no_descriptor() {
    std::panic::set_hook(Box::new(|info| {
        PANICKED.store(true, Ordering::SeqCst);
        eprintln!("{info}");
    }));
    let (max_connections, idle, drain) = (1, Duration::ZERO, Duration::from_secs(1));
    let limits = Limits {
        max_connections,
        idle,
        drain,
    };
    let sock = |tag| -> PathBuf {
        std::env::temp_dir().join(format!("scflt-{}-{tag}.sock", std::process::id()))
    };
    let basic = [
        Cut::Header,
        Cut::Payload,
        Cut::HalfCloseMidFrame,
        Cut::HalfCloseAfterRequest,
    ];
    let servers = [
        (sock("basic"), &basic[..], false),
        (sock("stream"), &[Cut::MidStream][..], true),
    ];
    for (path, cuts, streaming) in servers {
        let listener = bind(&path).unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let server = if streaming {
            tokio::spawn(serve(
                listener,
                Streaming("x".repeat(1 << 20)),
                stopped,
                limits,
            ))
        } else {
            tokio::spawn(serve(listener, Basic, stopped, limits))
        };
        let base = open_fds();
        for &how in cuts {
            for _ in 0..3 {
                cut(&path, how).await;
                served(&path).await;
            }
        }
        settled(base).await;
        stop.send(()).unwrap();
        timeout(PATIENCE, server).await.unwrap().unwrap();
        std::fs::remove_file(path).unwrap();
    }
    assert!(!PANICKED.load(Ordering::SeqCst), "algum pânico (stderr)");
}
