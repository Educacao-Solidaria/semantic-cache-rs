//! As métricas globais do IPC pelo `serve` e pelo `Basic` de produção. Binário
//! próprio com UM teste: os testes da lib correm em paralelo e atendem
//! pedidos, o que sujaria as contagens.

use std::io;
use std::time::{Duration, Instant};

use semantic_cache_ipc::frame::{discarded_frames, encode, kind, read_frame};
use semantic_cache_ipc::metrics::{connections, lookups, request_seconds, requests};
use semantic_cache_ipc::server::{serve, Basic, Limits, Listener};
use tokio::io::{duplex, AsyncWriteExt, DuplexStream};
use tokio::sync::{mpsc, oneshot};

const PATIENCE: Duration = Duration::from_secs(10); // folga para runner lento

/// Canal = listener: cada `DuplexStream` mandado é uma conexão aceita.
struct Chan(mpsc::Receiver<DuplexStream>);

impl Listener for Chan {
    type Conn = DuplexStream;
    async fn accept(&mut self) -> io::Result<DuplexStream> {
        match self.0.recv().await {
            Some(conn) => Ok(conn),
            None => std::future::pending().await,
        }
    }
}

async fn until(what: &str, done: impl Fn() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < PATIENCE, "{what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn count(kind: &str, outcome: &str) -> u64 {
    let mut all = requests();
    all.find(|&(k, o, _)| (k, o) == (kind, outcome)).unwrap().2
}

#[tokio::test]
async fn requests_connections_and_discarded_frames_are_counted() {
    let (dial, accepted) = mpsc::channel(4);
    let idle = Duration::from_secs(2); // a janela para ver a conexão parada
    let limits = Limits {
        max_connections: 4,
        idle,
        drain: PATIENCE,
    };
    let (stop, stopping) = oneshot::channel::<()>();
    let server = tokio::spawn(serve(Chan(accepted), Basic, stopping, limits));
    let ping = |id| encode(kind::PING, 0, id, b"").unwrap();

    // PING e STATS: ok; LOOKUP (sem store) e kind desconhecido: ERROR. O frame
    // com o CRC trocado é descartado (não é pedido) e fecha a conexão.
    let (mut client, conn) = duplex(1 << 16);
    dial.send(conn).await.unwrap();
    let mut batch = ping(1);
    for (k, id) in [(kind::STATS_REQ, 2), (kind::LOOKUP_REQ, 3), (0x55, 4)] {
        batch.extend(encode(k, 0, id, b"{}").unwrap());
    }
    batch.extend(ping(5));
    *batch.last_mut().unwrap() ^= 1;
    client.write_all(&batch).await.unwrap();
    for id in 1..=5 {
        let reply = read_frame(&mut client).await.unwrap().unwrap();
        assert_eq!(reply.request_id, id);
    }
    assert!(read_frame(&mut client).await.unwrap().is_none()); // fechou
    let answered = [("ping", "ok"), ("stats", "ok"), ("lookup", "error")];
    assert!(answered.iter().all(|&(k, o)| count(k, o) == 1));
    assert_eq!(count("unknown", "error"), 1);
    let all: u64 = requests().map(|(.., n)| n).sum();
    assert_eq!((all, discarded_frames()), (4, 1));
    assert!(lookups().all(|(_, n)| n == 0)); // LOOKUP sem busca não é lookup
    until("a conexão fechada ainda conta", || connections() == 0).await;

    // O par que sai antes da resposta: cancelado.
    let (mut gone, conn) = duplex(1 << 16);
    dial.send(conn).await.unwrap();
    gone.write_all(&ping(6)).await.unwrap();
    drop(gone);
    until("ping cancelado", || count("ping", "canceled") == 1).await;
    until("a conexão cancelada ainda conta", || connections() == 0).await;

    // O par que não lê: a resposta (84 bytes) não cabe no buffer de 32 e para
    // no prazo de ociosidade.
    let (mut stuck, conn) = duplex(32);
    dial.send(conn).await.unwrap();
    let stats = encode(kind::STATS_REQ, 0, 7, b"").unwrap();
    stuck.write_all(&stats).await.unwrap();
    until("a conexão parada não conta", || connections() == 1).await;
    until("stats no prazo", || count("stats", "timeout") == 1).await;
    until("a conexão vencida ainda conta", || connections() == 0).await;

    // A latência vai junto: 2 PING (ok e cancelado) e 2 STATS, o parado no +Inf.
    let latency = |kind| request_seconds().find(|&(k, _)| k == kind).unwrap().1;
    assert_eq!(latency("ping").snapshot().0[8], 2);
    let (stats, seconds) = latency("stats").snapshot();
    assert!(stats[7] <= 1 && stats[8] == 2, "{stats:?}");
    assert!(seconds >= 2.0, "o parado esperou o prazo: {seconds}");

    stop.send(()).unwrap();
    let ended = tokio::time::timeout(PATIENCE, server).await;
    ended.unwrap().unwrap();
}
