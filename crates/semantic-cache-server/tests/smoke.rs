//! Smoke do container final: a imagem (`Dockerfile` da raiz) de pé como o
//! compose a sobe (`deploy/compose/compose.yaml`: volumes nomeados, rede
//! interna, raiz só leitura, sem capabilities), conferida de fora contra o que o
//! contrato da Fase 5 (§3) promete do socket. Ele nasce `0600` com dono 65532
//! na pasta `0700` do volume; do outro lado está o PID 1 do container, como
//! 65532, e o PING responde; o `/health` responde na rede interna, pelo nome do
//! serviço; no SIGTERM o servidor fecha as conexões abertas, sai sozinho com 0
//! e leva o socket; e o socket morto de uma queda (SIGKILL) não impede a
//! partida seguinte.
//!
//! Precisa do Docker, da imagem e de root (o volume e o socket só abrem para o
//! 65532 e para root), então fica `#[ignore]`. O job da imagem
//! (`.github/workflows/docker.yml`) roda assim, e localmente é igual:
//!
//! ```text
//! cargo test --locked --test smoke --no-run   # mostra o executável
//! sudo SEMCACHE_SMOKE_IMAGE=educacao-solidaria/semantic-cache:dev <executável> --ignored --nocapture
//! ```
//!
//! Os outros testes, sem Docker, conferem os instrumentos do smoke.
#![cfg(target_os = "linux")]

use std::io::ErrorKind;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use semantic_cache_ipc::frame::{encode, kind};
use semantic_cache_server::config::Config;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::unix::UCred;
use tokio::net::UnixStream;

/// A tag do contrato (§2), a que o job da imagem constrói.
const IMAGE: &str = "educacao-solidaria/semantic-cache:dev";
/// O cliente HTTP da rede interna (o `wget` do busybox), fixado pelo digest.
const BUSYBOX: &str =
    "busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e";
/// O endurecimento do compose, nos dois containers da rodada.
const HARDENED: &str = "--read-only --cap-drop ALL --security-opt no-new-privileges:true";
/// Até o HEALTHCHECK da imagem (o `--ping` pelo socket) passar.
const STARTUP: Duration = Duration::from_secs(60);
/// Para fechar a conexão, sair depois do sinal ou responder. O servidor do
/// smoke drena por 20 s: no aviso de parada ele fecha em milissegundos, e só um
/// que esperasse o prazo de drenagem passaria daqui.
const PROMPT: Duration = Duration::from_secs(5);
const PING_ID: u32 = 7;

/// `docker <args>`, com os argumentos separados por espaço.
fn run(args: &str) -> std::io::Result<Output> {
    let words = args.split_whitespace();
    Command::new("docker").args(words).output()
}

/// O mesmo, exigindo sucesso: a saída padrão, sem as pontas.
fn docker(args: &str) -> String {
    let out = run(args).expect("docker no PATH");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "docker {args}: {err}");
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

/// O caminho como o `stat -c '%a %u:%g %F'` o mostra, sem seguir symlink.
fn entry(path: &Path) -> std::io::Result<String> {
    let meta = std::fs::symlink_metadata(path)?;
    let file_type = meta.file_type();
    let kind = if file_type.is_socket() {
        "socket"
    } else if file_type.is_dir() {
        "directory"
    } else if file_type.is_symlink() {
        "symbolic link"
    } else {
        "other"
    };
    let mode = meta.mode() & 0o7777;
    Ok(format!("{mode:o} {}:{} {kind}", meta.uid(), meta.gid()))
}

/// Uma rodada: o container do cache, os dois volumes e a rede interna, com
/// nomes próprios. Tudo sai no `Drop`, e numa falha o log do cache vem antes.
struct Stack(String);

impl Stack {
    fn up(image: &str) -> Self {
        let stack = Self(format!("scsmoke-{}", std::process::id()));
        let name = &stack.0;
        docker(&format!("network create --internal {name}-net"));
        docker(&format!(
            "run --detach --name {name} --network {name}-net --network-alias cache {HARDENED} \
             --env SEMCACHE_HTTP_ENABLED=true --env SEMCACHE_HTTP_LISTEN=0.0.0.0:8099 \
             --env SEMCACHE_IPC_DRAIN_TIMEOUT_MS=20000 \
             --volume {name}-ipc:/run/semantic-cache --volume {name}-data:/var/lib/semantic-cache \
             {image}"
        ));
        stack
    }

    /// `docker inspect` do container com o `template` (Go template, sem espaço).
    fn inspect(&self, template: &str) -> String {
        docker(&format!("inspect --format {template} {}", self.0))
    }

    /// A pasta do volume `{nome}-{which}` no host.
    fn volume(&self, which: &str) -> PathBuf {
        let (template, name) = ("{{.Mountpoint}}", &self.0);
        let inspect = format!("volume inspect --format {template} {name}-{which}");
        PathBuf::from(docker(&inspect))
    }

    /// Espera o HEALTHCHECK passar e devolve quanto levou.
    fn healthy(&self) -> Duration {
        let started = Instant::now();
        loop {
            let state = self.inspect("{{.State.Status}}/{{.State.Health.Status}}");
            if state == "running/healthy" {
                return started.elapsed();
            }
            let starting = state == "running/starting" && started.elapsed() < STARTUP;
            assert!(starting, "o cache não ficou saudável: {state}");
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// Espera o container sair, por até `PROMPT`, e devolve o código de saída.
    fn exit_code(&self) -> String {
        let started = Instant::now();
        while self.inspect("{{.State.Status}}") != "exited" {
            assert!(started.elapsed() < PROMPT, "o container não saiu");
            std::thread::sleep(Duration::from_millis(100));
        }
        self.inspect("{{.State.ExitCode}}")
    }

    fn signal(&self, signal: &str) {
        docker(&format!("kill --signal {signal} {}", self.0));
    }

    fn logs(&self) -> String {
        let out = run(&format!("logs {}", self.0));
        let both = out.map(|o| [o.stdout, o.stderr].concat());
        String::from_utf8_lossy(&both.unwrap_or_default()).into_owned()
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("--- log do cache ---\n{}", self.logs());
        }
        let name = &self.0;
        let _ = run(&format!("rm --force {name}"));
        let _ = run(&format!("volume rm --force {name}-ipc {name}-data"));
        let _ = run(&format!("network rm {name}-net"));
    }
}

/// Uma conexão aberta por um PING respondido.
struct Held(UnixStream);

impl Held {
    /// Conecta e manda um PING; só abre com o PONG do pedido, byte a byte
    /// (cabeçalho, id e CRC). Devolve também o processo do outro lado.
    async fn open(path: &Path) -> Result<(Self, UCred), String> {
        let pong = encode(kind::PONG, 0, PING_ID, b"").unwrap();
        let exchange = async {
            let mut conn = UnixStream::connect(path).await?;
            conn.write_all(&encode(kind::PING, 0, PING_ID, b"").unwrap())
                .await?;
            let mut reply = vec![0; pong.len()];
            conn.read_exact(&mut reply).await?;
            Ok::<_, std::io::Error>((conn, reply))
        };
        let (conn, reply) = match tokio::time::timeout(PROMPT, exchange).await {
            Ok(Ok(done)) => done,
            Ok(Err(err)) => return Err(format!("PING: {err}")),
            Err(_) => return Err(format!("PING sem resposta em {PROMPT:?}")),
        };
        if reply != pong {
            return Err(format!("a resposta ao PING não é o PONG: {reply:02x?}"));
        }
        let peer = conn.peer_cred().map_err(|err| err.to_string())?;
        Ok((Self(conn), peer))
    }

    /// Espera o servidor fechar a conexão (EOF) por até `limit`.
    async fn closes_within(&mut self, limit: Duration) -> Result<(), String> {
        let mut byte = [0];
        match tokio::time::timeout(limit, self.0.read(&mut byte)).await {
            Ok(Ok(0)) => Ok(()),
            Ok(Ok(_)) => Err("o servidor mandou dados em vez de fechar".to_owned()),
            Ok(Err(err)) => Err(format!("a conexão caiu com erro: {err}")),
            Err(_) => Err(format!("a conexão seguiu aberta por {limit:?}")),
        }
    }
}

#[tokio::test]
#[ignore = "precisa do Docker, da imagem e de root; o job da imagem roda com --ignored"]
async fn the_final_container_serves_the_socket_and_takes_it_along_on_sigterm() {
    let image = std::env::var("SEMCACHE_SMOKE_IMAGE").unwrap_or_else(|_| IMAGE.to_owned());
    let stack = Stack::up(&image);
    println!("smoke: saudável em {:?}", stack.healthy());
    let (ipc, data) = (stack.volume("ipc"), stack.volume("data"));
    let socket = ipc.join("semantic-cache.sock");
    let (dir, sock) = ("700 65532:65532 directory", "600 65532:65532 socket");
    // §3: os volumes novos com as pastas da imagem, e o socket só do 65532.
    assert_eq!(entry(&ipc).unwrap(), dir);
    assert_eq!(entry(&data).unwrap(), dir);
    assert_eq!(entry(&socket).unwrap(), sock);
    // O PING responde, e quem atende é o PID 1 do container, como 65532.
    let (mut held, peer) = Held::open(&socket).await.unwrap();
    let pid = stack.inspect("{{.State.Pid}}").parse().ok();
    assert_eq!((peer.uid(), peer.gid(), peer.pid()), (65532, 65532, pid));
    // O `/health` na rede interna, pelo nome do serviço, como o studio chega.
    let net = format!("{}-net", stack.0);
    let url = "http://cache:8099/health";
    let wget = format!("run --rm --network {net} --user 65534:65534 {HARDENED} {BUSYBOX}");
    let health = docker(&format!("{wget} wget -q -O - -T 5 {url}"));
    let health: Value = serde_json::from_str(&health).unwrap();
    assert_eq!(health["status"], "ok", "{health}");
    // Sob musl, o alocador segue o jemalloc, com as estatísticas lidas.
    let jemalloc = |line: &str| {
        let event: Value = serde_json::from_str(line).unwrap_or_default();
        event["allocator"] == "jemalloc" && event["heap_bytes"].as_u64() > Some(0)
    };
    let logs = stack.logs();
    assert!(logs.lines().any(jemalloc), "o log não mostra o jemalloc");
    // SIGTERM: a conexão aberta fecha no aviso, o servidor sai sozinho com 0 e
    // leva o socket; a pasta do volume fica.
    stack.signal("TERM");
    held.closes_within(PROMPT).await.unwrap();
    assert_eq!(stack.exit_code(), "0");
    assert_eq!(entry(&socket).unwrap_err().kind(), ErrorKind::NotFound);
    assert_eq!(entry(&ipc).unwrap(), dir);
    // Queda: o SIGKILL deixa o socket morto no volume, e a partida seguinte o
    // troca por um novo.
    docker(&format!("start {}", stack.0));
    stack.healthy();
    stack.signal("KILL");
    assert_eq!(stack.exit_code(), "137");
    assert_eq!(entry(&socket).unwrap(), sock);
    let dead = UnixStream::connect(&socket).await.unwrap_err();
    assert_eq!(dead.kind(), ErrorKind::ConnectionRefused);
    docker(&format!("start {}", stack.0));
    stack.healthy();
    Held::open(&socket).await.unwrap();
    // O `docker stop` (o STOPSIGNAL da imagem e o prazo padrão) faz o mesmo.
    docker(&format!("stop -t 10 {}", stack.0));
    assert_eq!(stack.exit_code(), "0");
    assert_eq!(entry(&socket).unwrap_err().kind(), ErrorKind::NotFound);
}

/// O `docker stop` e o `compose down` (o compose não muda o
/// `stop_grace_period`) mandam SIGKILL 10 s depois do SIGTERM, e um servidor
/// morto deixa o socket no volume: o prazo do servidor depois do sinal
/// (drenagem + 1 s, no `main.rs`) tem de caber antes.
#[test]
fn the_default_shutdown_fits_in_dockers_stop_grace_period() {
    let drain = Config::default().ipc.drain_timeout();
    let deadline = drain + Duration::from_secs(1);
    assert!(deadline < Duration::from_secs(10), "{deadline:?}");
}

#[test]
fn entries_read_like_stat_and_do_not_follow_links() {
    use std::fs::{create_dir, remove_dir_all, set_permissions, Permissions};
    use std::os::unix::fs::PermissionsExt;

    let dir = std::env::temp_dir().join(format!("scsmoke-entry-{}", std::process::id()));
    let _ = remove_dir_all(&dir);
    create_dir(&dir).unwrap();
    set_permissions(&dir, Permissions::from_mode(0o700)).unwrap();
    let (socket, link) = (dir.join("s.sock"), dir.join("link"));
    let _listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    set_permissions(&socket, Permissions::from_mode(0o600)).unwrap();
    std::os::unix::fs::symlink(&socket, &link).unwrap();
    let meta = std::fs::metadata(&dir).unwrap();
    let owner = format!("{}:{}", meta.uid(), meta.gid());
    assert_eq!(entry(&dir).unwrap(), format!("700 {owner} directory"));
    assert_eq!(entry(&socket).unwrap(), format!("600 {owner} socket"));
    assert_eq!(entry(&link).unwrap(), format!("777 {owner} symbolic link"));
    let missing = entry(&dir.join("nada")).unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::NotFound);
    remove_dir_all(&dir).unwrap();
}

/// O instrumento do SIGTERM contra o `serve` de produção: sem o aviso de parada
/// a conexão segue aberta (e o smoke acusaria), e com ele fecha bem antes do
/// prazo de drenagem.
#[tokio::test]
async fn a_held_connection_sees_the_draining_server_close_it() {
    use semantic_cache_ipc::server::{bind, serve, Basic, Limits};

    let path = std::env::temp_dir().join(format!("scsmoke-drain-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let (stop, stopping) = tokio::sync::oneshot::channel::<()>();
    let limits = Limits {
        max_connections: 4,
        idle: Duration::ZERO,
        drain: Duration::from_secs(60),
    };
    let server = tokio::spawn(serve(bind(&path).unwrap(), Basic, stopping, limits));
    let (mut held, peer) = Held::open(&path).await.unwrap();
    assert_eq!(peer.pid(), i32::try_from(std::process::id()).ok());
    let open = held.closes_within(Duration::from_millis(200)).await;
    assert!(open.unwrap_err().contains("seguiu aberta"));
    stop.send(()).unwrap();
    held.closes_within(PROMPT).await.unwrap();
    tokio::time::timeout(PROMPT, server).await.unwrap().unwrap();
    std::fs::remove_file(&path).unwrap();
}

/// O PING do smoke só passa com o PONG do pedido: outro id, um ERROR ou a
/// conexão fechada sem resposta reprovam.
#[tokio::test]
async fn a_ping_only_passes_with_the_pong_of_the_request() {
    let path = std::env::temp_dir().join(format!("scsmoke-ping-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&path);
    for (case, answer) in [
        ("pong de outro id", encode(kind::PONG, 0, PING_ID + 1, b"")),
        ("error", encode(kind::ERROR, 0, PING_ID, b"{}")),
        ("nada", Ok(Vec::new())),
    ] {
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let fake = tokio::spawn(async move {
            let (mut conn, _) = listener.accept().await.unwrap();
            let mut ping = encode(kind::PING, 0, PING_ID, b"").unwrap();
            conn.read_exact(&mut ping).await.unwrap();
            conn.write_all(&answer.unwrap()).await.unwrap();
        });
        let refused = Held::open(&path).await.err();
        assert!(refused.is_some(), "{case}");
        fake.await.unwrap();
        std::fs::remove_file(&path).unwrap();
    }
}
