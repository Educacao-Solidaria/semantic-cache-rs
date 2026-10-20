//! Named pipe do Windows (`\\.\pipe\semantic-cache` no contrato): o mesmo IPC
//! v1 do socket Unix, só local, só do dono, e Ctrl+C/Ctrl+Break.

use std::ffi::{OsStr, OsString};
use std::future::Future;
use std::io;

use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};
use tokio::signal::windows::{ctrl_break, ctrl_c};
use tracing::info;
use windows_sys::Win32::Foundation::LocalFree;
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;

use super::Listener;

/// DACL do pipe: protegido (não herda) e com acesso total só ao dono, como o
/// 0600 do socket. O padrão do Windows também dá leitura a Everyone e ao anônimo.
const OWNER_ONLY: &str = "D:P(A;;FA;;;OW)";

#[allow(clippy::cast_possible_truncation)] // 24 bytes
const ATTRIBUTES_LEN: u32 = std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32;

/// O pipe aberto pelo [`bind_pipe`]. Sempre há uma instância à espera do
/// próximo cliente: o nome nunca fica livre para outro processo tomar.
#[derive(Debug)]
pub struct PipeListener {
    name: OsString,
    next: NamedPipeServer,
}

/// Cria o pipe `name` (ex.: `\\.\pipe\semantic-cache`), sem cliente remoto.
///
/// # Errors
/// `AddrInUse` se o nome já existe (outro servidor, ou alguém que o tomou
/// antes) ou o acesso a ele é negado; nome inválido.
pub fn bind_pipe(name: impl AsRef<OsStr>) -> io::Result<PipeListener> {
    let name = name.as_ref().to_owned();
    let next = create(&name, true).map_err(|err| match err.kind() {
        // FILE_FLAG_FIRST_PIPE_INSTANCE com o nome já criado.
        io::ErrorKind::PermissionDenied => {
            let msg = "o pipe já existe ou o acesso a ele foi negado";
            io::Error::new(io::ErrorKind::AddrInUse, msg)
        }
        _ => err,
    })?;
    Ok(PipeListener { name, next })
}

impl Listener for PipeListener {
    type Conn = NamedPipeServer;

    async fn accept(&mut self) -> io::Result<NamedPipeServer> {
        let connected = self.next.connect().await;
        // A instância seguinte nasce antes de esta sair, e uma que falhou no
        // `connect` é trocada também. Se a criação falha, a conectada fica em
        // `next` e o próximo `accept` a devolve (o `connect` dela volta Ok).
        let next = create(&self.name, false)?;
        let conn = std::mem::replace(&mut self.next, next);
        connected.map(|()| conn)
    }
}

/// Uma instância do pipe com o [`OWNER_ONLY`].
#[allow(unsafe_code)]
fn create(name: &OsStr, first: bool) -> io::Result<NamedPipeServer> {
    let sddl: Vec<u16> = OWNER_ONLY.encode_utf16().chain([0]).collect();
    let mut descriptor = std::ptr::null_mut();
    // SAFETY: `sddl` termina em 0 e vive até o fim da chamada, que aloca o
    // descritor (liberado abaixo).
    let converted = unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            std::ptr::null_mut(),
        )
    };
    if converted == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut attributes = SECURITY_ATTRIBUTES {
        nLength: ATTRIBUTES_LEN,
        lpSecurityDescriptor: descriptor,
        bInheritHandle: 0,
    };
    let mut options = ServerOptions::new();
    options
        .first_pipe_instance(first)
        .reject_remote_clients(true);
    let attributes = std::ptr::from_mut(&mut attributes).cast();
    // SAFETY: `attributes` aponta para um SECURITY_ATTRIBUTES válido durante a
    // chamada; o sistema copia o descritor.
    let created = unsafe { options.create_with_security_attributes_raw(name, attributes) };
    // SAFETY: o descritor veio da conversão acima e não é usado depois.
    unsafe { LocalFree(descriptor) };
    created
}

/// Resolve no primeiro Ctrl+C ou Ctrl+Break (o que um gerenciador de processos
/// manda por `GenerateConsoleCtrlEvent`). Os tratadores entram já na chamada.
///
/// # Errors
/// Falha ao instalar os tratadores.
pub fn shutdown_signal() -> io::Result<impl Future<Output = ()>> {
    let (mut int, mut brk) = (ctrl_c()?, ctrl_break()?);
    Ok(async move {
        tokio::select! {
            _ = int.recv() => info!("Ctrl+C: encerrando"),
            _ = brk.recv() => info!("Ctrl+Break: encerrando"),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::os::windows::io::{AsRawHandle, RawHandle};
    use std::time::{Duration, Instant};

    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeClient};
    use tokio::time::timeout;

    use super::*;
    use crate::frame::{encode, kind::*};
    use crate::server::tests::{limits, next, PATIENCE};
    use crate::server::{serve, Basic};

    fn pipe(tag: &str) -> String {
        format!(r"\\.\pipe\scipc-{}-{tag}", std::process::id())
    }

    /// Abre o pipe, esperando enquanto não há instância livre.
    async fn open(name: &str) -> NamedPipeClient {
        let started = Instant::now();
        loop {
            match ClientOptions::new().open(name) {
                Ok(conn) => return conn,
                Err(err) => assert!(started.elapsed() < PATIENCE, "pipe fechado: {err}"),
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn pipe_serves_concurrent_clients_refuses_a_twin_and_goes_away() {
        let name = pipe("serve");
        let listener = bind_pipe(&name).unwrap();
        let twin = bind_pipe(&name).unwrap_err();
        assert_eq!(twin.kind(), io::ErrorKind::AddrInUse);
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let patient = limits(Duration::from_secs(60));
        let server = tokio::spawn(serve(listener, Basic, stopped, patient));
        // Duas conexões abertas ao mesmo tempo: cada accept deixa outra instância.
        let (mut a, mut b) = (open(&name).await, open(&name).await);
        for (conn, id) in [(&mut a, 1), (&mut b, 2)] {
            conn.write_all(&encode(PING, 0, id, b"").unwrap())
                .await
                .unwrap();
            assert_eq!(next(conn).await, (PONG, id, Value::Null));
        }
        stop.send(()).unwrap();
        // As conexões ociosas recebem o aviso: o servidor sai sem esperar os 60 s.
        timeout(PATIENCE, server).await.unwrap().unwrap();
        for conn in [&mut a, &mut b] {
            assert_eq!(conn.read(&mut [0]).await.unwrap(), 0, "a conexão fecha");
        }
        drop((a, b));
        // O mio só fecha o handle quando o reator processa o cancelamento do
        // I/O pendente: até lá o `open` pode ver o pipe ocupado.
        let started = Instant::now();
        loop {
            match ClientOptions::new().open(&name) {
                Err(err) if err.kind() == io::ErrorKind::NotFound => break,
                other => assert!(started.elapsed() < PATIENCE, "o pipe ficou: {other:?}"),
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// O DACL do objeto de `handle`, em SDDL.
    #[allow(unsafe_code)]
    fn dacl(handle: RawHandle) -> String {
        use std::ptr::null_mut;
        use windows_sys::Win32::Security::Authorization::{
            ConvertSecurityDescriptorToStringSecurityDescriptorW, GetSecurityInfo, SE_FILE_OBJECT,
        };
        use windows_sys::Win32::Security::DACL_SECURITY_INFORMATION as DACL;
        let (mut descriptor, mut text, mut len) = (null_mut(), null_mut(), 0);
        // SAFETY: `handle` está aberto; a chamada aloca o descritor (liberado
        // abaixo). Só o DACL é pedido: dono, grupo e SACL ficam nulos.
        let failed = unsafe {
            GetSecurityInfo(
                handle,
                SE_FILE_OBJECT,
                DACL,
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
                &mut descriptor,
            )
        };
        assert_eq!(failed, 0, "GetSecurityInfo");
        // SAFETY: `descriptor` é válido; a chamada aloca o texto (liberado abaixo).
        let converted = unsafe {
            ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor,
                SDDL_REVISION_1,
                DACL,
                &mut text,
                &mut len,
            )
        };
        assert_ne!(
            converted, 0,
            "ConvertSecurityDescriptorToStringSecurityDescriptorW"
        );
        // SAFETY: `text` tem `len` u16, contando o 0 do fim.
        let wide = unsafe { std::slice::from_raw_parts(text, usize::try_from(len).unwrap()) };
        let sddl = String::from_utf16_lossy(wide);
        // SAFETY: os dois vieram das chamadas acima e não são usados depois.
        unsafe {
            LocalFree(text.cast());
            LocalFree(descriptor);
        }
        sddl.trim_end_matches('\0').to_owned()
    }

    #[tokio::test]
    async fn pipe_dacl_grants_access_only_to_the_owner() {
        let listener = bind_pipe(pipe("dacl")).unwrap();
        // Uma entrada só, sem herança: acesso total ao dono (OW). O padrão do
        // Windows teria SYSTEM, administradores, Everyone (WD) e anônimo (AN).
        assert_eq!(dacl(listener.next.as_raw_handle()), "D:P(A;;FA;;;OW)");
    }
}
