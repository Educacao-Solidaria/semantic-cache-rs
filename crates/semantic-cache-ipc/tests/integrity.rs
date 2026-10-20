//! Frames hostis contra o estado do processo: o contador de descartados é
//! global, e um alocador soma os bytes pedidos pela thread. Por isso um teste
//! só neste binário.

#![allow(unsafe_code)] // só o alocador de contagem deste teste

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::ErrorKind;

use semantic_cache_ipc::frame::{
    discarded_frames, encode, kind, read_frame, Frame, FrameError, HEADER_LEN, MAX_PAYLOAD,
};

thread_local! {
    static REQUESTED: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: repassa cada chamada ao `System` com os mesmos argumentos; só soma os
// bytes pedidos pela thread atual, num `Cell` de inicialização constante (que
// não aloca). O `alloc_zeroed` padrão passa pelo `alloc`.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        REQUESTED.with(|n| n.set(n.get() + layout.size()));
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        REQUESTED.with(|n| n.set(n.get() + size));
        System.realloc(ptr, layout, size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// O `read_frame` de `bytes` e quantos bytes ele pediu ao alocador.
fn read(bytes: &[u8]) -> (Result<Option<Frame>, FrameError>, usize) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async {
        let before = REQUESTED.with(Cell::get);
        let frame = read_frame(&mut &bytes[..]).await;
        (frame, REQUESTED.with(Cell::get) - before)
    })
}

#[test]
fn corrupt_frames_are_counted_and_a_lying_length_costs_only_what_arrived() {
    let ping = encode(kind::PING, 0, 1, b"").unwrap();
    let with_len = |len: u32, tail: &[u8]| {
        let mut frame = ping[..HEADER_LEN].to_vec();
        frame[12..].copy_from_slice(&len.to_be_bytes());
        frame.extend_from_slice(tail);
        frame
    };
    let before = discarded_frames();
    // Acima do teto: recusado antes de alocar qualquer coisa.
    let (frame, bytes) = read(&with_len(MAX_PAYLOAD + 1, b""));
    assert!(
        matches!(frame, Err(FrameError::TooLarge(1, _))),
        "{frame:?}"
    );
    assert_eq!(bytes, 0);
    // Promete o teto (16 MiB) e manda 10 bytes: pesa a reserva inicial (64 KiB)
    // e o erro, não os 16 MiB prometidos.
    let (frame, bytes) = read(&with_len(MAX_PAYLOAD, &[b'x'; 10]));
    let eof = |e: &std::io::Error| e.kind() == ErrorKind::UnexpectedEof;
    assert!(
        matches!(&frame, Err(FrameError::Io(e)) if eof(e)),
        "{frame:?}"
    );
    assert!(bytes < 1 << 20, "{bytes} bytes por 10 que chegaram");
    // CRC trocado e grande demais contam; o cortado no meio e o íntegro, não.
    let mut flipped = ping.clone();
    flipped[HEADER_LEN + 3] ^= 1;
    assert!(matches!(read(&flipped).0, Err(FrameError::BadFrame(1, _))));
    assert!(matches!(read(&ping).0, Ok(Some(_))));
    assert_eq!(discarded_frames() - before, 2);
}
