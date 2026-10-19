//! `Tokenizer::count` não aloca: um alocador global conta as alocações da
//! thread enquanto ela conta tokens.

#![allow(unsafe_code)] // só o alocador de contagem deste teste

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use semantic_cache_tokenizer::Tokenizer;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: repassa cada chamada ao `System` com os mesmos argumentos; só conta
// as da thread atual, num `Cell` de inicialização constante (que não aloca).
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        System.realloc(ptr, layout, size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocations(run: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    run();
    ALLOCATIONS.with(Cell::get) - before
}

/// Os 256 bytes isolados (`XY==`) e junções para o BPE trabalhar: he, ll,
/// hell, hello, " w", " wo", aa, aaaa.
fn tokenizer() -> Tokenizer {
    const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut file = String::new();
    for b in 0..=u8::MAX {
        let char_at = |i: u8| char::from(B64[usize::from(i)]);
        file.extend([char_at(b >> 2), char_at((b & 3) << 4), '=', '=', ' ']);
        file.push_str(&b.to_string());
        file.push('\n');
    }
    file.push_str("aGU= 256\nbGw= 257\naGVsbA== 258\naGVsbG8= 259\nIHc= 260\nIHdv 261\n");
    file.push_str("YWE= 262\nYWFhYQ== 263\n");
    Tokenizer::cl100k_base(file.as_bytes()).unwrap()
}

#[test]
fn count_does_not_allocate_and_matches_encode() {
    let tokenizer = tokenizer();
    let long = "a".repeat(20_000);
    // A primeira contagem da thread dimensiona os buffers do BPE (o pedaço
    // de 20 mil letras é o maior); os textos novos depois disso não alocam,
    // nem os de Unicode que esta thread ainda não viu.
    let _ = tokenizer.count(&long);
    let texts = [
        long.as_str(),
        "hello world",
        "Olá, mundo! ação 🎉 日本語 e\u{301} x'ſ ٣٤٥",
        "a   b\n\n  \t x \r\n\u{3000}fim \n ",
        "1234567 (hello) $5 'll 'RE !!!\n",
        "",
    ];
    for text in texts {
        let mut count = 0;
        assert_eq!(allocations(|| count = tokenizer.count(text)), 0, "{text:?}");
        assert_eq!(count, tokenizer.encode(text).len(), "{text:?}");
    }
    // O controle: o mesmo alocador vê o `encode`, que monta um vetor.
    assert!(allocations(|| drop(tokenizer.encode("hello world"))) > 0);
}
