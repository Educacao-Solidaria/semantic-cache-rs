//! Alocações por operação no caminho do LOOKUP, contadas pelo dhat no modo de
//! testes. O alocador do dhat só existe neste binário de teste (dev-dependency
//! do testkit): nada disso chega ao binário do servidor. Binário próprio com UM
//! teste: o dhat conta o processo inteiro, e testes em paralelo sujariam a conta.
//!
//! Para o perfil completo, com a pilha de cada alocação, troque o `.testing()`
//! por `dhat::Profiler::new_heap()` e abra o `dhat-heap.json` no DHAT viewer.

use std::hint::black_box;

use semantic_cache_ipc::frame::{encode, kind, read_frame, Frame};
use semantic_cache_simd::QuantizedVector;
use semantic_cache_testkit::ipc::SearchCache;
use semantic_cache_testkit::stress::{FlatF32, FlatI8, VectorIndex};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Blocos que cada chamada de `op` aloca, depois de uma de aquecimento; a
/// conta tem de ser a mesma nas 10 seguintes. Medir fora do `dhat::assert_eq!`:
/// ele avalia com o lock do dhat tomado (o `HeapStats::get` travaria).
fn blocks(mut op: impl FnMut()) -> u64 {
    op();
    let before = dhat::HeapStats::get().total_blocks;
    for _ in 0..10 {
        op();
    }
    let total = dhat::HeapStats::get().total_blocks - before;
    assert_eq!(
        total % 10,
        0,
        "conta que varia entre chamadas: {total} em 10"
    );
    total / 10
}

/// O vetor `axis` da base canônica de R^dim.
fn axis(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    v[axis] = 1.0;
    v
}

/// Um cache de `dim` dimensões com dois vetores, e os `LOOKUP_REQ` que acertam o
/// primeiro (hit) e que não chegam ao limiar de nenhum (miss, cosseno 0,707).
fn lookups<I: VectorIndex + Default>(dim: usize) -> (u64, u64) {
    let mut index = I::default();
    index.insert(&axis(dim, 0));
    index.insert(&axis(dim, 1));
    let texts = vec!["resposta curta".to_owned(), "outra".to_owned()];
    let cache = SearchCache::new(index, dim, texts);
    let request = |embedding: Vec<f32>| {
        let body = serde_json::json!({ "prompt": "p", "embedding": embedding, "threshold": 0.9 });
        let payload = body.to_string().into_bytes();
        Frame {
            kind: kind::LOOKUP_REQ,
            request_id: 1,
            payload,
        }
    };
    let (hit, mut miss) = (request(axis(dim, 0)), axis(dim, 0));
    miss[1] = 1.0;
    let miss = request(miss);
    let reply = cache.answer(&hit);
    assert_eq!(
        reply[5],
        kind::LOOKUP_RESP,
        "o hit é uma resposta, não ERROR"
    );
    let count = |frame: &Frame| blocks(|| drop(black_box(cache.answer(frame))));
    (count(&hit), count(&miss))
}

#[test]
fn the_lookup_path_allocates_only_what_it_must() {
    // Falhou? O perfil do momento fica em target/tmp/dhat-heap.json.
    let file = concat!(env!("CARGO_TARGET_TMPDIR"), "/dhat-heap.json");
    let _profiler = dhat::Profiler::builder().testing().file_name(file).build();
    let dim = 1536;
    // Um LOOKUP, hit ou miss: o embedding (1, já no tamanho do índice), o
    // top-k (1), o JSON da resposta (1) e o frame (1). Antes do PR 83/100 eram
    // 16 no hit: o embedding crescia dobrando, o prompt virava `String`, e o
    // vetor unitário e o texto guardado eram cópias.
    let flat = lookups::<FlatF32>(dim);
    dhat::assert_eq!(flat, (4, 4));
    // No int8, mais a consulta quantizada.
    let int8 = lookups::<FlatI8>(dim);
    dhat::assert_eq!(int8, (5, 5));

    // As peças de produção do caminho: montar e ler o frame, 1 bloco cada (o
    // payload); quantizar, 1 (os códigos); os kernels, nenhum.
    let frame = encode(kind::LOOKUP_RESP, 0, 1, &[b'x'; 300]).unwrap();
    let built = blocks(|| drop(black_box(encode(kind::LOOKUP_RESP, 0, 1, &[b'x'; 300]))));
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let read = blocks(|| drop(black_box(rt.block_on(read_frame(&mut frame.as_slice())))));
    let (v, w) = (axis(dim, 3), axis(dim, 4));
    let quantized = blocks(|| drop(black_box(QuantizedVector::quantize(&v))));
    let (a, b) = (QuantizedVector::quantize(&v), QuantizedVector::quantize(&w));
    let (a, b) = (a.unwrap(), b.unwrap());
    let cosine = blocks(|| {
        black_box(a.cosine(&b).unwrap());
    });
    let dot = blocks(|| {
        black_box(semantic_cache_simd::dot(&v, &w).unwrap());
    });
    dhat::assert_eq!((built, read, quantized, cosine, dot), (1, 1, 1, 0, 0));
}
