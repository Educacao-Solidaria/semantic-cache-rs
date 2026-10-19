//! Compatibilidade com o `cl100k_base` real (o vocabulário, 1,7 MB, não fica
//! no repositório): baixe
//! `https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`
//! e rode `SEMCACHE_CL100K=<arquivo> cargo test -- --ignored`.

use std::{fs::File, io::BufReader};

use semantic_cache_tokenizer::{Rank, Tokenizer};

include!("fixtures/cl100k_cases.rs");

#[test]
#[ignore = "precisa do cl100k_base.tiktoken em SEMCACHE_CL100K"]
fn matches_tiktoken_encode_ordinary() {
    let path = std::env::var("SEMCACHE_CL100K").expect("SEMCACHE_CL100K com o caminho");
    let file = File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let tokenizer = Tokenizer::cl100k_base(BufReader::new(file)).unwrap();
    assert_eq!(tokenizer.vocab_size(), 100_256);
    for &(text, want) in CASES {
        assert_eq!(tokenizer.encode(text), want, "{text:?}");
    }
}
