//! Tokenizador BPE compatível com o `cl100k_base` (GPT-4, GPT-3.5-turbo,
//! `text-embedding-3-*`), em Rust puro: dá os mesmos tokens que o
//! `encode_ordinary` do tiktoken (tokens especiais como `<|endoftext|>` são
//! texto comum).
//!
//! O vocabulário não vem embutido (o arquivo tem 1,7 MB):
//! [`Tokenizer::cl100k_base`] lê o `cl100k_base.tiktoken` público e o compila
//! uma vez num mapa bytes → rank. [`Tokenizer::count`] dá o número de tokens
//! sem alocar, para medir prompt no caminho quente.

mod bpe;
mod pretok;
mod vocab;

use std::cell::RefCell;
use std::collections::HashMap;
use std::{fmt, io};

/// Id de um token: o rank dele no vocabulário.
pub type Rank = u32;

thread_local! {
    /// Buffers do BPE de cada thread: crescem até o maior pedaço já visto
    /// (com teto) e ficam, então contar não aloca.
    static MERGER: RefCell<bpe::Merger> = RefCell::default();
}

/// Vocabulário carregado e pronto para codificar; `Sync`, para dividir entre
/// threads num `Arc`.
pub struct Tokenizer {
    ranks: HashMap<Vec<u8>, Rank>,
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Carrega um vocabulário no formato `.tiktoken` (o `cl100k_base.tiktoken`
    /// público) com a pré-tokenização do `cl100k_base`.
    ///
    /// # Errors
    ///
    /// O erro de leitura, ou `InvalidData` (com a linha) se o arquivo está
    /// fora do formato.
    pub fn cl100k_base(vocab: impl io::BufRead) -> io::Result<Self> {
        pretok::load_tables();
        vocab::read(vocab).map(|ranks| Self { ranks })
    }

    /// Tokens do vocabulário (100 256 no `cl100k_base`).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.ranks.len()
    }

    /// Tokens de `text`, iguais aos do `encode_ordinary` do tiktoken.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<Rank> {
        let mut tokens = Vec::new();
        self.each_token(text, |rank| tokens.push(rank));
        tokens
    }

    /// Quantos tokens [`Tokenizer::encode`] devolveria, sem montar o vetor e
    /// sem alocar: a varredura não usa regex e os buffers do BPE são da
    /// thread. Só aloca a primeira vez que a thread vê um pedaço maior que os
    /// anteriores, e todo pedaço sem espaço acima de 64 KiB (que devolve a
    /// memória no fim).
    #[must_use]
    pub fn count(&self, text: &str) -> usize {
        let mut count = 0;
        self.each_token(text, |_| count += 1);
        count
    }

    fn each_token(&self, text: &str, mut emit: impl FnMut(Rank)) {
        MERGER.with_borrow_mut(|merger| {
            for piece in pretok::pieces(text) {
                // Pedaço que já é token sai direto, como no tiktoken: o BPE
                // nem sempre chegaria nele.
                match self.ranks.get(piece.as_bytes()) {
                    Some(&rank) => emit(rank),
                    None => merger.merge(&self.ranks, piece.as_bytes(), &mut emit),
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Os 256 bytes isolados (ranks 0..256) e `merges` em seguida.
    pub(crate) fn ranks(merges: &[&[u8]]) -> HashMap<Vec<u8>, Rank> {
        let merges = merges.iter().map(|m| m.to_vec());
        (0..=u8::MAX)
            .map(|b| vec![b])
            .chain(merges)
            .zip(0..)
            .collect()
    }

    #[test]
    fn encodes_pieces_whole_or_by_merges() {
        let merges: [&[u8]; 7] = [b"he", b"ll", b"hell", b"hello", b" w", b" wo", b"xyz"];
        let tok = Tokenizer {
            ranks: ranks(&merges),
        };
        assert_eq!(tok.vocab_size(), 263);
        // "hello" é token; " world" passa pelo BPE.
        assert_eq!(tok.encode("hello world"), [259, 261, 114, 108, 100]);
        // Pedaço inteiro no vocabulário sai direto, mesmo sem junção que o
        // alcance (não há "xy" nem "yz"), como no tiktoken.
        assert_eq!(tok.encode("xyz xy"), [262, 32, 120, 121]);
        // Sem junção: um token por byte do UTF-8.
        assert_eq!(tok.encode("ação 🎉").len(), "ação 🎉".len());
        assert_eq!(tok.encode(""), Vec::<Rank>::new());
        for text in [
            "hello world",
            "xyz xy",
            "ação 🎉",
            "",
            "  \n hello\t\tworld 123456 !!",
        ] {
            assert_eq!(tok.count(text), tok.encode(text).len(), "{text:?}");
        }
        assert_eq!(format!("{tok:?}"), "Tokenizer { vocab_size: 263, .. }");
    }
}
