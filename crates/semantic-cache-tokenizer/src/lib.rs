//! Tokenizador BPE compatível com o `cl100k_base` (GPT-4, GPT-3.5-turbo,
//! `text-embedding-3-*`), em Rust puro: dá os mesmos tokens que o
//! `encode_ordinary` do tiktoken (tokens especiais como `<|endoftext|>` são
//! texto comum).
//!
//! O vocabulário não vem embutido (o arquivo tem 1,7 MB):
//! [`Tokenizer::cl100k_base`] lê o `cl100k_base.tiktoken` público e o compila
//! uma vez num mapa bytes → rank.

mod bpe;
mod vocab;

use std::collections::HashMap;
use std::sync::LazyLock;
use std::{fmt, io};

use regex::Regex;

/// Id de um token: o rank dele no vocabulário.
pub type Rank = u32;

/// Pré-tokenização do `cl100k_base` no tiktoken 0.14:
/// `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s`
///
/// O crate `regex` não tem quantificador possessivo nem lookahead. Os
/// possessivos não mudam o resultado (devolver caractere nunca faz o resto da
/// alternativa casar), e `\s+(?!\S)|\s` vira `\s+`, cortado em `pieces`.
static PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}",
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+",
    ))
    .expect("expressão fixa e válida")
});

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
        for piece in pieces(text) {
            // Pedaço que já é token sai direto, como no tiktoken: o BPE nem
            // sempre chegaria nele.
            match self.ranks.get(piece.as_bytes()) {
                Some(&rank) => tokens.push(rank),
                None => bpe::merge(&self.ranks, piece.as_bytes(), |r| tokens.push(r)),
            }
        }
        tokens
    }
}

/// Pedaços da pré-tokenização, em ordem e cobrindo `text` inteiro.
fn pieces(text: &str) -> impl Iterator<Item = &str> {
    let mut start = 0;
    std::iter::from_fn(move || {
        // Todo caractere é letra, número, espaço ou outro: sempre há pedaço
        // começando onde o anterior terminou.
        let found = PATTERN.find_at(text, start)?;
        debug_assert_eq!(found.start(), start);
        let mut end = found.end();
        // Só a alternativa `\s+` termina antes do fim do texto num espaço que
        // não é quebra de linha (`\s+$` vai até o fim; `\s*[\r\n]` termina em
        // quebra). Aí vale o `\s+(?!\S)`: com dois ou mais espaços, o último
        // fica para o pedaço seguinte.
        let mut chars = found.as_str().chars();
        if let (Some(last), Some(_)) = (chars.next_back(), chars.next()) {
            if end < text.len() && last.is_whitespace() && !matches!(last, '\r' | '\n') {
                end -= last.len_utf8();
            }
        }
        let piece = &text[start..end];
        start = end;
        Some(piece)
    })
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

    fn split(text: &str) -> Vec<&str> {
        pieces(text).collect()
    }

    #[test]
    fn pieces_follow_the_cl100k_expression() {
        assert_eq!(split("hello world"), ["hello", " world"]);
        assert_eq!(split(""), Vec::<&str>::new());
        // Contrações, com maiúscula e com `ſ` (que dobra para `s`).
        let contractions = ["I", "'m", " can", "'t", " You", "'RE", " x", "'ſ"];
        assert_eq!(split("I'm can't You'RE x'ſ"), contractions);
        // Até 3 dígitos; pontuação leva o espaço antes e as quebras depois.
        assert_eq!(split("12345 !!!\n\nx"), ["123", "45", " !!!\n\n", "x"]);
        assert_eq!(split("(hello) $100"), ["(hello", ")", " $", "100"]);
        // `\s+(?!\S)`: o último espaço vai para a palavra seguinte.
        assert_eq!(split("a   b"), ["a", "  ", " b"]);
        assert_eq!(split("a\t\u{3000}b 1"), ["a", "\t", "\u{3000}b", " ", "1"]);
        // Espaços até a última quebra de linha; o resto segue a regra acima.
        assert_eq!(split("a \n\n  b"), ["a", " \n\n", " ", " b"]);
        assert_eq!(split("a\r\n\r\nb"), ["a", "\r\n\r\n", "b"]);
        // `\s++$`: espaço até o fim é um pedaço só, mesmo com quebra no meio.
        assert_eq!(split("fim \n "), ["fim", " \n "]);
        // Marca combinante não é letra.
        assert_eq!(split("e\u{301}"), ["e", "\u{301}"]);
        let text = "Olá,  mundo!\r\n\t 日本語 ٣٤٥ ²³ 🎉🎉 'll ''s x  \n";
        assert_eq!(split(text).concat(), text);
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
        assert_eq!(format!("{tok:?}"), "Tokenizer { vocab_size: 263, .. }");
    }
}
