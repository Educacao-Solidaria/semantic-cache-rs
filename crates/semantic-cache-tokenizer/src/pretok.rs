//! Pré-tokenização do `cl100k_base` no tiktoken 0.14, sem regex e sem alocar:
//!
//! `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s`
//!
//! `piece_end` testa as alternativas na ordem, como o motor do tiktoken. As
//! classes `\p{L}` e `\p{N}` vêm das tabelas Unicode do `regex-syntax` (as do
//! crate `regex`), montadas uma vez; `\s` é `char::is_whitespace` (`White_Space`).

use std::cmp::Ordering;
use std::sync::LazyLock;

use regex_syntax::hir::{Class, HirKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Letter,
    Number,
    Space,
    Other,
}

/// Intervalos ordenados de `\p{L}` e de `\p{N}`.
static TABLES: LazyLock<[Vec<(char, char)>; 2]> = LazyLock::new(|| {
    [r"\p{L}", r"\p{N}"].map(|class| {
        match regex_syntax::parse(class).map(regex_syntax::hir::Hir::into_kind) {
            Ok(HirKind::Class(Class::Unicode(set))) => {
                set.ranges().iter().map(|r| (r.start(), r.end())).collect()
            }
            other => unreachable!("{class} deveria ser classe Unicode: {other:?}"),
        }
    })
});

/// Monta as tabelas agora (na construção do tokenizador), e não na primeira
/// contagem com caractere fora do ASCII.
pub(crate) fn load_tables() {
    LazyLock::force(&TABLES);
}

fn kind(c: char) -> Kind {
    let within = |table: &[(char, char)]| {
        let side = |&(lo, hi): &(char, char)| match (hi < c, lo > c) {
            (true, _) => Ordering::Less,
            (_, true) => Ordering::Greater,
            _ => Ordering::Equal,
        };
        table.binary_search_by(side).is_ok()
    };
    match c {
        _ if c.is_whitespace() => Kind::Space,
        'a'..='z' | 'A'..='Z' => Kind::Letter,
        '0'..='9' => Kind::Number,
        _ if c.is_ascii() => Kind::Other,
        _ if within(&TABLES[0]) => Kind::Letter,
        _ if within(&TABLES[1]) => Kind::Number,
        _ => Kind::Other,
    }
}

/// Pedaços em ordem, cobrindo `text` inteiro; nenhum é vazio.
pub(crate) fn pieces(text: &str) -> impl Iterator<Item = &str> {
    let mut start = 0;
    std::iter::from_fn(move || {
        (start < text.len()).then(|| {
            let end = piece_end(text, start);
            debug_assert!(end > start, "pedaço vazio em {start}");
            let piece = &text[start..end];
            start = end;
            piece
        })
    })
}

/// Fim (exclusivo) do pedaço que começa em `start` (< `text.len()`).
fn piece_end(text: &str, start: usize) -> usize {
    let rest = &text[start..];
    let mut chars = rest.chars();
    let first = chars.next().expect("start antes do fim");
    let second = chars.next().map(kind);
    let after_first = start + first.len_utf8();
    let run = |from: usize, want: Kind| skip(text, from, |c| kind(c) == want);
    // `'(?i:[sdmt]|ll|ve|re)`
    if first == '\'' {
        if let Some(len) = contraction(&rest[1..]) {
            return after_first + len;
        }
    }
    // `[^\r\n\p{L}\p{N}]?+\p{L}++`
    match kind(first) {
        Kind::Letter => return run(start, Kind::Letter),
        Kind::Number => {
            // `\p{N}{1,3}+`
            let digits = rest
                .chars()
                .take(3)
                .take_while(|&c| kind(c) == Kind::Number);
            return start + digits.map(char::len_utf8).sum::<usize>();
        }
        _ if second == Some(Kind::Letter) && !matches!(first, '\r' | '\n') => {
            return run(after_first, Kind::Letter);
        }
        _ => {}
    }
    // ` ?[^\s\p{L}\p{N}]++[\r\n]*+`
    let other_from = match (first, kind(first), second) {
        (' ', _, Some(Kind::Other)) => Some(after_first),
        (_, Kind::Other, _) => Some(start),
        _ => None,
    };
    if let Some(from) = other_from {
        return skip(text, run(from, Kind::Other), |c| matches!(c, '\r' | '\n'));
    }
    // Daqui em diante `first` é espaço.
    let end = skip(text, start, char::is_whitespace);
    let spaces = &text[start..end];
    if end == text.len() {
        return end; // `\s++$`
    }
    if let Some(newline) = spaces.rfind(['\r', '\n']) {
        return start + newline + 1; // `\s*[\r\n]`: até a última quebra
    }
    // `\s+(?!\S)` deixa o último espaço para o pedaço seguinte; `\s`, um só.
    let last = spaces.chars().next_back().map_or(0, char::len_utf8);
    if spaces.len() > last {
        end - last
    } else {
        end
    }
}

/// Fim da sequência de caracteres que satisfazem `keep` a partir de `from`.
fn skip(text: &str, from: usize, keep: impl Fn(char) -> bool) -> usize {
    text[from..]
        .find(|c| !keep(c))
        .map_or(text.len(), |i| from + i)
}

/// Bytes de `[sdmt]|ll|ve|re` sem diferenciar maiúsculas no começo de `rest`.
fn contraction(rest: &str) -> Option<usize> {
    let mut chars = rest.chars();
    let a = chars.next()?;
    if matches!(fold(a), 's' | 'd' | 'm' | 't') {
        return Some(a.len_utf8());
    }
    let b = chars.next()?;
    let pair = matches!((fold(a), fold(b)), ('l', 'l') | ('v' | 'r', 'e'));
    pair.then(|| a.len_utf8() + b.len_utf8())
}

/// Dobra de caixa do `(?i)` para as letras das contrações: fora do ASCII, só o
/// `ſ` (U+017F) vira `s` (o teste confere contra o `regex-syntax`).
fn fold(c: char) -> char {
    if c == 'ſ' {
        's'
    } else {
        c.to_ascii_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use regex::Regex;

    use super::*;

    fn split(text: &str) -> Vec<&str> {
        pieces(text).collect()
    }

    /// A pré-tokenização do PR 25/100 (regex sem lookahead + corte do `\s+`),
    /// como oráculo.
    fn by_regex<'t>(re: &Regex, text: &'t str) -> Vec<&'t str> {
        let (mut out, mut start) = (Vec::new(), 0);
        while let Some(found) = re.find_at(text, start) {
            let mut end = found.end();
            let mut chars = found.as_str().chars();
            if let (Some(last), Some(_)) = (chars.next_back(), chars.next()) {
                if end < text.len() && last.is_whitespace() && !matches!(last, '\r' | '\n') {
                    end -= last.len_utf8();
                }
            }
            out.push(&text[start..end]);
            start = end;
        }
        out
    }

    #[test]
    fn pieces_follow_the_cl100k_expression() {
        assert_eq!(split("hello world"), ["hello", " world"]);
        assert_eq!(split(""), Vec::<&str>::new());
        // Contrações, com maiúscula e com `ſ` (que dobra para `s`); depois de
        // outro apóstrofo não há contração.
        let contractions = [
            "I", "'m", " can", "'t", " You", "'RE", " x", "'ſ", "''", "ll",
        ];
        assert_eq!(split("I'm can't You'RE x'ſ''ll"), contractions);
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
    }

    #[test]
    fn contraction_folds_match_the_regex_case_folds() {
        for letter in "sdmtlver".chars() {
            let hir = regex_syntax::parse(&format!("(?i){letter}")).unwrap();
            let HirKind::Class(Class::Unicode(set)) = hir.into_kind() else {
                panic!("(?i){letter} deveria ser classe")
            };
            let regex: Vec<char> = set.iter().flat_map(|r| r.start()..=r.end()).collect();
            let ours: Vec<char> = ('\0'..=char::MAX).filter(|&c| fold(c) == letter).collect();
            assert_eq!(ours, regex, "{letter}");
        }
    }

    #[test]
    fn matches_the_regex_oracle_on_random_text() {
        let re = Regex::new(concat!(
            r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+",
        ))
        .unwrap();
        let pool: Vec<char> = concat!(
            "aZ09 \t\n\r\u{b}\u{c}\u{85}\u{a0}\u{2028}\u{3000}'!$(._sStTdDmMlLvVeErRſ",
            "áçÉ\u{301}\u{903}日本한\u{94d}نПΩ٣²½Ⅻ🎉\u{200d}\u{0}\u{7f}"
        )
        .chars()
        .collect();
        let mut seed = 0x2545_F491_4F6C_DD1D_u64;
        let mut rand = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            usize::try_from(seed % n as u64).unwrap()
        };
        for _ in 0..3000 {
            let text: String = (0..rand(40))
                .map(|_| match rand(12) {
                    0 => char::from_u32(u32::try_from(rand(0x11_0000)).unwrap()).unwrap_or('?'),
                    _ => pool[rand(pool.len())],
                })
                .collect();
            assert_eq!(split(&text), by_regex(&re, &text), "{text:?}");
        }
    }
}
