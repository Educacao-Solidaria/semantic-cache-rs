//! Leitura do formato `.tiktoken`: uma linha `<bytes do token em base64> <rank>`
//! por token, como em `cl100k_base.tiktoken`.

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead};

use crate::Rank;

/// Bytes do token → rank (o id do token). O arquivo é entrada externa:
/// formato errado vira `InvalidData` com a linha, nunca panic nem token errado
/// em silêncio; faltar um dos 256 bytes isolados também (haveria texto
/// impossível de codificar).
pub(crate) fn read(reader: impl BufRead) -> io::Result<HashMap<Vec<u8>, Rank>> {
    let invalid = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
    let (mut ranks, mut seen) = (HashMap::new(), HashSet::new());
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        let bad = |reason: &str| invalid(format!("vocabulário, linha {}: {reason}", index + 1));
        let mut fields = line.split_ascii_whitespace();
        let (token, rank) = match (fields.next(), fields.next(), fields.next()) {
            (Some(token), Some(rank), None) => (token, rank),
            (None, ..) => continue, // linha em branco, como no tiktoken
            _ => return Err(bad("esperado `<base64> <rank>`")),
        };
        let token = decode_base64(token)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| bad("token em base64 inválido ou vazio"))?;
        let rank: Rank = rank.parse().map_err(|_| bad("rank inválido"))?;
        if !seen.insert(rank) {
            return Err(bad("rank repetido"));
        }
        if ranks.insert(token, rank).is_some() {
            return Err(bad("token repetido"));
        }
    }
    match (0..=u8::MAX).find(|&byte| !ranks.contains_key([byte].as_slice())) {
        Some(byte) => Err(invalid(format!(
            "vocabulário sem o byte isolado {byte:#04x}"
        ))),
        None => Ok(ranks),
    }
}

/// Base64 padrão, estrito: tamanho múltiplo de 4 e até dois `=`, só no fim.
fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let body = text.trim_end_matches('=');
    if text.len() % 4 != 0 || text.len() - body.len() > 2 {
        return None;
    }
    let (mut out, mut bits, mut pending) = (Vec::with_capacity(body.len()), 0u32, 0);
    for c in body.bytes() {
        let value = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        // Só os bits ainda não emitidos importam (no máximo 12).
        bits = ((bits << 6) | u32::from(value)) & 0xFFFF;
        pending += 6;
        if pending >= 8 {
            pending -= 8;
            out.push((bits >> pending).to_le_bytes()[0]);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Os 256 bytes isolados com ranks 0..256 (byte `b` = `XY==`).
    fn bytes_file() -> String {
        const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let char_at = |i: u8| char::from(B64[usize::from(i)]);
        let mut file = String::new();
        for b in 0..=u8::MAX {
            file.extend([char_at(b >> 2), char_at((b & 3) << 4), '=', '=', ' ']);
            file.push_str(&b.to_string());
            file.push('\n');
        }
        file
    }

    fn error(text: &str) -> String {
        read(text.as_bytes()).unwrap_err().to_string()
    }

    #[test]
    fn reads_every_byte_and_merge() {
        // " daycare" e " Conveyor" com o base64 do cl100k_base real.
        let file = bytes_file() + "\nIGRheWNhcmU= 256\n   \nIENvbnZleW9y 100255\n";
        let ranks = read(file.as_bytes()).unwrap();
        assert_eq!(ranks.len(), 258);
        assert_eq!(ranks[b"A".as_slice()], 65);
        assert_eq!(ranks[b" daycare".as_slice()], 256);
        assert_eq!(ranks[b" Conveyor".as_slice()], 100_255);
    }

    #[test]
    fn rejects_malformed_lines_with_their_number() {
        let line = |text: &str| error(&format!("IQ== 0\n{text}"));
        let msg = |reason: &str| format!("vocabulário, linha 2: {reason}");
        assert_eq!(line("aGk= 1 x"), msg("esperado `<base64> <rank>`"));
        assert_eq!(line("aGk="), msg("esperado `<base64> <rank>`"));
        for bad in ["aGk", "a!k=", "a=k=", "a===", "====", "IQ==IQ=="] {
            assert_eq!(
                line(&format!("{bad} 1")),
                msg("token em base64 inválido ou vazio")
            );
        }
        assert_eq!(line("aGk= -1"), msg("rank inválido"));
        assert_eq!(line("aGk= 4294967296"), msg("rank inválido"));
        assert_eq!(line("aGk= 0"), msg("rank repetido"));
        assert_eq!(line("IQ== 1"), msg("token repetido"));
    }

    #[test]
    fn rejects_missing_bytes_and_unreadable_input() {
        assert_eq!(error("IQ== 0"), "vocabulário sem o byte isolado 0x00");
        let not_utf8 = read([0xFF, b'\n'].as_slice()).unwrap_err();
        assert_eq!(not_utf8.kind(), io::ErrorKind::InvalidData);
    }
}
