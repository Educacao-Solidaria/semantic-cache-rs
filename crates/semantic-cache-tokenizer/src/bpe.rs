//! BPE de bytes: junta pares de partes adjacentes pelo rank.
//!
//! Mesmo resultado do `byte_pair_merge` do tiktoken: junta sempre o par de
//! menor rank, o mais à esquerda primeiro, até nenhum par adjacente estar no
//! vocabulário. Em vez de revarrer os pares a cada junção (O(n²) num pedaço
//! longo sem espaço: hash, texto minificado, palavra repetida), uma fila de
//! prioridade com descarte preguiçoso: O(n log n).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::Rank;

/// Rank de "sem par": última parte, parte absorvida ou par fora do vocabulário.
const NONE: Rank = Rank::MAX;

/// Chama `emit` com o rank de cada parte final de `piece`, da esquerda para a
/// direita. `ranks` contém os 256 bytes isolados (o carregador garante).
pub(crate) fn merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8], mut emit: impl FnMut(Rank)) {
    let len = piece.len();
    // Cada parte viva é identificada pelo byte onde começa; `prev[len]` é só
    // uma sentinela, para atualizar o anterior do fim sem testar.
    let mut next: Vec<usize> = (1..=len).collect();
    let mut prev: Vec<usize> = (0..=len).map(|at| at.wrapping_sub(1)).collect();
    let pair_rank = |next: &[usize], at: usize| match next[at] {
        right if right < len => ranks.get(&piece[at..next[right]]).copied().unwrap_or(NONE),
        _ => NONE,
    };
    // Rank do par (parte, seguinte) de cada parte; `NONE` se não há par.
    let mut rank: Vec<Rank> = (0..len).map(|at| pair_rank(&next, at)).collect();
    // Menor rank primeiro e, no empate, o mais à esquerda.
    let mut heap: BinaryHeap<_> = (0..len)
        .filter(|&at| rank[at] != NONE)
        .map(|at| Reverse((rank[at], at)))
        .collect();
    while let Some(Reverse((popped, at))) = heap.pop() {
        // Entrada velha: a parte foi absorvida ou o par mudou. Os bytes de um
        // par só crescem e cada token tem um rank, então um rank antigo não
        // volta a valer na mesma posição.
        if rank[at] != popped {
            continue;
        }
        let absorbed = next[at];
        next[at] = next[absorbed];
        prev[next[at]] = at;
        rank[absorbed] = NONE;
        // Mudaram o par desta parte e o da anterior (a primeira nunca morre).
        for part in [Some(at), (at > 0).then(|| prev[at])].into_iter().flatten() {
            rank[part] = pair_rank(&next, part);
            if rank[part] != NONE {
                heap.push(Reverse((rank[part], part)));
            }
        }
    }
    let mut at = 0;
    while at < len {
        // Toda parte final é um byte isolado ou o token de uma junção.
        emit(ranks[&piece[at..next[at]]]);
        at = next[at];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::ranks as vocab;

    /// O algoritmo clássico do tiktoken, quadrático: a cada passo varre os
    /// pares e junta o de menor rank, o primeiro no empate.
    fn reference(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<Rank> {
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
        while let Some((_, i)) = (0..parts.len().saturating_sub(1))
            .filter_map(|i| Some((*ranks.get(&[&parts[i][..], &parts[i + 1]].concat())?, i)))
            .min()
        {
            let right = parts.remove(i + 1);
            parts[i].extend(right);
        }
        parts.iter().map(|p| ranks[p]).collect()
    }

    fn merged(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<Rank> {
        let mut out = Vec::new();
        merge(ranks, piece, |r| out.push(r));
        out
    }

    #[test]
    fn lowest_rank_first_then_leftmost() {
        // "bc" (256) antes de "ab" (257), mesmo à direita; "aa" empata consigo.
        let ranks = vocab(&[b"bc", b"ab", b"aa"]);
        assert_eq!(merged(&ranks, b"abc"), [97, 256]);
        assert_eq!(merged(&ranks, b"aaa"), [258, 97]);
        assert_eq!(merged(&ranks, b"xyz"), [120, 121, 122]);
        assert_eq!(merged(&ranks, b""), Vec::<Rank>::new());
    }

    #[test]
    fn matches_the_quadratic_reference_on_random_vocabularies() {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut rand = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            usize::try_from(seed % n).unwrap()
        };
        for _ in 0..300 {
            // Alfabeto de 3 letras: muitas junções e muitos empates.
            let mut ranks = vocab(&[]);
            for _ in 0..rand(40) {
                let token: Vec<u8> = (0..2 + rand(5)).map(|_| b"abc"[rand(3)]).collect();
                let next = Rank::try_from(ranks.len()).unwrap();
                ranks.entry(token).or_insert(next);
            }
            let piece: Vec<u8> = (0..=rand(60)).map(|_| b"abc"[rand(3)]).collect();
            let (got, want) = (merged(&ranks, &piece), reference(&ranks, &piece));
            assert_eq!(got, want, "{piece:?}");
        }
    }

    #[test]
    fn long_piece_is_not_quadratic() {
        // 400 mil bytes de um par só: revarrer os pares a cada junção daria
        // ~10¹¹ comparações; com a fila são ~10⁶ operações de heap.
        let ranks = vocab(&[b"aa", b"aaaa", b"aaaaaaaa"]);
        let out = merged(&ranks, &vec![b'a'; 400_003]);
        assert_eq!(out.len(), 50_002);
        assert!(out[..50_000].iter().all(|&r| r == 258));
        assert_eq!(out[50_000..], [256, 97]);
    }
}
