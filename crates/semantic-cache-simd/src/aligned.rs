//! Vetor `f32` que começa num endereço múltiplo de 32 bytes (256 bits, um
//! registrador AVX) e é completado com zeros até um múltiplo de 8 valores.
//!
//! Um `Vec<f32>` só garante 4 bytes de alinhamento (o malloc do glibc entrega
//! 16), e uma fatia no meio de um buffer, qualquer coisa. Uma carga de 32
//! bytes que começa a menos de 32 do fim de uma linha de cache (64 bytes) lê
//! duas linhas: com o início desalinhado, é metade das cargas do kernel. O
//! `loadu` em si não custa mais que o `load` alinhado; o custo é a linha
//! partida. Medido (Ryzen 7 8700G, AVX2, 1536 dimensões): o produto escalar
//! com os dois vetores na L1 leva 43 ns alinhado e 76 ns desalinhado; numa
//! varredura, 1,3× na L2, 1,15× na L3 e nada na DRAM, onde a banda manda.
//!
//! Os zeros do fim são neutros em produto escalar, cosseno e distância L2, e
//! com eles o kernel não tem cauda: [`AlignedVector::padded`].

use std::ops::Deref;

/// Valores por bloco: um registrador AVX de `f32`.
const LANES: usize = 8;

/// Oito `f32` alinhados a 32 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C, align(32))]
struct Block([f32; LANES]);

// Sem preenchimento entre blocos: `n` blocos são `8n` `f32` contíguos.
const _: () = assert!(size_of::<Block>() == 32 && align_of::<Block>() == 32);

/// Vetor `f32` alinhado a 32 bytes; derreferencia para os `len` valores.
/// Do `Vector` do crate de domínio: `AlignedVector::from(v.data.as_slice())`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AlignedVector {
    blocks: Vec<Block>,
    len: usize,
}

impl AlignedVector {
    /// Os valores e os zeros até o próximo múltiplo de 8, que é o que o kernel
    /// deve receber: dois vetores da mesma dimensão têm o mesmo comprimento
    /// completado, e o resultado é o mesmo dos valores sem os zeros.
    #[must_use]
    #[allow(unsafe_code)]
    pub fn padded(&self) -> &[f32] {
        // SAFETY: `Block` é `repr(C)` com um `[f32; 8]` e tem 32 bytes
        // (conferido acima), então os blocos do `Vec` são `8 · len` `f32`
        // contíguos e inicializados. O ponteiro do `Vec` está alinhado a 32,
        // múltiplo dos 4 do `f32` (vazio, é o `dangling` do `Block`, alinhado
        // também). A fatia empresta `self`, e o `Vec` não muda enquanto isso.
        unsafe {
            std::slice::from_raw_parts(
                self.blocks.as_ptr().cast::<f32>(),
                self.blocks.len() * LANES,
            )
        }
    }
}

impl From<&[f32]> for AlignedVector {
    fn from(values: &[f32]) -> Self {
        let mut blocks = vec![Block::default(); values.len().div_ceil(LANES)];
        for (block, chunk) in blocks.iter_mut().zip(values.chunks(LANES)) {
            block.0[..chunk.len()].copy_from_slice(chunk);
        }
        Self {
            blocks,
            len: values.len(),
        }
    }
}

impl Deref for AlignedVector {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.padded()[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use semantic_cache_rs::math::euclidean_distance;
    use semantic_cache_rs::Vector;

    use super::*;
    use crate::{CpuFeatures, Kernel};

    /// Inteiros pequenos: toda soma é exata em `f32`, em qualquer ordem.
    #[allow(clippy::cast_precision_loss)]
    fn ints(m: usize, len: usize) -> Vec<f32> {
        (0..len).map(|i| (i % m) as f32 - (m / 2) as f32).collect()
    }

    const LENS: [usize; 6] = [383, 384, 385, 1536, 1537, 3072];

    #[test]
    fn starts_on_32_bytes_and_pads_with_positive_zeros() {
        for len in (0..=70).chain(LENS) {
            let values = ints(7, len);
            let aligned = AlignedVector::from(values.as_slice());
            for v in [&aligned, &aligned.clone()] {
                let padded = v.padded();
                assert_eq!(padded.as_ptr() as usize % 32, 0, "len {len}");
                assert_eq!(padded.len(), len.div_ceil(8) * 8);
                assert_eq!(**v, *values, "len {len}");
                assert!(padded[len..].iter().all(|z| z.to_bits() == 0), "len {len}");
            }
        }
        // O `Vector` do crate de domínio, pela fatia.
        let vector = Vector {
            data: ints(5, 1536),
        };
        let aligned = AlignedVector::from(vector.data.as_slice());
        assert_eq!((&*aligned, aligned.len()), (vector.data.as_slice(), 1536));
        assert_eq!(AlignedVector::default().padded(), &[0.0f32; 0]);
    }

    #[test]
    fn zero_padding_changes_no_dot_cosine_or_distance() {
        let kernels: Vec<Kernel> = Kernel::PREFERENCE
            .into_iter()
            .filter(|k| k.supported(CpuFeatures::get()))
            .collect();
        for len in (0..=70).chain(LENS) {
            let (a, b) = (ints(7, len), ints(5, len));
            let (pa, pb) = (AlignedVector::from(&a[..]), AlignedVector::from(&b[..]));
            let (pa, pb) = (pa.padded(), pb.padded());
            // O mesmo kernel, bit a bit, com e sem os zeros (o resultado
            // contra a referência escalar é conferido em `similarity`).
            for k in &kernels {
                let dot = |x, y| k.dot(x, y).unwrap().to_bits();
                assert_eq!(dot(pa, pb), dot(&a, &b), "{k} dot, len {len}");
                let cos = |x, y| k.cosine(x, y).ok().map(f32::to_bits);
                assert_eq!(cos(pa, pb), cos(&a, &b), "{k} cosine, len {len}");
            }
            let l2 = |x: &[f32], y: &[f32]| euclidean_distance(x, y).unwrap().to_bits();
            assert_eq!(l2(pa, pb), l2(&a, &b), "l2, len {len}");
        }
    }
}
