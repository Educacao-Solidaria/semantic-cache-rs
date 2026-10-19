//! Kernel AVX2 do produto de códigos `i8`: 32 por instrução (`__m256i`).
//!
//! `_mm256_maddubs_epi16` multiplica `u8 × i8` e soma os pares em `i16`, com
//! saturação. Com os dois lados `i8`, `|a| · (b com o sinal de a)` é o mesmo
//! `a · b` com o primeiro fator sem sinal, e é exato porque os códigos ficam em
//! `[-127, 127]` (invariante de `QuantizedVector`): um par soma no máximo
//! `2 · 127² = 32 258 < 32 767`. Com `-128` não seria — `abs` e `sign` de
//! `-128` dão `-128`. `_mm256_madd_epi16` com uns junta os pares em `i32`.

#![allow(unsafe_code)]

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m256i, _mm256_abs_epi8, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
    _mm256_setzero_si256, _mm256_sign_epi8, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_abs_epi8, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
    _mm256_setzero_si256, _mm256_sign_epi8, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32,
};

/// Códigos por registrador.
const LANES: usize = 32;

/// `Σ aᵢ bᵢ`.
///
/// # Safety
///
/// A CPU precisa ter AVX2, e `a.len() == b.len()`. Para o resultado ser exato
/// (não é questão de memória): códigos em `[-127, 127]` e `len · 127² < 2³¹`.
#[target_feature(enable = "avx2")]
#[allow(clippy::cast_ptr_alignment)] // `loadu` não exige alinhamento
pub(super) unsafe fn dot(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    // SAFETY (os loads): cada leitura cobre `[i, i + LANES)` com
    // `i + LANES <= len`, dentro dos dois slices; `loadu` aceita qualquer
    // alinhamento.
    while i + LANES <= len {
        let va = _mm256_loadu_si256(pa.add(i).cast::<__m256i>());
        let vb = _mm256_loadu_si256(pb.add(i).cast::<__m256i>());
        let pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(va), _mm256_sign_epi8(vb, va));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, ones));
        i += LANES;
    }
    hsum(acc) + super::dot_scalar(&a[i..], &b[i..])
}

/// Soma das 8 faixas `i32`: 256 → 128 bits, depois 4 → 2 → 1.
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256i) -> i32 {
    let quad = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let pair = _mm_add_epi32(quad, _mm_shuffle_epi32(quad, 0b01_00_11_10));
    _mm_cvtsi128_si32(_mm_add_epi32(pair, _mm_shuffle_epi32(pair, 0b00_00_00_01)))
}
