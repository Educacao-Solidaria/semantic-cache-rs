//! Preset de medição compartilhado pelos benchmarks.
//!
//! O criterion mede em nanossegundos e reporta média, intervalo de confiança,
//! outliers e a variação em relação à rodada anterior (salva em
//! `target/criterion`). Este módulo fixa os parâmetros estatísticos num lugar
//! só, para que todo benchmark do projeto seja comparável entre si.

use std::time::Duration;

use criterion::Criterion;

/// Dimensões de embedding medidas: `MiniLM` (384), BERT-base/nomic (768),
/// `text-embedding-3-small` (1536) e `text-embedding-3-large` (3072).
pub const EMBEDDING_DIMS: [usize; 4] = [384, 768, 1536, 3072];

/// Variável de ambiente que troca o preset para [`Precision::QUICK`].
pub const QUICK_ENV: &str = "SEMCACHE_BENCH_QUICK";

/// Parâmetros estatísticos de uma rodada de benchmarks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Precision {
    /// Amostras por benchmark (o criterion exige no mínimo 10).
    pub sample_size: usize,
    /// Aquecimento antes de medir: caches, branch predictor e frequência da CPU.
    pub warm_up: Duration,
    /// Tempo alvo de medição por benchmark.
    pub measurement: Duration,
    /// Variação relativa abaixo da qual uma mudança é tratada como ruído.
    pub noise_threshold: f64,
    /// Nível de significância do teste de hipótese de regressão.
    pub significance_level: f64,
    /// Nível de confiança do intervalo reportado.
    pub confidence_level: f64,
}

impl Precision {
    /// Rodada completa, usada para comparar kernels e detectar regressão.
    /// Ruído de 2%: variações menores que isso não são reportadas como mudança.
    pub const DEFAULT: Self = Self {
        sample_size: 100,
        warm_up: Duration::from_secs(3),
        measurement: Duration::from_secs(5),
        noise_threshold: 0.02,
        significance_level: 0.05,
        confidence_level: 0.95,
    };

    /// Rodada curta para conferir que os benchmarks executam; os números
    /// servem de ordem de grandeza, não de comparação.
    pub const QUICK: Self = Self {
        sample_size: 10,
        warm_up: Duration::from_millis(200),
        measurement: Duration::from_millis(500),
        noise_threshold: 0.05,
        significance_level: 0.05,
        confidence_level: 0.95,
    };

    /// Escolhe o preset pelo valor de [`QUICK_ENV`]: `1` ou `true` dá
    /// [`Precision::QUICK`]; ausente ou qualquer outro valor, o padrão.
    #[must_use]
    pub fn from_flag(flag: Option<&str>) -> Self {
        match flag.map(str::trim) {
            Some("1" | "true") => Self::QUICK,
            _ => Self::DEFAULT,
        }
    }

    /// [`Precision::from_flag`] lido do ambiente.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_flag(std::env::var(QUICK_ENV).ok().as_deref())
    }

    /// Monta o [`Criterion`] com estes parâmetros. Flags de linha de comando
    /// (`--sample-size`, `--save-baseline`...) continuam valendo por cima, via
    /// `configure_from_args` dentro de `criterion_group!`.
    ///
    /// # Panics
    ///
    /// Se algum parâmetro estiver fora do que o criterion aceita
    /// (`sample_size < 10`, níveis fora de `(0, 1)`, tempo zero).
    #[must_use]
    pub fn criterion(self) -> Criterion {
        Criterion::default()
            .sample_size(self.sample_size)
            .warm_up_time(self.warm_up)
            .measurement_time(self.measurement)
            .noise_threshold(self.noise_threshold)
            .significance_level(self.significance_level)
            .confidence_level(self.confidence_level)
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_accepted_by_criterion() {
        // O criterion valida cada parâmetro com `assert!`: montar não pode panicar.
        let _ = Precision::DEFAULT.criterion();
        let _ = Precision::QUICK.criterion();
    }

    #[test]
    fn quick_is_strictly_cheaper_than_default() {
        let (q, d) = (Precision::QUICK, Precision::DEFAULT);
        assert!(q.sample_size < d.sample_size);
        assert!(q.warm_up < d.warm_up);
        assert!(q.measurement < d.measurement);
    }

    #[test]
    fn flag_selects_preset() {
        assert_eq!(Precision::from_flag(Some("1")), Precision::QUICK);
        assert_eq!(Precision::from_flag(Some(" true ")), Precision::QUICK);
        assert_eq!(Precision::from_flag(Some("0")), Precision::DEFAULT);
        assert_eq!(Precision::from_flag(Some("")), Precision::DEFAULT);
        assert_eq!(Precision::from_flag(None), Precision::DEFAULT);
        assert_eq!(Precision::default(), Precision::DEFAULT);
    }

    #[test]
    fn dims_are_sorted_and_distinct() {
        assert!(EMBEDDING_DIMS.windows(2).all(|w| w[0] < w[1]));
    }
}
