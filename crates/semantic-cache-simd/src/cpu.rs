//! Detecção de extensões SIMD em tempo de execução e seleção de kernel.
//!
//! O binário é compilado para o x86-64/aarch64 base; o que a máquina tem a
//! mais (AVX2, FMA, AVX-512, NEON) só se sabe ao rodar. A detecção acontece
//! uma vez por processo ([`CpuFeatures::get`]) e o kernel escolhido também
//! ([`Kernel::active`]): o caminho quente lê um valor em cache, sem `cpuid`.

use std::fmt;
use std::str::FromStr;
use std::sync::OnceLock;

/// Variável que força um kernel (`scalar`, `avx2`, `avx512`, `neon`). Serve
/// para rebaixar em produção sem recompilar; um kernel que a CPU não suporta
/// é ignorado.
pub const KERNEL_ENV: &str = "SEMCACHE_SIMD";

/// Extensões relevantes para os kernels de similaridade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)] // um bool por extensão é o dado
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub neon: bool,
}

impl CpuFeatures {
    /// Consulta a CPU agora. Prefira [`CpuFeatures::get`].
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                avx2: is_x86_feature_detected!("avx2"),
                fma: is_x86_feature_detected!("fma"),
                avx512f: is_x86_feature_detected!("avx512f"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                neon: std::arch::is_aarch64_feature_detected!("neon"),
                ..Self::default()
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    /// Detecção feita uma única vez por processo.
    pub fn get() -> &'static Self {
        static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
        FEATURES.get_or_init(Self::detect)
    }
}

impl fmt::Display for CpuFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let flags = [
            ("avx2", self.avx2),
            ("fma", self.fma),
            ("avx512f", self.avx512f),
            ("neon", self.neon),
        ];
        let on: Vec<&str> = flags.iter().filter(|(_, b)| *b).map(|(n, _)| *n).collect();
        if on.is_empty() {
            f.write_str("nenhuma")
        } else {
            f.write_str(&on.join(","))
        }
    }
}

/// Implementação de similaridade a usar, da mais portável à mais larga.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Referência escalar; roda em qualquer CPU.
    Scalar,
    /// 128 bits, aarch64.
    Neon,
    /// 256 bits com multiplica-e-soma fundido (x86-64-v3).
    Avx2Fma,
    /// 512 bits; só com a feature `avx512`.
    Avx512,
}

impl Kernel {
    /// Todos, do mais preferido ao menos.
    pub const PREFERENCE: [Self; 4] = [Self::Avx512, Self::Avx2Fma, Self::Neon, Self::Scalar];

    /// `true` se a CPU tem tudo que o kernel usa (e, para AVX-512, se a
    /// feature `avx512` está ligada).
    #[must_use]
    pub fn supported(self, cpu: &CpuFeatures) -> bool {
        match self {
            Self::Scalar => true,
            Self::Neon => cpu.neon,
            Self::Avx2Fma => cpu.avx2 && cpu.fma,
            Self::Avx512 => cfg!(feature = "avx512") && cpu.avx512f && cpu.avx2 && cpu.fma,
        }
    }

    /// Kernel para `cpu`: o pedido, se suportado; senão o melhor suportado.
    #[must_use]
    pub fn select(cpu: &CpuFeatures, requested: Option<Self>) -> Self {
        requested
            .filter(|k| k.supported(cpu))
            .or_else(|| Self::PREFERENCE.into_iter().find(|k| k.supported(cpu)))
            .unwrap_or(Self::Scalar)
    }

    /// Kernel do processo, escolhido uma vez a partir de [`CpuFeatures::get`]
    /// e de [`KERNEL_ENV`] (valor inválido é ignorado).
    pub fn active() -> Self {
        static KERNEL: OnceLock<Kernel> = OnceLock::new();
        *KERNEL.get_or_init(|| {
            let requested = std::env::var(KERNEL_ENV).ok().and_then(|v| v.parse().ok());
            Self::select(CpuFeatures::get(), requested)
        })
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Avx2Fma => "avx2",
            Self::Avx512 => "avx512",
        }
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Kernel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        Self::PREFERENCE
            .into_iter()
            .find(|k| k.name() == s)
            .ok_or_else(|| format!("kernel desconhecido `{s}` (scalar|neon|avx2|avx512)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X86_V3: CpuFeatures = CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: false,
        neon: false,
    };
    const X86_V4: CpuFeatures = CpuFeatures {
        avx512f: true,
        ..X86_V3
    };
    const ARM: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        avx512f: false,
        neon: true,
    };
    const BASE: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        avx512f: false,
        neon: false,
    };

    #[test]
    fn selects_widest_supported() {
        assert_eq!(Kernel::select(&BASE, None), Kernel::Scalar);
        assert_eq!(Kernel::select(&ARM, None), Kernel::Neon);
        assert_eq!(Kernel::select(&X86_V3, None), Kernel::Avx2Fma);
        let v4 = if cfg!(feature = "avx512") {
            Kernel::Avx512
        } else {
            Kernel::Avx2Fma
        };
        assert_eq!(Kernel::select(&X86_V4, None), v4);
    }

    #[test]
    fn avx2_without_fma_is_not_enough() {
        let cpu = CpuFeatures {
            fma: false,
            ..X86_V3
        };
        assert_eq!(Kernel::select(&cpu, None), Kernel::Scalar);
    }

    #[test]
    fn request_downgrades_but_never_upgrades() {
        assert_eq!(
            Kernel::select(&X86_V3, Some(Kernel::Scalar)),
            Kernel::Scalar
        );
        // Pedido sem suporte cai no melhor suportado: executar AVX2 numa CPU
        // sem AVX2 é instrução ilegal, não lentidão.
        assert_eq!(Kernel::select(&BASE, Some(Kernel::Avx2Fma)), Kernel::Scalar);
        assert_eq!(Kernel::select(&ARM, Some(Kernel::Avx512)), Kernel::Neon);
    }

    #[test]
    fn names_roundtrip() {
        for k in Kernel::PREFERENCE {
            assert_eq!(k.name().parse::<Kernel>(), Ok(k));
        }
        assert_eq!(" AVX2 ".parse::<Kernel>(), Ok(Kernel::Avx2Fma));
        assert!("sse9".parse::<Kernel>().unwrap_err().contains("sse9"));
    }

    #[test]
    fn display_lists_enabled_flags() {
        assert_eq!(BASE.to_string(), "nenhuma");
        assert_eq!(X86_V4.to_string(), "avx2,fma,avx512f");
        assert_eq!(ARM.to_string(), "neon");
    }

    #[test]
    fn detection_is_cached_and_consistent() {
        assert!(std::ptr::eq(CpuFeatures::get(), CpuFeatures::get()));
        assert_eq!(*CpuFeatures::get(), CpuFeatures::detect());
        let active = Kernel::active();
        assert!(active.supported(CpuFeatures::get()), "{active} sem suporte");
        assert_eq!(Kernel::active(), active);
    }

    #[test]
    fn detection_matches_target() {
        let cpu = CpuFeatures::detect();
        // Extensões ligadas em tempo de compilação estão presentes em runtime.
        assert!(!cfg!(target_feature = "avx2") || cpu.avx2);
        assert!(!cfg!(target_feature = "fma") || cpu.fma);
        if cfg!(target_arch = "aarch64") {
            assert!(cpu.neon, "NEON é obrigatório no aarch64");
        } else {
            assert!(!cpu.neon);
        }
    }
}
