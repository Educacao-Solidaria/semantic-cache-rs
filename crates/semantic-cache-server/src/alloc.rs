//! Alocador global do servidor.
//!
//! O cache aloca e libera embeddings em muitas threads ao mesmo tempo. O
//! alocador do sistema (glibc malloc) fragmenta e disputa locks nesse padrão;
//! o jemalloc usa arenas por thread e mantém a latência de alocação estável.
//!
//! Fora do MSVC o jemalloc é o `#[global_allocator]` de todo binário que
//! linka esta lib. No MSVC fica o alocador do sistema.

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Nome do alocador global em uso, para logs de inicialização.
pub const ALLOCATOR: &str = if cfg!(target_env = "msvc") {
    "system"
} else {
    "jemalloc"
};

/// Fotografia das estatísticas do jemalloc, em bytes.
///
/// Invariante do jemalloc: `allocated <= active <= resident <= mapped`
/// (este último a menos de páginas já devolvidas, contadas em `retained`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocStats {
    /// Bytes entregues à aplicação e ainda não liberados.
    pub allocated: usize,
    /// Bytes em páginas ativas (inclui fragmentação interna).
    pub active: usize,
    /// Bytes fisicamente residentes mantidos pelo alocador.
    pub resident: usize,
    /// Bytes mapeados pelo alocador.
    pub mapped: usize,
    /// Bytes devolvidos ao SO mas ainda reservados no espaço de endereços.
    pub retained: usize,
}

impl AllocStats {
    /// Fragmentação: fração dos bytes ativos que não estão em uso pela
    /// aplicação. `0.0` quando nada está alocado.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // métrica aproximada, f64 basta
    pub fn fragmentation(&self) -> f64 {
        if self.active == 0 {
            return 0.0;
        }
        self.active.saturating_sub(self.allocated) as f64 / self.active as f64
    }
}

/// Lê as estatísticas atuais do jemalloc.
///
/// As estatísticas são cacheadas pelo jemalloc; a leitura avança a `epoch`
/// antes, para que os números reflitam o estado deste instante. Devolve
/// `None` no MSVC (sem jemalloc) ou se a leitura via `mallctl` falhar.
#[must_use]
pub fn stats() -> Option<AllocStats> {
    #[cfg(not(target_env = "msvc"))]
    {
        use tikv_jemalloc_ctl::{epoch, stats};

        epoch::advance().ok()?;
        Some(AllocStats {
            allocated: stats::allocated::read().ok()?,
            active: stats::active::read().ok()?,
            resident: stats::resident::read().ok()?,
            mapped: stats::mapped::read().ok()?,
            retained: stats::retained::read().ok()?,
        })
    }
    #[cfg(target_env = "msvc")]
    None
}

#[cfg(all(test, not(target_env = "msvc")))]
mod tests {
    use super::*;

    #[test]
    fn jemalloc_is_the_global_allocator() {
        assert_eq!(ALLOCATOR, "jemalloc");
        // Só há estatísticas se o jemalloc estiver de fato servindo o heap.
        assert!(stats().is_some_and(|s| s.allocated > 0));
    }

    #[test]
    fn stats_respect_jemalloc_invariants() {
        let s = stats().expect("stats do jemalloc");
        assert!(s.allocated <= s.active);
        assert!(s.active <= s.resident);
        assert!(s.active <= s.mapped);
        assert!((0.0..1.0).contains(&s.fragmentation()));
    }

    #[test]
    fn allocated_grows_with_a_large_allocation() {
        const SIZE: usize = 64 * 1024 * 1024;
        let before = stats().expect("stats").allocated;
        let buf = vec![1u8; SIZE];
        let during = stats().expect("stats").allocated;
        // Outras threads de teste alocam em paralelo; a margem cobre isso.
        assert!(during >= before + SIZE / 2, "{before} -> {during}");
        drop(buf);
    }

    #[test]
    fn fragmentation_of_empty_stats_is_zero() {
        let empty = AllocStats {
            allocated: 0,
            active: 0,
            resident: 0,
            mapped: 0,
            retained: 0,
        };
        assert!(empty.fragmentation().abs() < f64::EPSILON);
    }
}
