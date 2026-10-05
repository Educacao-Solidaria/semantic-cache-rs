use serde::{Deserialize, Serialize};

/// Classificação do nível de correspondência semântica entre vetores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchLevel {
    /// Similaridade quase idêntica (ex: >= 0.98), resposta direta sem ressalvas.
    ExactHit,
    /// Similaridade semântica robusta (ex: >= 0.85 e < 0.98), hit válido no cache.
    SemanticHit,
    /// Similaridade limítrofe (ex: >= 0.75 e < 0.85), requer revalidação ou fallback.
    Ambiguous,
    /// Score insuficiente (ex: < 0.75), miss garantido no cache.
    Miss,
}

/// Configuração granular de faixas de limiar de similaridade vetorial.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimilarityThresholdConfig {
    pub exact_threshold: f32,
    pub semantic_threshold: f32,
    pub ambiguous_threshold: f32,
}

impl Default for SimilarityThresholdConfig {
    fn default() -> Self {
        Self {
            exact_threshold: 0.98,
            semantic_threshold: 0.88,
            ambiguous_threshold: 0.75,
        }
    }
}

impl SimilarityThresholdConfig {
    /// Instancia uma nova configuração validando a ordem decrescente dos limiares.
    pub fn new(exact: f32, semantic: f32, ambiguous: f32) -> Result<Self, &'static str> {
        if !(0.0..=1.0).contains(&exact)
            || !(0.0..=1.0).contains(&semantic)
            || !(0.0..=1.0).contains(&ambiguous)
        {
            return Err("todos os limiares devem pertencer ao intervalo [0.0, 1.0]");
        }
        if exact < semantic || semantic < ambiguous {
            return Err("os limiares devem satisfazer exact >= semantic >= ambiguous");
        }
        Ok(Self {
            exact_threshold: exact,
            semantic_threshold: semantic,
            ambiguous_threshold: ambiguous,
        })
    }

    /// Classifica um score de similaridade cosseno em um nível de correspondência.
    pub fn classify(&self, score: f32) -> MatchLevel {
        if score >= self.exact_threshold {
            MatchLevel::ExactHit
        } else if score >= self.semantic_threshold {
            MatchLevel::SemanticHit
        } else if score >= self.ambiguous_threshold {
            MatchLevel::Ambiguous
        } else {
            MatchLevel::Miss
        }
    }

    /// Retorna verdadeiro se o score for considerado um hit aproveitável (Exact ou Semantic).
    pub fn is_hit(&self, score: f32) -> bool {
        matches!(
            self.classify(score),
            MatchLevel::ExactHit | MatchLevel::SemanticHit
        )
    }
}
