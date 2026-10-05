use semantic_cache_rs::threshold::{MatchLevel, SimilarityThresholdConfig};

#[test]
fn test_default_threshold_classification() {
    let cfg = SimilarityThresholdConfig::default();

    assert_eq!(cfg.classify(0.99), MatchLevel::ExactHit);
    assert_eq!(cfg.classify(0.98), MatchLevel::ExactHit);
    assert_eq!(cfg.classify(0.95), MatchLevel::SemanticHit);
    assert_eq!(cfg.classify(0.88), MatchLevel::SemanticHit);
    assert_eq!(cfg.classify(0.80), MatchLevel::Ambiguous);
    assert_eq!(cfg.classify(0.75), MatchLevel::Ambiguous);
    assert_eq!(cfg.classify(0.70), MatchLevel::Miss);
    assert_eq!(cfg.classify(0.10), MatchLevel::Miss);

    assert!(cfg.is_hit(0.99));
    assert!(cfg.is_hit(0.90));
    assert!(!cfg.is_hit(0.80));
    assert!(!cfg.is_hit(0.50));
}

#[test]
fn test_custom_threshold_validations() {
    // Configuração válida
    let valid = SimilarityThresholdConfig::new(0.95, 0.85, 0.70);
    assert!(valid.is_ok());

    // Fora do intervalo [0.0, 1.0]
    let invalid_bound = SimilarityThresholdConfig::new(1.05, 0.85, 0.70);
    assert!(invalid_bound.is_err());

    // Ordem invertida (exact < semantic)
    let invalid_order = SimilarityThresholdConfig::new(0.80, 0.90, 0.70);
    assert!(invalid_order.is_err());

    // Ordem invertida (semantic < ambiguous)
    let invalid_order2 = SimilarityThresholdConfig::new(0.95, 0.60, 0.70);
    assert!(invalid_order2.is_err());
}

#[test]
fn test_threshold_serialization() {
    let cfg = SimilarityThresholdConfig::default();
    let json = serde_json::to_string(&cfg).expect("serializacao json");
    let deserialized: SimilarityThresholdConfig =
        serde_json::from_str(&json).expect("desserializacao json");
    assert_eq!(cfg, deserialized);
}
