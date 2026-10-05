use semantic_cache_rs::domain::{CacheEntry, SimilarityThreshold, Vector, VectorError};
use semantic_cache_rs::mcp::LookupRequest;

#[test]
fn test_vector_cosine_similarity() {
    let v1 = Vector::new(vec![1.0, 0.0, 0.0]).unwrap();
    let v2 = Vector::new(vec![1.0, 0.0, 0.0]).unwrap();
    let v3 = Vector::new(vec![0.0, 1.0, 0.0]).unwrap();

    let sim_identicos = v1.cosine_similarity(&v2).unwrap();
    assert!((sim_identicos - 1.0).abs() < 1e-6);

    let sim_ortogonais = v1.cosine_similarity(&v3).unwrap();
    assert!(sim_ortogonais.abs() < 1e-6);
}

#[test]
fn test_vector_dimension_mismatch() {
    let v1 = Vector::new(vec![1.0, 2.0]).unwrap();
    let v2 = Vector::new(vec![1.0, 2.0, 3.0]).unwrap();

    let err = v1.cosine_similarity(&v2).unwrap_err();
    assert_eq!(
        err,
        VectorError::DimensionMismatch {
            expected: 2,
            actual: 3
        }
    );
}

#[test]
fn test_similarity_threshold() {
    let thresh = SimilarityThreshold::default();
    assert!(thresh.is_hit(0.95));
    assert!(thresh.is_hit(0.92));
    assert!(!thresh.is_hit(0.919));
}

#[test]
fn test_cache_entry_expiration() {
    let entry = CacheEntry {
        id: "entry-1".to_string(),
        prompt: "como funciona o cache?".to_string(),
        response: "resposta rapida".to_string(),
        embedding: Vector::new(vec![0.5, 0.5]).unwrap(),
        hit_count: 0,
        created_at: 1000,
        ttl_seconds: 60,
    };

    assert!(!entry.is_expired(1050));
    assert!(entry.is_expired(1061));
}

#[test]
fn test_mcp_json_serialization() {
    let req = LookupRequest {
        prompt: "ola mundo".to_string(),
        embedding: Vector::new(vec![0.1, 0.2]).unwrap(),
        threshold: Some(0.90),
        tenant_id: Some("org-test".to_string()),
    };

    let json_str = serde_json::to_string(&req).expect("erro ao serializar json");
    let deserialized: LookupRequest =
        serde_json::from_str(&json_str).expect("erro ao desserializar");

    assert_eq!(req.prompt, deserialized.prompt);
    assert_eq!(deserialized.threshold, Some(0.90));
}
