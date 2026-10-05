use semantic_cache_rs::domain::VectorError;
use semantic_cache_rs::error::CacheError;

#[test]
fn test_cache_error_from_vector_error() {
    let vec_err = VectorError::EmptyVector;
    let cache_err: CacheError = vec_err.into();

    assert_eq!(cache_err, CacheError::Vector(VectorError::EmptyVector));
    assert!(cache_err.to_string().contains("vetor nao pode ser vazio"));
}

#[test]
fn test_cache_error_display_formatting() {
    let err = CacheError::CapacityExceeded {
        max: 1000,
        current: 1001,
    };
    assert_eq!(
        err.to_string(),
        "capacidade maxima do cache atingida: atual 1001, maximo 1000"
    );

    let not_found = CacheError::EntryNotFound("prompt_hash_123".to_string());
    assert_eq!(
        not_found.to_string(),
        "entrada com chave 'prompt_hash_123' nao encontrada"
    );
}

#[test]
fn test_cache_error_helpers() {
    let err = CacheError::capacity_exceeded(100, 105);
    assert!(err.is_recoverable());

    let corrupted = CacheError::corrupted("checksum invalido");
    assert!(!corrupted.is_recoverable());
}
