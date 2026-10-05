use semantic_cache_rs::codec::{
    deserialize_entry, deserialize_vector, serialize_entry, serialize_vector, BinaryCodec,
    BincodeCodec,
};
use semantic_cache_rs::domain::{CacheEntry, Vector};

fn create_sample_entry() -> CacheEntry {
    CacheEntry {
        id: "c-123".to_string(),
        prompt: "Qual a capital da Franca?".to_string(),
        response: "A capital da Franca eh Paris.".to_string(),
        embedding: Vector::new(vec![0.1, 0.5, 0.9, -0.3]).unwrap(),
        hit_count: 5,
        created_at: 1700000000,
        ttl_seconds: 7200,
    }
}

#[test]
fn test_entry_roundtrip() {
    let entry = create_sample_entry();
    let encoded = serialize_entry(&entry).expect("falha ao serializar");
    assert!(!encoded.is_empty());

    let decoded = deserialize_entry(&encoded).expect("falha ao desserializar");
    assert_eq!(decoded.id, entry.id);
    assert_eq!(decoded.prompt, entry.prompt);
    assert_eq!(decoded.response, entry.response);
    assert_eq!(decoded.embedding.data, entry.embedding.data);
    assert_eq!(decoded.hit_count, entry.hit_count);
    assert_eq!(decoded.ttl_seconds, entry.ttl_seconds);
}

#[test]
fn test_vector_roundtrip() {
    let vec = Vector::new(vec![1.0, 2.0, 3.5, -4.25]).unwrap();
    let encoded = serialize_vector(&vec).expect("falha ao serializar vetor");
    let decoded = deserialize_vector(&encoded).expect("falha ao desserializar vetor");
    assert_eq!(vec, decoded);
}

#[test]
fn test_max_size_enforcement() {
    let codec = BincodeCodec::new(10); // limite restrito de 10 bytes
    let entry = create_sample_entry();
    let encoded = bincode::serialize(&entry).unwrap();

    // Tentar decodificar payload maior que o limite configurado
    let res: Result<CacheEntry, _> = codec.decode(&encoded);
    assert!(res.is_err());
    let err_str = res.unwrap_err().to_string();
    assert!(err_str.contains("payload excede tamanho maximo"));
}
