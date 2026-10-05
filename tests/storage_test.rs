use semantic_cache_rs::domain::{CacheEntry, Vector};
use semantic_cache_rs::error::CacheError;
use semantic_cache_rs::storage::{CacheStore, InMemoryStore};

fn create_sample_entry(prompt: &str) -> CacheEntry {
    CacheEntry {
        id: "entry-1".to_string(),
        prompt: prompt.to_string(),
        response: "Resposta cacheada".to_string(),
        embedding: Vector::new(vec![1.0, 0.0, 0.0]).unwrap(),
        hit_count: 0,
        created_at: 1000,
        ttl_seconds: 3600,
    }
}

#[test]
fn test_in_memory_store_crud() {
    let mut store = InMemoryStore::new();
    assert!(store.is_empty());

    let entry = create_sample_entry("qual o significado da vida?");
    store.set("key-1".to_string(), entry.clone()).unwrap();

    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());

    // Get
    let fetched = store.get("key-1").unwrap().expect("chave deve existir");
    assert_eq!(fetched.prompt, "qual o significado da vida?");

    // Evict
    let evicted = store.evict("key-1").unwrap();
    assert!(evicted);
    assert_eq!(store.len(), 0);

    // Evict inexistente
    let evicted_again = store.evict("key-1").unwrap();
    assert!(!evicted_again);
}

#[test]
fn test_in_memory_store_capacity_limit() {
    let mut store = InMemoryStore::with_capacity(2);
    assert_eq!(store.capacity_limit(), Some(2));

    store
        .set("k1".to_string(), create_sample_entry("p1"))
        .unwrap();
    store
        .set("k2".to_string(), create_sample_entry("p2"))
        .unwrap();

    // Atualizar chave existente não deve exceder capacidade
    store
        .set("k1".to_string(), create_sample_entry("p1_updated"))
        .unwrap();

    // Inserir terceira chave deve falhar com CapacityExceeded
    let err = store
        .set("k3".to_string(), create_sample_entry("p3"))
        .unwrap_err();
    assert_eq!(err, CacheError::CapacityExceeded { max: 2, current: 2 });
}

#[test]
fn test_in_memory_store_clear() {
    let mut store = InMemoryStore::new();
    store
        .set("k1".to_string(), create_sample_entry("p1"))
        .unwrap();
    store
        .set("k2".to_string(), create_sample_entry("p2"))
        .unwrap();

    assert_eq!(store.len(), 2);
    store.clear().unwrap();
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
}
