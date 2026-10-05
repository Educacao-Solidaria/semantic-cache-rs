use semantic_cache_rs::domain::{Metadata, Vector, VectorBatch, VectorError};

#[test]
fn test_metadata_tenant_matching() {
    let meta = Metadata::new("tenant-abc", "deepseek-v3")
        .with_tag("producao")
        .with_attribute("region", "sa-east-1");

    assert!(meta.matches_tenant("tenant-abc"));
    assert!(!meta.matches_tenant("tenant-xyz"));
    assert_eq!(meta.tags.len(), 1);
    assert_eq!(meta.attributes.get("region").unwrap(), "sa-east-1");
}

#[test]
fn test_vector_batch_operations() {
    let mut batch = VectorBatch::new(3);

    let v1 = Vector::new(vec![3.0, 0.0, 0.0]).unwrap();
    let v2 = Vector::new(vec![0.0, 4.0, 0.0]).unwrap();

    assert!(batch.push(v1).is_ok());
    assert!(batch.push(v2).is_ok());

    // Dimensão incompatível deve falhar
    let invalid_v = Vector::new(vec![1.0, 2.0]).unwrap();
    assert_eq!(
        batch.push(invalid_v).unwrap_err(),
        VectorError::DimensionMismatch {
            expected: 3,
            actual: 2
        }
    );

    assert_eq!(batch.len(), 2);

    // Normalização em lote
    assert!(batch.normalize_all().is_ok());
    assert!((batch.vectors[0].norm() - 1.0).abs() < 1e-6);
    assert!((batch.vectors[1].norm() - 1.0).abs() < 1e-6);
}
