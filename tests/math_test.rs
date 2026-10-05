use semantic_cache_rs::domain::Vector;
use semantic_cache_rs::math::{CosineDistance, NormalizedVector};
use semantic_cache_rs::metric::DistanceMetric;

#[test]
fn test_cosine_distance_identical_and_orthogonal() {
    let metric = CosineDistance;

    let v1 = Vector::new(vec![1.0, 0.0]).unwrap();
    let v2 = Vector::new(vec![1.0, 0.0]).unwrap();
    let v3 = Vector::new(vec![0.0, 1.0]).unwrap();

    // Vetores idênticos: distância cosseno = 0.0
    let dist_identica = metric.distance(&v1, &v2).unwrap();
    assert!(dist_identica < 1e-6);

    // Vetores ortogonais: similaridade = 0.0, distância = 1.0
    let dist_ortogonal = metric.distance(&v1, &v3).unwrap();
    assert!((dist_ortogonal - 1.0).abs() < 1e-6);
}

#[test]
fn test_normalized_vector_caching() {
    let raw = Vector::new(vec![3.0, 4.0]).unwrap();
    let norm_vec = NormalizedVector::new(raw).unwrap();

    assert!((norm_vec.norm - 5.0).abs() < 1e-6);

    let raw2 = Vector::new(vec![6.0, 8.0]).unwrap();
    let norm_vec2 = NormalizedVector::new(raw2).unwrap();

    let sim = norm_vec.cosine_similarity_to(&norm_vec2).unwrap();
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_is_orthogonal_and_normalize_slice() {
    let v1 = Vector::new(vec![1.0, 0.0]).unwrap();
    let v2 = Vector::new(vec![0.0, 1.0]).unwrap();
    assert!(semantic_cache_rs::math::is_orthogonal(&v1, &v2, 1e-6).unwrap());

    let mut slice = vec![3.0, 4.0];
    let norm = semantic_cache_rs::math::normalize_slice_in_place(&mut slice).unwrap();
    assert!((norm - 5.0).abs() < 1e-6);
    assert!((slice[0] - 0.6).abs() < 1e-6);
    assert!((slice[1] - 0.8).abs() < 1e-6);
}
