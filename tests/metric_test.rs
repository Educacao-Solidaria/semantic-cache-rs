use semantic_cache_rs::domain::Vector;
use semantic_cache_rs::metric::{
    CosineSimilarity, DistanceMetric, DotProduct, EuclideanDistance, SimilarityCalculator,
};

#[test]
fn test_cosine_similarity_metric() {
    let metric = CosineSimilarity;
    assert_eq!(metric.name(), "cosine");

    let v1 = Vector::new(vec![2.0, 0.0]).unwrap();
    let v2 = Vector::new(vec![5.0, 0.0]).unwrap();

    let sim = metric.similarity(&v1, &v2).unwrap();
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_metric() {
    let metric = EuclideanDistance;
    assert_eq!(metric.name(), "euclidean_l2");

    let v1 = Vector::new(vec![0.0, 0.0]).unwrap();
    let v2 = Vector::new(vec![3.0, 4.0]).unwrap();

    let dist = metric.distance(&v1, &v2).unwrap();
    assert!((dist - 5.0).abs() < 1e-6);
}

#[test]
fn test_dot_product_metric() {
    let metric = DotProduct;
    assert_eq!(metric.name(), "dot_product");

    let v1 = Vector::new(vec![1.0, 2.0, 3.0]).unwrap();
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]).unwrap();

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let dot = metric.similarity(&v1, &v2).unwrap();
    assert!((dot - 32.0).abs() < 1e-6);
}
