use semantic_cache_rs::domain::VectorError;
use semantic_cache_rs::math::{
    dot_product, euclidean_distance, normalized_dot_product, squared_euclidean_distance,
};

#[test]
fn test_squared_euclidean_distance() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 6.0, 3.0];
    // diffs: (3)^2 + (4)^2 + (0)^2 = 9 + 16 = 25
    let sq = squared_euclidean_distance(&a, &b).unwrap();
    assert!((sq - 25.0).abs() < 1e-5);
}

#[test]
fn test_euclidean_distance_identical_and_known() {
    let a = [0.0, 0.0, 0.0];
    let b = [3.0, 4.0, 0.0];
    let dist = euclidean_distance(&a, &b).unwrap();
    assert!((dist - 5.0).abs() < 1e-5);

    // Vetores idênticos devem ter distância zero
    let zero_dist = euclidean_distance(&a, &a).unwrap();
    assert!(zero_dist.abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_dimension_mismatch() {
    let a = [1.0, 2.0];
    let b = [1.0, 2.0, 3.0];
    let res = euclidean_distance(&a, &b);
    assert_eq!(
        res,
        Err(VectorError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    );
}

#[test]
fn test_dot_product_and_normalized() {
    let a = [1.0, 0.0, 0.0];
    let b = [0.0, 1.0, 0.0];
    // Ortogonais: dot product = 0
    let dot = dot_product(&a, &b).unwrap();
    assert!(dot.abs() < 1e-6);

    let v1 = [2.0, 0.0];
    let v2 = [3.0, 0.0];
    let norm_dot = normalized_dot_product(&v1, &v2).unwrap();
    assert!((norm_dot - 1.0).abs() < 1e-5);
}

#[test]
fn test_normalized_dot_product_zero_norm() {
    let a = [0.0, 0.0];
    let b = [1.0, 1.0];
    let res = normalized_dot_product(&a, &b);
    assert_eq!(res, Err(VectorError::ZeroNorm));
}
