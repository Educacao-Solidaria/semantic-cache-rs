//! Testes de integridade e validação dos contratos matemáticos da Fase 1.
//!
//! Valida invariantes de cálculo vetorial, métricas de distância,
//! propriedades de normalização e contratos de erro do motor semântico.

use semantic_cache_rs::domain::{Metadata, SimilarityThreshold, Vector, VectorBatch, VectorError};
use semantic_cache_rs::math::{is_orthogonal, normalize_slice_in_place, CosineDistance, NormalizedVector};
use semantic_cache_rs::metric::{CosineSimilarity, DistanceMetric, DotProduct, EuclideanDistance, SimilarityCalculator};

#[test]
fn test_contract_unit_vector_cosine_and_dot_product_equivalence() {
    // Contrato: Para vetores normalizados (norma = 1.0), Cosseno == Dot Product.
    let mut data_a = vec![0.6_f32, 0.8_f32];
    let mut data_b = vec![0.8_f32, 0.6_f32];

    normalize_slice_in_place(&mut data_a).expect("falha ao normalizar a");
    normalize_slice_in_place(&mut data_b).expect("falha ao normalizar b");

    let vec_a = Vector::new(data_a).unwrap();
    let vec_b = Vector::new(data_b).unwrap();

    let cos_metric = CosineSimilarity;
    let dot_metric = DotProduct;

    let sim_cos = cos_metric.similarity(&vec_a, &vec_b).expect("calc cosseno");
    let sim_dot = dot_metric.similarity(&vec_a, &vec_b).expect("calc dot");

    let diff = (sim_cos - sim_dot).abs();
    assert!(
        diff < 1e-6,
        "Contrato violado: Cosseno ({sim_cos}) difere de Produto Escalar ({sim_dot}) em vetores unitarios"
    );
}

#[test]
fn test_contract_euclidean_distance_axioms() {
    let metric = EuclideanDistance;
    let v_a = Vector::new(vec![1.0, 2.0, 3.0]).unwrap();
    let v_b = Vector::new(vec![4.0, 6.0, 3.0]).unwrap();
    let v_c = Vector::new(vec![1.0, 6.0, 3.0]).unwrap();

    // 1. Identidade dos indiscerníveis: d(x, x) == 0
    let d_self = metric.distance(&v_a, &v_a).unwrap();
    assert!(d_self < 1e-6, "d(x, x) deve ser zero, obteve {d_self}");

    // 2. Não-negatividade: d(x, y) >= 0
    let d_ab = metric.distance(&v_a, &v_b).unwrap();
    assert!(d_ab > 0.0, "d(a, b) deve ser estritamente positiva para a != b");

    // 3. Simetria: d(a, b) == d(b, a)
    let d_ba = metric.distance(&v_b, &v_a).unwrap();
    assert!((d_ab - d_ba).abs() < 1e-6, "Simetria violada: d(a, b) != d(b, a)");

    // 4. Desigualdade triangular: d(a, b) <= d(a, c) + d(c, b)
    let d_ac = metric.distance(&v_a, &v_c).unwrap();
    let d_cb = metric.distance(&v_c, &v_b).unwrap();
    assert!(
        d_ab <= d_ac + d_cb + 1e-6,
        "Desigualdade triangular violada: d(a,b)={d_ab} > d(a,c)={d_ac} + d(c,b)={d_cb}"
    );
}

#[test]
fn test_contract_cosine_distance_relation() {
    let cos_dist = CosineDistance;
    let cos_sim = CosineSimilarity;

    let v1 = Vector::new(vec![1.0, 2.0, 0.0]).unwrap();
    let v2 = Vector::new(vec![2.0, 1.0, 0.0]).unwrap();

    let dist = cos_dist.distance(&v1, &v2).unwrap();
    let sim = cos_sim.similarity(&v1, &v2).unwrap();

    // Contrato: distance = 1.0 - similarity
    assert!(
        (dist - (1.0 - sim)).abs() < 1e-6,
        "Contrato violado: dist ({dist}) != 1.0 - sim ({sim})"
    );
}

#[test]
fn test_contract_orthogonality_invariants() {
    let v_x = Vector::new(vec![1.0, 0.0, 0.0]).unwrap();
    let v_y = Vector::new(vec![0.0, 1.0, 0.0]).unwrap();
    let v_z = Vector::new(vec![0.0, 0.0, 1.0]).unwrap();

    assert!(is_orthogonal(&v_x, &v_y, 1e-6).unwrap());
    assert!(is_orthogonal(&v_x, &v_z, 1e-6).unwrap());
    assert!(is_orthogonal(&v_y, &v_z, 1e-6).unwrap());

    let sim_xy = CosineSimilarity.similarity(&v_x, &v_y).unwrap();
    assert!(sim_xy.abs() < 1e-6, "Vetores ortogonais devem ter cosseno zero");

    let dist_xy = CosineDistance.distance(&v_x, &v_y).unwrap();
    assert!((dist_xy - 1.0).abs() < 1e-6, "Vetores ortogonais devem ter distancia cosseno 1.0");
}

#[test]
fn test_contract_error_invariants() {
    // 1. Vetor vazio deve retornar EmptyVector
    let empty_res = Vector::new(vec![]);
    assert_eq!(empty_res.unwrap_err(), VectorError::EmptyVector);

    // 2. Dimensão incompatível
    let v2d = Vector::new(vec![1.0, 2.0]).unwrap();
    let v3d = Vector::new(vec![1.0, 2.0, 3.0]).unwrap();
    let mismatch = CosineSimilarity.similarity(&v2d, &v3d).unwrap_err();
    assert_eq!(
        mismatch,
        VectorError::DimensionMismatch {
            expected: 2,
            actual: 3,
        }
    );

    // 3. Vetor de norma zero
    let v_zero = Vector::new(vec![0.0, 0.0]).unwrap();
    let zero_norm = NormalizedVector::new(v_zero).unwrap_err();
    assert_eq!(zero_norm, VectorError::ZeroNorm);
}

#[test]
fn test_contract_batch_invariants() {
    let mut batch = VectorBatch::new(4);
    assert_eq!(batch.len(), 0);

    let v1 = Vector::new(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let v2 = Vector::new(vec![2.0, 0.0, 0.0, 0.0]).unwrap();
    assert!(batch.push(v1).is_ok());
    assert!(batch.push(v2).is_ok());

    assert_eq!(batch.len(), 2);
    assert!(batch.normalize_all().is_ok());

    for v in &batch.vectors {
        assert!((v.norm() - 1.0).abs() < 1e-6, "Vetor no batch deve ter norma unitaria");
    }

    let meta = Metadata::new("tenant-contrato", "deepseek-v3");
    assert!(meta.matches_tenant("tenant-contrato"));

    let thresh = SimilarityThreshold { min_score: 0.90 };
    assert!(thresh.is_hit(0.90));
    assert!(thresh.is_hit(0.95));
    assert!(!thresh.is_hit(0.89));
}
