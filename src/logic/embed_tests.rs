pub(crate) use super::*;

#[test]
fn test_embedding_space_creation() {
    let space = EmbeddingSpace::new(10, 8);
    assert_eq!(space.num_entities(), 10);
    assert_eq!(space.dim(), 8);
}

#[test]
fn test_relation_matrix() {
    let mut space = EmbeddingSpace::new(5, 4);
    space.add_relation("knows");

    let matrix = space.get_relation_matrix("knows");
    assert!(matrix.is_some());
    assert_eq!(matrix.unwrap().len(), 4);
}

#[test]
fn test_bilinear_scoring() {
    let mut space = EmbeddingSpace::new(3, 4);
    space.add_relation("likes");

    let score = space.score(0, "likes", 1);
    assert!(score.is_finite());
}

#[test]
fn test_relation_composition() {
    let mut space = EmbeddingSpace::new(3, 4);
    space.add_relation("parent");

    let composed = space.compose_relations(&["parent", "parent"]);
    assert_eq!(composed.len(), 4);
    assert_eq!(composed[0].len(), 4);
}

#[test]
fn test_rescal_factorization() {
    let factorizer = RescalFactorizer::new(5, 4, 2);

    let triples = vec![(0, 0, 1), (1, 0, 2), (2, 1, 3)];

    let result = factorizer.factorize(&triples, 5);

    assert_eq!(result.entity_embeddings.len(), 5);
    assert_eq!(result.relation_cores.len(), 2);
}

#[test]
fn test_bilinear_scorer_predictions() {
    let mut space = EmbeddingSpace::new(5, 4);
    space.add_relation("knows");

    let scorer = BilinearScorer::new(space);
    let predictions = scorer.predict_tails(0, "knows", 3);

    assert_eq!(predictions.len(), 3);
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_score_unknown_relation() {
    let space = EmbeddingSpace::new(3, 4);
    // No relations added
    let score = space.score(0, "unknown", 1);
    assert_eq!(score, 0.0);
}

#[test]
fn test_get_entity() {
    let space = EmbeddingSpace::new(5, 4);
    assert!(space.get_entity(0).is_some());
    assert!(space.get_entity(4).is_some());
    assert!(space.get_entity(5).is_none()); // Out of bounds
}

#[test]
fn test_set_entity() {
    let mut space = EmbeddingSpace::new(3, 4);
    let new_embedding = vec![1.0, 2.0, 3.0, 4.0];
    space.set_entity(0, new_embedding.clone());
    assert_eq!(space.get_entity(0).unwrap(), &new_embedding);
}

#[test]
fn test_set_entity_invalid_index() {
    let mut space = EmbeddingSpace::new(3, 4);
    let orig = space.get_entity(0).unwrap().clone();
    // Out of bounds - should do nothing
    space.set_entity(10, vec![1.0, 2.0, 3.0, 4.0]);
    // Entity 0 unchanged
    assert_eq!(space.get_entity(0).unwrap(), &orig);
}

#[test]
fn test_set_entity_wrong_dimension() {
    let mut space = EmbeddingSpace::new(3, 4);
    let orig = space.get_entity(0).unwrap().clone();
    // Wrong dimension (3 instead of 4) - should do nothing
    space.set_entity(0, vec![1.0, 2.0, 3.0]);
    assert_eq!(space.get_entity(0).unwrap(), &orig);
}

#[test]
fn test_compose_relations_empty() {
    let space = EmbeddingSpace::new(3, 4);
    let composed = space.compose_relations(&[]);
    // Should return zero matrix
    assert_eq!(composed.len(), 4);
    for row in &composed {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
}

#[test]
fn test_compose_relations_unknown() {
    let space = EmbeddingSpace::new(3, 4);
    // Unknown relation - should return zero matrix
    let composed = space.compose_relations(&["unknown"]);
    assert_eq!(composed.len(), 4);
}

#[test]
fn test_compose_relations_mixed() {
    let mut space = EmbeddingSpace::new(3, 4);
    space.add_relation("parent");
    // One known, one unknown
    let composed = space.compose_relations(&["parent", "unknown"]);
    assert_eq!(composed.len(), 4);
}

#[test]
fn test_relation_matrix_wrapper() {
    let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let rm = RelationMatrix::new(data.clone());
    assert_eq!(rm.len(), 2);
    assert!(!rm.is_empty());
    assert_eq!(rm.data, data);
}

#[test]
fn test_relation_matrix_empty() {
    let rm = RelationMatrix::new(vec![]);
    assert!(rm.is_empty());
    assert_eq!(rm.len(), 0);
}

#[test]
fn test_relation_matrix_debug_clone() {
    let rm = RelationMatrix::new(vec![vec![1.0]]);
    let debug_str = format!("{:?}", rm);
    assert!(debug_str.contains("RelationMatrix"));
    let cloned = rm.clone();
    assert_eq!(cloned.data, rm.data);
}

#[test]
fn test_bilinear_scorer_score_heads() {
    let mut space = EmbeddingSpace::new(5, 4);
    space.add_relation("knows");

    let scorer = BilinearScorer::new(space);
    let scores = scorer.score_heads("knows", 0);
    assert_eq!(scores.len(), 5);
}

#[test]
fn test_bilinear_scorer_score_tails() {
    let mut space = EmbeddingSpace::new(5, 4);
    space.add_relation("knows");

    let scorer = BilinearScorer::new(space);
    let scores = scorer.score_tails(0, "knows");
    assert_eq!(scores.len(), 5);
}

#[test]
fn test_bilinear_scorer_debug() {
    let space = EmbeddingSpace::new(3, 2);
    let scorer = BilinearScorer::new(space);
    let debug_str = format!("{:?}", scorer);
    assert!(debug_str.contains("BilinearScorer"));
}

#[test]
fn test_rescal_result_debug() {
    let factorizer = RescalFactorizer::new(3, 2, 1);
    let result = factorizer.factorize(&[(0, 0, 1)], 1);
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("RescalResult"));
}

#[test]
fn test_rescal_factorizer_debug() {
    let factorizer = RescalFactorizer::new(3, 2, 1);
    let debug_str = format!("{:?}", factorizer);
    assert!(debug_str.contains("RescalFactorizer"));
}

#[test]
fn test_rescal_triples_out_of_bounds() {
    let factorizer = RescalFactorizer::new(3, 2, 1);
    // Triples with out-of-bounds indices should be ignored
    let result = factorizer.factorize(&[(0, 0, 1), (10, 0, 1), (0, 5, 1)], 2);
    assert_eq!(result.entity_embeddings.len(), 3);
}

#[test]
fn test_embedding_space_debug() {
    let space = EmbeddingSpace::new(2, 2);
    let debug_str = format!("{:?}", space);
    assert!(debug_str.contains("EmbeddingSpace"));
}

#[test]
fn test_matrix_multiply_edge_cases() {
    // Empty matrices
    let empty: Vec<Vec<f64>> = vec![];
    let result = matrix_multiply(&empty, &empty);
    assert!(result.is_empty());

    // Single element
    let a = vec![vec![2.0]];
    let b = vec![vec![3.0]];
    let result = matrix_multiply(&a, &b);
    assert!((result[0][0] - 6.0).abs() < 1e-10);
}
