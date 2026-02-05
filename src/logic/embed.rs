//! Embedding Space for Knowledge Graph Reasoning
//!
//! Implements embedding-based relational learning:
//! - **Entity embeddings**: Each entity is a learned vector
//! - **Relation matrices**: Each relation is a learned matrix transformation
//! - **Bilinear scoring**: score(s, r, o) = s^T × `W_r` × o
//! - **RESCAL factorization**: `X_k` ≈ A × `R_k` × A^T
//!
//! # References
//!
//! - Nickel et al. (2011): "RESCAL: A Three-Way Model for Collective Learning"
//! - Bordes et al. (2013): "`TransE`: Translating Embeddings for Multi-relational Data"

use rand::Rng;
use std::collections::HashMap;

/// Embedding space for knowledge graph reasoning
#[derive(Debug)]
pub struct EmbeddingSpace {
    /// Number of entities
    num_entities: usize,
    /// Embedding dimension
    dim: usize,
    /// Entity embeddings [`num_entities`, dim]
    entity_embeddings: Vec<Vec<f64>>,
    /// Relation matrices [dim, dim] per relation
    relation_matrices: HashMap<String, Vec<Vec<f64>>>,
}

impl EmbeddingSpace {
    /// Create a new embedding space with random initialization
    #[must_use]
    pub fn new(num_entities: usize, dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize entity embeddings randomly
        let entity_embeddings: Vec<Vec<f64>> = (0..num_entities)
            .map(|_| (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        Self {
            num_entities,
            dim,
            entity_embeddings,
            relation_matrices: HashMap::new(),
        }
    }

    /// Add a relation with random initialization
    pub fn add_relation(&mut self, name: &str) {
        let mut rng = rand::thread_rng();

        // Initialize relation matrix randomly
        let matrix: Vec<Vec<f64>> = (0..self.dim)
            .map(|_| (0..self.dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        self.relation_matrices.insert(name.to_string(), matrix);
    }

    /// Get a relation matrix by name
    #[must_use]
    pub fn get_relation_matrix(&self, name: &str) -> Option<&Vec<Vec<f64>>> {
        self.relation_matrices.get(name)
    }

    /// Score a triple (subject, relation, object) using bilinear scoring
    ///
    /// score = subject^T × `W_relation` × object
    #[must_use]
    pub fn score(&self, subject: usize, relation: &str, object: usize) -> f64 {
        let s = &self.entity_embeddings[subject];
        let o = &self.entity_embeddings[object];

        let Some(w) = self.relation_matrices.get(relation) else {
            return 0.0;
        };

        // Compute s^T × W × o via two matrix-vector products
        let temp: Vec<f64> = (0..self.dim)
            .map(|i| (0..self.dim).map(|j| w[i][j] * o[j]).sum())
            .collect();

        // s^T × temp gives final score
        s.iter().zip(temp.iter()).map(|(si, ti)| si * ti).sum()
    }

    /// Compose multiple relations by matrix multiplication
    ///
    /// Example: `grandparent = compose(&[parent, parent])`
    #[must_use]
    pub fn compose_relations(&self, relations: &[&str]) -> Vec<Vec<f64>> {
        if relations.is_empty() {
            return vec![vec![0.0; self.dim]; self.dim];
        }

        // Return zero matrix if first relation unknown (graceful handling)
        let Some(mut result) = self.relation_matrices.get(relations[0]).cloned() else {
            return vec![vec![0.0; self.dim]; self.dim];
        };

        for &rel_name in relations.iter().skip(1) {
            if let Some(m) = self.relation_matrices.get(rel_name) {
                result = matrix_multiply(&result, m);
            }
        }

        result
    }

    /// Get entity embedding
    #[must_use]
    pub fn get_entity(&self, idx: usize) -> Option<&Vec<f64>> {
        self.entity_embeddings.get(idx)
    }

    /// Set entity embedding
    pub fn set_entity(&mut self, idx: usize, embedding: Vec<f64>) {
        if idx < self.num_entities && embedding.len() == self.dim {
            self.entity_embeddings[idx] = embedding;
        }
    }

    /// Number of entities
    #[must_use]
    pub fn num_entities(&self) -> usize {
        self.num_entities
    }

    /// Embedding dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Relation matrix wrapper (for type-safe access)
#[derive(Debug, Clone)]
pub struct RelationMatrix {
    /// Matrix data [dim, dim]
    pub data: Vec<Vec<f64>>,
}

impl RelationMatrix {
    /// Create from data
    #[must_use]
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        Self { data }
    }

    /// Get dimension
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Bilinear scorer for knowledge graph completion
#[derive(Debug)]
pub struct BilinearScorer {
    space: EmbeddingSpace,
}

impl BilinearScorer {
    /// Create a new scorer
    #[must_use]
    pub fn new(space: EmbeddingSpace) -> Self {
        Self { space }
    }

    /// Score all entities as objects for (subject, relation, ?)
    #[must_use]
    pub fn score_tails(&self, subject: usize, relation: &str) -> Vec<f64> {
        (0..self.space.num_entities)
            .map(|o| self.space.score(subject, relation, o))
            .collect()
    }

    /// Score all entities as subjects for (?, relation, object)
    #[must_use]
    pub fn score_heads(&self, relation: &str, object: usize) -> Vec<f64> {
        (0..self.space.num_entities)
            .map(|s| self.space.score(s, relation, object))
            .collect()
    }

    /// Get top-K predictions for (subject, relation, ?)
    #[must_use]
    pub fn predict_tails(&self, subject: usize, relation: &str, k: usize) -> Vec<(usize, f64)> {
        let scores = self.score_tails(subject, relation);
        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
}

/// RESCAL Tensor Factorization for predicate invention
///
/// Decomposes a 3-way tensor X into:
/// `X_k` ≈ A × `R_k` × A^T
///
/// Where:
/// - `X_k` is the adjacency matrix for relation k
/// - A contains entity embeddings
/// - `R_k` is the core tensor for relation k
#[derive(Debug)]
pub struct RescalFactorizer {
    /// Number of entities
    num_entities: usize,
    /// Embedding dimension
    dim: usize,
    /// Number of relations (including latent)
    num_relations: usize,
}

/// Result of RESCAL factorization
#[derive(Debug)]
pub struct RescalResult {
    /// Entity embeddings [`num_entities`, dim]
    pub entity_embeddings: Vec<Vec<f64>>,
    /// Relation core tensors [`num_relations`, dim, dim]
    pub relation_cores: Vec<Vec<Vec<f64>>>,
}

impl RescalFactorizer {
    /// Create a new factorizer
    #[must_use]
    pub fn new(num_entities: usize, dim: usize, num_relations: usize) -> Self {
        Self {
            num_entities,
            dim,
            num_relations,
        }
    }

    /// Factorize a set of triples
    ///
    /// # Arguments
    /// * `triples` - List of (head, relation, tail) indices
    /// * `iterations` - Number of ALS iterations
    #[must_use]
    pub fn factorize(&self, triples: &[(usize, usize, usize)], iterations: usize) -> RescalResult {
        let mut rng = rand::thread_rng();

        // Initialize A randomly
        let mut a: Vec<Vec<f64>> = (0..self.num_entities)
            .map(|_| (0..self.dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        // Initialize R_k randomly
        let r: Vec<Vec<Vec<f64>>> = (0..self.num_relations)
            .map(|_| {
                (0..self.dim)
                    .map(|_| (0..self.dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
                    .collect()
            })
            .collect();

        // Build adjacency tensors from triples
        let mut x: Vec<Vec<Vec<f64>>> =
            vec![vec![vec![0.0; self.num_entities]; self.num_entities]; self.num_relations];
        for &(h, rel, t) in triples {
            if rel < self.num_relations && h < self.num_entities && t < self.num_entities {
                x[rel][h][t] = 1.0;
            }
        }

        // Simplified ALS iteration (for demonstration)
        for _ in 0..iterations {
            // Update A (simplified)
            for i in 0..self.num_entities {
                for d in 0..self.dim {
                    let mut sum = 0.0;
                    for k in 0..self.num_relations {
                        for j in 0..self.num_entities {
                            if x[k][i][j] > 0.0 {
                                sum += r[k][d][0] * a[j][d];
                            }
                        }
                    }
                    a[i][d] = a[i][d] * 0.9 + sum * 0.1; // Momentum update
                }
            }

            // Normalize A
            for embedding in &mut a {
                let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-6 {
                    for v in embedding.iter_mut() {
                        *v /= norm;
                    }
                }
            }
        }

        RescalResult {
            entity_embeddings: a,
            relation_cores: r,
        }
    }
}

/// Matrix multiplication helper
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let inner = if a.is_empty() { 0 } else { a[0].len() };
    let cols = if b.is_empty() { 0 } else { b[0].len() };

    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
