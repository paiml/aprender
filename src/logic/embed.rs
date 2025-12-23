//! Embedding Space for Knowledge Graph Reasoning
//!
//! Implements embedding-based relational learning:
//! - **Entity embeddings**: Each entity is a learned vector
//! - **Relation matrices**: Each relation is a learned matrix transformation
//! - **Bilinear scoring**: score(s, r, o) = s^T × W_r × o
//! - **RESCAL factorization**: X_k ≈ A × R_k × A^T
//!
//! # References
//!
//! - Nickel et al. (2011): "RESCAL: A Three-Way Model for Collective Learning"
//! - Bordes et al. (2013): "TransE: Translating Embeddings for Multi-relational Data"

use rand::Rng;
use std::collections::HashMap;

/// Embedding space for knowledge graph reasoning
#[derive(Debug)]
pub struct EmbeddingSpace {
    /// Number of entities
    num_entities: usize,
    /// Embedding dimension
    dim: usize,
    /// Entity embeddings [num_entities, dim]
    entity_embeddings: Vec<Vec<f64>>,
    /// Relation matrices [dim, dim] per relation
    relation_matrices: HashMap<String, Vec<Vec<f64>>>,
}

impl EmbeddingSpace {
    /// Create a new embedding space with random initialization
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
    pub fn get_relation_matrix(&self, name: &str) -> Option<&Vec<Vec<f64>>> {
        self.relation_matrices.get(name)
    }

    /// Score a triple (subject, relation, object) using bilinear scoring
    ///
    /// score = subject^T × W_relation × object
    pub fn score(&self, subject: usize, relation: &str, object: usize) -> f64 {
        let s = &self.entity_embeddings[subject];
        let o = &self.entity_embeddings[object];

        let Some(w) = self.relation_matrices.get(relation) else {
            return 0.0;
        };

        // Compute s^T × W × o
        // First: temp = W × o
        let temp: Vec<f64> = (0..self.dim)
            .map(|i| (0..self.dim).map(|j| w[i][j] * o[j]).sum())
            .collect();

        // Then: s^T × temp
        s.iter().zip(temp.iter()).map(|(si, ti)| si * ti).sum()
    }

    /// Compose multiple relations by matrix multiplication
    ///
    /// Example: `grandparent = compose(&[parent, parent])`
    pub fn compose_relations(&self, relations: &[&str]) -> Vec<Vec<f64>> {
        if relations.is_empty() {
            return vec![vec![0.0; self.dim]; self.dim];
        }

        let mut result = self
            .relation_matrices
            .get(relations[0])
            .cloned()
            .unwrap_or_else(|| vec![vec![0.0; self.dim]; self.dim]);

        for &rel_name in relations.iter().skip(1) {
            if let Some(m) = self.relation_matrices.get(rel_name) {
                result = matrix_multiply(&result, m);
            }
        }

        result
    }

    /// Get entity embedding
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
    pub fn num_entities(&self) -> usize {
        self.num_entities
    }

    /// Embedding dimension
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
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        Self { data }
    }

    /// Get dimension
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
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
    pub fn new(space: EmbeddingSpace) -> Self {
        Self { space }
    }

    /// Score all entities as objects for (subject, relation, ?)
    pub fn score_tails(&self, subject: usize, relation: &str) -> Vec<f64> {
        (0..self.space.num_entities)
            .map(|o| self.space.score(subject, relation, o))
            .collect()
    }

    /// Score all entities as subjects for (?, relation, object)
    pub fn score_heads(&self, relation: &str, object: usize) -> Vec<f64> {
        (0..self.space.num_entities)
            .map(|s| self.space.score(s, relation, object))
            .collect()
    }

    /// Get top-K predictions for (subject, relation, ?)
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
/// X_k ≈ A × R_k × A^T
///
/// Where:
/// - X_k is the adjacency matrix for relation k
/// - A contains entity embeddings
/// - R_k is the core tensor for relation k
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
    /// Entity embeddings [num_entities, dim]
    pub entity_embeddings: Vec<Vec<f64>>,
    /// Relation core tensors [num_relations, dim, dim]
    pub relation_cores: Vec<Vec<Vec<f64>>>,
}

impl RescalFactorizer {
    /// Create a new factorizer
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
}
