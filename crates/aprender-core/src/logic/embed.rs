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

/// Generate a random vector of given dimension with values in `[-0.1, 0.1)`.
fn rand_vec(rng: &mut impl Rng, dim: usize) -> Vec<f64> {
    (0..dim).map(|_| rng.random_range(-0.1..0.1)).collect()
}

/// Generate a random matrix (rows x cols) with values in `[-0.1, 0.1)`.
fn rand_matrix(rng: &mut impl Rng, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows).map(|_| rand_vec(rng, cols)).collect()
}

/// Score all entities by varying either the subject or object position.
///
/// When `vary_subject` is true, iterates subjects for a fixed object;
/// otherwise iterates objects for a fixed subject.
fn score_all_entities(
    space: &EmbeddingSpace,
    fixed: usize,
    relation: &str,
    vary_subject: bool,
) -> Vec<f64> {
    (0..space.num_entities)
        .map(|i| {
            if vary_subject {
                space.score(i, relation, fixed)
            } else {
                space.score(fixed, relation, i)
            }
        })
        .collect()
}

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
        let mut rng = rand::rng();

        Self {
            num_entities,
            dim,
            entity_embeddings: rand_matrix(&mut rng, num_entities, dim),
            relation_matrices: HashMap::new(),
        }
    }

    /// Add a relation with random initialization
    pub fn add_relation(&mut self, name: &str) {
        let mut rng = rand::rng();
        self.relation_matrices
            .insert(name.to_string(), rand_matrix(&mut rng, self.dim, self.dim));
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
        score_all_entities(&self.space, subject, relation, false)
    }

    /// Score all entities as subjects for (?, relation, object)
    #[must_use]
    pub fn score_heads(&self, relation: &str, object: usize) -> Vec<f64> {
        score_all_entities(&self.space, object, relation, true)
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
        let mut rng = rand::rng();

        let mut a = rand_matrix(&mut rng, self.num_entities, self.dim);
        let r: Vec<Vec<Vec<f64>>> = (0..self.num_relations)
            .map(|_| rand_matrix(&mut rng, self.dim, self.dim))
            .collect();

        let x = build_adjacency_tensors(triples, self.num_relations, self.num_entities);

        for _ in 0..iterations {
            als_update_a(&mut a, &r, &x, self.dim);
            normalize_rows(&mut a);
        }

        RescalResult {
            entity_embeddings: a,
            relation_cores: r,
        }
    }
}

/// Build adjacency tensors from triples. Returns `x[relation][head][tail]`.
fn build_adjacency_tensors(
    triples: &[(usize, usize, usize)],
    num_relations: usize,
    num_entities: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut x = vec![vec![vec![0.0; num_entities]; num_entities]; num_relations];
    for &(h, rel, t) in triples {
        if rel < num_relations && h < num_entities && t < num_entities {
            x[rel][h][t] = 1.0;
        }
    }
    x
}

/// Simplified ALS update for entity embeddings A.
fn als_update_a(a: &mut [Vec<f64>], r: &[Vec<Vec<f64>>], x: &[Vec<Vec<f64>>], dim: usize) {
    let num_entities = a.len();
    let num_relations = r.len();
    for i in 0..num_entities {
        for d in 0..dim {
            let mut sum = 0.0;
            for k in 0..num_relations {
                for j in 0..num_entities {
                    if x[k][i][j] > 0.0 {
                        sum += r[k][d][0] * a[j][d];
                    }
                }
            }
            a[i][d] = a[i][d] * 0.9 + sum * 0.1;
        }
    }
}

/// Normalize each row (entity embedding) to unit L2 norm.
fn normalize_rows(a: &mut [Vec<f64>]) {
    for embedding in a.iter_mut() {
        let norm: f64 = embedding.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-6 {
            for v in embedding.iter_mut() {
                *v /= norm;
            }
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
#[path = "embed_tests.rs"]
mod tests;
