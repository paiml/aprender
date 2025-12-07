//! Code2Vec embedding encoder
//!
//! Implements the embedding layer from code2vec:
//! - Maps tokens to dense vectors
//! - Maps path node sequences to dense vectors
//! - Aggregates path embeddings using attention mechanism

use super::ast::AstNodeType;
use super::path::AstPath;
use crate::primitives::Vector;
use std::collections::HashMap;

/// Code embedding representation
#[derive(Debug, Clone)]
pub struct CodeEmbedding {
    /// The embedding vector
    vector: Vector<f64>,
    /// Optional attention weights for interpretability
    attention_weights: Option<Vec<f64>>,
}

impl CodeEmbedding {
    /// Create a new code embedding
    #[must_use]
    pub fn new(vector: Vector<f64>) -> Self {
        Self {
            vector,
            attention_weights: None,
        }
    }

    /// Create embedding with attention weights
    #[must_use]
    pub fn with_attention(vector: Vector<f64>, weights: Vec<f64>) -> Self {
        Self {
            vector,
            attention_weights: Some(weights),
        }
    }

    /// Get the embedding dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Get the embedding vector
    #[must_use]
    pub fn vector(&self) -> &Vector<f64> {
        &self.vector
    }

    /// Get attention weights if available
    #[must_use]
    pub fn attention_weights(&self) -> Option<&[f64]> {
        self.attention_weights.as_deref()
    }

    /// Compute cosine similarity with another embedding
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        if self.dim() != other.dim() {
            return 0.0;
        }

        let dot: f64 = self
            .vector
            .as_slice()
            .iter()
            .zip(other.vector.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f64 = self
            .vector
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let norm_other: f64 = other
            .vector
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        if norm_self < 1e-10 || norm_other < 1e-10 {
            0.0
        } else {
            dot / (norm_self * norm_other)
        }
    }
}

/// Code2Vec encoder for generating embeddings from AST paths
#[derive(Debug)]
pub struct Code2VecEncoder {
    /// Embedding dimension
    dim: usize,
    /// Token vocabulary embeddings
    token_embeddings: HashMap<String, Vec<f64>>,
    /// Path vocabulary embeddings
    path_embeddings: HashMap<String, Vec<f64>>,
    /// Random seed for reproducible embeddings
    seed: u64,
}

impl Code2VecEncoder {
    /// Create a new Code2Vec encoder
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            token_embeddings: HashMap::new(),
            path_embeddings: HashMap::new(),
            seed: 42,
        }
    }

    /// Set random seed for reproducible embeddings
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Get the embedding dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode a single path to an embedding vector
    #[must_use]
    pub fn encode_path(&self, path: &AstPath) -> Vec<f64> {
        // Get or generate embeddings for source, path, and target
        let source_emb = self.get_token_embedding(path.source().value());
        let path_emb = self.get_path_embedding(path.path_nodes());
        let target_emb = self.get_token_embedding(path.target().value());

        // Combine embeddings: concatenate and project to dim
        // In practice, this would use learned weights
        self.combine_embeddings(&source_emb, &path_emb, &target_emb)
    }

    /// Aggregate multiple path embeddings into a single code embedding
    #[must_use]
    pub fn aggregate_paths(&self, paths: &[AstPath]) -> CodeEmbedding {
        if paths.is_empty() {
            return CodeEmbedding::new(Vector::from_vec(vec![0.0; self.dim]));
        }

        // Encode all paths
        let path_embeddings: Vec<Vec<f64>> = paths.iter().map(|p| self.encode_path(p)).collect();

        // Compute attention weights (simplified: uniform attention for now)
        let attention_weights = Self::compute_attention(&path_embeddings);

        // Weighted sum of path embeddings
        let mut aggregated = vec![0.0; self.dim];
        for (emb, &weight) in path_embeddings.iter().zip(attention_weights.iter()) {
            for (i, &val) in emb.iter().enumerate() {
                aggregated[i] += val * weight;
            }
        }

        CodeEmbedding::with_attention(Vector::from_vec(aggregated), attention_weights)
    }

    /// Get or generate embedding for a token
    fn get_token_embedding(&self, token: &str) -> Vec<f64> {
        if let Some(emb) = self.token_embeddings.get(token) {
            emb.clone()
        } else {
            // Generate deterministic random embedding based on token hash
            self.generate_embedding(token)
        }
    }

    /// Get or generate embedding for a path
    fn get_path_embedding(&self, path_nodes: &[AstNodeType]) -> Vec<f64> {
        let path_key: String = path_nodes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("_");

        if let Some(emb) = self.path_embeddings.get(&path_key) {
            emb.clone()
        } else {
            // Generate deterministic random embedding based on path hash
            self.generate_embedding(&path_key)
        }
    }

    /// Generate a deterministic pseudo-random embedding
    fn generate_embedding(&self, key: &str) -> Vec<f64> {
        // Simple hash-based pseudo-random embedding
        // In practice, this would be learned during training
        let mut embedding = Vec::with_capacity(self.dim);
        let hash = self.hash_string(key);

        for i in 0..self.dim {
            // Use XORShift-like mixing
            let mixed = hash.wrapping_mul(0x5851_f42d_4c95_7f2d_u64.wrapping_add(i as u64));
            let val = ((mixed >> 32) as f64) / f64::from(u32::MAX) * 2.0 - 1.0;
            // Xavier initialization scale
            let scale = (2.0 / self.dim as f64).sqrt();
            embedding.push(val * scale);
        }

        embedding
    }

    /// Simple string hash function
    fn hash_string(&self, s: &str) -> u64 {
        let mut hash = self.seed;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(u64::from(byte));
        }
        hash
    }

    /// Combine source, path, and target embeddings
    fn combine_embeddings(&self, source: &[f64], path: &[f64], target: &[f64]) -> Vec<f64> {
        // Average pooling across the three components
        // In practice, this would use learned projection weights
        let mut combined = vec![0.0; self.dim];
        for i in 0..self.dim {
            combined[i] = (source[i] + path[i] + target[i]) / 3.0;
        }
        combined
    }

    /// Compute attention weights for path embeddings
    fn compute_attention(embeddings: &[Vec<f64>]) -> Vec<f64> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        // Simplified attention: compute L2 norm of each embedding as "importance"
        // In practice, this would use learned attention weights
        let scores: Vec<f64> = embeddings
            .iter()
            .map(|emb| {
                let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                norm
            })
            .collect();

        // Softmax normalization
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        if sum_exp < 1e-10 {
            // Uniform weights if sum is too small
            vec![1.0 / embeddings.len() as f64; embeddings.len()]
        } else {
            exp_scores.iter().map(|e| e / sum_exp).collect()
        }
    }

    /// Add a token embedding to the vocabulary
    pub fn add_token_embedding(&mut self, token: impl Into<String>, embedding: Vec<f64>) {
        assert_eq!(embedding.len(), self.dim, "Embedding dimension mismatch");
        self.token_embeddings.insert(token.into(), embedding);
    }

    /// Add a path embedding to the vocabulary
    pub fn add_path_embedding(&mut self, path_key: impl Into<String>, embedding: Vec<f64>) {
        assert_eq!(embedding.len(), self.dim, "Embedding dimension mismatch");
        self.path_embeddings.insert(path_key.into(), embedding);
    }

    /// Get vocabulary size for tokens
    #[must_use]
    pub fn token_vocab_size(&self) -> usize {
        self.token_embeddings.len()
    }

    /// Get vocabulary size for paths
    #[must_use]
    pub fn path_vocab_size(&self) -> usize {
        self.path_embeddings.len()
    }
}

impl Default for Code2VecEncoder {
    fn default() -> Self {
        Self::new(super::DEFAULT_EMBEDDING_DIM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::ast::{Token, TokenType};

    #[test]
    fn test_code_embedding_creation() {
        let vec = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let emb = CodeEmbedding::new(vec);

        assert_eq!(emb.dim(), 3);
        assert!(emb.attention_weights().is_none());
    }

    #[test]
    fn test_code_embedding_with_attention() {
        let vec = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        let emb = CodeEmbedding::with_attention(vec, weights);

        assert_eq!(emb.dim(), 4);
        assert!(emb.attention_weights().is_some());
    }

    #[test]
    fn test_cosine_similarity() {
        let emb1 = CodeEmbedding::new(Vector::from_vec(vec![1.0, 0.0, 0.0]));
        let emb2 = CodeEmbedding::new(Vector::from_vec(vec![1.0, 0.0, 0.0]));
        let emb3 = CodeEmbedding::new(Vector::from_vec(vec![0.0, 1.0, 0.0]));

        // Same vector: similarity = 1
        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 1e-6);

        // Orthogonal vectors: similarity = 0
        assert!(emb1.cosine_similarity(&emb3).abs() < 1e-6);
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = Code2VecEncoder::new(128);
        assert_eq!(encoder.dim(), 128);
        assert_eq!(encoder.token_vocab_size(), 0);
        assert_eq!(encoder.path_vocab_size(), 0);
    }

    #[test]
    fn test_encode_path() {
        let encoder = Code2VecEncoder::new(64);

        let path = AstPath::new(
            Token::new(TokenType::Identifier, "x"),
            vec![
                AstNodeType::Parameter,
                AstNodeType::Function,
                AstNodeType::Return,
            ],
            Token::new(TokenType::Identifier, "result"),
        );

        let embedding = encoder.encode_path(&path);
        assert_eq!(embedding.len(), 64);
    }

    #[test]
    fn test_deterministic_embedding() {
        let encoder1 = Code2VecEncoder::new(32).with_seed(123);
        let encoder2 = Code2VecEncoder::new(32).with_seed(123);

        let path = AstPath::new(
            Token::new(TokenType::Identifier, "foo"),
            vec![AstNodeType::Function],
            Token::new(TokenType::Identifier, "bar"),
        );

        let emb1 = encoder1.encode_path(&path);
        let emb2 = encoder2.encode_path(&path);

        // Same seed should produce same embeddings
        for (a, b) in emb1.iter().zip(emb2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_aggregate_paths() {
        let encoder = Code2VecEncoder::new(32);

        let paths = vec![
            AstPath::new(
                Token::new(TokenType::Identifier, "a"),
                vec![AstNodeType::Parameter, AstNodeType::Function],
                Token::new(TokenType::Identifier, "b"),
            ),
            AstPath::new(
                Token::new(TokenType::Identifier, "c"),
                vec![AstNodeType::Return, AstNodeType::Function],
                Token::new(TokenType::Identifier, "d"),
            ),
        ];

        let embedding = encoder.aggregate_paths(&paths);
        assert_eq!(embedding.dim(), 32);
        assert!(embedding.attention_weights().is_some());

        // Attention weights should sum to 1
        let weights = embedding.attention_weights().unwrap();
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_paths_aggregation() {
        let encoder = Code2VecEncoder::new(32);
        let embedding = encoder.aggregate_paths(&[]);

        assert_eq!(embedding.dim(), 32);
        // Should be zero vector
        for val in embedding.vector().as_slice() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_add_embeddings() {
        let mut encoder = Code2VecEncoder::new(4);

        encoder.add_token_embedding("test", vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(encoder.token_vocab_size(), 1);

        encoder.add_path_embedding("Func_Param", vec![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(encoder.path_vocab_size(), 1);
    }

    #[test]
    #[should_panic(expected = "Embedding dimension mismatch")]
    fn test_add_embedding_wrong_dim() {
        let mut encoder = Code2VecEncoder::new(4);
        encoder.add_token_embedding("test", vec![1.0, 2.0]); // Wrong dimension
    }

    #[test]
    fn test_default_encoder() {
        let encoder = Code2VecEncoder::default();
        assert_eq!(encoder.dim(), super::super::DEFAULT_EMBEDDING_DIM);
    }
}
