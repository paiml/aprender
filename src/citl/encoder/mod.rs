//! Error encoder for program-feedback graph construction.
//!
//! Per Yasunaga & Liang (2020), the program-feedback graph connects:
//! - Symbols in source code (variables, types, functions)
//! - Diagnostic feedback (error codes, messages, spans)
//! - AST structure (parent-child, sibling relationships)
//!
//! # GNN-Based Encoding
//!
//! This module provides two encoder types:
//! - [`ErrorEncoder`]: Simple bag-of-features encoding (fast, CPU-only)
//! - [`GNNErrorEncoder`]: Graph neural network encoding (higher quality)
//!
//! The GNN encoder builds a program-feedback graph and uses message passing
//! to produce context-aware embeddings that capture code structure.

use super::diagnostic::{CompilerDiagnostic, SourceSpan};
use super::ErrorCode;
use crate::autograd::Tensor;
use crate::nn::gnn::{AdjacencyMatrix, GCNConv, SAGEAggregation, SAGEConv};
use std::collections::HashMap;
use trueno::Vector;

/// Error embedding vector.
///
/// A fixed-size vector representation of an error pattern
/// suitable for similarity search and ML training.
#[derive(Debug, Clone)]
pub struct ErrorEmbedding {
    /// The embedding vector (256 dimensions by default)
    pub vector: Vec<f32>,
    /// Error code for reference
    pub error_code: ErrorCode,
    /// Hash of surrounding context
    pub context_hash: u64,
}

impl ErrorEmbedding {
    /// Create a new error embedding.
    #[must_use]
    pub fn new(vector: Vec<f32>, error_code: ErrorCode, context_hash: u64) -> Self {
        Self {
            vector,
            error_code,
            context_hash,
        }
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// ONE PATH: Delegates to `nn::functional::cosine_similarity_slice` (UCBD §4).
    #[must_use]
    pub fn cosine_similarity(&self, other: &ErrorEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() || self.vector.is_empty() {
            return 0.0;
        }
        crate::nn::functional::cosine_similarity_slice(&self.vector, &other.vector)
    }

    /// Compute L2 distance to another embedding using trueno SIMD.
    #[must_use]
    pub fn l2_distance(&self, other: &ErrorEmbedding) -> f32 {
        if self.vector.len() != other.vector.len() || self.vector.is_empty() {
            return f32::MAX;
        }

        let va = Vector::from_slice(&self.vector);
        let vb = Vector::from_slice(&other.vector);

        va.sub(&vb)
            .and_then(|diff| diff.norm_l2())
            .unwrap_or(f32::MAX)
    }
}

/// Error encoder using simplified feature extraction.
///
/// In production, this would use a GNN per Yasunaga & Liang (2020).
/// For now, we use bag-of-features encoding suitable for pattern matching.
#[derive(Debug)]
pub struct ErrorEncoder {
    /// Embedding dimension
    dim: usize,
    /// Error code embeddings (learned or hashed)
    error_code_embeddings: HashMap<String, Vec<f32>>,
    /// Vocabulary for source tokens (reserved for future token-level encoding)
    #[allow(dead_code)]
    vocab: HashMap<String, usize>,
}

include!("gnn_error_encoder.rs");
include!("mod_part_03.rs");
include!("program_feedback_graph.rs");
