//! Configuration for the pattern store.

use serde::{Deserialize, Serialize};

/// Default embedding dimension for the pattern store
const DEFAULT_EMBEDDING_DIM: usize = 384;

/// Configuration for the pattern store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStoreConfig {
    /// Chunk size for the chunker (default: 256)
    pub chunk_size: usize,
    /// Embedding dimension (default: 384)
    pub embedding_dim: usize,
    /// RRF k constant (default: 60.0)
    pub rrf_k: f32,
}

impl Default for PatternStoreConfig {
    fn default() -> Self {
        Self { chunk_size: 256, embedding_dim: DEFAULT_EMBEDDING_DIM, rrf_k: 60.0 }
    }
}
