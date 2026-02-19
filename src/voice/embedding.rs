//! Speaker embedding extraction.
//!
//! Provides speaker embedding models and utilities:
//! - X-vector embeddings (Snyder et al., 2018)
//! - ECAPA-TDNN embeddings (Desplanques et al., 2020)
//! - Embedding comparison and clustering support
//!
//! # Architecture
//!
//! ```text
//! Audio Frames → TDNN → Statistics Pooling → FC → Embedding
//!                         ↓
//!                  Mean + Std aggregation
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::voice::embedding::{EmbeddingConfig, SpeakerEmbedding, cosine_similarity};
//!
//! let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
//! let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
//! let sim = cosine_similarity(&emb1, &emb2);
//! assert!((sim - 0.0).abs() < 1e-6); // Orthogonal vectors
//! ```
//!
//! # References
//!
//! - Snyder, D., et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition.
//! - Desplanques, B., et al. (2020). ECAPA-TDNN: Emphasized Channel Attention.
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible

use super::{VoiceError, VoiceResult};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for speaker embedding extraction
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Output embedding dimension (typically 192 or 512)
    pub embedding_dim: usize,
    /// Input sample rate (Hz)
    pub sample_rate: u32,
    /// Frame length in milliseconds
    pub frame_length_ms: u32,
    /// Frame shift in milliseconds
    pub frame_shift_ms: u32,
    /// Number of mel filterbank channels
    pub n_mels: usize,
    /// Normalize embeddings to unit length
    pub normalize: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 192,
            sample_rate: 16000,
            frame_length_ms: 25,
            frame_shift_ms: 10,
            n_mels: 80,
            normalize: true,
        }
    }
}

impl EmbeddingConfig {
    /// Configuration for ECAPA-TDNN model
    #[must_use]
    pub fn ecapa_tdnn() -> Self {
        Self {
            embedding_dim: 192,
            n_mels: 80,
            ..Self::default()
        }
    }

    /// Configuration for X-vector model
    #[must_use]
    pub fn x_vector() -> Self {
        Self {
            embedding_dim: 512,
            n_mels: 30,
            ..Self::default()
        }
    }

    /// Configuration for ResNet-based embeddings
    #[must_use]
    pub fn resnet() -> Self {
        Self {
            embedding_dim: 256,
            n_mels: 64,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> VoiceResult<()> {
        if self.embedding_dim == 0 {
            return Err(VoiceError::InvalidConfig(
                "embedding_dim must be > 0".to_string(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(VoiceError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if self.frame_length_ms == 0 {
            return Err(VoiceError::InvalidConfig(
                "frame_length_ms must be > 0".to_string(),
            ));
        }
        if self.n_mels == 0 {
            return Err(VoiceError::InvalidConfig("n_mels must be > 0".to_string()));
        }
        Ok(())
    }
}

// ============================================================================
// Speaker Embedding
// ============================================================================

/// A speaker embedding vector.
///
/// Represents a fixed-dimensional representation of a speaker's voice
/// characteristics, suitable for comparison and clustering.
#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    /// The embedding vector
    vector: Vec<f32>,
    /// Whether the embedding is normalized
    normalized: bool,
}

impl SpeakerEmbedding {
    /// Create embedding from vector
    #[must_use]
    pub fn from_vec(vector: Vec<f32>) -> Self {
        Self {
            vector,
            normalized: false,
        }
    }

    /// Create zero embedding of given dimension
    #[must_use]
    pub fn zeros(dim: usize) -> Self {
        Self {
            vector: vec![0.0; dim],
            normalized: false,
        }
    }

    /// Get embedding dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Get embedding vector as slice
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.vector
    }

    /// Get mutable embedding vector
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.vector
    }

    /// Convert to owned vector
    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        self.vector
    }

    /// Check if embedding is normalized
    #[must_use]
    pub fn is_normalized(&self) -> bool {
        self.normalized
    }

    /// Normalize embedding to unit length (in-place)
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > f32::EPSILON {
            for x in &mut self.vector {
                *x /= norm;
            }
        }
        self.normalized = true;
    }

    /// Get L2 norm of embedding
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        self.vector.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Compute dot product with another embedding
    ///
    /// # Errors
    /// Returns error if dimensions don't match.
    pub fn dot(&self, other: &Self) -> VoiceResult<f32> {
        if self.dim() != other.dim() {
            return Err(VoiceError::DimensionMismatch {
                expected: self.dim(),
                got: other.dim(),
            });
        }
        Ok(self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum())
    }

    /// Compute Euclidean distance to another embedding
    ///
    /// # Errors
    /// Returns error if dimensions don't match.
    /// ONE PATH: Core computation delegates to `nn::functional::euclidean_distance` (UCBD §4).
    pub fn euclidean_distance(&self, other: &Self) -> VoiceResult<f32> {
        if self.dim() != other.dim() {
            return Err(VoiceError::DimensionMismatch {
                expected: self.dim(),
                got: other.dim(),
            });
        }
        Ok(crate::nn::functional::euclidean_distance(
            &self.vector,
            &other.vector,
        ))
    }
}

// ============================================================================
// Embedding Extractor Trait
// ============================================================================

/// Trait for speaker embedding extraction models.
///
/// Implementations include:
/// - ECAPA-TDNN (best quality)
/// - X-Vector (classic)
/// - ResNet-based (fast)
pub trait EmbeddingExtractor {
    /// Extract speaker embedding from audio samples.
    ///
    /// # Arguments
    /// * `audio` - Audio samples (f32, mono, at configured sample rate)
    ///
    /// # Returns
    /// Speaker embedding vector
    fn extract(&self, audio: &[f32]) -> VoiceResult<SpeakerEmbedding>;

    /// Get the embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get the expected sample rate
    fn sample_rate(&self) -> u32;
}

// ============================================================================
// Stub Implementations
// ============================================================================

/// ECAPA-TDNN speaker embedding extractor.
///
/// Emphasized Channel Attention, Propagation and Aggregation in TDNN.
/// State-of-the-art for speaker verification (2020+).
///
/// # Note
/// This is a stub - actual implementation requires model weights.
#[derive(Debug)]
pub struct EcapaTdnn {
    config: EmbeddingConfig,
}

impl EcapaTdnn {
    /// Create new ECAPA-TDNN extractor
    ///
    /// # Note
    /// This is a stub - returns `NotImplemented` on `extract()`.
    #[must_use]
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }

    /// Create with default ECAPA configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(EmbeddingConfig::ecapa_tdnn())
    }
}

impl EmbeddingExtractor for EcapaTdnn {
    fn extract(&self, audio: &[f32]) -> VoiceResult<SpeakerEmbedding> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty audio".to_string()));
        }
        // Stub: Real implementation requires ECAPA-TDNN model
        Err(VoiceError::NotImplemented(
            "ECAPA-TDNN requires model weights (use from_apr to load)".to_string(),
        ))
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

/// X-Vector speaker embedding extractor.
///
/// Classic TDNN-based speaker embedding (Snyder et al., 2018).
/// Good balance of quality and speed.
///
/// # Note
/// This is a stub - actual implementation requires model weights.
#[derive(Debug)]
pub struct XVector {
    config: EmbeddingConfig,
}

impl XVector {
    /// Create new X-Vector extractor
    #[must_use]
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }

    /// Create with default X-Vector configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(EmbeddingConfig::x_vector())
    }
}

impl EmbeddingExtractor for XVector {
    fn extract(&self, audio: &[f32]) -> VoiceResult<SpeakerEmbedding> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty audio".to_string()));
        }
        // Stub: Real implementation requires X-Vector model
        Err(VoiceError::NotImplemented(
            "X-Vector requires model weights (use from_apr to load)".to_string(),
        ))
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute cosine similarity between two embeddings.
///
/// Returns value in [-1.0, 1.0] where:
/// - 1.0 = identical direction
/// - 0.0 = orthogonal
/// - -1.0 = opposite direction
///
/// For speaker verification, same speaker pairs typically have
/// cosine similarity > 0.7.
///
/// ONE PATH: Delegates to `nn::functional::cosine_similarity_slice` (UCBD §4).
#[must_use]
pub fn cosine_similarity(a: &SpeakerEmbedding, b: &SpeakerEmbedding) -> f32 {
    if a.dim() != b.dim() || a.dim() == 0 {
        return 0.0;
    }
    crate::nn::functional::cosine_similarity_slice(a.as_slice(), b.as_slice())
}

/// Normalize embedding to unit length.
///
/// Returns a new normalized embedding without modifying the original.
#[must_use]
pub fn normalize_embedding(embedding: &SpeakerEmbedding) -> SpeakerEmbedding {
    let mut normalized = embedding.clone();
    normalized.normalize();
    normalized
}

/// Compute average embedding from multiple embeddings.
///
/// Useful for creating speaker templates from multiple utterances.
///
/// # Errors
/// Returns error if embeddings have different dimensions or list is empty.
pub fn average_embeddings(embeddings: &[SpeakerEmbedding]) -> VoiceResult<SpeakerEmbedding> {
    if embeddings.is_empty() {
        return Err(VoiceError::InvalidConfig(
            "cannot average empty list".to_string(),
        ));
    }

    let dim = embeddings[0].dim();
    for emb in embeddings.iter().skip(1) {
        if emb.dim() != dim {
            return Err(VoiceError::DimensionMismatch {
                expected: dim,
                got: emb.dim(),
            });
        }
    }

    let mut avg = vec![0.0_f32; dim];
    let count = embeddings.len() as f32;

    for emb in embeddings {
        for (i, &val) in emb.as_slice().iter().enumerate() {
            avg[i] += val / count;
        }
    }

    Ok(SpeakerEmbedding::from_vec(avg))
}

/// Compute pairwise similarity matrix for embeddings.
///
/// Returns `NxN` matrix where entry `[i][j]` is cosine similarity
/// between embedding `i` and embedding `j`.
#[must_use]
pub fn similarity_matrix(embeddings: &[SpeakerEmbedding]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut matrix = vec![vec![0.0_f32; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0; // Self-similarity
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim; // Symmetric
        }
    }

    matrix
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "embedding_tests.rs"]
mod tests;
