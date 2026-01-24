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
    pub fn euclidean_distance(&self, other: &Self) -> VoiceResult<f32> {
        if self.dim() != other.dim() {
            return Err(VoiceError::DimensionMismatch {
                expected: self.dim(),
                got: other.dim(),
            });
        }
        let sum_sq: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        Ok(sum_sq.sqrt())
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
#[must_use]
pub fn cosine_similarity(a: &SpeakerEmbedding, b: &SpeakerEmbedding) -> f32 {
    if a.dim() != b.dim() || a.dim() == 0 {
        return 0.0;
    }

    let dot: f32 = a
        .as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(x, y)| x * y)
        .sum();

    let norm_a = a.l2_norm();
    let norm_b = b.l2_norm();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
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
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 192);
        assert_eq!(config.sample_rate, 16000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_embedding_config_ecapa() {
        let config = EmbeddingConfig::ecapa_tdnn();
        assert_eq!(config.embedding_dim, 192);
        assert_eq!(config.n_mels, 80);
    }

    #[test]
    fn test_embedding_config_xvector() {
        let config = EmbeddingConfig::x_vector();
        assert_eq!(config.embedding_dim, 512);
        assert_eq!(config.n_mels, 30);
    }

    #[test]
    fn test_embedding_config_validation() {
        let mut config = EmbeddingConfig::default();
        config.embedding_dim = 0;
        assert!(config.validate().is_err());

        config.embedding_dim = 192;
        config.sample_rate = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_speaker_embedding_from_vec() {
        let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.dim(), 3);
        assert!(!emb.is_normalized());
    }

    #[test]
    fn test_speaker_embedding_zeros() {
        let emb = SpeakerEmbedding::zeros(192);
        assert_eq!(emb.dim(), 192);
        assert_eq!(emb.l2_norm(), 0.0);
    }

    #[test]
    fn test_speaker_embedding_normalize() {
        let mut emb = SpeakerEmbedding::from_vec(vec![3.0, 4.0]);
        emb.normalize();
        assert!(emb.is_normalized());
        assert!((emb.l2_norm() - 1.0).abs() < 1e-6);
        assert!((emb.as_slice()[0] - 0.6).abs() < 1e-6);
        assert!((emb.as_slice()[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_speaker_embedding_dot() {
        let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
        let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0]);
        assert!((emb1.dot(&emb2).unwrap() - 0.0).abs() < 1e-6);

        let emb3 = SpeakerEmbedding::from_vec(vec![1.0, 1.0]);
        assert!((emb1.dot(&emb3).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_speaker_embedding_dot_dimension_mismatch() {
        let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
        let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
        assert!(emb1.dot(&emb2).is_err());
    }

    #[test]
    fn test_speaker_embedding_euclidean_distance() {
        let emb1 = SpeakerEmbedding::from_vec(vec![0.0, 0.0]);
        let emb2 = SpeakerEmbedding::from_vec(vec![3.0, 4.0]);
        assert!((emb1.euclidean_distance(&emb2).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&emb, &emb);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
        let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
        let sim = cosine_similarity(&emb1, &emb2);
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
        let emb2 = SpeakerEmbedding::from_vec(vec![-1.0, 0.0]);
        let sim = cosine_similarity(&emb1, &emb2);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0]);
        let emb2 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
        let sim = cosine_similarity(&emb1, &emb2);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_normalize_embedding() {
        let emb = SpeakerEmbedding::from_vec(vec![3.0, 4.0]);
        let normalized = normalize_embedding(&emb);
        assert!(normalized.is_normalized());
        assert!((normalized.l2_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_embeddings() {
        let embeddings = vec![
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0]),
        ];
        let avg = average_embeddings(&embeddings).unwrap();
        assert!((avg.as_slice()[0] - 0.5).abs() < 1e-6);
        assert!((avg.as_slice()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_average_embeddings_empty() {
        let embeddings: Vec<SpeakerEmbedding> = vec![];
        assert!(average_embeddings(&embeddings).is_err());
    }

    #[test]
    fn test_average_embeddings_dimension_mismatch() {
        let embeddings = vec![
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]),
        ];
        assert!(average_embeddings(&embeddings).is_err());
    }

    #[test]
    fn test_similarity_matrix() {
        let embeddings = vec![
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0]),
            SpeakerEmbedding::from_vec(vec![1.0, 1.0]),
        ];
        let matrix = similarity_matrix(&embeddings);

        assert_eq!(matrix.len(), 3);
        // Diagonal should be 1.0
        assert!((matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((matrix[1][1] - 1.0).abs() < 1e-6);
        // [0][1] and [1][0] should be 0.0 (orthogonal)
        assert!((matrix[0][1] - 0.0).abs() < 1e-6);
        assert!((matrix[1][0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ecapa_tdnn_stub() {
        let extractor = EcapaTdnn::default_config();
        assert_eq!(extractor.embedding_dim(), 192);
        assert_eq!(extractor.sample_rate(), 16000);

        let audio = vec![0.0_f32; 16000];
        let result = extractor.extract(&audio);
        assert!(result.is_err());
    }

    #[test]
    fn test_ecapa_tdnn_empty_audio() {
        let extractor = EcapaTdnn::default_config();
        let result = extractor.extract(&[]);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_xvector_stub() {
        let extractor = XVector::default_config();
        assert_eq!(extractor.embedding_dim(), 512);
        assert_eq!(extractor.sample_rate(), 16000);
    }
}
