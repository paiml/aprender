//! Voice processing module (GH-132).
//!
//! Provides voice-related primitives:
//! - Speaker embeddings (x-vector, ECAPA-TDNN)
//! - Voice style transfer
//! - Voice cloning
//! - Voice conversion
//!
//! # Architecture
//!
//! ```text
//! Audio → Feature Extraction → Embedding Model → Speaker Vector
//!                                    ↓
//!                            ECAPA-TDNN / X-Vector
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::voice::{EmbeddingConfig, SpeakerEmbedding};
//!
//! let config = EmbeddingConfig::default();
//! let embedding = SpeakerEmbedding::zeros(config.embedding_dim);
//! assert_eq!(embedding.dim(), 192);
//! ```
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>`
//! - Real-time safe where noted

pub mod embedding;

// Re-exports
pub use embedding::{
    cosine_similarity, normalize_embedding, EmbeddingConfig, EmbeddingExtractor, SpeakerEmbedding,
};

/// Voice processing error type
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Invalid audio input
    InvalidAudio(String),
    /// Embedding extraction failed
    ExtractionFailed(String),
    /// Model not loaded
    ModelNotLoaded,
    /// Feature not implemented
    NotImplemented(String),
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for VoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            Self::InvalidAudio(msg) => write!(f, "Invalid audio: {msg}"),
            Self::ExtractionFailed(msg) => write!(f, "Embedding extraction failed: {msg}"),
            Self::ModelNotLoaded => write!(f, "Embedding model not loaded"),
            Self::NotImplemented(msg) => write!(f, "Not implemented: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for VoiceError {}

/// Result type for voice operations
pub type VoiceResult<T> = Result<T, VoiceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_error_display() {
        let err = VoiceError::InvalidConfig("bad config".to_string());
        assert_eq!(err.to_string(), "Invalid configuration: bad config");

        let err = VoiceError::DimensionMismatch {
            expected: 192,
            got: 256,
        };
        assert_eq!(err.to_string(), "Dimension mismatch: expected 192, got 256");
    }

    #[test]
    fn test_voice_error_equality() {
        let err1 = VoiceError::ModelNotLoaded;
        let err2 = VoiceError::ModelNotLoaded;
        assert_eq!(err1, err2);
    }
}
