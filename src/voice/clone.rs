//! Voice cloning module (GH-132).
//!
//! Provides voice cloning primitives for:
//! - Few-shot voice cloning (3-10 seconds of reference audio)
//! - Zero-shot voice cloning (single utterance)
//! - Speaker adaptation (fine-tuning on target voice)
//!
//! # Architecture
//!
//! ```text
//! Reference Audio → Speaker Encoder → Speaker Embedding
//!                                           ↓
//! Text Input → Text Encoder → Linguistic Features → Synthesizer → Mel Spectrogram → Vocoder → Audio
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::voice::clone::{CloningConfig, VoiceProfile};
//!
//! let config = CloningConfig::default();
//! let profile = VoiceProfile::new("speaker_1".to_string());
//! assert_eq!(profile.speaker_id(), "speaker_1");
//! ```
//!
//! # References
//!
//! - Jia, Y., et al. (2018). Transfer Learning from Speaker Verification to TTS.
//! - Casanova, E., et al. (2022). `YourTTS`: Zero-Shot TTS and Voice Conversion.
//! - Arik, S., et al. (2018). Neural Voice Cloning with Few Samples.
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible

use super::{SpeakerEmbedding, StyleVector, VoiceError, VoiceResult};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for voice cloning.
#[derive(Debug, Clone)]
pub struct CloningConfig {
    /// Minimum reference audio duration in seconds
    pub min_reference_duration: f32,
    /// Maximum reference audio duration in seconds
    pub max_reference_duration: f32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Speaker embedding dimension
    pub embedding_dim: usize,
    /// Number of mel channels for synthesis
    pub n_mels: usize,
    /// Enable speaker adaptation (fine-tuning)
    pub enable_adaptation: bool,
    /// Similarity threshold for speaker verification
    pub similarity_threshold: f32,
}

impl Default for CloningConfig {
    fn default() -> Self {
        Self {
            min_reference_duration: 3.0,
            max_reference_duration: 30.0,
            sample_rate: 22050,
            embedding_dim: 256,
            n_mels: 80,
            enable_adaptation: false,
            similarity_threshold: 0.75,
        }
    }
}

impl CloningConfig {
    /// Create config for few-shot cloning (3-10 seconds)
    #[must_use]
    pub fn few_shot() -> Self {
        Self {
            min_reference_duration: 3.0,
            max_reference_duration: 10.0,
            enable_adaptation: false,
            ..Self::default()
        }
    }

    /// Create config for zero-shot cloning (single utterance)
    #[must_use]
    pub fn zero_shot() -> Self {
        Self {
            min_reference_duration: 1.0,
            max_reference_duration: 5.0,
            enable_adaptation: false,
            ..Self::default()
        }
    }

    /// Create config with speaker adaptation
    #[must_use]
    pub fn with_adaptation() -> Self {
        Self {
            enable_adaptation: true,
            min_reference_duration: 10.0,
            max_reference_duration: 60.0,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> VoiceResult<()> {
        if self.min_reference_duration <= 0.0 {
            return Err(VoiceError::InvalidConfig(
                "min_reference_duration must be > 0".to_string(),
            ));
        }
        if self.max_reference_duration <= self.min_reference_duration {
            return Err(VoiceError::InvalidConfig(
                "max_reference_duration must be > min_reference_duration".to_string(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(VoiceError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if self.embedding_dim == 0 {
            return Err(VoiceError::InvalidConfig(
                "embedding_dim must be > 0".to_string(),
            ));
        }
        if self.n_mels == 0 {
            return Err(VoiceError::InvalidConfig("n_mels must be > 0".to_string()));
        }
        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            return Err(VoiceError::InvalidConfig(
                "similarity_threshold must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }

    /// Get minimum samples for reference audio
    #[must_use]
    pub fn min_reference_samples(&self) -> usize {
        (self.min_reference_duration * self.sample_rate as f32) as usize
    }

    /// Get maximum samples for reference audio
    #[must_use]
    pub fn max_reference_samples(&self) -> usize {
        (self.max_reference_duration * self.sample_rate as f32) as usize
    }
}

// ============================================================================
// Voice Profile
// ============================================================================

/// A voice profile representing a cloned speaker.
///
/// Contains the speaker embedding and optionally style information
/// extracted from reference audio.
#[derive(Debug, Clone)]
pub struct VoiceProfile {
    /// Unique identifier for this voice
    speaker_id: String,
    /// Speaker embedding vector
    embedding: Option<SpeakerEmbedding>,
    /// Style characteristics (optional)
    style: Option<StyleVector>,
    /// Reference audio duration in seconds
    reference_duration: f32,
    /// Quality score [0.0, 1.0] based on reference audio
    quality_score: f32,
    /// Whether the profile has been adapted/fine-tuned
    adapted: bool,
}

impl VoiceProfile {
    /// Create a new empty voice profile
    #[must_use]
    pub fn new(speaker_id: String) -> Self {
        Self {
            speaker_id,
            embedding: None,
            style: None,
            reference_duration: 0.0,
            quality_score: 0.0,
            adapted: false,
        }
    }

    /// Create profile with embedding
    #[must_use]
    pub fn with_embedding(speaker_id: String, embedding: SpeakerEmbedding) -> Self {
        Self {
            speaker_id,
            embedding: Some(embedding),
            style: None,
            reference_duration: 0.0,
            quality_score: 0.5, // Default quality for embedding-only profile
            adapted: false,
        }
    }

    /// Get speaker ID
    #[must_use]
    pub fn speaker_id(&self) -> &str {
        &self.speaker_id
    }

    /// Get speaker embedding
    #[must_use]
    pub fn embedding(&self) -> Option<&SpeakerEmbedding> {
        self.embedding.as_ref()
    }

    /// Get style vector
    #[must_use]
    pub fn style(&self) -> Option<&StyleVector> {
        self.style.as_ref()
    }

    /// Set speaker embedding
    pub fn set_embedding(&mut self, embedding: SpeakerEmbedding) {
        self.embedding = Some(embedding);
    }

    /// Set style vector
    pub fn set_style(&mut self, style: StyleVector) {
        self.style = Some(style);
    }

    /// Get reference duration
    #[must_use]
    pub fn reference_duration(&self) -> f32 {
        self.reference_duration
    }

    /// Set reference duration
    pub fn set_reference_duration(&mut self, duration: f32) {
        self.reference_duration = duration;
    }

    /// Get quality score
    #[must_use]
    pub fn quality_score(&self) -> f32 {
        self.quality_score
    }

    /// Set quality score
    pub fn set_quality_score(&mut self, score: f32) {
        self.quality_score = score.clamp(0.0, 1.0);
    }

    /// Check if profile has been adapted
    #[must_use]
    pub fn is_adapted(&self) -> bool {
        self.adapted
    }

    /// Mark profile as adapted
    pub fn set_adapted(&mut self, adapted: bool) {
        self.adapted = adapted;
    }

    /// Check if profile is ready for synthesis
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.embedding.is_some()
    }

    /// Compute similarity to another profile
    ///
    /// # Errors
    /// Returns error if either profile lacks an embedding.
    pub fn similarity(&self, other: &Self) -> VoiceResult<f32> {
        let emb_a = self
            .embedding
            .as_ref()
            .ok_or_else(|| VoiceError::InvalidConfig("missing embedding in source".to_string()))?;
        let emb_b = other
            .embedding
            .as_ref()
            .ok_or_else(|| VoiceError::InvalidConfig("missing embedding in target".to_string()))?;

        if emb_a.dim() != emb_b.dim() {
            return Err(VoiceError::DimensionMismatch {
                expected: emb_a.dim(),
                got: emb_b.dim(),
            });
        }

        // Cosine similarity
        let dot: f32 = emb_a
            .as_slice()
            .iter()
            .zip(emb_b.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a = emb_a.l2_norm();
        let norm_b = emb_b.l2_norm();

        if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
            return Ok(0.0);
        }

        Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
    }
}

// ============================================================================
// Voice Cloner Trait
// ============================================================================

/// Trait for voice cloning operations.
pub trait VoiceCloner {
    /// Create a voice profile from reference audio.
    ///
    /// # Arguments
    /// * `reference_audio` - Reference audio samples (mono, at config sample rate)
    /// * `speaker_id` - Identifier for the cloned voice
    ///
    /// # Errors
    /// Returns error if cloning fails.
    fn create_profile(
        &self,
        reference_audio: &[f32],
        speaker_id: &str,
    ) -> VoiceResult<VoiceProfile>;

    /// Synthesize audio from text using a voice profile.
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `profile` - Voice profile to use
    ///
    /// # Returns
    /// Synthesized audio samples.
    ///
    /// # Errors
    /// Returns error if synthesis fails.
    fn synthesize(&self, text: &str, profile: &VoiceProfile) -> VoiceResult<Vec<f32>>;

    /// Adapt a voice profile with additional reference audio.
    ///
    /// # Arguments
    /// * `profile` - Profile to adapt
    /// * `additional_audio` - Additional reference audio
    ///
    /// # Errors
    /// Returns error if adaptation fails.
    fn adapt(&self, profile: &mut VoiceProfile, additional_audio: &[f32]) -> VoiceResult<()>;

    /// Get the configuration
    fn config(&self) -> &CloningConfig;
}

// ============================================================================
// Speaker Encoder Trait
// ============================================================================

/// Trait for speaker encoding in voice cloning.
pub trait SpeakerEncoder {
    /// Encode speaker from audio to embedding.
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono)
    ///
    /// # Errors
    /// Returns error if encoding fails.
    fn encode(&self, audio: &[f32]) -> VoiceResult<SpeakerEmbedding>;

    /// Get embedding dimension
    fn embedding_dim(&self) -> usize;
}

// ============================================================================
// Stub Implementations
// ============================================================================

/// YourTTS-based voice cloning.
///
/// Reference: Casanova et al., 2022 - `YourTTS`: Zero-Shot TTS and Voice Conversion.
///
/// # Note
/// This is a stub - requires model weights.
#[derive(Debug)]
pub struct YourTtsCloner {
    config: CloningConfig,
}

impl YourTtsCloner {
    /// Create new `YourTTS` cloner
    #[must_use]
    pub fn new(config: CloningConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(CloningConfig::default())
    }
}

include!("sv2tts_speaker_encoder.rs");
include!("clone_tests.rs");
