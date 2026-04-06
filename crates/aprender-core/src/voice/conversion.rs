//! Voice conversion module (GH-132).
//!
//! Provides voice conversion primitives for:
//! - Speaker identity conversion (change who is speaking)
//! - Non-parallel voice conversion (no paired training data needed)
//! - Any-to-any voice conversion
//!
//! # Architecture
//!
//! ```text
//! Source Audio → Content Encoder → Bottleneck → Target Speaker
//!                     ↓               ↓              ↓
//!               PPG/ASR Features  Speaker ID   → Decoder → Converted Audio
//! ```
//!
//! # Voice Conversion Methods
//!
//! - **`AutoVC`**: Autoencoder-based, disentangled content and speaker
//! - **StarGAN-VC**: GAN-based, non-parallel training
//! - **PPG-based**: Phonetic Posteriorgram bottleneck
//! - **VQVC+**: Vector-quantized with self-supervision
//!
//! # Example
//!
//! ```rust
//! use aprender::voice::conversion::{VoiceConversionConfig, ConversionMode};
//!
//! let config = VoiceConversionConfig::default();
//! assert_eq!(config.mode, ConversionMode::AnyToAny);
//! ```
//!
//! # References
//!
//! - Qian, K., et al. (2019). `AutoVC`: Zero-Shot Voice Style Transfer.
//! - Kameoka, H., et al. (2018). StarGAN-VC: Non-parallel VC with Star GAN.
//! - Sun, L., et al. (2016). Phonetic Posteriorgrams for VC.
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible

use super::{SpeakerEmbedding, VoiceError, VoiceResult};

// ============================================================================
// Configuration
// ============================================================================

/// Voice conversion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConversionMode {
    /// One-to-one: fixed source to fixed target
    OneToOne,
    /// Many-to-one: any source to fixed target
    ManyToOne,
    /// One-to-many: fixed source to any target
    OneToMany,
    /// Any-to-any: any source to any target (most flexible)
    #[default]
    AnyToAny,
}

/// Feature bottleneck type for voice conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BottleneckType {
    /// Phonetic Posteriorgram (PPG) from ASR
    Ppg,
    /// `AutoEncoder` bottleneck
    #[default]
    AutoEncoder,
    /// Vector Quantized (VQ) codebook
    VectorQuantized,
    /// Content Embedding (like HuBERT/wav2vec2)
    ContentEmbedding,
}

/// Configuration for voice conversion.
#[derive(Debug, Clone)]
pub struct VoiceConversionConfig {
    /// Conversion mode (one-to-one, many-to-one, etc.)
    pub mode: ConversionMode,
    /// Bottleneck feature type
    pub bottleneck: BottleneckType,
    /// Speaker embedding dimension
    pub speaker_dim: usize,
    /// Content feature dimension
    pub content_dim: usize,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Frame shift in milliseconds
    pub frame_shift_ms: u32,
    /// F0 (fundamental frequency) conversion enabled
    pub convert_f0: bool,
    /// Energy conversion enabled
    pub convert_energy: bool,
    /// Pitch shift ratio (1.0 = no shift)
    pub pitch_ratio: f32,
    /// Formant preservation factor [0.0, 1.0]
    pub formant_preservation: f32,
}

impl Default for VoiceConversionConfig {
    fn default() -> Self {
        Self {
            mode: ConversionMode::default(),
            bottleneck: BottleneckType::default(),
            speaker_dim: 256,
            content_dim: 512,
            sample_rate: 16000,
            frame_shift_ms: 10,
            convert_f0: true,
            convert_energy: true,
            pitch_ratio: 1.0,
            formant_preservation: 0.5,
        }
    }
}

impl VoiceConversionConfig {
    /// Create config for AutoVC-based conversion
    #[must_use]
    pub fn autovc() -> Self {
        Self {
            mode: ConversionMode::AnyToAny,
            bottleneck: BottleneckType::AutoEncoder,
            speaker_dim: 256,
            content_dim: 512,
            ..Self::default()
        }
    }

    /// Create config for StarGAN-VC style conversion
    #[must_use]
    pub fn stargan_vc() -> Self {
        Self {
            mode: ConversionMode::ManyToOne,
            bottleneck: BottleneckType::AutoEncoder,
            speaker_dim: 64,
            content_dim: 256,
            formant_preservation: 0.0,
            ..Self::default()
        }
    }

    /// Create config for PPG-based conversion
    #[must_use]
    pub fn ppg_based() -> Self {
        Self {
            mode: ConversionMode::AnyToAny,
            bottleneck: BottleneckType::Ppg,
            speaker_dim: 256,
            content_dim: 144, // Typical PPG dimension
            formant_preservation: 0.3,
            ..Self::default()
        }
    }

    /// Create config for real-time low-latency conversion
    #[must_use]
    pub fn realtime() -> Self {
        Self {
            mode: ConversionMode::ManyToOne,
            bottleneck: BottleneckType::AutoEncoder,
            frame_shift_ms: 5, // Lower latency
            formant_preservation: 0.7,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> VoiceResult<()> {
        if self.speaker_dim == 0 {
            return Err(VoiceError::InvalidConfig(
                "speaker_dim must be > 0".to_string(),
            ));
        }
        if self.content_dim == 0 {
            return Err(VoiceError::InvalidConfig(
                "content_dim must be > 0".to_string(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(VoiceError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if self.frame_shift_ms == 0 {
            return Err(VoiceError::InvalidConfig(
                "frame_shift_ms must be > 0".to_string(),
            ));
        }
        if !(0.0..=10.0).contains(&self.pitch_ratio) {
            return Err(VoiceError::InvalidConfig(
                "pitch_ratio must be in [0.0, 10.0]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.formant_preservation) {
            return Err(VoiceError::InvalidConfig(
                "formant_preservation must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }

    /// Get frame length in samples
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        (self.sample_rate as usize * self.frame_shift_ms as usize) / 1000
    }
}

// ============================================================================
// Conversion Result
// ============================================================================

/// Result of voice conversion operation.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Converted audio samples
    pub audio: Vec<f32>,
    /// Sample rate of output
    pub sample_rate: u32,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
    /// Duration in seconds
    pub duration_secs: f32,
    /// Source speaker similarity (how much source identity remains)
    pub source_similarity: f32,
    /// Target speaker similarity (how close to target)
    pub target_similarity: f32,
}

impl ConversionResult {
    /// Create a new conversion result
    #[must_use]
    pub fn new(audio: Vec<f32>, sample_rate: u32) -> Self {
        let duration_secs = if sample_rate > 0 {
            audio.len() as f32 / sample_rate as f32
        } else {
            0.0
        };

        Self {
            audio,
            sample_rate,
            confidence: 0.0,
            duration_secs,
            source_similarity: 0.0,
            target_similarity: 0.0,
        }
    }

    /// Set quality metrics
    #[must_use]
    pub fn with_metrics(mut self, confidence: f32, source_sim: f32, target_sim: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self.source_similarity = source_sim.clamp(0.0, 1.0);
        self.target_similarity = target_sim.clamp(0.0, 1.0);
        self
    }
}

// ============================================================================
// Voice Converter Trait
// ============================================================================

/// Trait for voice conversion systems.
pub trait VoiceConverter: Send + Sync {
    /// Get the configuration
    fn config(&self) -> &VoiceConversionConfig;

    /// Convert voice using speaker embeddings.
    ///
    /// # Arguments
    /// * `source_audio` - Source audio samples
    /// * `source_embedding` - Source speaker embedding (optional for some modes)
    /// * `target_embedding` - Target speaker embedding
    ///
    /// # Errors
    /// Returns error if conversion fails.
    fn convert(
        &self,
        source_audio: &[f32],
        source_embedding: Option<&SpeakerEmbedding>,
        target_embedding: &SpeakerEmbedding,
    ) -> VoiceResult<ConversionResult>;

    /// Extract content features from audio (speaker-independent).
    ///
    /// # Errors
    /// Returns error if extraction fails.
    fn extract_content(&self, audio: &[f32]) -> VoiceResult<Vec<Vec<f32>>>;

    /// Synthesize audio from content features and speaker embedding.
    ///
    /// # Errors
    /// Returns error if synthesis fails.
    fn synthesize(&self, content: &[Vec<f32>], speaker: &SpeakerEmbedding)
        -> VoiceResult<Vec<f32>>;
}

// ============================================================================
// AutoVC Converter
// ============================================================================

/// AutoVC-based voice converter.
///
/// Implements the `AutoVC` architecture for zero-shot voice conversion:
/// - Content encoder: Extracts speaker-independent features
/// - Speaker encoder: Extracts speaker embedding
/// - Decoder: Reconstructs mel spectrogram from content + speaker
///
/// Reference: Qian et al. (2019) "`AutoVC`: Zero-Shot Voice Style Transfer"
#[derive(Debug, Clone)]
pub struct AutoVcConverter {
    /// Configuration
    config: VoiceConversionConfig,
    /// Content encoder downsample factor
    downsample_factor: usize,
}

impl AutoVcConverter {
    /// Create a new `AutoVC` converter
    #[must_use]
    pub fn new(config: VoiceConversionConfig) -> Self {
        Self {
            config,
            downsample_factor: 32, // Typical AutoVC bottleneck
        }
    }

    /// Create with default `AutoVC` config
    #[must_use]
    pub fn default_autovc() -> Self {
        Self::new(VoiceConversionConfig::autovc())
    }

    /// Get downsample factor
    #[must_use]
    pub fn downsample_factor(&self) -> usize {
        self.downsample_factor
    }
}

include!("ppg_converter.rs");
include!("conversion_tests.rs");
