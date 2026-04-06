//! Voice style transfer module (GH-132).
//!
//! Provides voice style transfer primitives for:
//! - Prosody transfer (pitch, rhythm, energy)
//! - Timbre conversion (spectral characteristics)
//! - Cross-lingual style transfer
//!
//! # Architecture
//!
//! ```text
//! Source Audio → Content Encoder → Linguistic Features
//!                                         ↓
//! Reference Audio → Style Encoder → Style Vector → Decoder → Styled Audio
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::voice::style::{StyleConfig, StyleVector, prosody_distance};
//!
//! let style_a = StyleVector::new(vec![0.5, 0.3, 0.2], vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]);
//! let style_b = StyleVector::new(vec![0.6, 0.4, 0.3], vec![0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7]);
//! let distance = prosody_distance(&style_a, &style_b);
//! assert!(distance >= 0.0);
//! ```
//!
//! # References
//!
//! - Qian, K., et al. (2019). `AutoVC`: Zero-Shot Voice Style Transfer.
//! - Wang, Y., et al. (2018). Style Tokens for Expressive Speech Synthesis.
//! - Chen, M., et al. (2021). Adaspeech: Adaptive Text to Speech for Custom Voice.
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible

use super::{SpeakerEmbedding, VoiceError, VoiceResult};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for voice style transfer.
#[derive(Debug, Clone)]
pub struct StyleConfig {
    /// Dimension of prosody features (pitch, energy, duration)
    pub prosody_dim: usize,
    /// Dimension of timbre features (spectral envelope)
    pub timbre_dim: usize,
    /// Dimension of speaking rate features
    pub rhythm_dim: usize,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Frame shift in milliseconds for analysis
    pub frame_shift_ms: u32,
    /// Blend factor for style interpolation [0.0, 1.0]
    pub style_strength: f32,
    /// Preserve source pitch contour
    pub preserve_pitch_contour: bool,
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            prosody_dim: 64,
            timbre_dim: 128,
            rhythm_dim: 32,
            sample_rate: 16000,
            frame_shift_ms: 10,
            style_strength: 1.0,
            preserve_pitch_contour: false,
        }
    }
}

impl StyleConfig {
    /// Create config for prosody-only transfer (pitch, energy, rhythm)
    #[must_use]
    pub fn prosody_only() -> Self {
        Self {
            style_strength: 0.5,
            preserve_pitch_contour: true,
            ..Self::default()
        }
    }

    /// Create config for full voice conversion
    #[must_use]
    pub fn full_conversion() -> Self {
        Self {
            style_strength: 1.0,
            preserve_pitch_contour: false,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> VoiceResult<()> {
        if self.prosody_dim == 0 {
            return Err(VoiceError::InvalidConfig(
                "prosody_dim must be > 0".to_string(),
            ));
        }
        if self.timbre_dim == 0 {
            return Err(VoiceError::InvalidConfig(
                "timbre_dim must be > 0".to_string(),
            ));
        }
        if self.rhythm_dim == 0 {
            return Err(VoiceError::InvalidConfig(
                "rhythm_dim must be > 0".to_string(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(VoiceError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.style_strength) {
            return Err(VoiceError::InvalidConfig(
                "style_strength must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }

    /// Total style vector dimension
    #[must_use]
    pub fn total_dim(&self) -> usize {
        self.prosody_dim + self.timbre_dim + self.rhythm_dim
    }
}

// ============================================================================
// Style Vector
// ============================================================================

/// Voice style vector capturing prosody, timbre, and rhythm.
///
/// Represents the speaking style characteristics of a speaker,
/// independent of linguistic content.
#[derive(Debug, Clone)]
pub struct StyleVector {
    /// Prosody features (pitch contour, energy pattern)
    prosody: Vec<f32>,
    /// Timbre features (spectral envelope, formants)
    timbre: Vec<f32>,
    /// Rhythm features (speaking rate, pausing patterns)
    rhythm: Vec<f32>,
}

impl StyleVector {
    /// Create a new style vector from components
    #[must_use]
    pub fn new(prosody: Vec<f32>, timbre: Vec<f32>, rhythm: Vec<f32>) -> Self {
        Self {
            prosody,
            timbre,
            rhythm,
        }
    }

    /// Create zero-initialized style vector
    #[must_use]
    pub fn zeros(config: &StyleConfig) -> Self {
        Self {
            prosody: vec![0.0; config.prosody_dim],
            timbre: vec![0.0; config.timbre_dim],
            rhythm: vec![0.0; config.rhythm_dim],
        }
    }

    /// Create from a flattened slice
    ///
    /// # Errors
    /// Returns error if slice length doesn't match config dimensions.
    pub fn from_flat(vector: &[f32], config: &StyleConfig) -> VoiceResult<Self> {
        let expected_len = config.total_dim();
        if vector.len() != expected_len {
            return Err(VoiceError::DimensionMismatch {
                expected: expected_len,
                got: vector.len(),
            });
        }

        let prosody_end = config.prosody_dim;
        let timbre_end = prosody_end + config.timbre_dim;

        Ok(Self {
            prosody: vector[..prosody_end].to_vec(),
            timbre: vector[prosody_end..timbre_end].to_vec(),
            rhythm: vector[timbre_end..].to_vec(),
        })
    }

    /// Get prosody features
    #[must_use]
    pub fn prosody(&self) -> &[f32] {
        &self.prosody
    }

    /// Get timbre features
    #[must_use]
    pub fn timbre(&self) -> &[f32] {
        &self.timbre
    }

    /// Get rhythm features
    #[must_use]
    pub fn rhythm(&self) -> &[f32] {
        &self.rhythm
    }

    /// Get total dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.prosody.len() + self.timbre.len() + self.rhythm.len()
    }

    /// Flatten to single vector
    #[must_use]
    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.dim());
        flat.extend_from_slice(&self.prosody);
        flat.extend_from_slice(&self.timbre);
        flat.extend_from_slice(&self.rhythm);
        flat
    }

    /// Interpolate between two styles
    ///
    /// # Arguments
    /// * `other` - Target style
    /// * `t` - Interpolation factor [0.0, 1.0] where 0.0 = self, 1.0 = other
    ///
    /// # Errors
    /// Returns error if styles have different dimensions.
    pub fn interpolate(&self, other: &Self, t: f32) -> VoiceResult<Self> {
        if self.prosody.len() != other.prosody.len() {
            return Err(VoiceError::DimensionMismatch {
                expected: self.prosody.len(),
                got: other.prosody.len(),
            });
        }
        if self.timbre.len() != other.timbre.len() {
            return Err(VoiceError::DimensionMismatch {
                expected: self.timbre.len(),
                got: other.timbre.len(),
            });
        }
        if self.rhythm.len() != other.rhythm.len() {
            return Err(VoiceError::DimensionMismatch {
                expected: self.rhythm.len(),
                got: other.rhythm.len(),
            });
        }

        let t = t.clamp(0.0, 1.0);
        let one_minus_t = 1.0 - t;

        let prosody = self
            .prosody
            .iter()
            .zip(other.prosody.iter())
            .map(|(a, b)| a * one_minus_t + b * t)
            .collect();

        let timbre = self
            .timbre
            .iter()
            .zip(other.timbre.iter())
            .map(|(a, b)| a * one_minus_t + b * t)
            .collect();

        let rhythm = self
            .rhythm
            .iter()
            .zip(other.rhythm.iter())
            .map(|(a, b)| a * one_minus_t + b * t)
            .collect();

        Ok(Self {
            prosody,
            timbre,
            rhythm,
        })
    }

    /// Compute L2 norm
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        let sum_sq: f32 = self
            .prosody
            .iter()
            .chain(self.timbre.iter())
            .chain(self.rhythm.iter())
            .map(|x| x * x)
            .sum();
        sum_sq.sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > f32::EPSILON {
            for x in &mut self.prosody {
                *x /= norm;
            }
            for x in &mut self.timbre {
                *x /= norm;
            }
            for x in &mut self.rhythm {
                *x /= norm;
            }
        }
    }
}

// ============================================================================
// Style Encoder Trait
// ============================================================================

/// Trait for style encoding from audio.
pub trait StyleEncoder {
    /// Extract style vector from audio.
    ///
    /// # Arguments
    /// * `audio` - Audio samples (f32, mono)
    ///
    /// # Errors
    /// Returns error if extraction fails.
    fn encode(&self, audio: &[f32]) -> VoiceResult<StyleVector>;

    /// Get the configuration
    fn config(&self) -> &StyleConfig;
}

// ============================================================================
// Style Transfer Trait
// ============================================================================

/// Trait for voice style transfer.
pub trait StyleTransfer {
    /// Apply style transfer to audio.
    ///
    /// # Arguments
    /// * `source_audio` - Source audio to convert (content)
    /// * `target_style` - Target style to apply
    ///
    /// # Returns
    /// Converted audio with target style applied.
    ///
    /// # Errors
    /// Returns error if transfer fails.
    fn transfer(&self, source_audio: &[f32], target_style: &StyleVector) -> VoiceResult<Vec<f32>>;

    /// Apply style transfer using reference audio.
    ///
    /// Extracts style from reference and applies to source.
    ///
    /// # Errors
    /// Returns error if style extraction or transfer fails.
    fn transfer_from_reference(
        &self,
        source_audio: &[f32],
        reference_audio: &[f32],
    ) -> VoiceResult<Vec<f32>>;

    /// Get the configuration
    fn config(&self) -> &StyleConfig;
}

// ============================================================================
// Stub Implementations
// ============================================================================

/// Global Style Token (GST) based style encoder.
///
/// Reference: Wang et al., 2018 - Style Tokens for Expressive Speech Synthesis.
///
/// # Note
/// This is a stub - requires model weights.
#[derive(Debug)]
pub struct GstEncoder {
    config: StyleConfig,
}

impl GstEncoder {
    /// Create new GST encoder
    #[must_use]
    pub fn new(config: StyleConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(StyleConfig::default())
    }
}

impl StyleEncoder for GstEncoder {
    fn encode(&self, audio: &[f32]) -> VoiceResult<StyleVector> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty audio".to_string()));
        }
        Err(VoiceError::NotImplemented(
            "GST encoder requires model weights".to_string(),
        ))
    }

    fn config(&self) -> &StyleConfig {
        &self.config
    }
}

/// AutoVC-based voice style transfer.
///
/// Reference: Qian et al., 2019 - `AutoVC`: Zero-Shot Voice Style Transfer.
///
/// # Note
/// This is a stub - requires model weights.
#[derive(Debug)]
pub struct AutoVcTransfer {
    config: StyleConfig,
}

impl AutoVcTransfer {
    /// Create new `AutoVC` transfer
    #[must_use]
    pub fn new(config: StyleConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(StyleConfig::default())
    }
}

include!("compute.rs");
include!("style_tests.rs");
