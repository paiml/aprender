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
//! - Qian, K., et al. (2019). AutoVC: Zero-Shot Voice Style Transfer.
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
/// Reference: Qian et al., 2019 - AutoVC: Zero-Shot Voice Style Transfer.
///
/// # Note
/// This is a stub - requires model weights.
#[derive(Debug)]
pub struct AutoVcTransfer {
    config: StyleConfig,
}

impl AutoVcTransfer {
    /// Create new AutoVC transfer
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

impl StyleTransfer for AutoVcTransfer {
    fn transfer(&self, source_audio: &[f32], _target_style: &StyleVector) -> VoiceResult<Vec<f32>> {
        if source_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty source audio".to_string()));
        }
        Err(VoiceError::NotImplemented(
            "AutoVC requires model weights".to_string(),
        ))
    }

    fn transfer_from_reference(
        &self,
        source_audio: &[f32],
        reference_audio: &[f32],
    ) -> VoiceResult<Vec<f32>> {
        if source_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty source audio".to_string()));
        }
        if reference_audio.is_empty() {
            return Err(VoiceError::InvalidAudio(
                "empty reference audio".to_string(),
            ));
        }
        Err(VoiceError::NotImplemented(
            "AutoVC requires model weights".to_string(),
        ))
    }

    fn config(&self) -> &StyleConfig {
        &self.config
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute prosody distance between two styles.
///
/// Focuses on pitch, energy, and rhythm differences.
#[must_use]
pub fn prosody_distance(a: &StyleVector, b: &StyleVector) -> f32 {
    if a.prosody.len() != b.prosody.len() {
        return f32::MAX;
    }
    if a.rhythm.len() != b.rhythm.len() {
        return f32::MAX;
    }

    let prosody_dist: f32 = a
        .prosody
        .iter()
        .zip(b.prosody.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    let rhythm_dist: f32 = a
        .rhythm
        .iter()
        .zip(b.rhythm.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    (prosody_dist + rhythm_dist).sqrt()
}

/// Compute timbre distance between two styles.
///
/// Focuses on spectral envelope differences.
#[must_use]
pub fn timbre_distance(a: &StyleVector, b: &StyleVector) -> f32 {
    if a.timbre.len() != b.timbre.len() {
        return f32::MAX;
    }

    let dist: f32 = a
        .timbre
        .iter()
        .zip(b.timbre.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    dist.sqrt()
}

/// Compute total style distance (Euclidean).
#[must_use]
pub fn style_distance(a: &StyleVector, b: &StyleVector) -> f32 {
    if a.dim() != b.dim() {
        return f32::MAX;
    }

    let flat_a = a.to_flat();
    let flat_b = b.to_flat();

    let dist: f32 = flat_a
        .iter()
        .zip(flat_b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    dist.sqrt()
}

/// Create style from speaker embedding (approximate).
///
/// Maps speaker embedding to style vector space.
/// Useful when only speaker embeddings are available.
#[must_use]
pub fn style_from_embedding(embedding: &SpeakerEmbedding, config: &StyleConfig) -> StyleVector {
    let emb_slice = embedding.as_slice();
    let emb_len = emb_slice.len();

    // Simple projection: split embedding into style components
    let prosody_len = config.prosody_dim.min(emb_len);
    let timbre_len = config.timbre_dim.min(emb_len.saturating_sub(prosody_len));
    let rhythm_len = config
        .rhythm_dim
        .min(emb_len.saturating_sub(prosody_len + timbre_len));

    let mut prosody = vec![0.0_f32; config.prosody_dim];
    let mut timbre = vec![0.0_f32; config.timbre_dim];
    let mut rhythm = vec![0.0_f32; config.rhythm_dim];

    // Copy available values
    prosody[..prosody_len].copy_from_slice(&emb_slice[..prosody_len]);

    if timbre_len > 0 {
        timbre[..timbre_len].copy_from_slice(&emb_slice[prosody_len..prosody_len + timbre_len]);
    }

    if rhythm_len > 0 {
        rhythm[..rhythm_len].copy_from_slice(
            &emb_slice[prosody_len + timbre_len..prosody_len + timbre_len + rhythm_len],
        );
    }

    StyleVector::new(prosody, timbre, rhythm)
}

/// Average multiple style vectors.
///
/// # Errors
/// Returns error if styles have different dimensions or list is empty.
pub fn average_styles(styles: &[StyleVector]) -> VoiceResult<StyleVector> {
    if styles.is_empty() {
        return Err(VoiceError::InvalidConfig(
            "cannot average empty style list".to_string(),
        ));
    }

    let first = &styles[0];
    let prosody_len = first.prosody.len();
    let timbre_len = first.timbre.len();
    let rhythm_len = first.rhythm.len();

    // Verify all dimensions match
    for style in styles.iter().skip(1) {
        if style.prosody.len() != prosody_len {
            return Err(VoiceError::DimensionMismatch {
                expected: prosody_len,
                got: style.prosody.len(),
            });
        }
        if style.timbre.len() != timbre_len {
            return Err(VoiceError::DimensionMismatch {
                expected: timbre_len,
                got: style.timbre.len(),
            });
        }
        if style.rhythm.len() != rhythm_len {
            return Err(VoiceError::DimensionMismatch {
                expected: rhythm_len,
                got: style.rhythm.len(),
            });
        }
    }

    let count = styles.len() as f32;

    let mut prosody = vec![0.0_f32; prosody_len];
    let mut timbre = vec![0.0_f32; timbre_len];
    let mut rhythm = vec![0.0_f32; rhythm_len];

    for style in styles {
        for (i, &v) in style.prosody.iter().enumerate() {
            prosody[i] += v / count;
        }
        for (i, &v) in style.timbre.iter().enumerate() {
            timbre[i] += v / count;
        }
        for (i, &v) in style.rhythm.iter().enumerate() {
            rhythm[i] += v / count;
        }
    }

    Ok(StyleVector::new(prosody, timbre, rhythm))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_config_default() {
        let config = StyleConfig::default();
        assert_eq!(config.prosody_dim, 64);
        assert_eq!(config.timbre_dim, 128);
        assert_eq!(config.rhythm_dim, 32);
        assert_eq!(config.total_dim(), 224);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_style_config_prosody_only() {
        let config = StyleConfig::prosody_only();
        assert!(config.preserve_pitch_contour);
        assert!((config.style_strength - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_config_full_conversion() {
        let config = StyleConfig::full_conversion();
        assert!(!config.preserve_pitch_contour);
        assert!((config.style_strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_config_validation_prosody() {
        let mut config = StyleConfig::default();
        config.prosody_dim = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_style_config_validation_strength() {
        let mut config = StyleConfig::default();
        config.style_strength = 1.5;
        assert!(config.validate().is_err());

        config.style_strength = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_style_vector_new() {
        let style = StyleVector::new(vec![1.0, 2.0], vec![3.0, 4.0, 5.0], vec![6.0]);
        assert_eq!(style.prosody().len(), 2);
        assert_eq!(style.timbre().len(), 3);
        assert_eq!(style.rhythm().len(), 1);
        assert_eq!(style.dim(), 6);
    }

    #[test]
    fn test_style_vector_zeros() {
        let config = StyleConfig::default();
        let style = StyleVector::zeros(&config);
        assert_eq!(style.prosody().len(), config.prosody_dim);
        assert_eq!(style.timbre().len(), config.timbre_dim);
        assert_eq!(style.rhythm().len(), config.rhythm_dim);
        assert!((style.l2_norm()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_vector_from_flat() {
        let config = StyleConfig {
            prosody_dim: 2,
            timbre_dim: 3,
            rhythm_dim: 1,
            ..StyleConfig::default()
        };
        let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let style = StyleVector::from_flat(&flat, &config).expect("from_flat failed");

        assert_eq!(style.prosody(), &[1.0, 2.0]);
        assert_eq!(style.timbre(), &[3.0, 4.0, 5.0]);
        assert_eq!(style.rhythm(), &[6.0]);
        assert_eq!(style.to_flat(), flat.to_vec());
    }

    #[test]
    fn test_style_vector_from_flat_wrong_size() {
        let config = StyleConfig::default();
        let flat = [1.0, 2.0, 3.0]; // Wrong size
        assert!(StyleVector::from_flat(&flat, &config).is_err());
    }

    #[test]
    fn test_style_vector_interpolate() {
        let style_a = StyleVector::new(vec![0.0, 0.0], vec![0.0, 0.0, 0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0, 1.0], vec![1.0, 1.0, 1.0], vec![1.0]);

        let mid = style_a
            .interpolate(&style_b, 0.5)
            .expect("interpolate failed");
        assert!((mid.prosody()[0] - 0.5).abs() < 1e-6);
        assert!((mid.timbre()[0] - 0.5).abs() < 1e-6);
        assert!((mid.rhythm()[0] - 0.5).abs() < 1e-6);

        // Edge cases
        let start = style_a
            .interpolate(&style_b, 0.0)
            .expect("interpolate 0 failed");
        assert!((start.prosody()[0] - 0.0).abs() < 1e-6);

        let end = style_a
            .interpolate(&style_b, 1.0)
            .expect("interpolate 1 failed");
        assert!((end.prosody()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_vector_interpolate_dimension_mismatch() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0, 1.0], vec![1.0], vec![1.0]);
        assert!(style_a.interpolate(&style_b, 0.5).is_err());
    }

    #[test]
    fn test_style_vector_l2_norm() {
        let style = StyleVector::new(vec![3.0], vec![4.0], vec![0.0]);
        assert!((style.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_vector_normalize() {
        let mut style = StyleVector::new(vec![3.0], vec![4.0], vec![0.0]);
        style.normalize();
        assert!((style.l2_norm() - 1.0).abs() < 1e-6);
        assert!((style.prosody()[0] - 0.6).abs() < 1e-6);
        assert!((style.timbre()[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_prosody_distance() {
        let style_a = StyleVector::new(vec![0.0, 0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![3.0, 4.0], vec![0.0], vec![0.0]);
        let dist = prosody_distance(&style_a, &style_b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_prosody_distance_mismatch() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![0.0, 0.0], vec![0.0], vec![0.0]);
        assert_eq!(prosody_distance(&style_a, &style_b), f32::MAX);
    }

    #[test]
    fn test_timbre_distance() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0, 0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![0.0], vec![3.0, 4.0], vec![0.0]);
        let dist = timbre_distance(&style_a, &style_b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_distance() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![3.0], vec![4.0], vec![0.0]);
        let dist = style_distance(&style_a, &style_b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_from_embedding() {
        let config = StyleConfig {
            prosody_dim: 64,
            timbre_dim: 64,
            rhythm_dim: 64,
            ..StyleConfig::default()
        };
        let embedding = SpeakerEmbedding::from_vec(vec![1.0; 192]);
        let style = style_from_embedding(&embedding, &config);

        assert_eq!(style.prosody().len(), 64);
        assert_eq!(style.timbre().len(), 64);
        assert_eq!(style.rhythm().len(), 64);
        assert!((style.prosody()[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_from_embedding_small() {
        let config = StyleConfig::default(); // 64 + 128 + 32 = 224
        let embedding = SpeakerEmbedding::from_vec(vec![1.0; 100]); // Smaller than total
        let style = style_from_embedding(&embedding, &config);

        // Should copy what's available, pad with zeros
        assert_eq!(style.dim(), config.total_dim());
    }

    #[test]
    fn test_average_styles() {
        let style_a = StyleVector::new(vec![0.0, 0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0, 1.0], vec![1.0], vec![1.0]);
        let styles = vec![style_a, style_b];

        let avg = average_styles(&styles).expect("average_styles failed");
        assert!((avg.prosody()[0] - 0.5).abs() < 1e-6);
        assert!((avg.timbre()[0] - 0.5).abs() < 1e-6);
        assert!((avg.rhythm()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_average_styles_empty() {
        let styles: Vec<StyleVector> = vec![];
        assert!(average_styles(&styles).is_err());
    }

    #[test]
    fn test_average_styles_dimension_mismatch() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0, 1.0], vec![1.0], vec![1.0]);
        let styles = vec![style_a, style_b];
        assert!(average_styles(&styles).is_err());
    }

    #[test]
    fn test_gst_encoder_stub() {
        let encoder = GstEncoder::default_config();
        let audio = vec![0.0_f32; 16000];
        assert!(encoder.encode(&audio).is_err());
    }

    #[test]
    fn test_gst_encoder_empty_audio() {
        let encoder = GstEncoder::default_config();
        let result = encoder.encode(&[]);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_autovc_transfer_stub() {
        let transfer = AutoVcTransfer::default_config();
        let source = vec![0.0_f32; 16000];
        let style = StyleVector::zeros(&StyleConfig::default());
        assert!(transfer.transfer(&source, &style).is_err());
    }

    #[test]
    fn test_autovc_transfer_empty_source() {
        let transfer = AutoVcTransfer::default_config();
        let style = StyleVector::zeros(&StyleConfig::default());
        let result = transfer.transfer(&[], &style);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_autovc_transfer_from_reference() {
        let transfer = AutoVcTransfer::default_config();
        let source = vec![0.0_f32; 16000];
        let reference = vec![0.0_f32; 16000];
        assert!(transfer
            .transfer_from_reference(&source, &reference)
            .is_err());
    }

    #[test]
    fn test_autovc_transfer_from_reference_empty() {
        let transfer = AutoVcTransfer::default_config();
        let source = vec![0.0_f32; 16000];

        let result = transfer.transfer_from_reference(&[], &source);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));

        let result = transfer.transfer_from_reference(&source, &[]);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }
}
