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
//! - **AutoVC**: Autoencoder-based, disentangled content and speaker
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
//! - Qian, K., et al. (2019). AutoVC: Zero-Shot Voice Style Transfer.
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
    /// AutoEncoder bottleneck
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
    pub fn with_metrics(
        mut self,
        confidence: f32,
        source_sim: f32,
        target_sim: f32,
    ) -> Self {
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
    fn synthesize(
        &self,
        content: &[Vec<f32>],
        speaker: &SpeakerEmbedding,
    ) -> VoiceResult<Vec<f32>>;
}

// ============================================================================
// AutoVC Converter
// ============================================================================

/// AutoVC-based voice converter.
///
/// Implements the AutoVC architecture for zero-shot voice conversion:
/// - Content encoder: Extracts speaker-independent features
/// - Speaker encoder: Extracts speaker embedding
/// - Decoder: Reconstructs mel spectrogram from content + speaker
///
/// Reference: Qian et al. (2019) "AutoVC: Zero-Shot Voice Style Transfer"
#[derive(Debug, Clone)]
pub struct AutoVcConverter {
    /// Configuration
    config: VoiceConversionConfig,
    /// Content encoder downsample factor
    downsample_factor: usize,
}

impl AutoVcConverter {
    /// Create a new AutoVC converter
    #[must_use]
    pub fn new(config: VoiceConversionConfig) -> Self {
        Self {
            config,
            downsample_factor: 32, // Typical AutoVC bottleneck
        }
    }

    /// Create with default AutoVC config
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

impl VoiceConverter for AutoVcConverter {
    fn config(&self) -> &VoiceConversionConfig {
        &self.config
    }

    fn convert(
        &self,
        source_audio: &[f32],
        _source_embedding: Option<&SpeakerEmbedding>,
        target_embedding: &SpeakerEmbedding,
    ) -> VoiceResult<ConversionResult> {
        if source_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty source audio".to_string()));
        }

        // Validate target embedding dimension
        if target_embedding.dim() != self.config.speaker_dim {
            return Err(VoiceError::DimensionMismatch {
                expected: self.config.speaker_dim,
                got: target_embedding.dim(),
            });
        }

        // Extract content features (speaker-independent)
        let content = self.extract_content(source_audio)?;

        // Synthesize with target speaker
        let converted_audio = self.synthesize(&content, target_embedding)?;

        let mut result = ConversionResult::new(converted_audio, self.config.sample_rate);

        // Estimate quality metrics (placeholder - would use neural network)
        result = result.with_metrics(0.85, 0.15, 0.75);

        Ok(result)
    }

    fn extract_content(&self, audio: &[f32]) -> VoiceResult<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // Simulated content extraction (would use neural network)
        let frame_samples = self.config.frame_samples();
        let num_frames = (audio.len() / frame_samples).max(1);
        let content_dim = self.config.content_dim;

        // Create bottleneck features (downsampled content)
        let downsampled_frames = num_frames / self.downsample_factor.max(1);
        let downsampled_frames = downsampled_frames.max(1);

        let content: Vec<Vec<f32>> = (0..downsampled_frames)
            .map(|i| {
                // Simplified: average energy over frame range as placeholder
                let start_sample = i * self.downsample_factor * frame_samples;
                let end_sample =
                    ((i + 1) * self.downsample_factor * frame_samples).min(audio.len());

                let energy = if end_sample > start_sample {
                    audio[start_sample..end_sample]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt()
                        / (end_sample - start_sample) as f32
                } else {
                    0.0
                };

                // Create content vector (placeholder)
                vec![energy; content_dim]
            })
            .collect();

        Ok(content)
    }

    fn synthesize(
        &self,
        content: &[Vec<f32>],
        _speaker: &SpeakerEmbedding,
    ) -> VoiceResult<Vec<f32>> {
        if content.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty content features".to_string()));
        }

        // Simulated synthesis (would use neural vocoder)
        let frame_samples = self.config.frame_samples();
        let num_output_frames = content.len() * self.downsample_factor;
        let output_len = num_output_frames * frame_samples;

        // Generate placeholder audio (would use decoder + vocoder)
        let audio: Vec<f32> = (0..output_len)
            .map(|i| {
                let frame_idx = (i / frame_samples) / self.downsample_factor.max(1);
                let frame_idx = frame_idx.min(content.len().saturating_sub(1));

                // Use content energy to modulate output
                let energy = content.get(frame_idx).and_then(|f| f.first()).copied().unwrap_or(0.0);

                // Generate simple sine wave modulated by energy
                let t = i as f32 / self.config.sample_rate as f32;
                let freq = 200.0; // Base frequency
                (2.0 * std::f32::consts::PI * freq * t).sin() * energy * 0.5
            })
            .collect();

        Ok(audio)
    }
}

impl Default for AutoVcConverter {
    fn default() -> Self {
        Self::default_autovc()
    }
}

// ============================================================================
// PPG-based Converter
// ============================================================================

/// PPG (Phonetic Posteriorgram) based voice converter.
///
/// Uses ASR-derived phonetic features as the content bottleneck:
/// - More explicit phonetic representation
/// - Better for cross-lingual conversion
/// - Requires pre-trained ASR model for PPG extraction
#[derive(Debug, Clone)]
pub struct PpgConverter {
    /// Configuration
    config: VoiceConversionConfig,
    /// Number of phoneme classes (typical: 144 for multilingual)
    num_phonemes: usize,
}

impl PpgConverter {
    /// Create a new PPG-based converter
    #[must_use]
    pub fn new(config: VoiceConversionConfig, num_phonemes: usize) -> Self {
        Self {
            config,
            num_phonemes,
        }
    }

    /// Create with default PPG config
    #[must_use]
    pub fn default_ppg() -> Self {
        Self::new(VoiceConversionConfig::ppg_based(), 144)
    }

    /// Get number of phoneme classes
    #[must_use]
    pub fn num_phonemes(&self) -> usize {
        self.num_phonemes
    }
}

impl VoiceConverter for PpgConverter {
    fn config(&self) -> &VoiceConversionConfig {
        &self.config
    }

    fn convert(
        &self,
        source_audio: &[f32],
        _source_embedding: Option<&SpeakerEmbedding>,
        target_embedding: &SpeakerEmbedding,
    ) -> VoiceResult<ConversionResult> {
        if source_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty source audio".to_string()));
        }

        // Extract PPG features
        let ppg = self.extract_content(source_audio)?;

        // Synthesize with target speaker
        let converted_audio = self.synthesize(&ppg, target_embedding)?;

        let mut result = ConversionResult::new(converted_audio, self.config.sample_rate);
        result = result.with_metrics(0.80, 0.10, 0.80);

        Ok(result)
    }

    fn extract_content(&self, audio: &[f32]) -> VoiceResult<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // Simulated PPG extraction (would use ASR model)
        let frame_samples = self.config.frame_samples();
        let num_frames = (audio.len() / frame_samples).max(1);

        // Generate PPG (phonetic posteriors)
        let ppg: Vec<Vec<f32>> = (0..num_frames)
            .map(|i| {
                let start = i * frame_samples;
                let end = ((i + 1) * frame_samples).min(audio.len());

                // Simple energy-based placeholder
                let energy = if end > start {
                    audio[start..end].iter().map(|x| x.abs()).sum::<f32>()
                        / (end - start) as f32
                } else {
                    0.0
                };

                // Create sparse phonetic posterior (placeholder)
                let mut posteriors = vec![0.0f32; self.num_phonemes];
                let dominant_phoneme = ((energy * 100.0) as usize) % self.num_phonemes;
                if let Some(p) = posteriors.get_mut(dominant_phoneme) {
                    *p = 0.8;
                }
                // Spread remaining probability
                for p in &mut posteriors {
                    *p += 0.2 / self.num_phonemes as f32;
                }
                posteriors
            })
            .collect();

        Ok(ppg)
    }

    fn synthesize(
        &self,
        content: &[Vec<f32>],
        _speaker: &SpeakerEmbedding,
    ) -> VoiceResult<Vec<f32>> {
        if content.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty PPG features".to_string()));
        }

        // Simulated synthesis (would use acoustic model + vocoder)
        let frame_samples = self.config.frame_samples();
        let output_len = content.len() * frame_samples;

        let audio: Vec<f32> = (0..output_len)
            .map(|i| {
                let frame_idx = (i / frame_samples).min(content.len().saturating_sub(1));
                let posteriors = &content[frame_idx];

                // Find dominant phoneme
                let (max_idx, max_val) = posteriors
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));

                // Generate frequency based on phoneme class
                let base_freq = 100.0 + (max_idx as f32 * 10.0);
                let t = i as f32 / self.config.sample_rate as f32;
                (2.0 * std::f32::consts::PI * base_freq * t).sin() * max_val * 0.3
            })
            .collect();

        Ok(audio)
    }
}

impl Default for PpgConverter {
    fn default() -> Self {
        Self::default_ppg()
    }
}

// ============================================================================
// Pitch Conversion Utilities
// ============================================================================

/// Convert F0 (fundamental frequency) contour from source to target speaker.
///
/// # Arguments
/// * `f0_contour` - Source F0 values in Hz (0.0 for unvoiced)
/// * `source_mean` - Mean F0 of source speaker
/// * `source_std` - Standard deviation of source F0
/// * `target_mean` - Mean F0 of target speaker
/// * `target_std` - Standard deviation of target F0
///
/// # Returns
/// Converted F0 contour
#[must_use]
pub fn convert_f0(
    f0_contour: &[f32],
    source_mean: f32,
    source_std: f32,
    target_mean: f32,
    target_std: f32,
) -> Vec<f32> {
    if f0_contour.is_empty() || source_std <= 0.0 {
        return f0_contour.to_vec();
    }

    let target_std = if target_std <= 0.0 { source_std } else { target_std };

    f0_contour
        .iter()
        .map(|&f0| {
            if f0 <= 0.0 {
                0.0 // Preserve unvoiced
            } else {
                // Linear transformation: (f0 - src_mean) / src_std * tgt_std + tgt_mean
                let normalized = (f0 - source_mean) / source_std;
                (normalized * target_std + target_mean).max(50.0) // Clamp to reasonable range
            }
        })
        .collect()
}

/// Calculate F0 statistics from contour (mean, std of voiced frames).
///
/// # Returns
/// (mean, std) of voiced F0 values
#[must_use]
pub fn f0_statistics(f0_contour: &[f32]) -> (f32, f32) {
    let voiced: Vec<f32> = f0_contour.iter().copied().filter(|&f| f > 0.0).collect();

    if voiced.is_empty() {
        return (0.0, 0.0);
    }

    let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
    let variance = voiced.iter().map(|&f| (f - mean).powi(2)).sum::<f32>() / voiced.len() as f32;
    let std = variance.sqrt();

    (mean, std)
}

/// Convert pitch ratio to semitone shift.
#[must_use]
pub fn ratio_to_semitones(ratio: f32) -> f32 {
    if ratio <= 0.0 {
        0.0
    } else {
        12.0 * ratio.log2()
    }
}

/// Convert semitone shift to pitch ratio.
#[must_use]
pub fn semitones_to_ratio(semitones: f32) -> f32 {
    2.0_f32.powf(semitones / 12.0)
}

// ============================================================================
// Quality Metrics
// ============================================================================

/// Calculate voice conversion quality metrics.
///
/// # Arguments
/// * `source_embedding` - Original speaker embedding
/// * `target_embedding` - Target speaker embedding
/// * `converted_embedding` - Embedding of converted audio
///
/// # Returns
/// (source_similarity, target_similarity, conversion_score)
#[must_use]
pub fn conversion_quality(
    source_embedding: &SpeakerEmbedding,
    target_embedding: &SpeakerEmbedding,
    converted_embedding: &SpeakerEmbedding,
) -> (f32, f32, f32) {
    // Calculate cosine similarities
    let source_sim = cosine_similarity(converted_embedding.as_slice(), source_embedding.as_slice());
    let target_sim = cosine_similarity(converted_embedding.as_slice(), target_embedding.as_slice());

    // Conversion score: high target similarity, low source similarity
    let conversion_score = target_sim * (1.0 - source_sim);

    (source_sim, target_sim, conversion_score)
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_mode_default() {
        assert_eq!(ConversionMode::default(), ConversionMode::AnyToAny);
    }

    #[test]
    fn test_bottleneck_type_default() {
        assert_eq!(BottleneckType::default(), BottleneckType::AutoEncoder);
    }

    #[test]
    fn test_config_default() {
        let config = VoiceConversionConfig::default();
        assert_eq!(config.mode, ConversionMode::AnyToAny);
        assert_eq!(config.speaker_dim, 256);
        assert_eq!(config.sample_rate, 16000);
        assert!(config.convert_f0);
    }

    #[test]
    fn test_config_autovc() {
        let config = VoiceConversionConfig::autovc();
        assert_eq!(config.mode, ConversionMode::AnyToAny);
        assert_eq!(config.bottleneck, BottleneckType::AutoEncoder);
    }

    #[test]
    fn test_config_stargan_vc() {
        let config = VoiceConversionConfig::stargan_vc();
        assert_eq!(config.mode, ConversionMode::ManyToOne);
        assert_eq!(config.speaker_dim, 64);
    }

    #[test]
    fn test_config_ppg_based() {
        let config = VoiceConversionConfig::ppg_based();
        assert_eq!(config.bottleneck, BottleneckType::Ppg);
        assert_eq!(config.content_dim, 144);
    }

    #[test]
    fn test_config_realtime() {
        let config = VoiceConversionConfig::realtime();
        assert_eq!(config.frame_shift_ms, 5);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = VoiceConversionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_speaker_dim() {
        let config = VoiceConversionConfig {
            speaker_dim: 0,
            ..VoiceConversionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_content_dim() {
        let config = VoiceConversionConfig {
            content_dim: 0,
            ..VoiceConversionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_pitch_ratio() {
        let config = VoiceConversionConfig {
            pitch_ratio: -1.0,
            ..VoiceConversionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_formant() {
        let config = VoiceConversionConfig {
            formant_preservation: 1.5,
            ..VoiceConversionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_frame_samples() {
        let config = VoiceConversionConfig::default();
        // 16000 Hz * 10 ms / 1000 = 160 samples
        assert_eq!(config.frame_samples(), 160);
    }

    #[test]
    fn test_conversion_result_new() {
        let audio = vec![0.1, 0.2, 0.3];
        let result = ConversionResult::new(audio.clone(), 16000);
        assert_eq!(result.audio, audio);
        assert_eq!(result.sample_rate, 16000);
        assert!(result.duration_secs > 0.0);
    }

    #[test]
    fn test_conversion_result_with_metrics() {
        let result = ConversionResult::new(vec![0.1; 1600], 16000)
            .with_metrics(0.9, 0.1, 0.85);
        assert_eq!(result.confidence, 0.9);
        assert_eq!(result.source_similarity, 0.1);
        assert_eq!(result.target_similarity, 0.85);
    }

    #[test]
    fn test_autovc_converter_new() {
        let converter = AutoVcConverter::default();
        assert_eq!(converter.config().mode, ConversionMode::AnyToAny);
        assert_eq!(converter.downsample_factor(), 32);
    }

    #[test]
    fn test_autovc_extract_content() {
        let converter = AutoVcConverter::default();
        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();
        let content = converter.extract_content(&audio);
        assert!(content.is_ok());
        let features = content.expect("extraction failed");
        assert!(!features.is_empty());
    }

    #[test]
    fn test_autovc_extract_content_empty() {
        let converter = AutoVcConverter::default();
        let result = converter.extract_content(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_autovc_convert() {
        let converter = AutoVcConverter::new(VoiceConversionConfig {
            speaker_dim: 192,
            ..VoiceConversionConfig::default()
        });

        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();
        let target = SpeakerEmbedding::zeros(192);

        let result = converter.convert(&audio, None, &target);
        assert!(result.is_ok());
        let conversion = result.expect("conversion failed");
        assert!(!conversion.audio.is_empty());
    }

    #[test]
    fn test_autovc_convert_empty() {
        let converter = AutoVcConverter::default();
        let target = SpeakerEmbedding::zeros(256);
        let result = converter.convert(&[], None, &target);
        assert!(result.is_err());
    }

    #[test]
    fn test_autovc_convert_dim_mismatch() {
        let converter = AutoVcConverter::default(); // expects 256
        let target = SpeakerEmbedding::zeros(128); // wrong dimension
        let audio = vec![0.1f32; 1600];
        let result = converter.convert(&audio, None, &target);
        assert!(result.is_err());
    }

    #[test]
    fn test_ppg_converter_new() {
        let converter = PpgConverter::default();
        assert_eq!(converter.num_phonemes(), 144);
    }

    #[test]
    fn test_ppg_extract_content() {
        let converter = PpgConverter::default();
        let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.002).sin()).collect();
        let ppg = converter.extract_content(&audio);
        assert!(ppg.is_ok());
        let features = ppg.expect("extraction failed");
        assert!(!features.is_empty());
        // Each frame should have 144 phoneme posteriors
        assert_eq!(features[0].len(), 144);
    }

    #[test]
    fn test_ppg_convert() {
        let converter = PpgConverter::new(
            VoiceConversionConfig {
                speaker_dim: 192,
                ..VoiceConversionConfig::ppg_based()
            },
            144,
        );

        let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.002).sin()).collect();
        let target = SpeakerEmbedding::zeros(192);

        let result = converter.convert(&audio, None, &target);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convert_f0_basic() {
        let f0 = vec![200.0, 220.0, 0.0, 180.0]; // Hz, with unvoiced
        let converted = convert_f0(&f0, 200.0, 20.0, 150.0, 15.0);

        assert_eq!(converted.len(), f0.len());
        assert_eq!(converted[2], 0.0); // Unvoiced preserved
        assert!(converted[0] < f0[0]); // Pitch lowered
    }

    #[test]
    fn test_convert_f0_empty() {
        let converted = convert_f0(&[], 200.0, 20.0, 150.0, 15.0);
        assert!(converted.is_empty());
    }

    #[test]
    fn test_f0_statistics() {
        let f0 = vec![200.0, 220.0, 0.0, 180.0];
        let (mean, std) = f0_statistics(&f0);

        // Mean of [200, 220, 180] = 200
        assert!((mean - 200.0).abs() < 1.0);
        assert!(std > 0.0);
    }

    #[test]
    fn test_f0_statistics_all_unvoiced() {
        let f0 = vec![0.0, 0.0, 0.0];
        let (mean, std) = f0_statistics(&f0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_ratio_to_semitones() {
        // Octave up = 12 semitones
        assert!((ratio_to_semitones(2.0) - 12.0).abs() < 0.01);
        // No change = 0 semitones
        assert!((ratio_to_semitones(1.0) - 0.0).abs() < 0.01);
        // Octave down = -12 semitones
        assert!((ratio_to_semitones(0.5) - (-12.0)).abs() < 0.01);
    }

    #[test]
    fn test_semitones_to_ratio() {
        // 12 semitones = octave up (2x)
        assert!((semitones_to_ratio(12.0) - 2.0).abs() < 0.01);
        // 0 semitones = no change
        assert!((semitones_to_ratio(0.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_conversion_quality() {
        let source = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
        let target = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
        let converted = SpeakerEmbedding::from_vec(vec![0.1, 0.9, 0.1]);

        let (src_sim, tgt_sim, score) = conversion_quality(&source, &target, &converted);

        // Should be more similar to target than source
        assert!(tgt_sim > src_sim);
        assert!(score > 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
