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

impl VoiceCloner for YourTtsCloner {
    fn create_profile(
        &self,
        reference_audio: &[f32],
        _speaker_id: &str,
    ) -> VoiceResult<VoiceProfile> {
        if reference_audio.is_empty() {
            return Err(VoiceError::InvalidAudio(
                "empty reference audio".to_string(),
            ));
        }

        let min_samples = self.config.min_reference_samples();
        if reference_audio.len() < min_samples {
            return Err(VoiceError::InvalidAudio(format!(
                "reference audio too short: {} samples, need at least {}",
                reference_audio.len(),
                min_samples
            )));
        }

        Err(VoiceError::NotImplemented(
            "YourTTS requires model weights".to_string(),
        ))
    }

    fn synthesize(&self, text: &str, profile: &VoiceProfile) -> VoiceResult<Vec<f32>> {
        if text.is_empty() {
            return Err(VoiceError::InvalidConfig("empty text".to_string()));
        }
        if !profile.is_ready() {
            return Err(VoiceError::ModelNotLoaded);
        }

        Err(VoiceError::NotImplemented(
            "YourTTS requires model weights".to_string(),
        ))
    }

    fn adapt(&self, profile: &mut VoiceProfile, additional_audio: &[f32]) -> VoiceResult<()> {
        if additional_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty audio".to_string()));
        }
        if !self.config.enable_adaptation {
            return Err(VoiceError::InvalidConfig(
                "adaptation not enabled in config".to_string(),
            ));
        }
        let _ = profile; // Mark as used

        Err(VoiceError::NotImplemented(
            "YourTTS adaptation requires model weights".to_string(),
        ))
    }

    fn config(&self) -> &CloningConfig {
        &self.config
    }
}

/// SV2TTS-based speaker encoder.
///
/// Reference: Jia et al., 2018 - Transfer Learning from Speaker Verification to TTS.
///
/// # Note
/// This is a stub - requires model weights.
#[derive(Debug)]
pub struct Sv2TtsSpeakerEncoder {
    embedding_dim: usize,
}

impl Sv2TtsSpeakerEncoder {
    /// Create new SV2TTS speaker encoder
    #[must_use]
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }

    /// Create with default dimension (256)
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(256)
    }
}

impl SpeakerEncoder for Sv2TtsSpeakerEncoder {
    fn encode(&self, audio: &[f32]) -> VoiceResult<SpeakerEmbedding> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty audio".to_string()));
        }

        Err(VoiceError::NotImplemented(
            "SV2TTS requires model weights".to_string(),
        ))
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Verify that two voice profiles represent the same speaker.
///
/// # Arguments
/// * `profile_a` - First voice profile
/// * `profile_b` - Second voice profile
/// * `threshold` - Similarity threshold (default 0.75)
///
/// # Returns
/// True if profiles likely represent the same speaker.
pub fn verify_same_speaker(
    profile_a: &VoiceProfile,
    profile_b: &VoiceProfile,
    threshold: f32,
) -> VoiceResult<bool> {
    let similarity = profile_a.similarity(profile_b)?;
    Ok(similarity >= threshold)
}

/// Estimate quality score from reference audio.
///
/// Based on:
/// - Duration (longer is better, up to a point)
/// - Signal-to-noise ratio (estimated)
/// - Speech activity (percentage of non-silence)
#[must_use]
pub fn estimate_quality(audio: &[f32], sample_rate: u32) -> f32 {
    if audio.is_empty() || sample_rate == 0 {
        return 0.0;
    }

    // Duration score (3-30 seconds is optimal)
    let duration = audio.len() as f32 / sample_rate as f32;
    let duration_score = if duration < 3.0 {
        duration / 3.0
    } else if duration > 30.0 {
        30.0 / duration
    } else {
        1.0
    };

    // Energy score (check for silence)
    let rms: f32 = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
    let energy_score = (rms * 10.0).min(1.0); // Scale RMS to 0-1

    // Activity score (frames above threshold)
    let frame_size = sample_rate as usize / 100; // 10ms frames
    let frame_count = audio.len().saturating_sub(frame_size) / (frame_size / 2);
    if frame_count == 0 {
        return duration_score * energy_score;
    }

    let threshold = rms * 0.1;
    let mut active_frames = 0_usize;

    for i in 0..frame_count {
        let start = i * (frame_size / 2);
        let end = (start + frame_size).min(audio.len());
        let frame_energy: f32 =
            (audio[start..end].iter().map(|x| x * x).sum::<f32>() / (end - start) as f32).sqrt();
        if frame_energy > threshold {
            active_frames += 1;
        }
    }

    let activity_score = active_frames as f32 / frame_count as f32;

    // Combine scores (weighted average)
    duration_score * 0.3 + energy_score * 0.3 + activity_score * 0.4
}

/// Merge multiple voice profiles into one.
///
/// Averages embeddings from profiles representing the same speaker.
///
/// # Errors
/// Returns error if profiles are empty or have incompatible embeddings.
pub fn merge_profiles(profiles: &[VoiceProfile]) -> VoiceResult<VoiceProfile> {
    if profiles.is_empty() {
        return Err(VoiceError::InvalidConfig(
            "cannot merge empty profile list".to_string(),
        ));
    }

    // Collect all embeddings
    let embeddings: Vec<&SpeakerEmbedding> = profiles
        .iter()
        .filter_map(|p| p.embedding.as_ref())
        .collect();

    if embeddings.is_empty() {
        return Err(VoiceError::InvalidConfig(
            "no profiles have embeddings".to_string(),
        ));
    }

    let dim = embeddings[0].dim();
    for emb in &embeddings {
        if emb.dim() != dim {
            return Err(VoiceError::DimensionMismatch {
                expected: dim,
                got: emb.dim(),
            });
        }
    }

    // Average embeddings
    let mut avg = vec![0.0_f32; dim];
    let count = embeddings.len() as f32;

    for emb in &embeddings {
        for (i, &val) in emb.as_slice().iter().enumerate() {
            avg[i] += val / count;
        }
    }

    // Compute combined metrics
    let total_duration: f32 = profiles.iter().map(|p| p.reference_duration).sum();
    let avg_quality: f32 =
        profiles.iter().map(|p| p.quality_score).sum::<f32>() / profiles.len() as f32;
    let any_adapted = profiles.iter().any(VoiceProfile::is_adapted);

    let speaker_id = profiles[0].speaker_id.clone();
    let mut merged = VoiceProfile::new(speaker_id);
    merged.set_embedding(SpeakerEmbedding::from_vec(avg));
    merged.set_reference_duration(total_duration);
    merged.set_quality_score(avg_quality);
    merged.set_adapted(any_adapted);

    Ok(merged)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloning_config_default() {
        let config = CloningConfig::default();
        assert!((config.min_reference_duration - 3.0).abs() < f32::EPSILON);
        assert!((config.max_reference_duration - 30.0).abs() < f32::EPSILON);
        assert_eq!(config.sample_rate, 22050);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cloning_config_few_shot() {
        let config = CloningConfig::few_shot();
        assert!((config.min_reference_duration - 3.0).abs() < f32::EPSILON);
        assert!((config.max_reference_duration - 10.0).abs() < f32::EPSILON);
        assert!(!config.enable_adaptation);
    }

    #[test]
    fn test_cloning_config_zero_shot() {
        let config = CloningConfig::zero_shot();
        assert!((config.min_reference_duration - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cloning_config_with_adaptation() {
        let config = CloningConfig::with_adaptation();
        assert!(config.enable_adaptation);
        assert!((config.min_reference_duration - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cloning_config_validation() {
        let mut config = CloningConfig::default();

        config.min_reference_duration = 0.0;
        assert!(config.validate().is_err());

        config.min_reference_duration = 3.0;
        config.max_reference_duration = 2.0; // Less than min
        assert!(config.validate().is_err());

        config.max_reference_duration = 30.0;
        config.similarity_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cloning_config_samples() {
        let config = CloningConfig::default();
        let min_samples = config.min_reference_samples();
        let max_samples = config.max_reference_samples();

        assert_eq!(min_samples, (3.0 * 22050.0) as usize);
        assert_eq!(max_samples, (30.0 * 22050.0) as usize);
    }

    #[test]
    fn test_voice_profile_new() {
        let profile = VoiceProfile::new("speaker_1".to_string());
        assert_eq!(profile.speaker_id(), "speaker_1");
        assert!(profile.embedding().is_none());
        assert!(!profile.is_ready());
        assert!(!profile.is_adapted());
    }

    #[test]
    fn test_voice_profile_with_embedding() {
        let emb = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0]);
        let profile = VoiceProfile::with_embedding("speaker_2".to_string(), emb);

        assert_eq!(profile.speaker_id(), "speaker_2");
        assert!(profile.is_ready());
        assert!((profile.quality_score() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_voice_profile_setters() {
        let mut profile = VoiceProfile::new("test".to_string());

        profile.set_embedding(SpeakerEmbedding::from_vec(vec![1.0, 2.0]));
        assert!(profile.is_ready());

        profile.set_reference_duration(5.5);
        assert!((profile.reference_duration() - 5.5).abs() < f32::EPSILON);

        profile.set_quality_score(0.8);
        assert!((profile.quality_score() - 0.8).abs() < f32::EPSILON);

        // Test clamping
        profile.set_quality_score(1.5);
        assert!((profile.quality_score() - 1.0).abs() < f32::EPSILON);

        profile.set_adapted(true);
        assert!(profile.is_adapted());
    }

    #[test]
    fn test_voice_profile_similarity() {
        let emb1 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
        let emb2 = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
        let emb3 = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);

        let profile1 = VoiceProfile::with_embedding("a".to_string(), emb1);
        let profile2 = VoiceProfile::with_embedding("b".to_string(), emb2);
        let profile3 = VoiceProfile::with_embedding("c".to_string(), emb3);

        let sim_same = profile1.similarity(&profile2).expect("similarity failed");
        assert!((sim_same - 1.0).abs() < 1e-6);

        let sim_diff = profile1.similarity(&profile3).expect("similarity failed");
        assert!((sim_diff - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_voice_profile_similarity_missing_embedding() {
        let profile1 = VoiceProfile::new("a".to_string());
        let profile2 =
            VoiceProfile::with_embedding("b".to_string(), SpeakerEmbedding::from_vec(vec![1.0]));

        assert!(profile1.similarity(&profile2).is_err());
        assert!(profile2.similarity(&profile1).is_err());
    }

    #[test]
    fn test_voice_profile_similarity_dimension_mismatch() {
        let profile1 = VoiceProfile::with_embedding(
            "a".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        );
        let profile2 = VoiceProfile::with_embedding(
            "b".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]),
        );

        assert!(profile1.similarity(&profile2).is_err());
    }

    #[test]
    fn test_yourtts_cloner_stub() {
        let cloner = YourTtsCloner::default_config();
        let audio = vec![0.0_f32; 22050 * 5]; // 5 seconds
        assert!(cloner.create_profile(&audio, "test").is_err());
    }

    #[test]
    fn test_yourtts_cloner_empty_audio() {
        let cloner = YourTtsCloner::default_config();
        let result = cloner.create_profile(&[], "test");
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_yourtts_cloner_short_audio() {
        let cloner = YourTtsCloner::default_config();
        let audio = vec![0.0_f32; 1000]; // Too short
        let result = cloner.create_profile(&audio, "test");
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_yourtts_synthesize() {
        let cloner = YourTtsCloner::default_config();
        let profile = VoiceProfile::with_embedding(
            "test".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0; 256]),
        );

        // Empty text
        assert!(cloner.synthesize("", &profile).is_err());

        // Not implemented
        assert!(cloner.synthesize("Hello world", &profile).is_err());
    }

    #[test]
    fn test_yourtts_synthesize_not_ready() {
        let cloner = YourTtsCloner::default_config();
        let profile = VoiceProfile::new("test".to_string()); // No embedding

        let result = cloner.synthesize("Hello", &profile);
        assert!(matches!(result, Err(VoiceError::ModelNotLoaded)));
    }

    #[test]
    fn test_yourtts_adapt() {
        let mut cloner = YourTtsCloner::default_config();
        let mut profile = VoiceProfile::with_embedding(
            "test".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0; 256]),
        );
        let audio = vec![0.0_f32; 22050];

        // Adaptation disabled
        assert!(cloner.adapt(&mut profile, &audio).is_err());

        // Enable adaptation
        cloner = YourTtsCloner::new(CloningConfig::with_adaptation());
        // Still fails (not implemented)
        assert!(cloner.adapt(&mut profile, &audio).is_err());
    }

    #[test]
    fn test_sv2tts_encoder_stub() {
        let encoder = Sv2TtsSpeakerEncoder::default_config();
        assert_eq!(encoder.embedding_dim(), 256);

        let audio = vec![0.0_f32; 16000];
        assert!(encoder.encode(&audio).is_err());
    }

    #[test]
    fn test_sv2tts_encoder_empty() {
        let encoder = Sv2TtsSpeakerEncoder::default_config();
        assert!(matches!(
            encoder.encode(&[]),
            Err(VoiceError::InvalidAudio(_))
        ));
    }

    #[test]
    fn test_verify_same_speaker() {
        let profile1 = VoiceProfile::with_embedding(
            "a".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]),
        );
        let profile2 = VoiceProfile::with_embedding(
            "b".to_string(),
            SpeakerEmbedding::from_vec(vec![0.99, 0.1, 0.0]), // Similar
        );
        let profile3 = VoiceProfile::with_embedding(
            "c".to_string(),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]), // Different
        );

        assert!(verify_same_speaker(&profile1, &profile2, 0.9).expect("verify failed"));
        assert!(!verify_same_speaker(&profile1, &profile3, 0.5).expect("verify failed"));
    }

    #[test]
    fn test_estimate_quality() {
        // Empty audio
        assert_eq!(estimate_quality(&[], 16000), 0.0);

        // Very short audio
        let short = vec![0.5_f32; 16000]; // 1 second
        let quality = estimate_quality(&short, 16000);
        assert!(quality > 0.0 && quality < 1.0);

        // Optimal duration
        let optimal = vec![0.5_f32; 16000 * 10]; // 10 seconds
        let quality2 = estimate_quality(&optimal, 16000);
        assert!(quality2 >= quality);
    }

    #[test]
    fn test_estimate_quality_zero_sample_rate() {
        let audio = vec![0.5_f32; 16000];
        assert_eq!(estimate_quality(&audio, 0), 0.0);
    }

    #[test]
    fn test_merge_profiles() {
        let profile1 = VoiceProfile::with_embedding(
            "speaker".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        );
        let profile2 = VoiceProfile::with_embedding(
            "speaker".to_string(),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0]),
        );

        let merged = merge_profiles(&[profile1, profile2]).expect("merge failed");
        assert!(merged.is_ready());

        let emb = merged.embedding().expect("no embedding");
        assert!((emb.as_slice()[0] - 0.5).abs() < 1e-6);
        assert!((emb.as_slice()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_merge_profiles_empty() {
        let profiles: Vec<VoiceProfile> = vec![];
        assert!(merge_profiles(&profiles).is_err());
    }

    #[test]
    fn test_merge_profiles_no_embeddings() {
        let profiles = vec![
            VoiceProfile::new("a".to_string()),
            VoiceProfile::new("b".to_string()),
        ];
        assert!(merge_profiles(&profiles).is_err());
    }

    #[test]
    fn test_merge_profiles_dimension_mismatch() {
        let profile1 = VoiceProfile::with_embedding(
            "a".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0, 0.0]),
        );
        let profile2 = VoiceProfile::with_embedding(
            "b".to_string(),
            SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]),
        );

        assert!(merge_profiles(&[profile1, profile2]).is_err());
    }

    #[test]
    fn test_merge_profiles_metrics() {
        let mut profile1 = VoiceProfile::with_embedding(
            "speaker".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0]),
        );
        profile1.set_reference_duration(5.0);
        profile1.set_quality_score(0.8);

        let mut profile2 = VoiceProfile::with_embedding(
            "speaker".to_string(),
            SpeakerEmbedding::from_vec(vec![1.0]),
        );
        profile2.set_reference_duration(10.0);
        profile2.set_quality_score(0.6);
        profile2.set_adapted(true);

        let merged = merge_profiles(&[profile1, profile2]).expect("merge failed");
        assert!((merged.reference_duration() - 15.0).abs() < f32::EPSILON);
        assert!((merged.quality_score() - 0.7).abs() < 1e-6);
        assert!(merged.is_adapted());
    }
}
