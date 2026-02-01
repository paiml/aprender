//! Text-to-Speech (TTS) module (GH-133).
//!
//! Provides TTS primitives for:
//! - Neural TTS synthesis
//! - Mel spectrogram generation from text
//! - Vocoder integration (HiFi-GAN, `WaveGlow`, etc.)
//! - Multi-speaker synthesis
//!
//! # Architecture
//!
//! ```text
//! Text → Text Processing → Acoustic Model → Mel Spectrogram → Vocoder → Audio
//!              ↓                                    ↑
//!        Phoneme/Grapheme                   [Speaker Embedding]
//!        Encoding                           [Prosody Control]
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::speech::tts::{TtsConfig, SpeechSynthesizer, SynthesisRequest};
//!
//! let config = TtsConfig::default();
//! assert_eq!(config.sample_rate, 22050);
//! assert_eq!(config.n_mels, 80);
//! ```
//!
//! # Supported Models
//!
//! - Tacotron2-style (attention-based)
//! - FastSpeech2-style (non-autoregressive)
//! - VITS-style (end-to-end variational)
//!
//! # References
//!
//! - Wang, Y., et al. (2017). Tacotron: End-to-End Speech Synthesis.
//! - Ren, Y., et al. (2020). `FastSpeech` 2: Fast and High-Quality TTS.
//! - Kim, J., et al. (2021). Conditional Variational Autoencoder with Adversarial Learning.
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible

use super::{SpeechError, SpeechResult};

// ============================================================================
// Configuration
// ============================================================================

/// TTS configuration.
#[derive(Debug, Clone)]
pub struct TtsConfig {
    /// Output sample rate in Hz
    pub sample_rate: u32,
    /// Number of mel channels
    pub n_mels: usize,
    /// Hop size for mel spectrogram (samples)
    pub hop_size: usize,
    /// Window size for mel spectrogram (samples)
    pub win_size: usize,
    /// Speaking rate multiplier (1.0 = normal)
    pub speaking_rate: f32,
    /// Pitch shift in semitones
    pub pitch_shift: f32,
    /// Energy scale (1.0 = normal)
    pub energy_scale: f32,
    /// Maximum text length
    pub max_text_length: usize,
    /// Maximum output duration in seconds
    pub max_output_duration: f32,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22050,
            n_mels: 80,
            hop_size: 256,
            win_size: 1024,
            speaking_rate: 1.0,
            pitch_shift: 0.0,
            energy_scale: 1.0,
            max_text_length: 500,
            max_output_duration: 30.0,
        }
    }
}

impl TtsConfig {
    /// Create high-quality configuration (48kHz)
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            sample_rate: 48000,
            n_mels: 128,
            hop_size: 512,
            win_size: 2048,
            ..Self::default()
        }
    }

    /// Create fast configuration (lower quality, faster synthesis)
    #[must_use]
    pub fn fast() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            hop_size: 160,
            win_size: 640,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> SpeechResult<()> {
        if self.sample_rate == 0 {
            return Err(SpeechError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if self.n_mels == 0 {
            return Err(SpeechError::InvalidConfig("n_mels must be > 0".to_string()));
        }
        if self.hop_size == 0 {
            return Err(SpeechError::InvalidConfig(
                "hop_size must be > 0".to_string(),
            ));
        }
        if self.win_size == 0 || self.win_size < self.hop_size {
            return Err(SpeechError::InvalidConfig(
                "win_size must be > 0 and >= hop_size".to_string(),
            ));
        }
        if self.speaking_rate <= 0.0 || self.speaking_rate > 5.0 {
            return Err(SpeechError::InvalidConfig(
                "speaking_rate must be in (0, 5]".to_string(),
            ));
        }
        if self.pitch_shift < -24.0 || self.pitch_shift > 24.0 {
            return Err(SpeechError::InvalidConfig(
                "pitch_shift must be in [-24, 24] semitones".to_string(),
            ));
        }
        if self.energy_scale <= 0.0 || self.energy_scale > 3.0 {
            return Err(SpeechError::InvalidConfig(
                "energy_scale must be in (0, 3]".to_string(),
            ));
        }
        if self.max_text_length == 0 {
            return Err(SpeechError::InvalidConfig(
                "max_text_length must be > 0".to_string(),
            ));
        }
        if self.max_output_duration <= 0.0 {
            return Err(SpeechError::InvalidConfig(
                "max_output_duration must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Get frames per second
    #[must_use]
    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_size as f32
    }

    /// Get maximum output samples
    #[must_use]
    pub fn max_output_samples(&self) -> usize {
        (self.max_output_duration * self.sample_rate as f32) as usize
    }
}

// ============================================================================
// Synthesis Request/Result
// ============================================================================

/// A synthesis request with text and optional controls.
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// Text to synthesize
    pub text: String,
    /// Optional speaker ID for multi-speaker synthesis
    pub speaker_id: Option<String>,
    /// Speaking rate multiplier (overrides config)
    pub speaking_rate: Option<f32>,
    /// Pitch shift in semitones (overrides config)
    pub pitch_shift: Option<f32>,
    /// Energy scale (overrides config)
    pub energy_scale: Option<f32>,
    /// Language code (e.g., "en", "es", "zh")
    pub language: Option<String>,
}

impl SynthesisRequest {
    /// Create a new synthesis request
    #[must_use]
    pub fn new(text: String) -> Self {
        Self {
            text,
            speaker_id: None,
            speaking_rate: None,
            pitch_shift: None,
            energy_scale: None,
            language: None,
        }
    }

    /// Set speaker ID
    #[must_use]
    pub fn with_speaker(mut self, speaker_id: String) -> Self {
        self.speaker_id = Some(speaker_id);
        self
    }

    /// Set speaking rate
    #[must_use]
    pub fn with_speaking_rate(mut self, rate: f32) -> Self {
        self.speaking_rate = Some(rate);
        self
    }

    /// Set pitch shift
    #[must_use]
    pub fn with_pitch_shift(mut self, semitones: f32) -> Self {
        self.pitch_shift = Some(semitones);
        self
    }

    /// Set energy scale
    #[must_use]
    pub fn with_energy_scale(mut self, scale: f32) -> Self {
        self.energy_scale = Some(scale);
        self
    }

    /// Set language
    #[must_use]
    pub fn with_language(mut self, language: String) -> Self {
        self.language = Some(language);
        self
    }

    /// Validate the request
    pub fn validate(&self, config: &TtsConfig) -> SpeechResult<()> {
        if self.text.is_empty() {
            return Err(SpeechError::InvalidConfig("empty text".to_string()));
        }
        if self.text.len() > config.max_text_length {
            return Err(SpeechError::InvalidConfig(format!(
                "text too long: {} chars, max {}",
                self.text.len(),
                config.max_text_length
            )));
        }
        if let Some(rate) = self.speaking_rate {
            if rate <= 0.0 || rate > 5.0 {
                return Err(SpeechError::InvalidConfig(
                    "speaking_rate must be in (0, 5]".to_string(),
                ));
            }
        }
        if let Some(shift) = self.pitch_shift {
            if !(-24.0..=24.0).contains(&shift) {
                return Err(SpeechError::InvalidConfig(
                    "pitch_shift must be in [-24, 24]".to_string(),
                ));
            }
        }
        if let Some(scale) = self.energy_scale {
            if scale <= 0.0 || scale > 3.0 {
                return Err(SpeechError::InvalidConfig(
                    "energy_scale must be in (0, 3]".to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Synthesis result containing audio and metadata.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Synthesized audio samples (mono)
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Mel spectrogram (`n_mels` x frames)
    pub mel_spectrogram: Option<Vec<Vec<f32>>>,
    /// Alignment info (if available)
    pub alignment: Option<Vec<AlignmentInfo>>,
    /// Phoneme sequence (if available)
    pub phonemes: Option<Vec<String>>,
}

impl SynthesisResult {
    /// Create a new synthesis result
    #[must_use]
    pub fn new(audio: Vec<f32>, sample_rate: u32) -> Self {
        let duration = if sample_rate > 0 {
            audio.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        Self {
            audio,
            sample_rate,
            duration,
            mel_spectrogram: None,
            alignment: None,
            phonemes: None,
        }
    }

    /// Set mel spectrogram
    pub fn with_mel(&mut self, mel: Vec<Vec<f32>>) {
        self.mel_spectrogram = Some(mel);
    }

    /// Set alignment info
    pub fn with_alignment(&mut self, alignment: Vec<AlignmentInfo>) {
        self.alignment = Some(alignment);
    }

    /// Set phoneme sequence
    pub fn with_phonemes(&mut self, phonemes: Vec<String>) {
        self.phonemes = Some(phonemes);
    }

    /// Get number of samples
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.audio.len()
    }

    /// Check if mel spectrogram is available
    #[must_use]
    pub fn has_mel(&self) -> bool {
        self.mel_spectrogram.is_some()
    }
}

/// Alignment information between text and audio.
#[derive(Debug, Clone)]
pub struct AlignmentInfo {
    /// Character or phoneme
    pub token: String,
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
}

impl AlignmentInfo {
    /// Create new alignment info
    #[must_use]
    pub fn new(token: String, start: f32, end: f32) -> Self {
        Self {
            token,
            start,
            end,
            confidence: 1.0,
        }
    }

    /// Set confidence
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Get duration
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }
}

// ============================================================================
// Synthesizer Trait
// ============================================================================

/// Trait for speech synthesis.
pub trait SpeechSynthesizer {
    /// Synthesize speech from text.
    ///
    /// # Arguments
    /// * `request` - Synthesis request with text and controls
    ///
    /// # Returns
    /// Synthesis result with audio and metadata.
    ///
    /// # Errors
    /// Returns error if synthesis fails.
    fn synthesize(&self, request: &SynthesisRequest) -> SpeechResult<SynthesisResult>;

    /// Get the configuration
    fn config(&self) -> &TtsConfig;

    /// Get available speakers (for multi-speaker models)
    fn available_speakers(&self) -> Vec<String>;

    /// Check if model supports a language
    fn supports_language(&self, language: &str) -> bool;
}

// ============================================================================
// Vocoder Trait
// ============================================================================

/// Trait for neural vocoder (mel to audio).
pub trait Vocoder {
    /// Convert mel spectrogram to audio.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram (`n_mels` x frames)
    ///
    /// # Returns
    /// Audio samples.
    ///
    /// # Errors
    /// Returns error if conversion fails.
    fn vocalize(&self, mel: &[Vec<f32>]) -> SpeechResult<Vec<f32>>;

    /// Get output sample rate
    fn sample_rate(&self) -> u32;

    /// Get expected number of mel channels
    fn n_mels(&self) -> usize;
}

// ============================================================================
// Stub Implementations
// ============================================================================

/// FastSpeech2-style TTS synthesizer.
///
/// Non-autoregressive, parallel synthesis for fast inference.
///
/// # Note
/// This is a stub - requires model weights.
#[derive(Debug)]
pub struct FastSpeech2Synthesizer {
    config: TtsConfig,
    speakers: Vec<String>,
}

impl FastSpeech2Synthesizer {
    /// Create new `FastSpeech2` synthesizer
    #[must_use]
    pub fn new(config: TtsConfig) -> Self {
        Self {
            config,
            speakers: vec!["default".to_string()],
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(TtsConfig::default())
    }

    /// Add a speaker
    pub fn add_speaker(&mut self, speaker: String) {
        if !self.speakers.contains(&speaker) {
            self.speakers.push(speaker);
        }
    }
}

impl SpeechSynthesizer for FastSpeech2Synthesizer {
    fn synthesize(&self, request: &SynthesisRequest) -> SpeechResult<SynthesisResult> {
        request.validate(&self.config)?;

        if let Some(ref speaker) = request.speaker_id {
            if !self.speakers.contains(speaker) {
                return Err(SpeechError::InvalidConfig(format!(
                    "unknown speaker: {speaker}"
                )));
            }
        }

        Err(SpeechError::ProcessingError(
            "FastSpeech2 requires model weights".to_string(),
        ))
    }

    fn config(&self) -> &TtsConfig {
        &self.config
    }

    fn available_speakers(&self) -> Vec<String> {
        self.speakers.clone()
    }

    fn supports_language(&self, language: &str) -> bool {
        // Stub: assume English only
        language == "en"
    }
}

/// VITS-style end-to-end TTS.
///
/// Variational inference with adversarial learning.
///
/// # Note
/// This is a stub - requires model weights.
#[derive(Debug)]
pub struct VitsSynthesizer {
    config: TtsConfig,
    speakers: Vec<String>,
    languages: Vec<String>,
}

impl VitsSynthesizer {
    /// Create new VITS synthesizer
    #[must_use]
    pub fn new(config: TtsConfig) -> Self {
        Self {
            config,
            speakers: vec!["default".to_string()],
            languages: vec!["en".to_string()],
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(TtsConfig::default())
    }

    /// Add a speaker
    pub fn add_speaker(&mut self, speaker: String) {
        if !self.speakers.contains(&speaker) {
            self.speakers.push(speaker);
        }
    }

    /// Add a language
    pub fn add_language(&mut self, language: String) {
        if !self.languages.contains(&language) {
            self.languages.push(language);
        }
    }
}

impl SpeechSynthesizer for VitsSynthesizer {
    fn synthesize(&self, request: &SynthesisRequest) -> SpeechResult<SynthesisResult> {
        request.validate(&self.config)?;

        if let Some(ref speaker) = request.speaker_id {
            if !self.speakers.contains(speaker) {
                return Err(SpeechError::InvalidConfig(format!(
                    "unknown speaker: {speaker}"
                )));
            }
        }

        if let Some(ref lang) = request.language {
            if !self.languages.contains(lang) {
                return Err(SpeechError::InvalidConfig(format!(
                    "unsupported language: {lang}"
                )));
            }
        }

        Err(SpeechError::ProcessingError(
            "VITS requires model weights".to_string(),
        ))
    }

    fn config(&self) -> &TtsConfig {
        &self.config
    }

    fn available_speakers(&self) -> Vec<String> {
        self.speakers.clone()
    }

    fn supports_language(&self, language: &str) -> bool {
        self.languages.iter().any(|l| l == language)
    }
}

/// HiFi-GAN vocoder.
///
/// High-fidelity generative adversarial network for mel-to-audio.
///
/// # Note
/// This is a stub - requires model weights.
#[derive(Debug)]
pub struct HifiGanVocoder {
    sample_rate: u32,
    n_mels: usize,
}

impl HifiGanVocoder {
    /// Create new HiFi-GAN vocoder
    #[must_use]
    pub fn new(sample_rate: u32, n_mels: usize) -> Self {
        Self {
            sample_rate,
            n_mels,
        }
    }

    /// Create with default settings (22050Hz, 80 mels)
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(22050, 80)
    }
}

impl Vocoder for HifiGanVocoder {
    fn vocalize(&self, mel: &[Vec<f32>]) -> SpeechResult<Vec<f32>> {
        if mel.is_empty() {
            return Err(SpeechError::InvalidConfig(
                "empty mel spectrogram".to_string(),
            ));
        }

        if mel.len() != self.n_mels {
            return Err(SpeechError::InvalidConfig(format!(
                "mel has {} channels, expected {}",
                mel.len(),
                self.n_mels
            )));
        }

        Err(SpeechError::ProcessingError(
            "HiFi-GAN requires model weights".to_string(),
        ))
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn n_mels(&self) -> usize {
        self.n_mels
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Normalize text for TTS (lowercase, expand abbreviations, etc.).
#[must_use]
pub fn normalize_text(text: &str) -> String {
    let mut normalized = text.to_string();

    // Basic normalization
    normalized = normalized.trim().to_string();

    // Expand common abbreviations
    normalized = normalized
        .replace("Mr.", "Mister")
        .replace("Mrs.", "Missus")
        .replace("Dr.", "Doctor")
        .replace("St.", "Street")
        .replace("vs.", "versus")
        .replace("etc.", "et cetera");

    // Expand numbers (basic)
    // A real implementation would use a full text normalization library

    normalized
}

/// Estimate synthesis duration from text.
///
/// Based on average speaking rate of ~150 words per minute.
#[must_use]
pub fn estimate_duration(text: &str, speaking_rate: f32) -> f32 {
    if text.is_empty() || speaking_rate <= 0.0 {
        return 0.0;
    }

    let word_count = text.split_whitespace().count();
    let base_wpm = 150.0; // Average speaking rate

    let adjusted_wpm = base_wpm * speaking_rate;
    if adjusted_wpm <= 0.0 {
        return 0.0;
    }

    // Duration in seconds
    word_count as f32 * 60.0 / adjusted_wpm
}

/// Split text into sentences for chunked synthesis.
#[must_use]
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Add remaining text
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
