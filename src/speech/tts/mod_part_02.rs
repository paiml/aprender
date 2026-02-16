use crate::speech::{SpeechError, SpeechResult};
use super::{SpeechSynthesizer, SynthesisRequest, SynthesisResult, TtsConfig, Vocoder};

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

