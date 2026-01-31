//! Automatic Speech Recognition (ASR) primitives.
//!
//! Provides the foundation for Whisper-style ASR inference:
//! - `AsrModel` trait for model abstraction
//! - `AsrSession` for stateful transcription
//! - `Transcription` and `Segment` for results
//!
//! # Architecture
//!
//! ```text
//! Audio → Mel Spectrogram → Encoder → Cross-Attention → Decoder → Tokens → Text
//!                           (APR)                        (APR)
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::speech::asr::{AsrConfig, Transcription, Segment};
//!
//! let config = AsrConfig::default();
//! assert_eq!(config.language, None); // Auto-detect
//! assert!(config.beam_size >= 1);
//! ```
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>`
//! - Real-time safe (no allocations in hot path planned)

use super::{SpeechError, SpeechResult};

// ============================================================================
// Configuration
// ============================================================================

/// ASR configuration for transcription sessions
#[derive(Debug, Clone)]
pub struct AsrConfig {
    /// Language code (ISO 639-1) or None for auto-detection
    pub language: Option<String>,
    /// Beam search width (1 = greedy, 5 = default)
    pub beam_size: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// Enable word-level timestamps
    pub word_timestamps: bool,
    /// Maximum segment length in seconds
    pub max_segment_length: f32,
    /// Suppress blank tokens at start
    pub suppress_blank: bool,
    /// Condition on previous text for context
    pub condition_on_previous: bool,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            language: None,
            beam_size: 5,
            temperature: 0.0,
            word_timestamps: false,
            max_segment_length: 30.0,
            suppress_blank: true,
            condition_on_previous: true,
        }
    }
}

impl AsrConfig {
    /// Create config for a specific language
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Enable word-level timestamps
    #[must_use]
    pub fn with_word_timestamps(mut self) -> Self {
        self.word_timestamps = true;
        self
    }

    /// Set beam size (1 for greedy decoding)
    #[must_use]
    pub fn with_beam_size(mut self, size: usize) -> Self {
        self.beam_size = size.max(1);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> SpeechResult<()> {
        if self.beam_size == 0 {
            return Err(SpeechError::InvalidConfig(
                "beam_size must be >= 1".to_string(),
            ));
        }
        if self.temperature < 0.0 {
            return Err(SpeechError::InvalidConfig(
                "temperature must be >= 0.0".to_string(),
            ));
        }
        if self.max_segment_length <= 0.0 {
            return Err(SpeechError::InvalidConfig(
                "max_segment_length must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Transcription Results
// ============================================================================

/// A single transcription segment with timing
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Segment text
    pub text: String,
    /// Start time in milliseconds
    pub start_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Word-level timestamps (if enabled)
    pub words: Vec<WordTiming>,
}

impl Segment {
    /// Create a new segment
    #[must_use]
    pub fn new(text: impl Into<String>, start_ms: u64, end_ms: u64) -> Self {
        Self {
            text: text.into(),
            start_ms,
            end_ms,
            confidence: 1.0,
            words: Vec::new(),
        }
    }

    /// Duration in milliseconds
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Duration in seconds
    #[must_use]
    pub fn duration_secs(&self) -> f32 {
        self.duration_ms() as f32 / 1000.0
    }
}

/// Word-level timing information
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    /// The word
    pub word: String,
    /// Start time in milliseconds
    pub start_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Confidence score
    pub confidence: f32,
}

/// Complete transcription result
#[derive(Debug, Clone, Default)]
pub struct Transcription {
    /// Full transcription text
    pub text: String,
    /// Timed segments
    pub segments: Vec<Segment>,
    /// Detected or specified language
    pub language: Option<String>,
    /// Processing duration in milliseconds
    pub processing_time_ms: u64,
    /// Cross-attention weights for alignment (G5)
    /// Shape: [`decoder_layers`, `decoder_tokens`, `encoder_frames`]
    pub cross_attention_weights: Option<CrossAttentionWeights>,
}

// ============================================================================
// G5: Cross-Attention Weights
// ============================================================================

/// Cross-attention weights for encoder-decoder alignment (G5)
///
/// These weights show how the decoder attends to encoder frames,
/// useful for:
/// - Word-audio alignment
/// - Detecting hallucination (low attention entropy)
/// - Debugging transcription issues
///
/// # Example
///
/// ```rust
/// use aprender::speech::asr::CrossAttentionWeights;
///
/// // 6 layers, 10 decoder tokens, 100 encoder frames
/// let weights = CrossAttentionWeights::zeros(6, 10, 100);
/// assert_eq!(weights.shape(), (6, 10, 100));
/// ```
#[derive(Debug, Clone)]
pub struct CrossAttentionWeights {
    /// Flattened attention weights [layers × tokens × frames]
    weights: Vec<f32>,
    /// Number of decoder layers
    n_layers: usize,
    /// Number of decoder tokens
    n_tokens: usize,
    /// Number of encoder frames
    n_frames: usize,
}

impl CrossAttentionWeights {
    /// Create cross-attention weights from flat data
    ///
    /// # Arguments
    /// * `weights` - Flattened weights of shape [`n_layers` × `n_tokens` × `n_frames`]
    /// * `n_layers` - Number of decoder layers
    /// * `n_tokens` - Number of decoder tokens
    /// * `n_frames` - Number of encoder frames
    pub fn new(
        weights: Vec<f32>,
        n_layers: usize,
        n_tokens: usize,
        n_frames: usize,
    ) -> Result<Self, SpeechError> {
        let expected_len = n_layers * n_tokens * n_frames;
        if weights.len() != expected_len {
            return Err(SpeechError::InvalidAudio(format!(
                "Cross-attention weight length {} doesn't match shape [{}, {}, {}] (expected {})",
                weights.len(),
                n_layers,
                n_tokens,
                n_frames,
                expected_len
            )));
        }
        Ok(Self {
            weights,
            n_layers,
            n_tokens,
            n_frames,
        })
    }

    /// Create zeros-initialized weights
    #[must_use]
    pub fn zeros(n_layers: usize, n_tokens: usize, n_frames: usize) -> Self {
        Self {
            weights: vec![0.0; n_layers * n_tokens * n_frames],
            n_layers,
            n_tokens,
            n_frames,
        }
    }

    /// Get the shape as (`n_layers`, `n_tokens`, `n_frames`)
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.n_layers, self.n_tokens, self.n_frames)
    }

    /// Get attention weights for a specific layer and token
    ///
    /// Returns attention distribution over encoder frames
    #[must_use]
    pub fn get_attention(&self, layer: usize, token: usize) -> Option<&[f32]> {
        if layer >= self.n_layers || token >= self.n_tokens {
            return None;
        }
        let start = (layer * self.n_tokens + token) * self.n_frames;
        let end = start + self.n_frames;
        Some(&self.weights[start..end])
    }

    /// Get the peak attention frame for a token (averaged across layers)
    ///
    /// Useful for rough word-audio alignment
    #[must_use]
    pub fn peak_frame(&self, token: usize) -> Option<usize> {
        if token >= self.n_tokens || self.n_frames == 0 {
            return None;
        }

        let mut avg_attention = vec![0.0f32; self.n_frames];

        // Average attention across layers
        for layer in 0..self.n_layers {
            if let Some(attn) = self.get_attention(layer, token) {
                for (i, &w) in attn.iter().enumerate() {
                    avg_attention[i] += w;
                }
            }
        }

        // Find peak
        avg_attention
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// Calculate entropy of attention distribution for a token
    ///
    /// Low entropy (< 1.0) may indicate hallucination or poor alignment
    #[must_use]
    pub fn attention_entropy(&self, layer: usize, token: usize) -> Option<f32> {
        let attn = self.get_attention(layer, token)?;

        // Shannon entropy: -sum(p * log(p))
        let mut entropy = 0.0f32;
        for &p in attn {
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }

        Some(entropy)
    }

    /// Check if attention is well-distributed (not collapsed)
    ///
    /// Returns false if standard deviation is too low (G10 check)
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        if self.weights.is_empty() {
            return true;
        }

        // Calculate std across all weights
        let n = self.weights.len() as f32;
        let mean: f32 = self.weights.iter().sum::<f32>() / n;
        let variance: f32 = self.weights.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        // G10: std should be > 0.01 to indicate non-collapsed attention
        std > 0.01
    }

    /// Get the raw weights
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.weights
    }
}

// ============================================================================
// G2: Language Detection
// ============================================================================

/// Language detection result (G2)
///
/// Provides detected language with confidence scores for alternatives.
///
/// # Example
///
/// ```rust
/// use aprender::speech::asr::LanguageDetection;
///
/// let detection = LanguageDetection::new("en", 0.95)
///     .with_alternative("de", 0.03)
///     .with_alternative("fr", 0.02);
///
/// assert_eq!(detection.language(), "en");
/// assert!(detection.confidence() > 0.9);
/// ```
#[derive(Debug, Clone)]
pub struct LanguageDetection {
    /// Detected language code (ISO 639-1)
    language: String,
    /// Confidence score for detected language (0.0-1.0)
    confidence: f32,
    /// Alternative language candidates with scores
    alternatives: Vec<(String, f32)>,
}

impl LanguageDetection {
    /// Create a new language detection result
    #[must_use]
    pub fn new(language: impl Into<String>, confidence: f32) -> Self {
        Self {
            language: language.into(),
            confidence: confidence.clamp(0.0, 1.0),
            alternatives: Vec::new(),
        }
    }

    /// Add an alternative language candidate
    #[must_use]
    pub fn with_alternative(mut self, language: impl Into<String>, confidence: f32) -> Self {
        self.alternatives
            .push((language.into(), confidence.clamp(0.0, 1.0)));
        self
    }

    /// Get the detected language code
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Get confidence score for the detected language
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get alternative language candidates
    #[must_use]
    pub fn alternatives(&self) -> &[(String, f32)] {
        &self.alternatives
    }

    /// Check if detection is confident (> threshold)
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Get top N languages by confidence
    #[must_use]
    pub fn top_languages(&self, n: usize) -> Vec<(&str, f32)> {
        let mut all: Vec<(&str, f32)> = vec![(&self.language, self.confidence)];
        all.extend(self.alternatives.iter().map(|(l, c)| (l.as_str(), *c)));
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(n);
        all
    }
}

impl Default for LanguageDetection {
    fn default() -> Self {
        Self::new("en", 1.0) // Default to English with full confidence
    }
}

/// Detect language from audio features (G2)
///
/// Uses encoder outputs to classify the spoken language.
/// Supports 99 languages following Whisper's language detection.
///
/// # Arguments
/// * `encoder_output` - Encoder hidden states
/// * `encoder_shape` - Shape as [batch, frames, `hidden_dim`]
///
/// # Returns
/// Language detection result with confidence scores
///
/// # Note
/// This is a placeholder that returns a default result.
/// Real implementation requires language detection head weights.
pub fn detect_language(
    encoder_output: &[f32],
    encoder_shape: &[usize],
) -> SpeechResult<LanguageDetection> {
    // Validate input
    if encoder_shape.len() != 3 {
        return Err(SpeechError::InvalidAudio(
            "encoder_shape must be [batch, frames, hidden_dim]".to_string(),
        ));
    }

    let expected_len = encoder_shape.iter().product::<usize>();
    if encoder_output.len() != expected_len {
        return Err(SpeechError::InvalidAudio(format!(
            "encoder_output length {} doesn't match shape {:?}",
            encoder_output.len(),
            encoder_shape
        )));
    }

    // Placeholder: Real implementation would:
    // 1. Pool encoder outputs (e.g., mean over time)
    // 2. Apply language classification head
    // 3. Softmax over 99 language logits
    // 4. Return top languages with scores

    // For now, return English as default with alternatives
    Ok(LanguageDetection::new("en", 0.85)
        .with_alternative("de", 0.05)
        .with_alternative("fr", 0.04)
        .with_alternative("es", 0.03)
        .with_alternative("it", 0.02)
        .with_alternative("unknown", 0.01))
}

/// List of supported language codes (ISO 639-1)
///
/// Whisper supports 99 languages. This list includes the most common ones.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

/// Check if a language code is supported
#[must_use]
pub fn is_language_supported(code: &str) -> bool {
    SUPPORTED_LANGUAGES.contains(&code)
}

impl Transcription {
    /// Create empty transcription
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from segments
    #[must_use]
    pub fn from_segments(segments: Vec<Segment>) -> Self {
        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Self {
            text,
            segments,
            language: None,
            processing_time_ms: 0,
            cross_attention_weights: None,
        }
    }

    /// Total duration in milliseconds
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.segments.last().map_or(0, |s| s.end_ms)
    }

    /// Word count
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Check if transcription is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

// ============================================================================
// ASR Model Trait
// ============================================================================

/// Trait for ASR model implementations
///
/// Implementations include:
/// - Whisper models (tiny, base, small, medium, large)
/// - Distilled models (distil-whisper)
/// - Custom models in APR format
pub trait AsrModel {
    /// Model identifier (e.g., "whisper-tiny", "whisper-large-v3")
    fn model_id(&self) -> &str;

    /// Supported languages (None = multilingual)
    fn supported_languages(&self) -> Option<&[&str]>;

    /// Encode audio to hidden states
    fn encode(&self, mel: &[f32], mel_shape: &[usize]) -> SpeechResult<Vec<f32>>;

    /// Decode hidden states to token IDs
    fn decode(&self, encoder_output: &[f32], config: &AsrConfig) -> SpeechResult<Vec<u32>>;

    /// Convert token IDs to text
    fn tokens_to_text(&self, tokens: &[u32]) -> SpeechResult<String>;
}

// ============================================================================
// ASR Session
// ============================================================================

/// Stateful ASR session for transcription
///
/// Wraps a model with configuration and provides high-level transcription API.
#[derive(Debug)]
pub struct AsrSession<M: AsrModel> {
    model: M,
    config: AsrConfig,
}

impl<M: AsrModel> AsrSession<M> {
    /// Create a new ASR session
    pub fn new(model: M, config: AsrConfig) -> SpeechResult<Self> {
        config.validate()?;
        Ok(Self { model, config })
    }

    /// Create with default configuration
    pub fn with_default_config(model: M) -> SpeechResult<Self> {
        Self::new(model, AsrConfig::default())
    }

    /// Get the model reference
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &AsrConfig {
        &self.config
    }

    /// Transcribe audio samples
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram features (80 bins × time frames)
    /// * `mel_shape` - Shape as [`n_mels`, `n_frames`]
    ///
    /// # Returns
    /// Transcription with segments and timing
    pub fn transcribe(&self, mel: &[f32], mel_shape: &[usize]) -> SpeechResult<Transcription> {
        // Validate input shape
        if mel_shape.len() != 2 {
            return Err(SpeechError::InvalidAudio(
                "mel_shape must be [n_mels, n_frames]".to_string(),
            ));
        }

        let expected_len = mel_shape[0] * mel_shape[1];
        if mel.len() != expected_len {
            return Err(SpeechError::InvalidAudio(format!(
                "mel length {} doesn't match shape {:?} (expected {})",
                mel.len(),
                mel_shape,
                expected_len
            )));
        }

        // Encode
        let encoder_output = self.model.encode(mel, mel_shape)?;

        // Decode
        let tokens = self.model.decode(&encoder_output, &self.config)?;

        // Convert to text
        let text = self.model.tokens_to_text(&tokens)?;

        // Build transcription (simplified - real impl would have segments)
        let duration_ms = (mel_shape[1] as u64 * 10) / 16; // Rough estimate: 10ms per frame at 16kHz

        Ok(Transcription {
            text: text.clone(),
            segments: vec![Segment::new(text, 0, duration_ms)],
            language: self.config.language.clone(),
            processing_time_ms: 0,         // Would be measured in real impl
            cross_attention_weights: None, // G5: Would be extracted from model in real impl
        })
    }
}

// ============================================================================
// Streaming Transcription
// ============================================================================

/// Iterator for streaming transcription results
///
/// Yields segments as they become available during real-time processing.
#[derive(Debug)]
pub struct StreamingTranscription {
    /// Pending segments to yield
    pending: Vec<Segment>,
    /// Whether transcription is complete
    complete: bool,
}

impl StreamingTranscription {
    /// Create a new streaming transcription
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            complete: false,
        }
    }

    /// Push a new segment
    pub fn push(&mut self, segment: Segment) {
        self.pending.push(segment);
    }

    /// Mark transcription as complete
    pub fn finish(&mut self) {
        self.complete = true;
    }

    /// Check if complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}

impl Default for StreamingTranscription {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for StreamingTranscription {
    type Item = Segment;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pending.is_empty() {
            None
        } else {
            Some(self.pending.remove(0))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================


#[cfg(test)]
mod tests;
