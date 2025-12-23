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
    /// Shape: [decoder_layers, decoder_tokens, encoder_frames]
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
    /// * `weights` - Flattened weights of shape [n_layers × n_tokens × n_frames]
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

    /// Get the shape as (n_layers, n_tokens, n_frames)
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
/// * `encoder_shape` - Shape as [batch, frames, hidden_dim]
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
    /// * `mel_shape` - Shape as [n_mels, n_frames]
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
mod tests {
    use super::*;

    // Test config validation
    #[test]
    fn test_asr_config_default() {
        let config = AsrConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.beam_size, 5);
        assert_eq!(config.temperature, 0.0);
        assert!(config.language.is_none());
    }

    #[test]
    fn test_asr_config_with_language() {
        let config = AsrConfig::default().with_language("en");
        assert_eq!(config.language, Some("en".to_string()));
    }

    #[test]
    fn test_asr_config_with_word_timestamps() {
        let config = AsrConfig::default().with_word_timestamps();
        assert!(config.word_timestamps);
    }

    #[test]
    fn test_asr_config_beam_size_min() {
        let config = AsrConfig::default().with_beam_size(0);
        assert_eq!(config.beam_size, 1, "beam_size should clamp to 1");
    }

    #[test]
    fn test_asr_config_validation_beam_size() {
        let mut config = AsrConfig::default();
        config.beam_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_asr_config_validation_temperature() {
        let mut config = AsrConfig::default();
        config.temperature = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_asr_config_validation_segment_length() {
        let mut config = AsrConfig::default();
        config.max_segment_length = 0.0;
        assert!(config.validate().is_err());
    }

    // Test Segment
    #[test]
    fn test_segment_new() {
        let seg = Segment::new("hello world", 0, 1000);
        assert_eq!(seg.text, "hello world");
        assert_eq!(seg.start_ms, 0);
        assert_eq!(seg.end_ms, 1000);
        assert_eq!(seg.confidence, 1.0);
    }

    #[test]
    fn test_segment_duration() {
        let seg = Segment::new("test", 500, 1500);
        assert_eq!(seg.duration_ms(), 1000);
        assert!((seg.duration_secs() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_segment_duration_saturating() {
        let seg = Segment::new("test", 1000, 500); // Invalid but shouldn't panic
        assert_eq!(seg.duration_ms(), 0);
    }

    // Test Transcription
    #[test]
    fn test_transcription_empty() {
        let t = Transcription::new();
        assert!(t.is_empty());
        assert_eq!(t.word_count(), 0);
        assert_eq!(t.duration_ms(), 0);
    }

    #[test]
    fn test_transcription_from_segments() {
        let segments = vec![
            Segment::new("Hello", 0, 500),
            Segment::new("world", 500, 1000),
        ];
        let t = Transcription::from_segments(segments);
        assert_eq!(t.text, "Hello world");
        assert_eq!(t.word_count(), 2);
        assert_eq!(t.duration_ms(), 1000);
    }

    #[test]
    fn test_transcription_word_count() {
        let t = Transcription {
            text: "one two three four five".to_string(),
            segments: vec![],
            language: None,
            processing_time_ms: 0,
            cross_attention_weights: None,
        };
        assert_eq!(t.word_count(), 5);
    }

    // Test StreamingTranscription
    #[test]
    fn test_streaming_transcription_iterator() {
        let mut stream = StreamingTranscription::new();
        stream.push(Segment::new("first", 0, 100));
        stream.push(Segment::new("second", 100, 200));

        let first = stream.next();
        assert!(first.is_some());
        assert_eq!(first.unwrap().text, "first");

        let second = stream.next();
        assert!(second.is_some());
        assert_eq!(second.unwrap().text, "second");

        assert!(stream.next().is_none());
    }

    #[test]
    fn test_streaming_transcription_complete() {
        let mut stream = StreamingTranscription::new();
        assert!(!stream.is_complete());
        stream.finish();
        assert!(stream.is_complete());
    }

    // Test mock model
    struct MockAsrModel;

    impl AsrModel for MockAsrModel {
        fn model_id(&self) -> &str {
            "mock-model"
        }

        fn supported_languages(&self) -> Option<&[&str]> {
            Some(&["en", "es", "fr"])
        }

        fn encode(&self, _mel: &[f32], _shape: &[usize]) -> SpeechResult<Vec<f32>> {
            Ok(vec![0.0; 256])
        }

        fn decode(&self, _encoder_output: &[f32], _config: &AsrConfig) -> SpeechResult<Vec<u32>> {
            Ok(vec![1, 2, 3, 4, 5])
        }

        fn tokens_to_text(&self, _tokens: &[u32]) -> SpeechResult<String> {
            Ok("hello world".to_string())
        }
    }

    #[test]
    fn test_asr_session_creation() {
        let model = MockAsrModel;
        let session = AsrSession::with_default_config(model);
        assert!(session.is_ok());
    }

    #[test]
    fn test_asr_session_transcribe() {
        let model = MockAsrModel;
        let session = AsrSession::with_default_config(model).unwrap();

        // 80 mels × 100 frames
        let mel = vec![0.0f32; 80 * 100];
        let result = session.transcribe(&mel, &[80, 100]);

        assert!(result.is_ok());
        let transcription = result.unwrap();
        assert_eq!(transcription.text, "hello world");
        assert!(!transcription.is_empty());
    }

    #[test]
    fn test_asr_session_invalid_shape() {
        let model = MockAsrModel;
        let session = AsrSession::with_default_config(model).unwrap();

        let mel = vec![0.0f32; 100];
        let result = session.transcribe(&mel, &[100]); // Wrong shape

        assert!(result.is_err());
    }

    #[test]
    fn test_asr_session_shape_mismatch() {
        let model = MockAsrModel;
        let session = AsrSession::with_default_config(model).unwrap();

        let mel = vec![0.0f32; 100];
        let result = session.transcribe(&mel, &[80, 100]); // Mismatch

        assert!(result.is_err());
    }

    #[test]
    fn test_asr_model_trait() {
        let model = MockAsrModel;
        assert_eq!(model.model_id(), "mock-model");
        assert!(model.supported_languages().is_some());
    }

    // ========================================================================
    // G5: Cross-Attention Weights Tests
    // ========================================================================

    #[test]
    fn test_cross_attention_weights_zeros() {
        let weights = CrossAttentionWeights::zeros(6, 10, 100);
        assert_eq!(weights.shape(), (6, 10, 100));
        assert_eq!(weights.as_slice().len(), 6 * 10 * 100);
    }

    #[test]
    fn test_cross_attention_weights_new_valid() {
        let data = vec![0.1f32; 6 * 10 * 100];
        let weights = CrossAttentionWeights::new(data, 6, 10, 100);
        assert!(weights.is_ok());
    }

    #[test]
    fn test_cross_attention_weights_new_invalid_size() {
        let data = vec![0.1f32; 100]; // Wrong size
        let weights = CrossAttentionWeights::new(data, 6, 10, 100);
        assert!(weights.is_err());
    }

    #[test]
    fn test_cross_attention_get_attention() {
        let mut data = vec![0.0f32; 2 * 3 * 4]; // 2 layers, 3 tokens, 4 frames
                                                // Set specific values
        data[0..4].copy_from_slice(&[0.1, 0.2, 0.3, 0.4]); // layer 0, token 0
        data[4..8].copy_from_slice(&[0.5, 0.6, 0.7, 0.8]); // layer 0, token 1

        let weights = CrossAttentionWeights::new(data, 2, 3, 4).unwrap();

        let attn = weights.get_attention(0, 0);
        assert!(attn.is_some());
        assert_eq!(attn.unwrap(), &[0.1, 0.2, 0.3, 0.4]);

        let attn = weights.get_attention(0, 1);
        assert!(attn.is_some());
        assert_eq!(attn.unwrap(), &[0.5, 0.6, 0.7, 0.8]);

        // Out of bounds
        assert!(weights.get_attention(10, 0).is_none());
        assert!(weights.get_attention(0, 10).is_none());
    }

    #[test]
    fn test_cross_attention_peak_frame() {
        let mut data = vec![0.0f32; 2 * 1 * 5]; // 2 layers, 1 token, 5 frames
                                                // Layer 0: peak at frame 2
        data[0..5].copy_from_slice(&[0.1, 0.2, 0.9, 0.1, 0.1]);
        // Layer 1: peak at frame 2
        data[5..10].copy_from_slice(&[0.1, 0.1, 0.8, 0.2, 0.1]);

        let weights = CrossAttentionWeights::new(data, 2, 1, 5).unwrap();

        let peak = weights.peak_frame(0);
        assert_eq!(peak, Some(2)); // Both layers peak at frame 2
    }

    #[test]
    fn test_cross_attention_entropy() {
        // Create uniform distribution (high entropy)
        let mut data = vec![0.0f32; 1 * 1 * 4]; // 1 layer, 1 token, 4 frames
        data[0..4].copy_from_slice(&[0.25, 0.25, 0.25, 0.25]);

        let weights = CrossAttentionWeights::new(data, 1, 1, 4).unwrap();

        let entropy = weights.attention_entropy(0, 0);
        assert!(entropy.is_some());
        // Uniform distribution of 4 elements has entropy = ln(4) ≈ 1.386
        assert!(entropy.unwrap() > 1.0);
    }

    #[test]
    fn test_cross_attention_is_healthy() {
        // Well-distributed weights (should be healthy)
        let data: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0).collect();
        let weights = CrossAttentionWeights::new(data, 1, 10, 10).unwrap();
        assert!(weights.is_healthy());

        // Collapsed weights (all same value, std near 0)
        let collapsed = vec![0.5f32; 100];
        let collapsed_weights = CrossAttentionWeights::new(collapsed, 1, 10, 10).unwrap();
        assert!(!collapsed_weights.is_healthy());
    }

    #[test]
    fn test_cross_attention_empty_healthy() {
        let weights = CrossAttentionWeights::zeros(0, 0, 0);
        assert!(weights.is_healthy()); // Empty is considered healthy
    }

    // ========================================================================
    // G2: Language Detection Tests
    // ========================================================================

    #[test]
    fn test_language_detection_new() {
        let detection = LanguageDetection::new("en", 0.95);
        assert_eq!(detection.language(), "en");
        assert!((detection.confidence() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_language_detection_confidence_clamped() {
        let detection = LanguageDetection::new("en", 1.5); // Above 1.0
        assert!((detection.confidence() - 1.0).abs() < 0.001);

        let detection = LanguageDetection::new("en", -0.5); // Below 0.0
        assert!((detection.confidence() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_language_detection_with_alternatives() {
        let detection = LanguageDetection::new("en", 0.85)
            .with_alternative("de", 0.08)
            .with_alternative("fr", 0.05);

        assert_eq!(detection.alternatives().len(), 2);
        assert_eq!(detection.alternatives()[0].0, "de");
    }

    #[test]
    fn test_language_detection_is_confident() {
        let detection = LanguageDetection::new("en", 0.85);
        assert!(detection.is_confident(0.8));
        assert!(!detection.is_confident(0.9));
    }

    #[test]
    fn test_language_detection_top_languages() {
        let detection = LanguageDetection::new("en", 0.80)
            .with_alternative("de", 0.10)
            .with_alternative("fr", 0.05)
            .with_alternative("es", 0.03);

        let top2 = detection.top_languages(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, "en");
        assert_eq!(top2[1].0, "de");
    }

    #[test]
    fn test_language_detection_default() {
        let detection = LanguageDetection::default();
        assert_eq!(detection.language(), "en");
        assert!((detection.confidence() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_detect_language_valid() {
        let encoder_output = vec![0.0f32; 1 * 100 * 512]; // [1, 100, 512]
        let result = detect_language(&encoder_output, &[1, 100, 512]);
        assert!(result.is_ok());

        let detection = result.unwrap();
        assert!(!detection.language().is_empty());
        assert!(detection.confidence() > 0.0);
    }

    #[test]
    fn test_detect_language_invalid_shape() {
        let encoder_output = vec![0.0f32; 1000];
        let result = detect_language(&encoder_output, &[1000]); // 1D, should be 3D
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_language_size_mismatch() {
        let encoder_output = vec![0.0f32; 100];
        let result = detect_language(&encoder_output, &[1, 100, 512]); // Mismatch
        assert!(result.is_err());
    }

    #[test]
    fn test_is_language_supported() {
        assert!(is_language_supported("en"));
        assert!(is_language_supported("de"));
        assert!(is_language_supported("ja"));
        assert!(!is_language_supported("xyz"));
        assert!(!is_language_supported(""));
    }

    #[test]
    fn test_supported_languages_count() {
        // Whisper supports 99 languages
        assert_eq!(SUPPORTED_LANGUAGES.len(), 99);
    }

    #[test]
    fn test_transcription_with_cross_attention() {
        let weights = CrossAttentionWeights::zeros(6, 10, 100);
        let mut t = Transcription::new();
        t.cross_attention_weights = Some(weights);

        assert!(t.cross_attention_weights.is_some());
        let w = t.cross_attention_weights.as_ref().unwrap();
        assert_eq!(w.shape(), (6, 10, 100));
    }
}
