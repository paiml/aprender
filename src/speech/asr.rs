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
            processing_time_ms: 0, // Would be measured in real impl
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
}
