
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
