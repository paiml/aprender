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

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_asr_session_model_and_config_accessors() {
    let model = MockAsrModel;
    let session = AsrSession::with_default_config(model).unwrap();

    // Test model() accessor
    assert_eq!(session.model().model_id(), "mock-model");

    // Test config() accessor
    assert_eq!(session.config().beam_size, 5);
    assert_eq!(session.config().temperature, 0.0);
}

#[test]
fn test_streaming_transcription_default() {
    // Test Default impl for StreamingTranscription
    let stream = StreamingTranscription::default();
    assert!(!stream.is_complete());
}

#[test]
fn test_cross_attention_peak_frame_zero_frames() {
    // Test peak_frame when n_frames == 0
    let weights = CrossAttentionWeights::zeros(2, 3, 0);
    assert_eq!(weights.peak_frame(0), None);
}

#[test]
fn test_cross_attention_peak_frame_out_of_bounds_token() {
    let weights = CrossAttentionWeights::zeros(2, 3, 10);
    // Token index out of bounds
    assert_eq!(weights.peak_frame(10), None);
}

#[test]
fn test_asr_session_with_custom_config() {
    let model = MockAsrModel;
    let config = AsrConfig::default()
        .with_language("es")
        .with_beam_size(10)
        .with_word_timestamps();

    let session = AsrSession::new(model, config).unwrap();

    // Verify custom config is preserved
    assert_eq!(session.config().language, Some("es".to_string()));
    assert_eq!(session.config().beam_size, 10);
    assert!(session.config().word_timestamps);
}

#[test]
fn test_asr_session_invalid_config() {
    let model = MockAsrModel;
    let mut config = AsrConfig::default();
    config.beam_size = 0; // Invalid

    let result = AsrSession::new(model, config);
    assert!(result.is_err());
}
