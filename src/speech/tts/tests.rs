pub(crate) use super::*;

#[test]
fn test_tts_config_default() {
    let config = TtsConfig::default();
    assert_eq!(config.sample_rate, 22050);
    assert_eq!(config.n_mels, 80);
    assert!((config.speaking_rate - 1.0).abs() < f32::EPSILON);
    assert!(config.validate().is_ok());
}

#[test]
fn test_tts_config_high_quality() {
    let config = TtsConfig::high_quality();
    assert_eq!(config.sample_rate, 48000);
    assert_eq!(config.n_mels, 128);
    assert!(config.validate().is_ok());
}

#[test]
fn test_tts_config_fast() {
    let config = TtsConfig::fast();
    assert_eq!(config.sample_rate, 16000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_tts_config_validation() {
    let mut config = TtsConfig::default();

    config.sample_rate = 0;
    assert!(config.validate().is_err());

    config.sample_rate = 22050;
    config.speaking_rate = 0.0;
    assert!(config.validate().is_err());

    config.speaking_rate = 1.0;
    config.pitch_shift = 30.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_tts_config_frames_per_second() {
    let config = TtsConfig::default();
    let fps = config.frames_per_second();
    assert!((fps - 86.13).abs() < 1.0); // 22050 / 256
}

#[test]
fn test_synthesis_request_new() {
    let request = SynthesisRequest::new("Hello world".to_string());
    assert_eq!(request.text, "Hello world");
    assert!(request.speaker_id.is_none());
}

#[test]
fn test_synthesis_request_builder() {
    let request = SynthesisRequest::new("Hello".to_string())
        .with_speaker("alice".to_string())
        .with_speaking_rate(1.2)
        .with_pitch_shift(-2.0)
        .with_energy_scale(1.1)
        .with_language("en".to_string());

    assert_eq!(request.speaker_id, Some("alice".to_string()));
    assert!((request.speaking_rate.unwrap() - 1.2).abs() < f32::EPSILON);
    assert!((request.pitch_shift.unwrap() - (-2.0)).abs() < f32::EPSILON);
    assert!((request.energy_scale.unwrap() - 1.1).abs() < f32::EPSILON);
    assert_eq!(request.language, Some("en".to_string()));
}

#[test]
fn test_synthesis_request_validate() {
    let config = TtsConfig::default();

    let request = SynthesisRequest::new("Hello".to_string());
    assert!(request.validate(&config).is_ok());

    let empty = SynthesisRequest::new(String::new());
    assert!(empty.validate(&config).is_err());

    let invalid_rate = SynthesisRequest::new("Hello".to_string()).with_speaking_rate(10.0);
    assert!(invalid_rate.validate(&config).is_err());
}

#[test]
fn test_synthesis_request_validate_too_long() {
    let config = TtsConfig {
        max_text_length: 10,
        ..TtsConfig::default()
    };
    let request = SynthesisRequest::new("This is a very long text".to_string());
    assert!(request.validate(&config).is_err());
}

#[test]
fn test_synthesis_result_new() {
    let audio = vec![0.0_f32; 22050];
    let result = SynthesisResult::new(audio, 22050);

    assert_eq!(result.num_samples(), 22050);
    assert!((result.duration - 1.0).abs() < 1e-6);
    assert!(!result.has_mel());
}

#[test]
fn test_synthesis_result_with_extras() {
    let mut result = SynthesisResult::new(vec![0.0_f32; 100], 22050);

    result.with_mel(vec![vec![0.0; 10]; 80]);
    assert!(result.has_mel());

    result.with_alignment(vec![AlignmentInfo::new("h".to_string(), 0.0, 0.1)]);
    assert!(result.alignment.is_some());

    result.with_phonemes(vec!["HH".to_string(), "AH".to_string()]);
    assert!(result.phonemes.is_some());
}

#[test]
fn test_alignment_info() {
    let align = AlignmentInfo::new("hello".to_string(), 0.0, 0.5);
    assert_eq!(align.token, "hello");
    assert!((align.duration() - 0.5).abs() < f32::EPSILON);
    assert!((align.confidence - 1.0).abs() < f32::EPSILON);

    let with_conf = align.with_confidence(0.8);
    assert!((with_conf.confidence - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_alignment_info_clamp_confidence() {
    let align = AlignmentInfo::new("x".to_string(), 0.0, 0.1).with_confidence(1.5);
    assert!((align.confidence - 1.0).abs() < f32::EPSILON);

    let align2 = AlignmentInfo::new("y".to_string(), 0.0, 0.1).with_confidence(-0.5);
    assert!((align2.confidence - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_fastspeech2_synthesizer() {
    let synth = FastSpeech2Synthesizer::default_config();
    assert!(synth.supports_language("en"));
    assert!(!synth.supports_language("es"));

    let speakers = synth.available_speakers();
    assert!(speakers.contains(&"default".to_string()));

    let request = SynthesisRequest::new("Hello".to_string());
    assert!(synth.synthesize(&request).is_err());
}

#[test]
fn test_fastspeech2_unknown_speaker() {
    let synth = FastSpeech2Synthesizer::default_config();
    let request = SynthesisRequest::new("Hello".to_string()).with_speaker("unknown".to_string());
    let result = synth.synthesize(&request);
    assert!(matches!(result, Err(SpeechError::InvalidConfig(_))));
}

#[test]
fn test_vits_synthesizer() {
    let mut synth = VitsSynthesizer::default_config();
    synth.add_speaker("alice".to_string());
    synth.add_language("es".to_string());

    assert!(synth.supports_language("en"));
    assert!(synth.supports_language("es"));
    assert!(!synth.supports_language("fr"));

    let speakers = synth.available_speakers();
    assert!(speakers.contains(&"alice".to_string()));
}

#[test]
fn test_vits_unsupported_language() {
    let synth = VitsSynthesizer::default_config();
    let request = SynthesisRequest::new("Hello".to_string()).with_language("fr".to_string());
    let result = synth.synthesize(&request);
    assert!(matches!(result, Err(SpeechError::InvalidConfig(_))));
}

#[test]
fn test_hifigan_vocoder() {
    let vocoder = HifiGanVocoder::default_config();
    assert_eq!(vocoder.sample_rate(), 22050);
    assert_eq!(vocoder.n_mels(), 80);

    // Empty mel
    assert!(vocoder.vocalize(&[]).is_err());

    // Wrong number of channels
    let bad_mel: Vec<Vec<f32>> = vec![vec![0.0; 10]; 40];
    assert!(vocoder.vocalize(&bad_mel).is_err());

    // Correct dimensions but not implemented
    let mel: Vec<Vec<f32>> = vec![vec![0.0; 10]; 80];
    assert!(vocoder.vocalize(&mel).is_err());
}

#[test]
fn test_normalize_text() {
    assert_eq!(normalize_text("Mr. Smith"), "Mister Smith");
    assert_eq!(normalize_text("Dr. Jones"), "Doctor Jones");
    assert_eq!(normalize_text("  hello  "), "hello");
    assert_eq!(normalize_text("A vs. B"), "A versus B");
}

#[test]
fn test_estimate_duration() {
    // Empty text
    assert_eq!(estimate_duration("", 1.0), 0.0);

    // Zero rate
    assert_eq!(estimate_duration("Hello", 0.0), 0.0);

    // Normal text (10 words at 150 wpm = 4 seconds)
    let text = "This is a test sentence with exactly ten words here";
    let duration = estimate_duration(text, 1.0);
    assert!((duration - 4.0).abs() < 0.1);

    // Faster rate (2x = half duration)
    let fast_duration = estimate_duration(text, 2.0);
    assert!((fast_duration - 2.0).abs() < 0.1);
}

#[test]
fn test_split_sentences() {
    let text = "Hello world. How are you? I'm fine!";
    let sentences = split_sentences(text);
    assert_eq!(sentences.len(), 3);
    assert_eq!(sentences[0], "Hello world.");
    assert_eq!(sentences[1], "How are you?");
    assert_eq!(sentences[2], "I'm fine!");
}

#[test]
fn test_split_sentences_no_punctuation() {
    let text = "Hello world";
    let sentences = split_sentences(text);
    assert_eq!(sentences.len(), 1);
    assert_eq!(sentences[0], "Hello world");
}

#[test]
fn test_split_sentences_empty() {
    let sentences = split_sentences("");
    assert!(sentences.is_empty());

    let sentences = split_sentences("   ");
    assert!(sentences.is_empty());
}

// =========================================================================
// Extended coverage tests
// =========================================================================

#[test]
fn test_tts_config_validate_n_mels() {
    let mut config = TtsConfig::default();
    config.n_mels = 0;
    assert!(config.validate().is_err());
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("n_mels"));
}

#[test]
fn test_tts_config_validate_hop_size() {
    let mut config = TtsConfig::default();
    config.hop_size = 0;
    assert!(config.validate().is_err());
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("hop_size"));
}

#[test]
fn test_tts_config_validate_win_size_less_than_hop() {
    let mut config = TtsConfig::default();
    config.win_size = 100;
    config.hop_size = 200;
    assert!(config.validate().is_err());
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("win_size"));
}

#[test]
fn test_tts_config_validate_win_size_zero() {
    let mut config = TtsConfig::default();
    config.win_size = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_tts_config_validate_energy_scale_zero() {
    let mut config = TtsConfig::default();
    config.energy_scale = 0.0;
    assert!(config.validate().is_err());
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("energy_scale"));
}

#[test]
fn test_tts_config_validate_energy_scale_too_high() {
    let mut config = TtsConfig::default();
    config.energy_scale = 5.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_tts_config_validate_max_text_length() {
    let mut config = TtsConfig::default();
    config.max_text_length = 0;
    assert!(config.validate().is_err());
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("max_text_length"));
}

#[test]
fn test_tts_config_validate_max_output_duration() {
    let mut config = TtsConfig::default();
    config.max_output_duration = 0.0;
    assert!(config.validate().is_err());
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("max_output_duration"));
}

#[test]
fn test_tts_config_max_output_samples() {
    let config = TtsConfig::default();
    let max_samples = config.max_output_samples();
    // 30 seconds * 22050 Hz = 661500 samples
    assert_eq!(max_samples, 661500);
}

#[test]
fn test_synthesis_request_validate_pitch_shift_out_of_range() {
    let config = TtsConfig::default();
    let request = SynthesisRequest::new("Hello".to_string()).with_pitch_shift(30.0);
    assert!(request.validate(&config).is_err());
    let err = request.validate(&config).unwrap_err().to_string();
    assert!(err.contains("pitch_shift"));
}

#[test]
fn test_synthesis_request_validate_energy_scale_out_of_range() {
    let config = TtsConfig::default();
    let request = SynthesisRequest::new("Hello".to_string()).with_energy_scale(5.0);
    assert!(request.validate(&config).is_err());
    let err = request.validate(&config).unwrap_err().to_string();
    assert!(err.contains("energy_scale"));
}

#[test]
fn test_synthesis_request_validate_energy_scale_zero() {
    let config = TtsConfig::default();
    let request = SynthesisRequest::new("Hello".to_string()).with_energy_scale(0.0);
    assert!(request.validate(&config).is_err());
}

#[test]
fn test_synthesis_result_zero_sample_rate() {
    let audio = vec![0.0_f32; 100];
    let result = SynthesisResult::new(audio, 0);
    assert_eq!(result.duration, 0.0);
}

#[test]
fn test_fastspeech2_add_speaker() {
    let mut synth = FastSpeech2Synthesizer::default_config();
    synth.add_speaker("alice".to_string());
    synth.add_speaker("bob".to_string());

    // Adding duplicate should not create duplicate
    synth.add_speaker("alice".to_string());

    let speakers = synth.available_speakers();
    assert!(speakers.contains(&"alice".to_string()));
    assert!(speakers.contains(&"bob".to_string()));
    assert_eq!(speakers.iter().filter(|&s| s == "alice").count(), 1);
}

#[test]
fn test_fastspeech2_new_custom_config() {
    let config = TtsConfig::high_quality();
    let synth = FastSpeech2Synthesizer::new(config);
    assert_eq!(synth.config().sample_rate, 48000);
}

#[test]
fn test_vits_unknown_speaker() {
    let synth = VitsSynthesizer::default_config();
    let request = SynthesisRequest::new("Hello".to_string()).with_speaker("unknown".to_string());
    let result = synth.synthesize(&request);
    assert!(matches!(result, Err(SpeechError::InvalidConfig(_))));
}

#[test]
fn test_vits_new_custom_config() {
    let config = TtsConfig::fast();
    let synth = VitsSynthesizer::new(config);
    assert_eq!(synth.config().sample_rate, 16000);
}

#[test]
fn test_hifigan_new() {
    let vocoder = HifiGanVocoder::new(44100, 128);
    assert_eq!(vocoder.sample_rate(), 44100);
    assert_eq!(vocoder.n_mels(), 128);
}

#[test]
fn test_tts_config_validate_speaking_rate_too_high() {
    let mut config = TtsConfig::default();
    config.speaking_rate = 6.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_tts_config_validate_pitch_shift_too_low() {
    let mut config = TtsConfig::default();
    config.pitch_shift = -30.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_synthesis_request_debug() {
    let request = SynthesisRequest::new("Hello".to_string());
    let debug_str = format!("{:?}", request);
    assert!(debug_str.contains("SynthesisRequest"));
}

#[test]
fn test_synthesis_result_debug() {
    let result = SynthesisResult::new(vec![0.0], 22050);
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("SynthesisResult"));
}

#[test]
fn test_tts_config_debug() {
    let config = TtsConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("TtsConfig"));
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
