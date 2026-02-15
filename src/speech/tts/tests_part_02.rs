
#[test]
fn test_alignment_info_debug() {
    let align = AlignmentInfo::new("a".to_string(), 0.0, 0.1);
    let debug_str = format!("{:?}", align);
    assert!(debug_str.contains("AlignmentInfo"));
}

#[test]
fn test_fastspeech2_debug() {
    let synth = FastSpeech2Synthesizer::default_config();
    let debug_str = format!("{:?}", synth);
    assert!(debug_str.contains("FastSpeech2"));
}

#[test]
fn test_vits_debug() {
    let synth = VitsSynthesizer::default_config();
    let debug_str = format!("{:?}", synth);
    assert!(debug_str.contains("VitsSynthesizer"));
}

#[test]
fn test_hifigan_debug() {
    let vocoder = HifiGanVocoder::default_config();
    let debug_str = format!("{:?}", vocoder);
    assert!(debug_str.contains("HifiGanVocoder"));
}

#[test]
fn test_tts_config_clone() {
    let config = TtsConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.sample_rate, config.sample_rate);
    assert_eq!(cloned.n_mels, config.n_mels);
}

#[test]
fn test_synthesis_request_clone() {
    let request = SynthesisRequest::new("Hello".to_string()).with_speaker("alice".to_string());
    let cloned = request.clone();
    assert_eq!(cloned.text, request.text);
    assert_eq!(cloned.speaker_id, request.speaker_id);
}

#[test]
fn test_synthesis_result_clone() {
    let result = SynthesisResult::new(vec![1.0, 2.0, 3.0], 22050);
    let cloned = result.clone();
    assert_eq!(cloned.audio, result.audio);
    assert_eq!(cloned.sample_rate, result.sample_rate);
}

#[test]
fn test_alignment_info_clone() {
    let align = AlignmentInfo::new("hello".to_string(), 0.0, 0.5);
    let cloned = align.clone();
    assert_eq!(cloned.token, align.token);
    assert!((cloned.start - align.start).abs() < f32::EPSILON);
}
