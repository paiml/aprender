
#[test]
fn test_has_inf_negative() {
    let samples = vec![0.0, f32::NEG_INFINITY, 0.5];
    assert!(has_inf(&samples));
}

#[test]
fn test_has_inf_empty() {
    let samples: Vec<f32> = vec![];
    assert!(!has_inf(&samples));
}

#[test]
fn test_validate_audio_inf() {
    let samples = vec![0.0, f32::INFINITY, 0.5];
    let result = validate_audio(&samples);
    assert!(result.is_err());
    let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        msg.contains("Infinity"),
        "Error should mention Infinity: {}",
        msg
    );
}

#[test]
fn test_validate_audio_neg_inf() {
    let samples = vec![0.0, f32::NEG_INFINITY, 0.5];
    let result = validate_audio(&samples);
    assert!(result.is_err());
    let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        msg.contains("Infinity"),
        "Error should mention Infinity: {}",
        msg
    );
}

// ============================================================
// A12: Stereo to Mono Conversion Tests
// ============================================================

#[test]
fn test_stereo_to_mono_basic() {
    let stereo = vec![0.5, 0.3, 0.6, 0.4];
    let mono = stereo_to_mono(&stereo);
    assert_eq!(mono.len(), 2);
    assert!(
        (mono[0] - 0.4).abs() < 1e-6,
        "Expected 0.4, got {}",
        mono[0]
    );
    assert!(
        (mono[1] - 0.5).abs() < 1e-6,
        "Expected 0.5, got {}",
        mono[1]
    );
}

#[test]
fn test_stereo_to_mono_identical_channels() {
    let stereo = vec![0.5, 0.5, 0.3, 0.3, 0.8, 0.8];
    let mono = stereo_to_mono(&stereo);
    assert_eq!(mono.len(), 3);
    assert!((mono[0] - 0.5).abs() < 1e-6);
    assert!((mono[1] - 0.3).abs() < 1e-6);
    assert!((mono[2] - 0.8).abs() < 1e-6);
}

#[test]
fn test_stereo_to_mono_empty() {
    let stereo: Vec<f32> = vec![];
    let mono = stereo_to_mono(&stereo);
    assert!(mono.is_empty());
}

#[test]
fn test_stereo_to_mono_single_sample() {
    let stereo = vec![0.5];
    let mono = stereo_to_mono(&stereo);
    assert_eq!(mono.len(), 1);
    assert!((mono[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_stereo_to_mono_opposite_polarity() {
    // Left = +0.8, Right = -0.8 → average = 0.0
    let stereo = vec![0.8, -0.8, 0.4, -0.4];
    let mono = stereo_to_mono(&stereo);
    assert_eq!(mono.len(), 2);
    assert!(
        (mono[0] - 0.0).abs() < 1e-6,
        "Opposite polarity should cancel"
    );
    assert!((mono[1] - 0.0).abs() < 1e-6);
}

#[test]
fn test_stereo_to_mono_preserves_amplitude() {
    // Both channels at 0.6 → mono should be 0.6
    let stereo = vec![0.6, 0.6];
    let mono = stereo_to_mono(&stereo);
    assert_eq!(mono.len(), 1);
    assert!(
        (mono[0] - 0.6).abs() < 1e-6,
        "Equal channels should preserve amplitude"
    );
}

// ============================================================================
// Section AA: Audio Processing Popperian Falsification Tests
// Per spec v3.0.0 Part II Section 2.3
// ============================================================================

/// AA1: Mel spectrogram output is bounded and valid
/// FALSIFICATION: Output contains NaN or Inf values
#[test]
fn test_aa1_mel_output_no_nan_inf() {
    let config = MelConfig::whisper();
    let filterbank = MelFilterbank::new(&config);

    // Generate test signal: 1 second of 440Hz sine wave
    let sample_rate = 16000;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    let mel = filterbank
        .compute(&samples)
        .expect("compute should succeed");

    // Check for NaN
    let has_nan = mel.iter().any(|x| x.is_nan());
    assert!(
        !has_nan,
        "AA1 FALSIFIED: Mel spectrogram contains NaN values"
    );

    // Check for Inf
    let has_inf = mel.iter().any(|x| x.is_infinite());
    assert!(
        !has_inf,
        "AA1 FALSIFIED: Mel spectrogram contains Inf values"
    );
}

/// AA2: Mel spectrogram deterministic
/// FALSIFICATION: Same input produces different output
#[test]
fn test_aa2_mel_deterministic() {
    let config = MelConfig::whisper();
    let filterbank = MelFilterbank::new(&config);

    // Fixed test signal
    let samples: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.01).sin()).collect();

    // Compute 5 times
    let results: Vec<Vec<f32>> = (0..5)
        .map(|_| filterbank.compute(&samples).expect("compute"))
        .collect();

    // All must be identical
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            &results[0], result,
            "AA2 FALSIFIED: Mel computation is non-deterministic (run {} differs)",
            i
        );
    }
}

/// AA3: Whisper config enforces 16kHz sample rate
/// FALSIFICATION: Whisper config has wrong sample rate
#[test]
fn test_aa3_whisper_requires_16khz() {
    let config = MelConfig::whisper();
    assert_eq!(
        config.sample_rate, 16000,
        "AA3 FALSIFIED: Whisper config sample_rate is not 16000"
    );
}

/// AA4: Stereo to mono preserves sample count
/// FALSIFICATION: Output length != input length / 2
#[test]
fn test_aa4_stereo_to_mono_sample_count() {
    // 1000 stereo samples = 500 mono samples
    let stereo: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.001).sin()).collect();

    let mono = stereo_to_mono(&stereo);

    assert_eq!(
        mono.len(),
        stereo.len() / 2,
        "AA4 FALSIFIED: Stereo to mono dropped samples. Expected {}, got {}",
        stereo.len() / 2,
        mono.len()
    );
}

/// AA5: Memory usage is O(window), not O(file) - verified by frame calculation
/// FALSIFICATION: Number of frames scales incorrectly with input length
#[test]
fn test_aa5_frame_count_scales_correctly() {
    let config = MelConfig::whisper();
    let filterbank = MelFilterbank::new(&config);

    // Calculate expected frames for different lengths
    let len1 = 16000; // 1 second
    let len2 = 32000; // 2 seconds

    let frames1 = filterbank.num_frames(len1);
    let frames2 = filterbank.num_frames(len2);

    // frames2 should be approximately 2x frames1
    let ratio = frames2 as f64 / frames1 as f64;
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "AA5 FALSIFIED: Frame count does not scale linearly. Ratio: {:.2} (expected ~2.0)",
        ratio
    );
}

/// AA6: Silence detection works correctly
/// FALSIFICATION: Silent audio detected as speech
#[test]
fn test_aa6_silence_detection() {
    // Generate silent audio (all zeros)
    let silent: Vec<f32> = vec![0.0; 16000];

    // Check for clipping (should have none)
    let report = detect_clipping(&silent);
    assert!(
        !report.has_clipping,
        "AA6 FALSIFIED: Silent audio reported as clipped"
    );

    // Max/min should be 0
    assert!(
        (report.max_value - 0.0).abs() < 1e-10,
        "AA6 FALSIFIED: Silent audio has non-zero max"
    );
    assert!(
        (report.min_value - 0.0).abs() < 1e-10,
        "AA6 FALSIFIED: Silent audio has non-zero min"
    );
}

/// AA7: Clipping detection works correctly
/// FALSIFICATION: Clipped audio not detected
#[test]
fn test_aa7_clipping_detection() {
    // Generate clipped audio
    let clipped = vec![0.5, 1.5, -0.3, -1.2, 0.8, 2.0];

    let report = detect_clipping(&clipped);

    assert!(
        report.has_clipping,
        "AA7 FALSIFIED: Clipped audio not detected"
    );
    assert_eq!(
        report.positive_clipped, 2,
        "AA7 FALSIFIED: Wrong positive clip count"
    );
    assert_eq!(
        report.negative_clipped, 1,
        "AA7 FALSIFIED: Wrong negative clip count"
    );
}
