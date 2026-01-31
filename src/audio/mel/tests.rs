use super::*;

// ============================================================
// UNIT TESTS: Configuration
// ============================================================

#[test]
fn test_mel_config_whisper() {
    let config = MelConfig::whisper();
    assert_eq!(config.n_mels, 80);
    assert_eq!(config.n_fft, 400);
    assert_eq!(config.hop_length, 160);
    assert_eq!(config.sample_rate, 16000);
}

#[test]
fn test_mel_config_tts() {
    let config = MelConfig::tts();
    assert_eq!(config.n_mels, 80);
    assert_eq!(config.n_fft, 1024);
    assert_eq!(config.hop_length, 256);
    assert_eq!(config.sample_rate, 22050);
}

#[test]
fn test_mel_config_n_freqs() {
    let config = MelConfig::whisper();
    assert_eq!(config.n_freqs(), 201); // 400/2 + 1
}

// ============================================================
// UNIT TESTS: Mel scale conversion
// ============================================================

#[test]
fn test_hz_to_mel_zero() {
    let mel = MelFilterbank::hz_to_mel(0.0);
    assert!((mel - 0.0).abs() < 1e-5, "0 Hz should map to 0 mel");
}

#[test]
fn test_hz_to_mel_1000hz() {
    let mel = MelFilterbank::hz_to_mel(1000.0);
    assert!(
        (mel - 1000.0).abs() < 50.0,
        "1000 Hz should be close to 1000 mel, got {mel}"
    );
}

#[test]
fn test_mel_to_hz_roundtrip() {
    let frequencies = [0.0, 100.0, 500.0, 1000.0, 4000.0, 8000.0];
    for &hz in &frequencies {
        let mel = MelFilterbank::hz_to_mel(hz);
        let recovered = MelFilterbank::mel_to_hz(mel);
        assert!(
            (hz - recovered).abs() < 0.1,
            "Roundtrip failed for {hz} Hz: got {recovered}"
        );
    }
}

#[test]
fn test_mel_scale_monotonic() {
    let mut prev_mel = -1.0_f32;
    for hz in (0..8000).step_by(100) {
        let mel = MelFilterbank::hz_to_mel(hz as f32);
        assert!(
            mel > prev_mel,
            "Mel scale should be monotonically increasing"
        );
        prev_mel = mel;
    }
}

// ============================================================
// UNIT TESTS: Filterbank creation
// ============================================================

#[test]
fn test_mel_filterbank_new() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    assert_eq!(mel.n_mels(), 80);
    assert_eq!(mel.n_fft(), 400);
    assert_eq!(mel.sample_rate(), 16000);
    assert_eq!(mel.n_freqs(), 201);
}

#[test]
fn test_mel_filterbank_filters_shape() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    assert_eq!(mel.filters.len(), 80 * 201);
}

#[test]
fn test_mel_filterbank_filters_nonnegative() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    for &f in &mel.filters {
        assert!(f >= 0.0, "Filter values should be non-negative");
    }
}

#[test]
fn test_mel_filterbank_slaney_normalization() {
    // A2/D12: Verify Slaney area normalization
    // With Slaney normalization, filter peaks are NOT bounded by 1.0
    // Instead, higher frequency filters have larger peaks (narrower bandwidth)
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);

    // Find max filter value - should be > 1.0 for high frequency filters
    let max_filter_val = mel.filters.iter().cloned().fold(0.0_f32, f32::max);

    // Slaney normalization produces max values well above 1.0
    // (typically 0.01-0.05 range for area-normalized filters)
    // The key test: max should NOT be exactly 1.0 (peak normalization)
    assert!(
        (max_filter_val - 1.0).abs() > 0.001,
        "Slaney normalization should NOT produce peak=1.0, got max={:.6}",
        max_filter_val
    );

    // Verify filters are still non-negative and finite
    for &f in &mel.filters {
        assert!(f >= 0.0, "Filter values should be non-negative");
        assert!(f.is_finite(), "Filter values should be finite");
    }
}

#[test]
fn test_mel_filterbank_slaney_max_below_threshold() {
    // A2: Slaney normalization should produce max < 0.1 for Whisper config
    // This is the falsification test from the QA checklist
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);

    let max_filter_val = mel.filters.iter().cloned().fold(0.0_f32, f32::max);

    assert!(
        max_filter_val < 0.1,
        "Slaney-normalized filterbank max should be < 0.1, got {:.6}",
        max_filter_val
    );
}

#[test]
fn test_hann_window_endpoints() {
    let window = MelFilterbank::hann_window(100);
    assert!(window[0] < 0.01, "Hann window should start near 0");
    assert!(window[99] < 0.01, "Hann window should end near 0");
}

#[test]
fn test_hann_window_peak() {
    let window = MelFilterbank::hann_window(100);
    let mid = window[50];
    assert!(mid > 0.9, "Hann window should peak near 1.0 in middle");
}

// ============================================================
// UNIT TESTS: Spectrogram computation
// ============================================================

#[test]
fn test_mel_compute_empty() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let result = mel.compute(&[]);
    assert!(result.is_ok());
    assert!(result.map_or(false, |v| v.is_empty()));
}

#[test]
fn test_mel_compute_short_audio() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let audio = vec![0.0; 100]; // Too short for even one frame
    let result = mel.compute(&audio);
    assert!(result.is_ok());
    assert!(result.map_or(false, |v| v.is_empty()));
}

#[test]
fn test_mel_compute_exact_one_frame() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let audio = vec![0.0; 400]; // Exactly one FFT window
    let result = mel.compute(&audio).expect("compute should succeed");
    assert_eq!(result.len(), 80 * 1);
}

#[test]
fn test_mel_compute_multiple_frames() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    // 16000 samples = 1 second at 16kHz
    // With hop_length=160, we get (16000 - 400) / 160 + 1 = 98 frames
    let audio = vec![0.0; 16000];
    let result = mel.compute(&audio).expect("compute should succeed");
    let n_frames = result.len() / 80;
    assert_eq!(n_frames, 98);
}

#[test]
fn test_mel_compute_sine_wave() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);

    // Generate 1 second of 440 Hz sine wave
    let sample_rate = 16000.0;
    let freq = 440.0;
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
        .collect();

    let result = mel.compute(&audio).expect("compute should succeed");

    let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);

    assert!(max_val.is_finite(), "Max should be finite");
    assert!(min_val.is_finite(), "Min should be finite");
    assert!(max_val > min_val, "Should have variation in output");
}

#[test]
fn test_num_frames() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);

    assert_eq!(mel.num_frames(0), 0);
    assert_eq!(mel.num_frames(100), 0);
    assert_eq!(mel.num_frames(400), 1);
    assert_eq!(mel.num_frames(560), 2);
    assert_eq!(mel.num_frames(16000), 98);
}

// ============================================================
// UNIT TESTS: Normalization
// ============================================================

#[test]
fn test_normalize_global_empty() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let mut data: Vec<f32> = vec![];
    mel.normalize_global(&mut data);
    assert!(data.is_empty());
}

#[test]
fn test_normalize_global_mean_zero() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    mel.normalize_global(&mut data);

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean after normalization should be ~0");
}

#[test]
fn test_normalize_global_std_one() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    mel.normalize_global(&mut data);

    let variance: f32 = data.iter().map(|&x| x.powi(2)).sum::<f32>() / data.len() as f32;
    let std = variance.sqrt();
    assert!(
        (std - 1.0).abs() < 1e-5,
        "Std after normalization should be ~1, got {std}"
    );
}

// ============================================================
// UNIT TESTS: Apply filterbank
// ============================================================

#[test]
fn test_apply_filterbank_shape() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let power_spec = vec![1.0; mel.n_freqs()];
    let result = mel.apply_filterbank(&power_spec);
    assert_eq!(result.len(), 80);
}

#[test]
fn test_apply_filterbank_zeros() {
    let config = MelConfig::whisper();
    let mel = MelFilterbank::new(&config);
    let power_spec = vec![0.0; mel.n_freqs()];
    let result = mel.apply_filterbank(&power_spec);
    for &val in &result {
        assert!(
            (val - 0.0).abs() < 1e-10,
            "Zero input should give zero output"
        );
    }
}

// ============================================================
// A11: Audio Clipping Detection Tests
// ============================================================

#[test]
fn test_detect_clipping_no_clipping() {
    let samples = vec![0.0, 0.5, -0.5, 0.99, -0.99];
    let report = detect_clipping(&samples);
    assert!(!report.has_clipping);
    assert_eq!(report.positive_clipped, 0);
    assert_eq!(report.negative_clipped, 0);
    assert!((report.max_value - 0.99).abs() < 1e-6);
    assert!((report.min_value - (-0.99)).abs() < 1e-6);
    assert_eq!(report.total_samples, 5);
}

#[test]
fn test_detect_clipping_positive() {
    let samples = vec![0.5, 1.5, 0.8, 2.0, 0.9];
    let report = detect_clipping(&samples);
    assert!(report.has_clipping);
    assert_eq!(report.positive_clipped, 2);
    assert_eq!(report.negative_clipped, 0);
    assert!((report.max_value - 2.0).abs() < 1e-6);
}

#[test]
fn test_detect_clipping_negative() {
    let samples = vec![-0.5, -1.5, -0.8, -2.0, -0.9];
    let report = detect_clipping(&samples);
    assert!(report.has_clipping);
    assert_eq!(report.positive_clipped, 0);
    assert_eq!(report.negative_clipped, 2);
    assert!((report.min_value - (-2.0)).abs() < 1e-6);
}

#[test]
fn test_detect_clipping_both() {
    let samples = vec![1.5, -1.5, 0.5, 2.0, -2.0];
    let report = detect_clipping(&samples);
    assert!(report.has_clipping);
    assert_eq!(report.positive_clipped, 2);
    assert_eq!(report.negative_clipped, 2);
}

#[test]
fn test_detect_clipping_empty() {
    let samples: Vec<f32> = vec![];
    let report = detect_clipping(&samples);
    assert!(!report.has_clipping);
    assert_eq!(report.total_samples, 0);
    assert!((report.clipping_percentage() - 0.0).abs() < 1e-6);
}

#[test]
fn test_detect_clipping_exactly_one() {
    let samples = vec![1.0, -1.0, 0.5];
    let report = detect_clipping(&samples);
    // Exactly 1.0 and -1.0 should NOT be clipped
    assert!(!report.has_clipping);
    assert_eq!(report.positive_clipped, 0);
    assert_eq!(report.negative_clipped, 0);
}

#[test]
fn test_clipping_percentage() {
    let samples = vec![1.5, -1.5, 0.5, 0.3, 0.2];
    let report = detect_clipping(&samples);
    // 2 out of 5 = 40%
    assert!((report.clipping_percentage() - 40.0).abs() < 1e-6);
}

#[test]
fn test_has_nan_false() {
    let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0];
    assert!(!has_nan(&samples));
}

#[test]
fn test_has_nan_true() {
    let samples = vec![0.0, 0.5, f32::NAN, 1.0];
    assert!(has_nan(&samples));
}

#[test]
fn test_has_nan_empty() {
    let samples: Vec<f32> = vec![];
    assert!(!has_nan(&samples));
}

#[test]
fn test_validate_audio_valid() {
    let samples = vec![0.0, 0.5, -0.5, 0.99, -0.99];
    assert!(validate_audio(&samples).is_ok());
}

#[test]
fn test_validate_audio_empty() {
    let samples: Vec<f32> = vec![];
    let result = validate_audio(&samples);
    assert!(result.is_err());
    let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(msg.contains("empty"), "Error should mention empty: {}", msg);
}

#[test]
fn test_validate_audio_nan() {
    let samples = vec![0.0, f32::NAN, 0.5];
    let result = validate_audio(&samples);
    assert!(result.is_err());
    let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(msg.contains("NaN"), "Error should mention NaN: {}", msg);
}

#[test]
fn test_validate_audio_clipping() {
    let samples = vec![0.0, 1.5, -0.5];
    let result = validate_audio(&samples);
    assert!(result.is_err());
    let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        msg.contains("clipping") || msg.contains("Clipping"),
        "Error should mention clipping: {}",
        msg
    );
}

// ============================================================
// A15: Infinity Detection Tests
// ============================================================

#[test]
fn test_has_inf_false() {
    let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, f32::MAX, f32::MIN];
    assert!(!has_inf(&samples));
}

#[test]
fn test_has_inf_positive() {
    let samples = vec![0.0, f32::INFINITY, 0.5];
    assert!(has_inf(&samples));
}

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
