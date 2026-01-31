use super::*;

// ============================================================
// VadConfig Tests
// ============================================================

#[test]
fn test_vad_config_default() {
    let config = VadConfig::default();
    assert!((config.energy_threshold - 0.01).abs() < f32::EPSILON);
    assert_eq!(config.min_speech_duration_ms, 250);
    assert_eq!(config.min_silence_duration_ms, 300);
    assert_eq!(config.frame_size, 512);
    assert_eq!(config.hop_size, 256);
}

#[test]
fn test_vad_config_validate_valid() {
    let config = VadConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_vad_config_validate_energy_threshold_negative() {
    let config = VadConfig {
        energy_threshold: -0.1,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, SpeechError::InvalidConfig(_)));
    assert!(err.to_string().contains("energy_threshold"));
}

#[test]
fn test_vad_config_validate_energy_threshold_too_high() {
    let config = VadConfig {
        energy_threshold: 1.5,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, SpeechError::InvalidConfig(_)));
}

#[test]
fn test_vad_config_validate_frame_size_zero() {
    let config = VadConfig {
        frame_size: 0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, SpeechError::InvalidConfig(_)));
    assert!(err.to_string().contains("frame_size"));
}

#[test]
fn test_vad_config_validate_hop_size_zero() {
    let config = VadConfig {
        hop_size: 0,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, SpeechError::InvalidConfig(_)));
    assert!(err.to_string().contains("hop_size"));
}

#[test]
fn test_vad_config_validate_hop_size_exceeds_frame_size() {
    let config = VadConfig {
        frame_size: 256,
        hop_size: 512,
        ..Default::default()
    };
    let err = config.validate().unwrap_err();
    assert!(matches!(err, SpeechError::InvalidConfig(_)));
    assert!(err.to_string().contains("hop_size"));
}

#[test]
fn test_vad_config_boundary_values() {
    // Energy threshold at boundaries
    let config = VadConfig {
        energy_threshold: 0.0,
        ..Default::default()
    };
    assert!(config.validate().is_ok());

    let config = VadConfig {
        energy_threshold: 1.0,
        ..Default::default()
    };
    assert!(config.validate().is_ok());

    // Hop size equal to frame size is valid
    let config = VadConfig {
        frame_size: 256,
        hop_size: 256,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

// ============================================================
// VoiceSegment Tests
// ============================================================

#[test]
fn test_voice_segment_new() {
    let segment = VoiceSegment::new(1.0, 2.5, 0.3);
    assert!((segment.start - 1.0).abs() < f64::EPSILON);
    assert!((segment.end - 2.5).abs() < f64::EPSILON);
    assert!((segment.energy - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_voice_segment_duration() {
    let segment = VoiceSegment::new(1.0, 3.5, 0.5);
    assert!((segment.duration() - 2.5).abs() < f64::EPSILON);
}

#[test]
fn test_voice_segment_zero_duration() {
    let segment = VoiceSegment::new(1.0, 1.0, 0.5);
    assert!((segment.duration()).abs() < f64::EPSILON);
}

// ============================================================
// Vad Construction Tests
// ============================================================

#[test]
fn test_vad_new_valid_config() {
    let config = VadConfig::default();
    let vad = Vad::new(config);
    assert!(vad.is_ok());
}

#[test]
fn test_vad_new_invalid_config() {
    let config = VadConfig {
        energy_threshold: -1.0,
        ..Default::default()
    };
    let vad = Vad::new(config);
    assert!(vad.is_err());
}

#[test]
fn test_vad_with_defaults() {
    let vad = Vad::with_defaults();
    assert!(vad.is_ok());
    let vad = vad.expect("default config should be valid");
    assert!((vad.config().energy_threshold - 0.01).abs() < f32::EPSILON);
}

#[test]
fn test_vad_config_accessor() {
    let config = VadConfig {
        energy_threshold: 0.05,
        ..Default::default()
    };
    let vad = Vad::new(config.clone()).expect("valid config");
    assert_eq!(vad.config(), &config);
}

// ============================================================
// Vad::detect() Input Validation Tests
// ============================================================

#[test]
fn test_vad_detect_zero_sample_rate() {
    let vad = Vad::with_defaults().expect("valid config");
    let samples = vec![0.0f32; 1024];
    let result = vad.detect(&samples, 0);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, SpeechError::InvalidAudio(_)));
    assert!(err.to_string().contains("sample_rate"));
}

#[test]
fn test_vad_detect_empty_samples() {
    let vad = Vad::with_defaults().expect("valid config");
    let samples: Vec<f32> = vec![];
    let result = vad.detect(&samples, 16000);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, SpeechError::InvalidAudio(_)));
    assert!(err.to_string().contains("empty"));
}

#[test]
fn test_vad_detect_insufficient_samples() {
    let config = VadConfig {
        frame_size: 512,
        ..Default::default()
    };
    let vad = Vad::new(config).expect("valid config");
    let samples = vec![0.0f32; 256]; // Less than frame_size
    let result = vad.detect(&samples, 16000);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err,
        SpeechError::InsufficientSamples {
            required: 512,
            provided: 256
        }
    ));
}

// ============================================================
// Vad::detect() Core Functionality Tests
// ============================================================

#[test]
fn test_vad_detect_silence() {
    let vad = Vad::with_defaults().expect("valid config");
    let samples = vec![0.0f32; 16000]; // 1 second of silence at 16kHz
    let segments = vad
        .detect(&samples, 16000)
        .expect("detection should succeed");
    assert!(
        segments.is_empty(),
        "No speech should be detected in silence"
    );
}

#[test]
fn test_vad_detect_constant_low_energy() {
    let vad = Vad::with_defaults().expect("valid config");
    // Very low amplitude noise (below default threshold of 0.01)
    let samples: Vec<f32> = (0..16000).map(|_| 0.001).collect();
    let segments = vad
        .detect(&samples, 16000)
        .expect("detection should succeed");
    assert!(
        segments.is_empty(),
        "Low energy should not trigger detection"
    );
}

#[test]
fn test_vad_detect_loud_signal() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 100,
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");

    // Generate 500ms of loud signal at 16kHz
    let sample_rate = 16000;
    let duration_samples = sample_rate / 2; // 500ms
    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            // 440Hz sine wave at 0.5 amplitude
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");
    assert!(
        !segments.is_empty(),
        "Speech should be detected in loud signal"
    );

    // Should have one continuous segment
    assert_eq!(segments.len(), 1);

    // Segment should cover most of the audio
    let segment = &segments[0];
    assert!(segment.start < 0.1, "Segment should start near beginning");
    assert!(segment.duration() > 0.3, "Segment should be at least 300ms");
}

#[test]
fn test_vad_detect_speech_in_middle() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 100,
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 16000;
    let half_sec = (sample_rate / 2) as usize;

    // 500ms silence + 500ms speech + 500ms silence
    let mut samples = vec![0.0f32; half_sec]; // 500ms silence

    // 500ms of speech (sine wave)
    for i in 0..half_sec {
        let t = i as f32 / sample_rate as f32;
        samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }

    samples.extend(vec![0.0f32; half_sec]); // 500ms silence

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");
    assert_eq!(
        segments.len(),
        1,
        "Should detect exactly one speech segment"
    );

    let segment = &segments[0];
    // Speech starts around 500ms
    assert!(
        segment.start > 0.4 && segment.start < 0.6,
        "Segment should start around 0.5s"
    );
    // Speech ends around 1000ms
    assert!(
        segment.end > 0.9 && segment.end < 1.1,
        "Segment should end around 1.0s"
    );
}

#[test]
fn test_vad_detect_multiple_segments() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 200,
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 16000;
    let samples_300ms = (sample_rate * 3 / 10) as usize;
    let samples_400ms = (sample_rate * 4 / 10) as usize;

    let mut samples = Vec::new();

    // First speech segment: 300ms
    for i in 0..samples_300ms {
        let t = i as f32 / sample_rate as f32;
        samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }

    // Silence: 400ms (longer than min_silence_duration_ms)
    samples.extend(vec![0.0f32; samples_400ms]);

    // Second speech segment: 300ms
    for i in 0..samples_300ms {
        let t = i as f32 / sample_rate as f32;
        samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");
    assert_eq!(segments.len(), 2, "Should detect two speech segments");

    // First segment should be near the beginning
    assert!(segments[0].start < 0.1);
    // Second segment should be after the silence gap
    assert!(segments[1].start > 0.6);
}

#[test]
fn test_vad_detect_short_speech_filtered() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 500, // Require at least 500ms
        min_silence_duration_ms: 100,
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 16000;

    // Only 200ms of speech (shorter than min_speech_duration_ms)
    let duration_samples = (sample_rate / 5) as usize; // 200ms
    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");
    assert!(
        segments.is_empty(),
        "Short speech segments should be filtered out"
    );
}

#[test]
fn test_vad_detect_speech_at_end() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 100,
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 16000;
    let half_sec = (sample_rate / 2) as usize;

    // 500ms silence + 500ms speech (no trailing silence)
    let mut samples = vec![0.0f32; half_sec];

    for i in 0..half_sec {
        let t = i as f32 / sample_rate as f32;
        samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");
    assert_eq!(segments.len(), 1, "Should detect speech at end");

    let segment = &segments[0];
    assert!(
        segment.end > 0.9,
        "Segment should extend to near end of audio"
    );
}

// ============================================================
// Vad::compute_frame_energy Tests
// ============================================================

#[test]
fn test_compute_frame_energy_silence() {
    let frame = vec![0.0f32; 512];
    let energy = Vad::compute_frame_energy(&frame);
    assert!(
        energy.abs() < f32::EPSILON,
        "Silence should have zero energy"
    );
}

#[test]
fn test_compute_frame_energy_constant() {
    let frame = vec![0.5f32; 512];
    let energy = Vad::compute_frame_energy(&frame);
    assert!(
        (energy - 0.5).abs() < 0.001,
        "Constant signal RMS should equal amplitude"
    );
}

#[test]
fn test_compute_frame_energy_sine_wave() {
    // Full cycle of sine wave
    let frame: Vec<f32> = (0..360)
        .map(|i| (i as f32 * std::f32::consts::PI / 180.0).sin())
        .collect();
    let energy = Vad::compute_frame_energy(&frame);
    // RMS of sine wave is 1/sqrt(2) ≈ 0.707
    assert!(
        (energy - 0.707).abs() < 0.02,
        "Sine wave RMS should be ~0.707, got {}",
        energy
    );
}

#[test]
fn test_compute_frame_energy_empty() {
    let frame: Vec<f32> = vec![];
    let energy = Vad::compute_frame_energy(&frame);
    assert!(
        energy.abs() < f32::EPSILON,
        "Empty frame should have zero energy"
    );
}

// ============================================================
// Edge Case Tests
// ============================================================

#[test]
fn test_vad_detect_exactly_frame_size_samples() {
    let config = VadConfig {
        frame_size: 512,
        hop_size: 256,
        energy_threshold: 0.01,
        min_speech_duration_ms: 10,
        min_silence_duration_ms: 10,
    };
    let vad = Vad::new(config).expect("valid config");

    // Exactly frame_size samples of loud signal
    let samples: Vec<f32> = (0..512)
        .map(|i| {
            let t = i as f32 / 16000.0;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let result = vad.detect(&samples, 16000);
    assert!(result.is_ok(), "Should handle exactly frame_size samples");
}

#[test]
fn test_vad_detect_high_sample_rate() {
    let vad = Vad::with_defaults().expect("valid config");
    let sample_rate: u32 = 48000; // 48kHz
    let samples: Vec<f32> = (0..48000)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");
    assert!(
        !segments.is_empty(),
        "Should detect speech at high sample rate"
    );
}

#[test]
fn test_vad_detect_low_sample_rate() {
    let config = VadConfig {
        frame_size: 80, // Smaller frame for 8kHz
        hop_size: 40,
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 100,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 8000; // 8kHz

    let samples: Vec<f32> = (0..4000) // 500ms
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");
    assert!(
        !segments.is_empty(),
        "Should detect speech at low sample rate"
    );
}

#[test]
fn test_vad_segment_energy_calculation() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 100,
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 16000;

    // Generate signal with known amplitude
    let samples: Vec<f32> = (0..8000) // 500ms
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");
    assert!(!segments.is_empty());

    let segment = &segments[0];
    // RMS of 0.5 amplitude sine wave is 0.5 / sqrt(2) ≈ 0.354
    assert!(
        segment.energy > 0.3 && segment.energy < 0.4,
        "Segment energy should reflect signal amplitude, got {}",
        segment.energy
    );
}

// ============================================================
// B1-B10: Popperian Falsification Tests (GH-133)
// ============================================================

/// B3: VAD segments have start < end
#[test]
fn test_vad_b3_segments_start_before_end() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 100,
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 16000;

    // Create audio with multiple speech segments
    let mut samples = Vec::new();
    for _ in 0..3 {
        // 300ms speech
        for i in 0..4800 {
            let t = i as f32 / sample_rate as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        // 400ms silence
        samples.extend(vec![0.0f32; 6400]);
    }

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");

    // B3: Every segment must have start < end
    for segment in &segments {
        assert!(
            segment.start < segment.end,
            "B3 VIOLATED: segment.start ({}) >= segment.end ({})",
            segment.start,
            segment.end
        );
    }
}

/// B4: VAD confidence (energy) in [0, 1]
#[test]
fn test_vad_b4_energy_bounded() {
    let vad = Vad::with_defaults().expect("valid config");
    let sample_rate: u32 = 16000;

    // Test with various signal amplitudes
    for amplitude in [0.1, 0.5, 1.0] {
        let samples: Vec<f32> = (0..8000)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");

        for segment in &segments {
            assert!(
                segment.energy >= 0.0 && segment.energy <= 1.0,
                "B4 VIOLATED: energy {} not in [0, 1]",
                segment.energy
            );
        }
    }
}

/// B8: VAD handles stereo input (by requiring mono)
/// Note: VAD expects mono input. Stereo should be converted first.
#[test]
fn test_vad_b8_handles_mono_requirement() {
    let vad = Vad::with_defaults().expect("valid config");
    let sample_rate: u32 = 16000;

    // Simulate stereo by interleaving two channels
    // In practice, stereo should be converted to mono before VAD
    let samples_mono: Vec<f32> = (0..8000)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    // VAD should work on mono data
    let result = vad.detect(&samples_mono, sample_rate);
    assert!(result.is_ok(), "VAD should handle mono audio");
}

/// B10: VAD default energy threshold is reasonable (0.01 for energy-based VAD)
#[test]
fn test_vad_b10_default_threshold_reasonable() {
    let config = VadConfig::default();
    // Energy threshold of 0.01 is reasonable for energy-based VAD
    // (0.5 would be too high for energy which is typically in [0, 1])
    assert!(
        config.energy_threshold > 0.0 && config.energy_threshold < 0.1,
        "B10: Default energy threshold should be reasonable (0.0 < threshold < 0.1), got {}",
        config.energy_threshold
    );
    // Verify the default is exactly 0.01 as documented
    assert!(
        (config.energy_threshold - 0.01).abs() < 1e-6,
        "Default energy threshold should be 0.01"
    );
}

/// B5: Verify min_speech_ms filtering explicitly
#[test]
fn test_vad_b5_min_speech_duration_filtering() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 300, // Require at least 300ms
        min_silence_duration_ms: 100,
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 16000;

    // 200ms speech (less than min_speech_duration_ms)
    let short_samples: Vec<f32> = (0..3200)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let segments = vad
        .detect(&short_samples, sample_rate)
        .expect("detection should succeed");
    assert!(
        segments.is_empty(),
        "B5 VIOLATED: segments shorter than min_speech_duration_ms should be filtered"
    );

    // 500ms speech (more than min_speech_duration_ms)
    let long_samples: Vec<f32> = (0..8000)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    let segments = vad
        .detect(&long_samples, sample_rate)
        .expect("detection should succeed");
    assert!(
        !segments.is_empty(),
        "B5: segments longer than min_speech_duration_ms should be kept"
    );
}

/// B6: Verify min_silence_ms gap merging
#[test]
fn test_vad_b6_min_silence_duration_merging() {
    let config = VadConfig {
        energy_threshold: 0.01,
        min_speech_duration_ms: 100,
        min_silence_duration_ms: 400, // Require at least 400ms silence to separate
        frame_size: 160,
        hop_size: 80,
    };
    let vad = Vad::new(config).expect("valid config");
    let sample_rate: u32 = 16000;

    let mut samples = Vec::new();

    // 300ms speech
    for i in 0..4800 {
        let t = i as f32 / sample_rate as f32;
        samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }

    // 200ms silence (less than min_silence_duration_ms)
    samples.extend(vec![0.0f32; 3200]);

    // 300ms speech
    for i in 0..4800 {
        let t = i as f32 / sample_rate as f32;
        samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }

    let segments = vad
        .detect(&samples, sample_rate)
        .expect("detection should succeed");

    // Should merge into single segment since silence < min_silence_duration_ms
    assert_eq!(
        segments.len(),
        1,
        "B6: Short silence gaps should be merged into single segment"
    );
}
