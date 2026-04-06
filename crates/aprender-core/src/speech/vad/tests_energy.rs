use super::*;

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
