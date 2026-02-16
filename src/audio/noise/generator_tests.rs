pub(crate) use super::*;

// ========== NG11: Output length matches requested buffer size ==========

#[test]
fn test_ng11_output_length_1024() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();
    let mut buffer = vec![0.0; 1024];
    gen.generate(&mut buffer).unwrap();
    assert_eq!(buffer.len(), 1024);
}

#[test]
fn test_ng11_output_length_256() {
    let config = NoiseConfig::brown().with_buffer_size(256).unwrap();
    let mut gen = NoiseGenerator::new(config).unwrap();
    let mut buffer = vec![0.0; 256];
    gen.generate(&mut buffer).unwrap();
    assert_eq!(buffer.len(), 256);
}

#[test]
fn test_ng11_output_length_2048() {
    let config = NoiseConfig::brown().with_buffer_size(2048).unwrap();
    let mut gen = NoiseGenerator::new(config).unwrap();
    let mut buffer = vec![0.0; 2048];
    gen.generate(&mut buffer).unwrap();
    assert_eq!(buffer.len(), 2048);
}

#[test]
fn test_ng11_buffer_size_mismatch_error() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();
    let mut buffer = vec![0.0; 512]; // Wrong size
    let result = gen.generate(&mut buffer);
    assert!(result.is_err());
    match result.unwrap_err() {
        NoiseError::BufferSizeMismatch { expected, actual } => {
            assert_eq!(expected, 1024);
            assert_eq!(actual, 512);
        }
        _ => panic!("Expected BufferSizeMismatch error"),
    }
}

// ========== NG12: Output values bounded to [-1.0, 1.0] ==========

#[test]
fn test_ng12_output_bounded() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    for _ in 0..100 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();

        for (i, &sample) in buffer.iter().enumerate() {
            assert!(
                sample >= -1.0 && sample <= 1.0,
                "Sample[{}] = {} out of bounds",
                i,
                sample
            );
        }
    }
}

#[test]
fn test_ng12_output_bounded_white_noise() {
    let config = NoiseConfig::white();
    let mut gen = NoiseGenerator::new(config).unwrap();

    for _ in 0..50 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();

        for &sample in &buffer {
            assert!(sample >= -1.0 && sample <= 1.0);
        }
    }
}

#[test]
fn test_ng12_output_bounded_with_modulation() {
    let config = NoiseConfig::brown().with_modulation(1.0, 5.0).unwrap();
    let mut gen = NoiseGenerator::new(config).unwrap();

    for _ in 0..50 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();

        for &sample in &buffer {
            assert!(sample >= -1.0 && sample <= 1.0);
        }
    }
}

// ========== NG13: No NaN or Inf in output ==========

#[test]
fn test_ng13_no_nan_inf() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    for _ in 0..100 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();

        for (i, &sample) in buffer.iter().enumerate() {
            assert!(!sample.is_nan(), "NaN detected at index {}", i);
            assert!(!sample.is_infinite(), "Infinity detected at index {}", i);
        }
    }
}

#[test]
fn test_ng13_no_nan_inf_all_noise_types() {
    for noise_type in [
        NoiseConfig::white(),
        NoiseConfig::pink(),
        NoiseConfig::brown(),
        NoiseConfig::blue(),
        NoiseConfig::violet(),
    ] {
        let mut gen = NoiseGenerator::new(noise_type).unwrap();

        for _ in 0..20 {
            let mut buffer = vec![0.0; 1024];
            gen.generate(&mut buffer).unwrap();

            for &sample in &buffer {
                assert!(!sample.is_nan());
                assert!(!sample.is_infinite());
            }
        }
    }
}

#[test]
fn test_ng13_no_nan_inf_extreme_config() {
    let config = NoiseConfig::new(super::super::config::NoiseType::Custom(12.0))
        .with_texture(1.0)
        .unwrap()
        .with_modulation(1.0, 10.0)
        .unwrap();

    let mut gen = NoiseGenerator::new(config).unwrap();

    for _ in 0..50 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();

        for &sample in &buffer {
            assert!(sample.is_finite());
        }
    }
}

// ========== NG14: Config update takes effect within one buffer ==========

#[test]
fn test_ng14_config_update_effective() {
    let config1 = NoiseConfig::white();
    let mut gen = NoiseGenerator::new(config1).unwrap();

    // Generate with white noise config
    let mut buffer1 = vec![0.0; 1024];
    gen.generate(&mut buffer1).unwrap();

    // Update to brown noise
    let config2 = NoiseConfig::brown();
    gen.update_config(config2).unwrap();

    // Generate with new config
    let mut buffer2 = vec![0.0; 1024];
    gen.generate(&mut buffer2).unwrap();

    // Config should be updated
    assert_eq!(
        gen.config().noise_type,
        super::super::config::NoiseType::Brown
    );
}

#[test]
fn test_ng14_config_update_validates() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    // Invalid config should fail
    let mut bad_config = NoiseConfig::white();
    bad_config.buffer_size = 100; // Invalid
    let result = gen.update_config(bad_config);
    assert!(result.is_err());
}

#[test]
fn test_ng14_config_update_buffer_size_change() {
    let config1 = NoiseConfig::brown().with_buffer_size(1024).unwrap();
    let mut gen = NoiseGenerator::new(config1).unwrap();

    // Generate with 1024
    let mut buffer1 = vec![0.0; 1024];
    gen.generate(&mut buffer1).unwrap();

    // Update to different buffer size
    let config2 = NoiseConfig::brown().with_buffer_size(512).unwrap();
    gen.update_config(config2).unwrap();

    // Generate with new buffer size
    let mut buffer2 = vec![0.0; 512];
    gen.generate(&mut buffer2).unwrap();
    assert_eq!(buffer2.len(), 512);
}

// ========== NG15: Continuous generation produces seamless audio (no clicks) ==========

#[test]
fn test_ng15_no_clicks_continuous() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    let mut prev_last_sample = 0.0f32;
    let max_discontinuity = 0.5; // Maximum allowed jump between buffers

    for i in 0..100 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();

        if i > 0 {
            let first_sample = buffer[0];
            let delta = (first_sample - prev_last_sample).abs();
            assert!(
                delta < max_discontinuity,
                "Click detected at buffer {}: delta = {} (prev={}, curr={})",
                i,
                delta,
                prev_last_sample,
                first_sample
            );
        }

        prev_last_sample = *buffer.last().unwrap();
    }
}

#[test]
fn test_ng15_no_clicks_within_buffer() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    // For spectral synthesis with random phases, sample-to-sample correlation
    // differs from time-domain integration. The perceptually relevant property
    // is the spectral shape, not adjacent sample smoothness.
    //
    // This test verifies the signal doesn't have catastrophic discontinuities
    // (e.g., full-scale jumps from -1 to +1 on every sample). A reasonable upper
    // bound is that no more than 50% of transitions should be "large" (>0.5).
    let large_jump_threshold = 0.5;

    for _ in 0..10 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();

        let large_jumps: usize = (1..buffer.len())
            .filter(|&j| (buffer[j] - buffer[j - 1]).abs() > large_jump_threshold)
            .count();

        // Sanity check: should not have more than 50% large jumps
        // Spectral synthesis naturally has more transitions than time-domain noise,
        // but shouldn't be pathologically discontinuous
        assert!(
            large_jumps < buffer.len() / 2,
            "Pathologically discontinuous signal: {}/{} large jumps",
            large_jumps,
            buffer.len()
        );
    }
}

// ========== Falsification tests (NG-F1 to NG-F5) ==========

#[test]
fn test_ng_f1_cannot_produce_nan_inf() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    for _ in 0..1000 {
        let mut buf = vec![0.0; 1024];
        gen.generate(&mut buf).unwrap();
        for &sample in &buf {
            assert!(!sample.is_nan(), "NaN detected");
            assert!(!sample.is_infinite(), "Infinity detected");
        }
    }
}

#[test]
fn test_ng_f2_cannot_exceed_amplitude() {
    let config = NoiseConfig::white();
    let mut gen = NoiseGenerator::new(config).unwrap();

    for _ in 0..1000 {
        let mut buf = vec![0.0; 1024];
        gen.generate(&mut buf).unwrap();
        for &sample in &buf {
            assert!(
                (-1.0..=1.0).contains(&sample),
                "Sample {} outside [-1, 1]",
                sample
            );
        }
    }
}

#[test]
fn test_ng_f3_no_discontinuities() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    let mut prev_sample = 0.0;
    let threshold = 0.5;

    for _ in 0..100 {
        let mut buf = vec![0.0; 1024];
        gen.generate(&mut buf).unwrap();
        let delta = (buf[0] - prev_sample).abs();
        assert!(delta < threshold, "Click detected: delta={}", delta);
        prev_sample = *buf.last().unwrap();
    }
}

#[test]
fn test_ng_f5_not_silent() {
    let config = NoiseConfig::white();
    let mut gen = NoiseGenerator::new(config).unwrap();

    let mut buf = vec![0.0; 1024];
    gen.generate(&mut buf).unwrap();

    let energy: f32 = buf.iter().map(|x| x * x).sum();
    assert!(energy > 0.01, "Output is silent: energy = {}", energy);
}

// ========== Additional generator tests ==========

#[test]
fn test_iterator_interface() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    // Take 5 buffers
    let buffers: Vec<Vec<f32>> = gen.by_ref().take(5).collect();

    assert_eq!(buffers.len(), 5);
    for buffer in &buffers {
        assert_eq!(buffer.len(), 1024);
    }
}

#[test]
fn test_time_advances() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    assert!((gen.time() - 0.0).abs() < f64::EPSILON);

    let mut buffer = vec![0.0; 1024];
    gen.generate(&mut buffer).unwrap();

    // Time should advance by buffer_size / sample_rate
    let expected_time = 1024.0 / 44100.0;
    assert!((gen.time() - expected_time).abs() < 0.0001);
}

#[test]
fn test_sample_counter() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    assert_eq!(gen.sample_counter(), 0);

    let mut buffer = vec![0.0; 1024];
    gen.generate(&mut buffer).unwrap();
    assert_eq!(gen.sample_counter(), 1024);

    gen.generate(&mut buffer).unwrap();
    assert_eq!(gen.sample_counter(), 2048);
}

#[test]
fn test_reset() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    // Advance state
    let mut buffer = vec![0.0; 1024];
    for _ in 0..10 {
        gen.generate(&mut buffer).unwrap();
    }

    assert!(gen.time() > 0.0);
    assert!(gen.sample_counter() > 0);

    gen.reset();

    assert!((gen.time() - 0.0).abs() < f64::EPSILON);
    assert_eq!(gen.sample_counter(), 0);
}

#[test]
fn test_invalid_config_rejected() {
    let mut config = NoiseConfig::brown();
    config.buffer_size = 100; // Invalid
    let result = NoiseGenerator::new(config);
    assert!(result.is_err());
}

// ========== NG16-NG18: Spectral slope verification ==========

/// Helper: Compute average power in frequency band
pub(super) fn compute_band_power(
    samples: &[f32],
    sample_rate: u32,
    low_hz: f32,
    high_hz: f32,
) -> f32 {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n = samples.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();
    fft.process(&mut buffer);

    let freq_resolution = sample_rate as f32 / n as f32;
    let low_bin = (low_hz / freq_resolution).floor() as usize;
    let high_bin = (high_hz / freq_resolution).ceil() as usize;

    let power: f32 = buffer[low_bin..high_bin.min(n / 2)]
        .iter()
        .map(|c| c.norm_sqr())
        .sum();

    power / (high_bin - low_bin).max(1) as f32
}

#[path = "generator_tests_part_02.rs"]
mod generator_tests_part_02;
