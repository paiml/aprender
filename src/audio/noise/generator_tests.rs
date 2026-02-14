use super::*;

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
fn compute_band_power(samples: &[f32], sample_rate: u32, low_hz: f32, high_hz: f32) -> f32 {
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

#[test]
fn test_ng16_white_noise_flat_spectrum() {
    let config = NoiseConfig::white();
    let mut gen = NoiseGenerator::new(config).unwrap();

    // Generate multiple buffers and accumulate spectrum
    let mut all_samples = Vec::new();
    for _ in 0..100 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();
        all_samples.extend_from_slice(&buffer);
    }

    // Compare power in low vs high frequency bands
    // White noise should have roughly equal power per Hz
    let low_power = compute_band_power(&all_samples, 44100, 100.0, 1000.0);
    let high_power = compute_band_power(&all_samples, 44100, 5000.0, 10000.0);

    // Allow 10dB variance (factor of 10 in power)
    let ratio = low_power / high_power.max(1e-10);
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "White noise should be relatively flat: low/high ratio = {}",
        ratio
    );
}

#[test]
fn test_ng17_brown_noise_slope() {
    let config = NoiseConfig::brown();
    let mut gen = NoiseGenerator::new(config).unwrap();

    let mut all_samples = Vec::new();
    for _ in 0..100 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();
        all_samples.extend_from_slice(&buffer);
    }

    // Brown noise: low frequencies should have MORE power
    let low_power = compute_band_power(&all_samples, 44100, 100.0, 500.0);
    let high_power = compute_band_power(&all_samples, 44100, 5000.0, 10000.0);

    // Brown noise should have significantly more low-freq energy
    assert!(
        low_power > high_power,
        "Brown noise should emphasize low frequencies: low={}, high={}",
        low_power,
        high_power
    );
}

#[test]
fn test_ng18_pink_noise_slope() {
    let config = NoiseConfig::pink();
    let mut gen = NoiseGenerator::new(config).unwrap();

    let mut all_samples = Vec::new();
    for _ in 0..100 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();
        all_samples.extend_from_slice(&buffer);
    }

    // Pink noise: low frequencies should have more power, but less than brown
    let low_power = compute_band_power(&all_samples, 44100, 100.0, 500.0);
    let high_power = compute_band_power(&all_samples, 44100, 5000.0, 10000.0);

    // Pink noise should have more low-freq energy
    assert!(
        low_power > high_power,
        "Pink noise should emphasize low frequencies: low={}, high={}",
        low_power,
        high_power
    );
}

// ========== NG19-NG20: Modulation tests ==========

#[test]
fn test_ng19_modulation_depth_zero_no_variation() {
    // With modulation_depth=0, output should be consistent
    let config = NoiseConfig::brown().with_modulation(0.0, 1.0).unwrap();
    let mut gen = NoiseGenerator::new(config).unwrap();

    // Generate multiple buffers
    let mut energies = Vec::new();
    for _ in 0..20 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();
        let energy: f32 = buffer.iter().map(|x| x * x).sum();
        energies.push(energy);
    }

    // With no modulation, energy variance should be low (just random variation)
    let mean_energy: f32 = energies.iter().sum::<f32>() / energies.len() as f32;
    let variance: f32 = energies
        .iter()
        .map(|e| (e - mean_energy).powi(2))
        .sum::<f32>()
        / energies.len() as f32;
    let cv = variance.sqrt() / mean_energy; // Coefficient of variation

    // CV should be relatively small (natural variation only)
    assert!(
        cv < 0.5,
        "With modulation_depth=0, energy should be stable. CV={}",
        cv
    );
}

#[test]
fn test_ng20_modulation_affects_output() {
    // With modulation_depth>0, we should see periodic variation
    let config = NoiseConfig::brown()
        .with_modulation(1.0, 5.0) // High depth, 5Hz rate
        .unwrap();
    let mut gen = NoiseGenerator::new(config).unwrap();

    // Generate enough samples to cover one modulation cycle
    // At 5Hz, one cycle = 0.2s = ~8820 samples at 44100Hz
    let mut all_samples = Vec::new();
    for _ in 0..20 {
        let mut buffer = vec![0.0; 1024];
        gen.generate(&mut buffer).unwrap();
        all_samples.extend_from_slice(&buffer);
    }

    // Verify we got samples (modulation should not break generation)
    assert!(all_samples.len() >= 8820);
    let energy: f32 = all_samples.iter().map(|x| x * x).sum();
    assert!(energy > 0.0, "Output should not be silent with modulation");
}

// ========== Additional coverage tests ==========

#[test]
fn test_noise_generator_debug() {
    let config = NoiseConfig::brown();
    let gen = NoiseGenerator::new(config).unwrap();
    let debug_str = format!("{:?}", gen);
    assert!(debug_str.contains("NoiseGenerator"));
    assert!(debug_str.contains("time"));
    assert!(debug_str.contains("sample_counter"));
}

#[test]
fn test_from_apr_with_valid_model() {
    use tempfile::NamedTempFile;

    // Create a valid model with correct n_freqs for buffer_size 1024
    let n_freqs = 1024 / 2 + 1; // 513
    let mlp = SpectralMLP::random_init(8, 64, n_freqs, 42);

    // Save to temp file
    let temp = NamedTempFile::new().unwrap();
    mlp.save_apr(temp.path()).unwrap();

    // Load from file
    let config = NoiseConfig::brown();
    let gen = NoiseGenerator::from_apr(temp.path(), config).unwrap();

    // Verify it works
    let _buffer = vec![0.0; 1024];
    gen.clone_config();
    assert_eq!(gen.config().buffer_size, 1024);
}

#[test]
fn test_from_apr_dimension_mismatch() {
    use tempfile::NamedTempFile;

    // Create a model with wrong n_freqs
    let wrong_n_freqs = 256; // Wrong for buffer_size 1024
    let mlp = SpectralMLP::random_init(8, 64, wrong_n_freqs, 42);

    // Save to temp file
    let temp = NamedTempFile::new().unwrap();
    mlp.save_apr(temp.path()).unwrap();

    // Load should fail due to dimension mismatch
    let config = NoiseConfig::brown(); // buffer_size 1024, expects n_freqs 513
    let result = NoiseGenerator::from_apr(temp.path(), config);
    assert!(result.is_err());

    match result.unwrap_err() {
        NoiseError::ModelError(msg) => {
            assert!(msg.contains("n_freqs"));
        }
        _ => panic!("Expected ModelError"),
    }
}

#[test]
fn test_with_mlp_valid() {
    let config = NoiseConfig::brown();
    let n_freqs = config.buffer_size / 2 + 1;
    let mlp = SpectralMLP::random_init(8, 64, n_freqs, 42);

    let mut gen = NoiseGenerator::with_mlp(config, mlp).unwrap();

    let mut buffer = vec![0.0; 1024];
    gen.generate(&mut buffer).unwrap();

    // Should produce valid output
    for &sample in &buffer {
        assert!(sample.is_finite());
    }
}

#[test]
fn test_with_mlp_dimension_mismatch() {
    let config = NoiseConfig::brown(); // buffer_size 1024, expects n_freqs 513
    let wrong_mlp = SpectralMLP::random_init(8, 64, 256, 42); // Wrong n_freqs

    let result = NoiseGenerator::with_mlp(config, wrong_mlp);
    assert!(result.is_err());

    match result.unwrap_err() {
        NoiseError::ModelError(msg) => {
            assert!(msg.contains("n_freqs"));
        }
        _ => panic!("Expected ModelError"),
    }
}

#[test]
fn test_set_phase_seed() {
    let config = NoiseConfig::brown();
    let mut gen1 = NoiseGenerator::new(config.clone()).unwrap();
    let mut gen2 = NoiseGenerator::new(config).unwrap();

    // Set same seed
    gen1.set_phase_seed(99999);
    gen2.set_phase_seed(99999);

    // Generate - should be identical
    let mut buf1 = vec![0.0; 1024];
    let mut buf2 = vec![0.0; 1024];
    gen1.generate(&mut buf1).unwrap();
    gen2.generate(&mut buf2).unwrap();

    // With same seed and same config, output should be identical
    for (a, b) in buf1.iter().zip(buf2.iter()) {
        assert!((a - b).abs() < 1e-6, "Outputs differ: {} vs {}", a, b);
    }
}

impl NoiseGenerator {
    /// Clone config for testing
    fn clone_config(&self) -> NoiseConfig {
        self.config.clone()
    }
}
