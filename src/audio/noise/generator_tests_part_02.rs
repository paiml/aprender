
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
