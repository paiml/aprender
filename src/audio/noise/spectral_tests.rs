pub(crate) use super::*;

// ========== NG4: Forward pass is deterministic ==========

#[test]
fn test_ng4_forward_deterministic() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);
    let config = vec![0.5, 0.3, 0.0, 0.1, 0.0, 1.0, 0.5, 0.9];

    let output1 = model.forward(&config);
    let output2 = model.forward(&config);

    assert_eq!(output1.len(), output2.len());
    for (a, b) in output1.iter().zip(output2.iter()) {
        assert!(
            (a - b).abs() < f32::EPSILON,
            "Forward pass not deterministic: {} != {}",
            a,
            b
        );
    }
}

#[test]
fn test_ng4_forward_deterministic_multiple_calls() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);

    for i in 0..10 {
        let config = vec![i as f32 / 10.0; 8];
        let output1 = model.forward(&config);
        let output2 = model.forward(&config);

        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }
}

#[test]
fn test_ng4_same_seed_same_model() {
    let model1 = SpectralMLP::random_init(8, 64, 513, 42);
    let model2 = SpectralMLP::random_init(8, 64, 513, 42);

    let config = vec![0.5; 8];
    let output1 = model1.forward(&config);
    let output2 = model2.forward(&config);

    for (a, b) in output1.iter().zip(output2.iter()) {
        assert!((a - b).abs() < f32::EPSILON);
    }
}

#[test]
fn test_ng4_different_seed_different_output() {
    let model1 = SpectralMLP::random_init(8, 64, 513, 42);
    let model2 = SpectralMLP::random_init(8, 64, 513, 43);

    let config = vec![0.5; 8];
    let output1 = model1.forward(&config);
    let output2 = model2.forward(&config);

    // At least some values should differ
    let mut all_same = true;
    for (a, b) in output1.iter().zip(output2.iter()) {
        if (a - b).abs() > f32::EPSILON {
            all_same = false;
            break;
        }
    }
    assert!(
        !all_same,
        "Different seeds should produce different outputs"
    );
}

// ========== NG5: Output dimensions match n_freqs ==========

#[test]
fn test_ng5_output_dimensions_513() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);
    let config = vec![0.5; 8];
    let output = model.forward(&config);
    assert_eq!(output.len(), 513, "Output should have 513 frequency bins");
}

#[test]
fn test_ng5_output_dimensions_256() {
    let model = SpectralMLP::random_init(8, 32, 256, 42);
    let config = vec![0.5; 8];
    let output = model.forward(&config);
    assert_eq!(output.len(), 256);
}

#[test]
fn test_ng5_output_dimensions_1024() {
    let model = SpectralMLP::random_init(8, 64, 1024, 42);
    let config = vec![0.5; 8];
    let output = model.forward(&config);
    assert_eq!(output.len(), 1024);
}

#[test]
fn test_ng5_config_dim_accessor() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);
    assert_eq!(model.config_dim(), 8);
}

#[test]
fn test_ng5_hidden_dim_accessor() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);
    assert_eq!(model.hidden_dim(), 64);
}

#[test]
fn test_ng5_n_freqs_accessor() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);
    assert_eq!(model.n_freqs(), 513);
}

// ========== NG6: All outputs are non-negative (magnitudes) ==========

#[test]
fn test_ng6_outputs_non_negative() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);

    // Test with various inputs
    for i in 0..100 {
        let config: Vec<f32> = (0..8)
            .map(|j| ((i + j) as f32 / 100.0) * 2.0 - 1.0)
            .collect();

        let output = model.forward(&config);

        for (idx, &val) in output.iter().enumerate() {
            assert!(
                val >= 0.0,
                "Output[{}] = {} is negative (input {})",
                idx,
                val,
                i
            );
        }
    }
}

#[test]
fn test_ng6_outputs_non_negative_negative_inputs() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);

    // All negative inputs
    let config = vec![-1.0, -0.5, -0.8, -0.3, -1.0, -0.2, -0.7, -0.9];
    let output = model.forward(&config);

    for &val in &output {
        assert!(
            val >= 0.0,
            "Output should be non-negative even with negative inputs"
        );
    }
}

#[test]
fn test_ng6_outputs_non_negative_zero_inputs() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);

    let config = vec![0.0; 8];
    let output = model.forward(&config);

    for &val in &output {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_ng6_softplus_properties() {
    // Softplus should be positive for any input
    assert!(softplus(-100.0) >= 0.0);
    assert!(softplus(0.0) > 0.0);
    assert!(softplus(100.0) > 0.0);

    // Softplus(0) ≈ ln(2)
    assert!((softplus(0.0) - 2.0_f32.ln()).abs() < 0.001);
}

// ========== NG7: APR round-trip preserves weights exactly ==========

#[test]
fn test_ng7_apr_roundtrip() {
    let original = SpectralMLP::random_init(8, 64, 513, 42);

    // Save to temp file
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_ng7_spectral_mlp.apr");

    original.save_apr(&path).expect("Failed to save");
    let loaded = SpectralMLP::load_apr(&path).expect("Failed to load");

    // Clean up
    std::fs::remove_file(&path).ok();

    // Compare dimensions
    assert_eq!(original.config_dim, loaded.config_dim);
    assert_eq!(original.hidden_dim, loaded.hidden_dim);
    assert_eq!(original.n_freqs, loaded.n_freqs);

    // Compare weights exactly
    assert_eq!(original.weights_1, loaded.weights_1);
    assert_eq!(original.bias_1, loaded.bias_1);
    assert_eq!(original.weights_2, loaded.weights_2);
    assert_eq!(original.bias_2, loaded.bias_2);
    assert_eq!(original.weights_3, loaded.weights_3);
    assert_eq!(original.bias_3, loaded.bias_3);
}

#[test]
fn test_ng7_apr_roundtrip_output_identical() {
    let original = SpectralMLP::random_init(8, 64, 513, 42);

    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("test_ng7_output_check.apr");

    original.save_apr(&path).expect("Failed to save");
    let loaded = SpectralMLP::load_apr(&path).expect("Failed to load");

    std::fs::remove_file(&path).ok();

    // Test that outputs are identical
    let config = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let output_original = original.forward(&config);
    let output_loaded = loaded.forward(&config);

    for (a, b) in output_original.iter().zip(output_loaded.iter()) {
        assert!(
            (a - b).abs() < f32::EPSILON,
            "Outputs differ after roundtrip"
        );
    }
}

#[test]
fn test_ng7_load_nonexistent_file() {
    let result = SpectralMLP::load_apr("/nonexistent/path/model.apr");
    assert!(result.is_err());
}

// ========== Additional spectral tests ==========

#[test]
fn test_from_weights_valid() {
    let config_dim = 4;
    let hidden_dim = 8;
    let n_freqs = 16;

    let result = SpectralMLP::from_weights(
        vec![0.0; config_dim * hidden_dim],
        vec![0.0; hidden_dim],
        vec![0.0; hidden_dim * hidden_dim],
        vec![0.0; hidden_dim],
        vec![0.0; hidden_dim * n_freqs],
        vec![0.0; n_freqs],
        config_dim,
        hidden_dim,
        n_freqs,
    );

    assert!(result.is_ok());
}

#[test]
fn test_from_weights_invalid_w1() {
    let result = SpectralMLP::from_weights(
        vec![0.0; 10], // Wrong size
        vec![0.0; 8],
        vec![0.0; 64],
        vec![0.0; 8],
        vec![0.0; 128],
        vec![0.0; 16],
        4,
        8,
        16,
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("weights_1"));
}

#[test]
fn test_from_weights_invalid_b1() {
    let result = SpectralMLP::from_weights(
        vec![0.0; 32],
        vec![0.0; 5], // Wrong size
        vec![0.0; 64],
        vec![0.0; 8],
        vec![0.0; 128],
        vec![0.0; 16],
        4,
        8,
        16,
    );

    assert!(result.is_err());
}

#[test]
fn test_num_parameters() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);
    let expected = 8 * 64 + 64 + 64 * 64 + 64 + 64 * 513 + 513;
    assert_eq!(model.num_parameters(), expected);
}

#[test]
fn test_relu() {
    assert!((relu(5.0) - 5.0).abs() < f32::EPSILON);
    assert!((relu(-5.0) - 0.0).abs() < f32::EPSILON);
    assert!((relu(0.0) - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_softplus_numerical_stability() {
    // Very large positive
    assert!((softplus(100.0) - 100.0).abs() < 0.01);
    // Very large negative
    assert!(softplus(-100.0) < 0.01);
    // Normal range
    assert!(softplus(1.0) > 1.0 && softplus(1.0) < 2.0);
}

#[test]
fn test_weights_accessors() {
    let mut model = SpectralMLP::random_init(8, 64, 513, 42);

    // Test immutable access
    let (w1, b1, w2, b2, w3, b3) = model.weights();
    assert_eq!(w1.len(), 8 * 64);
    assert_eq!(b1.len(), 64);
    assert_eq!(w2.len(), 64 * 64);
    assert_eq!(b2.len(), 64);
    assert_eq!(w3.len(), 64 * 513);
    assert_eq!(b3.len(), 513);

    // Test mutable access
    let (w1_mut, _, _, _, _, _) = model.weights_mut();
    w1_mut[0] = 999.0;
    assert!((model.weights_1[0] - 999.0).abs() < f32::EPSILON);
}

#[test]
#[should_panic(expected = "Config dimension mismatch")]
fn test_forward_wrong_config_dim() {
    let model = SpectralMLP::random_init(8, 64, 513, 42);
    let wrong_config = vec![0.5; 4]; // Wrong dimension
    let _ = model.forward(&wrong_config);
}
