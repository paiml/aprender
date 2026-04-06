//! Tests for LoRA adapter serialization

use super::*;
use crate::lora::LoRALayer;
use crate::Tensor;
use approx::assert_abs_diff_eq;
use proptest::prelude::*;
use std::fs;

// ========================================================================
// PROPERTY TESTS - Serialization correctness validation
// ========================================================================

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(100))]

    /// Round-trip serialization should preserve all data exactly
    #[test]
    fn prop_round_trip_preserves_data(
        d_out in 2usize..16,
        d_in in 2usize..16,
        rank in 1usize..4,
        alpha in 1.0f32..32.0,
    ) {
        let size = d_out * d_in;
        let base_weight = Tensor::from_vec(vec![1.0; size], false);
        let layer = LoRALayer::new(base_weight.clone(), d_out, d_in, rank, alpha);

        // Create adapter and save
        let adapter = LoRAAdapter::from_layer(&layer, rank, alpha);
        let path = format!("/tmp/prop_test_adapter_{d_out}_{d_in}_{rank}.json");
        adapter.save(&path).expect("save should succeed");

        // Load and reconstruct
        let loaded = LoRAAdapter::load(&path).expect("load should succeed");
        let loaded_layer = loaded.to_layer(base_weight).expect("load should succeed");

        // Verify all fields preserved
        prop_assert_eq!(loaded_layer.d_out(), d_out);
        prop_assert_eq!(loaded_layer.d_in(), d_in);
        prop_assert_eq!(loaded_layer.rank(), rank);
        prop_assert!((loaded_layer.scale() - alpha / rank as f32).abs() < 1e-5);

        // Cleanup
        fs::remove_file(&path).ok();
    }

    /// Metadata calculation should be correct for any dimensions
    #[test]
    fn prop_metadata_correct(
        d_out in 2usize..32,
        d_in in 2usize..32,
        rank in 1usize..8,
        alpha in 1.0f32..64.0,
    ) {
        let size = d_out * d_in;
        let base_weight = Tensor::from_vec(vec![1.0; size], false);
        let layer = LoRALayer::new(base_weight, d_out, d_in, rank, alpha);

        let adapter = LoRAAdapter::from_layer(&layer, rank, alpha);
        let metadata = adapter.metadata();

        // Verify metadata
        prop_assert_eq!(metadata.d_out, d_out);
        prop_assert_eq!(metadata.d_in, d_in);
        prop_assert_eq!(metadata.rank, rank);
        prop_assert!((metadata.alpha - alpha).abs() < 1e-6);
        prop_assert_eq!(metadata.num_params, rank * d_in + d_out * rank);
        prop_assert_eq!(metadata.version, "1.0");
    }

    /// Forward output should be identical after save/load
    #[test]
    fn prop_forward_invariant_after_save_load(
        d in 4usize..12,
        rank in 1usize..3,
    ) {
        let size = d * d;
        let base_weight = Tensor::from_vec(vec![0.5; size], false);
        let layer = LoRALayer::new(base_weight.clone(), d, d, rank, 4.0);

        let x = Tensor::from_vec(vec![0.1; d], true);
        let original_output = layer.forward(&x);

        // Save and load
        let path = format!("/tmp/prop_forward_test_{d}_{rank}.json");
        save_adapter(&layer, rank, 4.0, &path).expect("save should succeed");
        let loaded_layer = load_adapter(base_weight, &path).expect("load should succeed");
        let loaded_output = loaded_layer.forward(&x);

        // Forward outputs must match
        prop_assert_eq!(original_output.len(), loaded_output.len());
        for i in 0..original_output.len() {
            prop_assert!(
                (original_output.data()[i] - loaded_output.data()[i]).abs() < 1e-5,
                "Forward output mismatch at index {}", i
            );
        }

        fs::remove_file(&path).ok();
    }

    /// Invalid dimension should be caught during to_layer
    #[test]
    fn prop_dimension_validation_catches_mismatches(
        d_out in 2usize..8,
        d_in in 2usize..8,
        rank in 1usize..3,
    ) {
        let size = d_out * d_in;
        let base_weight = Tensor::from_vec(vec![1.0; size], false);
        let layer = LoRALayer::new(base_weight, d_out, d_in, rank, 4.0);

        let adapter = LoRAAdapter::from_layer(&layer, rank, 4.0);

        // Try loading with wrong size base weight
        let wrong_size = size + 1;
        let wrong_base = Tensor::from_vec(vec![1.0; wrong_size], false);
        let result = adapter.to_layer(wrong_base);

        // Should fail with dimension error
        prop_assert!(result.is_err());
    }
}

// ========================================================================
// UNIT TESTS
// ========================================================================

#[test]
fn test_adapter_serialization_round_trip() {
    // Create LoRA layer
    let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let mut layer = LoRALayer::new(base_weight.clone(), 2, 2, 2, 4.0);

    // Set known weights
    *layer.lora_a_mut().data_mut() = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
    *layer.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.6, 0.7, 0.8]);

    // Save adapter
    let path = "/tmp/test_adapter.json";
    let adapter = LoRAAdapter::from_layer(&layer, 2, 4.0);
    adapter.save(path).expect("save should succeed");

    // Load adapter
    let loaded_adapter = LoRAAdapter::load(path).expect("load should succeed");
    let loaded_layer = loaded_adapter.to_layer(base_weight).expect("load should succeed");

    // Verify weights match
    for (&orig, &loaded) in layer.lora_a().data().iter().zip(loaded_layer.lora_a().data().iter()) {
        assert_abs_diff_eq!(orig, loaded, epsilon = 1e-6);
    }

    for (&orig, &loaded) in layer.lora_b().data().iter().zip(loaded_layer.lora_b().data().iter()) {
        assert_abs_diff_eq!(orig, loaded, epsilon = 1e-6);
    }

    // Verify metadata
    assert_eq!(loaded_layer.rank(), 2);
    assert_eq!(loaded_layer.d_out(), 2);
    assert_eq!(loaded_layer.d_in(), 2);
    assert_abs_diff_eq!(loaded_layer.scale(), 2.0, epsilon = 1e-6); // 4.0 / 2

    // Cleanup
    fs::remove_file(path).expect("operation should succeed");
}

#[test]
fn test_adapter_forward_consistency() {
    // Test that forward pass gives same result after save/load
    let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
    let mut layer = LoRALayer::new(base_weight.clone(), 2, 2, 1, 1.0);

    *layer.lora_a_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
    *layer.lora_b_mut().data_mut() = ndarray::arr1(&[0.3, 0.3]);

    let x = Tensor::from_vec(vec![2.0, 3.0], true);
    let output_original = layer.forward(&x);

    // Save and load
    let path = "/tmp/test_adapter_forward.json";
    save_adapter(&layer, 1, 1.0, path).expect("save should succeed");
    let loaded_layer = load_adapter(base_weight, path).expect("load should succeed");

    let output_loaded = loaded_layer.forward(&x);

    // Verify outputs match
    assert_eq!(output_original.len(), output_loaded.len());
    for i in 0..output_original.len() {
        assert_abs_diff_eq!(output_original.data()[i], output_loaded.data()[i], epsilon = 1e-5);
    }

    // Cleanup
    fs::remove_file(path).expect("operation should succeed");
}

#[test]
fn test_adapter_dimension_validation() {
    let base_weight_2x2 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let layer = LoRALayer::new(base_weight_2x2, 2, 2, 2, 4.0);

    let adapter = LoRAAdapter::from_layer(&layer, 2, 4.0);

    // Try to load with wrong size base weight
    let wrong_base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false); // 3x2
    let result = adapter.to_layer(wrong_base);

    assert!(result.is_err());
    match result {
        Err(AdapterError::DimensionMismatch { .. }) => {}
        _ => panic!("Expected DimensionMismatch error"),
    }
}

#[test]
fn test_adapter_metadata() {
    let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
    let layer = LoRALayer::new(base_weight, 3, 2, 2, 8.0);

    let adapter = LoRAAdapter::from_layer(&layer, 2, 8.0);
    let metadata = adapter.metadata();

    assert_eq!(metadata.rank, 2);
    assert_abs_diff_eq!(metadata.alpha, 8.0, epsilon = 1e-6);
    assert_eq!(metadata.d_out, 3);
    assert_eq!(metadata.d_in, 2);
    assert_abs_diff_eq!(metadata.scale, 4.0, epsilon = 1e-6); // 8.0 / 2
    assert_eq!(metadata.num_params, 4 + 6); // A: 2*2=4, B: 3*2=6
    assert_eq!(metadata.version, "1.0");
}

#[test]
fn test_multiple_adapters_same_base() {
    // Test loading multiple different adapters on the same base weight
    let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);

    // Create and save adapter 1
    let mut layer1 = LoRALayer::new(base_weight.clone(), 2, 2, 1, 1.0);
    *layer1.lora_a_mut().data_mut() = ndarray::arr1(&[1.0, 1.0]);
    *layer1.lora_b_mut().data_mut() = ndarray::arr1(&[1.0, 1.0]);
    save_adapter(&layer1, 1, 1.0, "/tmp/adapter1.json").expect("save should succeed");

    // Create and save adapter 2
    let mut layer2 = LoRALayer::new(base_weight.clone(), 2, 2, 1, 1.0);
    *layer2.lora_a_mut().data_mut() = ndarray::arr1(&[2.0, 2.0]);
    *layer2.lora_b_mut().data_mut() = ndarray::arr1(&[2.0, 2.0]);
    save_adapter(&layer2, 1, 1.0, "/tmp/adapter2.json").expect("save should succeed");

    // Load both adapters
    let loaded1 =
        load_adapter(base_weight.clone(), "/tmp/adapter1.json").expect("load should succeed");
    let loaded2 = load_adapter(base_weight, "/tmp/adapter2.json").expect("load should succeed");

    // Verify they're different
    assert_abs_diff_eq!(loaded1.lora_a().data()[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(loaded2.lora_a().data()[0], 2.0, epsilon = 1e-6);

    // Cleanup
    fs::remove_file("/tmp/adapter1.json").expect("operation should succeed");
    fs::remove_file("/tmp/adapter2.json").expect("operation should succeed");
}

#[test]
fn test_adapter_file_format_readable() {
    // Verify JSON is human-readable
    let base_weight = Tensor::from_vec(vec![1.0, 2.0], false);
    let layer = LoRALayer::new(base_weight, 2, 1, 1, 2.0);

    let path = "/tmp/test_readable.json";
    save_adapter(&layer, 1, 2.0, path).expect("save should succeed");

    // Read raw file content
    let content = fs::read_to_string(path).expect("file read should succeed");

    // Should contain key fields
    assert!(content.contains("\"version\""));
    assert!(content.contains("\"rank\""));
    assert!(content.contains("\"alpha\""));
    assert!(content.contains("\"lora_a\""));
    assert!(content.contains("\"lora_b\""));

    // Cleanup
    fs::remove_file(path).expect("operation should succeed");
}
