// GitHub Issue #5: SafeTensors Model Serialization (Phase 1)
// Tests for SafeTensors export functionality
//
// Acceptance Criteria:
// - [ ] LinearRegression::save_safetensors() exports valid SafeTensors format
// - [ ] SafeTensors header is 8-byte u64 (little-endian metadata length)
// - [ ] JSON metadata includes dtype, shape, data_offsets for each tensor
// - [ ] Coefficients serialized as F32 tensor (little-endian)
// - [ ] Intercept serialized as F32 tensor (little-endian)
// - [ ] Roundtrip: save → load produces identical model

use aprender::linear_model::LinearRegression;
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;
use std::fs;
use std::path::Path;

#[test]
fn test_linear_regression_save_safetensors_creates_file() {
    // RED PHASE: This test should FAIL because save_safetensors() doesn't exist yet

    // Train a simple model (need enough samples: n >= p + 1)
    // Use independent features to avoid collinearity
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Test data should be valid");

    // Save to SafeTensors format
    let path = "test_model.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");

    // Verify file was created
    assert!(
        Path::new(path).exists(),
        "SafeTensors file should be created"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_safetensors_header_format() {
    // RED PHASE: Verify SafeTensors header is 8-byte u64 little-endian

    let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Test data should be valid");
    let y = Vector::from_vec(vec![3.0, 4.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Test data should be valid");

    let path = "test_header.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");

    // Read first 8 bytes (header)
    let bytes = fs::read(path).expect("Test data should be valid");
    assert!(bytes.len() >= 8, "File must be at least 8 bytes");

    // First 8 bytes should be u64 little-endian (metadata length)
    let header_bytes: [u8; 8] = bytes[0..8].try_into().expect("Test data should be valid");
    let metadata_len = u64::from_le_bytes(header_bytes);

    // Metadata length should be reasonable (not zero, not huge)
    assert!(metadata_len > 0, "Metadata length must be > 0");
    assert!(metadata_len < 10000, "Metadata length should be reasonable");

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
#[allow(clippy::disallowed_methods)] // serde_json::json! macro uses unwrap internally
fn test_safetensors_json_metadata_structure() {
    // RED PHASE: Verify JSON metadata has correct structure

    // Need at least 3 samples for 2 features (n >= p + 1)
    let x = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Test data should be valid");

    let path = "test_metadata.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");

    let bytes = fs::read(path).expect("Test data should be valid");

    // Extract metadata
    let header_bytes: [u8; 8] = bytes[0..8].try_into().expect("Test data should be valid");
    let metadata_len = u64::from_le_bytes(header_bytes) as usize;
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_json).expect("Test data should be valid");

    // Parse JSON
    let metadata: serde_json::Value =
        serde_json::from_str(metadata_str).expect("Test data should be valid");

    // Verify "coefficients" tensor metadata
    assert!(
        metadata.get("coefficients").is_some(),
        "Must have 'coefficients' tensor"
    );
    let coeff_meta = &metadata["coefficients"];
    assert_eq!(coeff_meta["dtype"], "F32", "Coefficients must be F32");
    assert!(coeff_meta.get("shape").is_some(), "Must have shape");
    assert!(
        coeff_meta.get("data_offsets").is_some(),
        "Must have data_offsets"
    );

    // Verify "intercept" tensor metadata
    assert!(
        metadata.get("intercept").is_some(),
        "Must have 'intercept' tensor"
    );
    let intercept_meta = &metadata["intercept"];
    assert_eq!(intercept_meta["dtype"], "F32", "Intercept must be F32");
    assert_eq!(
        intercept_meta["shape"],
        serde_json::json!([1]),
        "Intercept shape must be [1]"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_safetensors_coefficients_serialization() {
    // RED PHASE: Verify coefficients are serialized correctly as F32 little-endian

    // Need at least 3 samples for 2 features (n >= p + 1)
    // Use independent features
    let x = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![2.0, 3.0, 5.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Test data should be valid");

    let path = "test_coeffs.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");

    let bytes = fs::read(path).expect("Test data should be valid");

    // Extract metadata to find data offsets
    let header_bytes: [u8; 8] = bytes[0..8].try_into().expect("Test data should be valid");
    let metadata_len = u64::from_le_bytes(header_bytes) as usize;
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(metadata_json).expect("Test data should be valid"),
    )
    .expect("Test data should be valid");

    // Get coefficients data offsets
    let offsets = metadata["coefficients"]["data_offsets"]
        .as_array()
        .expect("Test data should be valid");
    let start = offsets[0].as_u64().expect("Test data should be valid") as usize + 8 + metadata_len;
    let end = offsets[1].as_u64().expect("Test data should be valid") as usize + 8 + metadata_len;

    // Read coefficient bytes
    let coeff_bytes = &bytes[start..end];

    // Each F32 is 4 bytes
    assert_eq!(
        coeff_bytes.len() % 4,
        0,
        "Coefficient data must be multiple of 4 bytes"
    );

    // Verify we can parse F32 values
    let n_coeffs = coeff_bytes.len() / 4;
    for i in 0..n_coeffs {
        let f32_bytes: [u8; 4] = coeff_bytes[i * 4..(i + 1) * 4]
            .try_into()
            .expect("Test data should be valid");
        let _value = f32::from_le_bytes(f32_bytes); // Should not panic
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_safetensors_roundtrip() {
    // RED PHASE: Verify save → load produces identical model
    // This is a CRITICAL property test for data integrity

    // Use well-conditioned data (independent features)
    let x = Matrix::from_vec(
        5,
        3,
        vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ],
    )
    .expect("Test data should be valid");
    let y = Vector::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

    // Train original model
    let mut model_original = LinearRegression::new();
    model_original
        .fit(&x, &y)
        .expect("Test data should be valid");

    // Get original coefficients and intercept
    let original_coeffs = model_original.coefficients();
    let original_intercept = model_original.intercept();

    // Save to SafeTensors
    let path = "test_roundtrip.safetensors";
    model_original
        .save_safetensors(path)
        .expect("Test data should be valid");

    // Load from SafeTensors
    let model_loaded = LinearRegression::load_safetensors(path).expect("Test data should be valid");

    // Verify coefficients match (within floating-point tolerance)
    let loaded_coeffs = model_loaded.coefficients();
    assert_eq!(
        loaded_coeffs.len(),
        original_coeffs.len(),
        "Coefficient count must match"
    );

    for i in 0..original_coeffs.len() {
        let diff = (loaded_coeffs[i] - original_coeffs[i]).abs();
        assert!(
            diff < 1e-6,
            "Coefficient {} must match: {} vs {}",
            i,
            original_coeffs[i],
            loaded_coeffs[i]
        );
    }

    // Verify intercept matches
    let diff = (model_loaded.intercept() - original_intercept).abs();
    assert!(
        diff < 1e-6,
        "Intercept must match: {} vs {}",
        original_intercept,
        model_loaded.intercept()
    );

    // Verify predictions match
    let pred_original = model_original.predict(&x);
    let pred_loaded = model_loaded.predict(&x);

    for i in 0..pred_original.len() {
        let diff = (pred_loaded[i] - pred_original[i]).abs();
        assert!(diff < 1e-5, "Prediction {i} must match");
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_safetensors_file_does_not_exist_error() {
    // RED PHASE: Verify proper error handling for missing files

    let result = LinearRegression::load_safetensors("nonexistent.safetensors");
    assert!(
        result.is_err(),
        "Loading nonexistent file should return error"
    );
    let error_msg = result.expect_err("Expected error in test");
    assert!(
        error_msg.contains("No such file") || error_msg.contains("not found"),
        "Error should mention file not found, got: {error_msg}"
    );
}

#[test]
fn test_safetensors_corrupted_header_error() {
    // RED PHASE: Verify proper error handling for corrupted files

    // Create file with invalid header (less than 8 bytes)
    let path = "test_corrupted.safetensors";
    fs::write(path, [1, 2, 3]).expect("Test data should be valid");

    let result = LinearRegression::load_safetensors(path);
    assert!(
        result.is_err(),
        "Loading corrupted file should return error"
    );

    // Cleanup
    fs::remove_file(path).ok();
}
