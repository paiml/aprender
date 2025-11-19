// GitHub Issue #8: Complete SafeTensors Model Serialization
// Tests for Ridge SafeTensors export functionality
//
// Acceptance Criteria:
// - Ridge::save_safetensors() exports valid SafeTensors format
// - SafeTensors includes coefficients, intercept, and alpha tensors
// - Roundtrip: save → load produces identical predictions
// - Cross-platform compatibility verified

use aprender::linear_model::Ridge;
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;
use std::fs;
use std::path::Path;

#[test]
fn test_ridge_save_safetensors_creates_file() {
    // Train a simple Ridge model
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model = Ridge::new(1.0); // alpha = 1.0
    model.fit(&x, &y).unwrap();

    // Save to SafeTensors format
    let path = "test_ridge_model.safetensors";
    model.save_safetensors(path).unwrap();

    // Verify file was created
    assert!(
        Path::new(path).exists(),
        "SafeTensors file should be created"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_save_load_roundtrip() {
    // Train Ridge model
    let x = Matrix::from_vec(5, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 8.0, 11.0, 14.0, 17.0]);

    let mut model = Ridge::new(0.5); // alpha = 0.5
    model.fit(&x, &y).unwrap();

    // Get original predictions
    let pred_original = model.predict(&x);

    // Save and load
    let path = "test_ridge_roundtrip.safetensors";
    model.save_safetensors(path).unwrap();
    let loaded_model = Ridge::load_safetensors(path).unwrap();

    // Get loaded model predictions
    let pred_loaded = loaded_model.predict(&x);

    // Verify predictions match (within floating point tolerance)
    assert_eq!(pred_original.len(), pred_loaded.len());
    for i in 0..pred_original.len() {
        assert!(
            (pred_original[i] - pred_loaded[i]).abs() < 1e-6,
            "Prediction {} mismatch: original={}, loaded={}",
            i,
            pred_original[i],
            pred_loaded[i]
        );
    }

    // Verify alpha is preserved
    assert_eq!(model.alpha(), loaded_model.alpha());

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_safetensors_metadata_includes_alpha() {
    // Create and fit Ridge model with specific alpha
    let x = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let mut model = Ridge::new(2.5); // alpha = 2.5
    model.fit(&x, &y).unwrap();

    let path = "test_ridge_alpha.safetensors";
    model.save_safetensors(path).unwrap();

    let bytes = fs::read(path).unwrap();

    // Extract metadata
    let header_bytes: [u8; 8] = bytes[0..8].try_into().unwrap();
    let metadata_len = u64::from_le_bytes(header_bytes) as usize;
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_json).unwrap();

    // Parse JSON
    let metadata: serde_json::Value = serde_json::from_str(metadata_str).unwrap();

    // Verify all required tensors exist
    assert!(
        metadata.get("coefficients").is_some(),
        "Metadata must include 'coefficients' tensor"
    );
    assert!(
        metadata.get("intercept").is_some(),
        "Metadata must include 'intercept' tensor"
    );
    assert!(
        metadata.get("alpha").is_some(),
        "Metadata must include 'alpha' tensor"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_save_unfitted_model_fails() {
    // Unfitted model should not be saveable
    let model = Ridge::new(1.0);

    let path = "test_unfitted_ridge.safetensors";
    let result = model.save_safetensors(path);

    assert!(
        result.is_err(),
        "Saving unfitted model should return an error"
    );
    let error_msg = result.unwrap_err();
    assert!(
        error_msg.contains("unfitted") || error_msg.contains("fit"),
        "Error message should mention model is unfitted. Got: {}",
        error_msg
    );

    // Ensure no file was created
    assert!(
        !Path::new(path).exists(),
        "No file should be created for unfitted model"
    );
}

#[test]
fn test_ridge_load_nonexistent_file_fails() {
    let result = Ridge::load_safetensors("nonexistent_ridge_file.safetensors");
    assert!(
        result.is_err(),
        "Loading nonexistent file should return an error"
    );
}

#[test]
fn test_ridge_load_corrupted_metadata_fails() {
    // Create a file with invalid SafeTensors format
    let path = "test_corrupted_ridge.safetensors";
    fs::write(path, b"invalid safetensors data").unwrap();

    let result = Ridge::load_safetensors(path);
    assert!(
        result.is_err(),
        "Loading corrupted file should return an error"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_coefficients_preserved() {
    // Verify coefficients are exactly preserved through save/load
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model = Ridge::new(0.1);
    model.fit(&x, &y).unwrap();

    let original_coef = model.coefficients();
    let original_intercept = model.intercept();
    let original_alpha = model.alpha();

    // Save and load
    let path = "test_ridge_coef.safetensors";
    model.save_safetensors(path).unwrap();
    let loaded_model = Ridge::load_safetensors(path).unwrap();

    let loaded_coef = loaded_model.coefficients();
    let loaded_intercept = loaded_model.intercept();
    let loaded_alpha = loaded_model.alpha();

    // Verify coefficients
    assert_eq!(original_coef.len(), loaded_coef.len());
    for i in 0..original_coef.len() {
        assert!(
            (original_coef[i] - loaded_coef[i]).abs() < 1e-6,
            "Coefficient {} mismatch",
            i
        );
    }

    // Verify intercept
    assert!(
        (original_intercept - loaded_intercept).abs() < 1e-6,
        "Intercept mismatch"
    );

    // Verify alpha
    assert_eq!(original_alpha, loaded_alpha, "Alpha mismatch");

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_multiple_save_load_cycles() {
    // Property test: Multiple save/load cycles should be idempotent
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let mut model = Ridge::new(1.0);
    model.fit(&x, &y).unwrap();

    let path1 = "test_ridge_cycle1.safetensors";
    let path2 = "test_ridge_cycle2.safetensors";
    let path3 = "test_ridge_cycle3.safetensors";

    // Cycle 1: original → save → load
    model.save_safetensors(path1).unwrap();
    let model1 = Ridge::load_safetensors(path1).unwrap();

    // Cycle 2: loaded → save → load
    model1.save_safetensors(path2).unwrap();
    let model2 = Ridge::load_safetensors(path2).unwrap();

    // Cycle 3: loaded again → save → load
    model2.save_safetensors(path3).unwrap();
    let model3 = Ridge::load_safetensors(path3).unwrap();

    // All models should produce identical predictions
    let pred_orig = model.predict(&x);
    let pred1 = model1.predict(&x);
    let pred2 = model2.predict(&x);
    let pred3 = model3.predict(&x);

    for i in 0..pred_orig.len() {
        assert!((pred_orig[i] - pred1[i]).abs() < 1e-6);
        assert!((pred_orig[i] - pred2[i]).abs() < 1e-6);
        assert!((pred_orig[i] - pred3[i]).abs() < 1e-6);
    }

    // Cleanup
    fs::remove_file(path1).ok();
    fs::remove_file(path2).ok();
    fs::remove_file(path3).ok();
}

#[test]
fn test_ridge_different_alphas_produce_different_models() {
    // Verify that models with different alphas save/load correctly
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model1 = Ridge::new(0.1);
    model1.fit(&x, &y).unwrap();

    let mut model2 = Ridge::new(10.0);
    model2.fit(&x, &y).unwrap();

    // Save both models
    let path1 = "test_ridge_alpha_01.safetensors";
    let path2 = "test_ridge_alpha_10.safetensors";

    model1.save_safetensors(path1).unwrap();
    model2.save_safetensors(path2).unwrap();

    // Load and verify alphas are preserved
    let loaded1 = Ridge::load_safetensors(path1).unwrap();
    let loaded2 = Ridge::load_safetensors(path2).unwrap();

    assert_eq!(loaded1.alpha(), 0.1);
    assert_eq!(loaded2.alpha(), 10.0);

    // Predictions should differ (different regularization)
    let pred1 = loaded1.predict(&x);
    let pred2 = loaded2.predict(&x);

    let mut predictions_differ = false;
    for i in 0..pred1.len() {
        if (pred1[i] - pred2[i]).abs() > 1e-3 {
            predictions_differ = true;
            break;
        }
    }

    assert!(
        predictions_differ,
        "Models with different alphas should produce different predictions"
    );

    // Cleanup
    fs::remove_file(path1).ok();
    fs::remove_file(path2).ok();
}

#[test]
fn test_ridge_file_size_reasonable() {
    // Verify SafeTensors file size is reasonable (not bloated)
    let x = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let mut model = Ridge::new(1.0);
    model.fit(&x, &y).unwrap();

    let path = "test_ridge_size.safetensors";
    model.save_safetensors(path).unwrap();

    let file_size = fs::metadata(path).unwrap().len();

    // File should be reasonable:
    // - Metadata: < 1KB
    // - Coefficients: 2 features × 4 bytes = 8 bytes
    // - Intercept: 1 × 4 bytes = 4 bytes
    // - Alpha: 1 × 4 bytes = 4 bytes
    // - Total: < 2KB for this small model
    assert!(
        file_size < 2048,
        "SafeTensors file should be compact. Got {} bytes",
        file_size
    );

    assert!(
        file_size > 16,
        "SafeTensors file should contain data. Got {} bytes",
        file_size
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_r2_score_preserved() {
    // Property test: R² score should be identical after roundtrip
    let x = Matrix::from_vec(5, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 8.0, 11.0, 14.0, 17.0]);

    let mut model = Ridge::new(1.0);
    model.fit(&x, &y).unwrap();

    let original_r2 = model.score(&x, &y);

    // Save and load
    let path = "test_ridge_r2.safetensors";
    model.save_safetensors(path).unwrap();
    let loaded_model = Ridge::load_safetensors(path).unwrap();

    let loaded_r2 = loaded_model.score(&x, &y);

    // R² scores should be identical
    assert!(
        (original_r2 - loaded_r2).abs() < 1e-6,
        "R² score should be preserved: original={}, loaded={}",
        original_r2,
        loaded_r2
    );

    // Cleanup
    fs::remove_file(path).ok();
}
