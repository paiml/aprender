// GitHub Issue #8: Complete SafeTensors Model Serialization
// Tests for Lasso SafeTensors export functionality
//
// Acceptance Criteria:
// - Lasso::save_safetensors() exports valid SafeTensors format
// - SafeTensors includes coefficients, intercept, alpha, max_iter, tol tensors
// - Roundtrip: save → load produces identical predictions
// - Hyperparameters (max_iter, tol) preserved

use aprender::linear_model::Lasso;
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;
use std::fs;
use std::path::Path;

#[test]
fn test_lasso_save_safetensors_creates_file() {
    // Train a simple Lasso model
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model = Lasso::new(0.1); // alpha = 0.1
    model.fit(&x, &y).expect("Test data should be valid");

    // Save to SafeTensors format
    let path = "test_lasso_model.safetensors";
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
fn test_lasso_save_load_roundtrip() {
    // Train Lasso model
    let x = Matrix::from_vec(5, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![5.0, 8.0, 11.0, 14.0, 17.0]);

    let mut model = Lasso::new(0.5).with_max_iter(500).with_tol(1e-5);
    model.fit(&x, &y).expect("Test data should be valid");

    // Get original predictions
    let pred_original = model.predict(&x);

    // Save and load
    let path = "test_lasso_roundtrip.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_model = Lasso::load_safetensors(path).expect("Test data should be valid");

    // Get loaded model predictions
    let pred_loaded = loaded_model.predict(&x);

    // Verify predictions match (within floating point tolerance)
    assert_eq!(pred_original.len(), pred_loaded.len());
    for i in 0..pred_original.len() {
        assert!(
            (pred_original[i] - pred_loaded[i]).abs() < 1e-5,
            "Prediction {} mismatch: original={}, loaded={}",
            i,
            pred_original[i],
            pred_loaded[i]
        );
    }

    // Verify hyperparameters are preserved
    assert_eq!(model.alpha(), loaded_model.alpha());

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_lasso_safetensors_metadata_includes_all_hyperparameters() {
    // Create and fit Lasso model with specific hyperparameters
    let x = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let mut model = Lasso::new(2.5).with_max_iter(1500).with_tol(1e-6);
    model.fit(&x, &y).expect("Test data should be valid");

    let path = "test_lasso_hyperparams.safetensors";
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
    assert!(
        metadata.get("max_iter").is_some(),
        "Metadata must include 'max_iter' tensor"
    );
    assert!(
        metadata.get("tol").is_some(),
        "Metadata must include 'tol' tensor"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_lasso_save_unfitted_model_fails() {
    // Unfitted model should not be saveable
    let model = Lasso::new(1.0);

    let path = "test_unfitted_lasso.safetensors";
    let result = model.save_safetensors(path);

    assert!(
        result.is_err(),
        "Saving unfitted model should return an error"
    );
    let error_msg = result.unwrap_err();
    assert!(
        error_msg.contains("unfitted") || error_msg.contains("fit"),
        "Error message should mention model is unfitted. Got: {error_msg}"
    );

    // Ensure no file was created
    assert!(
        !Path::new(path).exists(),
        "No file should be created for unfitted model"
    );
}

#[test]
fn test_lasso_load_nonexistent_file_fails() {
    let result = Lasso::load_safetensors("nonexistent_lasso_file.safetensors");
    assert!(
        result.is_err(),
        "Loading nonexistent file should return an error"
    );
}

#[test]
fn test_lasso_load_corrupted_metadata_fails() {
    // Create a file with invalid SafeTensors format
    let path = "test_corrupted_lasso.safetensors";
    fs::write(path, b"invalid safetensors data").expect("Test data should be valid");

    let result = Lasso::load_safetensors(path);
    assert!(
        result.is_err(),
        "Loading corrupted file should return an error"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_lasso_coefficients_and_hyperparams_preserved() {
    // Verify coefficients and hyperparameters are exactly preserved
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model = Lasso::new(0.1).with_max_iter(800).with_tol(1e-5);
    model.fit(&x, &y).expect("Test data should be valid");

    let original_coef = model.coefficients();
    let original_intercept = model.intercept();
    let original_alpha = model.alpha();

    // Save and load
    let path = "test_lasso_params.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_model = Lasso::load_safetensors(path).expect("Test data should be valid");

    let loaded_coef = loaded_model.coefficients();
    let loaded_intercept = loaded_model.intercept();
    let loaded_alpha = loaded_model.alpha();

    // Verify coefficients
    assert_eq!(original_coef.len(), loaded_coef.len());
    for i in 0..original_coef.len() {
        assert!(
            (original_coef[i] - loaded_coef[i]).abs() < 1e-6,
            "Coefficient {i} mismatch"
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
fn test_lasso_multiple_save_load_cycles() {
    // Property test: Multiple save/load cycles should be idempotent
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Test data should be valid");
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let mut model = Lasso::new(0.05);
    model.fit(&x, &y).expect("Test data should be valid");

    let path1 = "test_lasso_cycle1.safetensors";
    let path2 = "test_lasso_cycle2.safetensors";
    let path3 = "test_lasso_cycle3.safetensors";

    // Cycle 1: original → save → load
    model
        .save_safetensors(path1)
        .expect("Test data should be valid");
    let model1 = Lasso::load_safetensors(path1).expect("Test data should be valid");

    // Cycle 2: loaded → save → load
    model1
        .save_safetensors(path2)
        .expect("Test data should be valid");
    let model2 = Lasso::load_safetensors(path2).expect("Test data should be valid");

    // Cycle 3: loaded again → save → load
    model2
        .save_safetensors(path3)
        .expect("Test data should be valid");
    let model3 = Lasso::load_safetensors(path3).expect("Test data should be valid");

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
fn test_lasso_sparsity_preserved() {
    // Property test: Lasso's sparsity (zero coefficients) should be preserved
    let x = Matrix::from_vec(
        5,
        3,
        vec![
            1.0, 0.1, 0.01, 2.0, 0.2, 0.02, 3.0, 0.1, 0.03, 4.0, 0.3, 0.01, 5.0, 0.2, 0.02,
        ],
    )
    .expect("Test data should be valid");
    let y = Vector::from_vec(vec![1.1, 2.2, 3.1, 4.3, 5.2]);

    // High alpha to force sparsity
    let mut model = Lasso::new(0.5);
    model.fit(&x, &y).expect("Test data should be valid");

    let original_coef = model.coefficients();

    // Count zeros in original model
    let mut original_zeros = 0;
    for i in 0..original_coef.len() {
        if original_coef[i].abs() < 1e-6 {
            original_zeros += 1;
        }
    }

    // Save and load
    let path = "test_lasso_sparsity.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_model = Lasso::load_safetensors(path).expect("Test data should be valid");

    let loaded_coef = loaded_model.coefficients();

    // Count zeros in loaded model
    let mut loaded_zeros = 0;
    for i in 0..loaded_coef.len() {
        if loaded_coef[i].abs() < 1e-6 {
            loaded_zeros += 1;
        }
    }

    // Sparsity pattern should be preserved
    assert_eq!(
        original_zeros, loaded_zeros,
        "Number of zero coefficients should be preserved"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_lasso_different_alphas_produce_different_models() {
    // Verify that models with different alphas save/load correctly
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model1 = Lasso::new(0.01);
    model1.fit(&x, &y).expect("Test data should be valid");

    let mut model2 = Lasso::new(1.0);
    model2.fit(&x, &y).expect("Test data should be valid");

    // Save both models
    let path1 = "test_lasso_alpha_001.safetensors";
    let path2 = "test_lasso_alpha_1.safetensors";

    model1
        .save_safetensors(path1)
        .expect("Test data should be valid");
    model2
        .save_safetensors(path2)
        .expect("Test data should be valid");

    // Load and verify alphas are preserved
    let loaded1 = Lasso::load_safetensors(path1).expect("Test data should be valid");
    let loaded2 = Lasso::load_safetensors(path2).expect("Test data should be valid");

    assert_eq!(loaded1.alpha(), 0.01);
    assert_eq!(loaded2.alpha(), 1.0);

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
fn test_lasso_file_size_reasonable() {
    // Verify SafeTensors file size is reasonable (not bloated)
    let x = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let mut model = Lasso::new(0.1);
    model.fit(&x, &y).expect("Test data should be valid");

    let path = "test_lasso_size.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");

    let file_size = fs::metadata(path).expect("Test data should be valid").len();

    // File should be reasonable:
    // - Metadata: < 1KB
    // - Coefficients: 2 features × 4 bytes = 8 bytes
    // - Intercept: 1 × 4 bytes = 4 bytes
    // - Alpha: 1 × 4 bytes = 4 bytes
    // - Max_iter: 1 × 4 bytes = 4 bytes
    // - Tol: 1 × 4 bytes = 4 bytes
    // - Total: < 2KB for this small model
    assert!(
        file_size < 2048,
        "SafeTensors file should be compact. Got {file_size} bytes"
    );

    assert!(
        file_size > 24,
        "SafeTensors file should contain data. Got {file_size} bytes"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_lasso_r2_score_preserved() {
    // Property test: R² score should be identical after roundtrip
    let x = Matrix::from_vec(5, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0])
        .expect("Test data should be valid");
    let y = Vector::from_vec(vec![5.0, 8.0, 11.0, 14.0, 17.0]);

    let mut model = Lasso::new(0.1);
    model.fit(&x, &y).expect("Test data should be valid");

    let original_r2 = model.score(&x, &y);

    // Save and load
    let path = "test_lasso_r2.safetensors";
    model
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_model = Lasso::load_safetensors(path).expect("Test data should be valid");

    let loaded_r2 = loaded_model.score(&x, &y);

    // R² scores should be identical
    assert!(
        (original_r2 - loaded_r2).abs() < 1e-6,
        "R² score should be preserved: original={original_r2}, loaded={loaded_r2}"
    );

    // Cleanup
    fs::remove_file(path).ok();
}
