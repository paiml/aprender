// GitHub Issue #8: Complete SafeTensors Model Serialization
// Tests for ElasticNet SafeTensors export functionality
//
// Acceptance Criteria:
// - ElasticNet::save_safetensors() exports valid SafeTensors format
// - SafeTensors includes coefficients, intercept, alpha, l1_ratio, max_iter, tol tensors
// - Roundtrip: save → load produces identical predictions
// - L1/L2 mix (l1_ratio) preserved through serialization

use aprender::linear_model::ElasticNet;
use aprender::primitives::{Matrix, Vector};
use aprender::traits::Estimator;
use std::fs;
use std::path::Path;

#[test]
fn test_elasticnet_save_safetensors_creates_file() {
    // Train a simple ElasticNet model
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model = ElasticNet::new(0.1, 0.5); // alpha = 0.1, l1_ratio = 0.5
    model.fit(&x, &y).unwrap();

    // Save to SafeTensors format
    let path = "test_elasticnet_model.safetensors";
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
fn test_elasticnet_save_load_roundtrip() {
    // Train ElasticNet model
    let x = Matrix::from_vec(5, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 8.0, 11.0, 14.0, 17.0]);

    let mut model = ElasticNet::new(0.5, 0.7).with_max_iter(500).with_tol(1e-5);
    model.fit(&x, &y).unwrap();

    // Get original predictions
    let pred_original = model.predict(&x);

    // Save and load
    let path = "test_elasticnet_roundtrip.safetensors";
    model.save_safetensors(path).unwrap();
    let loaded_model = ElasticNet::load_safetensors(path).unwrap();

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
    assert_eq!(model.l1_ratio(), loaded_model.l1_ratio());

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_elasticnet_safetensors_metadata_includes_all_hyperparameters() {
    // Create and fit ElasticNet model with specific hyperparameters
    let x = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let mut model = ElasticNet::new(2.5, 0.3).with_max_iter(1500).with_tol(1e-6);
    model.fit(&x, &y).unwrap();

    let path = "test_elasticnet_hyperparams.safetensors";
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
    assert!(
        metadata.get("l1_ratio").is_some(),
        "Metadata must include 'l1_ratio' tensor"
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
fn test_elasticnet_save_unfitted_model_fails() {
    // Unfitted model should not be saveable
    let model = ElasticNet::new(1.0, 0.5);

    let path = "test_unfitted_elasticnet.safetensors";
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
fn test_elasticnet_load_nonexistent_file_fails() {
    let result = ElasticNet::load_safetensors("nonexistent_elasticnet_file.safetensors");
    assert!(
        result.is_err(),
        "Loading nonexistent file should return an error"
    );
}

#[test]
fn test_elasticnet_load_corrupted_metadata_fails() {
    // Create a file with invalid SafeTensors format
    let path = "test_corrupted_elasticnet.safetensors";
    fs::write(path, b"invalid safetensors data").unwrap();

    let result = ElasticNet::load_safetensors(path);
    assert!(
        result.is_err(),
        "Loading corrupted file should return an error"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_elasticnet_l1_ratio_preserved() {
    // Verify l1_ratio is exactly preserved through save/load
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    let mut model = ElasticNet::new(0.1, 0.7); // l1_ratio = 0.7
    model.fit(&x, &y).unwrap();

    let original_l1_ratio = model.l1_ratio();
    let original_alpha = model.alpha();

    // Save and load
    let path = "test_elasticnet_l1_ratio.safetensors";
    model.save_safetensors(path).unwrap();
    let loaded_model = ElasticNet::load_safetensors(path).unwrap();

    let loaded_l1_ratio = loaded_model.l1_ratio();
    let loaded_alpha = loaded_model.alpha();

    // Verify l1_ratio
    assert_eq!(original_l1_ratio, loaded_l1_ratio, "L1 ratio mismatch");

    // Verify alpha
    assert_eq!(original_alpha, loaded_alpha, "Alpha mismatch");

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_elasticnet_multiple_save_load_cycles() {
    // Property test: Multiple save/load cycles should be idempotent
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let mut model = ElasticNet::new(0.05, 0.5);
    model.fit(&x, &y).unwrap();

    let path1 = "test_elasticnet_cycle1.safetensors";
    let path2 = "test_elasticnet_cycle2.safetensors";
    let path3 = "test_elasticnet_cycle3.safetensors";

    // Cycle 1: original → save → load
    model.save_safetensors(path1).unwrap();
    let model1 = ElasticNet::load_safetensors(path1).unwrap();

    // Cycle 2: loaded → save → load
    model1.save_safetensors(path2).unwrap();
    let model2 = ElasticNet::load_safetensors(path2).unwrap();

    // Cycle 3: loaded again → save → load
    model2.save_safetensors(path3).unwrap();
    let model3 = ElasticNet::load_safetensors(path3).unwrap();

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
fn test_elasticnet_different_l1_ratios() {
    // Verify that models with different l1_ratios save/load correctly
    let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 4.0, 11.0, 10.0]);

    // Pure Ridge (l1_ratio = 0.0)
    let mut model1 = ElasticNet::new(0.1, 0.0);
    model1.fit(&x, &y).unwrap();

    // Balanced (l1_ratio = 0.5)
    let mut model2 = ElasticNet::new(0.1, 0.5);
    model2.fit(&x, &y).unwrap();

    // Pure Lasso (l1_ratio = 1.0)
    let mut model3 = ElasticNet::new(0.1, 1.0);
    model3.fit(&x, &y).unwrap();

    // Save all models
    let path1 = "test_elasticnet_ridge.safetensors";
    let path2 = "test_elasticnet_balanced.safetensors";
    let path3 = "test_elasticnet_lasso.safetensors";

    model1.save_safetensors(path1).unwrap();
    model2.save_safetensors(path2).unwrap();
    model3.save_safetensors(path3).unwrap();

    // Load and verify l1_ratios are preserved
    let loaded1 = ElasticNet::load_safetensors(path1).unwrap();
    let loaded2 = ElasticNet::load_safetensors(path2).unwrap();
    let loaded3 = ElasticNet::load_safetensors(path3).unwrap();

    assert_eq!(loaded1.l1_ratio(), 0.0, "Pure Ridge l1_ratio");
    assert_eq!(loaded2.l1_ratio(), 0.5, "Balanced l1_ratio");
    assert_eq!(loaded3.l1_ratio(), 1.0, "Pure Lasso l1_ratio");

    // Cleanup
    fs::remove_file(path1).ok();
    fs::remove_file(path2).ok();
    fs::remove_file(path3).ok();
}

#[test]
fn test_elasticnet_file_size_reasonable() {
    // Verify SafeTensors file size is reasonable (not bloated)
    let x = Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);

    let mut model = ElasticNet::new(0.1, 0.5);
    model.fit(&x, &y).unwrap();

    let path = "test_elasticnet_size.safetensors";
    model.save_safetensors(path).unwrap();

    let file_size = fs::metadata(path).unwrap().len();

    // File should be reasonable:
    // - Metadata: < 1KB
    // - Coefficients: 2 features × 4 bytes = 8 bytes
    // - Intercept: 1 × 4 bytes = 4 bytes
    // - Alpha: 1 × 4 bytes = 4 bytes
    // - L1_ratio: 1 × 4 bytes = 4 bytes
    // - Max_iter: 1 × 4 bytes = 4 bytes
    // - Tol: 1 × 4 bytes = 4 bytes
    // - Total: < 2KB for this small model
    assert!(
        file_size < 2048,
        "SafeTensors file should be compact. Got {} bytes",
        file_size
    );

    assert!(
        file_size > 28,
        "SafeTensors file should contain data. Got {} bytes",
        file_size
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_elasticnet_r2_score_preserved() {
    // Property test: R² score should be identical after roundtrip
    let x = Matrix::from_vec(5, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]).unwrap();
    let y = Vector::from_vec(vec![5.0, 8.0, 11.0, 14.0, 17.0]);

    let mut model = ElasticNet::new(0.1, 0.5);
    model.fit(&x, &y).unwrap();

    let original_r2 = model.score(&x, &y);

    // Save and load
    let path = "test_elasticnet_r2.safetensors";
    model.save_safetensors(path).unwrap();
    let loaded_model = ElasticNet::load_safetensors(path).unwrap();

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

#[test]
fn test_elasticnet_combined_regularization_behavior() {
    // Property test: ElasticNet should combine L1 and L2 properties
    let x = Matrix::from_vec(
        5,
        3,
        vec![
            1.0, 0.1, 0.01, 2.0, 0.2, 0.02, 3.0, 0.1, 0.03, 4.0, 0.3, 0.01, 5.0, 0.2, 0.02,
        ],
    )
    .unwrap();
    let y = Vector::from_vec(vec![1.1, 2.2, 3.1, 4.3, 5.2]);

    // High alpha with balanced l1_ratio
    let mut model = ElasticNet::new(0.5, 0.5);
    model.fit(&x, &y).unwrap();

    let original_coef = model.coefficients();

    // Save and load
    let path = "test_elasticnet_combined.safetensors";
    model.save_safetensors(path).unwrap();
    let loaded_model = ElasticNet::load_safetensors(path).unwrap();

    let loaded_coef = loaded_model.coefficients();

    // Coefficients should match exactly
    assert_eq!(original_coef.len(), loaded_coef.len());
    for i in 0..original_coef.len() {
        assert!(
            (original_coef[i] - loaded_coef[i]).abs() < 1e-6,
            "Coefficient {} mismatch",
            i
        );
    }

    // Cleanup
    fs::remove_file(path).ok();
}
