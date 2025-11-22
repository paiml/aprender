// GitHub Issue #8: Complete SafeTensors Model Serialization
// Tests for StandardScaler SafeTensors export functionality
//
// Acceptance Criteria:
// - StandardScaler::save_safetensors() exports valid SafeTensors format
// - SafeTensors includes mean vector, std vector, with_mean, with_std flags
// - Roundtrip: save → load produces identical transformations
// - Scaling parameters (mean/std) preserved exactly

use aprender::preprocessing::StandardScaler;
use aprender::primitives::Matrix;
use aprender::traits::Transformer;
use std::fs;
use std::path::Path;

#[test]
fn test_scaler_save_safetensors_creates_file() {
    // Train a simple StandardScaler
    let x = Matrix::from_vec(4, 2, vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("Test data should be valid");

    // Save to SafeTensors format
    let path = "test_scaler_model.safetensors";
    scaler
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
fn test_scaler_save_load_roundtrip() {
    // Train StandardScaler
    let x = Matrix::from_vec(4, 2, vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("Test data should be valid");

    // Get original transformation
    let transformed_original = scaler.transform(&x).expect("Test data should be valid");

    // Save and load
    let path = "test_scaler_roundtrip.safetensors";
    scaler
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_scaler = StandardScaler::load_safetensors(path).expect("Test data should be valid");

    // Get loaded scaler transformation
    let transformed_loaded = loaded_scaler
        .transform(&x)
        .expect("Test data should be valid");

    // Verify transformations match (within floating point tolerance)
    let (n_rows, n_cols) = transformed_original.shape();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let orig = transformed_original.get(i, j);
            let loaded = transformed_loaded.get(i, j);
            assert!(
                (orig - loaded).abs() < 1e-6,
                "Transformed[{i},{j}] mismatch: original={orig}, loaded={loaded}"
            );
        }
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_scaler_safetensors_metadata_includes_all_tensors() {
    // Create and fit StandardScaler
    let x = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("Test data should be valid");

    let path = "test_scaler_metadata.safetensors";
    scaler
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
        metadata.get("mean").is_some(),
        "Metadata must include 'mean' tensor"
    );
    assert!(
        metadata.get("std").is_some(),
        "Metadata must include 'std' tensor"
    );
    assert!(
        metadata.get("with_mean").is_some(),
        "Metadata must include 'with_mean' tensor"
    );
    assert!(
        metadata.get("with_std").is_some(),
        "Metadata must include 'with_std' tensor"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_scaler_save_unfitted_model_fails() {
    // Unfitted scaler should not be saveable
    let scaler = StandardScaler::new();

    let path = "test_unfitted_scaler.safetensors";
    let result = scaler.save_safetensors(path);

    assert!(
        result.is_err(),
        "Saving unfitted scaler should return an error"
    );
    let error_msg = result.unwrap_err();
    assert!(
        error_msg.contains("unfitted") || error_msg.contains("fit"),
        "Error message should mention scaler is unfitted. Got: {error_msg}"
    );

    // Ensure no file was created
    assert!(
        !Path::new(path).exists(),
        "No file should be created for unfitted scaler"
    );
}

#[test]
fn test_scaler_load_nonexistent_file_fails() {
    let result = StandardScaler::load_safetensors("nonexistent_scaler_file.safetensors");
    assert!(
        result.is_err(),
        "Loading nonexistent file should return an error"
    );
}

#[test]
fn test_scaler_load_corrupted_metadata_fails() {
    // Create a file with invalid SafeTensors format
    let path = "test_corrupted_scaler.safetensors";
    fs::write(path, b"invalid safetensors data").expect("Test data should be valid");

    let result = StandardScaler::load_safetensors(path);
    assert!(
        result.is_err(),
        "Loading corrupted file should return an error"
    );

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_scaler_mean_and_std_preserved() {
    // Verify mean and std are exactly preserved through save/load
    let x = Matrix::from_vec(4, 2, vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("Test data should be valid");

    let original_mean = scaler.mean();
    let original_std = scaler.std();

    // Save and load
    let path = "test_scaler_params.safetensors";
    scaler
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_scaler = StandardScaler::load_safetensors(path).expect("Test data should be valid");

    let loaded_mean = loaded_scaler.mean();
    let loaded_std = loaded_scaler.std();

    // Verify lengths match
    assert_eq!(original_mean.len(), loaded_mean.len());
    assert_eq!(original_std.len(), loaded_std.len());

    // Verify all values match
    for i in 0..original_mean.len() {
        assert!(
            (original_mean[i] - loaded_mean[i]).abs() < 1e-6,
            "Mean[{}] mismatch: original={}, loaded={}",
            i,
            original_mean[i],
            loaded_mean[i]
        );
        assert!(
            (original_std[i] - loaded_std[i]).abs() < 1e-6,
            "Std[{}] mismatch: original={}, loaded={}",
            i,
            original_std[i],
            loaded_std[i]
        );
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_scaler_with_mean_flag_preserved() {
    // Test with_mean=false
    let x = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new().with_mean(false);
    scaler.fit(&x).expect("Test data should be valid");

    // Save and load
    let path = "test_scaler_no_mean.safetensors";
    scaler
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_scaler = StandardScaler::load_safetensors(path).expect("Test data should be valid");

    // Verify transformations match (with_mean=false should not center)
    let transformed_orig = scaler.transform(&x).expect("Test data should be valid");
    let transformed_loaded = loaded_scaler
        .transform(&x)
        .expect("Test data should be valid");

    let (n_rows, n_cols) = transformed_orig.shape();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let orig = transformed_orig.get(i, j);
            let loaded = transformed_loaded.get(i, j);
            assert!((orig - loaded).abs() < 1e-6, "Mismatch at [{i},{j}]");
        }
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_scaler_with_std_flag_preserved() {
    // Test with_std=false
    let x = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new().with_std(false);
    scaler.fit(&x).expect("Test data should be valid");

    // Save and load
    let path = "test_scaler_no_std.safetensors";
    scaler
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_scaler = StandardScaler::load_safetensors(path).expect("Test data should be valid");

    // Verify transformations match (with_std=false should not scale)
    let transformed_orig = scaler.transform(&x).expect("Test data should be valid");
    let transformed_loaded = loaded_scaler
        .transform(&x)
        .expect("Test data should be valid");

    let (n_rows, n_cols) = transformed_orig.shape();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let orig = transformed_orig.get(i, j);
            let loaded = transformed_loaded.get(i, j);
            assert!((orig - loaded).abs() < 1e-6, "Mismatch at [{i},{j}]");
        }
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_scaler_multiple_save_load_cycles() {
    // Property test: Multiple save/load cycles should be idempotent
    let x = Matrix::from_vec(4, 2, vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("Test data should be valid");

    let path1 = "test_scaler_cycle1.safetensors";
    let path2 = "test_scaler_cycle2.safetensors";
    let path3 = "test_scaler_cycle3.safetensors";

    // Cycle 1: original → save → load
    scaler
        .save_safetensors(path1)
        .expect("Test data should be valid");
    let scaler1 = StandardScaler::load_safetensors(path1).expect("Test data should be valid");

    // Cycle 2: loaded → save → load
    scaler1
        .save_safetensors(path2)
        .expect("Test data should be valid");
    let scaler2 = StandardScaler::load_safetensors(path2).expect("Test data should be valid");

    // Cycle 3: loaded again → save → load
    scaler2
        .save_safetensors(path3)
        .expect("Test data should be valid");
    let scaler3 = StandardScaler::load_safetensors(path3).expect("Test data should be valid");

    // All scalers should produce identical transformations
    let trans_orig = scaler.transform(&x).expect("Test data should be valid");
    let trans1 = scaler1.transform(&x).expect("Test data should be valid");
    let trans2 = scaler2.transform(&x).expect("Test data should be valid");
    let trans3 = scaler3.transform(&x).expect("Test data should be valid");

    let (n_rows, n_cols) = trans_orig.shape();
    for i in 0..n_rows {
        for j in 0..n_cols {
            assert!((trans_orig.get(i, j) - trans1.get(i, j)).abs() < 1e-6);
            assert!((trans_orig.get(i, j) - trans2.get(i, j)).abs() < 1e-6);
            assert!((trans_orig.get(i, j) - trans3.get(i, j)).abs() < 1e-6);
        }
    }

    // Cleanup
    fs::remove_file(path1).ok();
    fs::remove_file(path2).ok();
    fs::remove_file(path3).ok();
}

#[test]
fn test_scaler_inverse_transform_preserved() {
    // Verify inverse transformation works after roundtrip
    let x = Matrix::from_vec(4, 2, vec![1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("Test data should be valid");

    let transformed = scaler.transform(&x).expect("Test data should be valid");

    // Save and load
    let path = "test_scaler_inverse.safetensors";
    scaler
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_scaler = StandardScaler::load_safetensors(path).expect("Test data should be valid");

    // Inverse transform using loaded scaler
    let reconstructed = loaded_scaler
        .inverse_transform(&transformed)
        .expect("Test data should be valid");

    // Should match original data
    let (n_rows, n_cols) = x.shape();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let orig = x.get(i, j);
            let recon = reconstructed.get(i, j);
            assert!(
                (orig - recon).abs() < 1e-4,
                "Reconstructed[{i},{j}] mismatch: original={orig}, reconstructed={recon}"
            );
        }
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_scaler_high_dimensional_data() {
    // Test with higher-dimensional data
    let x = Matrix::from_vec(
        4,
        5,
        vec![
            1.0, 10.0, 100.0, 1000.0, 10000.0, 2.0, 20.0, 200.0, 2000.0, 20000.0, 3.0, 30.0, 300.0,
            3000.0, 30000.0, 4.0, 40.0, 400.0, 4000.0, 40000.0,
        ],
    )
    .expect("Test data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("Test data should be valid");

    // Save and load
    let path = "test_scaler_highdim.safetensors";
    scaler
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_scaler = StandardScaler::load_safetensors(path).expect("Test data should be valid");

    // Verify transformations match
    let transformed_orig = scaler.transform(&x).expect("Test data should be valid");
    let transformed_loaded = loaded_scaler
        .transform(&x)
        .expect("Test data should be valid");

    let (n_rows, n_cols) = transformed_orig.shape();
    for i in 0..n_rows {
        for j in 0..n_cols {
            let orig = transformed_orig.get(i, j);
            let loaded = transformed_loaded.get(i, j);
            assert!(
                (orig - loaded).abs() < 1e-6,
                "High-dimensional transformation mismatch at [{i},{j}]"
            );
        }
    }

    // Cleanup
    fs::remove_file(path).ok();
}

#[test]
fn test_scaler_file_size_reasonable() {
    // Verify SafeTensors file size is reasonable
    let x = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("Test data should be valid");

    let path = "test_scaler_size.safetensors";
    scaler
        .save_safetensors(path)
        .expect("Test data should be valid");

    let file_size = fs::metadata(path).expect("Test data should be valid").len();

    // File should be reasonable:
    // - Metadata: < 1KB
    // - Mean: 2 features × 4 bytes = 8 bytes
    // - Std: 2 features × 4 bytes = 8 bytes
    // - Flags: 2 × 4 bytes = 8 bytes
    // - Total: < 2KB for this small scaler
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
fn test_scaler_both_flags_false() {
    // Edge case: both with_mean and with_std are false (identity transformation)
    let x = Matrix::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .expect("Test data should be valid");

    let mut scaler = StandardScaler::new().with_mean(false).with_std(false);
    scaler.fit(&x).expect("Test data should be valid");

    let transformed = scaler.transform(&x).expect("Test data should be valid");

    // Should be identity (data unchanged)
    let (n_rows, n_cols) = x.shape();
    for i in 0..n_rows {
        for j in 0..n_cols {
            assert!(
                (x.get(i, j) - transformed.get(i, j)).abs() < 1e-6,
                "Identity transformation failed"
            );
        }
    }

    // Save and load
    let path = "test_scaler_identity.safetensors";
    scaler
        .save_safetensors(path)
        .expect("Test data should be valid");
    let loaded_scaler = StandardScaler::load_safetensors(path).expect("Test data should be valid");

    // Should still be identity after load
    let transformed_loaded = loaded_scaler
        .transform(&x)
        .expect("Test data should be valid");
    for i in 0..n_rows {
        for j in 0..n_cols {
            assert!(
                (x.get(i, j) - transformed_loaded.get(i, j)).abs() < 1e-6,
                "Identity transformation failed after load"
            );
        }
    }

    // Cleanup
    fs::remove_file(path).ok();
}
