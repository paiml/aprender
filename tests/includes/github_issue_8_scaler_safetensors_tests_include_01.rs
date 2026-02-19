
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
