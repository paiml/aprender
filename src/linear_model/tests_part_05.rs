
#[test]
fn test_lasso_multivariate_coverage() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[6.0, 8.0, 9.0, 11.0]);

    let mut model = Lasso::new(0.001);
    model.fit(&x, &y).expect("Fit should succeed");

    assert_eq!(model.coefficients().len(), 2);
}

#[test]
fn test_elastic_net_multivariate_coverage() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[6.0, 8.0, 9.0, 11.0]);

    let mut model = ElasticNet::new(0.001, 0.5);
    model.fit(&x, &y).expect("Fit should succeed");

    assert_eq!(model.coefficients().len(), 2);
}

// =========================================================================
// Coverage boost: Load error paths
// =========================================================================

#[test]
fn test_linear_regression_load_nonexistent_file() {
    let result = LinearRegression::load("/tmp/aprender_nonexistent_lr_model.bin");
    assert!(result.is_err());
}

#[test]
fn test_ridge_load_nonexistent_file() {
    let result = Ridge::load("/tmp/aprender_nonexistent_ridge_model.bin");
    assert!(result.is_err());
}

#[test]
fn test_lasso_load_nonexistent_file() {
    let result = Lasso::load("/tmp/aprender_nonexistent_lasso_model.bin");
    assert!(result.is_err());
}

#[test]
fn test_elastic_net_load_nonexistent_file() {
    let result = ElasticNet::load("/tmp/aprender_nonexistent_en_model.bin");
    assert!(result.is_err());
}

#[test]
fn test_linear_regression_load_safetensors_nonexistent() {
    let result = LinearRegression::load_safetensors("/tmp/aprender_nonexistent_lr.safetensors");
    assert!(result.is_err());
}

// =========================================================================
// Coverage boost: ElasticNet l1_ratio clamping
// =========================================================================

#[test]
fn test_elastic_net_l1_ratio_clamped_high() {
    let model = ElasticNet::new(0.1, 2.0); // l1_ratio > 1.0 should be clamped to 1.0
    assert!((model.l1_ratio() - 1.0).abs() < 1e-6);
}

#[test]
fn test_elastic_net_l1_ratio_clamped_low() {
    let model = ElasticNet::new(0.1, -0.5); // l1_ratio < 0.0 should be clamped to 0.0
    assert!(model.l1_ratio().abs() < 1e-6);
}

#[test]
fn test_elastic_net_l1_ratio_boundary_zero() {
    let model = ElasticNet::new(0.1, 0.0);
    assert!(model.l1_ratio().abs() < 1e-6);
}

#[test]
fn test_elastic_net_l1_ratio_boundary_one() {
    let model = ElasticNet::new(0.1, 1.0);
    assert!((model.l1_ratio() - 1.0).abs() < 1e-6);
}

// =========================================================================
// Coverage boost: ElasticNet builder methods
// =========================================================================

#[test]
fn test_elastic_net_with_intercept_false() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut model = ElasticNet::new(0.001, 0.5).with_intercept(false);
    model.fit(&x, &y).expect("Fit should succeed");
    assert_eq!(model.coefficients().len(), 1);
}

#[test]
fn test_elastic_net_with_max_iter_debug() {
    let model = ElasticNet::new(0.1, 0.5).with_max_iter(500);
    let _ = format!("{:?}", model); // Exercise Debug impl
}

#[test]
fn test_elastic_net_with_tol_debug() {
    let model = ElasticNet::new(0.1, 0.5).with_tol(1e-6);
    let _ = format!("{:?}", model);
}

// =========================================================================
// Coverage boost: Save/Load roundtrip for various models
// =========================================================================

#[test]
fn test_linear_regression_save_load_roundtrip() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Fit should succeed");

    let dir = std::env::temp_dir().join("aprender_test_lr_save_load");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("lr_model.bin");

    model.save(&path).expect("Save should succeed");
    let loaded = LinearRegression::load(&path).expect("Load should succeed");

    let orig_preds = model.predict(&x);
    let loaded_preds = loaded.predict(&x);
    for (a, b) in orig_preds
        .as_slice()
        .iter()
        .zip(loaded_preds.as_slice().iter())
    {
        assert!((a - b).abs() < 1e-6);
    }

    std::fs::remove_dir_all(&dir).ok();
}
