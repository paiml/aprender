
#[test]
fn test_linear_regression_save_load_safetensors() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_lr_safetensors.safetensors";
    model.save_safetensors(path).expect("Save should succeed");

    let loaded = LinearRegression::load_safetensors(path).expect("Load should succeed");
    assert!(loaded.is_fitted());
    assert!((loaded.intercept() - model.intercept()).abs() < 1e-6);

    fs::remove_file(path).ok();
}

#[test]
fn test_linear_regression_save_unfitted_error() {
    let model = LinearRegression::new();
    let result = model.save_safetensors("/tmp/unfitted.safetensors");
    assert!(result.is_err());
}

#[test]
fn test_ridge_save_load_binary() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Ridge::new(0.1);
    model.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_ridge_binary.bin";
    model.save(path).expect("Save should succeed");

    let loaded = Ridge::load(path).expect("Load should succeed");
    assert!(loaded.is_fitted());
    assert!((loaded.alpha() - model.alpha()).abs() < 1e-6);

    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_save_load_safetensors() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Ridge::new(0.1);
    model.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_ridge_safetensors.safetensors";
    model.save_safetensors(path).expect("Save should succeed");

    let loaded = Ridge::load_safetensors(path).expect("Load should succeed");
    assert!(loaded.is_fitted());
    assert!((loaded.alpha() - model.alpha()).abs() < 1e-6);

    fs::remove_file(path).ok();
}

#[test]
fn test_ridge_save_unfitted_error() {
    let model = Ridge::new(0.1);
    let result = model.save_safetensors("/tmp/unfitted_ridge.safetensors");
    assert!(result.is_err());
}

#[test]
fn test_lasso_save_load_binary() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Lasso::new(0.01);
    model.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_lasso_binary.bin";
    model.save(path).expect("Save should succeed");

    let loaded = Lasso::load(path).expect("Load should succeed");
    assert!(loaded.is_fitted());

    fs::remove_file(path).ok();
}

#[test]
fn test_lasso_save_load_safetensors() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Lasso::new(0.01);
    model.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_lasso_safetensors.safetensors";
    model.save_safetensors(path).expect("Save should succeed");

    let loaded = Lasso::load_safetensors(path).expect("Load should succeed");
    assert!(loaded.is_fitted());

    fs::remove_file(path).ok();
}

#[test]
fn test_lasso_save_unfitted_error() {
    let model = Lasso::new(0.1);
    let result = model.save_safetensors("/tmp/unfitted_lasso.safetensors");
    assert!(result.is_err());
}

#[test]
fn test_elastic_net_save_load_binary() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = ElasticNet::new(0.1, 0.5);
    model.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_elastic_binary.bin";
    model.save(path).expect("Save should succeed");

    let loaded = ElasticNet::load(path).expect("Load should succeed");
    assert!(loaded.is_fitted());

    fs::remove_file(path).ok();
}

#[test]
fn test_elastic_net_save_load_safetensors() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = ElasticNet::new(0.1, 0.5);
    model.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_elastic_safetensors.safetensors";
    model.save_safetensors(path).expect("Save should succeed");

    let loaded = ElasticNet::load_safetensors(path).expect("Load should succeed");
    assert!(loaded.is_fitted());

    fs::remove_file(path).ok();
}

#[test]
fn test_elastic_net_save_unfitted_error() {
    let model = ElasticNet::new(0.1, 0.5);
    let result = model.save_safetensors("/tmp/unfitted_elastic.safetensors");
    assert!(result.is_err());
}

// =========================================================================
// Coverage boost: Getters and builder methods
// =========================================================================

#[test]
fn test_ridge_alpha_getter() {
    let model = Ridge::new(0.5);
    assert!((model.alpha() - 0.5).abs() < 1e-6);
}

#[test]
fn test_ridge_with_intercept() {
    let model = Ridge::new(0.1).with_intercept(false);
    assert!(!model.fit_intercept);
}

#[test]
fn test_lasso_alpha_getter() {
    let model = Lasso::new(0.25);
    assert!((model.alpha() - 0.25).abs() < 1e-6);
}

#[test]
fn test_lasso_with_intercept() {
    let model = Lasso::new(0.1).with_intercept(false);
    assert!(!model.fit_intercept);
}

#[test]
fn test_elastic_net_alpha_getter() {
    let model = ElasticNet::new(0.3, 0.7);
    assert!((model.alpha() - 0.3).abs() < 1e-6);
}

#[test]
fn test_elastic_net_l1_ratio_getter() {
    let model = ElasticNet::new(0.3, 0.7);
    assert!((model.l1_ratio() - 0.7).abs() < 1e-6);
}

#[test]
fn test_elastic_net_with_intercept() {
    let model = ElasticNet::new(0.1, 0.5).with_intercept(false);
    assert!(!model.fit_intercept);
}

// =========================================================================
// Coverage boost: Error handling
// =========================================================================

#[test]
fn test_ridge_fit_empty_error() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![]);

    let mut model = Ridge::new(0.1);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_ridge_fit_dimension_mismatch() {
    let x = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length

    let mut model = Ridge::new(0.1);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_lasso_fit_empty_error() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![]);

    let mut model = Lasso::new(0.1);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_lasso_fit_dimension_mismatch() {
    let x = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length

    let mut model = Lasso::new(0.1);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_elastic_net_fit_empty_error() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![]);

    let mut model = ElasticNet::new(0.1, 0.5);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_elastic_net_fit_dimension_mismatch() {
    let x = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length

    let mut model = ElasticNet::new(0.1, 0.5);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

// =========================================================================
// Coverage boost: Score and prediction edge cases
// =========================================================================

#[test]
fn test_ridge_score_perfect() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut model = Ridge::new(0.0001); // Very small regularization
    model.fit(&x, &y).expect("Fit should succeed");

    let score = model.score(&x, &y);
    assert!(score > 0.99);
}

#[test]
fn test_lasso_score_perfect() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut model = Lasso::new(0.0001); // Very small regularization
    model.fit(&x, &y).expect("Fit should succeed");

    let score = model.score(&x, &y);
    assert!(score > 0.9);
}

#[test]
fn test_elastic_net_score_perfect() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut model = ElasticNet::new(0.0001, 0.5);
    model.fit(&x, &y).expect("Fit should succeed");

    let score = model.score(&x, &y);
    assert!(score > 0.9);
}

// =========================================================================
// Coverage boost: Debug and Clone implementations
// =========================================================================

#[test]
fn test_ridge_debug_trait() {
    let model = Ridge::new(0.1);
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("Ridge"));
}

#[test]
fn test_ridge_clone_trait() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Ridge::new(0.1);
    model.fit(&x, &y).expect("Fit should succeed");

    let cloned = model.clone();
    assert!(cloned.is_fitted());
    assert!((cloned.alpha() - model.alpha()).abs() < 1e-6);
}

#[test]
fn test_lasso_debug_trait() {
    let model = Lasso::new(0.1);
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("Lasso"));
}

#[test]
fn test_lasso_clone_trait() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Lasso::new(0.01);
    model.fit(&x, &y).expect("Fit should succeed");

    let cloned = model.clone();
    assert!(cloned.is_fitted());
}

#[test]
fn test_elastic_net_debug_trait() {
    let model = ElasticNet::new(0.1, 0.5);
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("ElasticNet"));
}

#[test]
fn test_elastic_net_clone_trait() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = ElasticNet::new(0.1, 0.5);
    model.fit(&x, &y).expect("Fit should succeed");

    let cloned = model.clone();
    assert!(cloned.is_fitted());
}

// =========================================================================
// Coverage boost: Soft threshold function (Lasso internal)
// =========================================================================

#[test]
fn test_soft_threshold_positive() {
    // x > lambda: should return x - lambda
    let result = Lasso::soft_threshold(5.0, 2.0);
    assert!((result - 3.0).abs() < 1e-6);
}

#[test]
fn test_soft_threshold_negative() {
    // x < -lambda: should return x + lambda
    let result = Lasso::soft_threshold(-5.0, 2.0);
    assert!((result - (-3.0)).abs() < 1e-6);
}

#[test]
fn test_soft_threshold_zero_region() {
    // |x| <= lambda: should return 0
    let result = Lasso::soft_threshold(1.0, 2.0);
    assert!(result.abs() < 1e-6);

    let result = Lasso::soft_threshold(-1.0, 2.0);
    assert!(result.abs() < 1e-6);

    let result = Lasso::soft_threshold(0.0, 2.0);
    assert!(result.abs() < 1e-6);
}

// =========================================================================
// Coverage boost: Add intercept column helper
// =========================================================================

#[test]
fn test_add_intercept_column_coverage() {
    let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Valid matrix dimensions for test");

    let x_with_intercept = LinearRegression::add_intercept_column(&x);
    assert_eq!(x_with_intercept.shape(), (3, 3));

    // First column should be all 1s
    assert!((x_with_intercept.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((x_with_intercept.get(1, 0) - 1.0).abs() < 1e-6);
    assert!((x_with_intercept.get(2, 0) - 1.0).abs() < 1e-6);
}

// =========================================================================
// Coverage boost: Multivariate regression
// =========================================================================

#[test]
fn test_ridge_multivariate_coverage() {
    let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[6.0, 8.0, 9.0, 11.0]);

    let mut model = Ridge::new(0.01);
    model.fit(&x, &y).expect("Fit should succeed");

    assert_eq!(model.coefficients().len(), 2);
    let score = model.score(&x, &y);
    assert!(score > 0.9);
}
