
#[test]
fn test_lasso_multivariate() {
    // y = 1 + 2*x1 + 3*x2
    let x = Matrix::from_vec(
        6,
        2,
        vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    )
    .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[6.0, 8.0, 9.0, 11.0, 16.0, 21.0]);

    let mut model = Lasso::new(0.01);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let r2 = model.score(&x, &y);
    assert!(r2 > 0.95, "R² should be > 0.95, got {r2}");
}

#[test]
fn test_lasso_no_intercept() {
    // y = 2x
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut model = Lasso::new(0.01).with_intercept(false);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!((model.intercept() - 0.0).abs() < 1e-6);
}

#[test]
fn test_lasso_dimension_mismatch_error() {
    let x = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length

    let mut model = Lasso::new(1.0);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_lasso_empty_data_error() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![]);

    let mut model = Lasso::new(1.0);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_lasso_clone() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

    let mut model = Lasso::new(0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let cloned = model.clone();
    assert!(cloned.is_fitted());
    assert!((cloned.alpha() - model.alpha()).abs() < 1e-6);
    assert!((cloned.intercept() - model.intercept()).abs() < 1e-6);
}

#[test]
fn test_lasso_save_load() {
    use std::fs;
    use std::path::Path;

    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Lasso::new(0.1);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let path = Path::new("/tmp/test_lasso.bin");
    model.save(path).expect("Failed to save model");

    let loaded = Lasso::load(path).expect("Failed to load model");

    assert!((loaded.alpha() - model.alpha()).abs() < 1e-6);
    let original_pred = model.predict(&x);
    let loaded_pred = loaded.predict(&x);

    for i in 0..original_pred.len() {
        assert!((original_pred[i] - loaded_pred[i]).abs() < 1e-6);
    }

    fs::remove_file(path).ok();
}

#[test]
fn test_lasso_with_intercept_builder() {
    let model = Lasso::new(1.0).with_intercept(false);

    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

    let mut model = model;
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let x_zero = Matrix::from_vec(1, 1, vec![0.0]).expect("Valid matrix dimensions for test");
    let pred = model.predict(&x_zero);

    assert!(
        pred[0].abs() < 1e-6,
        "Lasso without intercept should predict 0 at x=0"
    );
}

#[test]
fn test_lasso_coefficients_length() {
    let x = Matrix::from_vec(5, 3, vec![1.0; 15]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let mut model = Lasso::new(0.1);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert_eq!(model.coefficients().len(), 3);
}

#[test]
fn test_lasso_with_max_iter() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Lasso::new(0.1).with_max_iter(100);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!(model.is_fitted());
}

#[test]
fn test_lasso_with_tol() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = Lasso::new(0.1).with_tol(1e-6);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!(model.is_fitted());
}

#[test]
fn test_lasso_soft_threshold() {
    // Test the soft-thresholding function
    assert!((Lasso::soft_threshold(5.0, 2.0) - 3.0).abs() < 1e-6);
    assert!((Lasso::soft_threshold(-5.0, 2.0) - (-3.0)).abs() < 1e-6);
    assert!((Lasso::soft_threshold(1.0, 2.0) - 0.0).abs() < 1e-6);
    assert!((Lasso::soft_threshold(-1.0, 2.0) - 0.0).abs() < 1e-6);
}

// ==================== ElasticNet Tests ====================

#[test]
fn test_elastic_net_new() {
    let model = ElasticNet::new(1.0, 0.5);
    assert!(!model.is_fitted());
    assert!((model.alpha() - 1.0).abs() < 1e-6);
    assert!((model.l1_ratio() - 0.5).abs() < 1e-6);
}

#[test]
fn test_elastic_net_simple() {
    // y = 2x + 1
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = ElasticNet::new(0.01, 0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!(model.is_fitted());

    // Should recover approximately y = 2x + 1
    let coef = model.coefficients();
    assert!((coef[0] - 2.0).abs() < 0.5); // Some regularization effect
    assert!((model.intercept() - 1.0).abs() < 1.0);
}

#[test]
fn test_elastic_net_multivariate() {
    // y = 2*x1 + 3*x2
    let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[5.0, 7.0, 8.0, 10.0]);

    let mut model = ElasticNet::new(0.01, 0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let predictions = model.predict(&x);
    for i in 0..4 {
        assert!((predictions[i] - y[i]).abs() < 1.0);
    }
}

#[test]
fn test_elastic_net_l1_ratio_pure_l1() {
    // l1_ratio=1.0 should behave like Lasso
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut elastic = ElasticNet::new(0.1, 1.0);
    elastic
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let mut lasso = Lasso::new(0.1);
    lasso
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Should have similar coefficients
    let elastic_coef = elastic.coefficients();
    let lasso_coef = lasso.coefficients();
    assert!((elastic_coef[0] - lasso_coef[0]).abs() < 0.1);
}

#[test]
fn test_elastic_net_l1_ratio_pure_l2() {
    // l1_ratio=0.0 should behave like Ridge
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut elastic = ElasticNet::new(0.1, 0.0);
    elastic
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let mut ridge = Ridge::new(0.1);
    ridge
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    // Should have similar coefficients
    let elastic_coef = elastic.coefficients();
    let ridge_coef = ridge.coefficients();
    assert!((elastic_coef[0] - ridge_coef[0]).abs() < 0.5);
}

#[test]
fn test_elastic_net_dimension_mismatch() {
    let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length

    let mut model = ElasticNet::new(0.1, 0.5);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_elastic_net_empty_data() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Valid matrix dimensions for test");
    let y = Vector::from_vec(vec![]);

    let mut model = ElasticNet::new(0.1, 0.5);
    let result = model.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_elastic_net_predict_not_fitted() {
    let model = ElasticNet::new(0.1, 0.5);
    let x = Matrix::from_vec(1, 1, vec![1.0]).expect("Valid matrix dimensions for test");
    let _ = model.predict(&x);
}

#[test]
fn test_elastic_net_score() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = ElasticNet::new(0.01, 0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let r2 = model.score(&x, &y);
    assert!(r2 > 0.9); // Should fit well with small alpha
}

#[test]
fn test_elastic_net_clone() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

    let mut model = ElasticNet::new(0.5, 0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let cloned = model.clone();
    assert!(cloned.is_fitted());
    assert!((cloned.alpha() - model.alpha()).abs() < 1e-6);
    assert!((cloned.l1_ratio() - model.l1_ratio()).abs() < 1e-6);
    assert!((cloned.intercept() - model.intercept()).abs() < 1e-6);
}

#[test]
fn test_elastic_net_save_load() {
    use std::fs;
    use std::path::Path;

    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = ElasticNet::new(0.1, 0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let path = Path::new("/tmp/test_elastic_net.bin");
    model.save(path).expect("Failed to save model");

    let loaded = ElasticNet::load(path).expect("Failed to load model");

    assert!((loaded.alpha() - model.alpha()).abs() < 1e-6);
    assert!((loaded.l1_ratio() - model.l1_ratio()).abs() < 1e-6);
    let original_pred = model.predict(&x);
    let loaded_pred = loaded.predict(&x);

    for i in 0..original_pred.len() {
        assert!((original_pred[i] - loaded_pred[i]).abs() < 1e-6);
    }

    fs::remove_file(path).ok();
}

#[test]
fn test_elastic_net_with_intercept_builder() {
    let model = ElasticNet::new(1.0, 0.5).with_intercept(false);

    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

    let mut model = model;
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    let x_zero = Matrix::from_vec(1, 1, vec![0.0]).expect("Valid matrix dimensions for test");
    let pred = model.predict(&x_zero);
    assert!((pred[0] - 0.0).abs() < 1e-6); // No intercept
}

#[test]
fn test_elastic_net_multivariate_coefficients() {
    let x = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[6.0, 15.0, 24.0]);

    let mut model = ElasticNet::new(0.1, 0.5);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert_eq!(model.coefficients().len(), 3);
}

#[test]
fn test_elastic_net_with_max_iter() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = ElasticNet::new(0.1, 0.5).with_max_iter(100);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!(model.is_fitted());
}

#[test]
fn test_elastic_net_with_tol() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = ElasticNet::new(0.1, 0.5).with_tol(1e-6);
    model
        .fit(&x, &y)
        .expect("Fit should succeed with valid test data");

    assert!(model.is_fitted());
}

// =========================================================================
// Coverage boost tests: SafeTensors and binary serialization
// =========================================================================

#[test]
fn test_linear_regression_save_load_binary() {
    let x =
        Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Valid matrix dimensions for test");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_lr_binary.bin";
    model.save(path).expect("Save should succeed");

    let loaded = LinearRegression::load(path).expect("Load should succeed");
    assert!(loaded.is_fitted());
    assert!((loaded.intercept() - model.intercept()).abs() < 1e-6);

    fs::remove_file(path).ok();
}
