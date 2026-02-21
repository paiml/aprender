
#[test]
fn test_random_forest_regressor_predict_nonlinear() {
    // Non-linear data: y = x²
    let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

    // Train RF (with fixed random state for reproducibility)
    let mut rf = RandomForestRegressor::new(20)
        .with_max_depth(4)
        .with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let predictions = rf.predict(&x);
    assert_eq!(predictions.len(), 8);

    // Check predictions are reasonable (allow some error due to averaging and small dataset)
    let pred_slice = predictions.as_slice();
    let y_slice = y.as_slice();
    for i in 0..8 {
        let error = (pred_slice[i] - y_slice[i]).abs();
        assert!(
            error <= 12.0,
            "Prediction {} too far from true value {}: error {}",
            pred_slice[i],
            y_slice[i],
            error
        );
    }
}

#[test]
fn test_random_forest_regressor_score() {
    let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

    let mut rf = RandomForestRegressor::new(15).with_max_depth(3);
    rf.fit(&x, &y).expect("fit should succeed");

    let r2 = rf.score(&x, &y);
    // R² should be positive and high for this simple linear pattern
    assert!(r2 > 0.7, "R² score {r2} should be high");
    assert!(r2 <= 1.0, "R² score {r2} should be <= 1.0");
}

#[test]
fn test_random_forest_regressor_n_estimators_effect() {
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]);

    // Few trees
    let mut rf_few = RandomForestRegressor::new(5).with_max_depth(4);
    rf_few.fit(&x, &y).expect("fit should succeed");
    let r2_few = rf_few.score(&x, &y);

    // Many trees
    let mut rf_many = RandomForestRegressor::new(30).with_max_depth(4);
    rf_many.fit(&x, &y).expect("fit should succeed");
    let r2_many = rf_many.score(&x, &y);

    // More trees should generally give same or better performance
    // (at least not significantly worse)
    assert!(
        r2_many >= r2_few - 0.1,
        "More trees should not hurt performance"
    );
}

#[test]
fn test_random_forest_regressor_vs_single_tree() {
    // Random forest should generalize better than single tree
    let x = Matrix::from_vec(
        15,
        1,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[
        2.1, 4.2, 8.9, 16.1, 24.8, 36.2, 49.1, 63.8, 81.2, 100.1, 120.9, 144.2, 169.1, 195.8, 225.0,
    ]);

    // Single tree with high depth (prone to overfitting)
    let mut single_tree = DecisionTreeRegressor::new().with_max_depth(10);
    single_tree.fit(&x, &y).expect("fit should succeed");
    let single_r2 = single_tree.score(&x, &y);

    // Random forest with moderate depth
    let mut rf = RandomForestRegressor::new(20).with_max_depth(6);
    rf.fit(&x, &y).expect("fit should succeed");
    let rf_r2 = rf.score(&x, &y);

    // Both should fit well, but RF typically more stable
    assert!(single_r2 > 0.8, "Single tree R²: {single_r2}");
    assert!(rf_r2 > 0.8, "Random forest R²: {rf_r2}");
}

#[test]
fn test_random_forest_regressor_multidimensional() {
    // 2D features: [x1, x2], y = x1 + 2*x2
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 1.0, 4.0, 2.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 6.0, 8.0]);

    let mut rf = RandomForestRegressor::new(15).with_max_depth(5);
    rf.fit(&x, &y).expect("fit should succeed");

    let predictions = rf.predict(&x);
    assert_eq!(predictions.len(), 8);

    let r2 = rf.score(&x, &y);
    assert!(r2 > 0.6, "R² on 2D data should be reasonable: {r2}");
}

#[test]
fn test_random_forest_regressor_constant_target() {
    // All samples have same target value
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0, 7.0]);

    let mut rf = RandomForestRegressor::new(10).with_max_depth(3);
    rf.fit(&x, &y).expect("fit should succeed");

    let predictions = rf.predict(&x);
    for &pred in predictions.as_slice() {
        assert!(
            (pred - 7.0).abs() < 1e-5,
            "Prediction {pred} should be ~7.0 for constant target"
        );
    }
}

#[test]
fn test_random_forest_regressor_single_sample() {
    let x =
        Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[5.0]);

    let mut rf = RandomForestRegressor::new(5).with_max_depth(2);
    rf.fit(&x, &y).expect("fit should succeed");

    let predictions = rf.predict(&x);
    assert_eq!(predictions.len(), 1);
    assert!(
        (predictions.as_slice()[0] - 5.0).abs() < 1e-5,
        "Single sample prediction should be exact"
    );
}

#[test]
fn test_random_forest_regressor_random_state() {
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

    // Train two forests with same random state
    let mut rf1 = RandomForestRegressor::new(10)
        .with_max_depth(4)
        .with_random_state(42);
    rf1.fit(&x, &y).expect("fit should succeed");
    let pred1 = rf1.predict(&x);

    let mut rf2 = RandomForestRegressor::new(10)
        .with_max_depth(4)
        .with_random_state(42);
    rf2.fit(&x, &y).expect("fit should succeed");
    let pred2 = rf2.predict(&x);

    // Predictions should be identical
    for (p1, p2) in pred1.as_slice().iter().zip(pred2.as_slice().iter()) {
        assert!(
            (p1 - p2).abs() < 1e-10,
            "Predictions with same random_state should match: {p1} vs {p2}"
        );
    }
}

#[test]
fn test_random_forest_regressor_validation_errors() {
    // Mismatched dimensions
    let x = Matrix::from_vec(
        5,
        2,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]); // Wrong size

    let mut rf = RandomForestRegressor::new(5);
    let result = rf.fit(&x, &y);
    assert!(result.is_err(), "Should error on mismatched dimensions");

    // Zero samples
    let x_empty = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed in tests");
    let y_empty = Vector::from_slice(&[]);
    let mut rf_empty = RandomForestRegressor::new(5);
    let result_empty = rf_empty.fit(&x_empty, &y_empty);
    assert!(result_empty.is_err(), "Should error on zero samples");
}

#[test]
#[should_panic(expected = "Cannot predict with an unfitted Random Forest")]
fn test_random_forest_regressor_predict_before_fit() {
    let rf = RandomForestRegressor::new(5);
    let x =
        Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Matrix creation should succeed in tests");
    let _ = rf.predict(&x); // Should panic
}

#[test]
fn test_random_forest_regressor_default() {
    let rf1 = RandomForestRegressor::new(10);
    let rf2 = RandomForestRegressor::default();

    assert_eq!(rf1.n_estimators, rf2.n_estimators);
    assert_eq!(rf1.max_depth, rf2.max_depth);
}

#[test]
fn test_random_forest_regressor_comparison_with_linear_regression() {
    // On non-linear data, RF should significantly outperform linear regression
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]); // y = x²

    // Train RF (with fixed random state for reproducibility)
    let mut rf = RandomForestRegressor::new(30)
        .with_max_depth(5)
        .with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");
    let rf_r2 = rf.score(&x, &y);

    // Train linear regression
    let mut lr = crate::linear_model::LinearRegression::new();
    lr.fit(&x, &y).expect("fit should succeed");
    let lr_r2 = lr.score(&x, &y);

    // RF should handle non-linearity better
    assert!(
        rf_r2 > 0.9,
        "Random forest R² {rf_r2} should be high on quadratic data"
    );
    assert!(
        lr_r2 < 0.98,
        "Linear regression R² {lr_r2} should be lower on non-linear data"
    );
    assert!(
        rf_r2 > lr_r2,
        "RF R² {rf_r2} should exceed linear R² {lr_r2} on non-linear data"
    );
}

#[test]
fn test_random_forest_regressor_max_depth_effect() {
    let x = Matrix::from_vec(
        12,
        1,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[
        1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0,
    ]);

    // Shallow trees (with fixed random state for reproducibility)
    let mut rf_shallow = RandomForestRegressor::new(15)
        .with_max_depth(2)
        .with_random_state(42);
    rf_shallow.fit(&x, &y).expect("fit should succeed");
    let r2_shallow = rf_shallow.score(&x, &y);

    // Deep trees (with fixed random state for reproducibility)
    let mut rf_deep = RandomForestRegressor::new(15)
        .with_max_depth(8)
        .with_random_state(42);
    rf_deep.fit(&x, &y).expect("fit should succeed");
    let r2_deep = rf_deep.score(&x, &y);

    // Deeper trees should capture at least as much complexity as shallow trees
    // Note: On small datasets, the difference may be minimal due to variance
    assert!(
        r2_deep >= r2_shallow - 0.01,
        "Deeper trees R² {r2_deep} should be at least as good as shallow trees R² {r2_shallow}"
    );
}
