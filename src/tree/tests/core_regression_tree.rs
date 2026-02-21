
#[test]
fn test_gradient_boosting_n_estimators_effect() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2, // class 0
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0, 1, 1, 1];

    // Few estimators
    let mut gbm_few = GradientBoostingClassifier::new()
        .with_n_estimators(5)
        .with_learning_rate(0.1);
    gbm_few.fit(&x, &y).expect("fit should succeed");

    // Many estimators
    let mut gbm_many = GradientBoostingClassifier::new()
        .with_n_estimators(50)
        .with_learning_rate(0.1);
    gbm_many.fit(&x, &y).expect("fit should succeed");

    // More estimators should generally lead to more trees (up to limit)
    assert!(gbm_many.n_estimators() >= gbm_few.n_estimators());
}

#[test]
fn test_gradient_boosting_max_depth_effect() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2, // class 0
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0, 1, 1, 1];

    // Shallow trees
    let mut gbm_shallow = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .with_max_depth(1);
    gbm_shallow.fit(&x, &y).expect("fit should succeed");
    let pred_shallow = gbm_shallow.predict(&x).expect("predict should succeed");

    // Deeper trees
    let mut gbm_deep = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .with_max_depth(5);
    gbm_deep.fit(&x, &y).expect("fit should succeed");
    let pred_deep = gbm_deep.predict(&x).expect("predict should succeed");

    // Both should make predictions
    assert_eq!(pred_shallow.len(), 6);
    assert_eq!(pred_deep.len(), 6);
}

#[test]
fn test_gradient_boosting_binary_classification() {
    // More realistic binary classification problem
    let x = Matrix::from_vec(
        10,
        2,
        vec![
            // Class 0 (bottom-left cluster)
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.0, 0.1, 0.2, // Class 1 (top-right cluster)
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8, 0.8, 1.0, 0.9, 1.1,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(30)
        .with_learning_rate(0.1)
        .with_max_depth(3);

    gbm.fit(&x, &y).expect("fit should succeed");
    let predictions = gbm.predict(&x).expect("predict should succeed");

    // Should achieve reasonable accuracy
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| *pred == *true_label)
        .count();

    // Should get at least 7 out of 10 correct for well-separated clusters
    assert!(
        correct >= 7,
        "Expected at least 7/10 correct, got {correct}/10"
    );
}

#[test]
fn test_gradient_boosting_default() {
    let gbm1 = GradientBoostingClassifier::new();
    let gbm2 = GradientBoostingClassifier::default();

    assert_eq!(
        gbm1.configured_n_estimators(),
        gbm2.configured_n_estimators()
    );
    assert!((gbm1.learning_rate() - gbm2.learning_rate()).abs() < 1e-6);
    assert_eq!(gbm1.max_depth(), gbm2.max_depth());
}

// ========================================================================
// Decision Tree Regression Tests (RED Phase - Issue #29)
// ========================================================================

#[test]
fn test_regression_tree_creation() {
    let tree = DecisionTreeRegressor::new();
    assert!(tree.tree.is_none());
    assert!(tree.max_depth.is_none());
}

#[test]
fn test_regression_tree_with_max_depth() {
    let tree = DecisionTreeRegressor::new().with_max_depth(5);
    assert_eq!(tree.max_depth, Some(5));
}

#[test]
fn test_regression_tree_fit_simple_linear() {
    // Simple linear relationship: y = 2x + 1
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);

    // Tree should learn the linear pattern reasonably well
    let pred_slice = predictions.as_slice();
    let y_slice = y.as_slice();
    for i in 0..predictions.len() {
        assert!(
            (pred_slice[i] - y_slice[i]).abs() < 2.0,
            "Prediction {} too far from true value {}",
            pred_slice[i],
            y_slice[i]
        );
    }
}

#[test]
fn test_regression_tree_predict_nonlinear() {
    // Quadratic relationship: y = x^2
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);

    // Should capture quadratic pattern with enough depth
    let mut mse_sum = 0.0_f32;
    let pred_slice = predictions.as_slice();
    let y_slice = y.as_slice();
    for i in 0..predictions.len() {
        mse_sum += (pred_slice[i] - y_slice[i]).powi(2);
    }
    let mse = mse_sum / predictions.len() as f32;

    assert!(mse < 50.0, "MSE {mse} too high for quadratic fit");
}

#[test]
fn test_regression_tree_score() {
    // Perfect predictions should give R² = 1.0
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
    tree.fit(&x, &y).expect("fit should succeed");

    let r2 = tree.score(&x, &y);

    // R² should be high for training data
    assert!(r2 > 0.5, "R² score {r2} too low");
    assert!(r2 <= 1.0, "R² score {r2} exceeds maximum");
}

#[test]
fn test_regression_tree_max_depth_limits_complexity() {
    let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

    // Shallow tree
    let mut tree_shallow = DecisionTreeRegressor::new().with_max_depth(1);
    tree_shallow.fit(&x, &y).expect("fit should succeed");
    let depth_shallow = tree_shallow
        .tree
        .as_ref()
        .expect("tree should exist after fit")
        .depth();
    assert!(
        depth_shallow <= 1,
        "Shallow tree depth {depth_shallow} exceeds max"
    );

    // Deep tree
    let mut tree_deep = DecisionTreeRegressor::new().with_max_depth(5);
    tree_deep.fit(&x, &y).expect("fit should succeed");
    let depth_deep = tree_deep
        .tree
        .as_ref()
        .expect("tree should exist after fit")
        .depth();
    assert!(depth_deep <= 5, "Deep tree depth {depth_deep} exceeds max");

    // Deeper tree should fit better
    let r2_shallow = tree_shallow.score(&x, &y);
    let r2_deep = tree_deep.score(&x, &y);
    assert!(
        r2_deep >= r2_shallow,
        "Deeper tree R²={r2_deep} should be >= shallow tree R²={r2_shallow}"
    );
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_regression_tree_predict_before_fit_panics() {
    let tree = DecisionTreeRegressor::new();
    let x =
        Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Matrix creation should succeed in tests");
    let _ = tree.predict(&x); // Should panic
}

#[test]
fn test_regression_tree_multidimensional_features() {
    // 2D features: y = x1 + 2*x2
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // y = 3
            2.0, 1.0, // y = 4
            1.0, 2.0, // y = 5
            2.0, 2.0, // y = 6
            3.0, 1.0, // y = 5
            1.0, 3.0, // y = 7
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[3.0, 4.0, 5.0, 6.0, 5.0, 7.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
    tree.fit(&x, &y).expect("fit should succeed");

    let r2 = tree.score(&x, &y);
    assert!(r2 > 0.5, "R² score {r2} too low for 2D features");
}

#[test]
fn test_regression_tree_constant_target() {
    // All y values the same
    let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);

    let mut tree = DecisionTreeRegressor::new();
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);

    // Should predict the constant value
    for &pred in predictions.as_slice() {
        assert!(
            (pred - 5.0).abs() < 1e-5,
            "Prediction {pred} should be 5.0 for constant target"
        );
    }
}

#[test]
fn test_regression_tree_single_sample() {
    let x = Matrix::from_vec(1, 1, vec![5.0]).expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[10.0]);

    let mut tree = DecisionTreeRegressor::new();
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);
    assert_eq!(predictions.len(), 1);
    assert!((predictions[0] - 10.0).abs() < 1e-5);
}

#[test]
fn test_regression_tree_fit_validation() {
    let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong size

    let mut tree = DecisionTreeRegressor::new();
    let result = tree.fit(&x, &y);

    assert!(result.is_err(), "Should error on mismatched dimensions");
}

#[test]
fn test_regression_tree_zero_samples() {
    let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[]);

    let mut tree = DecisionTreeRegressor::new();
    let result = tree.fit(&x, &y);

    assert!(result.is_err(), "Should error on zero samples");
}

#[test]
fn test_regression_tree_min_samples_split() {
    let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);

    // Tree with min_samples_split=4 should not split nodes with fewer samples
    let mut tree = DecisionTreeRegressor::new()
        .with_max_depth(5)
        .with_min_samples_split(4);

    tree.fit(&x, &y).expect("fit should succeed");

    // Should still fit successfully
    let r2 = tree.score(&x, &y);
    assert!(r2 > 0.0, "Tree with min_samples_split should still fit");
}

#[test]
fn test_regression_tree_min_samples_leaf() {
    let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

    // Tree with min_samples_leaf=3 should ensure leaves have at least 3 samples
    let mut tree = DecisionTreeRegressor::new()
        .with_max_depth(5)
        .with_min_samples_leaf(3);

    tree.fit(&x, &y).expect("fit should succeed");

    // Should fit without error
    let predictions = tree.predict(&x);
    assert_eq!(predictions.len(), 8);
}

#[test]
fn test_regression_tree_default() {
    let tree1 = DecisionTreeRegressor::new();
    let tree2 = DecisionTreeRegressor::default();

    assert_eq!(tree1.max_depth, tree2.max_depth);
    assert_eq!(tree1.tree.is_none(), tree2.tree.is_none());
}

#[test]
fn test_regression_tree_comparison_with_linear_regression() {
    // On perfectly linear data, both should perform well
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    // Train tree
    let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
    tree.fit(&x, &y).expect("fit should succeed");
    let tree_r2 = tree.score(&x, &y);

    // Train linear model
    let mut lr = crate::linear_model::LinearRegression::new();
    lr.fit(&x, &y).expect("fit should succeed");
    let lr_r2 = lr.score(&x, &y);

    // Both should achieve high R² on linear data
    assert!(tree_r2 > 0.9, "Tree R² {tree_r2} too low on linear data");
    assert!(lr_r2 > 0.99, "Linear regression R² {lr_r2} too low");
}

// ===================================================================
// Random Forest Regression Tests
// ===================================================================

#[test]
fn test_random_forest_regressor_creation() {
    let rf = RandomForestRegressor::new(10);
    assert_eq!(rf.n_estimators, 10);
    assert!(rf.trees.is_empty());
    assert!(rf.max_depth.is_none());
}

#[test]
fn test_random_forest_regressor_with_max_depth() {
    let rf = RandomForestRegressor::new(5).with_max_depth(3);
    assert_eq!(rf.max_depth, Some(3));
}

#[test]
fn test_random_forest_regressor_fit_simple_linear() {
    // Simple linear data: y = 2x + 1
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);

    let mut rf = RandomForestRegressor::new(10).with_max_depth(5);
    rf.fit(&x, &y).expect("fit should succeed");

    // Should have trained 10 trees
    assert_eq!(rf.trees.len(), 10);

    // Should make reasonable predictions
    let _predictions = rf.predict(&x);
    let r2 = rf.score(&x, &y);
    assert!(r2 > 0.8, "R² should be high on training data: {r2}");
}
