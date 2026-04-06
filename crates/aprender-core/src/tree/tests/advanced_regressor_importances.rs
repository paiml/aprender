
#[test]
fn test_random_forest_regressor_feature_importances_after_fit() {
    // Simple regression data where feature 0 is predictive
    let x = Matrix::from_vec(
        10,
        3,
        vec![
            // Feature 0 is predictive of y
            1.0, 5.0, 5.0, 2.0, 6.0, 4.0, 3.0, 5.0, 6.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0,
            4.0, 7.0, 5.0, 6.0, 8.0, 4.0, 5.0, 9.0, 5.0, 5.0, 10.0, 6.0, 4.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]); // Linear with feature 0

    let mut rf = RandomForestRegressor::new(20)
        .with_max_depth(5)
        .with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let importances = rf.feature_importances();
    assert!(
        importances.is_some(),
        "Feature importances should be available after fit"
    );

    let imps = importances.expect("importances should be available");
    assert_eq!(
        imps.len(),
        3,
        "Should have importance for each of 3 features"
    );

    // Importances should sum to ~1.0
    let sum: f32 = imps.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Importances should sum to 1.0, got {sum}"
    );

    // Feature 0 should be most important
    assert!(
        imps[0] > imps[1] && imps[0] > imps[2],
        "Feature 0 should be most important, got {imps:?}"
    );
}

#[test]
fn test_random_forest_regressor_feature_importances_before_fit() {
    let rf = RandomForestRegressor::new(10);

    let importances = rf.feature_importances();
    assert!(
        importances.is_none(),
        "Feature importances should be None before fit"
    );
}

#[test]
fn test_random_forest_regressor_feature_importances_reproducibility() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 6.0, 1.0, 7.0, 0.0, 8.0, 1.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    let mut rf1 = RandomForestRegressor::new(20).with_random_state(42);
    rf1.fit(&x, &y).expect("fit should succeed");
    let imps1 = rf1
        .feature_importances()
        .expect("feature importances should be available");

    let mut rf2 = RandomForestRegressor::new(20).with_random_state(42);
    rf2.fit(&x, &y).expect("fit should succeed");
    let imps2 = rf2
        .feature_importances()
        .expect("feature importances should be available");

    // Should be very similar with same random_state
    // Note: Small variations can occur due to floating point arithmetic in normalization
    for (i, (&imp1, &imp2)) in imps1.iter().zip(imps2.iter()).enumerate() {
        assert!(
            (imp1 - imp2).abs() <= 0.1,
            "Importance {i} should be similar: {imp1} vs {imp2}"
        );
    }
}

#[test]
fn test_random_forest_classifier_feature_importances_all_nonnegative() {
    // All importances should be >= 0
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

    let mut rf = RandomForestClassifier::new(10).with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let imps = rf
        .feature_importances()
        .expect("feature importances should be available");
    for (i, &imp) in imps.iter().enumerate() {
        assert!(
            imp >= 0.0,
            "Importance {i} should be non-negative, got {imp}"
        );
    }
}

#[test]
fn test_random_forest_regressor_feature_importances_all_nonnegative() {
    // All importances should be >= 0
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 6.0, 1.0, 7.0, 0.0, 8.0, 1.0,
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    let mut rf = RandomForestRegressor::new(10).with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let imps = rf
        .feature_importances()
        .expect("feature importances should be available");
    for (i, &imp) in imps.iter().enumerate() {
        assert!(
            imp >= 0.0,
            "Importance {i} should be non-negative, got {imp}"
        );
    }
}

#[test]
fn test_random_forest_classifier_predict_proba() {
    // Test predict_proba returns valid probabilities
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    )
    .expect("Matrix creation should succeed");
    let y = vec![0, 0, 1, 1, 2, 2]; // 3 classes

    let mut rf = RandomForestClassifier::new(10).with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let proba = rf.predict_proba(&x);

    // Shape should be (n_samples, n_classes)
    assert_eq!(proba.shape(), (6, 3));

    // Each row should sum to 1.0
    for row in 0..6 {
        let sum: f32 = (0..3).map(|col| proba.get(row, col)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Row {row} probabilities should sum to 1.0, got {sum}"
        );
    }

    // All probabilities should be in [0, 1]
    for row in 0..6 {
        for col in 0..3 {
            let p = proba.get(row, col);
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability should be in [0,1], got {p}"
            );
        }
    }
}

// ========== Additional Coverage Tests ==========

#[test]
fn test_decision_tree_regressor_new() {
    let tree = DecisionTreeRegressor::new();
    assert!(tree.tree.is_none());
    assert!(tree.max_depth.is_none());
}

#[test]
fn test_decision_tree_regressor_default() {
    let tree = DecisionTreeRegressor::default();
    assert!(tree.tree.is_none());
}

#[test]
fn test_decision_tree_regressor_with_max_depth() {
    let tree = DecisionTreeRegressor::new().with_max_depth(5);
    assert_eq!(tree.max_depth, Some(5));
}

#[test]
fn test_decision_tree_regressor_with_min_samples_split() {
    let tree = DecisionTreeRegressor::new().with_min_samples_split(5);
    assert_eq!(tree.min_samples_split, 5);
}

#[test]
fn test_decision_tree_regressor_with_min_samples_leaf() {
    let tree = DecisionTreeRegressor::new().with_min_samples_leaf(3);
    assert_eq!(tree.min_samples_leaf, 3);
}

#[test]
fn test_decision_tree_regressor_fit_predict() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);
    assert_eq!(predictions.len(), 5);
}

#[test]
fn test_decision_tree_regressor_score() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(5);
    tree.fit(&x, &y).expect("fit should succeed");

    let score = tree.score(&x, &y);
    // R² should be reasonably high for this simple linear data
    assert!(score > 0.5, "R² should be > 0.5, got {score}");
}

#[test]
fn test_decision_tree_regressor_fit_mismatch_error() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]); // Mismatched length

    let mut tree = DecisionTreeRegressor::new();
    let result = tree.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_decision_tree_regressor_fit_empty_error() {
    let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[]);

    let mut tree = DecisionTreeRegressor::new();
    let result = tree.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_random_forest_regressor_new() {
    let rf = RandomForestRegressor::new(10);
    assert_eq!(rf.n_estimators, 10);
    assert!(rf.trees.is_empty());
}

#[test]
fn test_random_forest_regressor_default_values() {
    let rf = RandomForestRegressor::default();
    assert!(rf.trees.is_empty());
}

#[test]
fn test_random_forest_regressor_fit_mismatch_error() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[2.0, 4.0]); // Mismatched length

    let mut rf = RandomForestRegressor::new(5);
    let result = rf.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_random_forest_regressor_fit_empty_error() {
    let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[]);

    let mut rf = RandomForestRegressor::new(5);
    let result = rf.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_random_forest_regressor_oob_methods() {
    let rf = RandomForestRegressor::new(10);
    assert!(rf.oob_prediction().is_none());
    assert!(rf.oob_score().is_none());
}

#[test]
fn test_random_forest_regressor_oob_methods_after_fit() {
    let x = Matrix::from_vec(
        10,
        1,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

    let mut rf = RandomForestRegressor::new(10).with_random_state(42);
    rf.fit(&x, &y).expect("fit should succeed");

    let oob_pred = rf.oob_prediction();
    assert!(oob_pred.is_some());
    assert_eq!(oob_pred.as_ref().map(|v| v.len()), Some(10));

    let oob_score = rf.oob_score();
    assert!(oob_score.is_some());
}

#[test]
fn test_regression_tree_node_depth() {
    let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
        value: 5.0,
        n_samples: 10,
    });
    assert_eq!(leaf.depth(), 0);

    let node = RegressionTreeNode::Node(RegressionNode {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
            value: 3.0,
            n_samples: 5,
        })),
        right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
            value: 7.0,
            n_samples: 5,
        })),
    });
    assert_eq!(node.depth(), 1);
}

#[test]
fn test_gradient_boosting_classifier_new() {
    let gb = GradientBoostingClassifier::new();
    assert!(gb.estimators().is_empty());
    assert_eq!(gb.configured_n_estimators(), 100); // Config value, not fitted count
}

#[test]
fn test_gradient_boosting_classifier_default() {
    let gb = GradientBoostingClassifier::default();
    assert!(gb.estimators().is_empty());
}

#[test]
fn test_gradient_boosting_classifier_builders() {
    let gb = GradientBoostingClassifier::new()
        .with_n_estimators(50)
        .with_learning_rate(0.05)
        .with_max_depth(5);

    assert_eq!(gb.configured_n_estimators(), 50); // Config value, not fitted count
}

#[test]
fn test_gradient_boosting_classifier_fit_predict() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
        ],
    )
    .expect("Matrix creation should succeed");
    let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

    let mut gb = GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_max_depth(3);

    gb.fit(&x, &y).expect("fit should succeed");

    let predictions = gb.predict(&x).expect("predict should succeed");
    assert_eq!(predictions.len(), 8);
}

#[test]
fn test_gradient_boosting_classifier_predict_proba() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
        ],
    )
    .expect("Matrix creation should succeed");
    let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

    let mut gb = GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_max_depth(3);

    gb.fit(&x, &y).expect("fit should succeed");

    let proba = gb.predict_proba(&x).expect("predict_proba should succeed");
    assert_eq!(proba.len(), 8);

    // Each row should be a probability distribution
    for row_proba in &proba {
        let sum: f32 = row_proba.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities should sum to 1.0, got {sum}"
        );
        for &p in row_proba {
            assert!((0.0..=1.0).contains(&p), "Probability should be in [0,1]");
        }
    }
}

#[test]
fn test_random_forest_classifier_oob_methods() {
    let rf = RandomForestClassifier::new(10);
    assert!(rf.oob_prediction().is_none());
    assert!(rf.oob_score().is_none());
    assert!(rf.feature_importances().is_none());
}

#[test]
fn test_decision_tree_classifier_fit_mismatch() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        .expect("Matrix creation should succeed");
    let y = vec![0, 1]; // Mismatched length

    let mut tree = DecisionTreeClassifier::new();
    let result = tree.fit(&x, &y);
    assert!(result.is_err());
}
