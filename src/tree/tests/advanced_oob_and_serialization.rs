
#[test]
fn test_count_regression_tree_samples() {
    let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
        value: 5.0,
        n_samples: 15,
    });
    assert_eq!(count_regression_tree_samples(&leaf), 15);

    let node = RegressionTreeNode::Node(RegressionNode {
        feature_idx: 0,
        threshold: 0.5,
        left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
            value: 1.0,
            n_samples: 5,
        })),
        right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
            value: 2.0,
            n_samples: 8,
        })),
    });
    assert_eq!(count_regression_tree_samples(&node), 13);
}

// =========================================================================
// Coverage boost: OOB and feature importance tests
// =========================================================================

#[test]
fn test_random_forest_classifier_oob_score() {
    let x = Matrix::from_vec(
        20,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.0, 0.3, 0.1, 0.4, 0.2, 0.0, 0.3, 0.1, 0.0, 0.2, 0.1,
            0.3, 0.2, 0.1, // Class 1 samples
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8, 0.8, 1.0, 0.9, 1.1, 1.1, 0.9, 0.8, 0.8, 1.0, 0.9, 0.9,
            1.0, 1.0, 0.8,
        ],
    )
    .expect("Matrix creation should succeed");
    let y: Vec<usize> = (0..10).map(|_| 0).chain((0..10).map(|_| 1)).collect();

    let mut rf = RandomForestClassifier::new(10)
        .with_max_depth(3)
        .with_random_state(42);
    rf.fit(&x, &y).expect("Fit should succeed");

    // Test OOB prediction
    let oob_pred = rf.oob_prediction();
    assert!(oob_pred.is_some());
    assert_eq!(oob_pred.expect("Expected OOB predictions").len(), 20);

    // Test OOB score
    let oob_score = rf.oob_score();
    assert!(oob_score.is_some());
    let score = oob_score.expect("Expected OOB score");
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn test_random_forest_classifier_oob_unfitted() {
    let rf = RandomForestClassifier::new(5);
    assert!(rf.oob_prediction().is_none());
    assert!(rf.oob_score().is_none());
}

#[test]
fn test_random_forest_classifier_feature_importances() {
    let x = Matrix::from_vec(
        10,
        3,
        vec![
            0.0, 0.5, 0.1, 0.1, 0.5, 0.2, 0.0, 0.5, 0.0, 0.2, 0.5, 0.1, 0.1, 0.5, 0.3, 1.0, 0.5,
            0.8, 0.9, 0.5, 0.9, 1.0, 0.5, 1.0, 0.8, 0.5, 0.7, 0.9, 0.5, 0.8,
        ],
    )
    .expect("Matrix creation should succeed");
    let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    let mut rf = RandomForestClassifier::new(10)
        .with_max_depth(3)
        .with_random_state(42);
    rf.fit(&x, &y).expect("Fit should succeed");

    let importances = rf.feature_importances();
    assert!(importances.is_some());
    let imp = importances.expect("Expected feature importances");
    assert_eq!(imp.len(), 3);
    // Sum should be ~1.0
    let sum: f32 = imp.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn test_random_forest_classifier_feature_importances_unfitted() {
    let rf = RandomForestClassifier::new(5);
    assert!(rf.feature_importances().is_none());
}

#[test]
fn test_random_forest_regressor_oob_score() {
    let x = Matrix::from_vec(20, 1, (1..=20).map(|i| i as f32).collect())
        .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&(1..=20).map(|i| (i * 2) as f32).collect::<Vec<_>>());

    let mut rf = RandomForestRegressor::new(10)
        .with_max_depth(3)
        .with_random_state(42);
    rf.fit(&x, &y).expect("Fit should succeed");

    let oob_pred = rf.oob_prediction();
    assert!(oob_pred.is_some());

    let oob_score = rf.oob_score();
    assert!(oob_score.is_some());
}

#[test]
fn test_random_forest_regressor_oob_unfitted() {
    let rf = RandomForestRegressor::new(5);
    assert!(rf.oob_prediction().is_none());
    assert!(rf.oob_score().is_none());
}

#[test]
fn test_random_forest_regressor_feature_importances() {
    let x = Matrix::from_vec(
        10,
        2,
        vec![
            1.0, 0.5, 2.0, 0.5, 3.0, 0.5, 4.0, 0.5, 5.0, 0.5, 6.0, 0.5, 7.0, 0.5, 8.0, 0.5, 9.0,
            0.5, 10.0, 0.5,
        ],
    )
    .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

    let mut rf = RandomForestRegressor::new(10)
        .with_max_depth(3)
        .with_random_state(42);
    rf.fit(&x, &y).expect("Fit should succeed");

    let importances = rf.feature_importances();
    assert!(importances.is_some());
    let imp = importances.expect("Expected feature importances");
    assert_eq!(imp.len(), 2);
}

#[test]
fn test_random_forest_regressor_feature_importances_unfitted() {
    let rf = RandomForestRegressor::new(5);
    assert!(rf.feature_importances().is_none());
}

// =========================================================================
// Coverage boost: Default implementations
// =========================================================================

#[test]
fn test_rfr_default_coverage() {
    let rf = RandomForestRegressor::default();
    assert_eq!(rf.n_estimators, 10);
}

#[test]
fn test_dtr_default_coverage() {
    let tree = DecisionTreeRegressor::default();
    assert!(tree.tree.is_none());
}

// =========================================================================
// Coverage boost: Error handling
// =========================================================================

#[test]
fn test_dtc_fit_empty_coverage() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
    let y: Vec<usize> = vec![];
    let mut tree = DecisionTreeClassifier::new();
    let result = tree.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_dtc_fit_sample_mismatch() {
    let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
    let y = vec![0, 1]; // Wrong length
    let mut tree = DecisionTreeClassifier::new();
    let result = tree.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_dtr_fit_empty_coverage() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[]);
    let mut tree = DecisionTreeRegressor::new();
    let result = tree.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_dtr_fit_sample_mismatch() {
    let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length
    let mut tree = DecisionTreeRegressor::new();
    let result = tree.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
#[should_panic(expected = "low >= high")]
fn test_rfc_fit_empty_coverage() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
    let y: Vec<usize> = vec![];
    let mut rf = RandomForestClassifier::new(5);
    // RF doesn't check empty data explicitly and will panic in bootstrap_sample
    let _ = rf.fit(&x, &y);
}

#[test]
fn test_rfr_fit_empty_coverage() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[]);
    let mut rf = RandomForestRegressor::new(5);
    let result = rf.fit(&x, &y);
    assert!(result.is_err());
}

#[test]
fn test_rfr_fit_sample_mismatch() {
    let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[1.0, 2.0]);
    let mut rf = RandomForestRegressor::new(5);
    let result = rf.fit(&x, &y);
    assert!(result.is_err());
}

// =========================================================================
// Coverage boost: Regression tree builder options
// =========================================================================

#[test]
fn test_decision_tree_regressor_min_samples_split_coverage() {
    let tree = DecisionTreeRegressor::new().with_min_samples_split(5);
    assert_eq!(tree.min_samples_split, 5);
}

#[test]
fn test_decision_tree_regressor_min_samples_split_floor_coverage() {
    // min_samples_split should be at least 2
    let tree = DecisionTreeRegressor::new().with_min_samples_split(1);
    assert_eq!(tree.min_samples_split, 2);
}

#[test]
fn test_decision_tree_regressor_min_samples_leaf_coverage() {
    let tree = DecisionTreeRegressor::new().with_min_samples_leaf(3);
    assert_eq!(tree.min_samples_leaf, 3);
}

#[test]
fn test_decision_tree_regressor_min_samples_leaf_floor_coverage() {
    // min_samples_leaf should be at least 1
    let tree = DecisionTreeRegressor::new().with_min_samples_leaf(0);
    assert_eq!(tree.min_samples_leaf, 1);
}

#[test]
fn test_decision_tree_regressor_score_coverage() {
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("Matrix creation should succeed");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mut tree = DecisionTreeRegressor::new().with_max_depth(5);
    tree.fit(&x, &y).expect("Fit should succeed");

    let score = tree.score(&x, &y);
    // R² should be high for this simple linear relationship
    assert!(score > 0.8);
}

// =========================================================================
// Coverage boost: Gradient Boosting additional tests
// =========================================================================

#[test]
fn test_gradient_boosting_predict_proba_untrained() {
    let gbm = GradientBoostingClassifier::new();
    let x =
        Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("Matrix creation should succeed");
    let result = gbm.predict_proba(&x);
    assert!(result.is_err());
}

#[test]
fn test_gradient_boosting_n_estimators_getter() {
    let gbm = GradientBoostingClassifier::new().with_n_estimators(25);
    // Before training, n_estimators() returns 0 (number of trained estimators)
    assert_eq!(gbm.n_estimators(), 0);
}

// =========================================================================
// Coverage boost: predict_proba for RandomForestClassifier
// =========================================================================

#[test]
fn test_random_forest_predict_proba() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 1.0, 1.0, 0.9, 0.9, 1.0, 0.8],
    )
    .expect("Matrix creation should succeed");
    let y = vec![0, 0, 0, 1, 1, 1];

    let mut rf = RandomForestClassifier::new(10)
        .with_max_depth(3)
        .with_random_state(42);
    rf.fit(&x, &y).expect("Fit should succeed");

    let proba = rf.predict_proba(&x);
    let (n_samples, n_classes) = proba.shape();
    assert_eq!(n_samples, 6);
    assert_eq!(n_classes, 2);

    // Each row should sum to 1.0
    for i in 0..n_samples {
        let row_sum: f32 = (0..n_classes).map(|j| proba.get(i, j)).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

// =========================================================================
// Coverage boost: Bootstrap sample edge cases
// =========================================================================

#[test]
fn test_bootstrap_sample_no_seed() {
    let indices = bootstrap_sample(50, None);
    assert_eq!(indices.len(), 50);
    // All indices should be valid
    for &idx in &indices {
        assert!(idx < 50);
    }
}

#[test]
fn test_bootstrap_sample_different_seeds() {
    let indices1 = bootstrap_sample(50, Some(1));
    let indices2 = bootstrap_sample(50, Some(2));
    // Different seeds should (almost certainly) give different results
    assert_ne!(indices1, indices2);
}

// =========================================================================
// Coverage boost: SafeTensors serialization
// =========================================================================

#[test]
fn test_decision_tree_save_load_safetensors() {
    use std::fs;

    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation should succeed");
    let y = vec![0, 0, 1, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
    tree.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_tree_safetensors.safetensors";
    tree.save_safetensors(path).expect("Save should succeed");

    let loaded = DecisionTreeClassifier::load_safetensors(path).expect("Load should succeed");

    // Verify predictions match
    let orig_pred = tree.predict(&x);
    let loaded_pred = loaded.predict(&x);
    assert_eq!(orig_pred, loaded_pred);

    fs::remove_file(path).ok();
}

#[test]
fn test_decision_tree_save_safetensors_unfitted() {
    let tree = DecisionTreeClassifier::new();
    let result = tree.save_safetensors("/tmp/unfitted.safetensors");
    assert!(result.is_err());
}

#[test]
fn test_random_forest_save_load_safetensors() {
    use std::fs;

    let x = Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 1.0, 1.0, 0.9, 0.9, 1.0, 0.8],
    )
    .expect("Matrix creation should succeed");
    let y = vec![0, 0, 0, 1, 1, 1];

    let mut rf = RandomForestClassifier::new(3)
        .with_max_depth(2)
        .with_random_state(42);
    rf.fit(&x, &y).expect("Fit should succeed");

    let path = "/tmp/test_rf_safetensors.safetensors";
    rf.save_safetensors(path).expect("Save should succeed");

    let loaded = RandomForestClassifier::load_safetensors(path).expect("Load should succeed");

    // Verify structure
    assert_eq!(loaded.n_estimators, 3);
    assert_eq!(loaded.trees.len(), 3);

    fs::remove_file(path).ok();
}

#[test]
fn test_random_forest_save_safetensors_unfitted() {
    let rf = RandomForestClassifier::new(5);
    let result = rf.save_safetensors("/tmp/unfitted_rf.safetensors");
    assert!(result.is_err());
}
