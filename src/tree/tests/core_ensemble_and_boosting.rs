
#[test]
fn test_score_partial() {
    use crate::primitives::Matrix;

    // Train on simple data
    let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
        .expect("Matrix creation should succeed in tests");
    let y_train = vec![0, 0, 1, 1];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(1);
    tree.fit(&x_train, &y_train).expect("fit should succeed");

    // Score should be between 0 and 1
    let accuracy = tree.score(&x_train, &y_train);
    assert!((0.0..=1.0).contains(&accuracy));
}

#[test]
fn test_multiclass_classification() {
    use crate::primitives::Matrix;

    // 3-class problem
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("fit should succeed");

    let predictions = tree.predict(&x);
    assert_eq!(predictions.len(), 6);
    // Should classify perfectly
    assert_eq!(predictions, vec![0, 0, 1, 1, 2, 2]);
}

#[test]
fn test_save_load() {
    use crate::primitives::Matrix;
    use std::fs;
    use std::path::Path;

    // Train a tree
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2];

    let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
    tree.fit(&x, &y).expect("fit should succeed");

    // Save model
    let path = Path::new("/tmp/test_decision_tree.bin");
    tree.save(path).expect("Failed to save model");

    // Load model
    let loaded = DecisionTreeClassifier::load(path).expect("Failed to load model");

    // Verify predictions match
    let original_pred = tree.predict(&x);
    let loaded_pred = loaded.predict(&x);
    assert_eq!(original_pred, loaded_pred);

    // Verify accuracy matches
    let original_score = tree.score(&x, &y);
    let loaded_score = loaded.score(&x, &y);
    assert!((original_score - loaded_score).abs() < 1e-6);

    // Cleanup
    fs::remove_file(path).ok();
}

// Random Forest Tests

#[test]
fn test_bootstrap_sample_size() {
    let indices = bootstrap_sample(100, Some(42));
    assert_eq!(
        indices.len(),
        100,
        "Bootstrap sample should have same size as original"
    );
}

#[test]
fn test_bootstrap_sample_reproducible() {
    let indices1 = bootstrap_sample(50, Some(42));
    let indices2 = bootstrap_sample(50, Some(42));
    assert_eq!(
        indices1, indices2,
        "Same seed should give same bootstrap sample"
    );
}

#[test]
fn test_random_forest_creation() {
    let rf = RandomForestClassifier::new(10);
    assert_eq!(rf.n_estimators, 10);
}

#[test]
fn test_random_forest_builder() {
    let rf = RandomForestClassifier::new(5)
        .with_max_depth(3)
        .with_random_state(42);
    assert_eq!(rf.n_estimators, 5);
    assert_eq!(rf.max_depth, Some(3));
    assert_eq!(rf.random_state, Some(42));
}

#[test]
fn test_random_forest_fit_basic() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2];

    let mut rf = RandomForestClassifier::new(3)
        .with_max_depth(3)
        .with_random_state(42);

    rf.fit(&x, &y).expect("Fit should succeed");

    // Should have trained the correct number of trees
    assert_eq!(rf.trees.len(), 3, "Should have 3 trees");
}

#[test]
fn test_random_forest_predict() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 1.0, // class 0
            1.5, 1.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            9.0, 9.0, // class 2
            9.5, 9.5, // class 2
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2];

    let mut rf = RandomForestClassifier::new(5)
        .with_max_depth(5)
        .with_random_state(42);

    rf.fit(&x, &y).expect("fit should succeed");
    let predictions = rf.predict(&x);

    assert_eq!(predictions.len(), 6, "Should predict for all samples");

    // Perfect separation - should get perfect accuracy
    let score = rf.score(&x, &y);
    assert!(
        score > 0.8,
        "Random Forest should achieve >80% accuracy on simple data"
    );
}

#[test]
fn test_random_forest_reproducible() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            0.0, 0.0, // class 0
            0.5, 0.5, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            10.0, 10.0, // class 2
            10.5, 10.5, // class 2
            1.0, 1.0, // class 0
            6.0, 6.0, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1, 2, 2, 0, 1];

    let mut rf1 = RandomForestClassifier::new(5).with_random_state(42);
    rf1.fit(&x, &y).expect("fit should succeed");
    let pred1 = rf1.predict(&x);

    let mut rf2 = RandomForestClassifier::new(5).with_random_state(42);
    rf2.fit(&x, &y).expect("fit should succeed");
    let pred2 = rf2.predict(&x);

    assert_eq!(
        pred1, pred2,
        "Same random state should give same predictions"
    );
}

// ===== Gradient Boosting Tests =====

#[test]
fn test_gradient_boosting_new() {
    let gbm = GradientBoostingClassifier::new();
    assert_eq!(gbm.configured_n_estimators(), 100);
    assert!((gbm.learning_rate() - 0.1).abs() < 1e-6);
    assert_eq!(gbm.max_depth(), 3);
    assert_eq!(gbm.n_estimators(), 0); // No estimators before fit
}

#[test]
fn test_gradient_boosting_builder() {
    let gbm = GradientBoostingClassifier::new()
        .with_n_estimators(50)
        .with_learning_rate(0.05)
        .with_max_depth(5);

    assert_eq!(gbm.configured_n_estimators(), 50);
    assert!((gbm.learning_rate() - 0.05).abs() < 1e-6);
    assert_eq!(gbm.max_depth(), 5);
}

#[test]
fn test_gradient_boosting_fit_simple() {
    // Simple linearly separable data
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 1
            1.0, 1.0, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_learning_rate(0.1)
        .with_max_depth(2);

    let result = gbm.fit(&x, &y);
    assert!(result.is_ok());
    assert!(gbm.n_estimators() > 0); // Should have fitted some trees
}

#[test]
fn test_gradient_boosting_predict_simple() {
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 1
            1.0, 1.0, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .with_learning_rate(0.1)
        .with_max_depth(2);

    gbm.fit(&x, &y).expect("fit should succeed");
    let predictions = gbm.predict(&x).expect("predict should succeed");

    assert_eq!(predictions.len(), 4);

    // GBM should classify correctly with enough iterations
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| *pred == *true_label)
        .count();

    // Should get at least 3 out of 4 correct
    assert!(correct >= 3);
}

#[test]
fn test_gradient_boosting_predict_proba() {
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 1
            1.0, 1.0, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1, 1];

    let mut gbm = GradientBoostingClassifier::new()
        .with_n_estimators(20)
        .with_learning_rate(0.1)
        .with_max_depth(2);

    gbm.fit(&x, &y).expect("fit should succeed");
    let probas = gbm.predict_proba(&x).expect("predict_proba should succeed");

    assert_eq!(probas.len(), 4);

    // Each sample should have 2 probabilities (class 0 and class 1)
    for probs in &probas {
        assert_eq!(probs.len(), 2);
        // Probabilities should sum to ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        // Each probability should be between 0 and 1
        for &p in probs {
            assert!((0.0..=1.0).contains(&p));
        }
    }
}

#[test]
fn test_gradient_boosting_predict_untrained() {
    let gbm = GradientBoostingClassifier::new();
    let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation should succeed in tests");

    let result = gbm.predict(&x);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail when predicting with untrained model"),
        "Model not trained yet"
    );
}

#[test]
fn test_gradient_boosting_empty_data() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed in tests");
    let y = vec![];

    let mut gbm = GradientBoostingClassifier::new();
    let result = gbm.fit(&x, &y);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with empty data"),
        "Cannot fit with 0 samples"
    );
}

#[test]
fn test_gradient_boosting_mismatched_samples() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 1]; // Wrong length

    let mut gbm = GradientBoostingClassifier::new();
    let result = gbm.fit(&x, &y);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with mismatched sample counts"),
        "x and y must have the same number of samples"
    );
}

#[test]
fn test_gradient_boosting_learning_rate_effect() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, // class 0
            0.1, 0.1, // class 0
            0.0, 0.2, // class 0
            1.0, 1.0, // class 1
            0.9, 0.9, // class 1
            1.0, 0.8, // class 1
        ],
    )
    .expect("Matrix creation should succeed in tests");
    let y = vec![0, 0, 0, 1, 1, 1];

    // High learning rate
    let mut gbm_high_lr = GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_learning_rate(0.5);
    gbm_high_lr.fit(&x, &y).expect("fit should succeed");
    let pred_high = gbm_high_lr.predict(&x).expect("predict should succeed");

    // Low learning rate
    let mut gbm_low_lr = GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_learning_rate(0.01);
    gbm_low_lr.fit(&x, &y).expect("fit should succeed");
    let pred_low = gbm_low_lr.predict(&x).expect("predict should succeed");

    // Both should make predictions
    assert_eq!(pred_high.len(), 6);
    assert_eq!(pred_low.len(), 6);
}
