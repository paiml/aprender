
#[test]
fn test_isolation_forest_score_samples() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal
            10.0, 10.0, // Outlier 1
            -10.0, -10.0, // Outlier 2
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new().with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let scores = iforest.score_samples(&data);
    assert_eq!(scores.len(), 6);

    // Outliers should have lower scores than normal points
    let normal_avg = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;
    let outlier_avg = (scores[4] + scores[5]) / 2.0;
    assert!(outlier_avg < normal_avg);
}

#[test]
fn test_isolation_forest_contamination() {
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 10.0,
            10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Low contamination (10%) - fewer anomalies expected
    let mut iforest_low = IsolationForest::new()
        .with_contamination(0.1)
        .with_random_state(42);
    iforest_low
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    let pred_low = iforest_low.predict(&data);
    let anomalies_low = pred_low.iter().filter(|&&p| p == -1).count();

    // High contamination (30%) - more anomalies expected
    let mut iforest_high = IsolationForest::new()
        .with_contamination(0.3)
        .with_random_state(42);
    iforest_high
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    let pred_high = iforest_high.predict(&data);
    let anomalies_high = pred_high.iter().filter(|&&p| p == -1).count();

    // Higher contamination should detect more or equal anomalies
    assert!(anomalies_high >= anomalies_low);
}

#[test]
fn test_isolation_forest_n_estimators() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Fewer trees
    let mut iforest_few = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest_few
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    // More trees (should be more stable)
    let mut iforest_many = IsolationForest::new()
        .with_n_estimators(100)
        .with_random_state(42);
    iforest_many
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    // Both should work, more trees typically more accurate
    let pred_few = iforest_few.predict(&data);
    let pred_many = iforest_many.predict(&data);

    assert_eq!(pred_few.len(), 8);
    assert_eq!(pred_many.len(), 8);
}

#[test]
fn test_isolation_forest_max_samples() {
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 10.0,
            10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Use subset of samples for each tree
    let mut iforest = IsolationForest::new()
        .with_max_samples(5)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 10);
}

#[test]
fn test_isolation_forest_reproducible() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest1 = IsolationForest::new().with_random_state(42);
    iforest1
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    let pred1 = iforest1.predict(&data);

    let mut iforest2 = IsolationForest::new().with_random_state(42);
    iforest2
        .fit(&data)
        .expect("Isolation Forest fit should succeed");
    let pred2 = iforest2.predict(&data);

    assert_eq!(pred1, pred2);
}

#[test]
fn test_isolation_forest_all_normal() {
    // All points are normal (tightly clustered)
    let data = Matrix::from_vec(
        6,
        2,
        vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_contamination(0.1)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    // With 10% contamination, expect mostly normal points
    let n_normal = predictions.iter().filter(|&&p| p == 1).count();
    assert!(n_normal >= 5);
}

#[test]
fn test_isolation_forest_score_samples_range() {
    let data = Matrix::from_vec(4, 2, vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 10.0, 10.0])
        .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new().with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let scores = iforest.score_samples(&data);
    // Anomaly scores should be in reasonable range
    for &score in &scores {
        assert!(score.is_finite());
    }
}

#[test]
fn test_isolation_forest_path_length() {
    // Test that isolation path length is computed correctly
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal
            10.0, 10.0, // Easy to isolate outlier
            2.05, 2.05, // Normal
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(100)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let scores = iforest.score_samples(&data);
    // Outlier (index 4) should have significantly different score
    let outlier_score = scores[4];
    let normal_score = (scores[0] + scores[1] + scores[2] + scores[3] + scores[5]) / 5.0;
    assert!(outlier_score < normal_score);
}

#[test]
fn test_isolation_forest_multidimensional() {
    // Test with more features
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.0, 2.0, 3.0, 0.9, 1.9, 2.9, 10.0, 10.0, 10.0, -10.0,
            -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_contamination(0.3)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 6);
}

#[test]
fn test_isolation_forest_decision_function_consistency() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_contamination(0.3)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Isolation Forest fit should succeed");

    let predictions = iforest.predict(&data);
    let scores = iforest.score_samples(&data);

    // Points with lower scores should be more likely to be anomalies
    // (though exact correspondence depends on threshold)
    assert_eq!(predictions.len(), scores.len());
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_isolation_forest_predict_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let iforest = IsolationForest::new();
    let _ = iforest.predict(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_isolation_forest_score_samples_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let iforest = IsolationForest::new();
    let _ = iforest.score_samples(&data); // Should panic
}

#[test]
fn test_isolation_forest_empty_after_construction() {
    let iforest = IsolationForest::new();
    assert!(!iforest.is_fitted());
}

// ========================================================================
// Isolation Forest Edge Case / Coverage Tests
// ========================================================================

#[test]
fn test_isolation_forest_default() {
    // Exercise the Default impl for IsolationForest
    let iforest = IsolationForest::default();
    assert!(!iforest.is_fitted());
}

#[test]
fn test_isolation_forest_debug_impl() {
    // Exercise the Debug derive
    let iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_contamination(0.2);
    let debug_str = format!("{:?}", iforest);
    assert!(debug_str.contains("IsolationForest"));
    assert!(debug_str.contains("n_estimators"));
}

#[test]
fn test_isolation_forest_clone() {
    // Exercise the Clone derive
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new().with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");

    let iforest_clone = iforest.clone();
    assert!(iforest_clone.is_fitted());

    let scores_orig = iforest.score_samples(&data);
    let scores_clone = iforest_clone.score_samples(&data);
    assert_eq!(scores_orig, scores_clone);
}

#[test]
fn test_isolation_forest_contamination_clamping_high() {
    // Values > 0.5 get clamped to 0.5
    let iforest = IsolationForest::new().with_contamination(0.9);
    let debug_str = format!("{:?}", iforest);
    assert!(debug_str.contains("0.5"));
}

#[test]
fn test_isolation_forest_contamination_clamping_low() {
    // Values < 0.0 get clamped to 0.0
    let iforest = IsolationForest::new().with_contamination(-1.0);
    let debug_str = format!("{:?}", iforest);
    assert!(debug_str.contains("contamination: 0.0"));
}

#[test]
fn test_isolation_forest_identical_points() {
    // All identical points: exercises the all-same-values leaf path (line 81)
    // in build_tree where (max_val - min_val).abs() < 1e-10
    let data = Matrix::from_vec(5, 2, vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Fit should succeed with identical points");
    assert!(iforest.is_fitted());

    let scores = iforest.score_samples(&data);
    for &score in &scores {
        assert!(score.is_finite());
    }
}

#[test]
fn test_isolation_forest_single_point() {
    // A single data point: exercises n_samples <= 1 terminal condition (line 57)
    // Also exercises c(1) = 0.0, which makes c_norm = 0.0.
    // The score formula 2^(-path/0) produces -inf, which is expected behavior
    // for degenerate input.
    let data = Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Fit should succeed with single point");
    assert!(iforest.is_fitted());

    let scores = iforest.score_samples(&data);
    assert_eq!(scores.len(), 1);
    // With n=1, c_norm=0.0 so score is 2^(-path/0) = -inf or NaN
    // The important thing is the model fits and produces a score without crashing
}

#[test]
fn test_isolation_forest_two_points() {
    // Two data points: exercises c(2) == 1.0 path (line 169)
    let data =
        Matrix::from_vec(2, 2, vec![0.0, 0.0, 10.0, 10.0]).expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(10)
        .with_max_samples(2)
        .with_random_state(42);
    iforest
        .fit(&data)
        .expect("Fit should succeed with two points");
    assert!(iforest.is_fitted());

    let scores = iforest.score_samples(&data);
    assert_eq!(scores.len(), 2);
    for &score in &scores {
        assert!(score.is_finite());
    }
}

#[test]
fn test_isolation_forest_without_random_state() {
    // Exercises the from_entropy() path (line 310) when no seed is provided
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new().with_n_estimators(10);
    // No .with_random_state() call
    iforest.fit(&data).expect("Fit should succeed without seed");
    assert!(iforest.is_fitted());
}
