
#[test]
fn test_isolation_forest_single_feature() {
    // Exercise with 1D data
    let data = Matrix::from_vec(5, 1, vec![1.0, 1.1, 1.2, 1.0, 100.0])
        .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_contamination(0.2)
        .with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 5);

    // The outlier at 100.0 should likely be flagged
    let scores = iforest.score_samples(&data);
    // Outlier should have more negative score (more anomalous)
    assert!(scores[4] < scores[0]);
}

#[test]
fn test_isolation_forest_max_samples_larger_than_data() {
    // max_samples > n_samples: exercises default clamping n_samples.min(256)
    let data = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_max_samples(1000)
        .with_n_estimators(10)
        .with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");
    assert!(iforest.is_fitted());
}

#[test]
fn test_isolation_forest_contamination_zero() {
    // With contamination=0, threshold_idx=0
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_contamination(0.0)
        .with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");

    let predictions = iforest.predict(&data);
    assert_eq!(predictions.len(), 6);
}

#[test]
fn test_isolation_forest_refit_clears_trees() {
    // Exercises the self.trees.clear() path (line 314) on refit
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(10)
        .with_random_state(42);
    iforest.fit(&data).expect("First fit should succeed");

    let scores1 = iforest.score_samples(&data);

    // Refit on same data - exercises clear + rebuild
    iforest.fit(&data).expect("Second fit should succeed");

    let scores2 = iforest.score_samples(&data);
    // Same seed + same data = same scores
    assert_eq!(scores1, scores2);
}

#[test]
fn test_isolation_forest_many_features() {
    // Higher dimensionality exercises random feature selection more thoroughly
    let data = Matrix::from_vec(
        5,
        5,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 2.1, 3.1, 4.1, 5.1, 1.0, 2.0, 3.0, 4.0, 5.0, 0.9, 1.9,
            2.9, 3.9, 4.9, 50.0, 50.0, 50.0, 50.0, 50.0, // Outlier
        ],
    )
    .expect("Matrix creation should succeed");

    let mut iforest = IsolationForest::new()
        .with_n_estimators(50)
        .with_contamination(0.2)
        .with_random_state(42);
    iforest.fit(&data).expect("Fit should succeed");

    let scores = iforest.score_samples(&data);
    assert_eq!(scores.len(), 5);
    // Outlier at index 4 should have the most negative score
    let min_idx = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Valid floats"))
        .map(|(i, _)| i)
        .expect("Non-empty scores");
    assert_eq!(min_idx, 4);
}

// ========================================================================
// Local Outlier Factor (LOF) Tests
// ========================================================================

#[test]
fn test_lof_new() {
    let lof = LocalOutlierFactor::new();
    assert!(!lof.is_fitted());
}

#[test]
fn test_lof_fit_basic() {
    // Normal data clustered around (2, 2)
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 1.9,
            1.9, 2.1, 1.8,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(5);
    lof.fit(&data).expect("LOF fit should succeed");
    assert!(lof.is_fitted());
}

#[test]
fn test_lof_predict_anomalies() {
    // 8 normal points + 2 outliers
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 2.2, 2.0, 2.0, 2.2, 10.0,
            10.0, // Outlier 1
            -10.0, -10.0, // Outlier 2
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(5)
        .with_contamination(0.2);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    assert_eq!(predictions.len(), 10);

    // Check that predictions are either 1 (normal) or -1 (anomaly)
    for &pred in &predictions {
        assert!(pred == 1 || pred == -1);
    }

    // Should detect approximately 2 anomalies (20% contamination)
    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    assert!((1..=3).contains(&n_anomalies));
}

#[test]
fn test_lof_score_samples() {
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

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);
    assert_eq!(scores.len(), 6);

    // Outliers should have higher LOF scores than normal points
    let normal_avg = (scores[0] + scores[1] + scores[2] + scores[3]) / 4.0;
    let outlier_avg = (scores[4] + scores[5]) / 2.0;
    assert!(outlier_avg > normal_avg);
}

#[test]
fn test_lof_negative_outlier_factor() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let nof = lof.negative_outlier_factor();
    assert_eq!(nof.len(), 6);

    // Negative outlier factor should be opposite sign of LOF scores
    let scores = lof.score_samples(&data);
    for i in 0..6 {
        // NOF should be negative of LOF (approximately)
        assert!(nof[i] < 0.0 || scores[i] < 1.0);
    }
}

#[test]
fn test_lof_contamination() {
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
    let mut lof_low = LocalOutlierFactor::new()
        .with_contamination(0.1)
        .with_n_neighbors(5);
    lof_low.fit(&data).expect("LOF fit should succeed");
    let pred_low = lof_low.predict(&data);
    let anomalies_low = pred_low.iter().filter(|&&p| p == -1).count();

    // High contamination (30%) - more anomalies expected
    let mut lof_high = LocalOutlierFactor::new()
        .with_contamination(0.3)
        .with_n_neighbors(5);
    lof_high.fit(&data).expect("LOF fit should succeed");
    let pred_high = lof_high.predict(&data);
    let anomalies_high = pred_high.iter().filter(|&&p| p == -1).count();

    // Higher contamination should detect more or equal anomalies
    assert!(anomalies_high >= anomalies_low);
}

#[test]
fn test_lof_n_neighbors() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0, 10.0, 10.0, -10.0, -10.0,
        ],
    )
    .expect("Matrix creation should succeed");

    // Fewer neighbors
    let mut lof_few = LocalOutlierFactor::new().with_n_neighbors(3);
    lof_few.fit(&data).expect("LOF fit should succeed");
    let scores_few = lof_few.score_samples(&data);

    // More neighbors
    let mut lof_many = LocalOutlierFactor::new().with_n_neighbors(5);
    lof_many.fit(&data).expect("LOF fit should succeed");
    let scores_many = lof_many.score_samples(&data);

    // Both should work and produce scores
    assert_eq!(scores_few.len(), 8);
    assert_eq!(scores_many.len(), 8);

    // Scores should be different (different neighborhood sizes)
    let diff_exists = scores_few
        .iter()
        .zip(scores_many.iter())
        .any(|(a, b)| (a - b).abs() > 0.01);
    assert!(diff_exists);
}

#[test]
fn test_lof_varying_density_clusters() {
    // Two clusters with different densities
    // Cluster 1: Dense (points close together)
    // Cluster 2: Sparse (points far apart)
    // Outlier: Between clusters
    let data = Matrix::from_vec(
        9,
        2,
        vec![
            // Dense cluster (4 points around 0,0)
            0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.0, 0.1, // Sparse cluster (3 points around 10,10)
            10.0, 10.0, 12.0, 12.0, 11.0, 9.0, // Outlier between clusters
            5.0, 5.0, // Another outlier
            5.5, 5.5,
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_n_neighbors(3)
        .with_contamination(0.2);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);
    let predictions = lof.predict(&data);

    // LOF should detect outliers in varying density regions
    // Points 7 and 8 (between clusters) should have higher LOF scores
    assert!(scores[7] > 1.0 || scores[8] > 1.0);

    // Should detect some anomalies
    let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
    assert!(n_anomalies >= 1);
}

#[test]
fn test_lof_lof_score_interpretation() {
    let data = Matrix::from_vec(
        5,
        2,
        vec![
            2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, // Normal cluster
            10.0, 10.0, // Clear outlier
        ],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);

    // LOF ≈ 1: similar density to neighbors (normal)
    // LOF >> 1: lower density than neighbors (outlier)
    let normal_scores = &scores[0..4];
    let outlier_score = scores[4];

    // Normal points should have LOF close to 1
    for &score in normal_scores {
        assert!((0.5..2.0).contains(&score));
    }

    // Outlier should have LOF > 1 (significantly)
    assert!(outlier_score > 1.5);
}

#[test]
fn test_lof_all_normal() {
    // All points are normal (tightly clustered)
    let data = Matrix::from_vec(
        6,
        2,
        vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9, 2.1, 2.1, 1.8, 2.0],
    )
    .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new()
        .with_contamination(0.1)
        .with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    let scores = lof.score_samples(&data);

    // All LOF scores should be close to 1 (similar density)
    for &score in &scores {
        assert!((0.5..1.5).contains(&score));
    }

    // With 10% contamination, expect mostly normal points
    let n_normal = predictions.iter().filter(|&&p| p == 1).count();
    assert!(n_normal >= 5);
}

#[test]
fn test_lof_score_samples_finite() {
    let data = Matrix::from_vec(4, 2, vec![2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 10.0, 10.0])
        .expect("Matrix creation should succeed");

    let mut lof = LocalOutlierFactor::new().with_n_neighbors(2);
    lof.fit(&data).expect("LOF fit should succeed");

    let scores = lof.score_samples(&data);
    // All LOF scores should be finite
    for &score in &scores {
        assert!(score.is_finite());
        assert!(score > 0.0); // LOF is always positive
    }
}

#[test]
fn test_lof_multidimensional() {
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

    let mut lof = LocalOutlierFactor::new()
        .with_contamination(0.3)
        .with_n_neighbors(3);
    lof.fit(&data).expect("LOF fit should succeed");

    let predictions = lof.predict(&data);
    let scores = lof.score_samples(&data);

    assert_eq!(predictions.len(), 6);
    assert_eq!(scores.len(), 6);
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_lof_predict_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let lof = LocalOutlierFactor::new();
    let _ = lof.predict(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_lof_score_samples_before_fit() {
    let data =
        Matrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 2.0]).expect("Matrix creation should succeed");
    let lof = LocalOutlierFactor::new();
    let _ = lof.score_samples(&data); // Should panic
}

#[test]
#[should_panic(expected = "Model not fitted")]
fn test_lof_negative_outlier_factor_before_fit() {
    let lof = LocalOutlierFactor::new();
    let _ = lof.negative_outlier_factor(); // Should panic
}
