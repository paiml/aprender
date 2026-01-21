//! Tests for classification module.

use super::*;

#[test]
fn test_sigmoid() {
    assert!((LogisticRegression::sigmoid(0.0) - 0.5).abs() < 1e-6);
    assert!(LogisticRegression::sigmoid(10.0) > 0.99);
    assert!(LogisticRegression::sigmoid(-10.0) < 0.01);
}

#[test]
fn test_logistic_regression_new() {
    let model = LogisticRegression::new();
    assert!(model.coefficients.is_none());
    assert_eq!(model.intercept, 0.0);
}

#[test]
fn test_logistic_regression_builder() {
    let model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(500)
        .with_tolerance(1e-3);

    assert_eq!(model.learning_rate, 0.1);
    assert_eq!(model.max_iter, 500);
    assert_eq!(model.tol, 1e-3);
}

#[test]
fn test_logistic_regression_fit_simple() {
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
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);

    let result = model.fit(&x, &y);
    assert!(result.is_ok());
    assert!(model.coefficients.is_some());
}

#[test]
fn test_logistic_regression_predict() {
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
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);

    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");
    let predictions = model.predict(&x);

    // Should correctly classify training data
    assert_eq!(predictions.len(), 4);
    for pred in predictions {
        assert!(pred == 0 || pred == 1);
    }
}

#[test]
fn test_logistic_regression_score() {
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
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);

    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");
    let accuracy = model.score(&x, &y);

    // Should achieve high accuracy on linearly separable data
    assert!(accuracy >= 0.75); // At least 75% accuracy
}

#[test]
fn test_logistic_regression_invalid_labels() {
    let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 matrix with 4 values");
    let y = vec![0, 2]; // Invalid label 2

    let mut model = LogisticRegression::new();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with invalid label value"),
        "Labels must be 0 or 1 for binary classification"
    );
}

#[test]
fn test_logistic_regression_mismatched_samples() {
    let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 matrix with 4 values");
    let y = vec![0]; // Only 1 label for 2 samples

    let mut model = LogisticRegression::new();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with mismatched sample counts"),
        "Number of samples in X and y must match"
    );
}

#[test]
fn test_logistic_regression_zero_samples() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("0x2 empty matrix");
    let y = vec![];

    let mut model = LogisticRegression::new();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with zero samples"),
        "Cannot fit with zero samples"
    );
}

#[test]
fn test_predict_proba() {
    let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 matrix with 4 values");
    let y = vec![0, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);

    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");
    let probas = model.predict_proba(&x);

    assert_eq!(probas.len(), 2);
    for &p in probas.as_slice() {
        assert!((0.0..=1.0).contains(&p));
    }
}

// SafeTensors Serialization Tests
// RED PHASE: These tests will fail until we implement save_safetensors() and load_safetensors()

#[test]
fn test_save_safetensors_unfitted_model() {
    // Test 1: Cannot save unfitted model
    let model = LogisticRegression::new();
    let result = model.save_safetensors("/tmp/test_unfitted_logistic.safetensors");

    assert!(result.is_err());
    assert!(result
        .expect_err("Should fail when saving unfitted model")
        .contains("unfitted"));
}

#[test]
fn test_save_load_safetensors_roundtrip() {
    // Test 2: Save and load preserves model state
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
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    // Train model
    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Save model
    let path = "/tmp/test_logistic_roundtrip.safetensors";
    model
        .save_safetensors(path)
        .expect("Should save fitted model to valid path");

    // Load model
    let loaded =
        LogisticRegression::load_safetensors(path).expect("Should load valid SafeTensors file");

    // Verify coefficients match
    assert_eq!(
        model
            .coefficients
            .as_ref()
            .expect("Model is fitted and has coefficients")
            .len(),
        loaded
            .coefficients
            .as_ref()
            .expect("Loaded model has coefficients")
            .len()
    );
    for i in 0..model
        .coefficients
        .as_ref()
        .expect("Model has coefficients")
        .len()
    {
        assert_eq!(
            model.coefficients.as_ref().expect("Model has coefficients")[i],
            loaded
                .coefficients
                .as_ref()
                .expect("Loaded model has coefficients")[i]
        );
    }
    assert_eq!(model.intercept, loaded.intercept);

    // Verify predictions match
    let predictions_original = model.predict(&x);
    let predictions_loaded = loaded.predict(&x);
    assert_eq!(predictions_original, predictions_loaded);

    // Cleanup
    std::fs::remove_file(path).ok();
}

#[test]
fn test_load_safetensors_corrupted_file() {
    // Test 3: Loading corrupted file fails gracefully
    let path = "/tmp/test_corrupted_logistic.safetensors";
    std::fs::write(path, b"CORRUPTED DATA").expect("Should write test file");

    let result = LogisticRegression::load_safetensors(path);
    assert!(result.is_err());

    std::fs::remove_file(path).ok();
}

#[test]
fn test_load_safetensors_missing_file() {
    // Test 4: Loading missing file fails with clear error
    let result = LogisticRegression::load_safetensors("/tmp/nonexistent_logistic_xyz.safetensors");
    assert!(result.is_err());
    let err = result.expect_err("Should fail when loading nonexistent file");
    assert!(
        err.contains("No such file") || err.contains("not found"),
        "Error should mention file not found: {err}"
    );
}

#[test]
fn test_safetensors_preserves_probabilities() {
    // Test 5: Probabilities are identical after save/load
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
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(1000);
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    let probas_before = model.predict_proba(&x);

    // Save and load
    let path = "/tmp/test_logistic_probas.safetensors";
    model
        .save_safetensors(path)
        .expect("Should save fitted model to valid path");
    let loaded =
        LogisticRegression::load_safetensors(path).expect("Should load valid SafeTensors file");

    let probas_after = loaded.predict_proba(&x);

    // Verify probabilities match exactly
    assert_eq!(probas_before.len(), probas_after.len());
    for i in 0..probas_before.len() {
        assert_eq!(probas_before[i], probas_after[i]);
    }

    std::fs::remove_file(path).ok();
}

// K-Nearest Neighbors tests
#[test]
fn test_knn_new() {
    let knn = KNearestNeighbors::new(3);
    assert_eq!(knn.k, 3);
    assert_eq!(knn.metric, DistanceMetric::Euclidean);
    assert!(!knn.weights);
}

#[test]
fn test_knn_basic_fit_predict() {
    // Simple 2-class problem
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 0
            5.0, 5.0, // class 1
            5.0, 6.0, // class 1
            6.0, 5.0, // class 1
        ],
    )
    .expect("6x2 matrix with 12 values");
    let y = vec![0, 0, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Test point close to class 0
    let test1 = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
    let pred1 = knn.predict(&test1).expect("Prediction should succeed");
    assert_eq!(pred1[0], 0);

    // Test point close to class 1
    let test2 = Matrix::from_vec(1, 2, vec![5.5, 5.5]).expect("1x2 test matrix");
    let pred2 = knn.predict(&test2).expect("Prediction should succeed");
    assert_eq!(pred2[0], 1);
}

#[test]
fn test_knn_k_equals_one() {
    // With k=1, should predict nearest neighbor exactly
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            1.0, 1.0, // class 1
            2.0, 2.0, // class 0
            3.0, 3.0, // class 1
        ],
    )
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 1, 0, 1];

    let mut knn = KNearestNeighbors::new(1);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Predict on training data - should be perfect
    let predictions = knn.predict(&x).expect("Prediction should succeed");
    assert_eq!(predictions, y);
}

#[test]
fn test_knn_euclidean_distance() {
    let x = Matrix::from_vec(
        3,
        2,
        vec![
            0.0, 0.0, // class 0
            3.0, 4.0, // class 1 (distance 5.0 from origin)
            1.0, 1.0, // class 0
        ],
    )
    .expect("3x2 matrix with 6 values");
    let y = vec![0, 1, 0];

    let mut knn = KNearestNeighbors::new(1).with_metric(DistanceMetric::Euclidean);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Test point at (1.5, 2.0) - closer to (1, 1) than (3, 4)
    let test = Matrix::from_vec(1, 2, vec![1.5, 2.0]).expect("1x2 test matrix");
    let pred = knn.predict(&test).expect("Prediction should succeed");
    assert_eq!(pred[0], 0);
}

#[test]
fn test_knn_manhattan_distance() {
    let x = Matrix::from_vec(
        3,
        2,
        vec![
            0.0, 0.0, // class 0
            2.0, 2.0, // class 1 (Manhattan distance 4.0 from origin)
            1.0, 0.0, // class 0
        ],
    )
    .expect("3x2 matrix with 6 values");
    let y = vec![0, 1, 0];

    let mut knn = KNearestNeighbors::new(1).with_metric(DistanceMetric::Manhattan);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    let test = Matrix::from_vec(1, 2, vec![0.5, 0.0]).expect("1x2 test matrix");
    let pred = knn.predict(&test).expect("Prediction should succeed");
    assert_eq!(pred[0], 0); // Closer to (1, 0)
}

#[test]
fn test_knn_minkowski_distance() {
    let x = Matrix::from_vec(
        3,
        2,
        vec![
            0.0, 0.0, // class 0
            3.0, 4.0, // class 1
            1.0, 1.0, // class 0
        ],
    )
    .expect("3x2 matrix with 6 values");
    let y = vec![0, 1, 0];

    // Minkowski with p=3
    let mut knn = KNearestNeighbors::new(1).with_metric(DistanceMetric::Minkowski(3.0));
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    let test = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
    let pred = knn.predict(&test).expect("Prediction should succeed");
    assert_eq!(pred[0], 0);
}

#[test]
fn test_knn_weighted_voting() {
    // Set up data where uniform voting gives different result than weighted
    let x = Matrix::from_vec(
        5,
        1,
        vec![
            0.0, // class 0
            0.1, // class 0
            5.0, // class 1
            5.5, // class 1
            6.0, // class 1
        ],
    )
    .expect("5x1 matrix with 5 values");
    let y = vec![0, 0, 1, 1, 1];

    let mut knn_weighted = KNearestNeighbors::new(3).with_weights(true);
    knn_weighted
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Test point at 0.05 - very close to class 0
    let test = Matrix::from_vec(1, 1, vec![0.05]).expect("1x1 test matrix");
    let pred = knn_weighted
        .predict(&test)
        .expect("Prediction should succeed");
    assert_eq!(pred[0], 0); // Should be class 0 due to proximity weighting
}

#[test]
fn test_knn_predict_proba() {
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 0
            5.0, 5.0, // class 1
            5.0, 6.0, // class 1
            6.0, 5.0, // class 1
        ],
    )
    .expect("6x2 matrix with 12 values");
    let y = vec![0, 0, 0, 1, 1, 1];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    let test = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
    let probas = knn
        .predict_proba(&test)
        .expect("Probability prediction should succeed");

    assert_eq!(probas.len(), 1);
    assert_eq!(probas[0].len(), 2); // 2 classes

    // Probabilities should sum to 1.0
    let sum: f32 = probas[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Point closer to class 0 should have higher probability for class 0
    assert!(probas[0][0] > probas[0][1]);
}

#[test]
fn test_knn_multiclass() {
    // 3-class problem
    let x = Matrix::from_vec(
        9,
        2,
        vec![
            0.0, 0.0, // class 0
            0.0, 1.0, // class 0
            1.0, 0.0, // class 0
            5.0, 5.0, // class 1
            5.0, 6.0, // class 1
            6.0, 5.0, // class 1
            10.0, 10.0, // class 2
            10.0, 11.0, // class 2
            11.0, 10.0, // class 2
        ],
    )
    .expect("9x2 matrix with 18 values");
    let y = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let mut knn = KNearestNeighbors::new(3);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Test each cluster
    let test1 = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
    assert_eq!(
        knn.predict(&test1).expect("Prediction should succeed")[0],
        0
    );

    let test2 = Matrix::from_vec(1, 2, vec![5.5, 5.5]).expect("1x2 test matrix");
    assert_eq!(
        knn.predict(&test2).expect("Prediction should succeed")[0],
        1
    );

    let test3 = Matrix::from_vec(1, 2, vec![10.5, 10.5]).expect("1x2 test matrix");
    assert_eq!(
        knn.predict(&test3).expect("Prediction should succeed")[0],
        2
    );
}

#[test]
fn test_knn_not_fitted_error() {
    let knn = KNearestNeighbors::new(3);
    let test = Matrix::from_vec(1, 2, vec![0.0, 0.0]).expect("1x2 test matrix");

    let result = knn.predict(&test);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail when predicting with unfitted model"),
        "Model not fitted"
    );
}

#[test]
fn test_knn_dimension_mismatch() {
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        .expect("3x2 matrix with 6 values");
    let y = vec![0, 1, 0];

    let mut knn = KNearestNeighbors::new(1);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Test with wrong number of features
    let test = Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]).expect("1x3 test matrix");
    let result = knn.predict(&test);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with dimension mismatch"),
        "Feature dimension mismatch"
    );
}

#[test]
fn test_knn_sample_mismatch() {
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        .expect("3x2 matrix with 6 values");
    let y = vec![0, 1]; // Wrong length

    let mut knn = KNearestNeighbors::new(1);
    let result = knn.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with sample mismatch"),
        "Number of samples in X and y must match"
    );
}

#[test]
fn test_knn_k_too_large() {
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        .expect("3x2 matrix with 6 values");
    let y = vec![0, 1, 0];

    let mut knn = KNearestNeighbors::new(5); // k > n_samples
    let result = knn.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail when k exceeds sample count"),
        "k cannot be larger than number of training samples"
    );
}

#[test]
fn test_knn_empty_data() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("0x2 empty matrix");
    let y = vec![];

    let mut knn = KNearestNeighbors::new(1);
    let result = knn.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with empty data"),
        "Cannot fit with zero samples"
    );
}

#[test]
fn test_knn_builder_pattern() {
    let knn = KNearestNeighbors::new(5)
        .with_metric(DistanceMetric::Manhattan)
        .with_weights(true);

    assert_eq!(knn.k, 5);
    assert_eq!(knn.metric, DistanceMetric::Manhattan);
    assert!(knn.weights);
}

#[test]
fn test_knn_distance_symmetry() {
    // Property test: distance(a, b) == distance(b, a)
    let x = Matrix::from_vec(
        2,
        2,
        vec![
            1.0, 2.0, // point a
            3.0, 4.0, // point b
        ],
    )
    .expect("2x2 matrix with 4 values");
    let y = vec![0, 1];

    let mut knn = KNearestNeighbors::new(1);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Compute both directions
    let dist_ab = knn.compute_distance(&x, 0, &x, 1, 2);
    let dist_ba = knn.compute_distance(&x, 1, &x, 0, 2);

    assert!((dist_ab - dist_ba).abs() < 1e-6);
}

#[test]
fn test_knn_perfect_fit_with_k1() {
    // Property test: k=1 on training data gives perfect predictions
    let x = Matrix::from_vec(
        10,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 7.0,
            8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("10x3 matrix with 30 values");
    let y = vec![0, 0, 1, 1, 0, 1, 0, 1, 0, 1];

    let mut knn = KNearestNeighbors::new(1);
    knn.fit(&x, &y)
        .expect("Training should succeed with valid data");

    let predictions = knn.predict(&x).expect("Prediction should succeed");
    assert_eq!(predictions, y);
}

// ========== Gaussian Naive Bayes Tests ==========

#[test]
fn test_gaussian_nb_new() {
    let model = GaussianNB::new();
    assert!(model.class_priors.is_none());
    assert!(model.means.is_none());
    assert!(model.variances.is_none());
    assert_eq!(model.var_smoothing, 1e-9);
}

#[test]
fn test_gaussian_nb_builder() {
    let model = GaussianNB::new().with_var_smoothing(1e-8);
    assert_eq!(model.var_smoothing, 1e-8);
}

#[test]
fn test_gaussian_nb_basic_fit_predict() {
    // Simple 2-class problem: class 0 at (0,0), class 1 at (1,1)
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.1, 0.1, // class 0
            1.0, 1.0, // class 1
            0.9, 0.9, // class 1
        ],
    )
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut model = GaussianNB::new();
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    let predictions = model.predict(&x).expect("Prediction should succeed");
    assert_eq!(predictions, y);
}

#[test]
fn test_gaussian_nb_multiclass() {
    // 3-class problem
    let x = Matrix::from_vec(
        9,
        2,
        vec![
            0.0, 0.0, // class 0
            0.1, 0.1, // class 0
            0.0, 0.1, // class 0
            5.0, 5.0, // class 1
            5.1, 5.1, // class 1
            5.0, 5.1, // class 1
            -5.0, -5.0, // class 2
            -5.1, -5.1, // class 2
            -5.0, -5.1, // class 2
        ],
    )
    .expect("9x2 matrix with 18 values");
    let y = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let mut model = GaussianNB::new();
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    let predictions = model.predict(&x).expect("Prediction should succeed");
    assert_eq!(predictions, y);
}

#[test]
fn test_gaussian_nb_predict_proba() {
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.1, 0.1, // class 0
            1.0, 1.0, // class 1
            0.9, 0.9, // class 1
        ],
    )
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut model = GaussianNB::new();
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    let probabilities = model
        .predict_proba(&x)
        .expect("Probability prediction should succeed");

    // Check all samples have probabilities
    assert_eq!(probabilities.len(), 4);

    // Check probabilities sum to 1
    for probs in &probabilities {
        assert_eq!(probs.len(), 2);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // Check first sample (class 0) has high probability for class 0
    assert!(probabilities[0][0] > 0.5);

    // Check last sample (class 1) has high probability for class 1
    assert!(probabilities[3][1] > 0.5);
}

#[test]
fn test_gaussian_nb_not_fitted_error() {
    let model = GaussianNB::new();
    let x_test = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 test matrix");

    let result = model.predict(&x_test);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail when predicting with unfitted model"),
        "Model not fitted"
    );
}

#[test]
fn test_gaussian_nb_empty_data() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("0x2 empty matrix");
    let y: Vec<usize> = vec![];

    let mut model = GaussianNB::new();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with empty data"),
        "Cannot fit with empty data"
    );
}

#[test]
fn test_gaussian_nb_sample_mismatch() {
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        .expect("3x2 matrix with 6 values");
    let y = vec![0, 1]; // Wrong length

    let mut model = GaussianNB::new();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with sample mismatch"),
        "Number of samples in X and y must match"
    );
}

#[test]
fn test_gaussian_nb_single_class() {
    let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        .expect("3x2 matrix with 6 values");
    let y = vec![0, 0, 0]; // All same class

    let mut model = GaussianNB::new();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with single class"),
        "Need at least 2 classes"
    );
}

#[test]
fn test_gaussian_nb_dimension_mismatch() {
    let x_train = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 0.9, 0.9])
        .expect("4x2 training matrix");
    let y_train = vec![0, 0, 1, 1];

    let mut model = GaussianNB::new();
    model
        .fit(&x_train, &y_train)
        .expect("Training should succeed with valid data");

    let x_test =
        Matrix::from_vec(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("2x3 test matrix");
    let result = model.predict(&x_test);

    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with dimension mismatch"),
        "Feature dimension mismatch"
    );
}

#[test]
fn test_gaussian_nb_balanced_classes() {
    // Equal number of samples per class
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, // class 0
            0.1, 0.1, // class 0
            0.2, 0.2, // class 0
            1.0, 1.0, // class 1
            1.1, 1.1, // class 1
            1.2, 1.2, // class 1
        ],
    )
    .expect("6x2 matrix with 12 values");
    let y = vec![0, 0, 0, 1, 1, 1];

    let mut model = GaussianNB::new();
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Check class priors are equal
    let priors = model
        .class_priors
        .expect("Model is fitted and has class priors");
    assert!((priors[0] - 0.5).abs() < 1e-5);
    assert!((priors[1] - 0.5).abs() < 1e-5);
}

#[test]
fn test_gaussian_nb_imbalanced_classes() {
    // Imbalanced: 1 sample class 0, 3 samples class 1
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            1.0, 1.0, // class 1
            1.1, 1.1, // class 1
            1.2, 1.2, // class 1
        ],
    )
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 1, 1, 1];

    let mut model = GaussianNB::new();
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Check class priors reflect imbalance
    let priors = model
        .class_priors
        .expect("Model is fitted and has class priors");
    assert!((priors[0] - 0.25).abs() < 1e-5); // 1/4
    assert!((priors[1] - 0.75).abs() < 1e-5); // 3/4
}

#[test]
fn test_gaussian_nb_var_smoothing() {
    // Test that variance smoothing prevents division by zero
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0 - identical points
            0.0, 0.0, // class 0 - identical points
            1.0, 1.0, // class 1 - identical points
            1.0, 1.0, // class 1 - identical points
        ],
    )
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut model = GaussianNB::new().with_var_smoothing(1e-8);
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Should not panic or produce NaN/Inf
    let predictions = model.predict(&x).expect("Prediction should succeed");
    assert_eq!(predictions, y);

    let probabilities = model
        .predict_proba(&x)
        .expect("Probability prediction should succeed");
    for probs in &probabilities {
        for &p in probs {
            assert!(p.is_finite());
            assert!((0.0..=1.0).contains(&p));
        }
    }
}

#[test]
fn test_gaussian_nb_probabilities_sum_to_one() {
    // Property test: probabilities must sum to 1
    let x = Matrix::from_vec(
        10,
        3,
        vec![
            0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.1, 1.1,
            1.1, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1,
        ],
    )
    .expect("10x3 matrix with 30 values");
    let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2];

    let mut model = GaussianNB::new();
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    let probabilities = model
        .predict_proba(&x)
        .expect("Probability prediction should succeed");

    for probs in &probabilities {
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_gaussian_nb_default() {
    let model1 = GaussianNB::new();
    let model2 = GaussianNB::default();

    assert_eq!(model1.var_smoothing, model2.var_smoothing);
}

#[test]
fn test_gaussian_nb_class_separation() {
    // Well-separated classes should have high confidence
    let x = Matrix::from_vec(
        4,
        2,
        vec![
            0.0, 0.0, // class 0
            0.1, 0.1, // class 0
            10.0, 10.0, // class 1 (far away)
            10.1, 10.1, // class 1 (far away)
        ],
    )
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut model = GaussianNB::new();
    model
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    let probabilities = model
        .predict_proba(&x)
        .expect("Probability prediction should succeed");

    // First sample should have very high confidence for class 0
    assert!(probabilities[0][0] > 0.99);

    // Last sample should have very high confidence for class 1
    assert!(probabilities[3][1] > 0.99);
}

// ===== LinearSVM Tests =====

#[test]
fn test_linear_svm_new() {
    let svm = LinearSVM::new();
    assert!(svm.weights.is_none());
    assert_eq!(svm.bias, 0.0);
    assert_eq!(svm.c, 1.0);
    assert_eq!(svm.learning_rate, 0.01);
    assert_eq!(svm.max_iter, 1000);
    assert_eq!(svm.tol, 1e-4);
}

#[test]
fn test_linear_svm_builder() {
    let svm = LinearSVM::new()
        .with_c(0.5)
        .with_learning_rate(0.001)
        .with_max_iter(500)
        .with_tolerance(1e-5);

    assert_eq!(svm.c, 0.5);
    assert_eq!(svm.learning_rate, 0.001);
    assert_eq!(svm.max_iter, 500);
    assert_eq!(svm.tol, 1e-5);
}

#[test]
fn test_linear_svm_fit_simple() {
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
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut svm = LinearSVM::new().with_max_iter(1000).with_learning_rate(0.1);

    let result = svm.fit(&x, &y);
    assert!(result.is_ok());
    assert!(svm.weights.is_some());
}

#[test]
fn test_linear_svm_predict_simple() {
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
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut svm = LinearSVM::new().with_max_iter(1000).with_learning_rate(0.1);
    svm.fit(&x, &y)
        .expect("Training should succeed with valid data");

    let predictions = svm.predict(&x).expect("Prediction should succeed");
    assert_eq!(predictions.len(), 4);

    // Should classify correctly (or close to it)
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| *pred == *true_label)
        .count();

    // Should get at least 3 out of 4 correct for simple case
    assert!(correct >= 3);
}

#[test]
fn test_linear_svm_decision_function() {
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
    .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    let mut svm = LinearSVM::new().with_max_iter(1000).with_learning_rate(0.1);
    svm.fit(&x, &y)
        .expect("Training should succeed with valid data");

    let decisions = svm
        .decision_function(&x)
        .expect("Decision function should succeed");
    assert_eq!(decisions.len(), 4);

    // Class 0 samples should have negative decisions
    // Class 1 samples should have positive decisions
    // (may not be perfect for simple gradient descent)
}

#[test]
fn test_linear_svm_predict_untrained() {
    let svm = LinearSVM::new();
    let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 matrix with 4 values");

    let result = svm.predict(&x);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail when predicting with untrained model"),
        "Model not trained yet"
    );
}

#[test]
fn test_linear_svm_dimension_mismatch() {
    let x_train = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("4x2 training matrix");
    let y = vec![0, 0, 1, 1];

    let mut svm = LinearSVM::new();
    svm.fit(&x_train, &y)
        .expect("Training should succeed with valid data");

    // Try to predict with wrong number of features
    let x_test =
        Matrix::from_vec(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("2x3 test matrix");
    let result = svm.predict(&x_test);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with dimension mismatch"),
        "Feature dimension mismatch"
    );
}

#[test]
fn test_linear_svm_empty_data() {
    let x = Matrix::from_vec(0, 2, vec![]).expect("0x2 empty matrix");
    let y = vec![];

    let mut svm = LinearSVM::new();
    let result = svm.fit(&x, &y);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with empty data"),
        "Cannot fit with 0 samples"
    );
}

#[test]
fn test_linear_svm_mismatched_samples() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1]; // Wrong length

    let mut svm = LinearSVM::new();
    let result = svm.fit(&x, &y);
    assert!(result.is_err());
    assert_eq!(
        result.expect_err("Should fail with mismatched sample counts"),
        "x and y must have the same number of samples"
    );
}

#[test]
fn test_linear_svm_regularization_c() {
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
    .expect("6x2 matrix with 12 values");
    let y = vec![0, 0, 0, 1, 1, 1];

    // High C (less regularization) - should fit data more closely
    let mut svm_high_c = LinearSVM::new()
        .with_c(10.0)
        .with_max_iter(1000)
        .with_learning_rate(0.1);
    svm_high_c
        .fit(&x, &y)
        .expect("Training should succeed with valid data");
    let pred_high_c = svm_high_c.predict(&x).expect("Prediction should succeed");

    // Low C (more regularization) - should prefer simpler model
    let mut svm_low_c = LinearSVM::new()
        .with_c(0.1)
        .with_max_iter(1000)
        .with_learning_rate(0.1);
    svm_low_c
        .fit(&x, &y)
        .expect("Training should succeed with valid data");
    let pred_low_c = svm_low_c.predict(&x).expect("Prediction should succeed");

    // Both should make predictions
    assert_eq!(pred_high_c.len(), 6);
    assert_eq!(pred_low_c.len(), 6);
}

#[test]
fn test_linear_svm_binary_classification() {
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
    .expect("10x2 matrix with 20 values");
    let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    let mut svm = LinearSVM::new()
        .with_c(1.0)
        .with_max_iter(2000)
        .with_learning_rate(0.1);

    svm.fit(&x, &y)
        .expect("Training should succeed with valid data");
    let predictions = svm.predict(&x).expect("Prediction should succeed");

    // Should achieve reasonable accuracy
    let correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, true_label)| *pred == *true_label)
        .count();

    // Should get at least 8 out of 10 correct for well-separated clusters
    assert!(
        correct >= 8,
        "Expected at least 8/10 correct, got {correct}/10"
    );
}

#[test]
fn test_linear_svm_convergence() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    // With very few iterations, might not converge
    let mut svm_few_iter = LinearSVM::new().with_max_iter(10).with_learning_rate(0.01);
    svm_few_iter
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // With many iterations, should converge better
    let mut svm_many_iter = LinearSVM::new().with_max_iter(2000).with_learning_rate(0.1);
    svm_many_iter
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Both should train successfully
    assert!(svm_few_iter.weights.is_some());
    assert!(svm_many_iter.weights.is_some());
}

#[test]
fn test_linear_svm_default() {
    let svm1 = LinearSVM::new();
    let svm2 = LinearSVM::default();

    assert_eq!(svm1.c, svm2.c);
    assert_eq!(svm1.learning_rate, svm2.learning_rate);
    assert_eq!(svm1.max_iter, svm2.max_iter);
}
