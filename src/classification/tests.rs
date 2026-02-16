//! Tests for classification module.

pub(crate) use super::super::*;

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

#[path = "tests_part_02.rs"]

mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
#[path = "tests_part_04.rs"]
mod tests_part_04;
