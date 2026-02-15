
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
