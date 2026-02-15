
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
