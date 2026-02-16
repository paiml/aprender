use super::*;

#[test]
fn test_stratified_kfold_binary_classification() {
    // Binary classification with 50-50 split
    let y = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    let skfold = StratifiedKFold::new(4);

    let splits = skfold.split(&y);

    for (_, test_idx) in splits {
        assert_eq!(test_idx.len(), 2, "Each fold should have 2 samples");

        // Count classes
        let mut class_0_count = 0;
        let mut class_1_count = 0;
        for &idx in &test_idx {
            if y[idx] == 0.0 {
                class_0_count += 1;
            } else {
                class_1_count += 1;
            }
        }

        // Should have exactly one sample from each class
        assert_eq!(class_0_count, 1);
        assert_eq!(class_1_count, 1);
    }
}

#[test]
fn test_stratified_kfold_many_classes() {
    // 5 classes, 2 samples each
    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    let skfold = StratifiedKFold::new(2);

    let splits = skfold.split(&y);

    for (_, test_idx) in splits {
        assert_eq!(test_idx.len(), 5, "Each fold should have 5 samples");

        // Each class should appear exactly once
        let mut class_counts = vec![0; 5];
        for &idx in &test_idx {
            let label = y[idx] as usize;
            class_counts[label] += 1;
        }

        for &count in &class_counts {
            assert_eq!(count, 1, "Each class should appear once per fold");
        }
    }
}

#[test]
fn test_stratified_kfold_no_overlap() {
    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    let skfold = StratifiedKFold::new(3);

    let splits = skfold.split(&y);

    for (train_idx, test_idx) in splits {
        // Train and test should not overlap
        for &test in &test_idx {
            assert!(
                !train_idx.contains(&test),
                "Train and test indices should not overlap"
            );
        }
    }
}

#[test]
fn test_stratified_kfold_builder_pattern() {
    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0]);

    let skfold = StratifiedKFold::new(2)
        .with_shuffle(true)
        .with_random_state(42);

    let splits = skfold.split(&y);
    assert_eq!(splits.len(), 2);
}

// ==================== Grid Search Tests ====================

#[test]
fn test_grid_search_alpha_ridge() {
    let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    let x = Matrix::from_vec(50, 1, x_data)
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(y_data);

    let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
    let kfold = KFold::new(5).with_random_state(42);

    let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None)
        .expect("Grid search for ridge should succeed");

    assert!(alphas.contains(&result.best_alpha));
    assert!(result.best_score > 0.9);
    assert_eq!(result.alphas.len(), alphas.len());
    assert_eq!(result.scores.len(), alphas.len());
}

#[test]
fn test_grid_search_alpha_lasso() {
    let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    let x = Matrix::from_vec(50, 1, x_data)
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(y_data);

    let alphas = vec![0.001, 0.01, 0.1];
    let kfold = KFold::new(5).with_random_state(42);

    let result = grid_search_alpha("lasso", &alphas, &x, &y, &kfold, None)
        .expect("Grid search for lasso should succeed");

    assert!(alphas.contains(&result.best_alpha));
    assert!(result.best_score > 0.9);
    assert_eq!(result.alphas.len(), alphas.len());
    assert_eq!(result.scores.len(), alphas.len());
}

#[test]
fn test_grid_search_alpha_elastic_net() {
    let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    let x = Matrix::from_vec(50, 1, x_data)
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(y_data);

    let alphas = vec![0.001, 0.01, 0.1];
    let kfold = KFold::new(5).with_random_state(42);

    let result = grid_search_alpha("elastic_net", &alphas, &x, &y, &kfold, Some(0.5))
        .expect("Grid search for elastic_net should succeed");

    assert!(alphas.contains(&result.best_alpha));
    assert!(result.best_score > 0.9);
    assert_eq!(result.alphas.len(), alphas.len());
    assert_eq!(result.scores.len(), alphas.len());
}

#[test]
fn test_grid_search_result_best_index() {
    let result = GridSearchResult {
        best_alpha: 0.1,
        best_score: 0.95,
        alphas: vec![0.01, 0.1, 1.0],
        scores: vec![0.90, 0.95, 0.85],
    };

    assert_eq!(result.best_index(), 1);
}

#[test]
fn test_grid_search_empty_alphas() {
    let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect())
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(vec![0.0; 10]);

    let alphas: Vec<f32> = vec![];
    let kfold = KFold::new(3);

    let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None);
    assert!(result.is_err());
    assert!(result
        .expect_err("Should fail with empty alphas")
        .contains("empty"));
}

#[test]
fn test_grid_search_invalid_model_type() {
    let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect())
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(vec![0.0; 10]);

    let alphas = vec![0.1, 1.0];
    let kfold = KFold::new(3);

    let result = grid_search_alpha("invalid_model", &alphas, &x, &y, &kfold, None);
    assert!(result.is_err());
    assert!(result
        .expect_err("Should fail with invalid model type")
        .contains("Unknown model type"));
}

#[test]
fn test_grid_search_elastic_net_missing_l1_ratio() {
    let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect())
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(vec![0.0; 10]);

    let alphas = vec![0.1, 1.0];
    let kfold = KFold::new(3);

    let result = grid_search_alpha("elastic_net", &alphas, &x, &y, &kfold, None);
    assert!(result.is_err());
    assert!(result
        .expect_err("Should fail with missing l1_ratio")
        .contains("l1_ratio required"));
}

#[test]
fn test_grid_search_finds_optimal_alpha() {
    let x_data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 3.0 * x + 2.0).collect();

    let x = Matrix::from_vec(30, 1, x_data)
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(y_data);

    // Try range of alphas - should prefer smaller for this simple problem
    let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    let kfold = KFold::new(5).with_random_state(42);

    let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None)
        .expect("Grid search should find optimal alpha");

    // Best alpha should be one of the smaller values (less regularization needed)
    assert!(result.best_alpha <= 1.0, "Best alpha should be <= 1.0");

    // All alphas should be evaluated
    assert_eq!(result.scores.len(), alphas.len());

    // Scores should generally decrease with higher alpha (more regularization hurts)
    let first_score = result.scores[0];
    let last_score = result.scores[alphas.len() - 1];
    assert!(first_score > last_score);
}

#[test]
fn test_grid_search_single_alpha() {
    let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect())
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec((0..10).map(|i| i as f32).collect());

    let alphas = vec![0.1];
    let kfold = KFold::new(3);

    let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None)
        .expect("Grid search with single alpha should succeed");

    assert_eq!(result.best_alpha, 0.1);
    assert_eq!(result.alphas.len(), 1);
    assert_eq!(result.scores.len(), 1);
}

// ==================== Coverage gap tests (targeting 24 missed lines) ====================

#[test]
fn test_cross_validation_result_empty_scores_mean() {
    // Covers mean() with empty scores (line 22-23)
    let result = CrossValidationResult { scores: vec![] };
    assert_eq!(result.mean(), 0.0);
}

#[test]
fn test_cross_validation_result_empty_scores_std() {
    // Covers std() with empty scores (line 31-33)
    let result = CrossValidationResult { scores: vec![] };
    assert_eq!(result.std(), 0.0);
}

#[test]
fn test_cross_validation_result_empty_scores_min() {
    // Covers min() with empty scores - returns f32::INFINITY
    let result = CrossValidationResult { scores: vec![] };
    assert_eq!(result.min(), f32::INFINITY);
}

#[test]
fn test_cross_validation_result_empty_scores_max() {
    // Covers max() with empty scores - returns f32::NEG_INFINITY
    let result = CrossValidationResult { scores: vec![] };
    assert_eq!(result.max(), f32::NEG_INFINITY);
}

#[test]
fn test_cross_validation_result_single_score() {
    let result = CrossValidationResult { scores: vec![0.85] };
    assert!((result.mean() - 0.85).abs() < 1e-6);
    assert_eq!(result.std(), 0.0);
    assert_eq!(result.min(), 0.85);
    assert_eq!(result.max(), 0.85);
}

#[test]
fn test_cross_validation_result_debug_clone() {
    let result = CrossValidationResult {
        scores: vec![0.9, 0.95],
    };
    let cloned = result.clone();
    assert_eq!(cloned.scores, result.scores);

    let debug = format!("{result:?}");
    assert!(debug.contains("CrossValidationResult"));
    assert!(debug.contains("0.9"));
}

#[test]
fn test_train_test_split_invalid_test_size_zero() {
    // Covers test_size <= 0.0 error (line 618)
    let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect()).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0; 10]);

    let result = train_test_split(&x, &y, 0.0, Some(42));
    assert!(result.is_err());
    assert!(result
        .expect_err("zero test_size should fail")
        .contains("test_size must be between 0 and 1"));
}

#[test]
fn test_train_test_split_invalid_test_size_negative() {
    let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect()).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0; 10]);

    let result = train_test_split(&x, &y, -0.5, Some(42));
    assert!(result.is_err());
}

#[test]
fn test_train_test_split_invalid_test_size_one() {
    // Covers test_size >= 1.0 error (line 618)
    let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect()).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0; 10]);

    let result = train_test_split(&x, &y, 1.0, Some(42));
    assert!(result.is_err());
    assert!(result
        .expect_err("test_size=1.0 should fail")
        .contains("test_size must be between 0 and 1"));
}

#[test]
fn test_train_test_split_invalid_test_size_above_one() {
    let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect()).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0; 10]);

    let result = train_test_split(&x, &y, 1.5, Some(42));
    assert!(result.is_err());
}

#[test]
fn test_train_test_split_mismatched_dimensions() {
    // Covers n_samples != y.len() error (lines 625-631)
    let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect()).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0; 5]); // Mismatch: 10 samples vs 5 labels

    let result = train_test_split(&x, &y, 0.2, Some(42));
    assert!(result.is_err());
    assert!(result
        .expect_err("mismatched dims should fail")
        .contains("same number of samples"));
}

#[test]
fn test_train_test_split_empty_result_set() {
    // Covers n_test == 0 || n_train == 0 error (lines 636-639)
    // With 2 samples and test_size=0.01, n_test rounds to 0
    let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0, 1.0]);

    let result = train_test_split(&x, &y, 0.01, Some(42));
    assert!(result.is_err());
    assert!(result
        .expect_err("empty split should fail")
        .contains("empty train or test set"));
}

#[test]
fn test_train_test_split_without_random_state() {
    // Covers shuffle_indices without seed (line 655-657: thread_rng branch)
    let x = Matrix::from_vec(20, 2, (0..40).map(|i| i as f32).collect()).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0; 20]);

    let result = train_test_split(&x, &y, 0.3, None);
    assert!(result.is_ok());
    let (x_train, x_test, y_train, y_test) = result.expect("split should succeed");
    assert_eq!(x_train.shape().0, 14);
    assert_eq!(x_test.shape().0, 6);
    assert_eq!(y_train.len(), 14);
    assert_eq!(y_test.len(), 6);
}

#[test]
fn test_kfold_with_shuffle_no_random_state() {
    // Covers KFold shuffle without random_state (line 213: thread_rng)
    let kfold = KFold::new(3).with_shuffle(true);
    let splits = kfold.split(9);

    assert_eq!(splits.len(), 3);
    // All indices should appear
    let mut all_test: Vec<usize> = splits.iter().flat_map(|(_, t)| t).copied().collect();
    all_test.sort_unstable();
    assert_eq!(all_test, (0..9).collect::<Vec<_>>());
}

#[test]
fn test_kfold_debug_clone() {
    let kfold = KFold::new(5).with_random_state(42);
    let cloned = kfold.clone();
    let splits_orig = kfold.split(10);
    let splits_clone = cloned.split(10);
    assert_eq!(splits_orig, splits_clone);

    let debug = format!("{kfold:?}");
    assert!(debug.contains("KFold"));
}

#[test]
fn test_stratified_kfold_debug_clone() {
    let skfold = StratifiedKFold::new(3).with_random_state(42);
    let cloned = skfold.clone();

    let debug = format!("{skfold:?}");
    assert!(debug.contains("StratifiedKFold"));

    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    let splits_orig = skfold.split(&y);
    let splits_clone = cloned.split(&y);
    assert_eq!(splits_orig.len(), splits_clone.len());
}

#[test]
fn test_stratified_kfold_with_shuffle_false() {
    // Covers with_shuffle(false) explicitly (line 314)
    let skfold = StratifiedKFold::new(2).with_shuffle(false);
    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0]);
    let splits = skfold.split(&y);
    assert_eq!(splits.len(), 2);
}
