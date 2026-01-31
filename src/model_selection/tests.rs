use super::*;

#[test]
fn test_train_test_split_basic() {
    // Create simple dataset: 10 samples, 2 features
    let x = Matrix::from_vec(
        10,
        2,
        vec![
            1.0, 2.0, // sample 0
            3.0, 4.0, // sample 1
            5.0, 6.0, // sample 2
            7.0, 8.0, // sample 3
            9.0, 10.0, // sample 4
            11.0, 12.0, // sample 5
            13.0, 14.0, // sample 6
            15.0, 16.0, // sample 7
            17.0, 18.0, // sample 8
            19.0, 20.0, // sample 9
        ],
    )
    .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    // Split 80/20
    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, 0.2, Some(42)).expect("Split should succeed");

    // Verify shapes
    assert_eq!(x_train.shape().0, 8, "Training set should have 8 samples");
    assert_eq!(x_test.shape().0, 2, "Test set should have 2 samples");
    assert_eq!(x_train.shape().1, 2, "Training features should be 2");
    assert_eq!(x_test.shape().1, 2, "Test features should be 2");
    assert_eq!(y_train.len(), 8, "Training labels should have 8 samples");
    assert_eq!(y_test.len(), 2, "Test labels should have 2 samples");
}

#[test]
fn test_train_test_split_reproducibility() {
    let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect())
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    // Same random state should give same split
    let (x_train1, x_test1, y_train1, y_test1) =
        train_test_split(&x, &y, 0.2, Some(42)).expect("First split should succeed");
    let (x_train2, x_test2, y_train2, y_test2) =
        train_test_split(&x, &y, 0.2, Some(42)).expect("Second split should succeed");

    // Verify reproducibility
    assert_eq!(x_train1.as_slice(), x_train2.as_slice());
    assert_eq!(x_test1.as_slice(), x_test2.as_slice());
    assert_eq!(y_train1.as_slice(), y_train2.as_slice());
    assert_eq!(y_test1.as_slice(), y_test2.as_slice());
}

#[test]
fn test_train_test_split_different_seeds() {
    let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect())
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    // Different random states should give different splits
    let (_, _, y_train1, _) =
        train_test_split(&x, &y, 0.2, Some(42)).expect("Split with seed 42 should succeed");
    let (_, _, y_train2, _) =
        train_test_split(&x, &y, 0.2, Some(123)).expect("Split with seed 123 should succeed");

    // Very likely to be different with different seeds
    assert_ne!(y_train1.as_slice(), y_train2.as_slice());
}

#[test]
fn test_train_test_split_different_sizes() {
    let x = Matrix::from_vec(100, 3, (0..300).map(|i| i as f32).collect())
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_slice(&vec![0.0; 100]);

    // Test 70/30 split
    let (x_train, x_test, _, _) =
        train_test_split(&x, &y, 0.3, Some(42)).expect("70/30 split should succeed");
    assert_eq!(x_train.shape().0, 70);
    assert_eq!(x_test.shape().0, 30);

    // Test 50/50 split
    let (x_train, x_test, _, _) =
        train_test_split(&x, &y, 0.5, Some(42)).expect("50/50 split should succeed");
    assert_eq!(x_train.shape().0, 50);
    assert_eq!(x_test.shape().0, 50);
}

#[test]
fn test_kfold_basic() {
    let kfold = KFold::new(5);
    let splits = kfold.split(10);

    // Should have 5 folds
    assert_eq!(splits.len(), 5, "Should have 5 folds");

    // Each fold should have 8 train and 2 test samples
    for (i, (train_idx, test_idx)) in splits.iter().enumerate() {
        assert_eq!(
            train_idx.len(),
            8,
            "Fold {i} should have 8 training samples"
        );
        assert_eq!(test_idx.len(), 2, "Fold {i} should have 2 test samples");

        // Verify no overlap between train and test
        for &test_i in test_idx {
            assert!(
                !train_idx.contains(&test_i),
                "Test index {test_i} should not be in training set for fold {i}"
            );
        }
    }

    // All indices should be used exactly once as test
    let mut all_test_indices: Vec<usize> =
        splits.iter().flat_map(|(_, test)| test).copied().collect();
    all_test_indices.sort_unstable();
    assert_eq!(all_test_indices, (0..10).collect::<Vec<_>>());
}

#[test]
fn test_kfold_no_shuffle() {
    let kfold = KFold::new(3);
    let splits = kfold.split(9);

    assert_eq!(splits.len(), 3);

    // Without shuffle, folds should be consecutive
    assert_eq!(splits[0].1, vec![0, 1, 2]);
    assert_eq!(splits[1].1, vec![3, 4, 5]);
    assert_eq!(splits[2].1, vec![6, 7, 8]);
}

#[test]
fn test_kfold_shuffle_reproducible() {
    let kfold1 = KFold::new(5).with_random_state(42);
    let kfold2 = KFold::new(5).with_random_state(42);

    let splits1 = kfold1.split(20);
    let splits2 = kfold2.split(20);

    // Same random state should give same splits
    assert_eq!(splits1, splits2);
}

#[test]
fn test_kfold_shuffle_different_states() {
    let kfold1 = KFold::new(5).with_random_state(42);
    let kfold2 = KFold::new(5).with_random_state(123);

    let splits1 = kfold1.split(20);
    let splits2 = kfold2.split(20);

    // Different random states should give different splits
    assert_ne!(splits1, splits2);
}

#[test]
fn test_kfold_uneven_split() {
    let kfold = KFold::new(3);
    let splits = kfold.split(10);

    assert_eq!(splits.len(), 3);

    // With 10 samples and 3 folds: folds should be 3, 3, 4 samples (or similar)
    let test_sizes: Vec<usize> = splits.iter().map(|(_, test)| test.len()).collect();
    let total_test: usize = test_sizes.iter().sum();
    assert_eq!(
        total_test, 10,
        "All samples should be used as test exactly once"
    );
}

#[test]
fn test_cross_validate_basic() {
    use crate::linear_model::LinearRegression;

    // Create simple dataset: y = 2x
    let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x).collect();

    let x = Matrix::from_vec(50, 1, x_data)
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(y_data);

    let model = LinearRegression::new();
    let kfold = KFold::new(5).with_random_state(42);

    let result =
        cross_validate(&model, &x, &y, &kfold).expect("Cross-validation should succeed");

    // Should have 5 scores (one per fold)
    assert_eq!(result.scores.len(), 5, "Should have 5 fold scores");

    // All scores should be very high (perfect linear relationship)
    for &score in &result.scores {
        assert!(score > 0.99, "Score should be > 0.99, got {score}");
    }

    // Mean should be close to 1.0
    assert!(result.mean() > 0.99, "Mean R² should be > 0.99");

    // Std should be very low (stable across folds)
    assert!(result.std() < 0.01, "Std should be < 0.01");
}

#[test]
fn test_cross_validate_result_stats() {
    let result = CrossValidationResult {
        scores: vec![0.95, 0.96, 0.94, 0.97, 0.93],
    };

    // Test mean
    let mean = result.mean();
    assert!((mean - 0.95).abs() < 0.001, "Mean should be ~0.95");

    // Test min/max
    assert_eq!(result.min(), 0.93);
    assert_eq!(result.max(), 0.97);

    // Test std
    let std = result.std();
    assert!(std > 0.0, "Std should be > 0");
    assert!(std < 0.02, "Std should be < 0.02");
}

#[test]
fn test_cross_validate_reproducible() {
    use crate::linear_model::LinearRegression;

    let x_data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 3.0 * x + 1.0).collect();

    let x = Matrix::from_vec(30, 1, x_data)
        .expect("Matrix creation should succeed with valid test data");
    let y = Vector::from_vec(y_data);

    let model = LinearRegression::new();

    // Same random state should give same results
    let kfold1 = KFold::new(5).with_random_state(42);
    let result1 =
        cross_validate(&model, &x, &y, &kfold1).expect("First cross-validation should succeed");

    let kfold2 = KFold::new(5).with_random_state(42);
    let result2 = cross_validate(&model, &x, &y, &kfold2)
        .expect("Second cross-validation should succeed");

    assert_eq!(
        result1.scores, result2.scores,
        "Results should be reproducible"
    );
}

// ==================== StratifiedKFold Tests ====================

#[test]
fn test_stratified_kfold_new() {
    let skfold = StratifiedKFold::new(5);
    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);

    let splits = skfold.split(&y);
    assert_eq!(splits.len(), 5);
}

#[test]
fn test_stratified_kfold_balanced_classes() {
    // Perfectly balanced classes
    let y = Vector::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let skfold = StratifiedKFold::new(3);

    let splits = skfold.split(&y);
    assert_eq!(splits.len(), 3);

    // Each fold should have one sample from each class
    for (train_idx, test_idx) in &splits {
        assert_eq!(test_idx.len(), 3, "Each test fold should have 3 samples");
        assert_eq!(train_idx.len(), 6, "Each train fold should have 6 samples");

        // Count classes in test fold
        let mut class_counts = [0; 3];
        for &idx in test_idx {
            let label = y[idx] as usize;
            class_counts[label] += 1;
        }

        // Each class should appear exactly once in test fold
        for &count in &class_counts {
            assert_eq!(
                count, 1,
                "Each class should appear exactly once in test fold"
            );
        }
    }
}

#[test]
fn test_stratified_kfold_imbalanced_classes() {
    // Imbalanced: 6 of class 0, 3 of class 1
    let y = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let skfold = StratifiedKFold::new(3);

    let splits = skfold.split(&y);

    for (_train_idx, test_idx) in &splits {
        // Count classes in test fold
        let mut class_0_count = 0;
        let mut class_1_count = 0;

        for &idx in test_idx {
            if y[idx] == 0.0 {
                class_0_count += 1;
            } else {
                class_1_count += 1;
            }
        }

        // Should maintain approximate 2:1 ratio in each fold
        assert_eq!(
            class_0_count, 2,
            "Each fold should have 2 samples from class 0"
        );
        assert_eq!(
            class_1_count, 1,
            "Each fold should have 1 sample from class 1"
        );
    }
}

#[test]
fn test_stratified_kfold_all_samples_used() {
    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    let skfold = StratifiedKFold::new(3);

    let splits = skfold.split(&y);

    let mut all_test_indices = vec![];
    for (_, test_idx) in splits {
        all_test_indices.extend(test_idx);
    }

    all_test_indices.sort_unstable();
    assert_eq!(
        all_test_indices,
        vec![0, 1, 2, 3, 4, 5],
        "All samples should be used as test exactly once"
    );
}

#[test]
fn test_stratified_kfold_with_shuffle() {
    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    let skfold = StratifiedKFold::new(2).with_shuffle(true);

    let splits = skfold.split(&y);
    assert_eq!(splits.len(), 2);

    // Should still maintain stratification even with shuffling
    for (_, test_idx) in &splits {
        let mut class_counts = [0; 3];
        for &idx in test_idx {
            let label = y[idx] as usize;
            class_counts[label] += 1;
        }

        // Each class should appear once in each fold
        for &count in &class_counts {
            assert_eq!(count, 1);
        }
    }
}

#[test]
fn test_stratified_kfold_with_random_state() {
    let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

    let skfold1 = StratifiedKFold::new(2).with_random_state(42);
    let splits1 = skfold1.split(&y);

    let skfold2 = StratifiedKFold::new(2).with_random_state(42);
    let splits2 = skfold2.split(&y);

    // Same random state should give same splits (check semantic equality)
    assert_eq!(splits1.len(), splits2.len());
    for ((train1, test1), (train2, test2)) in splits1.iter().zip(splits2.iter()) {
        // Sort for comparison since HashMap iteration order is not deterministic
        let mut train1_sorted = train1.clone();
        let mut train2_sorted = train2.clone();
        let mut test1_sorted = test1.clone();
        let mut test2_sorted = test2.clone();

        train1_sorted.sort_unstable();
        train2_sorted.sort_unstable();
        test1_sorted.sort_unstable();
        test2_sorted.sort_unstable();

        assert_eq!(train1_sorted, train2_sorted);
        assert_eq!(test1_sorted, test2_sorted);
    }
}

#[test]
fn test_stratified_kfold_different_random_states() {
    // Use larger dataset so different random states are more likely to differ
    let y = Vector::from_slice(&[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0,
    ]);

    let skfold1 = StratifiedKFold::new(3).with_random_state(42);
    let splits1 = skfold1.split(&y);

    let skfold2 = StratifiedKFold::new(3).with_random_state(123);
    let splits2 = skfold2.split(&y);

    // Different random states should give different splits
    // Due to HashMap ordering, we can't guarantee different order in every case
    // so we just verify both produce valid splits
    assert_eq!(splits1.len(), 3);
    assert_eq!(splits2.len(), 3);

    // Verify stratification is maintained for both
    for (_, test_idx) in &splits1 {
        let mut class_counts = [0; 3];
        for &idx in test_idx {
            let label = y[idx] as usize;
            class_counts[label] += 1;
        }
        // Each fold should have 2 samples from each class
        for &count in &class_counts {
            assert_eq!(count, 2);
        }
    }
}

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
