
#[test]
fn test_stratified_kfold_shuffle_no_random_state() {
    // Covers shuffle without random_state (line 381: thread_rng)
    let skfold = StratifiedKFold::new(2).with_shuffle(true);
    let y = Vector::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let splits = skfold.split(&y);
    assert_eq!(splits.len(), 2);

    // Stratification should still hold
    for (_, test_idx) in &splits {
        let mut c0 = 0;
        let mut c1 = 0;
        for &idx in test_idx {
            if y[idx] == 0.0 {
                c0 += 1;
            } else {
                c1 += 1;
            }
        }
        // Each fold should maintain approximate class ratio
        assert!(c0 > 0 && c1 > 0);
    }
}

#[test]
fn test_grid_search_result_debug_clone() {
    let result = GridSearchResult {
        best_alpha: 0.1,
        best_score: 0.95,
        alphas: vec![0.01, 0.1],
        scores: vec![0.9, 0.95],
    };
    let cloned = result.clone();
    assert_eq!(cloned.best_alpha, 0.1);
    assert_eq!(cloned.best_score, 0.95);

    let debug = format!("{result:?}");
    assert!(debug.contains("GridSearchResult"));
}

#[test]
fn test_grid_search_result_best_index_single_element() {
    let result = GridSearchResult {
        best_alpha: 0.5,
        best_score: 0.8,
        alphas: vec![0.5],
        scores: vec![0.8],
    };
    assert_eq!(result.best_index(), 0);
}

#[test]
fn test_grid_search_result_best_index_descending() {
    // Best score is at the first position
    let result = GridSearchResult {
        best_alpha: 0.01,
        best_score: 0.99,
        alphas: vec![0.01, 0.1, 1.0],
        scores: vec![0.99, 0.95, 0.85],
    };
    assert_eq!(result.best_index(), 0);
}

#[test]
fn test_update_best_if_improved_no_improvement() {
    // Covers the case where score <= best_score (no update)
    let mut best_score = 0.9_f32;
    let mut best_alpha = 0.1_f32;

    update_best_if_improved(0.8, 0.5, &mut best_score, &mut best_alpha);
    assert_eq!(best_score, 0.9);
    assert_eq!(best_alpha, 0.1);
}

#[test]
fn test_update_best_if_improved_with_improvement() {
    let mut best_score = 0.5_f32;
    let mut best_alpha = 0.1_f32;

    update_best_if_improved(0.9, 0.01, &mut best_score, &mut best_alpha);
    assert_eq!(best_score, 0.9);
    assert_eq!(best_alpha, 0.01);
}

#[test]
fn test_kfold_split_single_sample_per_fold() {
    // 3 samples, 3 folds: each fold has exactly 1 test sample
    let kfold = KFold::new(3);
    let splits = kfold.split(3);

    for (train_idx, test_idx) in &splits {
        assert_eq!(test_idx.len(), 1);
        assert_eq!(train_idx.len(), 2);
    }
}

#[test]
fn test_kfold_split_two_folds() {
    let kfold = KFold::new(2);
    let splits = kfold.split(10);

    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].1.len(), 5);
    assert_eq!(splits[1].1.len(), 5);
}

#[test]
fn test_kfold_with_shuffle_true() {
    // Covers with_shuffle(true) (line 183)
    let kfold = KFold::new(3).with_shuffle(true);
    let splits = kfold.split(9);
    assert_eq!(splits.len(), 3);
}

#[test]
fn test_train_test_split_large_test_size() {
    // Test with test_size close to 1.0
    let x = Matrix::from_vec(100, 1, (0..100).map(|i| i as f32).collect()).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0; 100]);

    let result = train_test_split(&x, &y, 0.9, Some(42));
    assert!(result.is_ok());
    let (x_train, x_test, _, _) = result.expect("should succeed");
    assert_eq!(x_train.shape().0, 10);
    assert_eq!(x_test.shape().0, 90);
}

#[test]
fn test_train_test_split_small_test_size() {
    // Test with test_size close to 0.0
    let x = Matrix::from_vec(100, 1, (0..100).map(|i| i as f32).collect()).expect("valid matrix");
    let y = Vector::from_vec(vec![0.0; 100]);

    let result = train_test_split(&x, &y, 0.05, Some(42));
    assert!(result.is_ok());
    let (x_train, x_test, _, _) = result.expect("should succeed");
    assert_eq!(x_train.shape().0, 95);
    assert_eq!(x_test.shape().0, 5);
}

#[test]
fn test_stratified_kfold_single_class() {
    // All samples are the same class
    let y = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let skfold = StratifiedKFold::new(3);
    let splits = skfold.split(&y);

    assert_eq!(splits.len(), 3);
    // Each fold should have 2 test samples
    for (_, test_idx) in &splits {
        assert_eq!(test_idx.len(), 2);
    }
}
