
    // ====================================================================
    // Helper: build a small linearly-separable classification dataset
    // ====================================================================
    fn classification_data() -> (Matrix<f32>, Vec<usize>) {
        // 8 samples, 2 features, 2 classes
        // Class 0: low feature values; Class 1: high feature values
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.1, // 0
                0.1, 0.0, // 0
                0.2, 0.2, // 0
                0.3, 0.1, // 0
                0.8, 0.9, // 1
                0.9, 0.8, // 1
                1.0, 1.0, // 1
                0.7, 0.9, // 1
            ],
        )
        .expect("classification data matrix");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    // ====================================================================
    // Helper: build a small regression dataset  y ≈ 2*x1 + 3*x2
    // ====================================================================
    fn regression_data() -> (Matrix<f32>, Vector<f32>) {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                1.0, 0.0, // 2
                0.0, 1.0, // 3
                1.0, 1.0, // 5
                2.0, 0.0, // 4
                0.0, 2.0, // 6
                2.0, 1.0, // 7
                1.0, 2.0, // 8
                3.0, 1.0, // 9
            ],
        )
        .expect("regression data matrix");
        let y = Vector::from_slice(&[2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0, 9.0]);
        (x, y)
    }

    // ====================================================================
    // RandomForestClassifier — construction
    // ====================================================================

    #[test]
    fn test_classifier_new_sets_n_estimators() {
        let rf = RandomForestClassifier::new(5);
        assert_eq!(rf.n_estimators, 5);
        assert!(rf.trees.is_empty());
        assert!(rf.max_depth.is_none());
        assert!(rf.random_state.is_none());
    }

    #[test]
    fn test_classifier_with_max_depth() {
        let rf = RandomForestClassifier::new(3).with_max_depth(4);
        assert_eq!(rf.max_depth, Some(4));
    }

    #[test]
    fn test_classifier_with_random_state() {
        let rf = RandomForestClassifier::new(3).with_random_state(42);
        assert_eq!(rf.random_state, Some(42));
    }

    #[test]
    fn test_classifier_builder_chaining() {
        let rf = RandomForestClassifier::new(10)
            .with_max_depth(3)
            .with_random_state(99);
        assert_eq!(rf.n_estimators, 10);
        assert_eq!(rf.max_depth, Some(3));
        assert_eq!(rf.random_state, Some(99));
    }

    // ====================================================================
    // RandomForestClassifier — fit / predict
    // ====================================================================

    #[test]
    fn test_classifier_fit_creates_correct_number_of_trees() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 5);
    }

    #[test]
    fn test_classifier_predict_returns_correct_length() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_classifier_predict_reasonable_accuracy() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);

        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        // Should get at least 6 out of 8 correct on linearly separable data
        assert!(
            correct >= 6,
            "Expected >= 6 correct, got {correct} out of 8"
        );
    }

    #[test]
    fn test_classifier_score_returns_valid_range() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let score = rf.score(&x, &y);
        assert!(
            (0.0..=1.0).contains(&score),
            "Score {score} should be in [0.0, 1.0]"
        );
    }

    #[test]
    fn test_classifier_reproducibility_with_random_state() {
        let (x, y) = classification_data();
        let mut rf1 = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let preds1 = rf1.predict(&x);

        let mut rf2 = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let preds2 = rf2.predict(&x);

        assert_eq!(
            preds1, preds2,
            "Same random_state should yield same predictions"
        );
    }

    // ====================================================================
    // RandomForestClassifier — predict_proba
    // ====================================================================

    #[test]
    fn test_classifier_predict_proba_shape() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let proba = rf.predict_proba(&x);
        // 8 samples, 2 classes
        assert_eq!(proba.shape(), (8, 2));
    }

    #[test]
    fn test_classifier_predict_proba_sums_to_one() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let proba = rf.predict_proba(&x);

        for row in 0..proba.shape().0 {
            let sum: f32 = (0..proba.shape().1).map(|col| proba.get(row, col)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row {row} probabilities sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_classifier_predict_proba_values_in_range() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let proba = rf.predict_proba(&x);

        for row in 0..proba.shape().0 {
            for col in 0..proba.shape().1 {
                let val = proba.get(row, col);
                assert!(
                    (0.0..=1.0).contains(&val),
                    "Probability at ({row},{col}) = {val} out of range"
                );
            }
        }
    }

    // ====================================================================
    // RandomForestClassifier — OOB
    // ====================================================================

    #[test]
    fn test_classifier_oob_prediction_returns_some_after_fit() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let oob = rf.oob_prediction();
        assert!(oob.is_some(), "OOB prediction should be Some after fit");
        assert_eq!(oob.expect("checked above").len(), 8);
    }

    #[test]
    fn test_classifier_oob_prediction_returns_none_before_fit() {
        let rf = RandomForestClassifier::new(5);
        assert!(rf.oob_prediction().is_none());
    }

    #[test]
    fn test_classifier_oob_score_returns_some_after_fit() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
        let score_val = oob_score.expect("checked above");
        assert!(
            (0.0..=1.0).contains(&score_val),
            "OOB score {score_val} should be in [0, 1]"
        );
    }

    #[test]
    fn test_classifier_oob_score_returns_none_before_fit() {
        let rf = RandomForestClassifier::new(3);
        assert!(rf.oob_score().is_none());
    }

    // ====================================================================
    // RandomForestClassifier — feature importances
    // ====================================================================

    #[test]
    fn test_classifier_feature_importances_returns_some_after_fit() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let importances = rf.feature_importances();
        assert!(importances.is_some());
        let imp = importances.expect("checked above");
        assert_eq!(
            imp.len(),
            2,
            "Should have importance for each of 2 features"
        );
    }

    #[test]
    fn test_classifier_feature_importances_sum_to_one() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let imp = rf.feature_importances().expect("should be Some after fit");
        let sum: f32 = imp.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Feature importances sum to {sum}, expected ~1.0"
        );
    }

    #[test]
    fn test_classifier_feature_importances_nonnegative() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let imp = rf.feature_importances().expect("should be Some after fit");
        for (i, &val) in imp.iter().enumerate() {
            assert!(
                val >= 0.0,
                "Feature importance [{i}] = {val} should be >= 0"
            );
        }
    }

    #[test]
    fn test_classifier_feature_importances_returns_none_before_fit() {
        let rf = RandomForestClassifier::new(3);
        assert!(rf.feature_importances().is_none());
    }

    // ====================================================================
    // RandomForestClassifier — edge cases
    // ====================================================================

    #[test]
    fn test_classifier_single_tree() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(1)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 1);
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_classifier_single_feature() {
        // 1-dimensional feature space
        let x = Matrix::from_vec(6, 1, vec![0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
            .expect("single feature matrix");
        let y = vec![0, 0, 0, 1, 1, 1];
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_classifier_no_max_depth() {
        // Fit without setting max_depth (exercises the `else` branch)
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_classifier_many_trees() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(50)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 50);
        let score = rf.score(&x, &y);
        // With 50 trees on linearly separable data, score should be high
        assert!(
            score >= 0.5,
            "Score with 50 trees should be decent, got {score}"
        );
    }

    // ====================================================================
    // RandomForestClassifier — save/load safetensors
    // ====================================================================

    #[test]
    fn test_classifier_save_unfitted_returns_error() {
        let rf = RandomForestClassifier::new(3);
        let result = rf.save_safetensors("/tmp/aprender_test_unfitted_rf.safetensors");
        assert!(result.is_err());
        assert!(result.expect_err("should be error").contains("unfitted"),);
    }

    #[test]
    fn test_classifier_save_and_load_roundtrip() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let path = "/tmp/aprender_test_rf_roundtrip.safetensors";
        rf.save_safetensors(path).expect("save should succeed");

        let loaded = RandomForestClassifier::load_safetensors(path).expect("load should succeed");
        assert_eq!(loaded.n_estimators, 3);
        assert_eq!(loaded.max_depth, Some(4));
        assert_eq!(loaded.random_state, Some(42));
        assert_eq!(loaded.trees.len(), 3);

        // Loaded model should produce same predictions
        let orig_preds = rf.predict(&x);
        let loaded_preds = loaded.predict(&x);
        assert_eq!(
            orig_preds, loaded_preds,
            "Loaded model predictions should match original"
        );

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_classifier_load_nonexistent_file_returns_error() {
        let result =
            RandomForestClassifier::load_safetensors("/tmp/aprender_no_such_file.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_classifier_save_load_no_max_depth() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(2).with_random_state(7);
        rf.fit(&x, &y).expect("fit should succeed");

        let path = "/tmp/aprender_test_rf_no_depth.safetensors";
        rf.save_safetensors(path).expect("save should succeed");

        let loaded = RandomForestClassifier::load_safetensors(path).expect("load should succeed");
        assert!(
            loaded.max_depth.is_none(),
            "max_depth should remain None after round-trip"
        );

        let _ = std::fs::remove_file(path);
    }
