
    #[test]
    fn test_classifier_save_load_no_random_state() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(2).with_max_depth(2);
        rf.fit(&x, &y).expect("fit should succeed");

        let path = "/tmp/aprender_test_rf_no_rs.safetensors";
        rf.save_safetensors(path).expect("save should succeed");

        let loaded = RandomForestClassifier::load_safetensors(path).expect("load should succeed");
        assert!(
            loaded.random_state.is_none(),
            "random_state should remain None after round-trip"
        );

        let _ = std::fs::remove_file(path);
    }

    // ====================================================================
    // RandomForestRegressor — construction
    // ====================================================================

    #[test]
    fn test_regressor_new_sets_n_estimators() {
        let rf = RandomForestRegressor::new(7);
        assert_eq!(rf.n_estimators, 7);
        assert!(rf.trees.is_empty());
        assert!(rf.max_depth.is_none());
        assert!(rf.random_state.is_none());
    }

    #[test]
    fn test_regressor_default() {
        let rf = RandomForestRegressor::default();
        assert_eq!(rf.n_estimators, 10);
        assert!(rf.trees.is_empty());
    }

    #[test]
    fn test_regressor_with_max_depth() {
        let rf = RandomForestRegressor::new(3).with_max_depth(6);
        assert_eq!(rf.max_depth, Some(6));
    }

    #[test]
    fn test_regressor_with_random_state() {
        let rf = RandomForestRegressor::new(3).with_random_state(123);
        assert_eq!(rf.random_state, Some(123));
    }

    // ====================================================================
    // RandomForestRegressor — fit / predict
    // ====================================================================

    #[test]
    fn test_regressor_fit_creates_correct_number_of_trees() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 5);
    }

    #[test]
    fn test_regressor_predict_returns_correct_length() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_regressor_predictions_are_averaged() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);

        // Average predictions should be within a reasonable range of actual
        for i in 0..preds.len() {
            let pred = preds.as_slice()[i];
            let actual = y.as_slice()[i];
            assert!(
                (pred - actual).abs() < 6.0,
                "Prediction {pred} too far from actual {actual} at index {i}"
            );
        }
    }

    #[test]
    fn test_regressor_score_returns_valid_value() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let score = rf.score(&x, &y);
        // R² can be negative for very bad fits, but on training data it should be positive
        assert!(
            score > -1.0 && score <= 1.0,
            "R² score {score} seems unreasonable"
        );
    }

    #[test]
    fn test_regressor_reproducibility_with_random_state() {
        let (x, y) = regression_data();
        let mut rf1 = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let preds1 = rf1.predict(&x);

        let mut rf2 = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let preds2 = rf2.predict(&x);

        for i in 0..preds1.len() {
            assert!(
                (preds1.as_slice()[i] - preds2.as_slice()[i]).abs() < 1e-6,
                "Predictions differ at index {i}"
            );
        }
    }

    // ====================================================================
    // RandomForestRegressor — fit error paths
    // ====================================================================

    #[test]
    fn test_regressor_fit_mismatched_dimensions() {
        let x = Matrix::from_vec(4, 2, vec![1.0; 8]).expect("matrix creation");
        let y = Vector::from_slice(&[1.0, 2.0, 3.0]); // 3 != 4
        let mut rf = RandomForestRegressor::new(3);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_regressor_fit_zero_samples() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("empty matrix");
        let y = Vector::from_slice(&[]);
        let mut rf = RandomForestRegressor::new(3);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    // ====================================================================
    // RandomForestRegressor — OOB
    // ====================================================================

    #[test]
    fn test_regressor_oob_prediction_returns_some_after_fit() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let oob = rf.oob_prediction();
        assert!(oob.is_some(), "OOB prediction should be Some after fit");
        assert_eq!(oob.expect("checked above").len(), 8);
    }

    #[test]
    fn test_regressor_oob_prediction_returns_none_before_fit() {
        let rf = RandomForestRegressor::new(5);
        assert!(rf.oob_prediction().is_none());
    }

    #[test]
    fn test_regressor_oob_score_returns_some_after_fit() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
    }

    #[test]
    fn test_regressor_oob_score_returns_none_before_fit() {
        let rf = RandomForestRegressor::new(3);
        assert!(rf.oob_score().is_none());
    }

    // ====================================================================
    // RandomForestRegressor — feature importances
    // ====================================================================

    #[test]
    fn test_regressor_feature_importances_returns_some_after_fit() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let imp = rf.feature_importances();
        assert!(imp.is_some());
        assert_eq!(imp.expect("checked above").len(), 2);
    }

    #[test]
    fn test_regressor_feature_importances_sum_to_one() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
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
    fn test_regressor_feature_importances_nonnegative() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
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
    fn test_regressor_feature_importances_returns_none_before_fit() {
        let rf = RandomForestRegressor::new(3);
        assert!(rf.feature_importances().is_none());
    }

    // ====================================================================
    // RandomForestRegressor — edge cases
    // ====================================================================

    #[test]
    fn test_regressor_single_tree() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(1)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 1);
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_regressor_no_max_depth() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_regressor_single_feature() {
        let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("single feature matrix");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 6);
    }

    #[test]
    #[should_panic(expected = "Cannot predict with an unfitted Random Forest")]
    fn test_regressor_predict_before_fit_panics() {
        let rf = RandomForestRegressor::new(3);
        let x = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix");
        let _ = rf.predict(&x);
    }

    #[test]
    fn test_regressor_stores_training_data() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert!(rf.x_train.is_some());
        assert!(rf.y_train.is_some());
        assert_eq!(rf.oob_indices.len(), 3);
    }

    #[test]
    fn test_classifier_stores_training_data() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert!(rf.x_train.is_some());
        assert!(rf.y_train.is_some());
        assert_eq!(rf.oob_indices.len(), 3);
    }

    #[test]
    fn test_classifier_predict_proba_without_y_train_defaults_to_2_classes() {
        // Build a classifier, fit it, then clear y_train to exercise the default path
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        rf.y_train = None;
        let proba = rf.predict_proba(&x);
        // Without y_train, n_classes defaults to 2
        assert_eq!(proba.shape().1, 2);
    }

    #[test]
    fn test_regressor_fit_overwrites_previous_fit() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 3);

        // Fit again with different estimator count (by modifying n_estimators)
        rf.n_estimators = 5;
        rf.fit(&x, &y).expect("second fit should succeed");
        assert_eq!(rf.trees.len(), 5);
    }

    #[test]
    fn test_classifier_clone() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let cloned = rf.clone();
        let orig_preds = rf.predict(&x);
        let cloned_preds = cloned.predict(&x);
        assert_eq!(orig_preds, cloned_preds);
    }

    #[test]
    fn test_regressor_clone() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let cloned = rf.clone();
        let orig_preds = rf.predict(&x);
        let cloned_preds = cloned.predict(&x);
        for i in 0..orig_preds.len() {
            assert!(
                (orig_preds.as_slice()[i] - cloned_preds.as_slice()[i]).abs() < 1e-6,
                "Cloned predictions differ at index {i}"
            );
        }
    }
