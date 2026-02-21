
    #[test]
    fn test_linear_model() {
        let model = TinyModelRepr::linear(vec![0.5, -0.3, 0.8], 1.0);

        assert_eq!(model.model_type(), "linear");
        assert_eq!(model.size_bytes(), 16); // 3*4 + 4
        assert_eq!(model.n_parameters(), 4); // 3 coefs + 1 intercept
        assert_eq!(model.n_features(), Some(3));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_linear_predict() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0, 3.0], 1.0);

        // 1*1 + 2*2 + 3*3 + 1 = 1 + 4 + 9 + 1 = 15
        let pred = model.predict_linear(&[1.0, 2.0, 3.0]);
        assert!((pred.unwrap() - 15.0).abs() < f32::EPSILON);

        // Wrong number of features
        assert!(model.predict_linear(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_stump_model() {
        let model = TinyModelRepr::stump(2, 0.5, -1.0, 1.0);

        assert_eq!(model.model_type(), "stump");
        assert_eq!(model.size_bytes(), 14);
        assert_eq!(model.n_parameters(), 4);
    }

    #[test]
    fn test_stump_predict() {
        let model = TinyModelRepr::stump(1, 0.5, -1.0, 1.0);

        // Feature 1 < 0.5 -> left value (-1.0)
        assert_eq!(model.predict_stump(&[0.0, 0.3, 0.0]), Some(-1.0));

        // Feature 1 >= 0.5 -> right value (1.0)
        assert_eq!(model.predict_stump(&[0.0, 0.7, 0.0]), Some(1.0));

        // Feature 1 == 0.5 -> right value (>= threshold)
        assert_eq!(model.predict_stump(&[0.0, 0.5, 0.0]), Some(1.0));

        // Out of bounds feature index
        assert!(model.predict_stump(&[0.0]).is_none());
    }

    #[test]
    fn test_naive_bayes_model() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.5, 0.5],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        );

        assert_eq!(model.model_type(), "naive_bayes");
        assert_eq!(model.n_features(), Some(2));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_naive_bayes_invalid_variance() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.5, 0.5],
            vec![vec![1.0], vec![2.0]],
            vec![vec![0.1], vec![-0.1]], // negative variance
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidVariance { .. })
        ));
    }

    #[test]
    fn test_kmeans_model() {
        let model = TinyModelRepr::kmeans(vec![vec![1.0, 2.0], vec![4.0, 5.0], vec![7.0, 8.0]]);

        assert_eq!(model.model_type(), "kmeans");
        assert_eq!(model.size_bytes(), 24); // 3 * 2 * 4
        assert_eq!(model.n_features(), Some(2));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_kmeans_predict() {
        let model = TinyModelRepr::kmeans(vec![vec![0.0, 0.0], vec![10.0, 10.0]]);

        // Closer to cluster 0
        assert_eq!(model.predict_kmeans(&[1.0, 1.0]), Some(0));

        // Closer to cluster 1
        assert_eq!(model.predict_kmeans(&[9.0, 9.0]), Some(1));

        // Wrong feature count
        assert!(model.predict_kmeans(&[1.0]).is_none());
    }

    #[test]
    fn test_kmeans_shape_mismatch() {
        let model = TinyModelRepr::kmeans(vec![
            vec![1.0, 2.0],
            vec![3.0], // different length
        ]);

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_logistic_regression_model() {
        let model = TinyModelRepr::logistic_regression(
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            vec![0.5, 0.6],
        );

        assert_eq!(model.model_type(), "logistic_regression");
        assert_eq!(model.n_features(), Some(2));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_knn_model() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
        );

        assert_eq!(model.model_type(), "knn");
        assert_eq!(model.n_features(), Some(2));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_knn_invalid_k() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0, 1],
            5, // k > n_samples
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidK { .. })
        ));
    }

    #[test]
    fn test_compressed_model() {
        let model = TinyModelRepr::compressed(DataCompression::zstd(), vec![1, 2, 3, 4, 5], 100);

        assert_eq!(model.model_type(), "compressed");
        assert_eq!(model.size_bytes(), 5);
    }

    #[test]
    fn test_fits_within() {
        let small = TinyModelRepr::linear(vec![1.0, 2.0], 0.0);
        let large = TinyModelRepr::kmeans(vec![vec![0.0; 1000]; 100]);

        assert!(small.fits_within(100));
        assert!(!small.fits_within(5));
        assert!(large.fits_within(1_000_000));
        assert!(!large.fits_within(100));
    }

    #[test]
    fn test_summary() {
        let linear = TinyModelRepr::linear(vec![1.0, 2.0, 3.0], 0.0);
        let summary = linear.summary();
        assert!(summary.contains("Linear"));
        assert!(summary.contains("3 features"));

        let stump = TinyModelRepr::stump(5, 0.5, -1.0, 1.0);
        let summary = stump.summary();
        assert!(summary.contains("Stump"));
        assert!(summary.contains("feature[5]"));

        let kmeans = TinyModelRepr::kmeans(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let summary = kmeans.summary();
        assert!(summary.contains("KMeans"));
        assert!(summary.contains("2 clusters"));
    }

    #[test]
    fn test_empty_model_validation() {
        let empty_linear = TinyModelRepr::linear(vec![], 0.0);
        assert!(matches!(
            empty_linear.validate(),
            Err(TinyModelError::EmptyModel)
        ));

        let empty_kmeans = TinyModelRepr::kmeans(vec![]);
        assert!(matches!(
            empty_kmeans.validate(),
            Err(TinyModelError::EmptyModel)
        ));
    }

    #[test]
    fn test_invalid_coefficient() {
        let model = TinyModelRepr::linear(vec![1.0, f32::NAN, 3.0], 0.0);
        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidCoefficient { index: 1, .. })
        ));

        let model = TinyModelRepr::linear(vec![f32::INFINITY, 2.0], 0.0);
        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidCoefficient { index: 0, .. })
        ));
    }

    #[test]
    fn test_tiny_model_error_display() {
        let err = TinyModelError::EmptyModel;
        assert_eq!(format!("{err}"), "Model has no parameters");

        let err = TinyModelError::InvalidK {
            k: 10,
            n_samples: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_naive_bayes_shape_mismatch() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.5, 0.5],
            vec![vec![1.0]], // only 1 class but 2 priors
            vec![vec![0.1], vec![0.2]],
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_knn_labels_mismatch() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0], // only 1 label for 2 points
            1,
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::ShapeMismatch { .. })
        ));
    }

    // ============================================================================
    // Additional Coverage Tests
    // ============================================================================

    #[test]
    fn test_kmeans_predict_empty_centroids() {
        let model = TinyModelRepr::kmeans(vec![]);
        assert!(model.predict_kmeans(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_logistic_regression_size_bytes() {
        let model = TinyModelRepr::logistic_regression(
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            vec![0.7, 0.8],
        );

        // 2 classes * 3 features * 4 bytes + 2 intercepts * 4 bytes = 24 + 8 = 32
        assert_eq!(model.size_bytes(), 32);
    }

    #[test]
    fn test_knn_size_bytes() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
        );

        // 3 points * 2 features * 4 bytes + 3 labels * 4 bytes + 4 (k)
        // = 24 + 12 + 4 = 40
        assert_eq!(model.size_bytes(), 40);
    }

    #[test]
    fn test_compressed_summary() {
        let model = TinyModelRepr::compressed(DataCompression::None, vec![1, 2, 3, 4, 5], 100);

        let summary = model.summary();
        assert!(summary.contains("Compressed"));
        assert!(summary.contains("5 bytes"));
        assert!(summary.contains("ratio"));
    }

    #[test]
    fn test_compressed_summary_empty() {
        let model = TinyModelRepr::compressed(DataCompression::zstd(), vec![], 0);

        let summary = model.summary();
        assert!(summary.contains("Compressed"));
        // Should handle empty data without panicking
    }

    #[test]
    fn test_logistic_regression_summary() {
        let model = TinyModelRepr::logistic_regression(
            vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
            vec![0.1, 0.2, 0.3],
        );

        let summary = model.summary();
        assert!(summary.contains("LogisticRegression"));
        assert!(summary.contains("3 classes"));
        assert!(summary.contains("2 features"));
    }

    #[test]
    fn test_knn_summary() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
        );

        let summary = model.summary();
        assert!(summary.contains("KNN"));
        assert!(summary.contains("k=2"));
        assert!(summary.contains("3 samples"));
        assert!(summary.contains("2 features"));
    }

    #[test]
    fn test_naive_bayes_summary() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.3, 0.7],
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        );

        let summary = model.summary();
        assert!(summary.contains("NaiveBayes"));
        assert!(summary.contains("2 classes"));
        assert!(summary.contains("3 features"));
    }

    #[test]
    fn test_feature_mismatch_error_display() {
        let err = TinyModelError::FeatureMismatch {
            expected: 10,
            got: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_invalid_coefficient_display() {
        let err = TinyModelError::InvalidCoefficient {
            index: 3,
            value: f32::NAN,
        };
        let msg = format!("{err}");
        assert!(msg.contains("3"));
        assert!(msg.contains("NaN"));
    }

    #[test]
    fn test_invalid_variance_display() {
        let err = TinyModelError::InvalidVariance {
            class: 1,
            feature: 2,
            value: -0.5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("class 1"));
        assert!(msg.contains("feature 2"));
        assert!(msg.contains("-0.5"));
    }

    #[test]
    fn test_shape_mismatch_display() {
        let err = TinyModelError::ShapeMismatch {
            message: "test error".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("test error"));
    }

    #[test]
    fn test_n_features_stump() {
        let model = TinyModelRepr::stump(5, 0.5, -1.0, 1.0);
        assert!(model.n_features().is_none());
    }

    #[test]
    fn test_n_features_compressed() {
        let model = TinyModelRepr::compressed(DataCompression::None, vec![1, 2, 3], 100);
        assert!(model.n_features().is_none());
    }

    #[test]
    fn test_n_parameters_compressed() {
        let model = TinyModelRepr::compressed(DataCompression::None, vec![0; 40], 160);
        // original_size / 4 = 160 / 4 = 40
        assert_eq!(model.n_parameters(), 40);
    }

    #[test]
    fn test_naive_bayes_size_bytes() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.5, 0.5],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        );

        // 2 priors * 4 + 2 classes * 2 features * 8 (means + variances) = 8 + 32 = 40
        assert_eq!(model.size_bytes(), 40);
    }

    #[test]
    fn test_naive_bayes_empty_means() {
        let model = TinyModelRepr::naive_bayes(vec![1.0], vec![], vec![]);

        // With empty means, n_features should be 0
        assert_eq!(model.n_features(), None);
    }
