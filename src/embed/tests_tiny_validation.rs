
    #[test]
    fn test_knn_k_zero() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![0, 1],
            0, // k = 0
        );

        assert!(matches!(
            model.validate(),
            Err(TinyModelError::InvalidK { k: 0, .. })
        ));
    }

    #[test]
    fn test_knn_empty_reference_points() {
        let model = TinyModelRepr::knn(vec![], vec![], 1);

        assert!(matches!(model.validate(), Err(TinyModelError::EmptyModel)));
    }

    #[test]
    fn test_naive_bayes_empty_priors() {
        let model = TinyModelRepr::naive_bayes(vec![], vec![], vec![]);

        assert!(matches!(model.validate(), Err(TinyModelError::EmptyModel)));
    }

    #[test]
    fn test_linear_predict_on_non_linear() {
        let model = TinyModelRepr::stump(0, 0.5, -1.0, 1.0);
        assert!(model.predict_linear(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_stump_predict_on_non_stump() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0], 0.0);
        assert!(model.predict_stump(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_kmeans_predict_on_non_kmeans() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0], 0.0);
        assert!(model.predict_kmeans(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_stump_validate_always_ok() {
        let model = TinyModelRepr::stump(0, 0.5, -1.0, 1.0);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_logistic_regression_validate_always_ok() {
        let model = TinyModelRepr::logistic_regression(vec![vec![0.1, 0.2]], vec![0.3]);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_compressed_validate_always_ok() {
        let model = TinyModelRepr::compressed(DataCompression::zstd(), vec![1, 2, 3], 10);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_model_partial_eq() {
        let model1 = TinyModelRepr::linear(vec![1.0, 2.0], 0.5);
        let model2 = TinyModelRepr::linear(vec![1.0, 2.0], 0.5);
        let model3 = TinyModelRepr::linear(vec![1.0, 2.0], 0.6);

        assert_eq!(model1, model2);
        assert_ne!(model1, model3);
    }

    #[test]
    fn test_logistic_regression_n_parameters() {
        let model = TinyModelRepr::logistic_regression(
            vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            vec![0.7, 0.8],
        );

        // 2 classes * 3 features + 2 intercepts = 6 + 2 = 8
        assert_eq!(model.n_parameters(), 8);
    }

    #[test]
    fn test_knn_n_parameters() {
        let model = TinyModelRepr::knn(
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            vec![0, 1, 0],
            2,
        );

        // 3 points * 2 features = 6
        assert_eq!(model.n_parameters(), 6);
    }

    #[test]
    fn test_error_source() {
        let err = TinyModelError::EmptyModel;
        // Test that std::error::Error is implemented
        let _source = std::error::Error::source(&err);
    }

    #[test]
    fn test_tiny_model_clone() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0], 0.5);
        let cloned = model.clone();
        assert_eq!(model, cloned);
    }

    #[test]
    fn test_tiny_model_debug() {
        let model = TinyModelRepr::linear(vec![1.0, 2.0], 0.5);
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("Linear"));
        assert!(debug_str.contains("coefficients"));
    }

    #[test]
    fn test_naive_bayes_n_parameters() {
        let model = TinyModelRepr::naive_bayes(
            vec![0.3, 0.7],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        );

        // 2 priors + 2*2 means + 2*2 variances = 2 + 4 + 4 = 10
        assert_eq!(model.n_parameters(), 10);
    }

    #[test]
    fn test_kmeans_n_features_empty() {
        let model = TinyModelRepr::kmeans(vec![]);
        assert!(model.n_features().is_none());
    }

    #[test]
    fn test_logistic_regression_n_features_empty() {
        let model = TinyModelRepr::logistic_regression(vec![], vec![]);
        assert!(model.n_features().is_none());
    }

    #[test]
    fn test_knn_n_features_empty() {
        let model = TinyModelRepr::knn(vec![], vec![], 1);
        assert!(model.n_features().is_none());
    }
