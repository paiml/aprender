
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_state_creation() {
        let state = ServeState::new("test-model".to_string(), "v1.0".to_string());
        assert_eq!(state.model_name, "test-model");
        assert_eq!(state.model_version, "v1.0");
    }

    #[test]
    fn test_predict_request_serialization() {
        let request = PredictRequest {
            model_id: Some("sentiment-v1".to_string()),
            features: vec![0.5, 1.2, -0.3, 0.8],
            options: Some(PredictOptions {
                return_probabilities: true,
                top_k: Some(3),
            }),
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("sentiment-v1"));
        assert!(json.contains("0.5"));
        assert!(json.contains("return_probabilities"));
    }

    #[test]
    fn test_predict_response_serialization() {
        let response = PredictResponse {
            prediction: 1.0,
            probabilities: Some(vec![0.12, 0.85, 0.03]),
            latency_ms: 2.3,
            model_version: "v1.2.0".to_string(),
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("v1.2.0"));
        assert!(json.contains("2.3"));
        assert!(json.contains("0.12"));
    }

    #[test]
    fn test_batch_predict_request_serialization() {
        let request = BatchPredictRequest {
            model_id: Some("model-v1".to_string()),
            instances: vec![
                PredictInstance {
                    features: vec![0.5, 1.2],
                },
                PredictInstance {
                    features: vec![0.1, 0.9],
                },
            ],
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("model-v1"));
        assert!(json.contains("instances"));
        assert!(json.contains("0.5"));
        assert!(json.contains("0.9"));
    }

    #[test]
    fn test_health_response_format() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.2.0".to_string(),
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("healthy"));
        assert!(json.contains("0.2.0"));
    }

    #[test]
    fn test_ready_response_format() {
        let response = ReadyResponse {
            ready: true,
            model_loaded: true,
            model_name: "test-model".to_string(),
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("true"));
        assert!(json.contains("test-model"));
    }

    #[test]
    fn test_error_response_format() {
        let response = ErrorResponse {
            error: "Model not found".to_string(),
            code: Some("E404".to_string()),
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("Model not found"));
        assert!(json.contains("E404"));
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await;
        assert_eq!(response.0.status, "healthy");
        assert!(!response.0.version.is_empty());
    }

    #[tokio::test]
    async fn test_ready_handler_no_model() {
        let state = ServeState::new("test-model".to_string(), "v1.0".to_string());
        let response = ready_handler(State(state)).await;
        // Without a model loaded, ready should be false
        assert!(!response.0.ready);
        assert!(!response.0.model_loaded);
        assert_eq!(response.0.model_name, "test-model");
    }

    #[test]
    fn test_serve_state_has_model() {
        let state = ServeState::new("test".to_string(), "v1".to_string());
        assert!(!state.has_model());
    }

    #[test]
    fn test_models_info_serialization() {
        let info = ModelInfo {
            id: "mnist-v1".to_string(),
            model_type: "LogisticRegression".to_string(),
            version: "1.0.0".to_string(),
            loaded: true,
        };

        let json = serde_json::to_string(&info).expect("serialization failed");
        assert!(json.contains("mnist-v1"));
        assert!(json.contains("LogisticRegression"));
    }

    /// Integration test: Train a model, serve it, and make predictions
    #[cfg(feature = "aprender-serve")]
    #[tokio::test]
    async fn test_predict_with_loaded_model() {
        // Train a simple model
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 matrix");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(100);
        model.fit(&x, &y).expect("Training should succeed");

        // Create serve state with model
        let state = ServeState::with_logistic_regression(model, "test-v1".to_string(), 2);
        assert!(state.has_model());

        // Create prediction request
        let request = PredictRequest {
            model_id: None,
            features: vec![0.9, 0.9], // Should predict class 1
            options: Some(PredictOptions {
                return_probabilities: true,
                top_k: None,
            }),
        };

        // Call predict handler
        let result = predict_handler(State(state.clone()), Json(request)).await;
        let response = result.expect("Prediction should succeed");

        // Verify response
        assert_eq!(response.prediction, 1.0); // Should predict class 1
        assert!(response.probabilities.is_some());
        assert!(response.latency_ms < 10.0); // Should be sub-10ms
        assert_eq!(response.model_version, "test-v1");

        // Test class 0 prediction
        let request_0 = PredictRequest {
            model_id: None,
            features: vec![0.0, 0.0], // Should predict class 0
            options: None,
        };
        let result_0 = predict_handler(State(state), Json(request_0)).await;
        let response_0 = result_0.expect("Prediction should succeed");
        assert_eq!(response_0.prediction, 0.0);
    }

    /// Integration test: Batch prediction
    #[cfg(feature = "aprender-serve")]
    #[tokio::test]
    async fn test_batch_predict_with_loaded_model() {
        // Train a simple model
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 matrix");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(100);
        model.fit(&x, &y).expect("Training should succeed");

        let state = ServeState::with_logistic_regression(model, "batch-v1".to_string(), 2);

        // Create batch request
        let request = BatchPredictRequest {
            model_id: None,
            instances: vec![
                PredictInstance {
                    features: vec![0.0, 0.0],
                },
                PredictInstance {
                    features: vec![1.0, 1.0],
                },
            ],
        };

        let result = batch_predict_handler(State(state), Json(request)).await;
        let response = result.expect("Batch prediction should succeed");

        assert_eq!(response.predictions.len(), 2);
        assert_eq!(response.predictions[0].prediction, 0.0); // Class 0
        assert_eq!(response.predictions[1].prediction, 1.0); // Class 1
        assert!(response.total_latency_ms < 10.0);
    }

    /// Test error handling for invalid input dimensions
    #[cfg(feature = "aprender-serve")]
    #[tokio::test]
    async fn test_predict_invalid_dimensions() {
        // Train model with 2 features
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 matrix");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).expect("Training should succeed");

        let state = ServeState::with_logistic_regression(model, "v1".to_string(), 2);

        // Request with wrong number of features
        let request = PredictRequest {
            model_id: None,
            features: vec![1.0, 2.0, 3.0], // 3 features, expected 2
            options: None,
        };

        let result = predict_handler(State(state), Json(request)).await;
        assert!(result.is_err());
        let (status, error) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error.error.contains("Invalid input dimension"));
    }
}
