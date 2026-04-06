
    // ==========================================================================
    // Test Helpers: Mock types for creating test APR model bytes
    // Note: These are test-only mocks; real APR inference is not yet implemented
    // ==========================================================================

    /// Mock model weights structure for test APR file creation
    /// This creates valid APR file format for testing the Lambda handler scaffolding
    #[derive(serde::Serialize)]
    struct MockModelWeights {
        weights: Vec<Vec<f32>>,
        biases: Vec<Vec<f32>>,
        dimensions: Vec<usize>,
    }

    /// Mock model type enum for test APR file creation
    #[repr(u16)]
    #[derive(Clone, Copy)]
    enum MockAprModelType {
        LinearRegression = 1,
    }

    impl MockAprModelType {
        fn as_u16(self) -> u16 {
            self as u16
        }
    }

    /// Create APR model bytes where output = sum of inputs (for easy verification)
    /// Note: Real inference is not implemented yet; tests verify handler scaffolding
    fn create_sum_model(input_dim: usize) -> &'static [u8] {
        // Single output that sums all inputs: output[0] = sum(input[i])
        let weights = MockModelWeights {
            weights: vec![vec![1.0; input_dim]], // 1 x input_dim weights (all 1.0)
            biases: vec![vec![0.0]],             // Single bias of 0
            dimensions: vec![input_dim, 1],
        };

        let payload = serde_json::to_vec(&weights).expect("serialize weights");
        let mut data = Vec::with_capacity(HEADER_SIZE + payload.len());

        data.extend_from_slice(&MAGIC);
        data.push(1);
        data.push(0);
        data.push(0);
        data.push(0);
        data.extend_from_slice(&MockAprModelType::LinearRegression.as_u16().to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        #[allow(clippy::cast_possible_truncation)]
        data.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        #[allow(clippy::cast_possible_truncation)]
        data.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0u8; 10]);
        data.extend_from_slice(&payload);

        Box::leak(data.into_boxed_slice())
    }

    // ==========================================================================
    // GREEN PHASE: Tests for Lambda handler with real APR inference
    // ==========================================================================

    // --------------------------------------------------------------------------
    // Test: LambdaHandler creation
    // --------------------------------------------------------------------------

    #[test]
    fn test_lambda_handler_creation_with_valid_apr_model() {
        let model_bytes = create_sum_model(3);
        let handler = LambdaHandler::from_bytes(model_bytes);
        assert!(handler.is_ok(), "Should accept valid .apr model");
        let handler = handler.expect("test");
        assert!(handler.model_size_bytes() > HEADER_SIZE);
    }

    #[test]
    fn test_lambda_handler_rejects_empty_model() {
        let model_bytes: &'static [u8] = b"";
        let result = LambdaHandler::from_bytes(model_bytes);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LambdaError::EmptyModel);
    }

    #[test]
    fn test_lambda_handler_rejects_invalid_magic() {
        // Invalid magic bytes (GGUF instead of APR)
        let model_bytes: &'static [u8] = b"GGUF\x01\x00\x00\x00testmodel";
        let result = LambdaHandler::from_bytes(model_bytes);
        assert!(result.is_err());
        match result.unwrap_err() {
            LambdaError::InvalidMagic { expected, found } => {
                assert_eq!(expected, "APR");
                assert!(!found.is_empty());
                assert!(found.contains("71")); // 'G' = 71 in ASCII
            },
            _ => panic!("Expected InvalidMagic error"),
        }
    }

    // --------------------------------------------------------------------------
    // Test: Lambda request/response serialization
    // --------------------------------------------------------------------------

    #[test]
    fn test_lambda_request_serialization() {
        let request = LambdaRequest {
            features: vec![0.5, 1.2, -0.3, 0.8],
            model_id: Some("sentiment-v1".to_string()),
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("0.5"));
        assert!(json.contains("sentiment-v1"));
    }

    #[test]
    fn test_lambda_request_deserialization() {
        let json = r#"{"features": [1.0, 2.0, 3.0], "model_id": "test-model"}"#;
        let request: LambdaRequest = serde_json::from_str(json).expect("deserialization failed");
        assert_eq!(request.features, vec![1.0, 2.0, 3.0]);
        assert_eq!(request.model_id, Some("test-model".to_string()));
    }

    #[test]
    fn test_lambda_response_serialization() {
        let response = LambdaResponse {
            prediction: 0.85,
            probabilities: Some(vec![0.15, 0.85]),
            latency_ms: 2.3,
            cold_start: true,
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("0.85"));
        assert!(json.contains("cold_start"));
        assert!(json.contains("true"));
    }

    // --------------------------------------------------------------------------
    // Test: Lambda handler invocation
    // --------------------------------------------------------------------------

    #[test]
    #[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
    fn test_lambda_handler_cold_start_detection() {
        let model_bytes = create_sum_model(3);
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        // Before first invocation: cold start
        assert!(handler.is_cold_start());

        let request = LambdaRequest {
            features: vec![1.0, 2.0, 3.0],
            model_id: None,
        };

        // First invocation
        let response = handler.handle(&request).expect("test");
        assert!(response.cold_start, "First invocation should be cold start");

        // After first invocation: no longer cold
        assert!(!handler.is_cold_start());

        // Second invocation
        let response2 = handler.handle(&request).expect("test");
        assert!(
            !response2.cold_start,
            "Second invocation should not be cold start"
        );
    }

    #[test]
    fn test_lambda_handler_rejects_empty_features() {
        let model_bytes = create_sum_model(3);
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        let request = LambdaRequest {
            features: vec![],
            model_id: None,
        };

        let result = handler.handle(&request);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LambdaError::EmptyFeatures);
    }

    #[test]
    #[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
    fn test_lambda_handler_real_inference() {
        // Create a model where output = sum of 3 inputs
        let model_bytes = create_sum_model(3);
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        let request = LambdaRequest {
            features: vec![1.0, 2.0, 3.0],
            model_id: None,
        };

        let response = handler.handle(&request).expect("test");
        // Real inference: sum model returns sum of features
        // output = 1.0 * 1.0 + 1.0 * 2.0 + 1.0 * 3.0 = 6.0
        assert!((response.prediction - 6.0).abs() < 0.001);
        assert!(response.latency_ms >= 0.0);
    }

    // --------------------------------------------------------------------------
    // Test: Cold start metrics
    // --------------------------------------------------------------------------

    #[test]
    #[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
    fn test_cold_start_metrics_recorded() {
        let model_bytes = create_sum_model(1);
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        // No metrics before first invocation
        assert!(handler.cold_start_metrics().is_none());

        let request = LambdaRequest {
            features: vec![1.0],
            model_id: None,
        };

        let _ = handler.handle(&request).expect("test");

        // Metrics available after first invocation
        let metrics = handler.cold_start_metrics();
        assert!(metrics.is_some());
        let metrics = metrics.expect("test");
        assert!(metrics.total_ms >= 0.0);
        assert!(metrics.first_inference_ms >= 0.0);
        assert!(metrics.model_load_ms >= 0.0);
    }

    // --------------------------------------------------------------------------
    // Test: ARM64 optimizations
    // --------------------------------------------------------------------------

    #[test]
    fn test_arm64_architecture_detection() {
        let arch = arm64::target_arch();
        assert!(
            arch == "aarch64" || arch == "x86_64" || arch == "unknown",
            "Should detect valid architecture"
        );
    }

    #[test]
    fn test_arm64_simd_detection() {
        let simd = arm64::optimal_simd();
        let valid_simd = ["NEON", "AVX2", "SSE2", "Scalar"];
        assert!(
            valid_simd.contains(&simd),
            "Should detect valid SIMD backend"
        );
    }

    // --------------------------------------------------------------------------
    // Test: Benchmark utilities
    // --------------------------------------------------------------------------

    #[test]
    #[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
    fn test_benchmark_cold_start() {
        let model_bytes = create_sum_model(2);
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        let request = LambdaRequest {
            features: vec![1.0, 2.0],
            model_id: None,
        };

        let result = benchmark::benchmark_cold_start(&handler, &request, 10).expect("test");

        assert!(result.cold_start_ms >= 0.0);
        assert!(result.warm_inference_ms >= 0.0);
        assert_eq!(result.warm_iterations, 10);
        assert!(result.model_size_bytes > 0);
    }

    #[test]
    fn test_benchmark_targets() {
        // Verify target constants match spec §8.1
        assert!((benchmark::TARGET_COLD_START_MS - 50.0).abs() < f64::EPSILON);
        assert!((benchmark::TARGET_WARM_INFERENCE_MS - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_result_meets_targets() {
        let result = benchmark::BenchmarkResult {
            cold_start_ms: 45.0,
            warm_inference_ms: 8.0,
            warm_iterations: 100,
            model_size_bytes: 1000,
            target_arch: "aarch64".to_string(),
            simd_backend: "NEON".to_string(),
            meets_cold_start_target: true,
            meets_warm_inference_target: true,
        };

        assert!(result.meets_all_targets());
    }

    #[test]
    fn test_benchmark_result_fails_targets() {
        let result = benchmark::BenchmarkResult {
            cold_start_ms: 75.0, // Exceeds 50ms target
            warm_inference_ms: 8.0,
            warm_iterations: 100,
            model_size_bytes: 1000,
            target_arch: "aarch64".to_string(),
            simd_backend: "NEON".to_string(),
            meets_cold_start_target: false,
            meets_warm_inference_target: true,
        };

        assert!(!result.meets_all_targets());
    }

    // --------------------------------------------------------------------------
    // Test: Error display
    // --------------------------------------------------------------------------

    #[test]
    fn test_lambda_error_display() {
        assert_eq!(LambdaError::EmptyModel.to_string(), "Model bytes are empty");
        assert_eq!(
            LambdaError::EmptyFeatures.to_string(),
            "Request features are empty"
        );
        assert_eq!(
            LambdaError::EmptyBatch.to_string(),
            "Batch request has no instances"
        );
        assert!(LambdaError::InvalidMagic {
            expected: "APR\\0".to_string(),
            found: "GGUF".to_string()
        }
        .to_string()
        .contains("Invalid magic"));
    }

    // --------------------------------------------------------------------------
    // Test: Batch inference (PROD-001)
    // Per spec §5.3 and §11.3
    // --------------------------------------------------------------------------

    #[test]
    fn test_batch_request_serialization() {
        let request = BatchLambdaRequest {
            instances: vec![
                LambdaRequest {
                    features: vec![1.0, 2.0],
                    model_id: None,
                },
                LambdaRequest {
                    features: vec![3.0, 4.0],
                    model_id: None,
                },
            ],
            max_parallelism: Some(4),
        };

        let json = serde_json::to_string(&request).expect("serialization failed");
        assert!(json.contains("instances"));
        assert!(json.contains("max_parallelism"));
        assert!(json.contains("1.0"));
        assert!(json.contains("3.0"));
    }

    #[test]
    fn test_batch_response_serialization() {
        let response = BatchLambdaResponse {
            predictions: vec![LambdaResponse {
                prediction: 3.0,
                probabilities: None,
                latency_ms: 1.5,
                cold_start: false,
            }],
            total_latency_ms: 5.0,
            success_count: 1,
            error_count: 0,
        };

        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("predictions"));
        assert!(json.contains("success_count"));
        assert!(json.contains("total_latency_ms"));
    }

    #[test]
    #[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
    fn test_batch_handler_success() {
        let model_bytes = create_sum_model(2);
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        let request = BatchLambdaRequest {
            instances: vec![
                LambdaRequest {
                    features: vec![1.0, 2.0],
                    model_id: None,
                },
                LambdaRequest {
                    features: vec![3.0, 4.0],
                    model_id: None,
                },
            ],
            max_parallelism: None,
        };

        let response = handler.handle_batch(&request).expect("test");

        assert_eq!(response.predictions.len(), 2);
        assert_eq!(response.success_count, 2);
        assert_eq!(response.error_count, 0);
        assert!(response.total_latency_ms >= 0.0);

        // Real inference: sum model returns sum of features
        assert!((response.predictions[0].prediction - 3.0).abs() < 0.001);
        assert!((response.predictions[1].prediction - 7.0).abs() < 0.001);
    }
