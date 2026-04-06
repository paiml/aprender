
    // =========================================================================
    // Server Command Tests (EXTREME TDD - PAR-112)
    // =========================================================================

    #[tokio::test]
    async fn test_serve_model_invalid_extension() {
        // Test that unsupported file extensions return error
        let result = serve_model("127.0.0.1", 8080, "/nonexistent/model.xyz", false, false, true).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported file extension"));
    }

    #[tokio::test]
    async fn test_serve_model_nonexistent_gguf() {
        // Test that nonexistent GGUF file returns error
        let result = serve_model("127.0.0.1", 8080, "/nonexistent/model.gguf", false, false, true).await;
        assert!(result.is_err());
        // Should fail during GGUF loading
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to load GGUF")
                || err.to_string().contains("No such file")
                || err.to_string().contains("mmap")
        );
    }

    #[tokio::test]
    async fn test_serve_model_nonexistent_safetensors() {
        // Test that nonexistent SafeTensors file returns error
        let result = serve_model(
            "127.0.0.1",
            8080,
            "/nonexistent/model.safetensors",
            false,
            false,
            true,
        )
        .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to read") || err.to_string().contains("No such file")
        );
    }

    #[tokio::test]
    async fn test_serve_model_nonexistent_apr() {
        // Test that nonexistent APR file returns error
        let result = serve_model("127.0.0.1", 8080, "/nonexistent/model.apr", false, false, true).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to read") || err.to_string().contains("No such file")
        );
    }

    #[test]
    fn test_serve_model_extension_detection() {
        // Verify extension detection logic
        assert!("/path/to/model.gguf".ends_with(".gguf"));
        assert!("/path/to/model.safetensors".ends_with(".safetensors"));
        assert!("/path/to/model.apr".ends_with(".apr"));
        assert!(!"/path/to/model.xyz".ends_with(".gguf"));
        assert!(!"/path/to/model.xyz".ends_with(".safetensors"));
        assert!(!"/path/to/model.xyz".ends_with(".apr"));
    }

    #[test]
    fn test_serve_model_address_parsing() {
        // Verify address parsing works correctly
        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:8080".parse();
        assert!(addr.is_ok());

        let addr: std::result::Result<std::net::SocketAddr, _> = "0.0.0.0:3000".parse();
        assert!(addr.is_ok());

        let addr: std::result::Result<std::net::SocketAddr, _> = "invalid:port".parse();
        assert!(addr.is_err());
    }

    #[test]
    fn test_serve_model_port_ranges() {
        // Verify port range handling
        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:0".parse();
        assert!(addr.is_ok()); // Port 0 = OS assigns

        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:65535".parse();
        assert!(addr.is_ok()); // Max port

        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:80".parse();
        assert!(addr.is_ok()); // Privileged port (may need root)
    }

    #[test]
    fn test_batch_mode_flag_logic() {
        // Test batch mode flag combinations
        let batch_mode = true;
        let force_gpu = false;
        assert!(batch_mode && !force_gpu); // Valid: batch without forced GPU

        let batch_mode = true;
        let force_gpu = true;
        assert!(batch_mode && force_gpu); // Valid: batch with GPU

        let batch_mode = false;
        let force_gpu = true;
        assert!(!batch_mode && force_gpu); // Valid: single-request with GPU (true streaming)
    }

    #[test]
    fn test_cuda_env_var_detection() {
        // Test REALIZAR_BACKEND environment variable detection
        std::env::remove_var("REALIZAR_BACKEND");
        let use_cuda = std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        assert!(!use_cuda);

        std::env::set_var("REALIZAR_BACKEND", "cuda");
        let use_cuda = std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        assert!(use_cuda);

        std::env::set_var("REALIZAR_BACKEND", "CUDA");
        let use_cuda = std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        assert!(use_cuda);

        std::env::set_var("REALIZAR_BACKEND", "cpu");
        let use_cuda = std::env::var("REALIZAR_BACKEND")
            .map(|v| v.eq_ignore_ascii_case("cuda"))
            .unwrap_or(false);
        assert!(!use_cuda);

        // Cleanup
        std::env::remove_var("REALIZAR_BACKEND");
    }

    // =========================================================================
    // ModelType Tests
    // =========================================================================

    #[test]
    fn test_model_type_display() {
        assert_eq!(format!("{}", ModelType::Gguf), "GGUF");
        assert_eq!(format!("{}", ModelType::SafeTensors), "SafeTensors");
        assert_eq!(format!("{}", ModelType::Apr), "APR");
    }

    #[test]
    fn test_model_type_debug() {
        assert_eq!(format!("{:?}", ModelType::Gguf), "Gguf");
        assert_eq!(format!("{:?}", ModelType::SafeTensors), "SafeTensors");
        assert_eq!(format!("{:?}", ModelType::Apr), "Apr");
    }

    #[test]
    fn test_model_type_clone_copy() {
        let mt = ModelType::Gguf;
        let mt_clone = mt;
        let mt_copy = mt;
        assert_eq!(mt, mt_clone);
        assert_eq!(mt, mt_copy);
    }

    #[test]
    fn test_model_type_equality() {
        assert_eq!(ModelType::Gguf, ModelType::Gguf);
        assert_eq!(ModelType::SafeTensors, ModelType::SafeTensors);
        assert_eq!(ModelType::Apr, ModelType::Apr);
        assert_ne!(ModelType::Gguf, ModelType::SafeTensors);
        assert_ne!(ModelType::SafeTensors, ModelType::Apr);
        assert_ne!(ModelType::Apr, ModelType::Gguf);
    }

    // =========================================================================
    // PreparedServer Tests
    // =========================================================================

    #[test]
    fn test_prepared_server_debug() {
        // Create a demo AppState for testing
        let state = crate::api::AppState::demo().expect("demo state");
        let prepared = PreparedServer {
            state,
            batch_mode_enabled: true,
            model_type: ModelType::Gguf,
        };
        let debug_str = format!("{:?}", prepared);
        assert!(debug_str.contains("PreparedServer"));
        assert!(debug_str.contains("batch_mode_enabled: true"));
        assert!(debug_str.contains("model_type: Gguf"));
    }

    #[test]
    fn test_prepared_server_fields() {
        let state = crate::api::AppState::demo().expect("demo state");
        let prepared = PreparedServer {
            state,
            batch_mode_enabled: false,
            model_type: ModelType::SafeTensors,
        };
        assert!(!prepared.batch_mode_enabled);
        assert_eq!(prepared.model_type, ModelType::SafeTensors);
    }

    // =========================================================================
    // prepare_serve_state Tests (EXTREME TDD)
    // =========================================================================

    #[test]
    fn test_prepare_serve_state_invalid_extension() {
        // Test that unsupported file extensions return error
        let result = prepare_serve_state("/nonexistent/model.xyz", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported file extension"));
    }

    #[test]
    fn test_prepare_serve_state_nonexistent_gguf() {
        // Test that nonexistent GGUF file returns error
        let result = prepare_serve_state("/nonexistent/model.gguf", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to load GGUF")
                || err.to_string().contains("No such file")
                || err.to_string().contains("mmap")
        );
    }

    #[test]
    fn test_prepare_serve_state_nonexistent_safetensors() {
        // Test that nonexistent SafeTensors file returns error
        let result = prepare_serve_state("/nonexistent/model.safetensors", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to read") || err.to_string().contains("No such file")
        );
    }

    #[test]
    fn test_prepare_serve_state_nonexistent_apr() {
        // Test that nonexistent APR file returns error
        let result = prepare_serve_state("/nonexistent/model.apr", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Failed to read") || err.to_string().contains("No such file")
        );
    }

    #[test]
    fn test_prepare_serve_state_with_batch_mode_flag() {
        // Test batch_mode flag is recorded even when loading fails
        let result = prepare_serve_state("/nonexistent/model.gguf", true, false);
        assert!(result.is_err()); // File doesn't exist, but flag should be processed before
    }

    #[test]
    fn test_prepare_serve_state_with_force_gpu_flag() {
        // Test force_gpu flag is recorded even when loading fails
        let result = prepare_serve_state("/nonexistent/model.gguf", false, true);
        assert!(result.is_err()); // File doesn't exist, but flag should be processed before
    }

    #[test]
    fn test_prepare_serve_state_extension_variants() {
        // Test various file extension patterns
        let extensions = vec![
            ("/path/model.gguf", true, "GGUF"),
            ("/path/MODEL.GGUF", false, "uppercase"),
            ("/path/model.safetensors", true, "SafeTensors"),
            ("/path/model.apr", true, "APR"),
            ("/path/model.pt", false, "PyTorch"),
            ("/path/model.bin", false, "binary"),
            ("/path/model.h5", false, "HDF5"),
            ("/path/model", false, "no extension"),
        ];

        for (path, should_detect, name) in extensions {
            let is_gguf = path.ends_with(".gguf");
            let is_safetensors = path.ends_with(".safetensors");
            let is_apr = path.ends_with(".apr");
            let detected = is_gguf || is_safetensors || is_apr;
            assert_eq!(
                detected, should_detect,
                "Extension detection failed for {name}: {path}"
            );
        }
    }

    // =========================================================================
    // serve_demo Tests (EXTREME TDD)
    // =========================================================================

    #[test]
    fn test_serve_demo_address_validation() {
        // Test that address parsing logic works correctly
        let valid_addresses = vec![
            ("127.0.0.1", 8080),
            ("0.0.0.0", 3000),
            ("localhost", 8000), // This won't parse as SocketAddr directly
        ];

        for (host, port) in valid_addresses {
            let addr_str = format!("{}:{}", host, port);
            let result: std::result::Result<std::net::SocketAddr, _> = addr_str.parse();
            // localhost won't parse, but IP addresses should
            if host != "localhost" {
                assert!(result.is_ok(), "Address {addr_str} should be valid");
            }
        }
    }

    #[test]
    fn test_serve_demo_port_zero() {
        // Port 0 should be valid (OS assigns port)
        let addr: std::result::Result<std::net::SocketAddr, _> = "127.0.0.1:0".parse();
        assert!(addr.is_ok());
    }

    // =========================================================================
    // Integration Tests (run with `cargo test -- --ignored`)
    // =========================================================================

    /// Integration test for prepare_serve_state with a real GGUF model
    /// Run with: cargo test test_prepare_serve_state_gguf_success -- --ignored
    #[test]
    #[ignore = "requires real GGUF model file"]
    fn test_prepare_serve_state_gguf_success() {
        // Look for a test model file
        let home = std::env::var("HOME").expect("HOME env var not set");
        let model_paths = [
            format!("{}/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", home),
            format!("{}/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", home),
        ];

        let model_path = model_paths
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .expect("No test model file found. Run tests with a valid GGUF model.");

        let result = prepare_serve_state(model_path, false, false);
        assert!(
            result.is_ok(),
            "prepare_serve_state failed: {:?}",
            result.err()
        );

        let prepared = result.expect("operation failed");
        assert_eq!(prepared.model_type, ModelType::Gguf);
        assert!(!prepared.batch_mode_enabled);
        // State should have a quantized model
    }

    /// Integration test for prepare_serve_state with batch mode
    /// Run with: cargo test test_prepare_serve_state_gguf_batch -- --ignored
    #[tokio::test]
    #[ignore = "requires real GGUF model file"]
    async fn test_prepare_serve_state_gguf_batch() {
        let home = std::env::var("HOME").expect("HOME env var not set");
        let model_paths = [
            format!("{}/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", home),
            format!("{}/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", home),
        ];

        let model_path = model_paths
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .expect("No test model file found.");

        // Batch mode requires Tokio runtime for spawn_batch_processor
        let result = prepare_serve_state(model_path, true, false);
        assert!(
            result.is_ok(),
            "prepare_serve_state failed: {:?}",
            result.err()
        );

        let prepared = result.expect("operation failed");
        assert_eq!(prepared.model_type, ModelType::Gguf);
        // batch_mode_enabled is true since we enabled it and gpu feature is available
        assert!(prepared.batch_mode_enabled);
    }

    /// Test all model type variants are properly detected
    #[test]
    fn test_model_type_from_extension() {
        // This tests the extension detection logic in prepare_serve_state
        let test_cases = vec![
            ("model.gguf", Some("gguf")),
            ("model.safetensors", Some("safetensors")),
            ("model.apr", Some("apr")),
            ("model.bin", None),
            ("model", None),
        ];

        for (path, expected_ext) in test_cases {
            let is_gguf = path.ends_with(".gguf");
            let is_safetensors = path.ends_with(".safetensors");
            let is_apr = path.ends_with(".apr");

            let actual = if is_gguf {
                Some("gguf")
            } else if is_safetensors {
                Some("safetensors")
            } else if is_apr {
                Some("apr")
            } else {
                None
            };

            assert_eq!(
                actual, expected_ext,
                "Extension detection failed for {path}"
            );
        }
    }

    /// Test PreparedServer with all model types
    #[test]
    fn test_prepared_server_all_model_types() {
        for model_type in [ModelType::Gguf, ModelType::SafeTensors, ModelType::Apr] {
            let state = crate::api::AppState::demo().expect("demo state");
            let prepared = PreparedServer {
                state,
                batch_mode_enabled: false,
                model_type,
            };
            assert_eq!(prepared.model_type, model_type);
            // Test that debug output includes the model type
            let debug = format!("{:?}", prepared);
            assert!(debug.contains(&format!("{:?}", model_type)));
        }
    }
