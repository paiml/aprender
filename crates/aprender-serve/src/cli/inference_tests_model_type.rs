
// =========================================================================
// Server Commands Tests (gated by "server" feature)
// =========================================================================
#[cfg(all(test, feature = "server"))]
mod server_inference_tests {
    use crate::cli::ModelType;

    #[test]
    fn test_model_type_display_gguf() {
        let mt = ModelType::Gguf;
        assert_eq!(format!("{}", mt), "GGUF");
    }

    #[test]
    fn test_model_type_display_safetensors() {
        let mt = ModelType::SafeTensors;
        assert_eq!(format!("{}", mt), "SafeTensors");
    }

    #[test]
    fn test_model_type_display_apr() {
        let mt = ModelType::Apr;
        assert_eq!(format!("{}", mt), "APR");
    }

    #[test]
    fn test_model_type_equality() {
        assert_eq!(ModelType::Gguf, ModelType::Gguf);
        assert_ne!(ModelType::Gguf, ModelType::Apr);
        assert_ne!(ModelType::SafeTensors, ModelType::Apr);
    }

    #[test]
    fn test_model_type_clone() {
        let mt = ModelType::Gguf;
        let cloned = mt;
        assert_eq!(mt, cloned);
    }

    #[test]
    fn test_model_type_copy() {
        let mt = ModelType::Apr;
        let copied: ModelType = mt; // Copy trait
        assert_eq!(mt, copied);
    }

    #[test]
    fn test_model_type_debug() {
        let mt = ModelType::SafeTensors;
        let debug_str = format!("{:?}", mt);
        assert_eq!(debug_str, "SafeTensors");
    }

    #[test]
    fn test_prepare_serve_state_invalid_gguf() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/nonexistent/model.gguf", false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_serve_state_invalid_safetensors() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/nonexistent/model.safetensors", false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_serve_state_invalid_apr() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/nonexistent/model.apr", false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_serve_state_unsupported_extension() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/some/model.bin", false, false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported file extension"));
    }

    #[test]
    fn test_prepare_serve_state_with_batch_mode_flag() {
        use crate::cli::prepare_serve_state;

        // batch_mode flag should be processed even if model doesn't exist
        let result = prepare_serve_state("/nonexistent/model.gguf", true, false);
        assert!(result.is_err()); // Still fails, but batch_mode path exercised
    }

    #[test]
    fn test_prepare_serve_state_with_gpu_flag() {
        use crate::cli::prepare_serve_state;

        // force_gpu flag should be processed even if model doesn't exist
        let result = prepare_serve_state("/nonexistent/model.gguf", false, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_serve_state_with_both_flags() {
        use crate::cli::prepare_serve_state;

        let result = prepare_serve_state("/nonexistent/model.gguf", true, true);
        assert!(result.is_err());
    }
}

// =========================================================================
// CUDA Feature Tests
// =========================================================================
#[cfg(all(test, feature = "cuda"))]
mod cuda_inference_additional_tests {
    use crate::cli::inference;
    use crate::inference_trace::TraceConfig;

    #[test]
    fn test_gguf_inference_gpu_path_exists() {
        // When cuda feature is enabled, force_gpu should attempt GPU path
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "cuda path test",
            5,
            0.5,
            "text",
            true, // force_gpu - exercises GPU code path
            true, // verbose
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_gpu_path_exists() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "cuda apr test",
            5,
            0.5,
            "text",
            true,
            true,
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_gpu_json_format() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "cuda json test",
            5,
            0.5,
            "json",
            true, // force_gpu
            true, // verbose
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_gpu_json_format() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "cuda json test",
            5,
            0.5,
            "json",
            true,
            true,
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }
}
