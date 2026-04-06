
// =========================================================================
// CUDA Feature Tests (compiled only when cuda feature is enabled)
// =========================================================================

#[cfg(all(test, feature = "cuda"))]
mod cuda_inference_tests {
    use crate::cli::inference;
    use crate::inference_trace::TraceConfig;

    #[test]
    fn test_run_gguf_inference_gpu_function_exists() {
        // Verify the GPU function is available when cuda feature is enabled
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "cuda test",
            5,
            0.5,
            "text",
            true, // force_gpu - will try to use GPU path
            true, // verbose - shows GPU status
            None, // trace_config
        );
        assert!(result.is_err()); // File doesn't exist, but GPU path is attempted
    }

    #[test]
    fn test_run_apr_inference_gpu_function_exists() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "cuda apr test",
            5,
            0.5,
            "text",
            true, // force_gpu
            true, // verbose
            None, // trace_config
        );
        assert!(result.is_err());
    }
}
