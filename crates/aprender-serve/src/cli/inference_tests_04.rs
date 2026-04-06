//! T-COV-95 Artifact Falsification: CLI Inference via Temp Files
//!
//! Dr. Popper's directive: "Do not mock the file; *become* the file."
//!
//! These tests write 1KB Active Pygmies to disk as actual .gguf and .apr files,
//! then invoke the real CLI inference functions to exercise:
//! - File discovery and mmap loading
//! - Argument parsing and validation
//! - Error handling for filesystem operations
//! - Full inference pipeline from CLI entry point

#[cfg(test)]
mod artifact_falsification {
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    use crate::apr::test_factory::build_executable_pygmy_apr;
    use crate::cli::inference;
    use crate::gguf::test_factory::build_executable_pygmy_gguf;
    use crate::inference_trace::TraceConfig;

    #[test]
    fn test_load_gguf_model_truncated() {
        // Truncated GGUF (just magic)
        let truncated = b"GGUF\x03\x00\x00\x00";
        let result = crate::cli::load_gguf_model(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_truncated() {
        // APR magic but truncated
        let truncated = b"APR\x00\x02\x00";
        let result = crate::cli::load_apr_model(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_local_file_path_absolute() {
        assert!(crate::cli::is_local_file_path("/home/user/model.gguf"));
        assert!(crate::cli::is_local_file_path("/tmp/model.apr"));
    }

    #[test]
    fn test_is_local_file_path_relative() {
        assert!(crate::cli::is_local_file_path("./model.gguf"));
        assert!(crate::cli::is_local_file_path("../models/model.apr"));
    }

    #[test]
    fn test_is_local_file_path_registry_uri() {
        assert!(!crate::cli::is_local_file_path("hf://model/name"));
        assert!(!crate::cli::is_local_file_path("ollama://llama2"));
    }

    #[test]
    fn test_validate_suite_name_valid() {
        // These are the actual benchmark suite names
        assert!(crate::cli::validate_suite_name("tensor_ops"));
        assert!(crate::cli::validate_suite_name("inference"));
        assert!(crate::cli::validate_suite_name("cache"));
        assert!(crate::cli::validate_suite_name("tokenizer"));
        assert!(crate::cli::validate_suite_name("quantize"));
    }

    #[test]
    fn test_validate_suite_name_invalid() {
        assert!(!crate::cli::validate_suite_name(""));
        assert!(!crate::cli::validate_suite_name("nonexistent_suite"));
        assert!(!crate::cli::validate_suite_name("random"));
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(crate::cli::format_size(0), "0 B");
        assert_eq!(crate::cli::format_size(512), "512 B");
        assert_eq!(crate::cli::format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(crate::cli::format_size(1024), "1.0 KB");
        assert_eq!(crate::cli::format_size(2048), "2.0 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(crate::cli::format_size(1024 * 1024), "1.0 MB");
        assert_eq!(crate::cli::format_size(10 * 1024 * 1024), "10.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(crate::cli::format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    // =========================================================================
    // GGUF inference edge cases - more format/option combinations
    // =========================================================================

    #[test]
    fn test_gguf_artifact_show_probs() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().expect("to_str conversion");

        let result = inference::run_gguf_inference(
            path,
            &[],
            "Test",
            3,
            0.0,
            "text",
            false,
            true,
            None, // trace_config (no tracing for this test)
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_gguf_artifact_debug_mode() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().expect("to_str conversion");

        let result = inference::run_gguf_inference(
            path,
            &[],
            "Debug test",
            2,
            0.0,
            "text",
            false,
            false,
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_gguf_artifact_all_options() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().expect("to_str conversion");

        let result = inference::run_gguf_inference(
            path,
            &[],
            "All options",
            3,
            0.5,
            "json",
            false,                        // force_gpu (pygmy model can't run on GPU)
            true,                         // verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_ok());
    }
include!("inference_tests.rs");
}
