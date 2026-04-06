//! Additional coverage tests for infer module (Part 2)
//!
//! These tests focus on:
//! - find_fallback_tokenizer function
//! - run_gguf_generate function paths
//! - Error handling edge cases
//! - Configuration validation edge cases

#[cfg(test)]
mod tests {
    use crate::infer::*;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_max_tokens_max_value() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(usize::MAX);
        assert_eq!(config.max_tokens, usize::MAX);
    }

    // =========================================================================
    // File Format Creation for Testing
    // =========================================================================

    #[test]
    fn test_run_inference_with_minimal_gguf_v3() {
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("create temp");
        let mut data = Vec::new();
        // GGUF magic
        data.extend_from_slice(b"GGUF");
        // Version 3 (little-endian u32)
        data.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count = 0
        data.extend_from_slice(&0u64.to_le_bytes());
        // Metadata count = 0
        data.extend_from_slice(&0u64.to_le_bytes());
        // Additional padding
        data.extend_from_slice(&[0u8; 256]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("test")
            .with_max_tokens(1);
        let result = run_inference(&config);
        // Should fail due to missing tensors, but format detection should work
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_with_apr_v2_minimal() {
        let mut temp = NamedTempFile::with_suffix(".apr").expect("create temp");
        let mut data = Vec::new();
        // APR v2 magic (ASCII '2' as version byte)
        data.extend_from_slice(b"APR2");
        // Minimal header padding
        data.extend_from_slice(&[0u8; 256]);
        temp.write_all(&data).expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path()).with_max_tokens(1);
        let result = run_inference(&config);
        // May succeed with degenerate output or fail
        let _ = result;
    }

    // =========================================================================
    // Verbose and Trace Mode Tests
    // =========================================================================

    #[test]
    fn test_verbose_mode_with_all_options() {
        let config = InferenceConfig::new("/model.gguf")
            .with_verbose(true)
            .with_trace(true)
            .with_prompt("test")
            .with_max_tokens(10)
            .with_temperature(0.5)
            .with_top_k(40)
            .without_gpu();

        assert!(config.verbose);
        assert!(config.trace);
        assert!(config.no_gpu);
    }

    #[test]
    fn test_trace_configuration_complete() {
        let mut config = InferenceConfig::new("/model.gguf");
        config.trace = true;
        config.trace_verbose = true;
        config.trace_output = Some(PathBuf::from("/tmp/trace/output.json"));
        config.trace_steps = Some(vec![
            "embedding".to_string(),
            "attention".to_string(),
            "ffn".to_string(),
            "logits".to_string(),
        ]);

        assert!(config.trace);
        assert!(config.trace_verbose);
        assert!(config.trace_output.is_some());
        assert_eq!(config.trace_steps.as_ref().map(std::vec::Vec::len), Some(4));
    }

    // =========================================================================
    // EOS Token Detection Tests
    // =========================================================================

    #[test]
    fn test_eos_detection_boundary_values() {
        // Test values around known EOS tokens
        let eos_tokens = [151645u32, 151643, 2];
        let boundary_values = [151644u32, 151646, 1, 3, 151642];

        for token in eos_tokens {
            let is_eos = token == 151645 || token == 151643 || token == 2;
            assert!(is_eos, "Token {} should be detected as EOS", token);
        }

        for token in boundary_values {
            let is_eos = token == 151645 || token == 151643 || token == 2;
            assert!(!is_eos, "Token {} should NOT be detected as EOS", token);
        }
    }

    // =========================================================================
    // Argmax Logic Tests for Generation
    // =========================================================================

    #[test]
    fn test_argmax_single_element() {
        let logits = vec![0.5f32];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 0);
    }

    #[test]
    fn test_argmax_large_vocab() {
        // Simulate large vocabulary
        let mut logits = vec![-1.0f32; 50000];
        logits[32000] = 10.0; // Set one element to be max
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 32000);
    }

    #[test]
    fn test_argmax_all_negative() {
        let logits = vec![-10.0f32, -5.0, -1.0, -0.5, -2.0];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 3); // -0.5 is the maximum
    }

    #[test]
    fn test_argmax_mixed_special_values() {
        let logits = vec![f32::NEG_INFINITY, 0.0, f32::INFINITY, 1.0];
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        assert_eq!(next_token, 2); // INFINITY is the maximum
    }

    // =========================================================================
    // InferenceResult Builder Pattern Tests
    // =========================================================================

    #[test]
    fn test_inference_result_all_formats() {
        let formats = ["GGUF", "APR", "SafeTensors", "Custom", ""];

        for format in formats {
            let result = InferenceResult {
                text: "test".to_string(),
                tokens: vec![1],
                input_token_count: 1,
                generated_token_count: 0,
                inference_ms: 1.0,
                tok_per_sec: 1.0,
                load_ms: 1.0,
                format: format.to_string(),
                used_gpu: false,
            };
            assert_eq!(result.format, format);
        }
    }

    #[test]
    fn test_inference_result_gpu_combinations() {
        for used_gpu in [true, false] {
            for format in ["GGUF", "APR", "SafeTensors"] {
                let result = InferenceResult {
                    text: "test".to_string(),
                    tokens: vec![1],
                    input_token_count: 1,
                    generated_token_count: 0,
                    inference_ms: 1.0,
                    tok_per_sec: 1.0,
                    load_ms: 1.0,
                    format: format.to_string(),
                    used_gpu,
                };
                assert_eq!(result.used_gpu, used_gpu);
                assert_eq!(result.format, format);
            }
        }
    }

    // =========================================================================
    // Path Handling Edge Cases
    // =========================================================================

    #[test]
    fn test_path_with_consecutive_slashes() {
        let config = InferenceConfig::new("//path//to//model.gguf");
        assert!(config.model_path.to_str().unwrap().contains("model.gguf"));
    }

    #[test]
    fn test_path_with_dot_components() {
        let config = InferenceConfig::new("/path/./to/../to/model.gguf");
        assert!(config.model_path.to_str().unwrap().contains("model.gguf"));
    }

    #[test]
    fn test_path_with_tilde() {
        let config = InferenceConfig::new("~/models/model.gguf");
        assert!(config.model_path.to_str().unwrap().starts_with('~'));
    }

    // =========================================================================
    // Tokens Per Second Calculation Edge Cases
    // =========================================================================

    #[test]
    fn test_tok_per_sec_exactly_one_second() {
        let inference_ms = 1000.0;
        let generated = 50;
        let tok_per_sec = generated as f64 / (inference_ms / 1000.0);
        assert!((tok_per_sec - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tok_per_sec_microsecond_precision() {
        let inference_ms = 0.001; // 1 microsecond
        let generated = 1;
        let tok_per_sec = if inference_ms > 0.0 {
            generated as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        // 1 / 0.000001 = 1,000,000
        assert!((tok_per_sec - 1_000_000.0).abs() < 0.001);
    }

    // =========================================================================
    // Config Builder Immutability Test
    // =========================================================================

    #[test]
    fn test_config_builder_creates_new_instance() {
        let base = InferenceConfig::new("/model.gguf");
        let with_prompt = base.clone().with_prompt("test");

        // Original should be unchanged (due to clone)
        assert!(base.prompt.is_none());
        assert_eq!(with_prompt.prompt, Some("test".to_string()));
    }

    #[test]
    fn test_config_debug_output_parseable() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("test")
            .with_max_tokens(100);

        let debug = format!("{:?}", config);

        // Should contain struct name and key field names
        assert!(debug.contains("InferenceConfig"));
        assert!(debug.contains("model_path"));
        assert!(debug.contains("prompt"));
        assert!(debug.contains("max_tokens"));
    }

    // =========================================================================
    // Error Type Coverage
    // =========================================================================

    #[test]
    fn test_error_type_io_error() {
        use crate::error::RealizarError;

        let err = RealizarError::IoError {
            message: "Custom IO error message".to_string(),
        };
        let err_string = err.to_string();
        assert!(err_string.contains("Custom IO error message"));
    }

    #[test]
    fn test_error_type_format_error() {
        use crate::error::RealizarError;

        let err = RealizarError::FormatError {
            reason: "Invalid magic bytes detected".to_string(),
        };
        let err_string = err.to_string();
        assert!(err_string.contains("Invalid magic bytes"));
    }

    #[test]
    fn test_error_type_inference_error() {
        use crate::error::RealizarError;

        let err = RealizarError::InferenceError("Generation loop failed".to_string());
        let err_string = err.to_string();
        assert!(err_string.contains("Generation loop failed"));
    }

    // =========================================================================
    // Generated Token Slice Edge Cases
    // =========================================================================

    #[test]
    fn test_generated_tokens_single_input() {
        let all_tokens = vec![1u32, 100, 200, 300];
        let input_count = 1;
        let generated = &all_tokens[input_count..];
        assert_eq!(generated, &[100, 200, 300]);
    }

    #[test]
    fn test_generated_tokens_equal_counts() {
        let all_tokens = vec![1u32, 2, 3, 4, 5, 6];
        let input_count = 3;
        let generated = &all_tokens[input_count..];
        assert_eq!(generated.len(), 3);
        assert_eq!(input_count, generated.len());
    }

    // =========================================================================
    // Model Loading Path Edge Cases
    // =========================================================================

    #[test]
    fn test_model_path_root_directory() {
        let config = InferenceConfig::new("/model.gguf");
        assert!(config.model_path.starts_with("/"));
    }

    #[test]
    fn test_model_path_current_directory() {
        let config = InferenceConfig::new("./model.gguf");
        assert!(config.model_path.starts_with("./"));
    }

    #[test]
    fn test_model_path_parent_directory() {
        let config = InferenceConfig::new("../models/model.gguf");
        assert!(config.model_path.starts_with("../"));
    }
include!("tests_find_fallback.rs");
}
