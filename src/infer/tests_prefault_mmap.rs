//! Additional coverage tests for infer module (Part 3)
//!
//! These tests focus on:
//! - with_trace_output builder method
//! - Architecture detection from model filenames
//! - Edge cases for tok_per_sec calculation
//! - Format detection boundary cases
//! - InferenceConfig field combinations
//! - Error message formatting

#[cfg(test)]
mod tests {
    use crate::infer::*;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[test]
    fn test_prefault_mmap_just_under_page() {
        let data = vec![0xCDu8; 4095];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_just_over_page() {
        let data = vec![0xEFu8; 4097];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_very_large() {
        // Test with 100 pages (400KB)
        let data = vec![0x12u8; 4096 * 100];
        prefault_mmap(&data);
    }

    #[test]
    fn test_prefault_mmap_wrapping_checksum() {
        // Data that will cause checksum wrapping
        let data = vec![0xFFu8; 4096 * 256 + 1]; // Enough to wrap around u8
        prefault_mmap(&data);
    }

    // =========================================================================
    // Model Path Edge Cases
    // =========================================================================

    #[test]
    fn test_model_path_no_extension() {
        let config = InferenceConfig::new("/path/to/model");
        assert_eq!(config.model_path, PathBuf::from("/path/to/model"));
    }

    #[test]
    fn test_model_path_double_extension() {
        let config = InferenceConfig::new("/path/model.tar.gz");
        assert_eq!(config.model_path, PathBuf::from("/path/model.tar.gz"));
    }

    #[test]
    fn test_model_path_hidden_file() {
        let config = InferenceConfig::new("/path/.hidden_model.gguf");
        assert!(config
            .model_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with('.'));
    }

    #[test]
    fn test_model_path_only_extension() {
        let config = InferenceConfig::new("/path/.gguf");
        assert_eq!(config.model_path, PathBuf::from("/path/.gguf"));
    }

    // =========================================================================
    // File Size Boundary Tests
    // =========================================================================

    #[test]
    fn test_file_exactly_8_bytes() {
        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        temp.write_all(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            .expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        // Should not fail with "too small" error
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(!err_msg.contains("too small"));
    }

    #[test]
    fn test_file_7_bytes() {
        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        temp.write_all(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            .expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("too small"));
    }

    // =========================================================================
    // InferenceResult Clone and Debug Tests
    // =========================================================================

    #[test]
    fn test_result_clone_preserves_all_fields() {
        let original = InferenceResult {
            text: "original text".to_string(),
            tokens: vec![1, 2, 3, 4, 5],
            input_token_count: 2,
            generated_token_count: 3,
            inference_ms: 123.456,
            tok_per_sec: 24.35,
            load_ms: 789.012,
            format: "SafeTensors".to_string(),
            used_gpu: true,
        };

        let cloned = original.clone();

        assert_eq!(original.text, cloned.text);
        assert_eq!(original.tokens, cloned.tokens);
        assert_eq!(original.input_token_count, cloned.input_token_count);
        assert_eq!(original.generated_token_count, cloned.generated_token_count);
        assert!((original.inference_ms - cloned.inference_ms).abs() < f64::EPSILON);
        assert!((original.tok_per_sec - cloned.tok_per_sec).abs() < f64::EPSILON);
        assert!((original.load_ms - cloned.load_ms).abs() < f64::EPSILON);
        assert_eq!(original.format, cloned.format);
        assert_eq!(original.used_gpu, cloned.used_gpu);
    }

    #[test]
    fn test_result_debug_contains_all_field_names() {
        let result = InferenceResult {
            text: "t".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };

        let debug_str = format!("{:?}", result);

        assert!(debug_str.contains("text"));
        assert!(debug_str.contains("tokens"));
        assert!(debug_str.contains("input_token_count"));
        assert!(debug_str.contains("generated_token_count"));
        assert!(debug_str.contains("inference_ms"));
        assert!(debug_str.contains("tok_per_sec"));
        assert!(debug_str.contains("load_ms"));
        assert!(debug_str.contains("format"));
        assert!(debug_str.contains("used_gpu"));
    }
include!("tests_trace.rs");
}
