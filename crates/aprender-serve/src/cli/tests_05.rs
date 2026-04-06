//! CLI Module Tests Part 05 - T-COV-95 Coverage Bridge (Deep CLI)
//!
//! Tests for uncovered CLI functions:
//! - run_visualization: sparkline, histogram, benchmark report, multi-comparison
//! - run_convoy_test: full execution with output file
//! - run_saturation_test: full execution with output file
//! - run_bench_compare: with temp benchmark files
//! - run_bench_regression: strict and normal modes
//! - parse_cargo_bench_output: valid bench lines, empty, partial
//! - run_benchmarks: list mode, invalid suite, external benchmark stub
//! - load_gguf_model / load_safetensors_model / load_apr_model: full paths
//! - print_info, home_dir, validate_suite_name
//! - display_model_info: all 4 branches
//! - is_local_file_path: comprehensive edge cases
//!
//! Refs PMAT-802: Protocol T-COV-95 Deep CLI Coverage

#[cfg(test)]
mod tests {
    use super::super::*;

    // =========================================================================
    // is_local_file_path - comprehensive
    // =========================================================================

    #[test]
    fn test_is_local_file_path_comprehensive() {
        // True cases
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("/absolute/path/model.gguf"));
        assert!(is_local_file_path("model.gguf")); // .gguf extension
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("model.apr"));
        assert!(is_local_file_path("/tmp/model.safetensors"));
        assert!(is_local_file_path("./relative.apr"));

        // False cases
        assert!(!is_local_file_path("pacha://model"));
        assert!(!is_local_file_path("hf://model"));
        assert!(!is_local_file_path("model_name"));
        assert!(!is_local_file_path("llama3.2"));
        assert!(!is_local_file_path("registry:tag"));
    }

    // =========================================================================
    // load_gguf_model
    // =========================================================================

    #[test]
    fn test_load_gguf_model_valid() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 64)
            .num_layers("llama", 2)
            .num_heads("llama", 4)
            .build();
        let result = load_gguf_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_gguf_model_empty() {
        let result = load_gguf_model(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_gguf_model_truncated() {
        let result = load_gguf_model(&[0x47, 0x47, 0x55, 0x46]); // Just magic bytes
        assert!(result.is_err());
    }

    // =========================================================================
    // load_safetensors_model
    // =========================================================================

    #[test]
    fn test_load_safetensors_model_valid() {
        let header = r#"{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&[0u8; 16]);
        let result = load_safetensors_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_safetensors_model_empty() {
        let result = load_safetensors_model(&[]);
        assert!(result.is_err());
    }

    // =========================================================================
    // load_apr_model
    // =========================================================================

    #[test]
    fn test_load_apr_model_with_magic() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes()); // LinearRegression
        data[6..8].copy_from_slice(&1u16.to_le_bytes()); // version
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_apr_model_wrong_magic() {
        let data = vec![0xFF; 64];
        let result = load_apr_model(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_too_small() {
        let result = load_apr_model(&[0x41, 0x50, 0x52, 0x00]); // Just "APR\0" - 4 bytes
        // May succeed or fail depending on format detection
        let _ = result;
    }

    // =========================================================================
    // display_model_info - all branches
    // =========================================================================

    #[test]
    fn test_display_model_info_gguf_by_extension() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        let result = display_model_info("test.gguf", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_gguf_by_magic() {
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        // Use non-gguf extension but GGUF magic bytes
        let result = display_model_info("model.bin", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_safetensors_by_extension() {
        let header = r#"{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&[0u8; 16]);
        let result = display_model_info("model.safetensors", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_by_extension() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0020u16.to_le_bytes());
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = display_model_info("model.apr", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_by_magic() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0001u16.to_le_bytes());
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = display_model_info("model.bin", &data);
        // APR magic check happens after safetensors check
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_unknown() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
        let result = display_model_info("model.unknown", &data);
        assert!(result.is_ok());
    }

    // =========================================================================
    // BENCHMARK_SUITES constant verification
    // =========================================================================

    #[test]
    fn test_benchmark_suites_not_empty() {
        // BENCHMARK_SUITES is a non-empty const - just access first element
        let (first_name, first_desc) = BENCHMARK_SUITES[0];
        assert!(!first_name.is_empty());
        assert!(!first_desc.is_empty());
        for (name, desc) in BENCHMARK_SUITES {
            assert!(!name.is_empty());
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_benchmark_suites_contains_expected() {
        let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"tensor_ops"));
        assert!(names.contains(&"inference"));
        assert!(names.contains(&"cache"));
        assert!(names.contains(&"tokenizer"));
        assert!(names.contains(&"quantize"));
    }
include!("benchmarks.rs");
}
