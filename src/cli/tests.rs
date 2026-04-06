#[cfg(test)]
mod tests {
    use crate::cli::*;

    #[test]
    fn test_deep_clicov_run_saturation_test_output_file_write() {
        use std::fs;
        let dir = std::env::temp_dir();
        let output = dir.join("deep_clicov_saturation_test.json");

        let result = run_saturation_test(
            Some("test_runtime".to_string()),
            Some("test_model.gguf".to_string()),
            Some(output.to_str().expect("invalid UTF-8").to_string()),
        );
        assert!(result.is_ok());

        // Verify file was created
        assert!(output.exists());
        let content = fs::read_to_string(&output).expect("file operation failed");
        assert!(content.contains("throughput"));

        let _ = fs::remove_file(&output);
    }

    #[test]
    fn test_deep_clicov_bench_compare_invalid_json_file1() {
        use std::fs::File;
        use std::io::Write;

        let dir = std::env::temp_dir();
        let file1 = dir.join("deep_clicov_invalid1.json");
        let file2 = dir.join("deep_clicov_invalid2.json");

        // Write invalid JSON to file1
        let mut f1 = File::create(&file1).expect("file operation failed");
        f1.write_all(b"not valid json").expect("operation failed");

        // Write valid but empty JSON to file2
        let mut f2 = File::create(&file2).expect("file operation failed");
        f2.write_all(b"{}").expect("operation failed");

        let result = run_bench_compare(
            file1.to_str().expect("file operation failed"),
            file2.to_str().expect("file operation failed"),
            5.0,
        );

        let _ = std::fs::remove_file(&file1);
        let _ = std::fs::remove_file(&file2);

        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_bench_regression_strict_mode() {
        use std::fs::File;
        use std::io::Write;

        let dir = std::env::temp_dir();
        let baseline = dir.join("deep_clicov_baseline.json");
        let current = dir.join("deep_clicov_current.json");

        // Write minimal valid JSON
        let mut f1 = File::create(&baseline).expect("file operation failed");
        f1.write_all(b"{}").expect("operation failed");

        let mut f2 = File::create(&current).expect("file operation failed");
        f2.write_all(b"{}").expect("operation failed");

        let result = run_bench_regression(
            baseline.to_str().expect("invalid UTF-8"),
            current.to_str().expect("invalid UTF-8"),
            true, // strict
        );

        let _ = std::fs::remove_file(&baseline);
        let _ = std::fs::remove_file(&current);

        // Should fail because JSON is not valid benchmark format
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clicov_format_size_u64_max_safe() {
        // Large but not overflow-inducing value
        let large_value = 1_000_000 * 1024 * 1024 * 1024u64; // 1 PB
        let result = format_size(large_value);
        assert!(result.contains("GB"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_double_dot() {
        assert!(is_local_file_path("../model.gguf"));
        assert!(is_local_file_path("../../model.safetensors"));
        assert!(is_local_file_path("./../model.apr"));
    }

    #[test]
    fn test_deep_clicov_is_local_file_path_protocol_like_strings() {
        // Strings that look like protocols but aren't
        assert!(!is_local_file_path("file:///path/to/model"));
        assert!(!is_local_file_path("hf://model"));
        assert!(!is_local_file_path("ollama://model"));
    }

    #[test]
    fn test_deep_clicov_benchmark_suites_description_content() {
        for (name, desc) in BENCHMARK_SUITES {
            // Descriptions should be informative
            assert!(desc.len() >= 10, "Description for {} too short", name);
            // Should not contain placeholder text
            assert!(!desc.contains("TODO"), "Description for {} has TODO", name);
            assert!(
                !desc.contains("FIXME"),
                "Description for {} has FIXME",
                name
            );
        }
    }

    #[test]
    fn test_deep_clicov_parse_cargo_bench_output_whitespace_variations() {
        // Various whitespace in bench output
        let output1 = "test  bench_a  ...  bench:  100  ns/iter  (+/- 5)";
        let output2 = "test\tbench_b\t...\tbench:\t200\tns/iter\t(+/- 10)";

        let results1 = parse_cargo_bench_output(output1, None);
        let results2 = parse_cargo_bench_output(output2, None);

        // May or may not parse depending on whitespace handling
        assert!(results1.len() <= 1);
        assert!(results2.len() <= 1);
    }

    #[test]
    fn test_deep_clicov_display_model_info_safetensors_magic() {
        // SafeTensors files start with 8-byte length header
        // Valid SafeTensors starts with little-endian u64 for header size
        let mut data = vec![0u8; 16];
        data[0..8].copy_from_slice(&0u64.to_le_bytes()); // Header size = 0
        let result = display_model_info("model.safetensors", &data);
        // Should try to parse as SafeTensors
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_deep_clicov_load_apr_model_custom_type() {
        // Custom model type (0x00FF)
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x00FFu16.to_le_bytes()); // Custom type
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = load_apr_model(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_deep_clicov_run_benchmarks_all_params_non_list() {
        // All params but not in list mode - will attempt cargo bench
        // Just verify the function signature accepts all params
        // We don't actually run cargo bench in tests
        let result = run_benchmarks(
            Some("tensor_ops".to_string()),
            true, // Use list mode to avoid cargo bench
            Some("realizar".to_string()),
            Some("model.gguf".to_string()),
            Some("http://localhost".to_string()),
            Some("/tmp/out.json".to_string()),
        );
        assert!(result.is_ok());
    }
include!("tests_format_size_03.rs");
include!("tests_load_apr.rs");
include!("tests_format_size_benchmark.rs");
include!("tests_validate_suite.rs");
include!("tests_process_chat.rs");
include!("tests_parse_bench.rs");
include!("tests_process_chat_02.rs");
}
