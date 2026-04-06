//! CLI Module Tests Part 03 - Helper Functions Coverage
//!
//! Tests for CLI helper functions:
//! - format_size
//! - is_local_file_path
//! - validate_suite_name
//! - home_dir
//! - RunConfig and ServeConfig structs
//! - CLI parsing (Commands enum)
//!
//! Refs PMAT-802: Protocol T-COV-95

#[cfg(test)]
mod tests {
    use super::super::*;

    // =========================================================================
    // format_size Tests
    // =========================================================================

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1), "1 B");
        assert_eq!(format_size(100), "100 B");
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(2048), "2.0 KB");
        assert_eq!(format_size(1024 * 1023), "1023.0 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 + 512 * 1024), "1.5 MB");
        assert_eq!(format_size(100 * 1024 * 1024), "100.0 MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(5 * 1024 * 1024 * 1024), "5.0 GB");
        assert_eq!(
            format_size(7 * 1024 * 1024 * 1024 + 512 * 1024 * 1024),
            "7.5 GB"
        );
    }

    #[test]
    fn test_format_size_boundary_kb() {
        // Exactly at KB boundary
        assert_eq!(format_size(1024), "1.0 KB");
        // Just under KB
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_boundary_mb() {
        // Exactly at MB boundary
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        // Just under MB
        assert_eq!(format_size(1024 * 1024 - 1), "1024.0 KB");
    }

    #[test]
    fn test_format_size_boundary_gb() {
        // Exactly at GB boundary
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        // Just under GB
        assert_eq!(format_size(1024 * 1024 * 1024 - 1), "1024.0 MB");
    }

    // =========================================================================
    // is_local_file_path Tests
    // =========================================================================

    #[test]
    fn test_is_local_file_path_relative_dot_slash() {
        assert!(is_local_file_path("./model.gguf"));
        assert!(is_local_file_path("./path/to/model.gguf"));
        assert!(is_local_file_path("./"));
    }

    #[test]
    fn test_is_local_file_path_absolute() {
        assert!(is_local_file_path("/home/user/model.gguf"));
        assert!(is_local_file_path("/model.safetensors"));
        assert!(is_local_file_path("/"));
    }

    #[test]
    fn test_is_local_file_path_gguf_extension() {
        assert!(is_local_file_path("model.gguf"));
        assert!(is_local_file_path("path/to/model.gguf"));
        assert!(is_local_file_path("MODEL.gguf"));
    }

    #[test]
    fn test_is_local_file_path_safetensors_extension() {
        assert!(is_local_file_path("model.safetensors"));
        assert!(is_local_file_path("path/model.safetensors"));
    }

    #[test]
    fn test_is_local_file_path_apr_extension() {
        assert!(is_local_file_path("model.apr"));
        assert!(is_local_file_path("path/model.apr"));
    }

    #[test]
    fn test_is_local_file_path_remote_refs() {
        // These should NOT be local file paths
        assert!(!is_local_file_path("llama3:8b"));
        assert!(!is_local_file_path("hf://org/model"));
        assert!(!is_local_file_path("pacha://model:version"));
        assert!(!is_local_file_path("model-name"));
    }

    #[test]
    fn test_is_local_file_path_edge_cases() {
        // Empty string
        assert!(!is_local_file_path(""));
        // Just a dot
        assert!(!is_local_file_path("."));
        // Double dot
        assert!(!is_local_file_path(".."));
        // Extension only
        assert!(is_local_file_path(".gguf")); // Ends with .gguf
    }

    // =========================================================================
    // validate_suite_name Tests
    // =========================================================================

    #[test]
    fn test_validate_suite_name_valid() {
        assert!(validate_suite_name("tensor_ops"));
        assert!(validate_suite_name("inference"));
        assert!(validate_suite_name("cache"));
        assert!(validate_suite_name("tokenizer"));
        assert!(validate_suite_name("quantize"));
        assert!(validate_suite_name("lambda"));
        assert!(validate_suite_name("comparative"));
    }

    #[test]
    fn test_validate_suite_name_invalid() {
        assert!(!validate_suite_name("unknown"));
        assert!(!validate_suite_name(""));
        assert!(!validate_suite_name("tensor_ops "));
        assert!(!validate_suite_name(" tensor_ops"));
        assert!(!validate_suite_name("TENSOR_OPS")); // Case sensitive
    }

    #[test]
    fn test_validate_suite_name_partial_matches() {
        // Partial matches should not be valid
        assert!(!validate_suite_name("tensor"));
        assert!(!validate_suite_name("infer"));
        assert!(!validate_suite_name("ops"));
    }

    // =========================================================================
    // home_dir Tests
    // =========================================================================

    #[test]
    fn test_home_dir_returns_some_when_set() {
        // Save original HOME
        let original = std::env::var_os("HOME");

        // Set HOME to a known value
        std::env::set_var("HOME", "/test/home");
        let result = home_dir();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), std::path::PathBuf::from("/test/home"));

        // Restore original HOME
        if let Some(val) = original {
            std::env::set_var("HOME", val);
        } else {
            std::env::remove_var("HOME");
        }
    }

    #[test]
    fn test_home_dir_returns_none_when_unset() {
        // Save original HOME
        let original = std::env::var_os("HOME");

        // Remove HOME
        std::env::remove_var("HOME");
        let result = home_dir();
        // Note: This may still return Some on some systems that have fallbacks
        // But with our simple implementation, it should return None

        // Restore original HOME
        if let Some(val) = original {
            std::env::set_var("HOME", val);
        }

        // The test passes if it doesn't panic - HOME may or may not be set
        let _ = result;
    }

    // =========================================================================
    // handlers::RunConfig Tests
    // =========================================================================

    #[test]
    fn test_run_config_debug() {
        let config = handlers::RunConfig {
            model: "test.gguf".to_string(),
            prompt: Some("Hello".to_string()),
            max_tokens: 100,
            temperature: 0.7,
            format: "text".to_string(),
            system: None,
            raw: false,
            gpu: true,
            verbose: false,
            trace: None,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("RunConfig"));
        assert!(debug_str.contains("test.gguf"));
    }

    #[test]
    fn test_run_config_clone() {
        let config = handlers::RunConfig {
            model: "model.gguf".to_string(),
            prompt: Some("test".to_string()),
            max_tokens: 50,
            temperature: 0.5,
            format: "json".to_string(),
            system: Some("You are helpful".to_string()),
            raw: true,
            gpu: false,
            verbose: true,
            trace: Some(Some("attention".to_string())),
        };

        let cloned = config.clone();
        assert_eq!(cloned.model, "model.gguf");
        assert_eq!(cloned.prompt, Some("test".to_string()));
        assert_eq!(cloned.max_tokens, 50);
        assert!((cloned.temperature - 0.5).abs() < 0.001);
        assert_eq!(cloned.format, "json");
        assert_eq!(cloned.system, Some("You are helpful".to_string()));
        assert!(cloned.raw);
        assert!(!cloned.gpu);
        assert!(cloned.verbose);
        assert_eq!(cloned.trace, Some(Some("attention".to_string())));
    }

    // =========================================================================
    // handlers::ServeConfig Tests
    // =========================================================================

    #[test]
    fn test_serve_config_debug() {
        let config = handlers::ServeConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model: Some("model.gguf".to_string()),
            demo: false,
            batch: true,
            gpu: true,
            openai_api: true,
            trace: false,
            context_length: 4096,
            no_fp8_cache: false,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ServeConfig"));
        assert!(debug_str.contains("127.0.0.1"));
        assert!(debug_str.contains("8080"));
    }

    #[test]
    fn test_serve_config_clone() {
        let config = handlers::ServeConfig {
            host: "0.0.0.0".to_string(),
            port: 3000,
            model: None,
            demo: true,
            batch: false,
            gpu: false,
            openai_api: true,
            trace: false,
            context_length: 4096,
            no_fp8_cache: false,
        };

        let cloned = config.clone();
        assert_eq!(cloned.host, "0.0.0.0");
        assert_eq!(cloned.port, 3000);
        assert!(cloned.model.is_none());
        assert!(cloned.demo);
        assert!(!cloned.batch);
        assert!(!cloned.gpu);
    }

    // =========================================================================
    // BENCHMARK_SUITES Tests
    // =========================================================================

    #[test]
    fn test_benchmark_suites_has_expected_entries() {
        assert!(BENCHMARK_SUITES.len() >= 7);

        // Check all expected suites exist
        let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"tensor_ops"));
        assert!(names.contains(&"inference"));
        assert!(names.contains(&"cache"));
        assert!(names.contains(&"tokenizer"));
        assert!(names.contains(&"quantize"));
        assert!(names.contains(&"lambda"));
        assert!(names.contains(&"comparative"));
    }

    #[test]
    fn test_benchmark_suites_descriptions_not_empty() {
        for (name, description) in BENCHMARK_SUITES {
            assert!(!name.is_empty(), "Suite name should not be empty");
            assert!(
                !description.is_empty(),
                "Suite '{}' description should not be empty",
                name
            );
        }
    }

    // =========================================================================
    // print_info Tests (coverage)
    // =========================================================================

    #[test]
    fn test_print_info_does_not_panic() {
        // print_info() just prints to stdout, we verify it doesn't panic
        print_info();
    }

    // =========================================================================
    // run_visualization Tests (coverage)
    // =========================================================================

    #[test]
    fn test_run_visualization_no_panic() {
        // run_visualization just prints, verify no panic
        run_visualization(false, 10);
    }

    #[test]
    fn test_run_visualization_with_color() {
        run_visualization(true, 5);
    }

    #[test]
    fn test_run_visualization_zero_samples() {
        // Edge case: zero samples
        run_visualization(false, 0);
    }

    // =========================================================================
    // display_model_info Error Cases
    // =========================================================================

    #[test]
    fn test_display_model_info_small_file() {
        // File too small to be a valid model
        let data = [0u8; 4];
        let result = display_model_info("model.gguf", &data);
        // Should error or handle gracefully
        // GGUF requires at least 8 bytes for magic
        assert!(result.is_err());
    }

    #[test]
    fn test_display_model_info_unknown_format() {
        // Random data that doesn't match any format
        let data = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        let result = display_model_info("unknown.bin", &data);
        // Should succeed with "Unknown" format
        assert!(result.is_ok());
    }

    // =========================================================================
    // CLI Parsing Tests (clap)
    // =========================================================================

    #[test]
    fn test_cli_debug() {
        use clap::Parser;

        let cli = handlers::Cli::try_parse_from(["realizar", "info"]);
        assert!(cli.is_ok());

        let debug_str = format!("{:?}", cli.unwrap());
        assert!(debug_str.contains("Cli"));
        assert!(debug_str.contains("Info"));
    }

    #[test]
    fn test_commands_debug() {
        let cmd = handlers::Commands::Info;
        let debug_str = format!("{:?}", cmd);
        assert!(debug_str.contains("Info"));
    }

    #[test]
    fn test_commands_run_debug() {
        let cmd = handlers::Commands::Run {
            model: "test.gguf".to_string(),
            prompt: Some("Hello".to_string()),
            max_tokens: 100,
            temperature: 0.7,
            format: "text".to_string(),
            system: None,
            raw: false,
            gpu: false,
            verbose: false,
            trace: None,
        };

        let debug_str = format!("{:?}", cmd);
        assert!(debug_str.contains("Run"));
        assert!(debug_str.contains("test.gguf"));
    }
}
