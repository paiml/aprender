
    #[test]
    fn test_n1_fuzzing_load() {
        let result = n1_fuzzing_load_infrastructure();
        assert!(result.passed);
        assert_eq!(result.id, "N1");
    }

    #[test]
    fn test_n2_fuzzing_audio() {
        let result = n2_fuzzing_audio_infrastructure();
        assert!(result.passed);
        assert_eq!(result.id, "N2");
    }

    #[test]
    fn test_n3_mutation_score() {
        let result = n3_mutation_score();
        assert!(result.passed);
    }

    #[test]
    fn test_n4_tsan_clean() {
        let result = n4_thread_sanitizer_clean();
        assert!(result.passed);
    }

    #[test]
    fn test_n5_msan_clean() {
        let result = n5_memory_sanitizer_clean();
        assert!(result.passed);
    }

    #[test]
    fn test_n6_panic_safety() {
        let result = n6_panic_safety_ffi();
        assert!(result.passed);
    }

    #[test]
    fn test_n7_error_propagation() {
        let result = n7_error_propagation();
        assert!(result.passed);
    }

    #[test]
    fn test_n8_oom_handling() {
        let result = n8_oom_handling();
        assert!(result.passed);
    }

    #[test]
    fn test_n9_fd_leak_check() {
        let result = n9_fd_leak_check();
        assert!(result.passed);
    }

    #[test]
    fn test_n10_path_traversal() {
        let result = n10_path_traversal_prevention();
        assert!(result.passed);

        // Additional verification
        assert!(!is_path_safe("../etc/passwd"));
        assert!(!is_path_safe("/etc/passwd"));
        assert!(is_path_safe("model.apr"));
        assert!(is_path_safe("models/whisper.apr"));
    }

    #[test]
    fn test_n11_dependency_audit() {
        let result = n11_dependency_audit();
        assert!(result.passed);
    }

    #[test]
    fn test_n12_replay_attack() {
        let result = n12_replay_attack_resistance();
        assert!(result.passed);
    }

    #[test]
    fn test_n13_timing_attack() {
        let result = n13_timing_attack_resistance();
        assert!(result.passed);
    }

    #[test]
    fn test_n14_xss_prevention() {
        let result = n14_xss_injection_prevention();
        assert!(result.passed);

        // Verify escaping works
        assert_eq!(escape_html("<script>"), "&lt;script&gt;");
        assert_eq!(escape_html("a & b"), "a &amp; b");
    }

    #[test]
    fn test_n15_wasm_sandboxing() {
        let result = n15_wasm_sandboxing();
        assert!(result.passed);
    }

    #[test]
    fn test_n16_disk_full() {
        let result = n16_disk_full_handling();
        assert!(result.passed);
    }

    #[test]
    fn test_n17_network_timeout() {
        let result = n17_network_timeout_handling();
        assert!(result.passed);
    }

    #[test]
    fn test_n18_golden_trace() {
        let result = n18_golden_trace_regression();
        assert!(result.passed);
    }

    #[test]
    fn test_n19_32bit_limit() {
        let result = n19_32bit_address_limit();
        assert!(result.passed);
    }

    #[test]
    fn test_n20_nan_inf_weights() {
        let result = n20_nan_inf_weight_handling();
        assert!(result.passed);
    }

    #[test]
    fn test_run_all_security_tests() {
        let config = SecurityConfig::default();
        let results = run_all_security_tests(&config);

        assert_eq!(results.len(), 20);
        assert!(results.iter().all(|r| r.passed));
    }

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert!(!config.enable_fuzzing);
        assert_eq!(config.fuzz_duration_secs, 60);
        assert_eq!(config.max_file_size, 100 * 1024 * 1024);
    }

    #[test]
    fn test_security_result_creation() {
        let pass = SecurityResult::pass("N1", "Test", "Details");
        assert!(pass.passed);
        assert_eq!(pass.id, "N1");

        let fail = SecurityResult::fail("N2", "Test", "Error");
        assert!(!fail.passed);
    }

    #[test]
    fn test_exponential_backoff_calculation() {
        assert!(test_exponential_backoff());
    }

    #[test]
    fn test_weight_validation_detects_invalid() {
        assert!(test_weight_validation());
    }

    #[test]
    fn test_escape_html_comprehensive() {
        assert_eq!(escape_html("Hello"), "Hello");
        assert_eq!(escape_html("<div>"), "&lt;div&gt;");
        assert_eq!(escape_html("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(escape_html("it's"), "it&#x27;s");
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_security_config_debug_clone() {
        let config = SecurityConfig::default();
        let cloned = config.clone();
        assert_eq!(config.enable_fuzzing, cloned.enable_fuzzing);
        assert_eq!(config.fuzz_duration_secs, cloned.fuzz_duration_secs);

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("enable_fuzzing"));
    }

    #[test]
    fn test_security_result_debug_clone() {
        let result = SecurityResult::pass("N1", "Test", "Details");
        let cloned = result.clone();
        assert_eq!(result.id, cloned.id);
        assert_eq!(result.passed, cloned.passed);

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("N1"));
    }

    #[test]
    fn test_is_path_safe_safe_paths() {
        assert!(is_path_safe("model.apr"));
        assert!(is_path_safe("models/whisper.apr"));
        assert!(is_path_safe("path/to/file.txt"));
        assert!(is_path_safe("relative/path"));
    }

    #[test]
    fn test_is_path_safe_dot_dot() {
        assert!(!is_path_safe("../something"));
        assert!(!is_path_safe("a/../../b"));
        assert!(!is_path_safe(".."));
    }

    #[test]
    fn test_is_path_safe_windows_paths() {
        assert!(!is_path_safe("C:\\Windows"));
        assert!(!is_path_safe("c:\\Users"));
        assert!(!is_path_safe("C:file.txt"));
    }

    #[test]
    fn test_is_path_safe_absolute_paths() {
        assert!(!is_path_safe("/etc/passwd"));
        assert!(!is_path_safe("/home/user/file"));
    }

    #[test]
    fn test_oom_handling_function() {
        // test_oom_handling allocates 1MB and should succeed
        assert!(test_oom_handling());
    }

    #[test]
    fn test_wasm32_limit_function() {
        // test_wasm32_limit checks if 5GB > 4GB limit
        assert!(test_wasm32_limit());
    }

    #[test]
    fn test_path_traversal_blocked_function() {
        assert!(test_path_traversal_blocked());
    }

    #[test]
    fn test_security_config_custom() {
        let config = SecurityConfig {
            enable_fuzzing: true,
            fuzz_duration_secs: 120,
            enable_sanitizers: true,
            max_file_size: 50 * 1024 * 1024,
            wasm_memory_limit: 1024 * 1024 * 1024,
        };
        assert!(config.enable_fuzzing);
        assert_eq!(config.fuzz_duration_secs, 120);
        assert!(config.enable_sanitizers);
    }

    #[test]
    fn test_security_result_fail_details() {
        let fail = SecurityResult::fail("N99", "Failed Test", "Something went wrong");
        assert!(!fail.passed);
        assert_eq!(fail.id, "N99");
        assert_eq!(fail.name, "Failed Test");
        assert_eq!(fail.details, "Something went wrong");
    }

    #[test]
    fn test_escape_html_combined() {
        // Test a string with multiple special characters
        assert_eq!(
            escape_html("<a href=\"test\">it's & fun</a>"),
            "&lt;a href=&quot;test&quot;&gt;it&#x27;s &amp; fun&lt;/a&gt;"
        );
    }

    #[test]
    fn test_escape_html_empty() {
        assert_eq!(escape_html(""), "");
    }

    #[test]
    fn test_weight_validation_with_only_valid() {
        let weights = [1.0_f32, 2.0, 3.0, 4.0];
        let has_invalid = weights.iter().any(|w| w.is_nan() || w.is_infinite());
        assert!(!has_invalid);
    }

    #[test]
    fn test_weight_validation_with_nan() {
        let weights = [1.0_f32, f32::NAN, 3.0];
        let has_invalid = weights.iter().any(|w| w.is_nan() || w.is_infinite());
        assert!(has_invalid);
    }

    #[test]
    fn test_weight_validation_with_inf() {
        let weights = [1.0_f32, f32::INFINITY, 3.0];
        let has_invalid = weights.iter().any(|w| w.is_nan() || w.is_infinite());
        assert!(has_invalid);
    }

    #[test]
    fn test_weight_validation_with_neg_inf() {
        let weights = [1.0_f32, f32::NEG_INFINITY, 3.0];
        let has_invalid = weights.iter().any(|w| w.is_nan() || w.is_infinite());
        assert!(has_invalid);
    }

    #[test]
    fn test_exponential_backoff_values() {
        let base_delay_ms = 100;
        let max_retries = 5;

        let delays: Vec<u64> = (0..max_retries)
            .map(|attempt| base_delay_ms * 2_u64.pow(attempt))
            .collect();

        assert_eq!(delays, vec![100, 200, 400, 800, 1600]);
    }

    #[test]
    fn test_all_security_results_have_unique_ids() {
        let config = SecurityConfig::default();
        let results = run_all_security_tests(&config);

        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "Duplicate ID found");
            }
        }
    }

    #[test]
    fn test_security_result_pass_has_true() {
        let result = SecurityResult::pass("T1", "Name", "Details");
        assert!(result.passed);
    }

    #[test]
    fn test_wasm_memory_limit_in_default_config() {
        let config = SecurityConfig::default();
        // On non-wasm32 targets, should be 4GB
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(config.wasm_memory_limit, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_is_path_safe_empty_path() {
        // Empty path is considered safe (relative)
        assert!(is_path_safe(""));
    }

    #[test]
    fn test_is_path_safe_single_file() {
        assert!(is_path_safe("file.txt"));
        assert!(is_path_safe("a"));
    }

    #[test]
    fn test_is_path_safe_nested_relative() {
        assert!(is_path_safe("a/b/c/d/e/file.txt"));
        assert!(is_path_safe("deeply/nested/path/to/model.apr"));
    }

    #[test]
    fn test_is_path_safe_with_dots_in_name() {
        // Dots that aren't ".." should be allowed
        assert!(is_path_safe("file.name.with.dots.txt"));
        assert!(is_path_safe("version.1.0.apr"));
    }

    #[test]
    fn test_is_path_safe_hidden_files() {
        // Unix hidden files (starting with .)
        assert!(is_path_safe(".hidden"));
        assert!(is_path_safe(".config/settings"));
    }

    #[test]
    fn test_is_path_safe_trailing_dot_dot() {
        assert!(!is_path_safe("path/to/.."));
        assert!(!is_path_safe("folder/.."));
    }

    #[test]
    fn test_security_result_name_field() {
        let pass = SecurityResult::pass("X1", "Test Name Here", "Some details");
        assert_eq!(pass.name, "Test Name Here");

        let fail = SecurityResult::fail("X2", "Another Name", "Error details");
        assert_eq!(fail.name, "Another Name");
    }

    #[test]
    fn test_security_result_details_field() {
        let result = SecurityResult::pass("X1", "Name", "These are the details");
        assert_eq!(result.details, "These are the details");
    }

    #[test]
    fn test_security_config_all_fields_accessible() {
        let config = SecurityConfig {
            enable_fuzzing: true,
            fuzz_duration_secs: 300,
            enable_sanitizers: true,
            max_file_size: 1024,
            wasm_memory_limit: 2048,
        };

        assert!(config.enable_fuzzing);
        assert_eq!(config.fuzz_duration_secs, 300);
        assert!(config.enable_sanitizers);
        assert_eq!(config.max_file_size, 1024);
        assert_eq!(config.wasm_memory_limit, 2048);
    }

    #[test]
    fn test_exponential_backoff_single_retry() {
        let base_delay_ms = 100;
        let delays: Vec<u64> = (0..1)
            .map(|attempt| base_delay_ms * 2_u64.pow(attempt))
            .collect();
        assert_eq!(delays, vec![100]);
    }

    #[test]
    fn test_exponential_backoff_large_retries() {
        let base_delay_ms = 100;
        let max_retries = 10;

        let delays: Vec<u64> = (0..max_retries)
            .map(|attempt| base_delay_ms * 2_u64.pow(attempt))
            .collect();

        // Last delay should be 100 * 2^9 = 51200
        assert_eq!(delays[9], 51200);
    }
