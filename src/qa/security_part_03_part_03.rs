
    #[test]
    fn test_security_results_all_have_names() {
        let config = SecurityConfig::default();
        let results = run_all_security_tests(&config);

        for result in &results {
            assert!(
                !result.name.is_empty(),
                "Result {} has empty name",
                result.id
            );
        }
    }

    #[test]
    fn test_security_results_all_have_details() {
        let config = SecurityConfig::default();
        let results = run_all_security_tests(&config);

        for result in &results {
            assert!(
                !result.details.is_empty(),
                "Result {} has empty details",
                result.id
            );
        }
    }

    #[test]
    fn test_is_path_safe_backslash_traversal() {
        // Windows-style path traversal
        assert!(!is_path_safe("..\\etc\\passwd"));
        assert!(!is_path_safe("folder\\..\\..\\secret"));
    }

    #[test]
    fn test_n_tests_sequential_ids() {
        let config = SecurityConfig::default();
        let results = run_all_security_tests(&config);

        // Verify IDs are N1 through N20
        for (i, result) in results.iter().enumerate() {
            let expected_id = format!("N{}", i + 1);
            assert_eq!(
                result.id, expected_id,
                "Expected {} but got {}",
                expected_id, result.id
            );
        }
    }

    #[test]
    fn test_escape_html_no_special_chars() {
        let input = "Hello World 123";
        assert_eq!(escape_html(input), input);
    }

    #[test]
    fn test_escape_html_unicode() {
        // Unicode characters should pass through unchanged
        assert_eq!(escape_html("héllo wörld"), "héllo wörld");
        assert_eq!(escape_html("日本語"), "日本語");
    }

    #[test]
    fn test_security_config_debug_format() {
        let config = SecurityConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("SecurityConfig"));
        assert!(debug.contains("fuzz_duration_secs"));
        assert!(debug.contains("max_file_size"));
    }

    #[test]
    fn test_security_result_debug_format() {
        let result = SecurityResult::pass("N1", "Test", "Details");
        let debug = format!("{:?}", result);
        assert!(debug.contains("SecurityResult"));
        assert!(debug.contains("passed"));
    }

    // =====================================================================
    // Coverage boost: is_path_safe with lowercase c: prefix
    // =====================================================================

    #[test]
    fn test_is_path_safe_lowercase_c_drive() {
        assert!(!is_path_safe("c:file.txt"));
        assert!(!is_path_safe("c:\\Users\\test"));
    }

    #[test]
    fn test_is_path_safe_uppercase_c_drive() {
        assert!(!is_path_safe("C:file.txt"));
        assert!(!is_path_safe("C:\\Users\\test"));
    }

    // =====================================================================
    // Coverage boost: escape_html individual special characters
    // =====================================================================

    #[test]
    fn test_escape_html_ampersand_only() {
        assert_eq!(escape_html("&"), "&amp;");
    }

    #[test]
    fn test_escape_html_less_than_only() {
        assert_eq!(escape_html("<"), "&lt;");
    }

    #[test]
    fn test_escape_html_greater_than_only() {
        assert_eq!(escape_html(">"), "&gt;");
    }

    #[test]
    fn test_escape_html_double_quote_only() {
        assert_eq!(escape_html("\""), "&quot;");
    }

    #[test]
    fn test_escape_html_single_quote_only() {
        assert_eq!(escape_html("'"), "&#x27;");
    }

    // =====================================================================
    // Coverage boost: SecurityConfig with wasm_memory_limit variations
    // =====================================================================

    #[test]
    fn test_security_config_wasm_memory_limit_custom() {
        let config = SecurityConfig {
            enable_fuzzing: false,
            fuzz_duration_secs: 60,
            enable_sanitizers: false,
            max_file_size: 100 * 1024 * 1024,
            wasm_memory_limit: 512 * 1024 * 1024, // 512MB
        };
        assert_eq!(config.wasm_memory_limit, 512 * 1024 * 1024);
    }

    // =====================================================================
    // Coverage boost: SecurityResult fail with various IDs
    // =====================================================================

    #[test]
    fn test_security_result_fail_all_fields() {
        let result = SecurityResult::fail("N42", "Custom Fail", "Detailed error description");
        assert!(!result.passed);
        assert_eq!(result.id, "N42");
        assert_eq!(result.name, "Custom Fail");
        assert_eq!(result.details, "Detailed error description");
    }

    // =====================================================================
    // Coverage boost: test_path_traversal_blocked function directly
    // =====================================================================

    #[test]
    fn test_path_traversal_blocked_returns_true() {
        // All malicious paths should be blocked
        assert!(test_path_traversal_blocked());
    }

    // =====================================================================
    // Coverage boost: test_exponential_backoff directly
    // =====================================================================

    #[test]
    fn test_exponential_backoff_returns_true() {
        assert!(test_exponential_backoff());
    }

    // =====================================================================
    // Coverage boost: test_wasm32_limit directly
    // =====================================================================

    #[test]
    fn test_wasm32_limit_returns_true() {
        assert!(test_wasm32_limit());
    }

    // =====================================================================
    // Coverage boost: test_weight_validation directly
    // =====================================================================

    #[test]
    fn test_weight_validation_returns_true() {
        assert!(test_weight_validation());
    }

    // =====================================================================
    // Coverage boost: test_oom_handling directly
    // =====================================================================

    #[test]
    fn test_oom_handling_returns_true() {
        assert!(test_oom_handling());
    }

    // =====================================================================
    // Coverage boost: escape_html with multiple consecutive specials
    // =====================================================================

    #[test]
    fn test_escape_html_consecutive_specials() {
        assert_eq!(escape_html("<><>"), "&lt;&gt;&lt;&gt;");
        assert_eq!(escape_html("&&"), "&amp;&amp;");
        assert_eq!(escape_html("\"\""), "&quot;&quot;");
    }

    // =====================================================================
    // Coverage boost: run_all_security_tests comprehensive
    // =====================================================================

    #[test]
    fn test_run_all_security_tests_with_custom_config() {
        let config = SecurityConfig {
            enable_fuzzing: true,
            fuzz_duration_secs: 30,
            enable_sanitizers: true,
            max_file_size: 50 * 1024 * 1024,
            wasm_memory_limit: 2 * 1024 * 1024 * 1024,
        };
        let results = run_all_security_tests(&config);
        assert_eq!(results.len(), 20);
        // All should still pass (internal checks are design-based)
        for r in &results {
            assert!(r.passed, "Test {} should pass", r.id);
        }
    }
