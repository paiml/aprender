
    // ========================================================================
    // NEW: QaReport JSON deserialize with extra fields (forward compat)
    // ========================================================================

    /// JSON with extra unknown fields should still deserialize (serde default).
    #[test]
    fn qa_report_deserialize_ignores_unknown_fields() {
        let json = r#"{
            "model": "test.gguf",
            "passed": true,
            "gates": [],
            "total_duration_ms": 100,
            "timestamp": "2026-02-07T00:00:00Z",
            "summary": "ok",
            "extra_field": "should be ignored",
            "another_extra": 42
        }"#;
        let report: QaReport = serde_json::from_str(json).expect("deserialize with extras");
        assert_eq!(report.model, "test.gguf");
        assert!(report.passed);
    }

    /// GateResult JSON with extra unknown fields should still deserialize.
    #[test]
    fn gate_result_deserialize_ignores_unknown_fields() {
        let json = r#"{
            "name": "test",
            "passed": true,
            "message": "ok",
            "duration_ms": 100,
            "skipped": false,
            "future_field": "v2"
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize with extras");
        assert_eq!(result.name, "test");
        assert!(result.passed);
    }

    // ========================================================================
    // verify_output Tests (PMAT-QA-PROTOCOL-001 §7.4)
    // ========================================================================

    #[test]
    fn verify_output_rejects_empty() {
        let result = verify_output("", "test-001", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(reason.contains("Empty"), "Expected 'Empty', got: {reason}");
        }
    }

    #[test]
    fn verify_output_rejects_whitespace_only() {
        let result = verify_output("   \n\t  ", "test-002", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
    }

    #[test]
    fn verify_output_rejects_garbage_fffd() {
        let result = verify_output("The answer is \u{FFFD}\u{FFFD}", "test-003", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("Garbage"),
                "Expected 'Garbage', got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_rejects_garbage_unk() {
        let result = verify_output("Hello [UNK] world", "test-004", &["Hello"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("Garbage"),
                "Expected 'Garbage', got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_rejects_null_bytes() {
        let result = verify_output("Hello\0World", "test-005", &["Hello"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("null"),
                "Expected 'null bytes', got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_rejects_missing_expected() {
        let result = verify_output("The answer is five", "test-006", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("Expected"),
                "Expected mention of pattern, got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_accepts_correct() {
        let result = verify_output("The answer is 4.", "test-007", &["4"]);
        assert!(matches!(result, OutputVerification::Pass));
    }

    #[test]
    fn verify_output_accepts_any_expected_pattern() {
        let result = verify_output("Hi there!", "test-008", &["Hello", "Hi", "Hey"]);
        assert!(matches!(result, OutputVerification::Pass));
    }

    #[test]
    fn verify_output_case_insensitive() {
        let result = verify_output("HELLO WORLD", "test-009", &["hello"]);
        assert!(matches!(result, OutputVerification::Pass));
    }

    #[test]
    fn verify_output_garbage_check_before_answer_check() {
        // Even though output contains "4", garbage should fail first
        let result = verify_output("4 [UNK] answer", "test-010", &["4"]);
        assert!(matches!(result, OutputVerification::Fail { .. }));
        if let OutputVerification::Fail { reason } = result {
            assert!(
                reason.contains("Garbage"),
                "Garbage check must happen BEFORE answer check, got: {reason}"
            );
        }
    }

    #[test]
    fn verify_output_no_expected_patterns_passes() {
        // If no patterns expected, just check for emptiness and garbage
        let result = verify_output("Some valid output", "test-011", &[]);
        assert!(matches!(result, OutputVerification::Pass));
    }

    // ========================================================================
    // Ollama Parity Grade Tests (F-PROFILE-010)
    // ========================================================================

    #[cfg(feature = "inference")]
    #[test]
    fn ollama_parity_grade_boundaries() {
        // Grade F: <50% Ollama
        assert_eq!(ollama_parity_grade(0.0), "F");
        assert_eq!(ollama_parity_grade(0.3), "F");
        assert_eq!(ollama_parity_grade(0.49), "F");
        // Grade D: 50-75%
        assert_eq!(ollama_parity_grade(0.5), "D");
        assert_eq!(ollama_parity_grade(0.64), "D");
        assert_eq!(ollama_parity_grade(0.74), "D");
        // Grade C: 75-100% (parity)
        assert_eq!(ollama_parity_grade(0.75), "C");
        assert_eq!(ollama_parity_grade(0.99), "C");
        // Grade B: 100-150%
        assert_eq!(ollama_parity_grade(1.0), "B");
        assert_eq!(ollama_parity_grade(1.49), "B");
        // Grade A: 150-200%
        assert_eq!(ollama_parity_grade(1.5), "A");
        assert_eq!(ollama_parity_grade(1.99), "A");
        // Grade A+: 200%+
        assert_eq!(ollama_parity_grade(2.0), "A+");
        assert_eq!(ollama_parity_grade(3.5), "A+");
    }
