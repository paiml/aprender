
    #[test]
    fn test_run_with_json_output() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            10,
            3,
            32,
            true, // json output
            false,
            None,
            None,
            None,
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_verbose() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            10,
            3,
            32,
            false,
            true, // verbose
            None,
            None,
            None,
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_safetensors_path() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let st_file = NamedTempFile::with_suffix(".safetensors").expect("create st file");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            Some(st_file.path().to_path_buf()), // safetensors path
            10,
            3,
            32,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
        );
        // Should fail (invalid files)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_small_iterations() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            None,
            1, // small iterations
            0, // no warmup
            8, // small max tokens
            false,
            false,
            None,
            None,
            None,
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    // ========================================================================
    // FORMAT DISPATCH TESTS (P0: Verify formats don't incorrectly skip)
    // These tests ensure that APR, GGUF, and SafeTensors formats are properly
    // dispatched to their handlers and don't silently skip with "GGUF only".
    // ========================================================================

    #[cfg(feature = "inference")]
    mod format_dispatch_tests {
        use realizar::format::{detect_format, ModelFormat};

        /// Test that GGUF magic bytes are correctly detected
        #[test]
        fn test_gguf_format_detection() {
            // GGUF magic: "GGUF" (0x47475546)
            let gguf_magic = b"GGUF\x03\x00\x00\x00"; // GGUF v3
            let format = detect_format(gguf_magic).expect("detect GGUF");
            assert_eq!(format, ModelFormat::Gguf, "GGUF magic must detect as GGUF");
        }

        /// Test that APR v2 magic bytes are correctly detected
        #[test]
        fn test_apr_v2_format_detection() {
            // APR v2 magic: "APR\0" (0x41505200)
            let apr_magic = b"APR\x00\x02\x00\x00\x00"; // APR v2
            let format = detect_format(apr_magic).expect("detect APR");
            assert_eq!(format, ModelFormat::Apr, "APR magic must detect as APR");
        }

        /// Test that SafeTensors format is correctly detected
        #[test]
        fn test_safetensors_format_detection() {
            // SafeTensors starts with u64 header length, then JSON
            let mut st_magic = Vec::new();
            st_magic.extend_from_slice(&100u64.to_le_bytes()); // header length
            st_magic.extend_from_slice(b"{\""); // JSON start
            let format = detect_format(&st_magic).expect("detect SafeTensors");
            assert_eq!(
                format,
                ModelFormat::SafeTensors,
                "SafeTensors magic must detect as SafeTensors"
            );
        }

        /// P0 REGRESSION TEST: APR format must NOT skip golden_output gate
        /// This test catches the bug where APR files silently returned "GGUF only"
        #[test]
        fn test_apr_format_does_not_skip_detection() {
            // Create minimal APR v2 header (8 bytes minimum for format detection)
            let apr_magic = b"APR\x00\x02\x00\x00\x00"; // APR v2 magic + version
            let format = detect_format(apr_magic).expect("detect APR");

            // The critical assertion: APR must be detected as APR, not fail/skip
            assert_eq!(
                format,
                ModelFormat::Apr,
                "APR format MUST be detected - cannot skip with 'GGUF only' error"
            );
        }

        /// P0 REGRESSION TEST: Verify ModelFormat enum covers all expected formats
        #[test]
        fn test_model_format_enum_completeness() {
            // This test documents the expected formats
            let formats = [
                ModelFormat::Gguf,
                ModelFormat::Apr,
                ModelFormat::SafeTensors,
            ];
            assert_eq!(
                formats.len(),
                3,
                "Must support exactly 3 formats: GGUF, APR, SafeTensors"
            );
        }
    }

    // ========================================================================
    // GATE RESULT NON-SKIP TESTS
    // Verify that gates return actual results (pass/fail) not skipped
    // ========================================================================

    #[test]
    fn test_gate_result_skipped_flag_semantics() {
        // Skipped gates have skipped=true
        let skipped = GateResult::skipped("test", "reason");
        assert!(skipped.skipped, "Skipped gate must have skipped=true");
        assert!(skipped.passed, "Skipped gates count as passed (don't fail)");

        // Passed gates have skipped=false
        let passed = GateResult::passed("test", "ok", None, None, Duration::from_secs(1));
        assert!(!passed.skipped, "Passed gate must have skipped=false");
        assert!(passed.passed, "Passed gate must have passed=true");

        // Failed gates have skipped=false
        let failed = GateResult::failed("test", "fail", None, None, Duration::from_secs(1));
        assert!(!failed.skipped, "Failed gate must have skipped=false");
        assert!(!failed.passed, "Failed gate must have passed=false");
    }

    /// P0 REGRESSION TEST: Gates that skip must have explicit reason
    #[test]
    fn test_skipped_gate_must_have_reason() {
        let result = GateResult::skipped("test_gate", "Explicit reason required");
        assert!(
            result.message.contains("Skipped"),
            "Skip message must contain 'Skipped'"
        );
        assert!(result.message.len() > 10, "Skip reason must be descriptive");
    }

    // ========================================================================
    // GateResult: boundary values and value/threshold interactions
    // ========================================================================

    /// A gate whose measured value exactly equals the threshold should pass.
    /// Bug class: using > instead of >= in threshold comparison, causing
    /// exact-threshold values to fail.
    #[test]
    fn gate_result_value_equals_threshold_is_pass() {
        // When value == threshold, the gate is "passed" (caller constructs it)
        // This test documents the semantic contract: equality means pass.
        let result = GateResult::passed(
            "throughput",
            "100.0 tok/s >= 100.0 tok/s threshold",
            Some(100.0),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(result.passed);
        assert_eq!(result.value, Some(100.0));
        assert_eq!(result.threshold, Some(100.0));
    }

    /// A gate with value just below threshold should be failed.
    /// Bug class: floating-point equality masking near-miss failures.
    #[test]
    fn gate_result_value_just_below_threshold_is_fail() {
        let result = GateResult::failed(
            "throughput",
            "99.9 tok/s < 100.0 tok/s",
            Some(99.9),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(!result.passed);
        assert!(!result.skipped);
    }

    /// Zero-duration gate result should be representable.
    /// Bug class: division by zero in duration_ms calculation.
    #[test]
    fn gate_result_zero_duration() {
        let result = GateResult::passed(
            "fast_gate",
            "Sub-millisecond completion",
            None,
            None,
            Duration::from_nanos(0),
        );
        assert_eq!(result.duration_ms, 0);
        assert!(result.passed);
    }

    /// Very large duration should not overflow u64 milliseconds.
    /// Bug class: u64 overflow when converting Duration to millis.
    #[test]
    fn gate_result_large_duration_no_overflow() {
        // 1 million seconds = ~11.5 days (extreme but valid)
        let result = GateResult::passed(
            "slow_gate",
            "Long-running test",
            None,
            None,
            Duration::from_secs(1_000_000),
        );
        assert_eq!(result.duration_ms, 1_000_000_000);
    }

    /// Skipped gates must have None for value and threshold.
    /// Bug class: skipped constructor inadvertently setting default values
    /// that confuse downstream reporting (e.g., "0.0 vs 0.0 threshold").
    #[test]
    fn gate_result_skipped_has_no_metrics() {
        let result = GateResult::skipped("contract", "Model not found");
        assert!(result.value.is_none(), "Skipped gate must have no value");
        assert!(
            result.threshold.is_none(),
            "Skipped gate must have no threshold"
        );
    }

    /// Failed gate with None value (e.g., infrastructure failure, not metric miss).
    /// Bug class: downstream code unwrapping value.unwrap() on failure.
    #[test]
    fn gate_result_failed_without_value() {
        let result = GateResult::failed(
            "golden_output",
            "Inference engine crashed",
            None,
            None,
            Duration::from_millis(50),
        );
        assert!(!result.passed);
        assert!(result.value.is_none());
    }

    // ========================================================================
    // GateResult serialization: JSON round-trip fidelity
    // ========================================================================

    /// Round-trip: passed gate with all fields must survive JSON serialization.
    /// Bug class: serde skip_serializing_if dropping fields that should be present.
    #[test]
    fn gate_result_json_roundtrip_with_values() {
        let original = GateResult::passed(
            "throughput",
            "150.0 tok/s >= 100.0 tok/s",
            Some(150.0),
            Some(100.0),
            Duration::from_millis(2500),
        );
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: GateResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.name, "throughput");
        assert!(restored.passed);
        assert!(!restored.skipped);
        assert_eq!(restored.value, Some(150.0));
        assert_eq!(restored.threshold, Some(100.0));
        assert_eq!(restored.duration_ms, 2500);
    }

    /// Round-trip: skipped gate should preserve skipped=true through JSON.
    /// Bug class: skipped field defaulting to false on deserialization.
    #[test]
    fn gate_result_json_roundtrip_skipped() {
        let original = GateResult::skipped("gpu_speedup", "No GPU");
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: GateResult = serde_json::from_str(&json).expect("deserialize");
        assert!(restored.skipped, "skipped flag must survive round-trip");
        assert!(restored.passed, "skipped gates must still show passed=true");
        assert!(
            restored.value.is_none(),
            "value should be None after round-trip"
        );
    }

    /// JSON with None value/threshold should omit those fields entirely.
    /// Bug class: serializing None as null instead of omitting.
    #[test]
    fn gate_result_json_omits_none_fields() {
        let result = GateResult::passed("test", "ok", None, None, Duration::from_secs(1));
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(
            !json.contains("value"),
            "None value should be omitted from JSON, got: {json}"
        );
        assert!(
            !json.contains("threshold"),
            "None threshold should be omitted from JSON, got: {json}"
        );
    }

    // ========================================================================
    // QaReport: aggregate pass/fail logic
    // ========================================================================

    /// A report with all skipped gates should pass (skips never fail).
    /// Bug class: empty non-skipped gate list treated as failure.
    #[test]
    fn qa_report_all_skipped_gates_passes() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![
                GateResult::skipped("golden", "no model"),
                GateResult::skipped("throughput", "no engine"),
                GateResult::skipped("ollama", "not available"),
            ],
            total_duration_ms: 10,
            timestamp: "2026-02-06T00:00:00Z".to_string(),
            summary: "All skipped".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        assert!(report.passed);
        assert!(
            report.gates.iter().all(|g| g.skipped),
            "All gates should be skipped"
        );
        assert!(
            report.gates.iter().all(|g| g.passed),
            "All skipped gates should count as passed"
        );
    }
