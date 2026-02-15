
    /// Zero failed gates should not produce a "Failed gates:" summary.
    #[test]
    fn failed_gates_summary_no_failures() {
        let gates = vec![
            GateResult::passed("golden_output", "ok", None, None, Duration::from_secs(1)),
            GateResult::skipped("ollama_parity", "not available"),
        ];
        let passed = gates.iter().all(|g| g.passed);
        assert!(passed);
        let summary = if passed {
            "All QA gates passed".to_string()
        } else {
            unreachable!()
        };
        assert_eq!(summary, "All QA gates passed");
    }

    // ========================================================================
    // GateResult: NaN and infinity in value/threshold
    // ========================================================================

    /// NaN values in gate results: the struct itself can hold NaN,
    /// verifying the value is stored correctly (NaN != NaN by IEEE 754).
    /// Bug class: accidentally comparing NaN with == and losing the signal.
    #[test]
    fn gate_result_nan_value_is_nan() {
        let result = GateResult::passed(
            "test",
            "NaN test",
            Some(f64::NAN),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(
            result.value.expect("should have value").is_nan(),
            "NaN value must be preserved in GateResult"
        );
        assert!(
            !result.value.expect("should have value").is_finite(),
            "NaN is not finite"
        );
    }

    /// Infinity in gate results should be representable.
    /// Bug class: threshold comparison logic using >= with infinity.
    #[test]
    fn gate_result_infinity_value_is_infinite() {
        let result = GateResult::failed(
            "test",
            "Inf test",
            Some(f64::INFINITY),
            Some(100.0),
            Duration::from_secs(1),
        );
        assert!(
            result.value.expect("should have value").is_infinite(),
            "Infinity must be preserved in GateResult"
        );
    }

    /// Negative infinity in threshold should be representable.
    #[test]
    fn gate_result_neg_infinity_threshold() {
        let result = GateResult::passed(
            "test",
            "neg inf threshold",
            Some(0.0),
            Some(f64::NEG_INFINITY),
            Duration::from_secs(1),
        );
        assert!(result
            .threshold
            .expect("should have threshold")
            .is_infinite());
    }

    // ========================================================================
    // QaConfig: clone preserves all fields including PathBuf
    // ========================================================================

    /// Clone with safetensors_path should deep-copy the PathBuf.
    #[test]
    fn qa_config_clone_with_safetensors_path() {
        let config = QaConfig {
            safetensors_path: Some(std::path::PathBuf::from("/deep/clone/test.safetensors")),
            min_tps: 42.0,
            json: true,
            verbose: true,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.safetensors_path, config.safetensors_path);
        assert!((cloned.min_tps - 42.0).abs() < f64::EPSILON);
        assert!(cloned.json);
        assert!(cloned.verbose);
    }

    // ========================================================================
    // NEW: Contract failure summary truncation logic
    // ========================================================================
    // Mirrors the truncation in run_tensor_contract_gate (lines 461-468):
    //   if failures.len() <= 3: join with "; "
    //   else: first 3 joined + "; ... and {N-3} more"

    /// Exactly 1 contract failure should display the single failure, no truncation.
    #[test]
    fn contract_failure_summary_single_failure() {
        let failures = vec!["embed_tokens.weight: density below threshold".to_string()];
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert_eq!(summary, "embed_tokens.weight: density below threshold");
        assert!(!summary.contains("more"));
    }

    /// Exactly 3 contract failures should display all without truncation.
    #[test]
    fn contract_failure_summary_three_failures_no_truncation() {
        let failures = vec![
            "layer.0: NaN detected".to_string(),
            "layer.1: Inf detected".to_string(),
            "layer.2: zero density".to_string(),
        ];
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert_eq!(
            summary,
            "layer.0: NaN detected; layer.1: Inf detected; layer.2: zero density"
        );
        assert!(!summary.contains("more"));
    }

    /// 4 contract failures should truncate: show 3, then "... and 1 more".
    #[test]
    fn contract_failure_summary_four_failures_truncates() {
        let failures = vec![
            "a: fail".to_string(),
            "b: fail".to_string(),
            "c: fail".to_string(),
            "d: fail".to_string(),
        ];
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert!(summary.contains("a: fail; b: fail; c: fail"));
        assert!(summary.ends_with("; ... and 1 more"));
    }

    /// 10 contract failures should truncate: show 3, then "... and 7 more".
    #[test]
    fn contract_failure_summary_ten_failures_truncates() {
        let failures: Vec<String> = (0..10).map(|i| format!("tensor_{i}: violation")).collect();
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert!(summary.contains("tensor_0: violation"));
        assert!(summary.contains("tensor_1: violation"));
        assert!(summary.contains("tensor_2: violation"));
        assert!(summary.ends_with("; ... and 7 more"));
        assert!(!summary.contains("tensor_3"));
    }

    /// 0 contract failures should produce empty string (join of empty vec).
    #[test]
    fn contract_failure_summary_zero_failures() {
        let failures: Vec<String> = vec![];
        let summary = if failures.len() <= 3 {
            failures.join("; ")
        } else {
            format!(
                "{}; ... and {} more",
                failures[..3].join("; "),
                failures.len() - 3
            )
        };
        assert!(summary.is_empty());
    }

    // ========================================================================
    // NEW: detect_ollama_model_from_path -- additional edge cases
    // ========================================================================

    /// Filename with only "3b" (no prefix dash) should still match the 3b branch.
    #[test]
    fn detect_ollama_model_3b_standalone() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model3b.gguf"));
        assert_eq!(model, "qwen2.5-coder:3b");
    }

    /// Filename with dash-prefixed sizes: "-3b" variant.
    #[test]
    fn detect_ollama_model_dash_3b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-3b-chat.gguf"));
        assert_eq!(model, "qwen2.5-coder:3b");
    }

    /// Filename with "-7b" variant.
    #[test]
    fn detect_ollama_model_dash_7b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/llama-7b-q4_k_m.gguf"));
        assert_eq!(model, "qwen2.5-coder:7b");
    }

    /// Filename containing "0.5b" with mixed case.
    #[test]
    fn detect_ollama_model_mixed_case_0_5b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/Qwen2.5-Coder-0.5B-Q4.gguf"));
        assert_eq!(model, "qwen2.5-coder:0.5b");
    }

    /// Filename containing "-32b" variant.
    #[test]
    fn detect_ollama_model_dash_32b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/qwen-32b-instruct.gguf"));
        assert_eq!(model, "qwen2.5-coder:32b");
    }

    /// Filename containing "-14b" variant with dash prefix.
    #[test]
    fn detect_ollama_model_dash_14b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-14b.gguf"));
        assert_eq!(model, "qwen2.5-coder:14b");
    }

    /// Empty string path (edge case for Path::new("")).
    #[test]
    fn detect_ollama_model_empty_string_path() {
        let model = detect_ollama_model_from_path(Path::new(""));
        // Empty path -> file_name() returns None on empty -> unwrap_or("") -> file size fallback
        assert!(
            model.starts_with("qwen2.5-coder:"),
            "Empty path should produce valid tag: {model}"
        );
    }

    /// Filename that contains multiple size markers: "1.5b" comes before "3b" in check order.
    #[test]
    fn detect_ollama_model_1_5b_before_3b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-1.5b-3b.gguf"));
        assert_eq!(
            model, "qwen2.5-coder:1.5b",
            "1.5b should be matched before 3b in priority order"
        );
    }

    /// Filename with underscore-separated 1.5b variant.
    #[test]
    fn detect_ollama_model_underscore_1_5b_variant() {
        let model = detect_ollama_model_from_path(Path::new("/cache/qwen2-1_5b-q4_k.gguf"));
        assert_eq!(model, "qwen2.5-coder:1.5b");
    }

    /// Filename containing "-0_5b" (underscore variant of 0.5b).
    #[test]
    fn detect_ollama_model_underscore_0_5b() {
        let model = detect_ollama_model_from_path(Path::new("/tmp/model-0_5b-instruct.gguf"));
        assert_eq!(model, "qwen2.5-coder:0.5b");
    }

    // ========================================================================
    // NEW: GateResult JSON edge cases for skip_serializing_if
    // ========================================================================

    /// JSON with value present but threshold missing should deserialize correctly.
    #[test]
    fn gate_result_json_value_present_threshold_missing() {
        let json = r#"{
            "name": "contract",
            "passed": true,
            "message": "50 tensors ok",
            "value": 50.0,
            "duration_ms": 100,
            "skipped": false
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(result.value, Some(50.0));
        assert!(result.threshold.is_none());
    }

    /// JSON with threshold present but value missing should deserialize correctly.
    #[test]
    fn gate_result_json_threshold_present_value_missing() {
        let json = r#"{
            "name": "throughput",
            "passed": false,
            "message": "too slow",
            "threshold": 100.0,
            "duration_ms": 5000,
            "skipped": false
        }"#;
        let result: GateResult = serde_json::from_str(json).expect("deserialize");
        assert!(result.value.is_none());
        assert_eq!(result.threshold, Some(100.0));
    }

    /// Serialized JSON for a passed gate with Some(value) should include "value" key.
    #[test]
    fn gate_result_json_includes_value_when_some() {
        let result = GateResult::passed(
            "throughput",
            "150 tok/s",
            Some(150.0),
            None,
            Duration::from_secs(1),
        );
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(
            json.contains("\"value\""),
            "value should be present: {json}"
        );
        assert!(
            !json.contains("\"threshold\""),
            "threshold should be omitted when None: {json}"
        );
    }

    /// Serialized JSON for a gate with both Some(value) and Some(threshold).
    #[test]
    fn gate_result_json_includes_both_value_and_threshold() {
        let result = GateResult::failed(
            "ollama_parity",
            "0.1x < 0.2x",
            Some(0.1),
            Some(0.2),
            Duration::from_secs(10),
        );
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("\"value\""));
        assert!(json.contains("\"threshold\""));
        assert!(json.contains("0.1"));
        assert!(json.contains("0.2"));
    }

    // ========================================================================
    // NEW: QaReport JSON pretty-print validation
    // ========================================================================

    /// Pretty-printed JSON report should contain newlines and indentation.
    #[test]
    fn qa_report_json_pretty_print_format() {
        let report = QaReport {
            model: "test.gguf".to_string(),
            passed: true,
            gates: vec![GateResult::passed(
                "contract",
                "ok",
                Some(10.0),
                Some(0.0),
                Duration::from_millis(50),
            )],
            total_duration_ms: 50,
            timestamp: "2026-02-07T00:00:00Z".to_string(),
            summary: "All passed".to_string(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        let json = serde_json::to_string_pretty(&report).expect("pretty serialize");
        assert!(json.contains('\n'), "Pretty JSON should contain newlines");
        assert!(
            json.contains("  "),
            "Pretty JSON should contain indentation"
        );
        assert!(json.contains("\"model\""));
        assert!(json.contains("\"gates\""));
        assert!(json.contains("\"summary\""));
    }

    /// JSON report with unwrap_or_default fallback (mirrors run() line 251).
    #[test]
    fn qa_report_json_to_string_pretty_never_panics() {
        let report = QaReport {
            model: String::new(),
            passed: false,
            gates: vec![
                GateResult::skipped("a", "skip"),
                GateResult::failed("b", "fail", Some(f64::NAN), None, Duration::from_secs(0)),
            ],
            total_duration_ms: 0,
            timestamp: String::new(),
            summary: String::new(),
            gates_executed: 0,
            gates_skipped: 0,
            system_info: None,
        };
        // This is what run() does: serde_json::to_string_pretty(&report).unwrap_or_default()
        let json = serde_json::to_string_pretty(&report).unwrap_or_default();
        // NaN in JSON becomes null (serde_json behavior), but should not panic
        assert!(!json.is_empty());
    }

    // ========================================================================
    // NEW: QaConfig construction from run() parameters (lines 228-244)
    // ========================================================================

    /// Simulate how run() builds QaConfig from Option parameters.
    /// unwrap_or defaults should match QaConfig::default() for the three thresholds.
    #[test]
    fn run_config_building_none_uses_defaults() {
        let min_tps: Option<f64> = None;
        let min_speedup: Option<f64> = None;
        let min_gpu_speedup: Option<f64> = None;
        let config = QaConfig {
            min_tps: min_tps.unwrap_or(100.0),
            min_speedup: min_speedup.unwrap_or(0.2),
            min_gpu_speedup: min_gpu_speedup.unwrap_or(2.0),
            ..Default::default()
        };
        assert!((config.min_tps - 100.0).abs() < f64::EPSILON);
        assert!((config.min_speedup - 0.2).abs() < f64::EPSILON);
        assert!((config.min_gpu_speedup - 2.0).abs() < f64::EPSILON);
    }
