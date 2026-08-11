
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
    fn verify_output_rejects_qwen2_05b_observed_gibberish() {
        // Captured from `apr run qwen2.5-coder-0.5b-instruct-imported.apr --prompt "Hello"`
        // (2026-05-04). Pre-fix this gibberish was passing the Golden Output gate because the
        // hardcoded garbage list missed CJK/Polish/diacritic byte fragments.
        let observed =
            "ìłľê°Ģ udaÅĤo udaÅĤoá¿Ĩëĸ» Ãĥå udaÅĤo ÃĥÂłÂłÂłÂł zwiÄħzku";
        let result = verify_output(observed, "FALSIFY-QA-GIBBERISH-001", &["Hello", "!"]);
        assert!(
            matches!(result, OutputVerification::Fail { .. }),
            "Qwen2-0.5B observed gibberish must be rejected"
        );
    }

    #[test]
    fn verify_output_rejects_repeated_fragment() {
        let result = verify_output(
            "udaÅĤo udaÅĤo udaÅĤo udaÅĤo end",
            "FALSIFY-QA-GIBBERISH-002",
            &["end"],
        );
        assert!(
            matches!(result, OutputVerification::Fail { .. }),
            "4x-repeated fragment must trip the loop-pathology signal"
        );
    }

    #[test]
    fn verify_output_accepts_normal_english() {
        let result = verify_output(
            "The answer is 4. Hello world!",
            "FALSIFY-QA-GIBBERISH-003",
            &["4"],
        );
        assert!(
            matches!(result, OutputVerification::Pass),
            "Normal ASCII English must pass"
        );
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
    // strip_thinking_blocks Tests (GH-279-4)
    // ========================================================================

    #[test]
    fn strip_thinking_no_tags() {
        // Non-thinking model output: passthrough unchanged
        assert_eq!(strip_thinking_blocks("The answer is 4."), "The answer is 4.");
    }

    #[test]
    fn strip_thinking_complete_block() {
        // Thinking block followed by answer
        let input = "<think>Let me calculate 2+2. That's 4.</think>4";
        assert_eq!(strip_thinking_blocks(input), "4");
    }

    #[test]
    fn strip_thinking_unclosed() {
        // Model ran out of tokens during reasoning (unclosed <think>)
        let input = "<think>Let me think about this carefully...";
        assert_eq!(strip_thinking_blocks(input), "");
    }

    #[test]
    fn strip_thinking_multiline() {
        // Multi-line thinking block
        let input = "<think>\nStep 1: 2+2\nStep 2: =4\n</think>\nThe answer is 4.";
        assert_eq!(strip_thinking_blocks(input), "The answer is 4.");
    }

    #[test]
    fn strip_thinking_multiple_blocks() {
        // Multiple thinking blocks
        let input = "<think>first thought</think>Hello <think>second thought</think>world";
        assert_eq!(strip_thinking_blocks(input), "Hello world");
    }

    #[test]
    fn strip_thinking_preserves_surrounding() {
        // Only think tags are stripped, surrounding content preserved
        let input = "Before <think>reasoning</think> After";
        assert_eq!(strip_thinking_blocks(input), "Before  After");
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

    // ========================================================================
    // GH-279-2: strip_quant_suffix — Qwen3 patterns
    // ========================================================================

    #[test]
    fn strip_quant_suffix_qwen3_q4k() {
        assert_eq!(strip_quant_suffix("qwen3-8b-q4k"), "qwen3-8b");
    }

    #[test]
    fn strip_quant_suffix_qwen3_q4_k_m() {
        assert_eq!(strip_quant_suffix("qwen3-8b-q4_k_m"), "qwen3-8b");
    }

    #[test]
    fn strip_quant_suffix_qwen3_q6k() {
        assert_eq!(strip_quant_suffix("qwen3-8b-q6k"), "qwen3-8b");
    }

    #[test]
    fn strip_quant_suffix_qwen3_f16() {
        assert_eq!(strip_quant_suffix("qwen3-8b-f16"), "qwen3-8b");
    }

    #[test]
    fn strip_quant_suffix_no_suffix() {
        assert_eq!(strip_quant_suffix("qwen3-8b"), "qwen3-8b");
    }

    #[test]
    fn strip_quant_suffix_qwen3_q8_0() {
        assert_eq!(strip_quant_suffix("qwen3-8b-q8_0"), "qwen3-8b");
    }

    // ========================================================================
    // GH-279-2: discover_apr_cache — tempdir tests
    // ========================================================================

    #[test]
    fn discover_apr_cache_returns_none_for_nonexistent_model() {
        // When the base_name doesn't match any repo in ~/.apr/cache/hf/,
        // should return None
        let result = discover_apr_cache("nonexistent-model-xyzzy-9999");
        assert!(result.is_none());
    }

    #[test]
    fn find_sharded_safetensors_with_index_and_shard() {
        let tmp = tempfile::tempdir().expect("create tempdir");

        // Create sharded index + shard file
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            r#"{"weight_map": {}}"#,
        )
        .expect("write index");
        std::fs::write(
            tmp.path().join("model-00001-of-00004.safetensors"),
            b"fake shard",
        )
        .expect("write shard");

        let shard = find_sharded_safetensors(tmp.path());
        assert!(shard.is_some(), "Should find sharded safetensors");
        let shard_path = shard.expect("shard exists");
        assert!(
            shard_path.to_string_lossy().ends_with(".safetensors"),
            "Should find a .safetensors file"
        );
    }

    #[test]
    fn find_sharded_safetensors_returns_none_without_index() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        // No index file — just a shard
        std::fs::write(
            tmp.path().join("model-00001-of-00002.safetensors"),
            b"data",
        )
        .expect("write shard");

        let result = find_sharded_safetensors(tmp.path());
        assert!(
            result.is_none(),
            "Should return None without index.json present"
        );
    }

    #[test]
    fn discover_sibling_subdir_finds_single_model() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let subdir = tmp.path().join("qwen3-8b");
        std::fs::create_dir_all(&subdir).expect("create subdir");
        std::fs::write(subdir.join("model.safetensors"), b"fake model").expect("write model");

        let result = discover_sibling_subdir(tmp.path(), "qwen3-8b");
        assert!(result.is_some(), "Should find model.safetensors in subdir");
    }

    #[test]
    fn discover_sibling_subdir_returns_none_for_missing_dir() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let result = discover_sibling_subdir(tmp.path(), "nonexistent");
        assert!(result.is_none());
    }

    // ====================================================================
    // PTX Parity gate: zero comparisons is not a pass
    // ====================================================================

    /// Without the `cuda` feature `validate_all_kernel_pairs` returns an empty
    /// report. `all_passed()` is `failed == 0`, vacuously true when nothing was
    /// compared, so the gate reported "0/0 kernel pairs passed PTX parity" as
    /// PASS — a gate that could not fail by construction.
    #[cfg(feature = "inference")]
    #[test]
    fn ptx_gate_skips_when_no_kernel_pairs_were_compared() {
        let empty = realizar::ptx_parity::PtxParityReport {
            results: Vec::new(),
            total: 0,
            passed: 0,
            failed: 0,
        };
        assert!(
            empty.all_passed(),
            "precondition: an empty report still claims all_passed()"
        );

        let gate = ptx_gate_result(&empty, std::time::Duration::from_millis(1));
        assert!(gate.skipped, "0/0 comparisons must be reported as SKIP");
        assert!(
            !gate.message.contains("0/0 kernel pairs passed"),
            "must not claim a pass it did not earn, got: {}",
            gate.message
        );
    }

    #[cfg(feature = "inference")]
    #[test]
    fn ptx_gate_still_judges_real_comparisons() {
        let ok = realizar::ptx_parity::PtxParityReport {
            results: Vec::new(),
            total: 6,
            passed: 6,
            failed: 0,
        };
        let gate = ptx_gate_result(&ok, std::time::Duration::from_millis(1));
        assert!(!gate.skipped, "real comparisons must produce a verdict");
        assert!(gate.passed, "6/6 must PASS");

        let bad = realizar::ptx_parity::PtxParityReport {
            results: Vec::new(),
            total: 6,
            passed: 5,
            failed: 1,
        };
        let gate = ptx_gate_result(&bad, std::time::Duration::from_millis(1));
        assert!(!gate.skipped);
        assert!(!gate.passed, "1 failing kernel pair must FAIL");
    }

    // ====================================================================
    // Capability Match gate: no GPU claims from a build with no GPU backend
    // ====================================================================

    /// On a build without `cuda`, the gate asserted "all N required ops
    /// supported by GPU" — a claim about kernels the binary cannot reach.
    #[cfg(all(feature = "inference", not(feature = "cuda")))]
    #[test]
    fn capability_gate_makes_no_gpu_claim_without_cuda() {
        // A GGUF carrying general.architecture — without it the gate short
        // circuits on "missing architecture metadata" and never reaches the
        // GPU claim, so the test would prove nothing.
        let mut tmp = NamedTempFile::new().expect("tempfile");
        tmp.write_all(b"GGUF").expect("magic");
        tmp.write_all(&3u32.to_le_bytes()).expect("version");
        tmp.write_all(&0u64.to_le_bytes()).expect("tensor count");
        tmp.write_all(&1u64.to_le_bytes()).expect("kv count");
        let key = b"general.architecture";
        tmp.write_all(&(key.len() as u64).to_le_bytes())
            .expect("key len");
        tmp.write_all(key).expect("key");
        tmp.write_all(&8u32.to_le_bytes()).expect("STRING type tag");
        let value = b"qwen2";
        tmp.write_all(&(value.len() as u64).to_le_bytes())
            .expect("value len");
        tmp.write_all(value).expect("value");
        tmp.flush().expect("flush");

        let config = QaConfig::default();
        let gate = crate::commands::qa_capability::run_capability_gate(tmp.path(), &config)
            .expect("gate runs");
        assert!(
            !gate.message.contains("supported by GPU"),
            "a CPU-only build must not assert GPU support, got: {}",
            gate.message
        );
        // Proves the fixture got past architecture extraction — otherwise the
        // gate short circuits earlier and the assertion above is vacuous.
        assert!(
            gate.skipped && gate.message.contains("Requires 'inference' and 'cuda'"),
            "expected the cuda-feature SKIP, got: {}",
            gate.message
        );
    }
