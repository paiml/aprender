
    // ═══════════════════════════════════════════════════════════════════════════
    // Tests for rosetta_print_inference.rs functions
    // ═══════════════════════════════════════════════════════════════════════════

    // ── print_inference_diagnosis ──────────────────────────────────────────

    #[test]
    fn test_print_inference_diagnosis_zero_tokens() {
        // GH-188: zero tokens = inference failure
        print_inference_diagnosis(0, 0, 0.05, "text a", "text b");
    }

    #[test]
    fn test_print_inference_diagnosis_all_match() {
        print_inference_diagnosis(10, 0, 0.05, "hello world", "hello world");
    }

    #[test]
    fn test_print_inference_diagnosis_some_mismatch_within_tolerance() {
        // 2/10 = 20% mismatch, tolerance=30% => partial match
        print_inference_diagnosis(10, 2, 0.30, "hello world", "hello there");
    }

    #[test]
    fn test_print_inference_diagnosis_heavy_mismatch() {
        // 8/10 = 80% mismatch, tolerance=5% => inference mismatch
        print_inference_diagnosis(10, 8, 0.05, "foo", "bar");
    }

    #[test]
    fn test_print_inference_diagnosis_all_mismatch() {
        print_inference_diagnosis(5, 5, 0.05, "aaa", "bbb");
    }

    #[test]
    fn test_print_inference_diagnosis_text_a_has_content_b_empty() {
        // Model A produced text, model B nothing
        print_inference_diagnosis(5, 3, 0.05, "actual output text", "");
    }

    #[test]
    fn test_print_inference_diagnosis_text_b_has_content_a_empty() {
        // Model B produced text, model A nothing
        print_inference_diagnosis(5, 3, 0.05, "", "actual output text");
    }

    #[test]
    fn test_print_inference_diagnosis_toks_marker_filtered() {
        // Text with "tok/s" is considered non-content
        print_inference_diagnosis(5, 2, 0.05, "100 tok/s", "real text");
    }

    #[test]
    fn test_print_inference_diagnosis_both_empty() {
        print_inference_diagnosis(5, 0, 0.05, "", "");
    }

    #[test]
    fn test_print_inference_diagnosis_content_differs() {
        // Both have content but it differs
        print_inference_diagnosis(5, 0, 0.05, "The answer is 4", "The answer is 5");
    }

    // ── detect_quant_level_from_path ──────────────────────────────────────

    #[test]
    fn test_detect_quant_safetensors() {
        let path = Path::new("model.safetensors");
        let level = detect_quant_level_from_path(path);
        assert!(level.contains("unquantized"));
    }

    #[test]
    fn test_detect_quant_gguf_q4_k_m() {
        let path = Path::new("model-q4_k_m.gguf");
        let level = detect_quant_level_from_path(path);
        assert_eq!(level, "Q4_K_M");
    }

    #[test]
    fn test_detect_quant_gguf_q6_k() {
        let path = Path::new("model-q6_k.gguf");
        let level = detect_quant_level_from_path(path);
        assert_eq!(level, "Q6_K");
    }

    #[test]
    fn test_detect_quant_gguf_f16() {
        let path = Path::new("model-f16.gguf");
        let level = detect_quant_level_from_path(path);
        assert_eq!(level, "F16");
    }

    #[test]
    fn test_detect_quant_gguf_unknown() {
        let path = Path::new("model.gguf");
        let level = detect_quant_level_from_path(path);
        assert!(level.contains("unknown"));
    }

    #[test]
    fn test_detect_quant_apr_q4k() {
        let path = Path::new("model-q4k.apr");
        let level = detect_quant_level_from_path(path);
        assert_eq!(level, "Q4K");
    }

    #[test]
    fn test_detect_quant_apr_unknown() {
        let path = Path::new("model.apr");
        let level = detect_quant_level_from_path(path);
        assert!(level.contains("unknown"));
    }

    #[test]
    fn test_detect_quant_unknown_ext() {
        let path = Path::new("model.bin");
        let level = detect_quant_level_from_path(path);
        assert_eq!(level, "unknown");
    }

    // ── check_mixed_quant_warning ─────────────────────────────────────────

    #[test]
    fn test_check_mixed_quant_same_safetensors() {
        let a = Path::new("a.safetensors");
        let b = Path::new("b.safetensors");
        assert!(check_mixed_quant_warning(a, b).is_none());
    }

    #[test]
    fn test_check_mixed_quant_different() {
        let a = Path::new("model-q4_k_m.gguf");
        let b = Path::new("model.safetensors");
        let warning = check_mixed_quant_warning(a, b);
        assert!(warning.is_some());
        let msg = warning.expect("should have warning");
        assert!(msg.contains("F-GT-002"));
        assert!(msg.contains("Mixed quantization"));
    }

    #[test]
    fn test_check_mixed_quant_same_gguf() {
        let a = Path::new("model-q4_k_m.gguf");
        let b = Path::new("other-q4_k_m.gguf");
        assert!(check_mixed_quant_warning(a, b).is_none());
    }

    // ── match_quant_pattern ───────────────────────────────────────────────

    #[test]
    fn test_match_quant_pattern_found() {
        let result = match_quant_pattern("model-q4_k_m.gguf", &["q4_k_m", "q6_k"]);
        assert_eq!(result, Some("Q4_K_M".to_string()));
    }

    #[test]
    fn test_match_quant_pattern_not_found() {
        let result = match_quant_pattern("model.gguf", &["q4_k_m", "q6_k"]);
        assert!(result.is_none());
    }

    #[test]
    fn test_match_quant_pattern_first_match_wins() {
        let result = match_quant_pattern("model-q4_k_m-q6_k.gguf", &["q4_k_m", "q6_k"]);
        assert_eq!(result, Some("Q4_K_M".to_string()));
    }

    // ── print_diff_header (rosetta_diff_tensor.rs) ────────────────────────

    #[test]
    fn test_print_diff_header_equal_counts() {
        print_diff_header(Path::new("a.gguf"), Path::new("b.apr"), 100, 100);
    }

    #[test]
    fn test_print_diff_header_mismatched_counts() {
        print_diff_header(Path::new("a.gguf"), Path::new("b.apr"), 100, 90);
    }

    // ── print_diff_json_summary ───────────────────────────────────────────

    #[test]
    fn test_print_diff_json_summary_no_mismatches() {
        print_diff_json_summary(
            Path::new("a.gguf"),
            Path::new("b.gguf"),
            100,
            100,
            &[],
            &[],
            &[],
        );
    }

    #[test]
    fn test_print_diff_json_summary_with_mismatches() {
        let mismatches = vec![
            ("layer.0.weight".to_string(), vec![768, 3072], vec![3072, 768]),
            ("layer.1.weight".to_string(), vec![768, 3072], vec![3072, 768]),
        ];
        print_diff_json_summary(
            Path::new("a.gguf"),
            Path::new("b.apr"),
            50,
            48,
            &mismatches,
            &[("missing.bias".to_string(), vec![768])],
            &[],
        );
    }

    // ── print_diff_text_summary ───────────────────────────────────────────

    #[test]
    fn test_print_diff_text_summary_no_issues() {
        print_diff_text_summary(100, 100, &[], &[], &[]);
    }

    #[test]
    fn test_print_diff_text_summary_with_mismatches() {
        let mismatches = vec![
            ("weight.0".to_string(), vec![4, 8], vec![8, 4]),
        ];
        print_diff_text_summary(50, 50, &mismatches, &[], &[]);
    }

    #[test]
    fn test_print_diff_text_summary_with_missing() {
        print_diff_text_summary(
            50,
            45,
            &[],
            &[("bias.0".to_string(), vec![128])],
            &[("extra.0".to_string(), vec![256])],
        );
    }
