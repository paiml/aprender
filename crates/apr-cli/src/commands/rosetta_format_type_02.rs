
    #[test]
    fn test_format_type_from_magic_apr() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&[0u8; 8]); // Padding for 8+ bytes
        std::fs::write(&file_path, &data).expect("write");
        let result = FormatType::from_magic(&file_path);
        assert!(result.is_ok());
        assert_eq!(result.expect("format"), FormatType::Apr);
    }

    #[test]
    fn test_format_type_from_magic_unknown() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"UNKNOWN!MAGIC").expect("write");
        let result = FormatType::from_magic(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_type_from_magic_nonexistent() {
        let result = FormatType::from_magic(Path::new("/nonexistent/file"));
        assert!(result.is_err());
    }

    #[test]
    fn test_format_type_from_magic_too_short() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"GGU").expect("write"); // Too short for magic read
        let result = FormatType::from_magic(&file_path);
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: RosettaStone construction tests
    // ========================================================================

    #[test]
    fn test_rosetta_stone_new() {
        let rs = RosettaStone::new();
        let debug_str = format!("{rs:?}");
        assert!(debug_str.contains("RosettaStone"));
    }

    #[test]
    fn test_rosetta_stone_with_options() {
        let opts = ConversionOptions {
            quantization: Some("int8".to_string()),
            ..Default::default()
        };
        let rs = RosettaStone::with_options(opts);
        let debug_str = format!("{rs:?}");
        assert!(debug_str.contains("RosettaStone"));
    }

    // ========================================================================
    // NEW: dequantize multi-block tests
    // ========================================================================

    #[test]
    fn test_dequantize_q4k_multiple_blocks() {
        // Two complete Q4_K blocks
        let data = vec![0u8; 288]; // 2 * 144
        let result = dequantize_q4k_for_stats(&data, 512);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_dequantize_q6k_multiple_blocks() {
        // Two complete Q6_K blocks
        let data = vec![0u8; 420]; // 2 * 210
        let result = dequantize_q6k_for_stats(&data, 512);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_dequantize_q4k_partial_last_block() {
        // One and a half blocks - second block is incomplete
        let data = vec![0u8; 200]; // 144 + 56 (incomplete second block)
        let result = dequantize_q4k_for_stats(&data, 512);
        // Only first block should produce output
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6k_partial_last_block() {
        let data = vec![0u8; 300]; // 210 + 90 (incomplete second block)
        let result = dequantize_q6k_for_stats(&data, 512);
        assert_eq!(result.len(), 256);
    }

    // ========================================================================
    // NEW: Comprehensive convert flow tests
    // ========================================================================

    #[test]
    fn test_run_convert_with_quantize_and_verify() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(
            source.path(),
            target.path(),
            Some("fp16"),
            true,
            false,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_convert_json_with_quantize() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(
            source.path(),
            target.path(),
            Some("int4"),
            false,
            true,
            None,
        );
        assert!(result.is_err());
    }

    // ====================================================================
    // Coverage-boost tests: normalize_tensor_name exhaustive cases
    // ====================================================================

    #[test]
    fn test_normalize_tensor_name_gguf_attn_v_bias() {
        // GGUF bias tensors should also normalize
        assert_eq!(normalize_tensor_name("blk.2.attn_v.bias"), "2.v_proj.bias");
    }

    #[test]
    fn test_normalize_tensor_name_gguf_attn_output_bias() {
        assert_eq!(
            normalize_tensor_name("blk.0.attn_output.bias"),
            "0.o_proj.bias"
        );
    }

    #[test]
    fn test_normalize_tensor_name_apr_mlp_down_proj_bias() {
        assert_eq!(
            normalize_tensor_name("model.layers.3.mlp.down_proj.bias"),
            "3.down_proj.bias"
        );
    }

    #[test]
    fn test_normalize_tensor_name_gguf_norm_weights_both_types() {
        // attn_norm → input_layernorm, ffn_norm → post_attention_layernorm
        assert_eq!(
            normalize_tensor_name("blk.15.attn_norm.bias"),
            "15.input_layernorm.bias"
        );
        assert_eq!(
            normalize_tensor_name("blk.15.ffn_norm.bias"),
            "15.post_attention_layernorm.bias"
        );
    }

    #[test]
    fn test_normalize_tensor_name_double_prefix_model_layers() {
        // "model.layers." prefix is stripped as "model." then "layers."
        let result = normalize_tensor_name("model.layers.7.self_attn.v_proj.weight");
        assert_eq!(result, "7.v_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_only_model_prefix() {
        // Only "model." prefix, no "layers."
        let result = normalize_tensor_name("model.norm.weight");
        assert_eq!(result, "norm.weight");
    }

    #[test]
    fn test_normalize_tensor_name_only_blk_prefix() {
        // "blk." prefix with a simple suffix
        let result = normalize_tensor_name("blk.0.token_embd.weight");
        assert_eq!(result, "0.embed_tokens.weight");
    }

    #[test]
    fn test_normalize_tensor_name_output_weight_exact_match() {
        // "output.weight" should map to "lm_head.weight" (exact match)
        assert_eq!(normalize_tensor_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_normalize_tensor_name_output_norm_weight() {
        // "output_norm.weight" → "norm.weight" via replacement
        assert_eq!(normalize_tensor_name("output_norm.weight"), "norm.weight");
    }

    #[test]
    fn test_normalize_tensor_name_multiple_self_attn_occurrences() {
        // str::replace scans left-to-right without re-scanning replacements.
        // "0.self_attn.self_attn.q_proj.weight" → first ".self_attn." match
        // yields "0.self_attn.q_proj.weight" — the overlapping second occurrence
        // collapses but one "self_attn" remains as a name prefix, not a dotted segment.
        let result = normalize_tensor_name("model.layers.0.self_attn.self_attn.q_proj.weight");
        assert_eq!(result, "0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_token_embd_bias() {
        assert_eq!(
            normalize_tensor_name("token_embd.bias"),
            "embed_tokens.bias"
        );
    }

    #[test]
    fn test_normalize_tensor_name_preserves_layer_numbers() {
        // Layer numbers should be preserved exactly
        for layer_num in [0, 1, 10, 99, 127] {
            let gguf = format!("blk.{layer_num}.attn_q.weight");
            let apr = format!("model.layers.{layer_num}.self_attn.q_proj.weight");
            assert_eq!(normalize_tensor_name(&gguf), normalize_tensor_name(&apr));
        }
    }

    #[test]
    fn test_normalize_tensor_name_all_gguf_ffn_variants() {
        // Verify all 3 FFN mappings with different layer numbers
        assert_eq!(
            normalize_tensor_name("blk.10.ffn_gate.weight"),
            "10.gate_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.10.ffn_up.weight"),
            "10.up_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.10.ffn_down.weight"),
            "10.down_proj.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_cross_format_embedding_match() {
        // Both formats should match for embedding
        assert_eq!(
            normalize_tensor_name("token_embd.weight"),
            normalize_tensor_name("model.embed_tokens.weight")
        );
    }

    #[test]
    fn test_normalize_tensor_name_cross_format_lm_head_match() {
        // GGUF "output.weight" and APR "lm_head.weight" should match
        assert_eq!(
            normalize_tensor_name("output.weight"),
            normalize_tensor_name("lm_head.weight")
        );
    }

    #[test]
    fn test_normalize_tensor_name_cross_format_norm_match() {
        // GGUF "output_norm.weight" and APR "model.norm.weight" should match
        assert_eq!(
            normalize_tensor_name("output_norm.weight"),
            normalize_tensor_name("model.norm.weight")
        );
    }

    // ====================================================================
    // Coverage-boost tests: is_transposed_dims exhaustive edge cases
    // ====================================================================

    #[test]
    fn test_is_transposed_dims_large_dimensions() {
        assert!(is_transposed_dims(&[4096, 11008], &[11008, 4096]));
    }

    #[test]
    fn test_is_transposed_dims_one_dimension_is_one() {
        // [1, 768] vs [768, 1] - technically transposed
        assert!(is_transposed_dims(&[1, 768], &[768, 1]));
    }

    #[test]
    fn test_is_transposed_dims_both_one() {
        // [1, 1] vs [1, 1] - square, so NOT transposed
        assert!(!is_transposed_dims(&[1, 1], &[1, 1]));
    }

    #[test]
    fn test_is_transposed_dims_single_element_each() {
        assert!(!is_transposed_dims(&[5], &[5]));
    }

    #[test]
    fn test_is_transposed_dims_4d_tensors() {
        // 4D should always return false
        assert!(!is_transposed_dims(&[2, 3, 4, 5], &[5, 4, 3, 2]));
    }

    #[test]
    fn test_is_transposed_dims_mismatched_ndims() {
        assert!(!is_transposed_dims(&[768], &[768, 3072]));
        assert!(!is_transposed_dims(&[768, 3072], &[768]));
    }

    #[test]
    fn test_is_transposed_dims_completely_different() {
        // Shapes where neither dimension matches when swapped
        assert!(!is_transposed_dims(&[100, 200], &[300, 400]));
    }

    #[test]
    fn test_is_transposed_dims_partially_matching() {
        // Only one dim matches: [768, 3072] vs [3072, 1024]
        assert!(!is_transposed_dims(&[768, 3072], &[3072, 1024]));
    }

    #[test]
    fn test_is_transposed_dims_zero_dimensions() {
        // [0, 768] vs [768, 0]: a[0]==b[1] (0==0) AND a[1]==b[0] (768==768),
        // and shapes differ, so the function correctly reports them as transposed.
        assert!(is_transposed_dims(&[0, 768], &[768, 0]));
    }

    // ====================================================================
    // Coverage-boost tests: truncate_path additional cases
    // ====================================================================

    #[test]
    fn test_truncate_path_exactly_at_max_len() {
        let path = "abcde".to_string(); // 5 chars
        assert_eq!(truncate_path(path, 5), "abcde");
    }

    #[test]
    fn test_truncate_path_one_over_max_len() {
        let path = "abcdef".to_string(); // 6 chars
        let result = truncate_path(path, 5);
        // Should be "...ef" (3 dots + last 2 chars = 5 chars)
        assert!(result.starts_with("..."));
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_truncate_path_very_long() {
        let path = "a".repeat(200);
        let result = truncate_path(path, 20);
        assert_eq!(result.len(), 20);
        assert!(result.starts_with("..."));
    }

    #[test]
    fn test_truncate_path_max_len_three() {
        // Edge case: max_len == 3 means "..." fits exactly
        let path = "abcdef".to_string();
        let result = truncate_path(path, 3);
        assert_eq!(result, "...");
    }

    #[test]
    fn test_truncate_path_preserves_file_extension_when_possible() {
        let path = "/very/long/path/to/model.gguf".to_string();
        let result = truncate_path(path, 15);
        // Should end with "model.gguf" since it takes from end
        assert!(result.ends_with("model.gguf"));
    }

    // ====================================================================
    // Coverage-boost tests: strip_ansi additional patterns
    // ====================================================================

    #[test]
    fn test_strip_ansi_256_color() {
        // 256-color code: \x1b[38;5;196m
        let text = "\x1b[38;5;196mRed\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Red");
    }

    #[test]
    fn test_strip_ansi_cursor_movement() {
        // Cursor movement codes
        let text = "\x1b[2AUp two lines\x1b[3BDown three";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Up two linesDown three");
    }

    #[test]
    fn test_strip_ansi_mixed_content_and_escapes() {
        let text = "start\x1b[31m red \x1b[32m green \x1b[0mend";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "start red  green end");
    }

    #[test]
    fn test_strip_ansi_unicode_preserved() {
        let text = "\x1b[1mUnicode: \u{2713} \u{2717}\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Unicode: \u{2713} \u{2717}");
    }

    // ====================================================================
    // Coverage-boost tests: compute_tensor_stats more edge cases
    // ====================================================================

    #[test]
    fn test_compute_tensor_stats_two_values() {
        let data = vec![0.0, 10.0];
        let (mean, std, min, max, _p5, _p25, p50, _p75, _p95, _, _, _, _) =
            compute_tensor_stats(&data);
        assert!((mean - 5.0).abs() < 0.001);
        assert!((std - 5.0).abs() < 0.001);
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 10.0).abs() < 0.001);
        // Median of 2 values: p50 is index-based so it's the 0th value (0.0)
        assert!(p50 >= 0.0 && p50 <= 10.0);
    }

    #[test]
    fn test_compute_tensor_stats_large_uniform() {
        // 1000 identical values
        let data = vec![42.0; 1000];
        let (mean, std, min, max, p5, p25, p50, p75, p95, nan, inf, zero_frac, _) =
            compute_tensor_stats(&data);
        assert!((mean - 42.0).abs() < 0.001);
        assert!(std < 0.001); // No variance
        assert!((min - 42.0).abs() < 0.001);
        assert!((max - 42.0).abs() < 0.001);
        assert!((p5 - 42.0).abs() < 0.001);
        assert!((p25 - 42.0).abs() < 0.001);
        assert!((p50 - 42.0).abs() < 0.001);
        assert!((p75 - 42.0).abs() < 0.001);
        assert!((p95 - 42.0).abs() < 0.001);
        assert_eq!(nan, 0);
        assert_eq!(inf, 0);
        assert!((zero_frac - 0.0).abs() < 0.001);
    }
