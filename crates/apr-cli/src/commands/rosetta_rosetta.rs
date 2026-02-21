
    // ========================================================================
    // RosettaCommands Tests
    // ========================================================================

    #[test]
    fn test_rosetta_commands_inspect_default() {
        // Test that the Inspect variant can be created
        let cmd = RosettaCommands::Inspect {
            file: PathBuf::from("model.gguf"),
            hexdump: false,
            json: false,
        };
        match cmd {
            RosettaCommands::Inspect {
                file,
                hexdump,
                json,
            } => {
                assert_eq!(file.to_string_lossy(), "model.gguf");
                assert!(!hexdump);
                assert!(!json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_convert() {
        let cmd = RosettaCommands::Convert {
            source: PathBuf::from("model.gguf"),
            target: PathBuf::from("model.apr"),
            quantize: None,
            verify: false,
            json: false,
            tokenizer: None,
        };
        match cmd {
            RosettaCommands::Convert {
                source,
                target,
                quantize,
                ..
            } => {
                assert_eq!(source.to_string_lossy(), "model.gguf");
                assert_eq!(target.to_string_lossy(), "model.apr");
                assert!(quantize.is_none());
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_chain() {
        let cmd = RosettaCommands::Chain {
            source: PathBuf::from("model.gguf"),
            formats: vec!["safetensors".to_string(), "apr".to_string()],
            work_dir: PathBuf::from("./work"),
            json: false,
        };
        match cmd {
            RosettaCommands::Chain { formats, .. } => {
                assert_eq!(formats.len(), 2);
                assert_eq!(formats[0], "safetensors");
                assert_eq!(formats[1], "apr");
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_verify() {
        let cmd = RosettaCommands::Verify {
            source: PathBuf::from("model.gguf"),
            intermediate: "safetensors".to_string(),
            tolerance: 1e-5,
            json: false,
        };
        match cmd {
            RosettaCommands::Verify {
                tolerance,
                intermediate,
                ..
            } => {
                assert_eq!(tolerance, 1e-5);
                assert_eq!(intermediate, "safetensors");
            }
            _ => panic!("Wrong command variant"),
        }
    }

    // ========================================================================
    // Helper Function Tests (PMAT Coverage - Internal Functions)
    // ========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        // f16 zero: 0x0000
        let bytes = [0x00, 0x00];
        let result = f16_to_f32(&bytes);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // f16 1.0: 0x3C00
        let bytes = [0x00, 0x3C];
        let result = f16_to_f32(&bytes);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        // f16 -1.0: 0xBC00
        let bytes = [0x00, 0xBC];
        let result = f16_to_f32(&bytes);
        assert!((result + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_small_value() {
        // f16 0.5: 0x3800
        let bytes = [0x00, 0x38];
        let result = f16_to_f32(&bytes);
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_normalize_tensor_name_basic() {
        let name = "model.layers.0.attention.q_proj.weight";
        let normalized = normalize_tensor_name(name);
        assert!(normalized.contains("attention"));
        assert!(normalized.contains("q_proj"));
    }

    #[test]
    fn test_normalize_tensor_name_empty() {
        let name = "";
        let normalized = normalize_tensor_name(name);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_normalize_tensor_name_with_numbers() {
        let name = "layer_123_weight";
        let normalized = normalize_tensor_name(name);
        assert!(!normalized.is_empty());
    }

    // GH-202: Cross-format tensor name normalization tests
    #[test]
    fn test_normalize_tensor_name_gguf_to_canonical() {
        // GGUF style: blk.N.attn_q.weight → N.q_proj.weight
        assert_eq!(
            normalize_tensor_name("blk.0.attn_q.weight"),
            "0.q_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.5.attn_k.weight"),
            "5.k_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.12.attn_v.weight"),
            "12.v_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.attn_output.weight"),
            "0.o_proj.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_apr_to_canonical() {
        // APR/HF style: model.layers.N.self_attn.q_proj.weight → N.q_proj.weight
        assert_eq!(
            normalize_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            "0.q_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.5.self_attn.k_proj.weight"),
            "5.k_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.12.self_attn.v_proj.weight"),
            "12.v_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.0.self_attn.o_proj.weight"),
            "0.o_proj.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_ffn_mapping() {
        // GGUF FFN → HF MLP
        assert_eq!(
            normalize_tensor_name("blk.0.ffn_gate.weight"),
            "0.gate_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.ffn_up.weight"),
            "0.up_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.ffn_down.weight"),
            "0.down_proj.weight"
        );

        // APR/HF MLP
        assert_eq!(
            normalize_tensor_name("model.layers.0.mlp.gate_proj.weight"),
            "0.gate_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.0.mlp.up_proj.weight"),
            "0.up_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.0.mlp.down_proj.weight"),
            "0.down_proj.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_layernorm() {
        // GGUF: attn_norm/ffn_norm → HF: input_layernorm/post_attention_layernorm
        assert_eq!(
            normalize_tensor_name("blk.0.attn_norm.weight"),
            "0.input_layernorm.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.ffn_norm.weight"),
            "0.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_embeddings() {
        // token_embd → embed_tokens
        assert_eq!(
            normalize_tensor_name("token_embd.weight"),
            "embed_tokens.weight"
        );
        // output_norm → norm
        assert_eq!(normalize_tensor_name("output_norm.weight"), "norm.weight");
        // output → lm_head
        assert_eq!(normalize_tensor_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_normalize_tensor_name_cross_format_match() {
        // Verify GGUF and APR/HF normalize to the SAME canonical form (GH-202 core fix)
        let gguf_name = "blk.3.attn_q.weight";
        let apr_name = "model.layers.3.self_attn.q_proj.weight";
        assert_eq!(
            normalize_tensor_name(gguf_name),
            normalize_tensor_name(apr_name)
        );

        let gguf_ffn = "blk.7.ffn_down.weight";
        let apr_ffn = "model.layers.7.mlp.down_proj.weight";
        assert_eq!(
            normalize_tensor_name(gguf_ffn),
            normalize_tensor_name(apr_ffn)
        );
    }

    #[test]
    fn test_is_transposed_dims_true() {
        let shape_a = vec![768, 3072];
        let shape_b = vec![3072, 768];
        assert!(is_transposed_dims(&shape_a, &shape_b));
    }

    #[test]
    fn test_is_transposed_dims_false_same() {
        let shape_a = vec![768, 3072];
        let shape_b = vec![768, 3072];
        assert!(!is_transposed_dims(&shape_a, &shape_b));
    }

    #[test]
    fn test_is_transposed_dims_different_ndims() {
        let shape_a = vec![768, 3072];
        let shape_b = vec![768, 3072, 1];
        assert!(!is_transposed_dims(&shape_a, &shape_b));
    }

    #[test]
    fn test_is_transposed_dims_1d() {
        let shape_a = vec![768];
        let shape_b = vec![768];
        assert!(!is_transposed_dims(&shape_a, &shape_b));
    }

    #[test]
    fn test_strip_ansi_no_codes() {
        let text = "Hello, World!";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Hello, World!");
    }

    #[test]
    fn test_strip_ansi_with_codes() {
        let text = "\x1b[31mRed Text\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Red Text");
    }

    #[test]
    fn test_strip_ansi_multiple_codes() {
        let text = "\x1b[1m\x1b[32mBold Green\x1b[0m Normal";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Bold Green Normal");
    }

    #[test]
    fn test_strip_ansi_empty() {
        let text = "";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "");
    }

    #[test]
    fn test_truncate_path_short() {
        let path = "/short/path".to_string();
        let truncated = truncate_path(path.clone(), 50);
        assert_eq!(truncated, path);
    }

    #[test]
    fn test_truncate_path_long() {
        let path = "/very/long/path/to/some/deeply/nested/file.txt".to_string();
        let truncated = truncate_path(path, 20);
        assert!(truncated.len() <= 23); // max_len + "..."
        assert!(truncated.contains("...") || truncated.len() <= 20);
    }

    #[test]
    fn test_truncate_path_exact_length() {
        let path = "exactly20characters!".to_string();
        let truncated = truncate_path(path, 20);
        assert!(truncated.len() <= 23);
    }

    #[test]
    fn test_get_role_threshold_embedding() {
        let threshold = get_role_threshold("model.embed_tokens.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_attention() {
        let threshold = get_role_threshold("model.layers.0.self_attn.q_proj.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_mlp() {
        let threshold = get_role_threshold("model.layers.0.mlp.gate_proj.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_norm() {
        let threshold = get_role_threshold("model.layers.0.input_layernorm.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_lm_head() {
        let threshold = get_role_threshold("lm_head.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_unknown() {
        let threshold = get_role_threshold("unknown_tensor_name");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_compute_tensor_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = compute_tensor_stats(&data);
        // Empty data should return NaN or zeros
        assert!(stats.0.is_nan() || stats.0 == 0.0); // mean
    }

    #[test]
    fn test_compute_tensor_stats_single_value() {
        let data = vec![5.0];
        let stats = compute_tensor_stats(&data);
        assert!((stats.0 - 5.0).abs() < 0.001); // mean = 5.0
        assert!(stats.1 == 0.0 || stats.1.is_nan()); // std = 0 for single value
    }

    #[test]
    fn test_compute_tensor_stats_multiple_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_tensor_stats(&data);
        assert!((stats.0 - 3.0).abs() < 0.001); // mean = 3.0
        assert!((stats.2 - 1.0).abs() < 0.001); // min = 1.0
        assert!((stats.3 - 5.0).abs() < 0.001); // max = 5.0
    }

    #[test]
    fn test_compute_tensor_stats_negative_values() {
        let data = vec![-5.0, -3.0, 0.0, 3.0, 5.0];
        let stats = compute_tensor_stats(&data);
        assert!((stats.0 - 0.0).abs() < 0.001); // mean = 0.0
        assert!((stats.2 - (-5.0)).abs() < 0.001); // min = -5.0
        assert!((stats.3 - 5.0).abs() < 0.001); // max = 5.0
    }

    #[test]
    fn test_dequantize_q4k_empty() {
        let data: Vec<u8> = vec![];
        let result = dequantize_q4k_for_stats(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6k_empty() {
        let data: Vec<u8> = vec![];
        let result = dequantize_q6k_for_stats(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_fingerprints_to_json_empty() {
        let fingerprints: Vec<TensorFingerprint> = vec![];
        let json = fingerprints_to_json(&fingerprints);
        // Returns JSON object with empty fingerprints array
        assert!(json.contains("fingerprints"));
    }
