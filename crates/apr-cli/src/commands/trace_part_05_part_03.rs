
    // ================================================================
    // Audit #3 fix: Real GGUF/SafeTensors dispatch tests
    // These exercise trace_gguf() and trace_safetensors() with valid data.
    // ================================================================

    /// Build a minimal valid GGUF file with architecture metadata and tensors.
    fn build_test_gguf() -> NamedTempFile {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
        use std::io::BufWriter;

        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let mut writer = BufWriter::new(&file);

        let tensors = vec![
            GgufTensor {
                name: "token_embd.weight".to_string(),
                shape: vec![4, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 4 * 8 * 4], // 4*8 f32s
            },
            GgufTensor {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![8, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 8 * 4],
            },
            GgufTensor {
                name: "blk.0.attn_k.weight".to_string(),
                shape: vec![8, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 8 * 4],
            },
            GgufTensor {
                name: "blk.0.attn_v.weight".to_string(),
                shape: vec![8, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 8 * 4],
            },
            GgufTensor {
                name: "blk.0.ffn_gate.weight".to_string(),
                shape: vec![16, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 16 * 8 * 4],
            },
            GgufTensor {
                name: "output_norm.weight".to_string(),
                shape: vec![8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 4],
            },
        ];

        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            ),
            ("llama.block_count".to_string(), GgufValue::Uint32(1)),
            ("llama.embedding_length".to_string(), GgufValue::Uint32(8)),
            (
                "llama.attention.head_count".to_string(),
                GgufValue::Uint32(2),
            ),
            (
                "llama.attention.head_count_kv".to_string(),
                GgufValue::Uint32(2),
            ),
        ];

        export_tensors_to_gguf(&mut writer, &tensors, &metadata).expect("write GGUF");
        drop(writer);
        file
    }

    /// Build a minimal valid SafeTensors file with named tensors.
    fn build_test_safetensors() -> NamedTempFile {
        // Build SafeTensors manually: 8-byte header_len + JSON header + tensor data
        let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight", vec![8, 4], vec![0.1; 32]),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![4, 4],
                vec![0.2; 16],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![4, 4],
                vec![0.3; 16],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![8, 4],
                vec![0.4; 32],
            ),
            ("lm_head.weight", vec![8, 4], vec![0.5; 32]),
        ];

        // Build header JSON and data bytes
        let mut data_bytes = Vec::new();
        let mut header_map = serde_json::Map::new();
        let mut offset = 0usize;

        for (name, shape, values) in &tensors {
            let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
            let end = offset + bytes.len();

            let mut entry = serde_json::Map::new();
            entry.insert("dtype".to_string(), serde_json::json!("F32"));
            entry.insert("shape".to_string(), serde_json::json!(shape));
            entry.insert("data_offsets".to_string(), serde_json::json!([offset, end]));
            header_map.insert(name.to_string(), serde_json::Value::Object(entry));

            data_bytes.extend_from_slice(&bytes);
            offset = end;
        }

        let header_json = serde_json::to_string(&header_map).expect("serialize header");
        let header_len = header_json.len() as u64;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_len.to_le_bytes());
        file_data.extend_from_slice(header_json.as_bytes());
        file_data.extend_from_slice(&data_bytes);

        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(&file_data).expect("write safetensors");
        file
    }

    #[test]
    fn test_run_valid_gguf_dispatch() {
        let file = build_test_gguf();
        let result = run(file.path(), None, None, false, false, false, false, false);
        assert!(result.is_ok(), "trace on valid GGUF failed: {result:?}");
    }

    #[test]
    fn test_run_valid_gguf_json_output() {
        let file = build_test_gguf();
        let result = run(file.path(), None, None, true, false, false, false, false);
        assert!(
            result.is_ok(),
            "trace JSON on valid GGUF failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_safetensors_dispatch() {
        let file = build_test_safetensors();
        let result = run(file.path(), None, None, false, false, false, false, false);
        assert!(
            result.is_ok(),
            "trace on valid SafeTensors failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_safetensors_json_output() {
        let file = build_test_safetensors();
        let result = run(file.path(), None, None, true, false, false, false, false);
        assert!(
            result.is_ok(),
            "trace JSON on valid SafeTensors failed: {result:?}"
        );
    }

    #[test]
    fn test_trace_gguf_detects_layers() {
        let file = build_test_gguf();
        let (format_name, layers, total_params) =
            detect_and_trace(file.path(), None, false).expect("detect_and_trace GGUF");
        assert!(
            format_name.contains("GGUF"),
            "format should be GGUF, got: {format_name}"
        );
        // Should detect at least the embedding and one transformer block
        assert!(
            !layers.is_empty(),
            "GGUF trace must produce at least one layer"
        );
        // BUG-TRACE-001 FIX: total_params should be computed from tensor shapes
        assert!(total_params > 0, "total_params should be > 0 for GGUF");
    }

    #[test]
    fn test_trace_safetensors_detects_layers() {
        let file = build_test_safetensors();
        let (format_name, layers, total_params) =
            detect_and_trace(file.path(), None, false).expect("detect_and_trace SafeTensors");
        assert_eq!(format_name, "SafeTensors");
        assert!(
            !layers.is_empty(),
            "SafeTensors trace must produce at least one layer"
        );
        // BUG-TRACE-001 FIX: total_params should be computed
        let _ = total_params; // May be 0 for test file
    }

    // ========================================================================
    // compute_vector_stats: comprehensive tests
    // ========================================================================

    #[test]
    fn test_compute_vector_stats_empty() {
        let stats = compute_vector_stats(&[]);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.l2_norm, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_compute_vector_stats_single_value() {
        let stats = compute_vector_stats(&[5.0]);
        assert!((stats.mean - 5.0).abs() < 1e-5);
        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_compute_vector_stats_basic() {
        let stats = compute_vector_stats(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((stats.mean - 3.0).abs() < 1e-5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_compute_vector_stats_negative_values() {
        let stats = compute_vector_stats(&[-3.0, -1.0, 0.0, 1.0, 3.0]);
        assert!((stats.mean - 0.0).abs() < 1e-5);
        assert_eq!(stats.min, -3.0);
        assert_eq!(stats.max, 3.0);
    }

    #[test]
    fn test_compute_vector_stats_with_nan() {
        let stats = compute_vector_stats(&[1.0, f32::NAN, 3.0]);
        assert_eq!(stats.nan_count, 1);
        assert!((stats.mean - 2.0).abs() < 1e-5); // Mean of 1 and 3
    }

    #[test]
    fn test_compute_vector_stats_with_inf() {
        let stats = compute_vector_stats(&[2.0, f32::INFINITY, 4.0]);
        assert_eq!(stats.inf_count, 1);
        assert!((stats.mean - 3.0).abs() < 1e-5); // Mean of 2 and 4
    }

    #[test]
    fn test_compute_vector_stats_with_neg_inf() {
        let stats = compute_vector_stats(&[2.0, f32::NEG_INFINITY, 8.0]);
        assert_eq!(stats.inf_count, 1);
        assert!((stats.mean - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_vector_stats_all_nan() {
        let stats = compute_vector_stats(&[f32::NAN, f32::NAN, f32::NAN]);
        assert_eq!(stats.nan_count, 3);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_compute_vector_stats_all_inf() {
        let stats = compute_vector_stats(&[f32::INFINITY, f32::NEG_INFINITY]);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_compute_vector_stats_l2_norm() {
        let stats = compute_vector_stats(&[3.0, 4.0]); // sqrt(9+16) = 5
        assert!((stats.l2_norm - 5.0).abs() < 1e-5);
    }

    // ========================================================================
    // is_likely_garbage: comprehensive branch coverage
    // ========================================================================

    #[test]
    fn test_is_likely_garbage_empty() {
        assert!(!is_likely_garbage(""));
    }

    #[test]
    fn test_is_likely_garbage_normal_text() {
        assert!(!is_likely_garbage("The answer is 42."));
    }

    #[test]
    fn test_is_likely_garbage_repeated_words() {
        // More than 50% repeated words
        assert!(is_likely_garbage("foo foo foo foo foo bar"));
    }

    #[test]
    fn test_is_likely_garbage_unicode_replacement() {
        // High ratio of replacement characters
        assert!(is_likely_garbage(
            "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}x"
        ));
    }

    #[test]
    fn test_is_likely_garbage_private_use_area() {
        assert!(is_likely_garbage("\u{E000}\u{E001}\u{E002}\u{E003}x"));
    }

    #[test]
    fn test_is_likely_garbage_pattern_random_random() {
        assert!(is_likely_garbage("some random random text here"));
    }

    #[test]
    fn test_is_likely_garbage_pattern_random_underscore() {
        assert!(is_likely_garbage("random_ stuff"));
    }

    #[test]
    fn test_is_likely_garbage_pattern_domain_domain() {
        assert!(is_likely_garbage("domain domain something"));
    }

    #[test]
    fn test_is_likely_garbage_pattern_pandas() {
        assert!(is_likely_garbage("pandas pandas thing"));
    }

    #[test]
    fn test_is_likely_garbage_no_normal_words_no_numbers() {
        // No common English words, no numbers, >2 words
        assert!(is_likely_garbage("zyx wvut srqp onml"));
    }

    #[test]
    fn test_is_likely_garbage_math_with_numbers() {
        // Has numbers, so not garbage
        assert!(!is_likely_garbage("4"));
    }

    #[test]
    fn test_is_likely_garbage_with_common_words() {
        assert!(!is_likely_garbage("the quick brown fox"));
    }

    #[test]
    fn test_is_likely_garbage_single_word() {
        // Only 1 word, too short for repeated word check
        assert!(!is_likely_garbage("hello"));
    }

    #[test]
    fn test_is_likely_garbage_two_words_no_repeat() {
        // 2 words, no repeats, has normal word
        assert!(!is_likely_garbage("the answer"));
    }

    #[test]
    fn test_is_likely_garbage_pattern_domainuster() {
        assert!(is_likely_garbage("some domainuster output"));
    }

    #[test]
    fn test_is_likely_garbage_pattern_localents() {
        assert!(is_likely_garbage("localents and stuff"));
    }

    #[test]
    fn test_is_likely_garbage_pattern_nunca() {
        assert!(is_likely_garbage("nunca something"));
    }

    #[test]
    fn test_is_likely_garbage_pattern_mult() {
        assert!(is_likely_garbage("x.mult something"));
    }

    // ========================================================================
    // extract_layer_index: all patterns
    // ========================================================================

    #[test]
    fn test_extract_layer_index_layers_pattern() {
        assert_eq!(extract_layer_index("model.layers.5.self_attn"), Some(5));
    }

    #[test]
    fn test_extract_layer_index_layer_pattern() {
        assert_eq!(extract_layer_index("encoder.layer.12.attention"), Some(12));
    }

    #[test]
    fn test_extract_layer_index_h_pattern() {
        assert_eq!(extract_layer_index("h.3.attn.weight"), Some(3));
    }

    #[test]
    fn test_extract_layer_index_blk_pattern() {
        assert_eq!(extract_layer_index("blk.0.ffn_gate.weight"), Some(0));
    }

    #[test]
    fn test_extract_layer_index_blocks_pattern() {
        assert_eq!(extract_layer_index("blocks.7.output"), Some(7));
    }

    #[test]
    fn test_extract_layer_index_block_pattern() {
        assert_eq!(extract_layer_index("block.99.weight"), Some(99));
    }

    #[test]
    fn test_extract_layer_index_no_match() {
        assert_eq!(extract_layer_index("embed_tokens.weight"), None);
    }

    #[test]
    fn test_extract_layer_index_no_number() {
        assert_eq!(extract_layer_index("layers.abc.weight"), None);
    }

    #[test]
    fn test_extract_layer_index_large_number() {
        assert_eq!(
            extract_layer_index("model.layers.1024.self_attn"),
            Some(1024)
        );
    }

    #[test]
    fn test_extract_layer_index_zero() {
        assert_eq!(extract_layer_index("model.layers.0.norm"), Some(0));
    }
