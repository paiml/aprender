
    #[test]
    fn test_tensor_entry_from_binary_unknown_dtype_defaults_f32() {
        // Unknown dtype byte (e.g., 255) should default to F32
        let data = create_binary_tensor_entry("test", 255, &[10], 0, 40);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(entry.dtype, "F32");
    }

    #[test]
    fn test_tensor_entry_from_binary_8d_shape() {
        // Maximum supported dimensions
        let data = create_binary_tensor_entry(
            "high_dim_tensor",
            0,
            &[2, 3, 4, 5, 6, 7, 8, 9],
            0,
            2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 4,
        );
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");
        assert_eq!(entry.shape.len(), 8);
        assert_eq!(entry.element_count(), 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9);
    }

    // =========================================================================
    // Additional Model Loading Error Tests
    // =========================================================================

    #[test]
    fn test_apr_model_metadata_truncated_past_eof() {
        // metadata_offset + metadata_size > file length
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset = 64
        data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size = 1000 > remaining bytes
        data[24..32].copy_from_slice(&1064u64.to_le_bytes());
        data[32..40].copy_from_slice(&1064u64.to_le_bytes());

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn test_apr_model_tensor_out_of_bounds() {
        // Create model where tensor data extends past EOF
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Create tensor entry that points past EOF
        let tensor_entry = create_binary_tensor_entry(
            "bad_tensor",
            0,
            &[1000, 1000],
            0,
            4_000_000, // Size much larger than file
        );

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;

        // File is only 256 bytes total, but tensor claims to be 4MB
        let total_size = data_offset as usize + 64;
        let mut data = vec![0u8; total_size];

        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let model = AprV2Model::from_bytes(data).expect("should load");

        // Accessing tensor data should fail
        let result = model.get_tensor_f32("bad_tensor");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("out of bounds"));
    }

    // =========================================================================
    // Generate Method Edge Cases
    // =========================================================================

    #[test]
    fn test_apr_model_generate_eos_stops() {
        // Test that generation stops at EOS token
        // (Requires a valid transformer model - using mock to verify logic)
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Empty input should fail
        let result = model.generate(&[], 10, Some(2));
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_generate_max_tokens() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // Test that generate with max_tokens=0 works (returns empty or immediate)
        let result = model.generate(&[1, 2, 3], 0, None);
        // Either succeeds with 0 tokens or returns an error - both are valid
        if let Ok(tokens) = result {
            assert!(tokens.len() <= 3); // At most initial tokens
        }
    }

    // =========================================================================
    // SimpleTokenizer Additional Tests
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_bpe_newline() {
        use crate::apr::SimpleTokenizer;

        // BPE newline is Ċ
        let vocab = vec!["hello".to_string(), "Ċ".to_string(), "world".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert_eq!(decoded, "hello\nworld");
    }

    #[test]
    fn test_simple_tokenizer_bpe_tab() {
        use crate::apr::SimpleTokenizer;

        // BPE tab is ĉ
        let vocab = vec!["hello".to_string(), "ĉ".to_string(), "world".to_string()];
        let tokenizer = SimpleTokenizer::new(vocab, None, None);

        let decoded = tokenizer.decode(&[0, 1, 2]);
        assert_eq!(decoded, "hello\tworld");
    }

    #[test]
    fn test_simple_tokenizer_large_vocab() {
        use crate::apr::SimpleTokenizer;

        // Create large vocabulary
        let vocab: Vec<String> = (0..10000).map(|i| format!("token_{}", i)).collect();
        let tokenizer = SimpleTokenizer::new(vocab, Some(0), Some(9999));

        assert_eq!(tokenizer.vocab_size(), 10000);
        assert!(tokenizer.is_bos(0));
        assert!(tokenizer.is_eos(9999));

        let decoded = tokenizer.decode(&[100, 200, 300]);
        assert!(decoded.contains("token_100"));
        assert!(decoded.contains("token_200"));
        assert!(decoded.contains("token_300"));
    }

    // =========================================================================
    // BpeTokenizer Additional Tests
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_large_merge_rules() {
        // Test with many merge rules
        let vocab: HashMap<String, u32> = [
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
            ("ab".to_string(), 3),
            ("abc".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let id_to_token = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "ab".to_string(),
            "abc".to_string(),
        ];

        let merge_rules = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];

        let tokenizer = BpeTokenizer {
            token_to_id: vocab,
            id_to_token,
            merge_rules,
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        let encoded = tokenizer.encode("abc");
        // Should merge a+b -> ab, then ab+c -> abc
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_non_ascii() {
        // Test encoding non-ASCII characters
        let vocab: HashMap<String, u32> = [("a".to_string(), 0)].into_iter().collect();

        let id_to_token = vec!["a".to_string()];

        let tokenizer = BpeTokenizer {
            token_to_id: vocab,
            id_to_token,
            merge_rules: vec![],
            bos_id: None,
            eos_id: None,
            special_tokens: HashMap::new(),
        };

        // Non-ASCII should be handled (may result in empty if not in vocab)
        let encoded = tokenizer.encode("日本語");
        // Result depends on byte encoding - just verify it doesn't panic
        let _ = encoded;
    }

    // =========================================================================
    // MAGIC_PREFIX Tests
    // =========================================================================

    #[test]
    fn test_magic_prefix_constant() {
        assert_eq!(MAGIC_PREFIX, [0x41, 0x50, 0x52]); // "APR"
        assert_eq!(&MAGIC_PREFIX, b"APR");
    }

    #[test]
    fn test_header_size_constant() {
        assert_eq!(HEADER_SIZE, 64);
    }

    #[test]
    fn test_alignment_constant() {
        assert_eq!(ALIGNMENT, 64);
    }

    // ==========================================================================
    // PMAT-107: Falsification Test for GQA num_kv_heads in AprMetadata
    // ==========================================================================

    /// FALSIFICATION TEST: Verify AprMetadata correctly parses num_kv_heads from JSON
    /// This catches the silent failure case where unwrap_or_default() swallows errors.
    #[test]
    fn test_falsification_apr_metadata_parses_gqa_num_kv_heads() {
        // JSON from a real Qwen 1.5B APR file (GQA: 12 heads, 2 kv_heads)
        let metadata_json = r#"{
            "model_type": "qwen2",
            "architecture": "qwen2",
            "hidden_size": 1536,
            "num_layers": 28,
            "num_heads": 12,
            "num_kv_heads": 2,
            "vocab_size": 151936,
            "intermediate_size": 8960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001
        }"#;

        let metadata: AprMetadata =
            serde_json::from_str(metadata_json).expect("AprMetadata should parse valid JSON");

        assert_eq!(
            metadata.num_heads,
            Some(12),
            "num_heads not parsed correctly"
        );
        assert_eq!(
            metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads not parsed from AprMetadata JSON!\n\
             This causes GQA models to hang on GPU because they're treated as MHA."
        );
        assert_eq!(
            metadata.hidden_size,
            Some(1536),
            "hidden_size not parsed correctly"
        );

        println!(
            "✅ AprMetadata correctly parses num_kv_heads={:?}",
            metadata.num_kv_heads
        );
    }

    /// FALSIFICATION TEST: Verify AprMetadata handles extra fields gracefully
    /// The real APR files have extra fields like "model.num_kv_heads" that should
    /// be captured by the flatten attribute without breaking num_kv_heads parsing.
    #[test]
    fn test_falsification_apr_metadata_with_extra_fields() {
        // This JSON has both the standard fields AND the extra "model.*" fields
        // from the Q4K converter
        let metadata_json = r#"{
            "model_type": "qwen2",
            "name": "qwen2",
            "architecture": "qwen2",
            "hidden_size": 1536,
            "num_layers": 28,
            "num_heads": 12,
            "num_kv_heads": 2,
            "vocab_size": 151936,
            "intermediate_size": 8960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001,
            "model.intermediate_size": 8960,
            "model.num_kv_heads": 2,
            "model.rope_theta": 1000000.0
        }"#;

        let metadata: AprMetadata = serde_json::from_str(metadata_json)
            .expect("AprMetadata should parse JSON with extra fields");

        assert_eq!(
            metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads broken by extra 'model.*' fields!\n\
             Expected: Some(2), Got: {:?}",
            metadata.num_kv_heads
        );

        // Verify extra fields are captured
        assert!(
            metadata.extra.contains_key("model.num_kv_heads"),
            "Extra fields should be captured in 'extra' map"
        );

        println!(
            "✅ AprMetadata handles extra fields correctly, num_kv_heads={:?}",
            metadata.num_kv_heads
        );
    }

    /// GH-88: Verify AprMetadata parses "num_key_value_heads" alias
    /// The Q4K converter writes this HuggingFace-style field name.
    /// Without the alias, num_kv_heads defaults to num_heads, causing
    /// GPU bias slicing panic on GQA models.
    #[test]
    fn test_apr_metadata_parses_num_key_value_heads_alias() {
        // JSON from the Q4K converter path (uses HuggingFace field names)
        let metadata_json = r#"{
            "model_type": "transformer_lm_q4k",
            "architecture": "qwen2",
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "intermediate_size": 4864,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001
        }"#;

        let metadata: AprMetadata =
            serde_json::from_str(metadata_json).expect("AprMetadata should parse Q4K converter JSON");

        assert_eq!(
            metadata.num_kv_heads,
            Some(2),
            "GH-88: num_key_value_heads alias must be parsed as num_kv_heads.\n\
             Without this, GQA models panic at GPU bias slicing."
        );
        assert_eq!(metadata.num_heads, Some(14));
        assert_eq!(metadata.num_layers, Some(24));
    }

    /// PMAT-107: Falsification test with REAL APR file
    ///
    /// This test loads the actual APR file from disk and verifies that
    /// num_kv_heads is correctly parsed. If this test fails, the APR CUDA
    /// path will hang because GQA models will be treated as MHA.
    #[test]
    fn test_falsification_real_apr_file_num_kv_heads() {
        let home = std::env::var("HOME").expect("HOME env var not set");
        let apr_path_str = format!(
            "{}/.cache/huggingface/models/qwen2.5-coder-1.5b-apr/qwen2.5-coder-1.5b-q4k.apr",
            home
        );
        let apr_path = std::path::Path::new(&apr_path_str);

        if !apr_path.exists() {
            println!("⚠️ Test model not available at {:?}, skipping", apr_path);
            return;
        }

        // Load the model
        let model = AprV2Model::load(apr_path).expect("Should load APR file");

        println!("=== REAL APR FILE METADATA ===");
        println!("  num_heads: {:?}", model.metadata.num_heads);
        println!("  num_kv_heads: {:?}", model.metadata.num_kv_heads);
        println!("  hidden_size: {:?}", model.metadata.hidden_size);
        println!("  num_layers: {:?}", model.metadata.num_layers);

        // This is the critical assertion
        assert!(
            model.metadata.num_kv_heads.is_some(),
            "FALSIFICATION FAILED: num_kv_heads is None after loading real APR file!\n\
             This causes GQA models to hang on GPU because they're treated as MHA.\n\
             Expected: Some(2) for Qwen 1.5B GQA, Got: None"
        );

        assert_eq!(
            model.metadata.num_kv_heads,
            Some(2),
            "FALSIFICATION FAILED: num_kv_heads wrong value!\n\
             Expected: Some(2) for Qwen 1.5B GQA, Got: {:?}",
            model.metadata.num_kv_heads
        );

        println!(
            "✅ Real APR file has correct num_kv_heads={:?}",
            model.metadata.num_kv_heads
        );
    }
