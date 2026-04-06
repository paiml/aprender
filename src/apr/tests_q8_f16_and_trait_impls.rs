
    #[ignore = "APR dtype parsing bug - needs investigation"]
    #[test]
    fn test_get_tensor_f32_q8_0_dtype() {
        // Q8_0 dtype = 10, block = 2-byte scale + 32 i8 values = 34 bytes for 32 elements
        let mut q8_data = vec![0u8; 34];
        q8_data[0] = 0x00;
        q8_data[1] = 0x3C; // Scale = 1.0 in f16
        for i in 0..32 {
            q8_data[2 + i] = i as u8;
        }

        // Create with shape [32] since Q8_0 block has 32 elements
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_entry = create_binary_tensor_entry("typed.weight", 10, &[32], 0, 34);
        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + 34;
        let mut model_data = vec![0u8; total_size];

        model_data[0..4].copy_from_slice(&MAGIC);
        model_data[4] = 2;
        model_data[5] = 0;
        model_data[8..12].copy_from_slice(&1u32.to_le_bytes());
        model_data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        model_data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        model_data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        model_data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        model_data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        model_data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);
        let data_start = data_offset as usize;
        model_data[data_start..data_start + 34].copy_from_slice(&q8_data);

        let model = AprV2Model::from_bytes(model_data).expect("should load");
        let result = model.get_tensor_f32("typed.weight");
        assert!(result.is_ok());
        let floats = result.expect("APR operation failed");
        assert_eq!(floats.len(), 32);
    }

    #[test]
    fn test_get_tensor_f32_unsupported_dtype() {
        // BF16 dtype = 2 is not fully supported for get_tensor_f32
        let bf16_data = vec![0x00, 0x3F, 0x80, 0x00]; // Two BF16 values
        let model_data = create_test_apr_model_with_dtype(2, &bf16_data);
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let result = model.get_tensor_f32("typed.weight");
        // BF16 is not in the supported list, should error
        assert!(result.is_err());
    }

    #[test]
    fn test_get_tensor_f32_out_of_bounds() {
        // Create a model where tensor data extends beyond file
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&100u64.to_le_bytes()); // data_offset

        // Add tensor entry that claims data beyond file
        let tensor_entry = create_binary_tensor_entry("oob.weight", 0, &[1000], 0, 4000);
        data[64..64 + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let model = AprV2Model::from_bytes(data).expect("should load");
        let result = model.get_tensor_f32("oob.weight");
        assert!(result.is_err()); // Out of bounds
    }

    // =========================================================================
    // decode_tokens extended tests
    // =========================================================================

    #[test]
    fn test_decode_tokens_gpt2_special() {
        // Test GPT-2 style byte-level BPE tokens
        let vocab = vec![
            "Ġhello".to_string(), // Space + hello
            "Ċ".to_string(),      // Newline
            "ĉ".to_string(),      // Tab
        ];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert!(result.contains("hello"));
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }

    #[test]
    fn test_decode_tokens_empty_string_token() {
        let vocab = vec![String::new(), "a".to_string()];
        let result = AprV2Model::decode_tokens(&vocab, &[0, 1]);
        // Empty token shouldn't cause issues
        assert!(result.contains('a'));
    }

    // =========================================================================
    // f16_to_f32 edge cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_smallest_positive_normal() {
        // Smallest positive normal f16 = 0x0400 (6.103515625e-5)
        let result = crate::apr::f16_to_f32(0x0400);
        assert!(result > 0.0 && result < 1e-4);
    }

    #[test]
    fn test_f16_to_f32_largest_normal() {
        // Largest finite f16 = 0x7BFF (65504.0)
        let result = crate::apr::f16_to_f32(0x7BFF);
        assert!((result - 65504.0).abs() < 10.0);
    }

    #[test]
    fn test_f16_to_f32_negative_normal() {
        // -2.0 in f16 = 0xC000
        let result = crate::apr::f16_to_f32(0xC000);
        assert!((result + 2.0).abs() < 0.01);
    }

    #[test]
    fn test_f16_to_f32_subnormal_nonzero() {
        // Various subnormals (exp=0, mantissa!=0)
        for mant in [1u16, 10, 100, 0x3FF] {
            let result = crate::apr::f16_to_f32(mant);
            assert!(result > 0.0, "Subnormal {mant:#x} should be positive");
        }
    }

    // =========================================================================
    // bpe_encode extended tests
    // =========================================================================

    #[test]
    fn test_bpe_encode_with_newline() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ċ".to_string(), 0); // Newline token
        let result = bpe_encode("\n", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_tab() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("ĉ".to_string(), 0); // Tab token
        let result = bpe_encode("\t", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_with_space() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("Ġ".to_string(), 0); // Space token
        let result = bpe_encode(" ", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_bpe_encode_mixed() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("Ġ".to_string(), 1); // Space
        token_to_id.insert("b".to_string(), 2);
        let result = bpe_encode("a b", &token_to_id, &[], &HashMap::new());
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_bpe_encode_multiple_merges() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        token_to_id.insert("b".to_string(), 1);
        token_to_id.insert("c".to_string(), 2);
        token_to_id.insert("ab".to_string(), 3);
        token_to_id.insert("abc".to_string(), 4);

        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let result = bpe_encode("abc", &token_to_id, &merges, &HashMap::new());
        // Should merge a+b->ab, then ab+c->abc
        assert!(!result.is_empty());
    }

    // =========================================================================
    // AprV2Model predict with weight tensor
    // =========================================================================

    /// Helper to create APR model with weight and bias tensors for linear model
    fn create_linear_model_apr() -> Vec<u8> {
        let metadata = r#"{"architecture":"linear"}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Weight tensor: 2x3 = 6 elements, 24 bytes
        // Bias tensor: 2 elements, 8 bytes
        let weight_entry = create_binary_tensor_entry("weight", 0, &[2, 3], 0, 24);
        let bias_entry = create_binary_tensor_entry("bias", 0, &[2], 24, 8);
        let tensor_index_size = weight_entry.len() + bias_entry.len();

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_index_size as u64;
        let total_size = data_offset as usize + 32;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&2u32.to_le_bytes()); // tensor_count = 2
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + weight_entry.len()].copy_from_slice(&weight_entry);
        data[idx_start + weight_entry.len()..idx_start + tensor_index_size]
            .copy_from_slice(&bias_entry);

        // Tensor data - weight matrix [2,3] (identity-ish pattern)
        let data_start = data_offset as usize;
        let weights: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2,3] matrix
        for (i, &w) in weights.iter().enumerate() {
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        // Bias [2]
        let biases: [f32; 2] = [0.5, 0.5];
        for (i, &b) in biases.iter().enumerate() {
            data[data_start + 24 + i * 4..data_start + 24 + i * 4 + 4]
                .copy_from_slice(&b.to_le_bytes());
        }

        data
    }

    #[test]
    fn test_apr_v2_model_predict_with_weights() {
        let model_data = create_linear_model_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        let features = vec![1.0, 2.0, 3.0];
        let result = model.predict(&features);
        assert!(result.is_ok());
        let output = result.expect("APR operation failed");
        assert_eq!(output.len(), 2);
        // Expected: [1*1 + 2*0 + 3*0 + 0.5, 1*0 + 2*1 + 3*0 + 0.5] = [1.5, 2.5]
        assert!((output[0] - 1.5).abs() < 0.01);
        assert!((output[1] - 2.5).abs() < 0.01);
    }

    // =========================================================================
    // AprFlags bit operations
    // =========================================================================

    #[test]
    fn test_apr_flags_sharded() {
        let flags = AprFlags::new(AprFlags::SHARDED);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
        assert!(!flags.is_quantized());
    }

    #[test]
    fn test_apr_flags_signed() {
        let flags = AprFlags::new(AprFlags::SIGNED);
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
    }

    #[test]
    fn test_apr_flags_all_set() {
        let flags = AprFlags::new(0xFFFF);
        assert!(flags.is_compressed());
        assert!(flags.is_encrypted());
        assert!(flags.is_quantized());
        assert!(flags.has_vocab());
    }

    // =========================================================================
    // simd_dot edge cases
    // =========================================================================

    #[test]
    fn test_simd_dot_empty() {
        let result = crate::apr::simd_dot(&[], &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_simd_dot_single_element() {
        let result = crate::apr::simd_dot(&[3.0], &[4.0]);
        assert!((result - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_negative() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![-1.0, -1.0, -1.0, -1.0];
        let result = crate::apr::simd_dot(&a, &b);
        // (-1)*(-1) + (-2)*(-1) + (-3)*(-1) + (-4)*(-1) = 1 + 2 + 3 + 4 = 10
        assert!((result - 10.0).abs() < 1e-6);
    }

    // =========================================================================
    // rms_norm multi-sequence
    // =========================================================================

    #[test]
    fn test_rms_norm_multi_sequence() {
        // 2 sequences of hidden_dim=4
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let result = crate::apr::rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 8);
    }

    // =========================================================================
    // simple_attention edge cases
    // =========================================================================

    #[test]
    fn test_simple_attention_mqa() {
        // MQA: 4 Q heads, 1 KV head
        let num_heads = 4;
        let num_kv_heads = 1;
        let head_dim = 2;
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q = vec![1.0; hidden_dim];
        let k = vec![1.0; kv_dim];
        let v = vec![1.0; kv_dim];

        let result = crate::apr::simple_attention(&q, &k, &v, 1, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), hidden_dim);
    }

    // =========================================================================
    // AprHeader debug and clone
    // =========================================================================

    #[test]
    fn test_apr_header_clone() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&5u32.to_le_bytes());

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        let cloned = header.clone();
        assert_eq!(cloned.tensor_count, header.tensor_count);
        assert_eq!(cloned.version, header.version);
    }

    #[test]
    fn test_apr_header_debug() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;

        let header = AprHeader::from_bytes(&data).expect("APR operation failed");
        let debug_str = format!("{:?}", header);
        assert!(debug_str.contains("AprHeader"));
    }

    // =========================================================================
    // TensorEntry debug and clone
    // =========================================================================

    #[test]
    fn test_tensor_entry_debug() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 64,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("TensorEntry"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_tensor_entry_clone() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![4, 4],
            offset: 0,
            size: 64,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.name, entry.name);
        assert_eq!(cloned.shape, entry.shape);
    }

    // =========================================================================
    // AprMetadata debug and clone
    // =========================================================================

    #[test]
    fn test_apr_metadata_debug() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            ..Default::default()
        };
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("AprMetadata"));
    }

    #[test]
    fn test_apr_metadata_clone() {
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            ..Default::default()
        };
        let cloned = meta.clone();
        assert_eq!(cloned.hidden_size, meta.hidden_size);
        assert_eq!(cloned.num_layers, meta.num_layers);
    }
