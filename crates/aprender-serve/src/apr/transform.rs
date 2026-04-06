
    // =========================================================================
    // BpeTokenizer debug and clone
    // =========================================================================

    #[test]
    fn test_bpe_tokenizer_debug() {
        let tokenizer = BpeTokenizer {
            token_to_id: HashMap::new(),
            id_to_token: vec![],
            merge_rules: vec![],
            bos_id: Some(1),
            eos_id: Some(2),
            special_tokens: HashMap::new(),
        };
        let debug_str = format!("{:?}", tokenizer);
        assert!(debug_str.contains("BpeTokenizer"));
    }

    #[test]
    fn test_bpe_tokenizer_clone() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("a".to_string(), 0);
        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token: vec!["a".to_string()],
            merge_rules: vec![("a".to_string(), "b".to_string())],
            bos_id: Some(1),
            eos_id: Some(2),
            special_tokens: HashMap::new(),
        };
        let cloned = tokenizer.clone();
        assert_eq!(cloned.bos_id, tokenizer.bos_id);
        assert_eq!(cloned.id_to_token, tokenizer.id_to_token);
    }

    // =========================================================================
    // AprFlags debug
    // =========================================================================

    #[test]
    fn test_apr_flags_debug() {
        let flags = AprFlags::new(0x0025);
        let debug_str = format!("{:?}", flags);
        assert!(debug_str.contains("AprFlags"));
    }

    #[test]
    fn test_apr_flags_clone() {
        let flags = AprFlags::new(0x0025);
        let cloned = flags;
        assert_eq!(cloned.is_lz4(), flags.is_lz4());
        assert_eq!(cloned.is_quantized(), flags.is_quantized());
    }

    // =========================================================================
    // AprV2Model debug
    // =========================================================================

    #[test]
    fn test_apr_v2_model_debug() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("APR operation failed");
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("AprV2Model"));
    }

    // =========================================================================
    // ModelData extended edge cases
    // =========================================================================

    #[test]
    fn test_model_data_debug_heap() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        let debug_str = format!("{:?}", data);
        assert!(debug_str.contains("Heap"));
    }

    // =========================================================================
    // dequantize multiple blocks
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_three_blocks() {
        // Three Q8_0 blocks = 102 bytes, 96 elements
        let mut bytes = vec![0u8; 102];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0 block 1
        bytes[34] = 0x00;
        bytes[35] = 0x3C; // Scale = 1.0 block 2
        bytes[68] = 0x00;
        bytes[69] = 0x3C; // Scale = 1.0 block 3

        let result = crate::apr::dequantize_q8_0(&bytes, 96);
        assert_eq!(result.len(), 96);
    }

    #[test]
    fn test_dequantize_q4_k_with_nonzero_scales() {
        // Q4_K super-block with actual scale values
        let mut bytes = vec![0u8; 144];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // d = 1.0
        bytes[2] = 0x00;
        bytes[3] = 0x3C; // dmin = 1.0
                         // Set some scale values
        bytes[4] = 0x3F; // scales[0] = 63
        bytes[5] = 0x3F;

        let result = crate::apr::dequantize_q4_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6_k_with_nonzero_scales() {
        // Q6_K super-block with scale values
        let mut bytes = vec![0u8; 210];
        // scales at offset 192-207
        bytes[192] = 10;
        bytes[193] = 20;
        // d at offset 208-209
        bytes[208] = 0x00;
        bytes[209] = 0x3C; // d = 1.0

        let result = crate::apr::dequantize_q6_k(&bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // =========================================================================
    // Transformer Model Tests - Forward Pass Coverage
    // =========================================================================

    /// Helper to create a minimal transformer model for testing forward pass.
    /// This creates a 1-layer transformer with tiny dimensions for test purposes.
    fn create_mini_transformer_apr() -> Vec<u8> {
        let metadata = r#"{
            "architecture": "llama",
            "hidden_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "num_kv_heads": 2,
            "vocab_size": 10,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6
        }"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Tensors needed for forward:
        // - model.embed_tokens.weight [vocab=10, hidden=8] = 80 floats = 320 bytes
        // - layers.0.input_layernorm.weight [hidden=8] = 8 floats = 32 bytes
        // - layers.0.self_attn.q_proj.weight [hidden=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.k_proj.weight [kv_dim=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.v_proj.weight [kv_dim=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.self_attn.o_proj.weight [hidden=8, hidden=8] = 64 floats = 256 bytes
        // - layers.0.post_attention_layernorm.weight [hidden=8] = 8 floats = 32 bytes
        // - layers.0.mlp.gate_proj.weight [inter=16, hidden=8] = 128 floats = 512 bytes
        // - layers.0.mlp.up_proj.weight [inter=16, hidden=8] = 128 floats = 512 bytes
        // - layers.0.mlp.down_proj.weight [hidden=8, inter=16] = 128 floats = 512 bytes
        // - norm.weight [hidden=8] = 8 floats = 32 bytes
        // - lm_head.weight [vocab=10, hidden=8] = 80 floats = 320 bytes

        let tensor_defs: Vec<(&str, &[usize], usize)> = vec![
            ("model.embed_tokens.weight", &[10, 8], 320),
            ("layers.0.input_layernorm.weight", &[8], 32),
            ("layers.0.self_attn.q_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.k_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.v_proj.weight", &[8, 8], 256),
            ("layers.0.self_attn.o_proj.weight", &[8, 8], 256),
            ("layers.0.post_attention_layernorm.weight", &[8], 32),
            ("layers.0.mlp.gate_proj.weight", &[16, 8], 512),
            ("layers.0.mlp.up_proj.weight", &[16, 8], 512),
            ("layers.0.mlp.down_proj.weight", &[8, 16], 512),
            ("norm.weight", &[8], 32),
            ("lm_head.weight", &[10, 8], 320),
        ];

        let mut tensor_entries = Vec::new();
        let mut current_offset = 0u64;

        for (name, shape, byte_size) in &tensor_defs {
            let shape_vec: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
            let entry =
                create_binary_tensor_entry(name, 0, &shape_vec, current_offset, *byte_size as u64);
            tensor_entries.push(entry);
            current_offset += *byte_size as u64;
        }

        let tensor_index: Vec<u8> = tensor_entries
            .iter()
            .flat_map(|e| e.iter().copied())
            .collect();
        let tensor_count = tensor_defs.len() as u32;
        let total_data_size = current_offset as usize;

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_index.len() as u64;
        let total_size = data_offset as usize + total_data_size;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

        // Tensor data - initialize with small random-ish values
        let data_start = data_offset as usize;
        let num_floats = total_data_size / 4;
        for i in 0..num_floats {
            let val = ((i % 10) as f32 - 5.0) * 0.1; // Small values between -0.5 and 0.4
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        // Set layernorm weights to 1.0 (they need to be non-zero)
        let norm_weight_offsets = vec![320, 320 + 32 + 256 * 4, 320 + 32 + 256 * 4 + 32 + 512 * 3];
        for offset in norm_weight_offsets {
            for i in 0..8 {
                let val = 1.0f32;
                let pos = data_start + offset + i * 4;
                if pos + 4 <= data.len() {
                    data[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }

        data
    }

    #[test]
    fn test_transformer_model_loads() {
        let model_data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");
        assert!(model.metadata().is_transformer());
        assert_eq!(model.metadata().hidden_size, Some(8));
        assert_eq!(model.metadata().num_layers, Some(1));
        assert_eq!(model.metadata().vocab_size, Some(10));
    }

    #[test]
    fn test_transformer_has_all_tensors() {
        let model_data = create_mini_transformer_apr();
        let model = AprV2Model::from_bytes(model_data).expect("should load");

        // Check key tensors exist
        assert!(model.get_tensor("model.embed_tokens.weight").is_some());
        assert!(model
            .get_tensor("layers.0.input_layernorm.weight")
            .is_some());
        assert!(model
            .get_tensor("layers.0.self_attn.q_proj.weight")
            .is_some());
        assert!(model.get_tensor("norm.weight").is_some());
        assert!(model.get_tensor("lm_head.weight").is_some());
    }

    // =========================================================================
    // matmul edge cases
    // =========================================================================

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_matmul_out_of_bounds_x() {
        // x is shorter than expected — canonical cpu_matmul_transpose_b
        // correctly panics on dimension mismatch (programming error).
        let x = vec![1.0, 2.0]; // Only 2 elements, but seq=1, in_dim=4
        let w = vec![1.0; 8]; // 2 output dims, 4 input dims
        let _result = crate::apr::matmul(&x, &w, 1, 4, 2);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_matmul_out_of_bounds_w() {
        // w is shorter than expected — canonical cpu_matmul_transpose_b
        // correctly panics on dimension mismatch (programming error).
        let x = vec![1.0; 4]; // seq=1, in_dim=4
        let w = vec![1.0, 2.0]; // Too short
        let _result = crate::apr::matmul(&x, &w, 1, 4, 2);
    }

    // =========================================================================
    // dequantize with negative i8 values
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_negative_values() {
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
                         // Set negative i8 values (128-255 map to -128 to -1)
        bytes[2] = 255; // -1 as i8
        bytes[3] = 254; // -2 as i8
        bytes[4] = 128; // -128 as i8

        let result = crate::apr::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // First value should be negative: -1 * 1.0 = -1.0
        assert!(result[0] < 0.0);
    }

    // =========================================================================
    // f16_to_f32 quiet NaN
    // =========================================================================

    #[test]
    fn test_f16_to_f32_qnan() {
        // Quiet NaN in f16 (exp=0x1F, mantissa>0, sign=0)
        // 0x7E00 = quiet NaN
        let result = crate::apr::f16_to_f32(0x7E00);
        assert!(result.is_nan());
    }

    // =========================================================================
    // detect_format with magic bytes
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_apr_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&MAGIC).expect("write magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert_eq!(detect_format(temp.path()), "apr");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_gguf_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&[0x47, 0x47, 0x55, 0x46])
            .expect("write GGUF magic");
        temp.write_all(&[0u8; 60]).expect("write padding");

        assert_eq!(detect_format(temp.path()), "gguf");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detect_format_safetensors_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"{\"test\": 1}").expect("write JSON");

        assert_eq!(detect_format(temp.path()), "safetensors");
    }

    // =========================================================================
    // simd_dot_avx2 scalar fallback
    // =========================================================================

    #[test]
    fn test_simd_dot_non_multiple_of_8() {
        // Test with lengths that aren't multiples of 8 to exercise remainder handling
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 13];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum of 0..12 = 78
        assert!((result - 78.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_prime_length() {
        // Prime number length ensures both SIMD chunks and remainder are exercised
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 17];
        let result = crate::apr::simd_dot(&a, &b);
        // Sum of 0..16 = 136
        assert!((result - 136.0).abs() < 1e-6);
    }

    // =========================================================================
    // simple_attention multi-token
    // =========================================================================

    #[test]
    fn test_simple_attention_multiple_tokens() {
        // 3 tokens, 2 heads, head_dim=4
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let hidden_dim = num_heads * head_dim;

        let q = vec![1.0; seq_len * hidden_dim];
        let k = vec![1.0; seq_len * hidden_dim];
        let v = vec![1.0; seq_len * hidden_dim];

        let result =
            crate::apr::simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), seq_len * hidden_dim);
    }

    #[test]
    fn test_simple_attention_varying_values() {
        // Test with varying Q, K, V values
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;
        let seq_len = 2;
        let hidden_dim = num_heads * head_dim;

        // Different Q, K, V patterns
        let q = vec![1.0, 0.0, 0.0, 1.0]; // Token 1: [1,0], Token 2: [0,1]
        let k = vec![1.0, 0.0, 1.0, 0.0]; // Token 1: [1,0], Token 2: [1,0]
        let v = vec![1.0, 2.0, 3.0, 4.0]; // Token 1: [1,2], Token 2: [3,4]

        let result =
            crate::apr::simple_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim);
        assert_eq!(result.len(), seq_len * hidden_dim);
        // Output should be valid attention-weighted values
        assert!(!result.iter().any(|v| v.is_nan()));
    }

    // =========================================================================
    // ModelData methods
    // =========================================================================

    #[test]
    fn test_model_data_empty() {
        let data = ModelData::from_vec(vec![]);
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn test_model_data_as_slice_extended() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        let slice = data.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }
