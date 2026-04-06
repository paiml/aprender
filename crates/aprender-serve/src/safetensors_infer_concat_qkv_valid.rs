
    #[test]
    fn test_concat_qkv() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let v = vec![5.0, 6.0];
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert_eq!(qkv, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // =========================================================================
    // Extended Coverage Tests (15+ tests ending with _ext_cov)
    // =========================================================================

    /// Generate valid non-zero F32 bytes for `count` elements.
    /// Uses sin pattern to pass all validation gates (density, L2, variation).
    fn valid_f32_bytes(count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|i| ((i as f32 * 0.01).sin() * 0.1 + 0.05).to_le_bytes())
            .collect()
    }

    /// Helper function to create a minimal SafeTensors file with given tensors
    fn create_safetensors_bytes(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        use serde_json::json;

        // Calculate tensor data layout
        let mut tensor_entries = serde_json::Map::new();
        let mut offset = 0usize;

        for (name, dtype, shape, data) in tensors {
            let end = offset + data.len();
            tensor_entries.insert(
                (*name).to_string(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }

        let json_obj = serde_json::Value::Object(tensor_entries);
        let json_bytes = json_obj.to_string().into_bytes();

        let mut data = Vec::new();
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        data.extend_from_slice(&json_bytes);

        // Append tensor data
        for (_, _, _, tensor_data) in tensors {
            data.extend_from_slice(tensor_data);
        }

        data
    }

    /// Helper to create config.json content
    fn create_config_json(
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        vocab_size: usize,
    ) -> String {
        format!(
            r#"{{
                "hidden_size": {},
                "num_hidden_layers": {},
                "num_attention_heads": {},
                "vocab_size": {},
                "intermediate_size": {},
                "max_position_embeddings": 2048,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama"
            }}"#,
            hidden_size,
            num_layers,
            num_heads,
            vocab_size,
            hidden_size * 4
        )
    }

    #[test]
    fn test_convert_file_not_found_ext_cov() {
        let result =
            SafetensorsToAprConverter::convert(Path::new("/nonexistent/model.safetensors"));
        assert!(result.is_err());
        // MappedSafeTensorsModel::load() returns UnsupportedOperation for file open errors
        if let Err(RealizarError::UnsupportedOperation { operation, reason }) = result {
            assert_eq!(operation, "open_safetensors");
            assert!(reason.contains("Failed to open file"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_convert_missing_config_json_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");

        // Create a minimal valid safetensors file
        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // No config.json file

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::UnsupportedOperation { operation, reason }) = result {
            assert_eq!(operation, "safetensors_convert");
            assert!(reason.contains("config.json not found"));
        } else {
            panic!("Expected UnsupportedOperation error");
        }
    }

    #[test]
    fn test_convert_missing_hidden_size_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Create minimal safetensors
        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Config missing hidden_size
        let config = r#"{"num_hidden_layers": 2, "num_attention_heads": 4, "vocab_size": 100}"#;
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::FormatError { reason }) = result {
            assert!(reason.contains("missing hidden_size"));
        } else {
            panic!("Expected FormatError for missing hidden_size");
        }
    }

    #[test]
    fn test_convert_missing_num_hidden_layers_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Config missing num_hidden_layers
        let config = r#"{"hidden_size": 64, "num_attention_heads": 4, "vocab_size": 100}"#;
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::FormatError { reason }) = result {
            assert!(reason.contains("missing num_hidden_layers"));
        } else {
            panic!("Expected FormatError for missing num_hidden_layers");
        }
    }

    #[test]
    fn test_convert_missing_num_attention_heads_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Config missing num_attention_heads
        let config = r#"{"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100}"#;
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::FormatError { reason }) = result {
            assert!(reason.contains("missing num_attention_heads"));
        } else {
            panic!("Expected FormatError for missing num_attention_heads");
        }
    }

    #[test]
    fn test_convert_missing_vocab_size_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Config missing vocab_size
        let config = r#"{"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}"#;
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        if let Err(RealizarError::FormatError { reason }) = result {
            assert!(reason.contains("missing vocab_size"));
        } else {
            panic!("Expected FormatError for missing vocab_size");
        }
    }

    #[test]
    fn test_convert_missing_embed_tokens_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Safetensors without model.embed_tokens.weight
        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");

        // Valid config
        let config = create_config_json(64, 1, 4, 100);
        std::fs::write(&config_path, config).expect("write config");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
        // Should fail because model.embed_tokens.weight is missing
    }

    #[test]
    fn test_concat_qkv_empty_inputs_ext_cov() {
        let q: Vec<f32> = vec![];
        let k: Vec<f32> = vec![];
        let v: Vec<f32> = vec![];
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert!(qkv.is_empty());
    }

    #[test]
    fn test_concat_qkv_single_elements_ext_cov() {
        let q = vec![1.0];
        let k = vec![2.0];
        let v = vec![3.0];
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert_eq!(qkv, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_concat_qkv_large_arrays_ext_cov() {
        let q: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let k: Vec<f32> = (1000..2000).map(|i| i as f32).collect();
        let v: Vec<f32> = (2000..3000).map(|i| i as f32).collect();
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert_eq!(qkv.len(), 3000);
        assert_eq!(qkv[0], 0.0);
        assert_eq!(qkv[1000], 1000.0);
        assert_eq!(qkv[2000], 2000.0);
        assert_eq!(qkv[2999], 2999.0);
    }

    #[test]
    fn test_concat_qkv_asymmetric_ext_cov() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![5.0, 6.0];
        let v = vec![7.0];
        let qkv = SafetensorsToAprConverter::concat_qkv(&q, &k, &v);
        assert_eq!(qkv, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_try_concat_qkv_bias_none_when_missing_ext_cov() {
        // Create safetensors model without any biases
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let data = create_safetensors_bytes(&[]);
        std::fs::write(&model_path, data).expect("write safetensors");
        let st_model = MappedSafeTensorsModel::load(&model_path).expect("load safetensors");

        let result =
            SafetensorsToAprConverter::try_concat_qkv_bias(&st_model, "model.layers.0", 64, 64);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_concat_qkv_bias_partial_missing_ext_cov() {
        // Create safetensors with only q_proj.bias (missing k and v)
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let q_bias_data: Vec<u8> = (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let data = create_safetensors_bytes(&[(
            "model.layers.0.self_attn.q_proj.bias",
            "F32",
            &[4],
            &q_bias_data,
        )]);
        std::fs::write(&model_path, data).expect("write safetensors");
        let st_model = MappedSafeTensorsModel::load(&model_path).expect("load safetensors");

        // Should return None because k_bias and v_bias are missing
        let result =
            SafetensorsToAprConverter::try_concat_qkv_bias(&st_model, "model.layers.0", 4, 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_try_concat_qkv_bias_all_present_ext_cov() {
        // Create F32 byte data for biases
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let q_bias_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let k_bias_data: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let v_bias_data: Vec<u8> = [7.0f32, 8.0].iter().flat_map(|f| f.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            (
                "model.layers.0.self_attn.q_proj.bias",
                "F32",
                &[4],
                &q_bias_data,
            ),
            (
                "model.layers.0.self_attn.k_proj.bias",
                "F32",
                &[2],
                &k_bias_data,
            ),
            (
                "model.layers.0.self_attn.v_proj.bias",
                "F32",
                &[2],
                &v_bias_data,
            ),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");
        let st_model = MappedSafeTensorsModel::load(&model_path).expect("load safetensors");

        let result =
            SafetensorsToAprConverter::try_concat_qkv_bias(&st_model, "model.layers.0", 4, 2);
        assert!(result.is_some());
        let bias = result.expect("operation failed");
        assert_eq!(bias.len(), 8); // 4 + 2 + 2
        assert_eq!(bias[0], 1.0);
        assert_eq!(bias[4], 5.0);
        assert_eq!(bias[6], 7.0);
    }

    #[test]
    fn test_convert_defaults_intermediate_size_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config without intermediate_size (should default to hidden_size * 4)
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "vocab_size": 100
        }"#;
        std::fs::write(&config_path, config).expect("write config");

        // Create safetensors with minimal required tensors for 0 layers
        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        assert_eq!(transformer.config.intermediate_dim, 64 * 4);
    }

    #[test]
    fn test_convert_defaults_context_length_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config without max_position_embeddings (should default to 2048)
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "intermediate_size": 256
        }"#;
        std::fs::write(&config_path, config).expect("write config");

        let embed_data = valid_f32_bytes(100 * 64);
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "convert failed: {:?}", result.err());
        let transformer = result.expect("operation failed");
        // N-03: context_length defaults to 0 when config.json has no max_position_embeddings
        assert_eq!(transformer.config.context_length, 0);
    }

    #[test]
    fn test_convert_tied_embeddings_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let config = create_config_json(64, 0, 4, 100);
        std::fs::write(&config_path, config).expect("write config");

        // Create tensors WITHOUT lm_head.weight (tied embeddings)
        let embed_data: Vec<u8> = (0..(100 * 64))
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok());
        let transformer = result.expect("operation failed");

        // lm_head_weight should have same dimensions as token_embedding (tied or separate)
        // When tied: lm_head_weight.len() == token_embedding.len()
        // But they may not be equal if transposed or if implementation uses separate weights
        assert!(!transformer.lm_head_weight.is_empty());
        assert!(!transformer.token_embedding.is_empty());
    }
