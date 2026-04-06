
    #[test]
    fn test_convert_separate_lm_head_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let config = create_config_json(64, 0, 4, 100);
        std::fs::write(&config_path, config).expect("write config");

        // Create tensors WITH separate lm_head.weight
        let embed_data: Vec<u8> = (0..(100 * 64))
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let norm_data: Vec<u8> = (0..64).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let lm_head_data: Vec<u8> = (0..(100 * 64))
            .flat_map(|i| ((i + 1000) as f32).to_le_bytes())
            .collect();

        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[100, 64], &embed_data),
            ("model.norm.weight", "F32", &[64], &norm_data),
            ("lm_head.weight", "F32", &[100, 64], &lm_head_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok());
        let transformer = result.expect("operation failed");

        // lm_head_weight should NOT equal token_embedding
        assert_ne!(
            transformer.lm_head_weight[0],
            transformer.token_embedding[0]
        );
    }

    #[test]
    fn test_convert_with_rope_theta_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config with custom rope_theta
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "intermediate_size": 256,
            "rope_theta": 500000.0
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
        assert!((transformer.config.rope_theta - 500000.0).abs() < 1.0);
    }

    #[test]
    fn test_convert_with_rms_norm_eps_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        // Config with custom rms_norm_eps
        let config = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 0,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5
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
        assert!((transformer.config.eps - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn test_safetensors_to_apr_converter_struct_ext_cov() {
        // Test that SafetensorsToAprConverter is a unit struct
        let _converter = SafetensorsToAprConverter;
        // This just ensures the struct exists and can be instantiated
    }

    #[test]
    fn test_convert_architecture_from_config_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let config = create_config_json(64, 0, 4, 100);
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
        assert_eq!(transformer.config.architecture, "LlamaForCausalLM");
    }

    /// Helper to create all layer tensors for a single transformer layer
    fn create_layer_tensors(
        layer_idx: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Vec<(&'static str, String, Vec<usize>, Vec<u8>)> {
        let prefix = format!("model.layers.{layer_idx}");

        // Calculate tensor sizes
        let attn_norm_size = hidden_dim;
        let q_size = hidden_dim * hidden_dim;
        let k_size = hidden_dim * hidden_dim;
        let v_size = hidden_dim * hidden_dim;
        let o_size = hidden_dim * hidden_dim;
        let ffn_norm_size = hidden_dim;
        let gate_size = hidden_dim * intermediate_dim;
        let up_size = hidden_dim * intermediate_dim;
        let down_size = intermediate_dim * hidden_dim;

        vec![
            (
                "attn_norm",
                format!("{prefix}.input_layernorm.weight"),
                vec![attn_norm_size],
                // Norm vectors: ValidatedVector only checks NaN/Inf/length, 1.0 is safe
                (0..attn_norm_size)
                    .flat_map(|_| 1.0f32.to_le_bytes())
                    .collect(),
            ),
            (
                "q_proj",
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![hidden_dim, hidden_dim],
                valid_f32_bytes(q_size),
            ),
            (
                "k_proj",
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![hidden_dim, hidden_dim],
                valid_f32_bytes(k_size),
            ),
            (
                "v_proj",
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![hidden_dim, hidden_dim],
                valid_f32_bytes(v_size),
            ),
            (
                "o_proj",
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden_dim, hidden_dim],
                valid_f32_bytes(o_size),
            ),
            (
                "ffn_norm",
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![ffn_norm_size],
                (0..ffn_norm_size)
                    .flat_map(|_| 1.0f32.to_le_bytes())
                    .collect(),
            ),
            (
                "gate_proj",
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![intermediate_dim, hidden_dim],
                valid_f32_bytes(gate_size),
            ),
            (
                "up_proj",
                format!("{prefix}.mlp.up_proj.weight"),
                vec![intermediate_dim, hidden_dim],
                valid_f32_bytes(up_size),
            ),
            (
                "down_proj",
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden_dim, intermediate_dim],
                valid_f32_bytes(down_size),
            ),
        ]
    }

    #[test]
    fn test_convert_with_single_layer_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let hidden_dim = 16;
        let intermediate_dim = 64;
        let vocab_size = 50;
        let num_layers = 1;
        let num_heads = 4;

        // Config
        let config = format!(
            r#"{{
                "hidden_size": {},
                "num_hidden_layers": {},
                "num_attention_heads": {},
                "vocab_size": {},
                "intermediate_size": {},
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6
            }}"#,
            hidden_dim, num_layers, num_heads, vocab_size, intermediate_dim
        );
        std::fs::write(&config_path, config).expect("write config");

        // Build layer tensors
        let layer_tensors = create_layer_tensors(0, hidden_dim, intermediate_dim);

        // Build safetensors with all required tensors (valid non-zero data)
        let embed_data = valid_f32_bytes(vocab_size * hidden_dim);
        let norm_data: Vec<u8> = (0..hidden_dim).flat_map(|_| 1.0f32.to_le_bytes()).collect();

        // Create a comprehensive tensor list
        use serde_json::json;
        let mut tensor_entries = serde_json::Map::new();
        let mut all_data = Vec::new();
        let mut offset = 0usize;

        // Add embed_tokens
        tensor_entries.insert(
            "model.embed_tokens.weight".to_string(),
            json!({
                "dtype": "F32",
                "shape": [vocab_size, hidden_dim],
                "data_offsets": [offset, offset + embed_data.len()]
            }),
        );
        all_data.extend(&embed_data);
        offset += embed_data.len();

        // Add norm
        tensor_entries.insert(
            "model.norm.weight".to_string(),
            json!({
                "dtype": "F32",
                "shape": [hidden_dim],
                "data_offsets": [offset, offset + norm_data.len()]
            }),
        );
        all_data.extend(&norm_data);
        offset += norm_data.len();

        // Add layer tensors
        for (_, name, shape, data) in &layer_tensors {
            tensor_entries.insert(
                name.clone(),
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + data.len()]
                }),
            );
            all_data.extend(data);
            offset += data.len();
        }

        let json_obj = serde_json::Value::Object(tensor_entries);
        let json_bytes = json_obj.to_string().into_bytes();

        let mut safetensors_data = Vec::new();
        safetensors_data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        safetensors_data.extend_from_slice(&json_bytes);
        safetensors_data.extend_from_slice(&all_data);

        std::fs::write(&model_path, safetensors_data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "Conversion failed: {:?}", result.err());

        let transformer = result.expect("operation failed");
        assert_eq!(transformer.config.hidden_dim, hidden_dim);
        assert_eq!(transformer.config.num_layers, num_layers);
        assert_eq!(transformer.layers.len(), num_layers);

        // Verify layer structure
        let layer = &transformer.layers[0];
        assert_eq!(layer.attn_norm_weight.len(), hidden_dim);
        assert_eq!(layer.qkv_weight.len(), hidden_dim * 3 * hidden_dim);
        assert_eq!(layer.attn_output_weight.len(), hidden_dim * hidden_dim);
        assert_eq!(layer.ffn_up_weight.len(), hidden_dim * intermediate_dim);
        assert_eq!(layer.ffn_down_weight.len(), intermediate_dim * hidden_dim);
        assert!(layer.ffn_gate_weight.is_some());
        assert!(layer.ffn_norm_weight.is_some());
    }

    #[test]
    fn test_convert_with_multiple_layers_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let hidden_dim = 8;
        let intermediate_dim = 32;
        let vocab_size = 20;
        let num_layers = 2;
        let num_heads = 2;

        let config = format!(
            r#"{{
                "hidden_size": {},
                "num_hidden_layers": {},
                "num_attention_heads": {},
                "vocab_size": {},
                "intermediate_size": {}
            }}"#,
            hidden_dim, num_layers, num_heads, vocab_size, intermediate_dim
        );
        std::fs::write(&config_path, config).expect("write config");

        // Build tensors for multiple layers
        use serde_json::json;
        let mut tensor_entries = serde_json::Map::new();
        let mut all_data = Vec::new();
        let mut offset = 0usize;

        // Add embed_tokens (valid non-zero data)
        let embed_data = valid_f32_bytes(vocab_size * hidden_dim);
        tensor_entries.insert(
            "model.embed_tokens.weight".to_string(),
            json!({
                "dtype": "F32",
                "shape": [vocab_size, hidden_dim],
                "data_offsets": [offset, offset + embed_data.len()]
            }),
        );
        all_data.extend(&embed_data);
        offset += embed_data.len();

        // Add norm
        let norm_data: Vec<u8> = (0..hidden_dim).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        tensor_entries.insert(
            "model.norm.weight".to_string(),
            json!({
                "dtype": "F32",
                "shape": [hidden_dim],
                "data_offsets": [offset, offset + norm_data.len()]
            }),
        );
        all_data.extend(&norm_data);
        offset += norm_data.len();

        // Add all layer tensors
        for layer_idx in 0..num_layers {
            let layer_tensors = create_layer_tensors(layer_idx, hidden_dim, intermediate_dim);
            for (_, name, shape, data) in &layer_tensors {
                tensor_entries.insert(
                    name.clone(),
                    json!({
                        "dtype": "F32",
                        "shape": shape,
                        "data_offsets": [offset, offset + data.len()]
                    }),
                );
                all_data.extend(data);
                offset += data.len();
            }
        }

        let json_obj = serde_json::Value::Object(tensor_entries);
        let json_bytes = json_obj.to_string().into_bytes();

        let mut safetensors_data = Vec::new();
        safetensors_data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
        safetensors_data.extend_from_slice(&json_bytes);
        safetensors_data.extend_from_slice(&all_data);

        std::fs::write(&model_path, safetensors_data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_ok(), "Conversion failed: {:?}", result.err());

        let transformer = result.expect("operation failed");
        assert_eq!(transformer.layers.len(), num_layers);
    }

    #[test]
    fn test_extract_layer_missing_input_layernorm_ext_cov() {
        let temp_dir = TempDir::new().expect("create temp dir");
        let model_path = temp_dir.path().join("model.safetensors");
        let config_path = temp_dir.path().join("config.json");

        let config = create_config_json(16, 1, 4, 50);
        std::fs::write(&config_path, config).expect("write config");

        // Missing layer 0 input_layernorm
        let embed_data: Vec<u8> = vec![0u8; 50 * 16 * 4];
        let norm_data: Vec<u8> = vec![0u8; 16 * 4];
        let data = create_safetensors_bytes(&[
            ("model.embed_tokens.weight", "F32", &[50, 16], &embed_data),
            ("model.norm.weight", "F32", &[16], &norm_data),
        ]);
        std::fs::write(&model_path, data).expect("write safetensors");

        let result = SafetensorsToAprConverter::convert(&model_path);
        assert!(result.is_err());
    }
