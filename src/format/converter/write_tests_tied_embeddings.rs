
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // =========================================================================
    // resolve_f32_tied_embeddings
    // =========================================================================

    #[test]
    fn test_resolve_f32_tied_no_lm_head_has_embed_tokens_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![5.0, 6.0], vec![1, 2]),
        );

        let (result, has_tied) = resolve_f32_tied_embeddings(&tensors);

        assert!(has_tied);
        assert!(result.contains_key("lm_head.weight"));
        let (data, shape) = &result["lm_head.weight"];
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, &[2, 2]);
    }

    #[test]
    fn test_resolve_f32_tied_already_has_lm_head_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![1.0, 2.0], vec![1, 2]),
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            (vec![3.0, 4.0], vec![1, 2]),
        );

        let (result, has_tied) = resolve_f32_tied_embeddings(&tensors);

        assert!(!has_tied);
        // lm_head should remain unchanged
        assert_eq!(result["lm_head.weight"].0, vec![3.0, 4.0]);
    }

    #[test]
    fn test_resolve_f32_tied_no_lm_head_no_embed_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.weight".to_string(),
            (vec![1.0], vec![1]),
        );

        let (result, has_tied) = resolve_f32_tied_embeddings(&tensors);

        assert!(!has_tied);
        assert!(!result.contains_key("lm_head.weight"));
    }

    #[test]
    fn test_resolve_f32_tied_gguf_token_embd_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "token_embd.weight".to_string(),
            (vec![10.0, 20.0], vec![1, 2]),
        );

        let (result, has_tied) = resolve_f32_tied_embeddings(&tensors);

        assert!(has_tied);
        assert_eq!(result["lm_head.weight"].0, vec![10.0, 20.0]);
    }

    #[test]
    fn test_resolve_f32_tied_gpt2_wte_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "wte.weight".to_string(),
            (vec![7.0, 8.0, 9.0], vec![3]),
        );

        let (result, has_tied) = resolve_f32_tied_embeddings(&tensors);

        assert!(has_tied);
        assert_eq!(result["lm_head.weight"].0, vec![7.0, 8.0, 9.0]);
    }

    // =========================================================================
    // resolve_tied_embeddings (raw GgufRawTensor version)
    // =========================================================================

    #[test]
    fn test_resolve_tied_raw_no_lm_head_gh219() {
        let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            GgufRawTensor {
                data: vec![1, 2, 3, 4],
                shape: vec![2, 2],
                dtype: 0,
            },
        );

        let (result, has_tied) = resolve_tied_embeddings(&tensors);

        assert!(has_tied);
        assert!(result.contains_key("lm_head.weight"));
        assert_eq!(result["lm_head.weight"].data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_resolve_tied_raw_has_output_weight_gh219() {
        let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            GgufRawTensor {
                data: vec![1, 2],
                shape: vec![2],
                dtype: 0,
            },
        );
        tensors.insert(
            "output.weight".to_string(),
            GgufRawTensor {
                data: vec![3, 4],
                shape: vec![2],
                dtype: 0,
            },
        );

        let (result, has_tied) = resolve_tied_embeddings(&tensors);

        assert!(!has_tied);
        // output.weight counts as lm_head, so no tie
        assert!(!result.contains_key("lm_head.weight"));
    }

    #[test]
    fn test_resolve_tied_raw_has_lm_head_weight_gh219() {
        let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
        tensors.insert(
            "lm_head.weight".to_string(),
            GgufRawTensor {
                data: vec![5, 6],
                shape: vec![2],
                dtype: 0,
            },
        );

        let (result, has_tied) = resolve_tied_embeddings(&tensors);

        assert!(!has_tied);
        assert_eq!(result.len(), 1);
    }

    // =========================================================================
    // insert_f32_tokenizer_metadata
    // =========================================================================

    #[test]
    fn test_insert_f32_tokenizer_metadata_full_gh219() {
        let tok = GgufTokenizer {
            vocabulary: vec!["hello".to_string(), "world".to_string()],
            merges: vec!["h e".to_string(), "ll o".to_string()],
            model_type: Some("bpe".to_string()),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            architecture: Some("qwen2".to_string()),
            model_name: Some("test-model".to_string()),
            ..Default::default()
        };
        let mut custom = std::collections::HashMap::new();

        insert_f32_tokenizer_metadata(&tok, &mut custom);

        assert!(custom.contains_key("tokenizer.vocabulary"));
        assert!(custom.contains_key("tokenizer.vocab_size"));
        assert!(custom.contains_key("tokenizer.model_type"));
        assert!(custom.contains_key("tokenizer.bos_token_id"));
        assert!(custom.contains_key("tokenizer.eos_token_id"));
        assert!(custom.contains_key("tokenizer.architecture"));
        assert!(custom.contains_key("tokenizer.model_name"));
        assert!(custom.contains_key("tokenizer.merges"));

        // Check vocab_size value
        assert_eq!(
            custom["tokenizer.vocab_size"],
            serde_json::Value::Number(serde_json::Number::from(2))
        );
    }

    #[test]
    fn test_insert_f32_tokenizer_metadata_minimal_gh219() {
        let tok = GgufTokenizer::default(); // all empty/None
        let mut custom = std::collections::HashMap::new();

        insert_f32_tokenizer_metadata(&tok, &mut custom);

        // Empty vocabulary means no vocab keys inserted
        assert!(!custom.contains_key("tokenizer.vocabulary"));
        assert!(!custom.contains_key("tokenizer.vocab_size"));
        assert!(!custom.contains_key("tokenizer.model_type"));
        assert!(!custom.contains_key("tokenizer.bos_token_id"));
        assert!(!custom.contains_key("tokenizer.eos_token_id"));
        assert!(!custom.contains_key("tokenizer.merges"));
    }

    #[test]
    fn test_insert_f32_tokenizer_metadata_vocab_only_gh219() {
        let tok = GgufTokenizer {
            vocabulary: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            ..Default::default()
        };
        let mut custom = std::collections::HashMap::new();

        insert_f32_tokenizer_metadata(&tok, &mut custom);

        assert!(custom.contains_key("tokenizer.vocabulary"));
        assert_eq!(
            custom["tokenizer.vocab_size"],
            serde_json::Value::Number(serde_json::Number::from(3))
        );
        // No other keys should be present
        assert!(!custom.contains_key("tokenizer.model_type"));
        assert!(!custom.contains_key("tokenizer.merges"));
    }

    // =========================================================================
    // insert_tokenizer_metadata (raw/GGUF version)
    // =========================================================================

    #[test]
    fn test_insert_tokenizer_metadata_full_gh219() {
        let tok = GgufTokenizer {
            vocabulary: vec!["x".to_string()],
            merges: vec!["a b".to_string()],
            model_type: Some("spm".to_string()),
            bos_token_id: Some(0),
            eos_token_id: Some(1),
            token_type: vec![1, 3],
            padding_token_id: Some(99),
            add_bos_token: Some(true),
            chat_template: Some("{{prompt}}".to_string()),
            ..Default::default()
        };
        let mut custom = std::collections::HashMap::new();

        insert_tokenizer_metadata(&tok, &mut custom);

        assert!(custom.contains_key("tokenizer.vocabulary"));
        assert!(custom.contains_key("tokenizer.vocab_size"));
        assert!(custom.contains_key("tokenizer.model"));
        assert!(custom.contains_key("tokenizer.bos_token_id"));
        assert!(custom.contains_key("tokenizer.eos_token_id"));
        assert!(custom.contains_key("tokenizer.merges"));
        assert!(custom.contains_key("tokenizer.token_type"));
        assert!(custom.contains_key("tokenizer.padding_token_id"));
        assert!(custom.contains_key("tokenizer.add_bos_token"));
        assert!(custom.contains_key("tokenizer.chat_template"));
    }

    #[test]
    fn test_insert_tokenizer_metadata_empty_gh219() {
        let tok = GgufTokenizer::default();
        let mut custom = std::collections::HashMap::new();

        insert_tokenizer_metadata(&tok, &mut custom);

        assert!(custom.is_empty());
    }

    // =========================================================================
    // build_f32_custom_metadata
    // =========================================================================

    #[test]
    fn test_build_f32_custom_metadata_basic_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("weight".to_string(), (vec![1.0], vec![1]));
        let user_metadata = UserMetadata::new();

        let custom = build_f32_custom_metadata(&tensors, &user_metadata, false, None);

        assert!(custom.contains_key("tensor_shapes"));
        assert!(!custom.contains_key("tied_embeddings"));
        assert!(!custom.contains_key("source_metadata"));
    }

    #[test]
    fn test_build_f32_custom_metadata_with_tied_gh219() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let user_metadata = UserMetadata::new();

        let custom = build_f32_custom_metadata(&tensors, &user_metadata, true, None);

        assert!(custom.contains_key("tied_embeddings"));
        assert_eq!(
            custom["tied_embeddings"],
            serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn test_build_f32_custom_metadata_with_user_metadata_gh219() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let mut user_metadata = UserMetadata::new();
        user_metadata.insert("format".to_string(), "pt".to_string());

        let custom = build_f32_custom_metadata(&tensors, &user_metadata, false, None);

        assert!(custom.contains_key("source_metadata"));
    }

    #[test]
    fn test_build_f32_custom_metadata_with_tokenizer_gh219() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let user_metadata = UserMetadata::new();
        let tok = GgufTokenizer {
            vocabulary: vec!["tok".to_string()],
            ..Default::default()
        };

        let custom = build_f32_custom_metadata(&tensors, &user_metadata, false, Some(&tok));

        assert!(custom.contains_key("tokenizer.vocabulary"));
    }

    #[test]
    fn test_build_f32_custom_metadata_tensor_shapes_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("a".to_string(), (vec![1.0, 2.0], vec![2]));
        tensors.insert("b".to_string(), (vec![1.0; 6], vec![2, 3]));
        let user_metadata = UserMetadata::new();

        let custom = build_f32_custom_metadata(&tensors, &user_metadata, false, None);

        let shapes = &custom["tensor_shapes"];
        assert!(shapes.is_object());
        let obj = shapes.as_object().unwrap();
        assert!(obj.contains_key("a"));
        assert!(obj.contains_key("b"));
        // Check shape of "b" is [2, 3]
        let b_shape = obj["b"].as_array().unwrap();
        assert_eq!(b_shape.len(), 2);
        assert_eq!(b_shape[0].as_u64().unwrap(), 2);
        assert_eq!(b_shape[1].as_u64().unwrap(), 3);
    }

    // =========================================================================
    // insert_model_config_metadata
    // =========================================================================

    #[test]
    fn test_insert_model_config_metadata_full_gh219() {
        let cfg = GgufModelConfig {
            architecture: Some("llama".to_string()),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(8),
            vocab_size: Some(32000),
            intermediate_size: Some(11008),
            max_position_embeddings: Some(4096),
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-5),
            rope_type: Some(0),
        };
        let mut custom = std::collections::HashMap::new();

        insert_model_config_metadata(&cfg, &mut custom);

        assert_eq!(custom["model.architecture"], serde_json::json!("llama"));
        assert_eq!(custom["model.hidden_size"], serde_json::json!(4096));
        assert_eq!(custom["model.num_layers"], serde_json::json!(32));
        assert_eq!(custom["model.num_heads"], serde_json::json!(32));
        assert_eq!(custom["model.num_kv_heads"], serde_json::json!(8));
        assert_eq!(custom["model.vocab_size"], serde_json::json!(32000));
        assert_eq!(custom["model.intermediate_size"], serde_json::json!(11008));
        assert_eq!(custom["model.max_position_embeddings"], serde_json::json!(4096));
        assert!(custom.contains_key("model.rope_theta"));
        assert!(custom.contains_key("model.rms_norm_eps"));
        assert!(custom.contains_key("model.rope_type"));
    }

    #[test]
    fn test_insert_model_config_metadata_empty_gh219() {
        let cfg = GgufModelConfig::default();
        let mut custom = std::collections::HashMap::new();

        insert_model_config_metadata(&cfg, &mut custom);

        assert!(custom.is_empty());
    }

    // =========================================================================
    // map_gguf_dtype
    // =========================================================================

    #[test]
    fn test_map_gguf_dtype_supported_gh219() {
        assert!(matches!(map_gguf_dtype(0, "test"), Ok(TensorDType::F32)));
        assert!(matches!(map_gguf_dtype(1, "test"), Ok(TensorDType::F16)));
        assert!(matches!(map_gguf_dtype(12, "test"), Ok(TensorDType::Q4K)));
        assert!(matches!(map_gguf_dtype(14, "test"), Ok(TensorDType::Q6K)));
    }

    #[test]
    fn test_map_gguf_dtype_unsupported_with_suggestion_gh219() {
        // Q4_0 (2), Q4_1 (3), Q5_0 (6), Q8_0 (8), Q5_K (13)
        for dtype in [2, 3, 6, 8, 13] {
            let result = map_gguf_dtype(dtype, "test_tensor");
            assert!(result.is_err(), "dtype {dtype} should be unsupported");
            let msg = format!("{}", result.unwrap_err());
            assert!(msg.contains("test_tensor"), "Error should mention tensor name");
        }
    }

    #[test]
    fn test_map_gguf_dtype_unsupported_q5_1_q8_1_gh219() {
        for dtype in [7, 9] {
            let result = map_gguf_dtype(dtype, "layer.weight");
            assert!(result.is_err());
            let msg = format!("{}", result.unwrap_err());
            assert!(msg.contains("LAYOUT-002"));
        }
    }

    #[test]
    fn test_map_gguf_dtype_unknown_gh219() {
        let result = map_gguf_dtype(255, "unknown.tensor");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Unsupported GGUF dtype 255"));
    }
}
