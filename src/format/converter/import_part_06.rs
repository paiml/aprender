
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // =========================================================================
    // infer_embedding_dims
    // =========================================================================

    #[test]
    fn test_infer_embedding_dims_hf_order_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 4096], vec![32000, 4096]),
        );

        let result = infer_embedding_dims(&tensors);
        assert_eq!(result, Some((32000, 4096)));
    }

    #[test]
    fn test_infer_embedding_dims_gguf_order_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GGUF: [hidden_size, vocab_size] — smaller first
        tensors.insert(
            "token_embd.weight".to_string(),
            (vec![0.0; 4096 * 32000], vec![4096, 32000]),
        );

        let result = infer_embedding_dims(&tensors);
        assert_eq!(result, Some((32000, 4096)));
    }

    #[test]
    fn test_infer_embedding_dims_wte_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "wte.weight".to_string(),
            (vec![0.0; 50257 * 768], vec![50257, 768]),
        );

        let result = infer_embedding_dims(&tensors);
        assert_eq!(result, Some((50257, 768)));
    }

    #[test]
    fn test_infer_embedding_dims_no_embedding_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.weight".to_string(),
            (vec![0.0; 4], vec![2, 2]),
        );

        assert_eq!(infer_embedding_dims(&tensors), None);
    }

    #[test]
    fn test_infer_embedding_dims_1d_embedding_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 100], vec![100]),
        );

        assert_eq!(infer_embedding_dims(&tensors), None);
    }

    // =========================================================================
    // count_transformer_layers
    // =========================================================================

    #[test]
    fn test_count_transformer_layers_hf_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        for i in 0..4 {
            tensors.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                (vec![1.0], vec![1]),
            );
        }

        assert_eq!(count_transformer_layers(&tensors), 4);
    }

    #[test]
    fn test_count_transformer_layers_gguf_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        for i in 0..8 {
            tensors.insert(
                format!("blk.{i}.attn_q.weight"),
                (vec![1.0], vec![1]),
            );
        }

        assert_eq!(count_transformer_layers(&tensors), 8);
    }

    #[test]
    fn test_count_transformer_layers_gpt2_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        for i in 0..12 {
            tensors.insert(
                format!("h.{i}.attn.c_attn.weight"),
                (vec![1.0], vec![1]),
            );
        }

        assert_eq!(count_transformer_layers(&tensors), 12);
    }

    #[test]
    fn test_count_transformer_layers_empty_gh219() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        assert_eq!(count_transformer_layers(&tensors), 0);
    }

    #[test]
    fn test_count_transformer_layers_no_layer_tensors_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("model.embed_tokens.weight".to_string(), (vec![1.0], vec![1]));
        tensors.insert("model.norm.weight".to_string(), (vec![1.0], vec![1]));

        assert_eq!(count_transformer_layers(&tensors), 0);
    }

    // =========================================================================
    // find_projection_dim
    // =========================================================================

    #[test]
    fn test_find_projection_dim_found_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 4096 * 4096], vec![4096, 4096]),
        );

        let result = find_projection_dim(&tensors, &["q_proj"]);
        assert_eq!(result, Some(4096));
    }

    #[test]
    fn test_find_projection_dim_rectangular_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GQA: k_proj has smaller output dim
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 1024 * 4096], vec![1024, 4096]),
        );

        let result = find_projection_dim(&tensors, &["k_proj"]);
        assert_eq!(result, Some(1024));
    }

    #[test]
    fn test_find_projection_dim_not_found_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("model.norm.weight".to_string(), (vec![1.0], vec![1]));

        assert_eq!(find_projection_dim(&tensors, &["q_proj"]), None);
    }

    #[test]
    fn test_find_projection_dim_1d_tensor_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            (vec![0.0; 4096], vec![4096]),
        );

        // 1D tensor should return None
        assert_eq!(find_projection_dim(&tensors, &["q_proj"]), None);
    }

    // =========================================================================
    // infer_gqa_heads / infer_mha_heads / infer_head_counts
    // =========================================================================

    #[test]
    fn test_infer_gqa_heads_qwen2_gh219() {
        // HEAD_DIMS tries [64,128,96,80]; 4096/64=64 heads, 512/64=8 kv heads
        let (n_heads, n_kv) = infer_gqa_heads(4096, 512);
        assert_eq!(n_heads, Some(64));
        assert_eq!(n_kv, Some(8));
    }

    #[test]
    fn test_infer_gqa_heads_llama3_gh219() {
        // HEAD_DIMS tries 64 first: 4096/64=64 heads, 1024/64=16 kv heads
        let (n_heads, n_kv) = infer_gqa_heads(4096, 1024);
        assert_eq!(n_heads, Some(64));
        assert_eq!(n_kv, Some(16));
    }

    #[test]
    fn test_infer_gqa_heads_no_match_gh219() {
        // Dimensions don't match any common head_dim
        let (n_heads, n_kv) = infer_gqa_heads(1000, 500);
        assert_eq!(n_heads, None);
        assert_eq!(n_kv, None);
    }

    #[test]
    fn test_infer_mha_heads_4096_gh219() {
        // HEAD_DIMS tries 64 first: 4096/64 = 64 heads
        let (n_heads, n_kv) = infer_mha_heads(4096);
        assert_eq!(n_heads, Some(64));
        assert_eq!(n_kv, Some(64));
    }

    #[test]
    fn test_infer_mha_heads_768_gh219() {
        // 768 / 64 = 12 heads (GPT-2 small)
        let (n_heads, n_kv) = infer_mha_heads(768);
        assert_eq!(n_heads, Some(12));
        assert_eq!(n_kv, Some(12));
    }

    #[test]
    fn test_infer_mha_heads_no_match_gh219() {
        // 100 doesn't divide evenly by any HEAD_DIM
        let (n_heads, n_kv) = infer_mha_heads(100);
        assert_eq!(n_heads, None);
        assert_eq!(n_kv, None);
    }

    #[test]
    fn test_infer_head_counts_gqa_gh219() {
        // GQA: kv < q → dispatches to infer_gqa_heads (head_dim=64 tried first)
        let (n_heads, n_kv) = infer_head_counts(Some(4096), Some(512), 4096);
        assert_eq!(n_heads, Some(64));
        assert_eq!(n_kv, Some(8));
    }

    #[test]
    fn test_infer_head_counts_mha_gh219() {
        // MHA: q == hidden_size → dispatches to infer_mha_heads (head_dim=64 first)
        let (n_heads, n_kv) = infer_head_counts(Some(4096), Some(4096), 4096);
        assert_eq!(n_heads, Some(64));
        assert_eq!(n_kv, Some(64));
    }

    #[test]
    fn test_infer_head_counts_no_q_dim_gh219() {
        let (n_heads, n_kv) = infer_head_counts(None, Some(512), 4096);
        assert_eq!(n_heads, None);
        assert_eq!(n_kv, None);
    }

    // =========================================================================
    // infer_intermediate_size_from_tensors
    // =========================================================================

    #[test]
    fn test_infer_intermediate_size_gate_proj_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            (vec![0.0; 4096 * 11008], vec![11008, 4096]),
        );

        let result = infer_intermediate_size_from_tensors(&tensors);
        assert_eq!(result, Some(11008));
    }

    #[test]
    fn test_infer_intermediate_size_ffn_gate_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "blk.0.ffn_gate.weight".to_string(),
            (vec![0.0; 4096 * 11008], vec![4096, 11008]),
        );

        let result = infer_intermediate_size_from_tensors(&tensors);
        assert_eq!(result, Some(11008));
    }

    #[test]
    fn test_infer_intermediate_size_not_found_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("model.norm.weight".to_string(), (vec![1.0], vec![1]));

        assert_eq!(infer_intermediate_size_from_tensors(&tensors), None);
    }

    // =========================================================================
    // infer_architecture_from_names
    // =========================================================================

    #[test]
    fn test_infer_arch_qwen2_with_attn_bias_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![1.0], vec![1]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            (vec![1.0], vec![1]),
        );

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("qwen2".to_string()));
    }

    #[test]
    fn test_infer_arch_llama_no_bias_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![1.0], vec![1]),
        );
        // No bias → LLaMA

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("llama".to_string()));
    }

    #[test]
    fn test_infer_arch_gpt2_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "transformer.h.0.attn.c_attn.weight".to_string(),
            (vec![1.0], vec![1]),
        );

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("gpt2".to_string()));
    }

    #[test]
    fn test_infer_arch_gpt2_safetensors_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // GH-255: SafeTensors GPT-2 uses "h.N.*" prefix
        tensors.insert(
            "h.0.attn.c_attn.weight".to_string(),
            (vec![1.0], vec![1]),
        );

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("gpt2".to_string()));
    }

    #[test]
    fn test_infer_arch_gguf_blk_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            (vec![1.0], vec![1]),
        );

        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("unknown".to_string()));
    }

    #[test]
    fn test_infer_arch_empty_tensors_gh219() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let result = infer_architecture_from_names(&tensors);
        assert_eq!(result, Some("unknown".to_string()));
    }

    // =========================================================================
    // map_tensor_names
    // =========================================================================

    #[test]
    fn test_map_tensor_names_qwen2_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("blk.0.attn_q.weight".to_string(), (vec![1.0], vec![1]));
        tensors.insert("token_embd.weight".to_string(), (vec![2.0], vec![1]));

        let mapped = map_tensor_names(&tensors, Architecture::Qwen2);

        assert!(mapped.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(mapped.contains_key("model.embed_tokens.weight"));
        assert!(!mapped.contains_key("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_map_tensor_names_auto_preserves_gh219() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("model.layers.0.weight".to_string(), (vec![1.0], vec![1]));

        let mapped = map_tensor_names(&tensors, Architecture::Auto);

        assert!(mapped.contains_key("model.layers.0.weight"));
    }

    // =========================================================================
    // check_special_values
    // =========================================================================

    #[test]
    fn test_check_special_values_clean_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::Basic,
            ..Default::default()
        };

        let errors = check_special_values("test", &stats, &options);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_check_special_values_with_nan_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 5,
            inf_count: 0,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::Basic,
            ..Default::default()
        };

        let errors = check_special_values("test.weight", &stats, &options);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("NaN"));
        assert!(errors[0].contains("5"));
    }

    #[test]
    fn test_check_special_values_with_inf_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 3,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::Basic,
            ..Default::default()
        };

        let errors = check_special_values("test.weight", &stats, &options);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Inf"));
    }

    #[test]
    fn test_check_special_values_with_both_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 2,
            inf_count: 1,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::Strict,
            ..Default::default()
        };

        let errors = check_special_values("test.weight", &stats, &options);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_check_special_values_none_validation_gh219() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 10,
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            nan_count: 100,
            inf_count: 50,
            zero_count: 0,
        };
        let options = ImportOptions {
            validation: ValidationConfig::None,
            ..Default::default()
        };

        let errors = check_special_values("test", &stats, &options);
        assert!(errors.is_empty());
    }

    // =========================================================================
    // has_required_tensor
    // =========================================================================

    #[test]
    fn test_has_required_tensor_found_gh219() {
        let mut names = std::collections::HashSet::new();
        names.insert("model.norm.weight");
        names.insert("lm_head.weight");

        assert!(has_required_tensor(&names, &["model.norm.weight", "norm.weight"]));
    }

    #[test]
    fn test_has_required_tensor_not_found_gh219() {
        let mut names = std::collections::HashSet::new();
        names.insert("model.layers.0.weight");

        assert!(!has_required_tensor(&names, &["model.norm.weight", "norm.weight"]));
    }

    #[test]
    fn test_has_required_tensor_empty_names_gh219() {
        let names = std::collections::HashSet::new();
        assert!(!has_required_tensor(&names, &["model.norm.weight"]));
    }
}
