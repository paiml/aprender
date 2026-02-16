
    /// Check if any tensor name matches one of the given patterns.
    fn has_tensor(names: &[&str], patterns: &[&str]) -> bool {
        names.iter().any(|n| patterns.iter().any(|p| n.contains(p)))
    }

    #[test]
    fn test_full_model_tensor_inventory_hf() {
        let names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ];

        assert!(has_tensor(&names, &["emb", "wte", "token_embd"]));
        assert!(has_tensor(&names, &["q_proj", "attn_q"]));
        assert!(has_tensor(&names, &["k_proj", "attn_k"]));
        assert!(has_tensor(&names, &["v_proj", "attn_v"]));
        assert!(has_tensor(&names, &["o_proj", "attn_output"]));
        assert!(has_tensor(&names, &["gate_proj", "ffn_gate"]));
        assert!(has_tensor(&names, &["up_proj", "ffn_up"]));
        assert!(has_tensor(&names, &["down_proj", "ffn_down"]));
        assert!(has_tensor(&names, &["input_layernorm", "attn_norm"]));
        assert!(has_tensor(&names, &["post_attention_layernorm", "ffn_norm"]));
        assert!(names.iter().any(|n| n.contains("lm_head") || *n == "output.weight"));
    }
