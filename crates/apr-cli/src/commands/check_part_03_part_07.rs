
    #[test]
    fn test_full_model_tensor_inventory_hf() {
        // Simulate a complete HF-style model's tensor names
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

        // Check APR-style detection (used in run_real_checks_apr)
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        let has_q = names
            .iter()
            .any(|n| n.contains("q_proj") || n.contains("attn_q"));
        let has_k = names
            .iter()
            .any(|n| n.contains("k_proj") || n.contains("attn_k"));
        let has_v = names
            .iter()
            .any(|n| n.contains("v_proj") || n.contains("attn_v"));
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("o_proj") || n.contains("attn_output"));
        let has_gate = names
            .iter()
            .any(|n| n.contains("gate_proj") || n.contains("ffn_gate"));
        let has_up = names
            .iter()
            .any(|n| n.contains("up_proj") || n.contains("ffn_up"));
        let has_down = names
            .iter()
            .any(|n| n.contains("down_proj") || n.contains("ffn_down"));
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("input_layernorm") || n.contains("attn_norm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("post_attention_layernorm") || n.contains("ffn_norm"));
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");

        assert!(has_embed);
        assert!(has_q && has_k && has_v);
        assert!(has_attn_out);
        assert!(has_gate && has_up && has_down);
        assert!(has_attn_norm && has_ffn_norm);
        assert!(has_lm_head);
    }
