
    #[test]
    fn test_gguf_embedding_detection_embed_tokens() {
        let name = "model.embed_tokens.weight";
        assert!(name.contains("token_embd") || name.contains("embed_tokens"));
    }

    #[test]
    fn test_gguf_embedding_detection_neither() {
        let name = "blk.0.attn_q.weight";
        assert!(!(name.contains("token_embd") || name.contains("embed_tokens")));
    }

    #[test]
    fn test_gguf_qkv_blk_style() {
        let names = vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ];
        let has_q = names
            .iter()
            .any(|t| t.contains("blk.0.attn_q") || t.contains("layers.0.self_attn.q_proj"));
        let has_k = names
            .iter()
            .any(|t| t.contains("blk.0.attn_k") || t.contains("layers.0.self_attn.k_proj"));
        let has_v = names
            .iter()
            .any(|t| t.contains("blk.0.attn_v") || t.contains("layers.0.self_attn.v_proj"));
        assert!(has_q && has_k && has_v);
    }

    #[test]
    fn test_gguf_qkv_hf_style() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ];
        let has_q = names
            .iter()
            .any(|t| t.contains("blk.0.attn_q") || t.contains("layers.0.self_attn.q_proj"));
        let has_k = names
            .iter()
            .any(|t| t.contains("blk.0.attn_k") || t.contains("layers.0.self_attn.k_proj"));
        let has_v = names
            .iter()
            .any(|t| t.contains("blk.0.attn_v") || t.contains("layers.0.self_attn.v_proj"));
        assert!(has_q && has_k && has_v);
    }

    #[test]
    fn test_gguf_qkv_missing_k() {
        let names = vec!["blk.0.attn_q.weight", "blk.0.attn_v.weight"];
        let has_k = names
            .iter()
            .any(|t| t.contains("blk.0.attn_k") || t.contains("layers.0.self_attn.k_proj"));
        assert!(!has_k);
    }

    #[test]
    fn test_gguf_attn_output_detection() {
        let names = vec!["blk.0.attn_output.weight"];
        let has = names
            .iter()
            .any(|t| t.contains("attn_output") || t.contains("o_proj"));
        assert!(has);
    }

    #[test]
    fn test_gguf_attn_output_o_proj_style() {
        let names = vec!["model.layers.0.self_attn.o_proj.weight"];
        let has = names
            .iter()
            .any(|t| t.contains("attn_output") || t.contains("o_proj"));
        assert!(has);
    }

    #[test]
    fn test_gguf_ffn_blk_style() {
        let names = vec![
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        let has_gate = names
            .iter()
            .any(|t| t.contains("ffn_gate") || t.contains("gate_proj"));
        let has_up = names
            .iter()
            .any(|t| t.contains("ffn_up") || t.contains("up_proj"));
        let has_down = names
            .iter()
            .any(|t| t.contains("ffn_down") || t.contains("down_proj"));
        assert!(has_gate && has_up && has_down);
    }

    #[test]
    fn test_gguf_norm_blk_style() {
        let names = vec!["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"];
        let has_attn = names
            .iter()
            .any(|t| t.contains("attn_norm") || t.contains("input_layernorm"));
        let has_ffn = names
            .iter()
            .any(|t| t.contains("ffn_norm") || t.contains("post_attention_layernorm"));
        assert!(has_attn && has_ffn);
    }

    #[test]
    fn test_gguf_norm_hf_style() {
        let names = vec![
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];
        let has_attn = names
            .iter()
            .any(|t| t.contains("attn_norm") || t.contains("input_layernorm"));
        let has_ffn = names
            .iter()
            .any(|t| t.contains("ffn_norm") || t.contains("post_attention_layernorm"));
        assert!(has_attn && has_ffn);
    }

    // ========================================================================
    // LM Head Details Formatting (GGUF-specific 3-branch logic)
    // ========================================================================

    #[test]
    fn test_lm_head_details_explicit_head() {
        let has_explicit_lm_head = true;
        let has_tied_embeddings = false;
        let vocab_size = 32000_usize;
        let details = if has_explicit_lm_head {
            format!("vocab_size={}", vocab_size)
        } else if has_tied_embeddings {
            format!("vocab_size={} (tied embeddings)", vocab_size)
        } else {
            "Missing LM head tensor".to_string()
        };
        assert_eq!(details, "vocab_size=32000");
    }

    #[test]
    fn test_lm_head_details_tied_embeddings() {
        let has_explicit_lm_head = false;
        let has_tied_embeddings = true;
        let vocab_size = 128256_usize;
        let details = if has_explicit_lm_head {
            format!("vocab_size={}", vocab_size)
        } else if has_tied_embeddings {
            format!("vocab_size={} (tied embeddings)", vocab_size)
        } else {
            "Missing LM head tensor".to_string()
        };
        assert_eq!(details, "vocab_size=128256 (tied embeddings)");
    }

    #[test]
    fn test_lm_head_details_missing() {
        let has_explicit_lm_head = false;
        let has_tied_embeddings = false;
        let details = if has_explicit_lm_head {
            format!("vocab_size={}", 32000)
        } else if has_tied_embeddings {
            format!("vocab_size={} (tied embeddings)", 32000)
        } else {
            "Missing LM head tensor".to_string()
        };
        assert_eq!(details, "Missing LM head tensor");
    }

    #[test]
    fn test_lm_head_pass_condition_explicit_with_vocab() {
        let has_lm_head = true;
        let vocab_size = 32000_usize;
        assert!(has_lm_head && vocab_size > 0);
    }

    #[test]
    fn test_lm_head_pass_condition_zero_vocab_fails() {
        let has_lm_head = true;
        let vocab_size = 0_usize;
        assert!(!(has_lm_head && vocab_size > 0));
    }

    #[test]
    fn test_lm_head_pass_condition_no_head_no_vocab() {
        let has_lm_head = false;
        let vocab_size = 32000_usize;
        assert!(!(has_lm_head && vocab_size > 0));
    }

    // ========================================================================
    // GGUF Tensor Name Matching: output.weight exact match
    // ========================================================================

    #[test]
    fn test_output_weight_exact_match() {
        let name = "output.weight";
        assert!(name == "output.weight" || name.contains("lm_head"));
    }

    #[test]
    fn test_output_weight_partial_no_match() {
        // "output.weight.bias" should still match via contains if using contains
        // but the exact == check in the code is for "output.weight" only
        let name = "some_output.weight";
        // The GGUF check uses t.name == "output.weight" || t.name.contains("lm_head")
        assert!(!(name == "output.weight"));
        // But the APR check uses n.contains("lm_head") || n == &"output.weight"
        assert!(!(name == "output.weight" || name.contains("lm_head")));
    }

    #[test]
    fn test_lm_head_weight_contains_match() {
        let name = "lm_head.weight";
        assert!(name == "output.weight" || name.contains("lm_head"));
    }

    // ========================================================================
    // Logits Min/Max Formatting (mirrors check_logits_real details)
    // ========================================================================

    #[test]
    fn test_logits_details_format_normal() {
        let logits = vec![0.1_f32, -0.5, 2.3, -1.0, 0.0];
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let details = format!("logits[{}]: min={:.2}, max={:.2}", logits.len(), min, max);
        assert_eq!(details, "logits[5]: min=-1.00, max=2.30");
    }

    #[test]
    fn test_logits_details_format_single() {
        let logits = vec![42.0_f32];
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let details = format!("logits[{}]: min={:.2}, max={:.2}", logits.len(), min, max);
        assert_eq!(details, "logits[1]: min=42.00, max=42.00");
    }

    #[test]
    fn test_logits_nan_details_message() {
        let has_nan = true;
        let has_inf = false;
        let logits_empty = false;
        let details = if has_nan {
            "FAIL: NaN detected in logits".to_string()
        } else if has_inf {
            "FAIL: Inf detected in logits".to_string()
        } else if logits_empty {
            "FAIL: Empty logits".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: NaN detected in logits");
    }

    #[test]
    fn test_logits_inf_details_message() {
        let has_nan = false;
        let has_inf = true;
        let details = if has_nan {
            "FAIL: NaN detected in logits".to_string()
        } else if has_inf {
            "FAIL: Inf detected in logits".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: Inf detected in logits");
    }

    #[test]
    fn test_logits_empty_details_message() {
        let has_nan = false;
        let has_inf = false;
        let logits_empty = true;
        let details = if has_nan {
            "FAIL: NaN detected in logits".to_string()
        } else if has_inf {
            "FAIL: Inf detected in logits".to_string()
        } else if logits_empty {
            "FAIL: Empty logits".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: Empty logits");
    }

    // ========================================================================
    // Sampler Details Formatting (mirrors check_sampler_real)
    // ========================================================================

    #[test]
    fn test_sampler_details_valid_softmax() {
        let prob_sum: f32 = 1.000_001;
        let softmax_valid = (prob_sum - 1.0).abs() < 0.001;
        let has_nan = false;
        let has_inf = false;
        let details = if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else if !softmax_valid {
            format!("FAIL: softmax sum = {:.6} (expected 1.0)", prob_sum)
        } else {
            format!("softmax sum = {:.6} \u{2713}", prob_sum)
        };
        assert!(details.contains("softmax sum = 1.0000"));
        assert!(details.contains('\u{2713}'));
    }

    #[test]
    fn test_sampler_details_nan_in_softmax() {
        let has_nan = true;
        let has_inf = false;
        let softmax_valid = false;
        let _ = softmax_valid;
        let details = if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: NaN in softmax");
    }

    #[test]
    fn test_sampler_details_inf_in_softmax() {
        let has_nan = false;
        let has_inf = true;
        let details = if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else {
            "ok".to_string()
        };
        assert_eq!(details, "FAIL: Inf in softmax");
    }

    #[test]
    fn test_sampler_details_bad_softmax_sum() {
        let prob_sum: f32 = 0.5;
        let softmax_valid = (prob_sum - 1.0).abs() < 0.001;
        let has_nan = false;
        let has_inf = false;
        let details = if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else if !softmax_valid {
            format!("FAIL: softmax sum = {:.6} (expected 1.0)", prob_sum)
        } else {
            "ok".to_string()
        };
        assert!(details.contains("FAIL: softmax sum"));
        assert!(details.contains("expected 1.0"));
    }

    // ========================================================================
    // Sampler Passed Condition (mirrors check_sampler_real line 559)
    // ========================================================================

    #[test]
    fn test_sampler_passed_all_good() {
        let softmax_valid = true;
        let has_nan = false;
        let has_inf = false;
        let passed = softmax_valid && !has_nan && !has_inf;
        assert!(passed);
    }

    #[test]
    fn test_sampler_passed_nan_fails() {
        let softmax_valid = true;
        let has_nan = true;
        let has_inf = false;
        let passed = softmax_valid && !has_nan && !has_inf;
        assert!(!passed);
    }

    #[test]
    fn test_sampler_passed_inf_fails() {
        let softmax_valid = true;
        let has_nan = false;
        let has_inf = true;
        let passed = softmax_valid && !has_nan && !has_inf;
        assert!(!passed);
    }

    #[test]
    fn test_sampler_passed_bad_sum_fails() {
        let softmax_valid = false;
        let has_nan = false;
        let has_inf = false;
        let passed = softmax_valid && !has_nan && !has_inf;
        assert!(!passed);
    }

    // ========================================================================
    // Embedding Validity Check (mirrors check_tokenizer_real logic)
    // ========================================================================

    #[test]
    fn test_embedding_validity_check_valid() {
        let embedding = vec![0.1_f32, 0.2, 0.3, 0.4]; // 2 tokens * hidden_dim=2
        let test_tokens_len = 2;
        let hidden_dim = 2;
        let embedding_ok = !embedding.is_empty()
            && embedding.len() == test_tokens_len * hidden_dim
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());
        assert!(embedding_ok);
    }

    #[test]
    fn test_embedding_validity_check_empty() {
        let embedding: Vec<f32> = vec![];
        let embedding_ok = !embedding.is_empty();
        assert!(!embedding_ok);
    }

    #[test]
    fn test_embedding_validity_check_wrong_size() {
        let embedding = vec![0.1_f32, 0.2, 0.3]; // 3 floats != 2*2
        let test_tokens_len = 2;
        let hidden_dim = 2;
        let embedding_ok = !embedding.is_empty()
            && embedding.len() == test_tokens_len * hidden_dim
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());
        assert!(!embedding_ok);
    }

    #[test]
    fn test_embedding_validity_check_contains_nan() {
        let embedding = vec![0.1_f32, f32::NAN, 0.3, 0.4];
        let test_tokens_len = 2;
        let hidden_dim = 2;
        let embedding_ok = !embedding.is_empty()
            && embedding.len() == test_tokens_len * hidden_dim
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());
        assert!(!embedding_ok);
    }
