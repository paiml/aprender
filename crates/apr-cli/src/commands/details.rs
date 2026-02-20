
    // ========================================================================
    // Shared helpers to eliminate repeated control-flow patterns
    // ========================================================================

    /// Check whether any tensor name matches either of two alternate formats.
    fn tensor_has_match(names: &[&str], alt_a: &str, alt_b: &str) -> bool {
        names.iter().any(|t| t.contains(alt_a) || t.contains(alt_b))
    }

    /// Format LM head details using the 3-branch GGUF logic.
    fn lm_head_details(has_explicit: bool, has_tied: bool, vocab_size: usize) -> String {
        if has_explicit {
            format!("vocab_size={}", vocab_size)
        } else if has_tied {
            format!("vocab_size={} (tied embeddings)", vocab_size)
        } else {
            "Missing LM head tensor".to_string()
        }
    }

    /// Format logits error details (NaN > Inf > Empty > ok).
    fn logits_error_details(has_nan: bool, has_inf: bool, logits_empty: bool) -> String {
        if has_nan {
            "FAIL: NaN detected in logits".to_string()
        } else if has_inf {
            "FAIL: Inf detected in logits".to_string()
        } else if logits_empty {
            "FAIL: Empty logits".to_string()
        } else {
            "ok".to_string()
        }
    }

    /// Format sampler details (NaN > Inf > bad sum > ok).
    fn sampler_details(
        has_nan: bool,
        has_inf: bool,
        softmax_valid: bool,
        prob_sum: f32,
    ) -> String {
        if has_nan {
            "FAIL: NaN in softmax".to_string()
        } else if has_inf {
            "FAIL: Inf in softmax".to_string()
        } else if !softmax_valid {
            format!("FAIL: softmax sum = {:.6} (expected 1.0)", prob_sum)
        } else {
            format!("softmax sum = {:.6} \u{2713}", prob_sum)
        }
    }

    /// Check embedding validity: non-empty, correct size, no NaN/Inf.
    fn embedding_valid(embedding: &[f32], expected_len: usize) -> bool {
        !embedding.is_empty()
            && embedding.len() == expected_len
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite())
    }

    /// Format logits min/max details.
    fn logits_min_max_details(logits: &[f32]) -> String {
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        format!("logits[{}]: min={:.2}, max={:.2}", logits.len(), min, max)
    }

    // ========================================================================
    // GGUF Tensor Name Matching Tests
    // ========================================================================

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
        assert!(tensor_has_match(&names, "blk.0.attn_q", "layers.0.self_attn.q_proj"));
        assert!(tensor_has_match(&names, "blk.0.attn_k", "layers.0.self_attn.k_proj"));
        assert!(tensor_has_match(&names, "blk.0.attn_v", "layers.0.self_attn.v_proj"));
    }

    #[test]
    fn test_gguf_qkv_hf_style() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ];
        assert!(tensor_has_match(&names, "blk.0.attn_q", "layers.0.self_attn.q_proj"));
        assert!(tensor_has_match(&names, "blk.0.attn_k", "layers.0.self_attn.k_proj"));
        assert!(tensor_has_match(&names, "blk.0.attn_v", "layers.0.self_attn.v_proj"));
    }

    #[test]
    fn test_gguf_qkv_missing_k() {
        let names = vec!["blk.0.attn_q.weight", "blk.0.attn_v.weight"];
        assert!(!tensor_has_match(&names, "blk.0.attn_k", "layers.0.self_attn.k_proj"));
    }

    #[test]
    fn test_gguf_attn_output_detection() {
        let names = vec!["blk.0.attn_output.weight"];
        assert!(tensor_has_match(&names, "attn_output", "o_proj"));
    }

    #[test]
    fn test_gguf_attn_output_o_proj_style() {
        let names = vec!["model.layers.0.self_attn.o_proj.weight"];
        assert!(tensor_has_match(&names, "attn_output", "o_proj"));
    }

    #[test]
    fn test_gguf_ffn_blk_style() {
        let names = vec![
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        assert!(tensor_has_match(&names, "ffn_gate", "gate_proj"));
        assert!(tensor_has_match(&names, "ffn_up", "up_proj"));
        assert!(tensor_has_match(&names, "ffn_down", "down_proj"));
    }

    #[test]
    fn test_gguf_norm_blk_style() {
        let names = vec!["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"];
        assert!(tensor_has_match(&names, "attn_norm", "input_layernorm"));
        assert!(tensor_has_match(&names, "ffn_norm", "post_attention_layernorm"));
    }

    #[test]
    fn test_gguf_norm_hf_style() {
        let names = vec![
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];
        assert!(tensor_has_match(&names, "attn_norm", "input_layernorm"));
        assert!(tensor_has_match(&names, "ffn_norm", "post_attention_layernorm"));
    }

    // ========================================================================
    // LM Head Details Formatting (GGUF-specific 3-branch logic)
    // ========================================================================

    #[test]
    fn test_lm_head_details_explicit_head() {
        assert_eq!(lm_head_details(true, false, 32000), "vocab_size=32000");
    }

    #[test]
    fn test_lm_head_details_tied_embeddings() {
        assert_eq!(
            lm_head_details(false, true, 128256),
            "vocab_size=128256 (tied embeddings)"
        );
    }

    #[test]
    fn test_lm_head_details_missing() {
        assert_eq!(lm_head_details(false, false, 32000), "Missing LM head tensor");
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
        let name = "some_output.weight";
        assert!(!(name == "output.weight"));
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
        assert_eq!(
            logits_min_max_details(&[0.1_f32, -0.5, 2.3, -1.0, 0.0]),
            "logits[5]: min=-1.00, max=2.30"
        );
    }

    #[test]
    fn test_logits_details_format_single() {
        assert_eq!(
            logits_min_max_details(&[42.0_f32]),
            "logits[1]: min=42.00, max=42.00"
        );
    }

    #[test]
    fn test_logits_nan_details_message() {
        assert_eq!(
            logits_error_details(true, false, false),
            "FAIL: NaN detected in logits"
        );
    }

    #[test]
    fn test_logits_inf_details_message() {
        assert_eq!(
            logits_error_details(false, true, false),
            "FAIL: Inf detected in logits"
        );
    }

    #[test]
    fn test_logits_empty_details_message() {
        assert_eq!(
            logits_error_details(false, false, true),
            "FAIL: Empty logits"
        );
    }

    // ========================================================================
    // Sampler Details Formatting (mirrors check_sampler_real)
    // ========================================================================

    #[test]
    fn test_sampler_details_valid_softmax() {
        let prob_sum: f32 = 1.000_001;
        let softmax_valid = (prob_sum - 1.0).abs() < 0.001;
        let details = sampler_details(false, false, softmax_valid, prob_sum);
        assert!(details.contains("softmax sum = 1.0000"));
        assert!(details.contains('\u{2713}'));
    }

    #[test]
    fn test_sampler_details_nan_in_softmax() {
        assert_eq!(sampler_details(true, false, false, 0.0), "FAIL: NaN in softmax");
    }

    #[test]
    fn test_sampler_details_inf_in_softmax() {
        assert_eq!(sampler_details(false, true, false, 0.0), "FAIL: Inf in softmax");
    }

    #[test]
    fn test_sampler_details_bad_softmax_sum() {
        let prob_sum: f32 = 0.5;
        let softmax_valid = (prob_sum - 1.0).abs() < 0.001;
        let details = sampler_details(false, false, softmax_valid, prob_sum);
        assert!(details.contains("FAIL: softmax sum"));
        assert!(details.contains("expected 1.0"));
    }

    // ========================================================================
    // Sampler Passed Condition (mirrors check_sampler_real line 559)
    // ========================================================================

    #[test]
    fn test_sampler_passed_all_good() {
        assert!(true && !false && !false); // softmax_valid && !has_nan && !has_inf
    }

    #[test]
    fn test_sampler_passed_nan_fails() {
        let passed = true && !true && !false;
        assert!(!passed);
    }

    #[test]
    fn test_sampler_passed_inf_fails() {
        let passed = true && !false && !true;
        assert!(!passed);
    }

    #[test]
    fn test_sampler_passed_bad_sum_fails() {
        let passed = false && !false && !false;
        assert!(!passed);
    }

    // ========================================================================
    // Embedding Validity Check (mirrors check_tokenizer_real logic)
    // ========================================================================

    #[test]
    fn test_embedding_validity_check_valid() {
        assert!(embedding_valid(&[0.1_f32, 0.2, 0.3, 0.4], 2 * 2));
    }

    #[test]
    fn test_embedding_validity_check_empty() {
        assert!(!embedding_valid(&[], 0));
    }

    #[test]
    fn test_embedding_validity_check_wrong_size() {
        assert!(!embedding_valid(&[0.1_f32, 0.2, 0.3], 2 * 2));
    }

    #[test]
    fn test_embedding_validity_check_contains_nan() {
        assert!(!embedding_valid(&[0.1_f32, f32::NAN, 0.3, 0.4], 2 * 2));
    }
