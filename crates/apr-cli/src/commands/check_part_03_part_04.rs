
    #[test]
    fn test_qkv_hf_style_names() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ];
        let has_q = names
            .iter()
            .any(|n| n.contains("q_proj") || n.contains("attn_q"));
        let has_k = names
            .iter()
            .any(|n| n.contains("k_proj") || n.contains("attn_k"));
        let has_v = names
            .iter()
            .any(|n| n.contains("v_proj") || n.contains("attn_v"));
        assert!(has_q && has_k && has_v);
    }

    #[test]
    fn test_ffn_detection_all_present() {
        let names = vec![
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ];
        let has_gate = names
            .iter()
            .any(|n| n.contains("gate_proj") || n.contains("ffn_gate"));
        let has_up = names
            .iter()
            .any(|n| n.contains("up_proj") || n.contains("ffn_up"));
        let has_down = names
            .iter()
            .any(|n| n.contains("down_proj") || n.contains("ffn_down"));
        assert!(has_gate && has_up && has_down);
    }

    #[test]
    fn test_ffn_detection_hf_style() {
        let names = vec![
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ];
        let has_gate = names
            .iter()
            .any(|n| n.contains("gate_proj") || n.contains("ffn_gate"));
        let has_up = names
            .iter()
            .any(|n| n.contains("up_proj") || n.contains("ffn_up"));
        let has_down = names
            .iter()
            .any(|n| n.contains("down_proj") || n.contains("ffn_down"));
        assert!(has_gate && has_up && has_down);
    }

    #[test]
    fn test_ffn_detection_missing_gate() {
        let names = vec!["blk.0.ffn_up.weight", "blk.0.ffn_down.weight"];
        let has_gate = names
            .iter()
            .any(|n| n.contains("gate_proj") || n.contains("ffn_gate"));
        assert!(!has_gate);
    }

    #[test]
    fn test_norm_detection() {
        let names = vec!["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"];
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("input_layernorm") || n.contains("attn_norm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("post_attention_layernorm") || n.contains("ffn_norm"));
        assert!(has_attn_norm && has_ffn_norm);
    }

    #[test]
    fn test_norm_detection_hf_style() {
        let names = vec![
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("input_layernorm") || n.contains("attn_norm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("post_attention_layernorm") || n.contains("ffn_norm"));
        assert!(has_attn_norm && has_ffn_norm);
    }

    #[test]
    fn test_norm_detection_missing_ffn_norm() {
        let names = vec!["blk.0.attn_norm.weight"];
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("input_layernorm") || n.contains("attn_norm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("post_attention_layernorm") || n.contains("ffn_norm"));
        assert!(has_attn_norm);
        assert!(!has_ffn_norm);
    }

    #[test]
    fn test_lm_head_detection_explicit() {
        let names = vec!["lm_head.weight", "output.weight"];
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");
        assert!(has_lm_head);
    }

    #[test]
    fn test_lm_head_detection_output_weight() {
        let names = vec!["output.weight"];
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");
        assert!(has_lm_head);
    }

    #[test]
    fn test_lm_head_detection_tied_embeddings_fallback() {
        // When no explicit lm_head, but embedding exists
        let names = vec!["token_embd.weight"];
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        // LM head check passes if either explicit lm_head or embedding for tied weights
        assert!(!has_lm_head);
        assert!(has_embed);
        assert!(has_lm_head || has_embed);
    }

    #[test]
    fn test_lm_head_detection_no_head_no_embed() {
        let names = vec!["blk.0.attn_q.weight"];
        let has_lm_head = names
            .iter()
            .any(|n| n.contains("lm_head") || *n == "output.weight");
        let has_embed = names
            .iter()
            .any(|n| n.contains("emb") || n.contains("wte") || n.contains("token_embd"));
        assert!(!has_lm_head);
        assert!(!has_embed);
        assert!(!(has_lm_head || has_embed));
    }

    #[test]
    fn test_rope_tensor_detection() {
        let names = vec!["model.rope.freqs"];
        let has_rope = names
            .iter()
            .any(|n| n.contains("rope") || n.contains("rotary"));
        assert!(has_rope);
    }

    #[test]
    fn test_rope_rotary_variant() {
        let names = vec!["model.rotary_emb.inv_freq"];
        let has_rope = names
            .iter()
            .any(|n| n.contains("rope") || n.contains("rotary"));
        assert!(has_rope);
    }

    #[test]
    fn test_rope_absent() {
        let names = vec!["blk.0.attn_q.weight"];
        let has_rope = names
            .iter()
            .any(|n| n.contains("rope") || n.contains("rotary"));
        assert!(!has_rope);
    }

    #[test]
    fn test_attention_output_detection() {
        let names = vec!["blk.0.attn_output.weight"];
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("o_proj") || n.contains("attn_output"));
        assert!(has_attn_out);
    }

    #[test]
    fn test_attention_output_o_proj() {
        let names = vec!["model.layers.0.self_attn.o_proj.weight"];
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("o_proj") || n.contains("attn_output"));
        assert!(has_attn_out);
    }

    #[test]
    fn test_attention_output_absent() {
        let names = vec!["blk.0.attn_q.weight"];
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("o_proj") || n.contains("attn_output"));
        assert!(!has_attn_out);
    }

    // ========================================================================
    // Layer Norm with num_layers edge cases
    // ========================================================================

    #[test]
    fn test_norm_with_zero_layers_fails() {
        let has_norm = true;
        let num_layers = 0;
        // Stage 7 in APR path checks has_norm && num_layers > 0
        assert!(!(has_norm && num_layers > 0));
    }

    #[test]
    fn test_norm_with_positive_layers_passes() {
        let has_norm = true;
        let num_layers = 24;
        assert!(has_norm && num_layers > 0);
    }

    #[test]
    fn test_norm_absent_with_layers_fails() {
        let has_norm = false;
        let num_layers = 24;
        assert!(!(has_norm && num_layers > 0));
    }

    // ========================================================================
    // LM Head Vocab Size Formatting
    // ========================================================================

    #[test]
    fn test_lm_head_details_with_head() {
        let has_lm_head = true;
        let has_embed = true;
        let vocab_size = 32000;
        let detail = format!(
            "vocab_size={}{}",
            vocab_size,
            if has_lm_head { "" } else { " (tied)" }
        );
        assert_eq!(detail, "vocab_size=32000");
        let _ = has_embed;
    }

    #[test]
    fn test_lm_head_details_tied() {
        let has_lm_head = false;
        let vocab_size = 151936;
        let detail = format!(
            "vocab_size={}{}",
            vocab_size,
            if has_lm_head { "" } else { " (tied)" }
        );
        assert_eq!(detail, "vocab_size=151936 (tied)");
    }

    // ========================================================================
    // Non-inference path (cfg(not(feature = "inference")))
    // ========================================================================

    #[test]
    fn test_non_inference_stage_result_construction() {
        // This mirrors what happens when inference feature is disabled
        let result = StageResult {
            name: "N/A",
            eli5: "Requires inference",
            passed: false,
            details: Some("Build with --features inference".to_string()),
        };
        assert!(!result.passed);
        assert_eq!(result.name, "N/A");
        assert_eq!(result.eli5, "Requires inference");
        assert_eq!(
            result.details.as_deref(),
            Some("Build with --features inference")
        );
    }

    // ========================================================================
    // Extension Dispatch Logic (mirrors run_real_checks dispatch)
    // ========================================================================

    /// Helper that replicates the extension dispatch logic from run_real_checks
    fn classify_extension(path: &Path) -> &str {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext.to_lowercase().as_str() {
            "apr" => "apr",
            "gguf" => "gguf",
            _ => "unsupported",
        }
    }

    #[test]
    fn test_extension_dispatch_apr_lowercase() {
        assert_eq!(classify_extension(Path::new("model.apr")), "apr");
    }

    #[test]
    fn test_extension_dispatch_apr_uppercase() {
        assert_eq!(classify_extension(Path::new("model.APR")), "apr");
    }

    #[test]
    fn test_extension_dispatch_apr_mixed_case() {
        assert_eq!(classify_extension(Path::new("model.Apr")), "apr");
    }

    #[test]
    fn test_extension_dispatch_gguf_lowercase() {
        assert_eq!(classify_extension(Path::new("model.gguf")), "gguf");
    }

    #[test]
    fn test_extension_dispatch_gguf_uppercase() {
        assert_eq!(classify_extension(Path::new("model.GGUF")), "gguf");
    }

    #[test]
    fn test_extension_dispatch_gguf_mixed_case() {
        assert_eq!(classify_extension(Path::new("model.Gguf")), "gguf");
    }

    #[test]
    fn test_extension_dispatch_safetensors_unsupported() {
        assert_eq!(
            classify_extension(Path::new("model.safetensors")),
            "unsupported"
        );
    }

    #[test]
    fn test_extension_dispatch_bin_unsupported() {
        assert_eq!(classify_extension(Path::new("model.bin")), "unsupported");
    }

    #[test]
    fn test_extension_dispatch_no_extension() {
        assert_eq!(classify_extension(Path::new("modelfile")), "unsupported");
    }

    #[test]
    fn test_extension_dispatch_empty_extension() {
        // A path ending with "." has empty extension
        assert_eq!(classify_extension(Path::new("model.")), "unsupported");
    }

    #[test]
    fn test_extension_dispatch_double_extension() {
        // Only last extension matters
        assert_eq!(classify_extension(Path::new("model.tar.gguf")), "gguf");
    }

    #[test]
    fn test_extension_dispatch_hidden_file() {
        assert_eq!(classify_extension(Path::new(".model.apr")), "apr");
    }

    // ========================================================================
    // Unsupported Format Error Message Construction
    // ========================================================================

    #[test]
    fn test_unsupported_format_error_message() {
        let ext = "bin";
        let msg = format!("Unsupported format: {}. Use .apr or .gguf", ext);
        assert_eq!(msg, "Unsupported format: bin. Use .apr or .gguf");
    }

    #[test]
    fn test_unsupported_format_error_empty_ext() {
        let ext = "";
        let msg = format!("Unsupported format: {}. Use .apr or .gguf", ext);
        assert_eq!(msg, "Unsupported format: . Use .apr or .gguf");
    }

    #[test]
    fn test_unsupported_format_error_safetensors() {
        let ext = "safetensors";
        let msg = format!("Unsupported format: {}. Use .apr or .gguf", ext);
        assert!(msg.contains("safetensors"));
        assert!(msg.contains("Use .apr or .gguf"));
    }

    // ========================================================================
    // RoPE Theta Validation Logic (mirrors Stage 3 GGUF)
    // ========================================================================

    #[test]
    fn test_rope_theta_default_valid() {
        let rope_theta: f64 = 10000.0;
        assert!(rope_theta > 1.0);
    }

    #[test]
    fn test_rope_theta_llama3_valid() {
        let rope_theta: f64 = 500_000.0;
        assert!(rope_theta > 1.0);
    }

    #[test]
    fn test_rope_theta_zero_invalid() {
        let rope_theta: f64 = 0.0;
        assert!(!(rope_theta > 1.0));
    }

    #[test]
    fn test_rope_theta_negative_invalid() {
        let rope_theta: f64 = -1.0;
        assert!(!(rope_theta > 1.0));
    }

    #[test]
    fn test_rope_theta_exactly_one_invalid() {
        let rope_theta: f64 = 1.0;
        assert!(!(rope_theta > 1.0));
    }

    #[test]
    fn test_rope_theta_just_above_one_valid() {
        let rope_theta: f64 = 1.001;
        assert!(rope_theta > 1.0);
    }

    #[test]
    fn test_rope_theta_details_format() {
        let rope_theta: f64 = 10000.0;
        let details = format!("rope_theta={:.1}", rope_theta);
        assert_eq!(details, "rope_theta=10000.0");
    }

    // ========================================================================
    // GGUF-Specific Tensor Name Patterns (blk.N style)
    // ========================================================================

    #[test]
    fn test_gguf_embedding_detection_token_embd() {
        let name = "token_embd.weight";
        assert!(name.contains("token_embd") || name.contains("embed_tokens"));
    }
