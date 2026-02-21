
    #[test]
    fn test_compute_param_count_tied_removes_lm_head() {
        let size = ModelSizeConfig {
            parameters: "test".to_string(),
            hidden_dim: 100,
            num_layers: 0,
            num_heads: 1,
            num_kv_heads: 1,
            intermediate_dim: 200,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 100,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        };
        let mut constraints = make_test_constraints();
        constraints.tied_embeddings = true;

        let params = compute_param_count(&size, &constraints);
        // embedding (1000*100) + final_norm (100) + 0 layers, no lm_head
        let expected = 1000 * 100 + 100;
        assert_eq!(params, expected);
    }

    #[test]
    fn test_compute_memory_estimates_ratio() {
        // F16 = 2 bytes/param, Q4 = 0.5 bytes/param
        // So F16/Q4 ratio should be exactly 4.0
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (f16_mb, q4_mb) = compute_memory_estimates(&size, &constraints);
        let ratio = f16_mb / q4_mb;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "F16/Q4 ratio should be exactly 4.0, got {ratio}"
        );
    }

    #[test]
    fn test_kv_cache_formula_correctness() {
        // KV = 2 * L * kv_heads * head_dim * 2(f16)
        let size = ModelSizeConfig {
            parameters: "test".to_string(),
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 4,
            intermediate_dim: 512,
            vocab_size: 100,
            max_position_embeddings: 2048,
            head_dim: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        };
        let (per_token, _) = compute_kv_cache(&size);
        let expected: u64 = 2 * 4 * 4 * 32 * 2;
        assert_eq!(per_token, expected);
    }

    #[test]
    fn test_ffn_analysis_swiglu_explanation_contains_ratio() {
        let mut size = make_test_size();
        size.hidden_dim = 1000;
        size.intermediate_dim = 3000;
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::SwiGlu;
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert!((ratio - 3.0).abs() < 0.01);
        assert!(explanation.contains("3.00x"));
    }

    #[test]
    fn test_ffn_analysis_gelu_explanation_contains_ratio() {
        let mut size = make_test_size();
        size.hidden_dim = 1000;
        size.intermediate_dim = 4000;
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert!((ratio - 4.0).abs() < 0.01);
        assert!(explanation.contains("4.00x"));
    }

    #[test]
    fn test_rope_analysis_negative_theta() {
        let mut size = make_test_size();
        size.rope_theta = -1.0; // Negative should produce 0.0
        let (wavelength, _) = compute_rope_analysis(&size);
        assert_eq!(wavelength, 0.0);
    }

    #[test]
    fn test_flops_estimate_scales_with_layers() {
        let mut size = make_test_size();
        let constraints = make_test_constraints();

        size.num_layers = 10;
        let (attn_10, ffn_10) = compute_flops_estimate(&size, &constraints);

        size.num_layers = 20;
        let (attn_20, ffn_20) = compute_flops_estimate(&size, &constraints);

        // Doubling layers should double FLOPS
        assert_eq!(attn_20, attn_10 * 2);
        assert_eq!(ffn_20, ffn_10 * 2);
    }

    #[test]
    fn test_build_statistical_analysis_zero_rope_theta() {
        let mut size = make_test_size();
        size.rope_theta = 0.0;
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        assert_eq!(stats.rope_max_wavelength, 0.0);
    }

    #[test]
    fn test_build_statistical_analysis_zero_hidden_dim() {
        let mut size = make_test_size();
        size.hidden_dim = 0;
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        assert_eq!(stats.ffn_expansion_ratio, 0.0);
        assert!(stats.ffn_type_explanation.is_empty());
    }

    #[test]
    fn test_kernel_compatibility_geglu_kernel_string() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);
        assert!(kern.ffn_kernel.contains("GeGLU"));
        assert!(kern.ffn_kernel.contains("row-major"));
    }

    #[test]
    fn test_kernel_compatibility_f16_bits_per_weight() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        let f16 = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "F16")
            .expect("F16");
        assert!((f16.bits_per_weight - 16.0).abs() < 0.001);

        let q8 = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "Q8_0")
            .expect("Q8_0");
        assert!((q8.bits_per_weight - 8.0).abs() < 0.001);

        let q4 = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "Q4_K_M")
            .expect("Q4_K_M");
        assert!((q4.bits_per_weight - 4.5).abs() < 0.001);

        let q6 = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "Q6_K")
            .expect("Q6_K");
        assert!((q6.bits_per_weight - 6.5).abs() < 0.001);
    }

    #[test]
    fn test_kernel_compatibility_row_major_note_always_present() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);
        assert!(kern.notes.iter().any(|n| n.contains("ROW-MAJOR")));
    }

    #[test]
    fn test_architecture_explanation_gqa_kv_cache_comparison() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);
        // GQA explanation should mention cache reduction percentage
        assert!(expl.attention_explanation.contains("reduces KV cache"));
        assert!(expl.attention_explanation.contains("MB"));
    }

    #[test]
    fn test_architecture_explanation_rope_extrapolation() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);
        // RoPE explanation should mention extrapolation
        assert!(expl.positional_explanation.contains("YaRN"));
        let extrapolated = size.max_position_embeddings * 4;
        assert!(expl
            .positional_explanation
            .contains(&extrapolated.to_string()));
    }

    #[test]
    fn test_architecture_explanation_chinchilla_tokens() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);
        assert!(expl.scaling_analysis.contains("Chinchilla"));
        assert!(expl.scaling_analysis.contains("FLOPs"));
    }

    #[test]
    fn test_build_family_info_with_chat_template() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hf_pattern: "Qwen/Qwen2*".to_string(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: TensorTemplate {
                embedding: "embed.weight".to_string(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            gguf_tensor_template: GgufTensorTemplate::default(),
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: Some(ChatTemplateConfig {
                format: "chatml".to_string(),
                template: String::new(),
                bos_token: "<|im_start|>".to_string(),
                eos_token: "<|im_end|>".to_string(),
                special_tokens: HashMap::new(),
            }),
            certification: None,
        };

        let fi = build_family_info(&config);
        assert_eq!(fi.chat_template_format, Some("chatml".to_string()));
    }

    #[test]
    fn test_build_family_info_display_types() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "gpt2".to_string(),
            display_name: "GPT-2".to_string(),
            vendor: "OpenAI".to_string(),
            architectures: vec!["GPT2LMHeadModel".to_string()],
            hf_pattern: "openai/gpt2*".to_string(),
            size_variants: HashMap::new(),
            constraints: ModelConstraints {
                attention_type: AttentionType::Mha,
                activation: Activation::Gelu,
                norm_type: NormType::LayerNorm,
                has_bias: true,
                tied_embeddings: true,
                positional_encoding: PositionalEncoding::Absolute,
                mlp_type: MlpType::GeluMlp,
                qk_norm: false,
            },
            tensor_template: TensorTemplate {
                embedding: "wte.weight".to_string(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            gguf_tensor_template: GgufTensorTemplate::default(),
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let fi = build_family_info(&config);
        assert_eq!(fi.constraints.attention, "MHA");
        assert_eq!(fi.constraints.norm, "LayerNorm");
        assert_eq!(fi.constraints.mlp, "GELU MLP");
        assert_eq!(fi.constraints.positional_encoding, "Absolute");
        assert!(fi.constraints.bias);
        assert!(fi.constraints.tied_embeddings);
    }

    #[test]
    fn test_cross_validation_all_fields_match() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 8960,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "model_type": "qwen2"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert!(cv.mismatches.is_empty());
        assert!(cv.contract_only.is_empty());
        // 7 size fields + rope_theta + norm_eps + model_type = 10 matches
        assert!(
            cv.matches.len() >= 9,
            "Expected at least 9 matches, got {}",
            cv.matches.len()
        );
    }

    #[test]
    fn test_cross_validation_multiple_mismatches() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "hidden_size": 9999,
            "num_hidden_layers": 99,
            "num_attention_heads": 99,
            "num_key_value_heads": 99,
            "intermediate_size": 9999,
            "vocab_size": 9999,
            "max_position_embeddings": 9999
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert_eq!(cv.mismatches.len(), 7, "All 7 size fields should mismatch");
    }

    #[test]
    fn test_cross_validation_norm_eps_mismatch_value() {
        let mut size = make_test_size();
        size.norm_eps = 1e-6;
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "rms_norm_eps": 1e-5
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let entry = cv
            .mismatches
            .iter()
            .find(|e| e.field == "norm_eps")
            .expect("should mismatch");
        assert_eq!(entry.status, "mismatch");
    }

    #[test]
    fn test_format_params_large_values() {
        assert_eq!(format_params(70_000_000_000), "70.0B");
        assert_eq!(format_params(175_000_000_000), "175.0B");
    }

    #[test]
    fn test_format_params_exact_boundaries() {
        assert_eq!(format_params(1), "1");
        assert_eq!(format_params(10), "10");
        assert_eq!(format_params(100), "100");
    }

    #[test]
    fn test_statistical_analysis_complete_field_coverage() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);

        // Verify specific computed values
        let (expected_ratio, expected_reduction) = compute_gqa_analysis(&size);
        assert!((stats.gqa_ratio - expected_ratio).abs() < 1e-10);
        assert!((stats.kv_cache_reduction - expected_reduction).abs() < 1e-10);

        let expected_params = compute_param_count(&size, &constraints);
        assert_eq!(stats.model_params, expected_params);

        let (expected_per_token, expected_4k) = compute_kv_cache(&size);
        assert_eq!(stats.kv_cache_per_token_bytes, expected_per_token);
        assert!((stats.kv_cache_4k_mb - expected_4k).abs() < 1e-10);

        let (expected_ffn_ratio, _) = compute_ffn_analysis(&size, &constraints);
        assert!((stats.ffn_expansion_ratio - expected_ffn_ratio).abs() < 1e-10);

        let (expected_wavelength, expected_ctx) = compute_rope_analysis(&size);
        assert!((stats.rope_max_wavelength - expected_wavelength).abs() < 1e-10);
        assert_eq!(stats.effective_context_window, expected_ctx);

        let (expected_attn_flops, expected_ffn_flops) = compute_flops_estimate(&size, &constraints);
        assert_eq!(stats.attention_flops_per_token, expected_attn_flops);
        assert_eq!(stats.ffn_flops_per_token, expected_ffn_flops);
    }

    #[test]
    fn test_build_certification_size_template_replacement() {
        use aprender::format::model_family::CertificationConfig;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            gguf_tensor_template: aprender::format::model_family::GgufTensorTemplate::default(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: Some(CertificationConfig {
                playbook_path: "playbooks/{size}/run.yaml".to_string(),
                csv_family_key: "test".to_string(),
                size_categories: HashMap::new(),
            }),
        };

        let cert = build_certification(&config, Some("13b")).expect("cert exists");
        assert_eq!(
            cert.playbook_path,
            Some("playbooks/13b/run.yaml".to_string())
        );
    }
