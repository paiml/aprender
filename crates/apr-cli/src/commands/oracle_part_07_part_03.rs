
    #[test]
    fn test_cross_validation_mismatch() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config: serde_json::Value = serde_json::json!({
            "hidden_size": 2048,  // MISMATCH: 2048 vs 1536
            "num_hidden_layers": 28,
            "model_type": "qwen2"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert!(
            cv.mismatches.iter().any(|e| e.field == "hidden_dim"),
            "Expected hidden_dim mismatch, got: {:?}",
            cv.mismatches
        );
    }

    // ========================================================================
    // Phase 4: Architecture Explanation Tests
    // ========================================================================

    #[test]
    fn test_architecture_explanation() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.attention_explanation.contains("GQA"));
        assert!(expl.ffn_explanation.contains("SwiGLU"));
        assert!(expl.norm_explanation.contains("RMSNorm"));
        assert!(expl.positional_explanation.contains("RoPE"));
        assert!(expl.scaling_analysis.contains("parameters"));
    }

    // ========================================================================
    // Phase 5: Kernel Compatibility Tests
    // ========================================================================

    #[test]
    fn test_kernel_compatibility() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert_eq!(kern.supported_quantizations.len(), 4);
        assert!(kern.supported_quantizations.iter().all(|q| q.supported));
        assert!(kern.attention_kernel.contains("GQA"));
        assert!(kern.ffn_kernel.contains("SwiGLU"));
        assert!(kern.estimated_tps_cpu.is_some());
        assert!(kern.estimated_tps_gpu.is_some());
        assert!(kern.memory_required_mb > 0.0);
        assert!(!kern.notes.is_empty());
    }

    // ========================================================================
    // Phase 6: OracleFlags Tests
    // ========================================================================

    #[test]
    fn test_oracle_flags_default() {
        let flags = OracleFlags::default();
        assert!(!flags.show_stats());
        assert!(!flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_full() {
        let flags = OracleFlags {
            full: true,
            ..OracleFlags::default()
        };
        assert!(flags.show_stats());
        assert!(flags.show_explain());
        assert!(flags.show_kernels());
        assert!(flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_individual() {
        let flags = OracleFlags {
            stats: true,
            ..OracleFlags::default()
        };
        assert!(flags.show_stats());
        assert!(!flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_explain_only() {
        let flags = OracleFlags {
            explain: true,
            ..OracleFlags::default()
        };
        assert!(!flags.show_stats());
        assert!(flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_kernels_only() {
        let flags = OracleFlags {
            kernels: true,
            ..OracleFlags::default()
        };
        assert!(!flags.show_stats());
        assert!(!flags.show_explain());
        assert!(flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_validate_only() {
        let flags = OracleFlags {
            validate: true,
            ..OracleFlags::default()
        };
        assert!(!flags.show_stats());
        assert!(!flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_debug() {
        let flags = OracleFlags::default();
        let debug = format!("{flags:?}");
        assert!(debug.contains("OracleFlags"));
    }

    #[test]
    fn test_oracle_flags_clone() {
        let flags = OracleFlags {
            stats: true,
            explain: true,
            kernels: false,
            validate: false,
            full: false,
        };
        let cloned = flags;
        assert!(cloned.show_stats());
        assert!(cloned.show_explain());
    }

    // ========================================================================
    // GQA Analysis Edge Cases
    // ========================================================================

    #[test]
    fn test_gqa_analysis_zero_heads() {
        let mut size = make_test_size();
        size.num_heads = 0;
        let (ratio, reduction) = compute_gqa_analysis(&size);
        assert_eq!(ratio, 0.0);
        assert_eq!(reduction, 0.0);
    }

    #[test]
    fn test_gqa_analysis_single_kv_head() {
        let mut size = make_test_size();
        size.num_heads = 32;
        size.num_kv_heads = 1; // MQA-like
        let (ratio, reduction) = compute_gqa_analysis(&size);
        assert!((ratio - 1.0 / 32.0).abs() < 0.001);
        assert!((reduction - 31.0 / 32.0).abs() < 0.001);
    }

    #[test]
    fn test_gqa_analysis_equal_heads() {
        let mut size = make_test_size();
        size.num_heads = 32;
        size.num_kv_heads = 32; // MHA
        let (ratio, reduction) = compute_gqa_analysis(&size);
        assert!((ratio - 1.0).abs() < 0.001);
        assert!(reduction.abs() < 0.001);
    }

    // ========================================================================
    // Param Count Edge Cases
    // ========================================================================

    #[test]
    fn test_param_count_no_bias() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.has_bias = false;
        let params_no_bias = compute_param_count(&size, &constraints);

        constraints.has_bias = true;
        let params_with_bias = compute_param_count(&size, &constraints);

        assert!(
            params_with_bias > params_no_bias,
            "Bias should add parameters"
        );
    }

    #[test]
    fn test_param_count_tied_embeddings() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.tied_embeddings = false;
        let params_untied = compute_param_count(&size, &constraints);

        constraints.tied_embeddings = true;
        let params_tied = compute_param_count(&size, &constraints);

        // Tied embeddings removes lm_head = vocab_size * hidden_dim
        let lm_head_params = (size.vocab_size as u64) * (size.hidden_dim as u64);
        assert_eq!(params_untied - params_tied, lm_head_params);
    }

    #[test]
    fn test_param_count_gated_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let params_gated = compute_param_count(&size, &constraints);

        constraints.mlp_type = MlpType::SwiGlu;
        let params_swiglu = compute_param_count(&size, &constraints);

        // Both gated, should have same FFN param count
        assert_eq!(params_gated, params_swiglu);
    }

    #[test]
    fn test_param_count_standard_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let params_standard = compute_param_count(&size, &constraints);

        constraints.mlp_type = MlpType::SwiGlu;
        let params_gated = compute_param_count(&size, &constraints);

        // Standard uses 2 matrices, gated uses 3
        assert!(
            params_gated > params_standard,
            "Gated should have more params"
        );
    }

    #[test]
    fn test_param_count_minimal_model() {
        let size = ModelSizeConfig {
            parameters: "tiny".to_string(),
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 128,
            vocab_size: 100,
            max_position_embeddings: 512,
            head_dim: 32,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        };
        let constraints = ModelConstraints {
            attention_type: AttentionType::Mha,
            activation: aprender::format::model_family::Activation::Silu,
            norm_type: NormType::RmsNorm,
            has_bias: false,
            tied_embeddings: true,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::GeluMlp,
        };
        let params = compute_param_count(&size, &constraints);
        assert!(params > 0, "Even minimal model should have params");
        // Embedding: 100*64 = 6400, no lm_head (tied), 1 layer with small dims
        assert!(
            params < 1_000_000,
            "Tiny model shouldn't have millions of params"
        );
    }

    // ========================================================================
    // Memory Estimates Edge Cases
    // ========================================================================

    #[test]
    fn test_memory_estimates_f16_is_4x_q4() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (f16_mb, q4_mb) = compute_memory_estimates(&size, &constraints);
        // F16 = 2 bytes/param, Q4 = 0.5 bytes/param, ratio = 4
        assert!(
            (f16_mb / q4_mb - 4.0).abs() < 0.01,
            "F16/Q4 ratio should be ~4"
        );
    }

    // ========================================================================
    // KV Cache Edge Cases
    // ========================================================================

    #[test]
    fn test_kv_cache_zero_layers() {
        let mut size = make_test_size();
        size.num_layers = 0;
        let (per_token, cache_4k) = compute_kv_cache(&size);
        assert_eq!(per_token, 0);
        assert_eq!(cache_4k, 0.0);
    }

    #[test]
    fn test_kv_cache_zero_kv_heads() {
        let mut size = make_test_size();
        size.num_kv_heads = 0;
        let (per_token, cache_4k) = compute_kv_cache(&size);
        assert_eq!(per_token, 0);
        assert_eq!(cache_4k, 0.0);
    }

    #[test]
    fn test_kv_cache_large_context() {
        let size = make_test_size();
        let (per_token, cache_4k) = compute_kv_cache(&size);
        // 4K cache = per_token * 4096 / (1024*1024)
        let expected_4k = (per_token as f64 * 4096.0) / (1024.0 * 1024.0);
        assert!((cache_4k - expected_4k).abs() < 0.001);
    }

    // ========================================================================
    // FFN Analysis Edge Cases
    // ========================================================================

    #[test]
    fn test_ffn_analysis_zero_hidden_dim() {
        let mut size = make_test_size();
        size.hidden_dim = 0;
        let constraints = make_test_constraints();
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert_eq!(ratio, 0.0);
        assert!(explanation.is_empty());
    }

    #[test]
    fn test_ffn_analysis_gated_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert!(ratio > 0.0);
        assert!(explanation.contains("GeGLU"));
    }

    #[test]
    fn test_ffn_analysis_gelu_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        assert!(ratio > 0.0);
        assert!(explanation.contains("Standard GELU"));
    }

    // ========================================================================
    // RoPE Analysis Edge Cases
    // ========================================================================

    #[test]
    fn test_rope_analysis_zero_theta() {
        let mut size = make_test_size();
        size.rope_theta = 0.0;
        let (wavelength, ctx) = compute_rope_analysis(&size);
        assert_eq!(wavelength, 0.0);
        assert_eq!(ctx, size.max_position_embeddings);
    }

    #[test]
    fn test_rope_analysis_standard_theta() {
        let mut size = make_test_size();
        size.rope_theta = 10000.0;
        let (wavelength, _) = compute_rope_analysis(&size);
        let expected = 2.0 * std::f64::consts::PI * 10000.0;
        assert!((wavelength - expected).abs() < 1.0);
    }

    // ========================================================================
    // FLOPS Estimate Edge Cases
    // ========================================================================

    #[test]
    fn test_flops_estimate_standard_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let (attn_gelu, ffn_gelu) = compute_flops_estimate(&size, &constraints);

        constraints.mlp_type = MlpType::SwiGlu;
        let (attn_swiglu, ffn_swiglu) = compute_flops_estimate(&size, &constraints);

        // Attention FLOPS should be the same regardless of MLP type
        assert_eq!(attn_gelu, attn_swiglu);
        // Gated has 50% more FFN FLOPS (3 matmuls vs 2)
        assert_eq!(ffn_swiglu, ffn_gelu * 3 / 2);
    }

    #[test]
    fn test_flops_estimate_zero_layers() {
        let mut size = make_test_size();
        size.num_layers = 0;
        let constraints = make_test_constraints();
        let (attn, ffn) = compute_flops_estimate(&size, &constraints);
        assert_eq!(attn, 0);
        assert_eq!(ffn, 0);
    }

    // ========================================================================
    // Cross-Validation Extended Tests
    // ========================================================================

    #[test]
    fn test_cross_validation_empty_hf_config() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({});

        let cv = cross_validate(&size, &constraints, &hf_config);
        assert!(cv.matches.is_empty());
        assert!(cv.mismatches.is_empty());
        // All contract fields are contract_only since HF has none
        assert!(!cv.contract_only.is_empty());
    }
