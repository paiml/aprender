
    #[test]
    fn test_cross_validation_rope_theta_approximate() {
        let mut size = make_test_size();
        size.rope_theta = 1_000_000.0;
        let constraints = make_test_constraints();
        // Within 1.0 absolute tolerance => "match"
        let hf_config = serde_json::json!({
            "rope_theta": 1_000_000.5
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let rope_entry = cv.matches.iter().find(|e| e.field == "rope_theta");
        assert!(
            rope_entry.is_some(),
            "rope_theta should match within tolerance"
        );
    }

    #[test]
    fn test_cross_validation_rope_theta_mismatch() {
        let mut size = make_test_size();
        size.rope_theta = 1_000_000.0;
        let constraints = make_test_constraints();
        // Way off => "mismatch"
        let hf_config = serde_json::json!({
            "rope_theta": 500_000.0
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let rope_mismatch = cv.mismatches.iter().find(|e| e.field == "rope_theta");
        assert!(rope_mismatch.is_some(), "rope_theta should mismatch");
    }

    #[test]
    fn test_cross_validation_norm_eps_match() {
        let mut size = make_test_size();
        size.norm_eps = 1e-6;
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "rms_norm_eps": 1e-6
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let eps_entry = cv.matches.iter().find(|e| e.field == "norm_eps");
        assert!(eps_entry.is_some(), "norm_eps should match");
    }

    #[test]
    fn test_cross_validation_norm_eps_layer_norm_variant() {
        let mut size = make_test_size();
        size.norm_eps = 1e-5;
        let constraints = make_test_constraints();
        // Uses layer_norm_eps key instead of rms_norm_eps
        let hf_config = serde_json::json!({
            "layer_norm_eps": 1e-5
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let eps_entry = cv.matches.iter().find(|e| e.field == "norm_eps");
        assert!(
            eps_entry.is_some(),
            "norm_eps should match via layer_norm_eps key"
        );
    }

    #[test]
    fn test_cross_validation_norm_eps_epsilon_variant() {
        let mut size = make_test_size();
        size.norm_eps = 1e-5;
        let constraints = make_test_constraints();
        // Uses layer_norm_epsilon key
        let hf_config = serde_json::json!({
            "layer_norm_epsilon": 1e-5
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let eps_entry = cv.matches.iter().find(|e| e.field == "norm_eps");
        assert!(
            eps_entry.is_some(),
            "norm_eps should match via layer_norm_epsilon key"
        );
    }

    #[test]
    fn test_cross_validation_norm_eps_mismatch() {
        let mut size = make_test_size();
        size.norm_eps = 1e-6;
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "rms_norm_eps": 1e-5  // Different
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let eps_mismatch = cv.mismatches.iter().find(|e| e.field == "norm_eps");
        assert!(eps_mismatch.is_some(), "norm_eps should mismatch");
    }

    #[test]
    fn test_cross_validation_hf_only_interesting_fields() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "rope_scaling": {"type": "dynamic"},
            "sliding_window": 4096,
            "attention_dropout": 0.0,
            "use_cache": true,
            "tie_word_embeddings": false,
            "some_other_field": 42
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        // Should find 5 interesting HF-only fields
        assert!(
            cv.hf_only.len() >= 5,
            "Expected at least 5 HF-only fields, got {}",
            cv.hf_only.len()
        );
    }

    #[test]
    fn test_cross_validation_model_type_info() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "model_type": "qwen2"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let model_type_entry = cv.matches.iter().find(|e| e.field == "model_type");
        assert!(model_type_entry.is_some());
        assert_eq!(model_type_entry.expect("exists").status, "info");
    }

    #[test]
    fn test_cross_validation_hf_string_value() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        // String-valued HF field instead of number
        let hf_config = serde_json::json!({
            "hidden_size": "1536"
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        // Should match since both are "1536"
        let entry = cv.matches.iter().find(|e| e.field == "hidden_dim");
        assert!(entry.is_some(), "String-valued HF field should match");
    }

    #[test]
    fn test_cross_validation_contract_only_fields() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        // HF config missing most fields
        let hf_config = serde_json::json!({
            "hidden_size": 1536
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        // Should have several contract_only fields
        assert!(
            cv.contract_only.len() >= 5,
            "Expected many contract-only fields"
        );
    }

    // ========================================================================
    // Architecture Explanation Extended Tests
    // ========================================================================

    #[test]
    fn test_architecture_explanation_mha() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.attention_type = AttentionType::Mha;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.attention_explanation.contains("Multi-Head Attention"));
        assert!(expl
            .attention_explanation
            .contains(&size.num_heads.to_string()));
    }

    #[test]
    fn test_architecture_explanation_mqa() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.attention_type = AttentionType::Mqa;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.attention_explanation.contains("Multi-Query Attention"));
    }

    #[test]
    fn test_architecture_explanation_geglu() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.ffn_explanation.contains("GeGLU"));
    }

    #[test]
    fn test_architecture_explanation_gelu_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.ffn_explanation.contains("Standard GELU MLP"));
    }

    #[test]
    fn test_architecture_explanation_layer_norm() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.norm_type = NormType::LayerNorm;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.norm_explanation.contains("LayerNorm"));
    }

    #[test]
    fn test_architecture_explanation_absolute_pos() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.positional_encoding = PositionalEncoding::Absolute;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.positional_explanation.contains("Absolute position"));
    }

    #[test]
    fn test_architecture_explanation_alibi_pos() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.positional_encoding = PositionalEncoding::Alibi;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.positional_explanation.contains("ALiBi"));
    }

    #[test]
    fn test_architecture_explanation_relative_pos() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.positional_encoding = PositionalEncoding::Relative;
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.positional_explanation.contains("Relative"));
    }

    #[test]
    fn test_architecture_explanation_scaling_analysis() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let expl = build_architecture_explanation(&size, &constraints, &stats);

        assert!(expl.scaling_analysis.contains("parameters"));
        assert!(expl.scaling_analysis.contains("FLOPs"));
        assert!(expl.scaling_analysis.contains("Chinchilla"));
    }

    // ========================================================================
    // Kernel Compatibility Extended Tests
    // ========================================================================

    #[test]
    fn test_kernel_compatibility_mha() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.attention_type = AttentionType::Mha;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.attention_kernel.contains("MHA"));
    }

    #[test]
    fn test_kernel_compatibility_mqa() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.attention_type = AttentionType::Mqa;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.attention_kernel.contains("MQA"));
    }

    #[test]
    fn test_kernel_compatibility_gelu_mlp() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GeluMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.ffn_kernel.contains("standard GELU"));
    }

    #[test]
    fn test_kernel_compatibility_geglu() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.mlp_type = MlpType::GatedMlp;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.ffn_kernel.contains("GeGLU"));
    }

    #[test]
    fn test_kernel_compatibility_no_bias() {
        let mut size = make_test_size();
        size.num_kv_heads = size.num_heads; // MHA to remove GQA note
        let mut constraints = make_test_constraints();
        constraints.has_bias = false;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // No bias note should be absent
        assert!(!kern.notes.iter().any(|n| n.contains("Bias")));
    }

    #[test]
    fn test_kernel_compatibility_with_bias() {
        let size = make_test_size();
        let mut constraints = make_test_constraints();
        constraints.has_bias = true;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.notes.iter().any(|n| n.contains("Bias")));
    }

    #[test]
    fn test_kernel_compatibility_gqa_note() {
        let mut size = make_test_size();
        size.num_heads = 32;
        size.num_kv_heads = 8;
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        assert!(kern.notes.iter().any(|n| n.contains("GQA")));
    }

    #[test]
    fn test_kernel_compatibility_equal_heads_no_gqa_note() {
        let mut size = make_test_size();
        size.num_heads = 12;
        size.num_kv_heads = 12; // MHA
        let mut constraints = make_test_constraints();
        constraints.has_bias = false;
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // Should only have layout note, no GQA or bias notes
        assert!(
            kern.notes.len() == 1,
            "Expected only layout note, got: {:?}",
            kern.notes
        );
        assert!(kern.notes[0].contains("ROW-MAJOR"));
    }

    #[test]
    fn test_kernel_compatibility_quantization_sizes() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // F16 (16bpw) should be largest, Q4_K_M (4.5bpw) should be smallest
        let f16_entry = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "F16")
            .expect("F16");
        let q4_entry = kern
            .supported_quantizations
            .iter()
            .find(|q| q.format == "Q4_K_M")
            .expect("Q4_K_M");
        assert!(
            f16_entry.estimated_size_mb > q4_entry.estimated_size_mb,
            "F16 ({:.1} MB) should be larger than Q4_K_M ({:.1} MB)",
            f16_entry.estimated_size_mb,
            q4_entry.estimated_size_mb
        );

        // All sizes should be positive
        for q in &kern.supported_quantizations {
            assert!(
                q.estimated_size_mb > 0.0,
                "{} size should be positive",
                q.format
            );
        }
    }

    #[test]
    fn test_kernel_compatibility_tps_estimates() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        let cpu_tps = kern.estimated_tps_cpu.expect("should have CPU estimate");
        let gpu_tps = kern.estimated_tps_gpu.expect("should have GPU estimate");
        // GPU should be much faster than CPU (900 GB/s vs 50 GB/s bandwidth)
        assert!(
            gpu_tps > cpu_tps * 10.0,
            "GPU should be >10x faster than CPU"
        );
    }

    #[test]
    fn test_kernel_compatibility_memory_includes_kv() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // Memory should be > Q4 model size (includes KV cache)
        assert!(kern.memory_required_mb > stats.model_size_q4_mb);
    }

    // ========================================================================
    // format_params Extended Tests
    // ========================================================================

    #[test]
    fn test_format_params_boundary_1000() {
        assert_eq!(format_params(999), "999");
        assert_eq!(format_params(1000), "1.0K");
    }
