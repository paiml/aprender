
    #[test]
    fn test_oracle_mode_serialize() {
        let mode = OracleMode::Local;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"local\"");

        let mode = OracleMode::HuggingFace;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"hugging_face\"");

        let mode = OracleMode::Family;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"family\"");
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1_500), "1.5K");
        assert_eq!(format_params(1_500_000), "1.5M");
        assert_eq!(format_params(1_500_000_000), "1.5B");
        assert_eq!(format_params(7_000_000_000), "7.0B");
    }

    #[test]
    fn test_family_info_build() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test Model".to_string(),
            vendor: "TestCo".to_string(),
            architectures: vec!["TestForCausalLM".to_string()],
            hf_pattern: "test/*".to_string(),
            size_variants: HashMap::new(),
            constraints: ModelConstraints {
                attention_type: AttentionType::Gqa,
                activation: Activation::Silu,
                norm_type: NormType::RmsNorm,
                has_bias: true,
                tied_embeddings: false,
                positional_encoding: PositionalEncoding::Rope,
                mlp_type: MlpType::SwiGlu,
            },
            tensor_template: TensorTemplate {
                embedding: "embed.weight".to_string(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec!["q4_k_m".to_string()],
            chat_template: None,
            certification: None,
        };

        let fi = build_family_info(&config);
        assert_eq!(fi.name, "test");
        assert_eq!(fi.vendor, "TestCo");
        assert_eq!(fi.constraints.attention, "GQA");
        assert!(fi.constraints.bias);
    }

    #[test]
    fn test_compliance_result_serialize() {
        let cr = ComplianceResult {
            is_compliant: true,
            tensor_count_match: true,
            missing_tensors: vec![],
            unexpected_tensors: vec![],
        };
        let json = serde_json::to_string(&cr).expect("serialize");
        assert!(json.contains("\"is_compliant\":true"));
    }

    #[test]
    fn test_report_json_roundtrip() {
        let report = ModelOracleReport {
            source: "test.gguf".to_string(),
            mode: OracleMode::Local,
            family: Some(FamilyInfo {
                name: "qwen2".to_string(),
                display_name: "Qwen2".to_string(),
                vendor: "Alibaba".to_string(),
                architectures: vec!["Qwen2ForCausalLM".to_string()],
                constraints: ConstraintsSummary {
                    attention: "GQA".to_string(),
                    activation: "SiLU".to_string(),
                    norm: "RMSNorm".to_string(),
                    bias: true,
                    tied_embeddings: false,
                    mlp: "SwiGLU".to_string(),
                    positional_encoding: "RoPE".to_string(),
                },
                chat_template_format: Some("chatml".to_string()),
            }),
            size_variant: Some(SizeVariantInfo {
                name: "1.5b".to_string(),
                parameters: "1.5B".to_string(),
                hidden_dim: 1536,
                num_layers: 28,
                num_heads: 12,
                num_kv_heads: 2,
                intermediate_dim: 8960,
                vocab_size: 151936,
                expected_tensor_count: 339,
            }),
            format: Some(FormatInfo {
                format_type: "GGUF".to_string(),
                file_size: 1_000_000,
                tensor_count: 339,
                total_params: 1_500_000_000,
                quantization: Some("Q4_K_M".to_string()),
                architecture: Some("qwen2".to_string()),
            }),
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"source\": \"test.gguf\""));
        assert!(json.contains("\"mode\": \"local\""));
        assert!(json.contains("\"family\""));
        assert!(json.contains("\"hidden_dim\": 1536"));
    }

    #[test]
    fn test_certification_build_with_size() {
        use aprender::format::model_family::CertificationConfig;
        use std::collections::HashMap;

        let mut size_cats = HashMap::new();
        size_cats.insert("1.5b".to_string(), "small".to_string());

        let config = ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: aprender::format::model_family::ModelConstraints {
                attention_type: aprender::format::model_family::AttentionType::Gqa,
                activation: aprender::format::model_family::Activation::Silu,
                norm_type: aprender::format::model_family::NormType::RmsNorm,
                has_bias: false,
                tied_embeddings: false,
                positional_encoding: aprender::format::model_family::PositionalEncoding::Rope,
                mlp_type: aprender::format::model_family::MlpType::SwiGlu,
            },
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: Some(CertificationConfig {
                playbook_path: "../playbooks/{size}.yaml".to_string(),
                csv_family_key: "qwen2".to_string(),
                size_categories: size_cats,
            }),
        };

        let cert = build_certification(&config, Some("1.5b"));
        assert!(cert.is_some());
        let cert = cert.expect("cert exists");
        assert_eq!(cert.status, "PENDING");
        assert_eq!(
            cert.playbook_path,
            Some("../playbooks/1.5b.yaml".to_string())
        );
    }

    #[test]
    fn test_source_required_error() {
        let flags = OracleFlags::default();
        let result = run(None, None, None, false, false, false, false, false, flags);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("required"));
            }
            other => panic!("Expected InvalidFormat, got: {other:?}"),
        }
    }

    #[test]
    fn test_file_not_found() {
        let src = "/nonexistent/model.gguf".to_string();
        let flags = OracleFlags::default();
        let result = run(
            Some(&src),
            None,
            None,
            false,
            false,
            false,
            false,
            false,
            flags,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            other => panic!("Expected FileNotFound, got: {other:?}"),
        }
    }

    #[test]
    fn test_offline_hf_rejected() {
        let src = "hf://Qwen/Qwen2.5-Coder-1.5B".to_string();
        let flags = OracleFlags::default();
        let result = run(
            Some(&src),
            None,
            None,
            false,
            false,
            false,
            false,
            true, // offline
            flags,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::NetworkError(msg)) => {
                assert!(msg.contains("offline"));
            }
            other => panic!("Expected NetworkError, got: {other:?}"),
        }
    }

    #[test]
    fn test_load_registry_graceful_degradation() {
        // Should not error even if contracts dir doesn't exist nearby
        let registry = load_registry();
        // This should succeed (might be empty or populated depending on CWD)
        assert!(registry.is_ok());
    }

    #[test]
    fn test_tensor_compliance_entry_serialize() {
        let entry = TensorComplianceEntry {
            name: "model.embed_tokens.weight".to_string(),
            present: true,
            dtype: Some("F16".to_string()),
            shape: Some(vec![151936, 1536]),
            note: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("embed_tokens"));
        assert!(!json.contains("note")); // skip_serializing_if
    }

    // ========================================================================
    // Phase 1: Statistical Analysis Tests
    // ========================================================================

    fn make_test_size() -> ModelSizeConfig {
        ModelSizeConfig {
            parameters: "1.5B".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            intermediate_dim: 8960,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
        }
    }

    fn make_test_constraints() -> ModelConstraints {
        ModelConstraints {
            attention_type: AttentionType::Gqa,
            activation: aprender::format::model_family::Activation::Silu,
            norm_type: NormType::RmsNorm,
            has_bias: true,
            tied_embeddings: false,
            positional_encoding: PositionalEncoding::Rope,
            mlp_type: MlpType::SwiGlu,
        }
    }

    #[test]
    fn test_gqa_analysis() {
        let size = make_test_size();
        let (ratio, reduction) = compute_gqa_analysis(&size);
        // 2 kv heads / 12 heads = 1/6
        assert!((ratio - 1.0 / 6.0).abs() < 0.01);
        assert!((reduction - 5.0 / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_gqa_analysis_mha() {
        let mut size = make_test_size();
        size.num_kv_heads = size.num_heads; // MHA: ratio = 1.0
        let (ratio, reduction) = compute_gqa_analysis(&size);
        assert!((ratio - 1.0).abs() < 0.01);
        assert!(reduction.abs() < 0.01);
    }

    #[test]
    fn test_param_count_nonzero() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let params = compute_param_count(&size, &constraints);
        assert!(params > 0);
        // Qwen2 1.5B should be in the ballpark of 1.5B params
        assert!(params > 1_000_000_000, "params too small: {params}");
        assert!(params < 3_000_000_000, "params too large: {params}");
    }

    #[test]
    fn test_memory_estimates() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (f16_mb, q4_mb) = compute_memory_estimates(&size, &constraints);
        // F16 should be about 2x Q4
        assert!(f16_mb > q4_mb * 3.0, "F16 should be much larger than Q4");
        assert!(f16_mb > 0.0);
        assert!(q4_mb > 0.0);
    }

    #[test]
    fn test_kv_cache() {
        let size = make_test_size();
        let (per_token, cache_4k) = compute_kv_cache(&size);
        assert!(per_token > 0);
        assert!(cache_4k > 0.0);
        // Per-token should be 2 * 28 * 2 * 128 * 2 = 28672 bytes
        assert_eq!(per_token, 2 * 28 * 2 * 128 * 2);
    }

    #[test]
    fn test_ffn_analysis() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (ratio, explanation) = compute_ffn_analysis(&size, &constraints);
        // 8960 / 1536 ≈ 5.83
        assert!(
            ratio > 5.0 && ratio < 6.5,
            "FFN ratio {ratio} out of expected range"
        );
        assert!(explanation.contains("SwiGLU"));
    }

    #[test]
    fn test_rope_analysis() {
        let size = make_test_size();
        let (wavelength, ctx) = compute_rope_analysis(&size);
        assert!(wavelength > 0.0);
        assert_eq!(ctx, 32768);
    }

    #[test]
    fn test_flops_estimate() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let (attn, ffn) = compute_flops_estimate(&size, &constraints);
        assert!(attn > 0);
        assert!(ffn > 0);
    }

    #[test]
    fn test_statistical_analysis_complete() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);

        // Verify all fields are populated sensibly
        assert!(stats.gqa_ratio > 0.0 && stats.gqa_ratio <= 1.0);
        assert!(stats.kv_cache_reduction >= 0.0 && stats.kv_cache_reduction < 1.0);
        assert!(stats.model_params > 0);
        assert!(stats.model_size_f16_mb > 0.0);
        assert!(stats.model_size_q4_mb > 0.0);
        assert!(stats.kv_cache_per_token_bytes > 0);
        assert!(stats.kv_cache_4k_mb > 0.0);
        assert!(stats.ffn_expansion_ratio > 1.0);
        assert!(!stats.ffn_type_explanation.is_empty());
        assert!(stats.rope_max_wavelength > 0.0);
        assert!(stats.effective_context_window > 0);
        assert!(stats.attention_flops_per_token > 0);
        assert!(stats.ffn_flops_per_token > 0);

        // Verify JSON serialization
        let json = serde_json::to_string(&stats).expect("serialize stats");
        assert!(json.contains("gqa_ratio"));
        assert!(json.contains("model_params"));
    }

    // ========================================================================
    // Phase 3: Cross-Validation Tests
    // ========================================================================

    #[test]
    fn test_cross_validation_match() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config: serde_json::Value = serde_json::json!({
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
        assert!(
            cv.mismatches.is_empty(),
            "Expected no mismatches, got: {:?}",
            cv.mismatches
        );
        assert!(!cv.matches.is_empty(), "Expected matches");
    }
