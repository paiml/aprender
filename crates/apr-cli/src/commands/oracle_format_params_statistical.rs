
    #[test]
    fn test_format_params_boundary_1m() {
        assert_eq!(format_params(999_999), "1000.0K");
        assert_eq!(format_params(1_000_000), "1.0M");
    }

    #[test]
    fn test_format_params_boundary_1b() {
        assert_eq!(format_params(999_999_999), "1000.0M");
        assert_eq!(format_params(1_000_000_000), "1.0B");
    }

    #[test]
    fn test_format_params_zero() {
        assert_eq!(format_params(0), "0");
    }

    // ========================================================================
    // build_statistical_analysis Integration Tests
    // ========================================================================

    #[test]
    fn test_statistical_analysis_with_mha() {
        let mut size = make_test_size();
        size.num_kv_heads = size.num_heads;
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);

        assert!((stats.gqa_ratio - 1.0).abs() < 0.01);
        assert!(stats.kv_cache_reduction.abs() < 0.01);
    }

    #[test]
    fn test_statistical_analysis_serialization() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);

        let json = serde_json::to_string_pretty(&stats).expect("serialize");
        assert!(json.contains("gqa_ratio"));
        assert!(json.contains("kv_cache_reduction"));
        assert!(json.contains("model_params"));
        assert!(json.contains("model_size_f16_mb"));
        assert!(json.contains("model_size_q4_mb"));
        assert!(json.contains("kv_cache_per_token_bytes"));
        assert!(json.contains("kv_cache_4k_mb"));
        assert!(json.contains("ffn_expansion_ratio"));
        assert!(json.contains("ffn_type_explanation"));
        assert!(json.contains("rope_max_wavelength"));
        assert!(json.contains("effective_context_window"));
        assert!(json.contains("attention_flops_per_token"));
        assert!(json.contains("ffn_flops_per_token"));
    }

    // ========================================================================
    // Serialization Tests for Report Types
    // ========================================================================

    #[test]
    fn test_family_info_serialize() {
        let fi = FamilyInfo {
            name: "llama".to_string(),
            display_name: "LLaMA".to_string(),
            vendor: "Meta".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string()],
            constraints: ConstraintsSummary {
                attention: "GQA".to_string(),
                activation: "SiLU".to_string(),
                norm: "RMSNorm".to_string(),
                bias: false,
                tied_embeddings: false,
                mlp: "SwiGLU".to_string(),
                positional_encoding: "RoPE".to_string(),
            },
            chat_template_format: None,
        };
        let json = serde_json::to_string(&fi).expect("serialize");
        assert!(json.contains("\"name\":\"llama\""));
        // chat_template_format should be skipped
        assert!(!json.contains("chat_template_format"));
    }

    #[test]
    fn test_size_variant_info_serialize() {
        let svi = SizeVariantInfo {
            name: "7b".to_string(),
            parameters: "7B".to_string(),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 14336,
            vocab_size: 32000,
            expected_tensor_count: 291,
        };
        let json = serde_json::to_string(&svi).expect("serialize");
        assert!(json.contains("\"hidden_dim\":4096"));
        assert!(json.contains("\"expected_tensor_count\":291"));
    }

    #[test]
    fn test_format_info_serialize() {
        let fi = FormatInfo {
            format_type: "GGUF".to_string(),
            file_size: 4_000_000_000,
            tensor_count: 291,
            total_params: 7_000_000_000,
            quantization: Some("Q4_K_M".to_string()),
            architecture: Some("llama".to_string()),
        };
        let json = serde_json::to_string(&fi).expect("serialize");
        assert!(json.contains("\"format_type\":\"GGUF\""));
        assert!(json.contains("\"quantization\":\"Q4_K_M\""));
    }

    #[test]
    fn test_format_info_serialize_no_optional() {
        let fi = FormatInfo {
            format_type: "SafeTensors".to_string(),
            file_size: 2_000_000_000,
            tensor_count: 200,
            total_params: 1_500_000_000,
            quantization: None,
            architecture: None,
        };
        let json = serde_json::to_string(&fi).expect("serialize");
        assert!(!json.contains("quantization"));
        assert!(!json.contains("architecture"));
    }

    #[test]
    fn test_certification_info_serialize() {
        let ci = CertificationInfo {
            status: "PENDING".to_string(),
            playbook_path: Some("/playbooks/7b.yaml".to_string()),
        };
        let json = serde_json::to_string(&ci).expect("serialize");
        assert!(json.contains("PENDING"));
        assert!(json.contains("playbook_path"));
    }

    #[test]
    fn test_certification_info_no_playbook() {
        let ci = CertificationInfo {
            status: "APPROVED".to_string(),
            playbook_path: None,
        };
        let json = serde_json::to_string(&ci).expect("serialize");
        assert!(!json.contains("playbook_path"));
    }

    #[test]
    fn test_cross_validation_entry_serialize() {
        let entry = CrossValidationEntry {
            field: "hidden_dim".to_string(),
            contract_value: "1536".to_string(),
            hf_value: "1536".to_string(),
            status: "match".to_string(),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("\"status\":\"match\""));
    }

    #[test]
    fn test_cross_validation_serialize() {
        let cv = CrossValidation {
            matches: vec![CrossValidationEntry {
                field: "hidden_dim".to_string(),
                contract_value: "1536".to_string(),
                hf_value: "1536".to_string(),
                status: "match".to_string(),
            }],
            mismatches: vec![],
            contract_only: vec!["norm_eps=1e-6".to_string()],
            hf_only: vec!["rope_scaling=dynamic".to_string()],
        };
        let json = serde_json::to_string(&cv).expect("serialize");
        assert!(json.contains("matches"));
        assert!(json.contains("mismatches"));
        assert!(json.contains("contract_only"));
        assert!(json.contains("hf_only"));
    }

    #[test]
    fn test_kernel_compatibility_serialize() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![QuantizationSupport {
                format: "Q4_K_M".to_string(),
                supported: true,
                kernel: "fused_q4k".to_string(),
                bits_per_weight: 4.5,
                estimated_size_mb: 500.0,
            }],
            attention_kernel: "GQA fused".to_string(),
            ffn_kernel: "SwiGLU fused".to_string(),
            estimated_tps_cpu: Some(100.0),
            estimated_tps_gpu: Some(1000.0),
            memory_required_mb: 600.0,
            notes: vec!["ROW-MAJOR".to_string()],
        };
        let json = serde_json::to_string(&kern).expect("serialize");
        assert!(json.contains("supported_quantizations"));
        assert!(json.contains("estimated_tps_cpu"));
    }

    #[test]
    fn test_kernel_compatibility_no_tps() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "none".to_string(),
            ffn_kernel: "none".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let json = serde_json::to_string(&kern).expect("serialize");
        assert!(!json.contains("estimated_tps_cpu"));
        assert!(!json.contains("estimated_tps_gpu"));
    }

    #[test]
    fn test_architecture_explanation_serialize() {
        let expl = ArchitectureExplanation {
            attention_explanation: "GQA with ratio 0.17".to_string(),
            ffn_explanation: "SwiGLU gated".to_string(),
            norm_explanation: "RMSNorm".to_string(),
            positional_explanation: "RoPE theta=1000000".to_string(),
            scaling_analysis: "1.5B params".to_string(),
        };
        let json = serde_json::to_string(&expl).expect("serialize");
        assert!(json.contains("attention_explanation"));
        assert!(json.contains("ffn_explanation"));
        assert!(json.contains("norm_explanation"));
        assert!(json.contains("positional_explanation"));
        assert!(json.contains("scaling_analysis"));
    }

    #[test]
    fn test_quantization_support_serialize() {
        let qs = QuantizationSupport {
            format: "F16".to_string(),
            supported: true,
            kernel: "trueno::f16_matvec".to_string(),
            bits_per_weight: 16.0,
            estimated_size_mb: 3000.0,
        };
        let json = serde_json::to_string(&qs).expect("serialize");
        assert!(json.contains("\"format\":\"F16\""));
        assert!(json.contains("\"supported\":true"));
    }

    #[test]
    fn test_huggingface_data_serialize() {
        let hf = HuggingFaceData {
            repo: "Qwen/Qwen2.5-1.5B".to_string(),
            model_type: Some("qwen2".to_string()),
            pipeline_tag: Some("text-generation".to_string()),
            downloads: Some(1000),
            config_fields: serde_json::json!({"hidden_size": 1536}),
            generation_config: None,
        };
        let json = serde_json::to_string(&hf).expect("serialize");
        assert!(json.contains("\"repo\":\"Qwen/Qwen2.5-1.5B\""));
        assert!(!json.contains("generation_config"));
    }

    #[test]
    fn test_huggingface_data_all_none() {
        let hf = HuggingFaceData {
            repo: "test/model".to_string(),
            model_type: None,
            pipeline_tag: None,
            downloads: None,
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let json = serde_json::to_string(&hf).expect("serialize");
        assert!(!json.contains("model_type"));
        assert!(!json.contains("pipeline_tag"));
        assert!(!json.contains("downloads"));
    }

    // ========================================================================
    // Report with All Optional Fields Populated
    // ========================================================================

    #[test]
    fn test_report_with_all_fields() {
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
            size_variant: None,
            format: None,
            compliance: Some(ComplianceResult {
                is_compliant: false,
                tensor_count_match: false,
                missing_tensors: vec!["layer.0.attn.q_proj.weight".to_string()],
                unexpected_tensors: vec!["extra.weight".to_string()],
            }),
            certification: Some(CertificationInfo {
                status: "PENDING".to_string(),
                playbook_path: Some("/playbooks/1.5b.yaml".to_string()),
            }),
            tensors: Some(vec![TensorComplianceEntry {
                name: "embed.weight".to_string(),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![151936, 1536]),
                note: Some("embedding".to_string()),
            }]),
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: Some(CrossValidation {
                matches: vec![],
                mismatches: vec![],
                contract_only: vec![],
                hf_only: vec![],
            }),
            hf_data: None,
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("compliance"));
        assert!(json.contains("certification"));
        assert!(json.contains("tensors"));
        assert!(json.contains("cross_validation"));
    }

    // ========================================================================
    // OracleMode Tests
    // ========================================================================

    #[test]
    fn test_oracle_mode_debug() {
        let mode = OracleMode::Local;
        let debug = format!("{mode:?}");
        assert!(debug.contains("Local"));
    }

    #[test]
    fn test_oracle_mode_clone() {
        let mode = OracleMode::HuggingFace;
        let cloned = mode.clone();
        let json = serde_json::to_string(&cloned).expect("serialize");
        assert_eq!(json, "\"hugging_face\"");
    }

    // ========================================================================
    // build_certification Tests
    // ========================================================================

    #[test]
    fn test_build_certification_no_cert_config() {
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
            certification: None,
        };

        let cert = build_certification(&config, Some("7b"));
        assert!(cert.is_none());
    }

    #[test]
    fn test_build_certification_no_size() {
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
                playbook_path: "../playbooks/{size}.yaml".to_string(),
                csv_family_key: "test".to_string(),
                size_categories: HashMap::new(),
            }),
        };

        let cert = build_certification(&config, None);
        assert!(cert.is_some());
        let cert = cert.expect("cert exists");
        assert!(cert.playbook_path.is_none());
    }
