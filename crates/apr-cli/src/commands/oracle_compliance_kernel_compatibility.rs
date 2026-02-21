
    #[test]
    fn test_compliance_result_non_compliant_serialize() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: false,
            missing_tensors: vec![
                "layer.0.q.weight".to_string(),
                "layer.0.k.weight".to_string(),
            ],
            unexpected_tensors: vec!["extra.bias".to_string()],
        };
        let json = serde_json::to_string(&cr).expect("serialize");
        assert!(json.contains("\"is_compliant\":false"));
        assert!(json.contains("\"tensor_count_match\":false"));
        assert!(json.contains("layer.0.q.weight"));
        assert!(json.contains("extra.bias"));
    }

    #[test]
    fn test_kernel_compatibility_memory_calculation() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let kern = build_kernel_compatibility(&size, &constraints, &stats);

        // memory_required_mb = Q4 model size + KV cache
        let q4_size = (stats.model_params as f64 * 0.5625) / (1024.0 * 1024.0);
        let expected_mem = q4_size + stats.kv_cache_4k_mb;
        assert!(
            (kern.memory_required_mb - expected_mem).abs() < 0.01,
            "Memory should be Q4 model + KV cache"
        );
    }

    #[test]
    fn test_oracle_mode_family_serialize() {
        let mode = OracleMode::Family;
        let json = serde_json::to_string(&mode).expect("serialize");
        assert_eq!(json, "\"family\"");
    }

    #[test]
    fn test_cross_validation_entry_debug() {
        let entry = CrossValidationEntry {
            field: "hidden_dim".to_string(),
            contract_value: "1536".to_string(),
            hf_value: "2048".to_string(),
            status: "mismatch".to_string(),
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("CrossValidationEntry"));
        assert!(debug.contains("hidden_dim"));
    }

    #[test]
    fn test_cross_validation_debug() {
        let cv = CrossValidation {
            matches: vec![],
            mismatches: vec![],
            contract_only: vec![],
            hf_only: vec![],
        };
        let debug = format!("{cv:?}");
        assert!(debug.contains("CrossValidation"));
    }

    #[test]
    fn test_statistical_analysis_debug() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let debug = format!("{stats:?}");
        assert!(debug.contains("StatisticalAnalysis"));
        assert!(debug.contains("gqa_ratio"));
    }

    #[test]
    fn test_architecture_explanation_debug() {
        let expl = ArchitectureExplanation {
            attention_explanation: "test".to_string(),
            ffn_explanation: "test".to_string(),
            norm_explanation: "test".to_string(),
            positional_explanation: "test".to_string(),
            scaling_analysis: "test".to_string(),
        };
        let debug = format!("{expl:?}");
        assert!(debug.contains("ArchitectureExplanation"));
    }

    #[test]
    fn test_kernel_compatibility_debug() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "test".to_string(),
            ffn_kernel: "test".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let debug = format!("{kern:?}");
        assert!(debug.contains("KernelCompatibility"));
    }

    #[test]
    fn test_quantization_support_debug() {
        let qs = QuantizationSupport {
            format: "Q4_K_M".to_string(),
            supported: true,
            kernel: "fused_q4k".to_string(),
            bits_per_weight: 4.5,
            estimated_size_mb: 500.0,
        };
        let debug = format!("{qs:?}");
        assert!(debug.contains("QuantizationSupport"));
    }

    #[test]
    fn test_huggingface_data_debug() {
        let hf = HuggingFaceData {
            repo: "test/model".to_string(),
            model_type: None,
            pipeline_tag: None,
            downloads: None,
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let debug = format!("{hf:?}");
        assert!(debug.contains("HuggingFaceData"));
    }

    #[test]
    fn test_offline_hf_huggingface_prefix_rejected() {
        let src = "huggingface://Qwen/Qwen2.5-1.5B".to_string();
        let flags = OracleFlags::default();
        let result = run(
            Some(&src),
            None,
            None,
            false,
            false,
            false,
            false,
            true,
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

    // ========================================================================
    // Format Function Tests (coverage for output formatting)
    // ========================================================================

    #[test]
    fn test_format_text_report_basic() {
        let report = ModelOracleReport {
            source: "test.gguf".to_string(),
            mode: OracleMode::Local,
            family: None,
            size_variant: None,
            format: None,
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };
        let out = format_text_report(&report);
        assert!(out.contains("test.gguf"));
        assert!(out.contains("Local"));
    }

    #[test]
    fn test_format_text_report_hf_mode() {
        let report = ModelOracleReport {
            source: "hf://Qwen/Qwen2.5-1.5B".to_string(),
            mode: OracleMode::HuggingFace,
            family: None,
            size_variant: None,
            format: None,
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };
        let out = format_text_report(&report);
        assert!(out.contains("hf://Qwen/Qwen2.5-1.5B"));
        assert!(out.contains("HuggingFace"));
    }

    #[test]
    fn test_format_text_report_family_mode() {
        let report = ModelOracleReport {
            source: "qwen2".to_string(),
            mode: OracleMode::Family,
            family: None,
            size_variant: None,
            format: None,
            compliance: None,
            certification: None,
            tensors: None,
            stats: None,
            explanation: None,
            kernel_compatibility: None,
            cross_validation: None,
            hf_data: None,
        };
        let out = format_text_report(&report);
        assert!(out.contains("qwen2"));
        assert!(out.contains("Family"));
    }

    #[test]
    fn test_format_text_format_basic() {
        let fmt = FormatInfo {
            format_type: "GGUF".to_string(),
            file_size: 4_000_000_000,
            tensor_count: 291,
            total_params: 7_000_000_000,
            quantization: Some("Q4_K_M".to_string()),
            architecture: Some("LlamaForCausalLM".to_string()),
        };
        let out = format_text_format(&fmt);
        assert!(out.contains("GGUF"));
        assert!(out.contains("291"));
        assert!(out.contains("Q4_K_M"));
        assert!(out.contains("LlamaForCausalLM"));
        assert!(out.contains("7.0B"));
    }

    #[test]
    fn test_format_text_format_no_optionals() {
        let fmt = FormatInfo {
            format_type: "SafeTensors".to_string(),
            file_size: 1_000_000,
            tensor_count: 100,
            total_params: 500_000,
            quantization: None,
            architecture: None,
        };
        let out = format_text_format(&fmt);
        assert!(out.contains("SafeTensors"));
        assert!(out.contains("100"));
        assert!(!out.contains("Quantization"));
        assert!(!out.contains("Architecture"));
    }

    #[test]
    fn test_format_text_format_small_params() {
        let fmt = FormatInfo {
            format_type: "APR".to_string(),
            file_size: 1024,
            tensor_count: 5,
            total_params: 500,
            quantization: None,
            architecture: None,
        };
        let out = format_text_format(&fmt);
        assert!(out.contains("APR"));
        assert!(out.contains("500"));
    }

    #[test]
    fn test_format_text_family_basic() {
        let fi = FamilyInfo {
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
        };
        let out = format_text_family(&fi, false);
        assert!(out.contains("qwen2 (Qwen2)"));
        assert!(out.contains("Alibaba"));
        assert!(!out.contains("Qwen2ForCausalLM")); // not verbose
        assert!(out.contains("chatml"));
    }

    #[test]
    fn test_format_text_family_verbose() {
        let fi = FamilyInfo {
            name: "llama".to_string(),
            display_name: "LLaMA".to_string(),
            vendor: "Meta".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string(), "LlamaModel".to_string()],
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
        let out = format_text_family(&fi, true);
        assert!(out.contains("LlamaForCausalLM, LlamaModel"));
        assert!(!out.contains("Chat Template"));
    }

    #[test]
    fn test_format_text_family_no_chat_template() {
        let fi = FamilyInfo {
            name: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "TestCo".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "MHA".to_string(),
                activation: "GELU".to_string(),
                norm: "LayerNorm".to_string(),
                bias: true,
                tied_embeddings: true,
                mlp: "GELU MLP".to_string(),
                positional_encoding: "Absolute".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_family(&fi, false);
        assert!(!out.contains("Chat Template"));
    }

    #[test]
    fn test_format_text_family_empty_architectures_verbose() {
        let fi = FamilyInfo {
            name: "t".to_string(),
            display_name: "T".to_string(),
            vendor: "V".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "MHA".to_string(),
                activation: "GELU".to_string(),
                norm: "LayerNorm".to_string(),
                bias: false,
                tied_embeddings: false,
                mlp: "GELU MLP".to_string(),
                positional_encoding: "Absolute".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_family(&fi, true);
        assert!(out.contains("Architectures:"));
    }

    #[test]
    fn test_format_text_size_basic() {
        let svi = SizeVariantInfo {
            name: "1.5b".to_string(),
            parameters: "1.5B".to_string(),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            intermediate_dim: 8960,
            vocab_size: 151936,
            expected_tensor_count: 339,
        };
        let out = format_text_size(&svi);
        assert!(out.contains("1.5B"));
        assert!(out.contains("hidden=1536"));
        assert!(out.contains("layers=28"));
        assert!(out.contains("heads=12"));
        assert!(out.contains("kv_heads=2"));
        assert!(out.contains("8960"));
        assert!(out.contains("151936"));
        assert!(out.contains("339"));
    }

    #[test]
    fn test_format_text_size_large_model() {
        let svi = SizeVariantInfo {
            name: "70b".to_string(),
            parameters: "70B".to_string(),
            hidden_dim: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            intermediate_dim: 28672,
            vocab_size: 128256,
            expected_tensor_count: 723,
        };
        let out = format_text_size(&svi);
        assert!(out.contains("70B"));
        assert!(out.contains("hidden=8192"));
        assert!(out.contains("723"));
    }

    #[test]
    fn test_format_text_size_minimal() {
        let svi = SizeVariantInfo {
            name: "tiny".to_string(),
            parameters: "10M".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 128,
            vocab_size: 100,
            expected_tensor_count: 20,
        };
        let out = format_text_size(&svi);
        assert!(out.contains("10M"));
        assert!(out.contains("Intermediate Dim: 128"));
    }
