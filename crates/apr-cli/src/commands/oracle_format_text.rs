
    #[test]
    fn test_format_text_constraints_with_bias() {
        let fi = FamilyInfo {
            name: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "GQA".to_string(),
                activation: "SiLU".to_string(),
                norm: "RMSNorm".to_string(),
                bias: true,
                tied_embeddings: false,
                mlp: "SwiGLU".to_string(),
                positional_encoding: "RoPE".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_constraints(&fi);
        assert!(out.contains("GQA"));
        assert!(out.contains("SiLU"));
        assert!(out.contains("RMSNorm"));
        assert!(out.contains("Bias: yes"));
        assert!(out.contains("Tied: no"));
        assert!(out.contains("SwiGLU"));
        assert!(out.contains("RoPE"));
    }

    #[test]
    fn test_format_text_constraints_no_bias_tied() {
        let fi = FamilyInfo {
            name: "gpt2".to_string(),
            display_name: "GPT-2".to_string(),
            vendor: "OpenAI".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "MHA".to_string(),
                activation: "GELU".to_string(),
                norm: "LayerNorm".to_string(),
                bias: false,
                tied_embeddings: true,
                mlp: "GELU MLP".to_string(),
                positional_encoding: "Absolute".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_constraints(&fi);
        assert!(out.contains("Bias: no"));
        assert!(out.contains("Tied: yes"));
        assert!(out.contains("MHA"));
        assert!(out.contains("LayerNorm"));
    }

    #[test]
    fn test_format_text_constraints_header() {
        let fi = FamilyInfo {
            name: "t".to_string(),
            display_name: "T".to_string(),
            vendor: "V".to_string(),
            architectures: vec![],
            constraints: ConstraintsSummary {
                attention: "MQA".to_string(),
                activation: "SiLU".to_string(),
                norm: "RMSNorm".to_string(),
                bias: false,
                tied_embeddings: false,
                mlp: "SwiGLU".to_string(),
                positional_encoding: "ALiBi".to_string(),
            },
            chat_template_format: None,
        };
        let out = format_text_constraints(&fi);
        assert!(out.contains("Constraints:"));
        assert!(out.contains("MQA"));
        assert!(out.contains("ALiBi"));
    }

    #[test]
    fn test_format_text_stats_basic() {
        let stats = StatisticalAnalysis {
            gqa_ratio: 0.167,
            kv_cache_reduction: 0.833,
            model_params: 1_500_000_000,
            model_size_f16_mb: 2861.0,
            model_size_q4_mb: 715.0,
            kv_cache_per_token_bytes: 28672,
            kv_cache_4k_mb: 112.0,
            ffn_expansion_ratio: 5.83,
            ffn_type_explanation: "SwiGLU gated".to_string(),
            rope_max_wavelength: 6283185.0,
            effective_context_window: 32768,
            attention_flops_per_token: 100_000_000,
            ffn_flops_per_token: 200_000_000,
        };
        let out = format_text_stats(&stats);
        assert!(out.contains("0.17"));
        assert!(out.contains("83%"));
        assert!(out.contains("1.5B"));
        assert!(out.contains("2861.0 MB"));
        assert!(out.contains("715.0 MB"));
        assert!(out.contains("28672 bytes"));
        assert!(out.contains("112.0 MB"));
        assert!(out.contains("5.83x"));
        assert!(out.contains("SwiGLU gated"));
        assert!(out.contains("32768"));
    }

    #[test]
    fn test_format_text_stats_no_rope() {
        let stats = StatisticalAnalysis {
            gqa_ratio: 1.0,
            kv_cache_reduction: 0.0,
            model_params: 100_000,
            model_size_f16_mb: 0.2,
            model_size_q4_mb: 0.05,
            kv_cache_per_token_bytes: 100,
            kv_cache_4k_mb: 0.4,
            ffn_expansion_ratio: 4.0,
            ffn_type_explanation: "Standard GELU".to_string(),
            rope_max_wavelength: 0.0,
            effective_context_window: 2048,
            attention_flops_per_token: 1000,
            ffn_flops_per_token: 2000,
        };
        let out = format_text_stats(&stats);
        assert!(!out.contains("RoPE Wavelength"));
        assert!(out.contains("Standard GELU"));
    }

    #[test]
    fn test_format_text_stats_with_rope() {
        let stats = StatisticalAnalysis {
            gqa_ratio: 0.25,
            kv_cache_reduction: 0.75,
            model_params: 7_000_000_000,
            model_size_f16_mb: 13000.0,
            model_size_q4_mb: 3250.0,
            kv_cache_per_token_bytes: 65536,
            kv_cache_4k_mb: 256.0,
            ffn_expansion_ratio: 3.5,
            ffn_type_explanation: "SwiGLU".to_string(),
            rope_max_wavelength: 62831.0,
            effective_context_window: 131072,
            attention_flops_per_token: 500_000_000,
            ffn_flops_per_token: 800_000_000,
        };
        let out = format_text_stats(&stats);
        assert!(out.contains("RoPE Wavelength: 62831"));
        assert!(out.contains("131072"));
    }

    #[test]
    fn test_format_text_stats_flops_format() {
        let stats = StatisticalAnalysis {
            gqa_ratio: 0.5,
            kv_cache_reduction: 0.5,
            model_params: 1_000_000,
            model_size_f16_mb: 1.9,
            model_size_q4_mb: 0.48,
            kv_cache_per_token_bytes: 512,
            kv_cache_4k_mb: 2.0,
            ffn_expansion_ratio: 4.0,
            ffn_type_explanation: "test".to_string(),
            rope_max_wavelength: 100.0,
            effective_context_window: 1024,
            attention_flops_per_token: 123_456_789,
            ffn_flops_per_token: 987_654_321,
        };
        let out = format_text_stats(&stats);
        assert!(out.contains("Attn FLOPS/tok:"));
        assert!(out.contains("FFN FLOPS/tok:"));
        assert!(out.contains("e"));
    }

    #[test]
    fn test_format_text_explanation_basic() {
        let expl = ArchitectureExplanation {
            attention_explanation: "GQA with ratio 0.17".to_string(),
            ffn_explanation: "SwiGLU gated activation".to_string(),
            norm_explanation: "RMSNorm eps=1e-6".to_string(),
            positional_explanation: "RoPE theta=1000000".to_string(),
            scaling_analysis: "1.5B parameters, Chinchilla-optimal".to_string(),
        };
        let out = format_text_explanation(&expl);
        assert!(out.contains("GQA with ratio 0.17"));
        assert!(out.contains("SwiGLU gated activation"));
        assert!(out.contains("RMSNorm eps=1e-6"));
        assert!(out.contains("RoPE theta=1000000"));
        assert!(out.contains("Chinchilla-optimal"));
    }

    #[test]
    fn test_format_text_explanation_sections_labeled() {
        let expl = ArchitectureExplanation {
            attention_explanation: "attn".to_string(),
            ffn_explanation: "ffn".to_string(),
            norm_explanation: "norm".to_string(),
            positional_explanation: "pos".to_string(),
            scaling_analysis: "scale".to_string(),
        };
        let out = format_text_explanation(&expl);
        assert!(out.contains("Attention: attn"));
        assert!(out.contains("FFN: ffn"));
        assert!(out.contains("Normalization: norm"));
        assert!(out.contains("Position: pos"));
        assert!(out.contains("Scaling: scale"));
    }

    #[test]
    fn test_format_text_explanation_empty_strings() {
        let expl = ArchitectureExplanation {
            attention_explanation: String::new(),
            ffn_explanation: String::new(),
            norm_explanation: String::new(),
            positional_explanation: String::new(),
            scaling_analysis: String::new(),
        };
        let out = format_text_explanation(&expl);
        assert!(out.contains("Attention:"));
        assert!(out.contains("FFN:"));
    }

    #[test]
    fn test_format_text_kernels_basic() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![
                QuantizationSupport {
                    format: "F16".to_string(),
                    supported: true,
                    kernel: "trueno::f16_matvec".to_string(),
                    bits_per_weight: 16.0,
                    estimated_size_mb: 3000.0,
                },
                QuantizationSupport {
                    format: "Q4_K_M".to_string(),
                    supported: true,
                    kernel: "fused_q4k".to_string(),
                    bits_per_weight: 4.5,
                    estimated_size_mb: 750.0,
                },
            ],
            attention_kernel: "GQA fused QKV".to_string(),
            ffn_kernel: "SwiGLU fused".to_string(),
            estimated_tps_cpu: Some(60.0),
            estimated_tps_gpu: Some(1200.0),
            memory_required_mb: 850.0,
            notes: vec!["ROW-MAJOR layout".to_string(), "GQA: 6:1 ratio".to_string()],
        };
        let out = format_text_kernels(&kern);
        assert!(out.contains("GQA fused QKV"));
        assert!(out.contains("SwiGLU fused"));
        assert!(out.contains("F16"));
        assert!(out.contains("Q4_K_M"));
        assert!(out.contains("yes"));
        assert!(out.contains("60"));
        assert!(out.contains("1200"));
        assert!(out.contains("850.0 MB"));
        assert!(out.contains("ROW-MAJOR layout"));
        assert!(out.contains("GQA: 6:1 ratio"));
    }

    #[test]
    fn test_format_text_kernels_no_tps() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "MHA standard".to_string(),
            ffn_kernel: "GELU MLP".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 100.0,
            notes: vec![],
        };
        let out = format_text_kernels(&kern);
        assert!(!out.contains("Est. CPU"));
        assert!(!out.contains("Est. GPU"));
        assert!(out.contains("100.0 MB"));
    }

    #[test]
    fn test_format_text_kernels_unsupported_quant() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![QuantizationSupport {
                format: "Q2_K".to_string(),
                supported: false,
                kernel: "none".to_string(),
                bits_per_weight: 2.5,
                estimated_size_mb: 300.0,
            }],
            attention_kernel: "test".to_string(),
            ffn_kernel: "test".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let out = format_text_kernels(&kern);
        assert!(out.contains("Q2_K"));
        assert!(out.contains("no"));
    }

    #[test]
    fn test_format_text_kernels_header_line() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "a".to_string(),
            ffn_kernel: "f".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let out = format_text_kernels(&kern);
        assert!(out.contains("Quantization Support:"));
        assert!(out.contains("Format"));
        assert!(out.contains("Supported"));
        assert!(out.contains("BPW"));
    }

    #[test]
    fn test_format_text_cross_validation_all_match() {
        let cv = CrossValidation {
            matches: vec![
                CrossValidationEntry {
                    field: "hidden_dim".to_string(),
                    contract_value: "1536".to_string(),
                    hf_value: "1536".to_string(),
                    status: "match".to_string(),
                },
                CrossValidationEntry {
                    field: "num_layers".to_string(),
                    contract_value: "28".to_string(),
                    hf_value: "28".to_string(),
                    status: "match".to_string(),
                },
            ],
            mismatches: vec![],
            contract_only: vec![],
            hf_only: vec![],
        };
        let out = format_text_cross_validation(&cv);
        assert!(out.contains("Matches (2)"));
        assert!(out.contains("[OK] hidden_dim: 1536 == 1536"));
        assert!(out.contains("[OK] num_layers: 28 == 28"));
        assert!(!out.contains("Mismatches"));
    }

    #[test]
    fn test_format_text_cross_validation_with_mismatches() {
        let cv = CrossValidation {
            matches: vec![],
            mismatches: vec![CrossValidationEntry {
                field: "hidden_dim".to_string(),
                contract_value: "1536".to_string(),
                hf_value: "2048".to_string(),
                status: "mismatch".to_string(),
            }],
            contract_only: vec!["vocab_size=151936".to_string()],
            hf_only: vec!["rope_scaling=dynamic".to_string()],
        };
        let out = format_text_cross_validation(&cv);
        assert!(out.contains("Mismatches (1)"));
        assert!(out.contains("[!!] hidden_dim: contract=1536 vs hf=2048"));
        assert!(out.contains("Contract-only: vocab_size=151936"));
        assert!(out.contains("HF-only: rope_scaling=dynamic"));
    }

    #[test]
    fn test_format_text_cross_validation_empty() {
        let cv = CrossValidation {
            matches: vec![],
            mismatches: vec![],
            contract_only: vec![],
            hf_only: vec![],
        };
        let out = format_text_cross_validation(&cv);
        assert!(out.is_empty());
    }

    #[test]
    fn test_format_text_cross_validation_multiple_contract_only() {
        let cv = CrossValidation {
            matches: vec![],
            mismatches: vec![],
            contract_only: vec!["a=1".to_string(), "b=2".to_string(), "c=3".to_string()],
            hf_only: vec![],
        };
        let out = format_text_cross_validation(&cv);
        assert!(out.contains("Contract-only: a=1, b=2, c=3"));
    }

    #[test]
    fn test_format_text_hf_all_fields() {
        let hf = HuggingFaceData {
            repo: "Qwen/Qwen2.5-1.5B".to_string(),
            model_type: Some("qwen2".to_string()),
            pipeline_tag: Some("text-generation".to_string()),
            downloads: Some(50000),
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let out = format_text_hf(&hf);
        assert!(out.contains("Qwen/Qwen2.5-1.5B"));
        assert!(out.contains("qwen2"));
        assert!(out.contains("text-generation"));
        assert!(out.contains("50000"));
    }

    #[test]
    fn test_format_text_hf_minimal() {
        let hf = HuggingFaceData {
            repo: "test/model".to_string(),
            model_type: None,
            pipeline_tag: None,
            downloads: None,
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let out = format_text_hf(&hf);
        assert!(out.contains("test/model"));
        assert!(!out.contains("model_type"));
        assert!(!out.contains("pipeline_tag"));
        assert!(!out.contains("Downloads"));
    }

    #[test]
    fn test_format_text_hf_partial_fields() {
        let hf = HuggingFaceData {
            repo: "org/repo".to_string(),
            model_type: Some("llama".to_string()),
            pipeline_tag: None,
            downloads: Some(42),
            config_fields: serde_json::json!({}),
            generation_config: None,
        };
        let out = format_text_hf(&hf);
        assert!(out.contains("llama"));
        assert!(out.contains("42"));
        assert!(!out.contains("pipeline_tag"));
    }
