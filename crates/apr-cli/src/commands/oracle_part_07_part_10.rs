
    #[test]
    fn test_format_text_compliance_compliant() {
        let cr = ComplianceResult {
            is_compliant: true,
            tensor_count_match: true,
            missing_tensors: vec![],
            unexpected_tensors: vec![],
        };
        let out = format_text_compliance(&cr, false);
        assert!(out.contains("COMPLIANT"));
        assert!(!out.contains("NON-COMPLIANT"));
    }

    #[test]
    fn test_format_text_compliance_non_compliant() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: false,
            missing_tensors: vec![
                "layer.0.q.weight".to_string(),
                "layer.0.k.weight".to_string(),
            ],
            unexpected_tensors: vec!["extra.bias".to_string()],
        };
        let out = format_text_compliance(&cr, false);
        assert!(out.contains("NON-COMPLIANT"));
        assert!(out.contains("MISMATCH"));
        assert!(out.contains("2 tensor(s)"));
        assert!(!out.contains("layer.0.q.weight")); // not verbose
    }

    #[test]
    fn test_format_text_compliance_non_compliant_verbose() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: false,
            missing_tensors: vec!["layer.0.q.weight".to_string()],
            unexpected_tensors: vec!["extra.bias".to_string()],
        };
        let out = format_text_compliance(&cr, true);
        assert!(out.contains("NON-COMPLIANT"));
        assert!(out.contains("- layer.0.q.weight"));
        assert!(out.contains("+ extra.bias"));
        assert!(out.contains("Unexpected Tensors: 1 tensor(s)"));
    }

    #[test]
    fn test_format_text_compliance_count_mismatch_only() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: false,
            missing_tensors: vec![],
            unexpected_tensors: vec![],
        };
        let out = format_text_compliance(&cr, false);
        assert!(out.contains("NON-COMPLIANT"));
        assert!(out.contains("MISMATCH"));
        assert!(!out.contains("Missing"));
    }

    #[test]
    fn test_format_text_compliance_unexpected_hidden_not_verbose() {
        let cr = ComplianceResult {
            is_compliant: false,
            tensor_count_match: true,
            missing_tensors: vec![],
            unexpected_tensors: vec!["extra.weight".to_string()],
        };
        let out = format_text_compliance(&cr, false);
        assert!(!out.contains("Unexpected")); // not shown when not verbose
    }

    #[test]
    fn test_format_text_certification_with_playbook() {
        let cert = CertificationInfo {
            status: "PENDING".to_string(),
            playbook_path: Some("/playbooks/1.5b.yaml".to_string()),
        };
        let out = format_text_certification(&cert);
        assert!(out.contains("PENDING"));
        assert!(out.contains("/playbooks/1.5b.yaml"));
    }

    #[test]
    fn test_format_text_certification_no_playbook() {
        let cert = CertificationInfo {
            status: "APPROVED".to_string(),
            playbook_path: None,
        };
        let out = format_text_certification(&cert);
        assert!(out.contains("APPROVED"));
        assert!(!out.contains("Playbook"));
    }

    #[test]
    fn test_format_text_tensors_few() {
        let tensors = vec![
            TensorComplianceEntry {
                name: "model.embed_tokens.weight".to_string(),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![151936, 1536]),
                note: None,
            },
            TensorComplianceEntry {
                name: "lm_head.weight".to_string(),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![151936, 1536]),
                note: None,
            },
        ];
        let out = format_text_tensors(&tensors, false);
        assert!(out.contains("Tensors (2 total)"));
        assert!(out.contains("model.embed_tokens.weight"));
        assert!(out.contains("lm_head.weight"));
        assert!(out.contains("F16"));
        assert!(out.contains("[151936, 1536]"));
    }

    #[test]
    fn test_format_text_tensors_truncated() {
        // Create 25 tensors — should truncate at 20 when not verbose
        let tensors: Vec<TensorComplianceEntry> = (0..25)
            .map(|i| TensorComplianceEntry {
                name: format!("layer.{i}.weight"),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![100, 100]),
                note: None,
            })
            .collect();
        let out = format_text_tensors(&tensors, false);
        assert!(out.contains("Tensors (25 total)"));
        assert!(out.contains("... (3 more) ...")); // 25 - 20 - 2 = 3
        assert!(out.contains("layer.0.weight")); // first shown
        assert!(out.contains("layer.23.weight")); // last 2 always shown
        assert!(out.contains("layer.24.weight"));
    }

    #[test]
    fn test_format_text_tensors_verbose_all_shown() {
        let tensors: Vec<TensorComplianceEntry> = (0..25)
            .map(|i| TensorComplianceEntry {
                name: format!("layer.{i}.weight"),
                present: true,
                dtype: Some("F16".to_string()),
                shape: Some(vec![100]),
                note: None,
            })
            .collect();
        let out = format_text_tensors(&tensors, true);
        assert!(!out.contains("more")); // no truncation in verbose
        for i in 0..25 {
            assert!(
                out.contains(&format!("layer.{i}.weight")),
                "Missing layer.{i}.weight"
            );
        }
    }

    #[test]
    fn test_format_text_tensors_no_dtype_no_shape() {
        let tensors = vec![TensorComplianceEntry {
            name: "unknown.weight".to_string(),
            present: false,
            dtype: None,
            shape: None,
            note: Some("MISSING".to_string()),
        }];
        let out = format_text_tensors(&tensors, false);
        assert!(out.contains("unknown.weight"));
        assert!(out.contains("Tensors (1 total)"));
    }

    #[test]
    fn test_format_family_description_header_basic() {
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hf_pattern: "Qwen/Qwen2*".to_string(),
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
        let out = format_family_description_header(&config);
        assert!(out.contains("qwen2"));
        assert!(out.contains("Alibaba"));
        assert!(out.contains("Qwen2ForCausalLM"));
        assert!(out.contains("Qwen/Qwen2*"));
        assert!(out.contains("Constraints:"));
        assert!(out.contains("GQA"));
        assert!(out.contains("SiLU"));
        assert!(out.contains("Bias: yes"));
    }

    #[test]
    fn test_format_family_size_variant_basic() {
        let sc = make_test_size();
        let out = format_family_size_variant("1.5b", &sc);
        assert!(out.contains("1.5b (1.5B)"));
        assert!(out.contains("hidden_dim: 1536"));
        assert!(out.contains("num_layers: 28"));
        assert!(out.contains("num_heads: 12"));
        assert!(out.contains("num_kv_heads: 2"));
        assert!(out.contains("intermediate_dim: 8960"));
        assert!(out.contains("vocab_size: 151936"));
        assert!(out.contains("head_dim: 128"));
        assert!(out.contains("rope_theta: 1000000"));
        assert!(out.contains("norm_eps:"));
    }

    #[test]
    fn test_format_family_size_variant_no_rope() {
        let mut sc = make_test_size();
        sc.rope_theta = 0.0;
        let out = format_family_size_variant("test", &sc);
        assert!(!out.contains("rope_theta"));
    }

    #[test]
    fn test_format_family_variant_stats_basic() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let stats = build_statistical_analysis(&size, &constraints);
        let out = format_family_variant_stats(&stats);
        assert!(out.contains("GQA Ratio:"));
        assert!(out.contains("KV reduction"));
        assert!(out.contains("Est. Parameters:"));
        assert!(out.contains("Model Size (F16):"));
        assert!(out.contains("Model Size (Q4):"));
        assert!(out.contains("KV Cache (4K):"));
        assert!(out.contains("FFN Ratio:"));
    }

    #[test]
    fn test_format_family_variant_explain_basic() {
        let expl = ArchitectureExplanation {
            attention_explanation: "GQA attention".to_string(),
            ffn_explanation: "SwiGLU FFN".to_string(),
            norm_explanation: "RMSNorm".to_string(),
            positional_explanation: "RoPE".to_string(),
            scaling_analysis: "1.5B scaling".to_string(),
        };
        let out = format_family_variant_explain(&expl);
        assert!(out.contains("Attention: GQA attention"));
        assert!(out.contains("FFN: SwiGLU FFN"));
        assert!(out.contains("Scaling: 1.5B scaling"));
    }

    #[test]
    fn test_format_family_variant_kernels_basic() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "GQA fused".to_string(),
            ffn_kernel: "SwiGLU fused".to_string(),
            estimated_tps_cpu: Some(55.0),
            estimated_tps_gpu: Some(1100.0),
            memory_required_mb: 900.0,
            notes: vec![],
        };
        let out = format_family_variant_kernels(&kern);
        assert!(out.contains("Attn Kernel: GQA fused"));
        assert!(out.contains("FFN Kernel: SwiGLU fused"));
        assert!(out.contains("Est. CPU tok/s: 55"));
        assert!(out.contains("Est. GPU tok/s: 1100"));
        assert!(out.contains("900.0 MB"));
    }

    #[test]
    fn test_format_family_variant_kernels_no_tps() {
        let kern = KernelCompatibility {
            supported_quantizations: vec![],
            attention_kernel: "test".to_string(),
            ffn_kernel: "test".to_string(),
            estimated_tps_cpu: None,
            estimated_tps_gpu: None,
            memory_required_mb: 0.0,
            notes: vec![],
        };
        let out = format_family_variant_kernels(&kern);
        assert!(!out.contains("Est. CPU"));
        assert!(!out.contains("Est. GPU"));
    }

    #[test]
    fn test_format_family_description_footer_verbose() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let mut per_layer = HashMap::new();
        per_layer.insert(
            "q_proj".to_string(),
            Some("model.layers.{n}.q_proj.weight".to_string()),
        );

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: TensorTemplate {
                embedding: "embed.weight".to_string(),
                lm_head: Some("lm_head.weight".to_string()),
                final_norm: Some("norm.weight".to_string()),
                per_layer,
            },
            gguf_tensor_template: GgufTensorTemplate::default(),
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec!["q4_k_m".to_string(), "q6_k".to_string()],
            chat_template: Some(ChatTemplateConfig {
                format: "chatml".to_string(),
                template: String::new(),
                bos_token: "<|im_start|>".to_string(),
                eos_token: "<|im_end|>".to_string(),
                special_tokens: HashMap::new(),
            }),
            certification: None,
        };

        let out = format_family_description_footer(&config, true);
        assert!(out.contains("Tensor Template:"));
        assert!(out.contains("Embedding: embed.weight"));
        assert!(out.contains("LM Head: lm_head.weight"));
        assert!(out.contains("Final Norm: norm.weight"));
        assert!(out.contains("Per-layer:"));
        assert!(out.contains("q_proj: model.layers.{n}.q_proj.weight"));
        assert!(out.contains("Quantizations: q4_k_m, q6_k"));
        assert!(out.contains("Chat Template: chatml"));
        assert!(out.contains("BOS: <|im_start|>"));
        assert!(out.contains("EOS: <|im_end|>"));
    }

    #[test]
    fn test_format_family_description_footer_not_verbose() {
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
                embedding: "embed.weight".to_string(),
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

        let out = format_family_description_footer(&config, false);
        assert!(!out.contains("Tensor Template")); // hidden when not verbose
    }

    #[test]
    fn test_format_family_description_footer_empty_quantizations() {
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

        let out = format_family_description_footer(&config, false);
        assert!(!out.contains("Quantizations"));
    }
