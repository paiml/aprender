
    // ========================================================================
    // Tensor Compliance Entry Tests
    // ========================================================================

    #[test]
    fn test_tensor_compliance_entry_missing_tensor() {
        let entry = TensorComplianceEntry {
            name: "model.layers.0.attn.q_proj.weight".to_string(),
            present: false,
            dtype: None,
            shape: None,
            note: Some("MISSING".to_string()),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("\"present\":false"));
        assert!(json.contains("\"note\":\"MISSING\""));
        // dtype and shape should be skipped
        assert!(!json.contains("\"dtype\""));
        assert!(!json.contains("\"shape\""));
    }

    // ========================================================================
    // Constraints Summary Tests
    // ========================================================================

    #[test]
    fn test_constraints_summary_serialize() {
        let cs = ConstraintsSummary {
            attention: "GQA".to_string(),
            activation: "SiLU".to_string(),
            norm: "RMSNorm".to_string(),
            bias: true,
            tied_embeddings: false,
            mlp: "SwiGLU".to_string(),
            positional_encoding: "RoPE".to_string(),
        };
        let json = serde_json::to_string(&cs).expect("serialize");
        assert!(json.contains("\"bias\":true"));
        assert!(json.contains("\"tied_embeddings\":false"));
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_cross_validation_rope_theta_relative_approximate() {
        // Test the "approximate" path: abs diff > 1.0 but relative diff < 1%
        let mut size = make_test_size();
        size.rope_theta = 100_000.0;
        let constraints = make_test_constraints();
        // abs diff = 500 > 1.0, but relative = 500/100000 = 0.5% < 1%
        let hf_config = serde_json::json!({
            "rope_theta": 100_500.0
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let rope_entry = cv.matches.iter().find(|e| e.field == "rope_theta");
        assert!(
            rope_entry.is_some(),
            "rope_theta should approximately match"
        );
        assert_eq!(
            rope_entry.expect("exists").status,
            "approximate",
            "Status should be 'approximate'"
        );
    }

    #[test]
    fn test_cross_validation_hf_value_bool() {
        // Test when HF config has a non-number, non-string value (e.g. bool/object)
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "hidden_size": true
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        // "true" != "1536" => mismatch
        let entry = cv.mismatches.iter().find(|e| e.field == "hidden_dim");
        assert!(
            entry.is_some(),
            "bool HF value should mismatch with numeric contract value"
        );
    }

    #[test]
    fn test_cross_validation_hf_value_array() {
        let size = make_test_size();
        let constraints = make_test_constraints();
        let hf_config = serde_json::json!({
            "hidden_size": [1536]
        });

        let cv = cross_validate(&size, &constraints, &hf_config);
        let entry = cv.mismatches.iter().find(|e| e.field == "hidden_dim");
        assert!(entry.is_some(), "array HF value should mismatch");
    }

    #[test]
    fn test_expand_tensor_template_empty() {
        use std::collections::HashMap;

        let template = aprender::format::model_family::TensorTemplate {
            embedding: String::new(),
            lm_head: None,
            final_norm: None,
            per_layer: HashMap::new(),
        };
        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: template.clone(),
            gguf_tensor_template: aprender::format::model_family::GgufTensorTemplate::default(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let names = expand_tensor_template(&template, &config, "7b");
        assert!(names.is_empty(), "Empty template should produce no names");
    }

    #[test]
    fn test_expand_tensor_template_with_globals() {
        use std::collections::HashMap;

        let template = aprender::format::model_family::TensorTemplate {
            embedding: "model.embed_tokens.weight".to_string(),
            lm_head: Some("lm_head.weight".to_string()),
            final_norm: Some("model.norm.weight".to_string()),
            per_layer: HashMap::new(),
        };
        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: make_test_constraints(),
            tensor_template: template.clone(),
            gguf_tensor_template: aprender::format::model_family::GgufTensorTemplate::default(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let names = expand_tensor_template(&template, &config, "nonexistent_size");
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"lm_head.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
    }

    #[test]
    fn test_expand_tensor_template_with_per_layer() {
        use std::collections::HashMap;

        let mut per_layer = HashMap::new();
        per_layer.insert(
            "q_proj".to_string(),
            Some("model.layers.{n}.self_attn.q_proj.weight".to_string()),
        );
        per_layer.insert(
            "k_proj".to_string(),
            Some("model.layers.{n}.self_attn.k_proj.weight".to_string()),
        );

        let template = aprender::format::model_family::TensorTemplate {
            embedding: "embed.weight".to_string(),
            lm_head: None,
            final_norm: None,
            per_layer,
        };

        let mut size_variants = HashMap::new();
        size_variants.insert(
            "tiny".to_string(),
            ModelSizeConfig {
                parameters: "tiny".to_string(),
                hidden_dim: 64,
                num_layers: 2,
                num_heads: 2,
                num_kv_heads: 2,
                intermediate_dim: 128,
                vocab_size: 100,
                max_position_embeddings: 512,
                head_dim: 32,
                rope_theta: 10000.0,
                norm_eps: 1e-5,
            },
        );

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants,
            constraints: make_test_constraints(),
            tensor_template: template.clone(),
            gguf_tensor_template: aprender::format::model_family::GgufTensorTemplate::default(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let names = expand_tensor_template(&template, &config, "tiny");
        // 1 embedding + 2 layers * 2 per-layer tensors = 5
        assert_eq!(names.len(), 5);
        assert!(names.contains(&"embed.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.k_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.k_proj.weight".to_string()));
    }

    #[test]
    fn test_expand_tensor_template_per_layer_with_none_value() {
        use std::collections::HashMap;

        let mut per_layer = HashMap::new();
        per_layer.insert(
            "q_proj".to_string(),
            Some("model.layers.{n}.q.weight".to_string()),
        );
        per_layer.insert("bias".to_string(), None); // Optional tensor not present

        let template = aprender::format::model_family::TensorTemplate {
            embedding: "embed.weight".to_string(),
            lm_head: None,
            final_norm: None,
            per_layer,
        };

        let mut size_variants = HashMap::new();
        size_variants.insert(
            "tiny".to_string(),
            ModelSizeConfig {
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
            },
        );

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test".to_string(),
            vendor: "Test".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants,
            constraints: make_test_constraints(),
            tensor_template: template.clone(),
            gguf_tensor_template: aprender::format::model_family::GgufTensorTemplate::default(),
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: None,
        };

        let names = expand_tensor_template(&template, &config, "tiny");
        // 1 embedding + 1 layer * 1 per-layer (None skipped by flatten) = 2
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"embed.weight".to_string()));
        assert!(names.contains(&"model.layers.0.q.weight".to_string()));
    }

    #[test]
    fn test_oracle_flags_combined_stats_and_explain() {
        let flags = OracleFlags {
            stats: true,
            explain: true,
            kernels: false,
            validate: false,
            full: false,
        };
        assert!(flags.show_stats());
        assert!(flags.show_explain());
        assert!(!flags.show_kernels());
        assert!(!flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_combined_kernels_and_validate() {
        let flags = OracleFlags {
            stats: false,
            explain: false,
            kernels: true,
            validate: true,
            full: false,
        };
        assert!(!flags.show_stats());
        assert!(!flags.show_explain());
        assert!(flags.show_kernels());
        assert!(flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_full_overrides_individual() {
        let flags = OracleFlags {
            stats: false,
            explain: false,
            kernels: false,
            validate: false,
            full: true,
        };
        // full=true should make all show_* true even if individual flags are false
        assert!(flags.show_stats());
        assert!(flags.show_explain());
        assert!(flags.show_kernels());
        assert!(flags.show_validate());
    }

    #[test]
    fn test_oracle_flags_copy_semantics() {
        let flags = OracleFlags {
            stats: true,
            explain: false,
            kernels: true,
            validate: false,
            full: false,
        };
        let copied = flags;
        // After copy, original should still work (Copy trait)
        assert!(flags.show_stats());
        assert!(copied.show_stats());
        assert!(flags.show_kernels());
        assert!(copied.show_kernels());
    }

    #[test]
    fn test_report_all_none_fields_serialize() {
        let report = ModelOracleReport {
            source: "minimal.gguf".to_string(),
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

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"source\": \"minimal.gguf\""));
        assert!(json.contains("\"mode\": \"local\""));
        // All optional fields should be absent due to skip_serializing_if
        assert!(!json.contains("\"family\""));
        assert!(!json.contains("\"size_variant\""));
        assert!(!json.contains("\"format\""));
        assert!(!json.contains("\"compliance\""));
        assert!(!json.contains("\"certification\""));
        assert!(!json.contains("\"tensors\""));
        assert!(!json.contains("\"stats\""));
        assert!(!json.contains("\"explanation\""));
        assert!(!json.contains("\"kernel_compatibility\""));
        assert!(!json.contains("\"cross_validation\""));
        assert!(!json.contains("\"hf_data\""));
    }

    #[test]
    fn test_huggingface_data_with_generation_config() {
        let hf = HuggingFaceData {
            repo: "test/model".to_string(),
            model_type: Some("llama".to_string()),
            pipeline_tag: None,
            downloads: Some(42),
            config_fields: serde_json::json!({"hidden_size": 4096}),
            generation_config: Some(serde_json::json!({
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 2048
            })),
        };
        let json = serde_json::to_string(&hf).expect("serialize");
        assert!(json.contains("generation_config"));
        assert!(json.contains("temperature"));
        assert!(json.contains("0.7"));
    }

    #[test]
    fn test_compute_param_count_embedding_contribution() {
        // Verify embedding = vocab_size * hidden_dim
        let size = ModelSizeConfig {
            parameters: "test".to_string(),
            hidden_dim: 100,
            num_layers: 0, // zero layers to isolate embedding
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
        constraints.tied_embeddings = false;

        let params = compute_param_count(&size, &constraints);
        // embedding (1000*100) + lm_head (1000*100) + final_norm (100) + 0 layers
        let expected = 1000 * 100 + 1000 * 100 + 100;
        assert_eq!(params, expected);
    }
