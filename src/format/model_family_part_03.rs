
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_qwen2_config() -> ModelFamilyConfig {
        let mut size_variants = HashMap::new();
        size_variants.insert(
            "0.5b".to_string(),
            ModelSizeConfig {
                parameters: "0.5B".to_string(),
                hidden_dim: 896,
                num_layers: 24,
                num_heads: 14,
                num_kv_heads: 2,
                intermediate_dim: 4864,
                vocab_size: 151_936,
                max_position_embeddings: 32_768,
                head_dim: 64,
                rope_theta: 1_000_000.0,
                norm_eps: 0.000_001,
            },
        );
        size_variants.insert(
            "1.5b".to_string(),
            ModelSizeConfig {
                parameters: "1.5B".to_string(),
                hidden_dim: 1536,
                num_layers: 28,
                num_heads: 12,
                num_kv_heads: 2,
                intermediate_dim: 8960,
                vocab_size: 151_936,
                max_position_embeddings: 32_768,
                head_dim: 128,
                rope_theta: 1_000_000.0,
                norm_eps: 0.000_001,
            },
        );

        let mut per_layer = HashMap::new();
        per_layer.insert(
            "q_proj".to_string(),
            Some("model.layers.{n}.self_attn.q_proj.weight".to_string()),
        );
        per_layer.insert(
            "k_proj".to_string(),
            Some("model.layers.{n}.self_attn.k_proj.weight".to_string()),
        );
        per_layer.insert(
            "v_proj".to_string(),
            Some("model.layers.{n}.self_attn.v_proj.weight".to_string()),
        );
        per_layer.insert(
            "o_proj".to_string(),
            Some("model.layers.{n}.self_attn.o_proj.weight".to_string()),
        );
        per_layer.insert(
            "gate_proj".to_string(),
            Some("model.layers.{n}.mlp.gate_proj.weight".to_string()),
        );
        per_layer.insert(
            "up_proj".to_string(),
            Some("model.layers.{n}.mlp.up_proj.weight".to_string()),
        );
        per_layer.insert(
            "down_proj".to_string(),
            Some("model.layers.{n}.mlp.down_proj.weight".to_string()),
        );
        per_layer.insert(
            "input_layernorm".to_string(),
            Some("model.layers.{n}.input_layernorm.weight".to_string()),
        );
        per_layer.insert(
            "post_attention_layernorm".to_string(),
            Some("model.layers.{n}.post_attention_layernorm.weight".to_string()),
        );
        per_layer.insert(
            "q_proj_bias".to_string(),
            Some("model.layers.{n}.self_attn.q_proj.bias".to_string()),
        );
        per_layer.insert(
            "k_proj_bias".to_string(),
            Some("model.layers.{n}.self_attn.k_proj.bias".to_string()),
        );
        per_layer.insert(
            "v_proj_bias".to_string(),
            Some("model.layers.{n}.self_attn.v_proj.bias".to_string()),
        );

        let mut shapes = HashMap::new();
        shapes.insert(
            "embedding".to_string(),
            "[vocab_size, hidden_dim]".to_string(),
        );
        shapes.insert(
            "lm_head".to_string(),
            "[vocab_size, hidden_dim]".to_string(),
        );
        shapes.insert(
            "q_proj".to_string(),
            "[num_heads * head_dim, hidden_dim]".to_string(),
        );

        ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2 / Qwen2.5-Coder".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hf_pattern: "Qwen/Qwen2*".to_string(),
            size_variants,
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
                embedding: "model.embed_tokens.weight".to_string(),
                lm_head: Some("lm_head.weight".to_string()),
                final_norm: Some("model.norm.weight".to_string()),
                per_layer,
            },
            gguf_tensor_template: GgufTensorTemplate::default(),
            shape_template: ShapeTemplate { shapes },
            quantizations: vec!["q4_k_m".to_string(), "q6_k".to_string(), "f16".to_string()],
            chat_template: None,
            certification: None,
        }
    }

    #[test]
    fn test_dyn_family_detect_size() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        assert_eq!(family.detect_size(896, 24), Some("0.5b".to_string()));
        assert_eq!(family.detect_size(1536, 28), Some("1.5b".to_string()));
        assert_eq!(family.detect_size(999, 99), None);
    }

    #[test]
    fn test_dyn_family_expected_tensor_count() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        // 3 global (embedding, lm_head, final_norm) + 12 per-layer * 24 layers = 291
        assert_eq!(family.expected_tensor_count("0.5b"), Some(291));
        // 3 global + 12 per-layer * 28 layers = 339
        assert_eq!(family.expected_tensor_count("1.5b"), Some(339));
        assert_eq!(family.expected_tensor_count("unknown"), None);
    }

    #[test]
    fn test_registry_detect_family() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        let mut registry = FamilyRegistry::new();
        registry.register(Box::new(family));

        let names = vec![
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
        ];

        let detected = registry.detect_family(&names);
        assert!(detected.is_some());
        assert_eq!(detected.expect("family detected").family_name(), "qwen2");
    }

    #[test]
    fn test_registry_detect_from_model_type() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        let mut registry = FamilyRegistry::new();
        registry.register(Box::new(family));

        let detected = registry.detect_from_model_type("qwen2");
        assert!(detected.is_some());
        assert_eq!(detected.expect("family detected").family_name(), "qwen2");
    }

    #[test]
    fn test_attention_type_parsing() {
        assert_eq!(
            AttentionType::from_str_contract("gqa").expect("parse"),
            AttentionType::Gqa
        );
        assert_eq!(
            AttentionType::from_str_contract("mha").expect("parse"),
            AttentionType::Mha
        );
        assert_eq!(
            AttentionType::from_str_contract("mqa").expect("parse"),
            AttentionType::Mqa
        );
        assert!(AttentionType::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_activation_parsing() {
        assert_eq!(
            Activation::from_str_contract("silu").expect("parse"),
            Activation::Silu
        );
        assert_eq!(
            Activation::from_str_contract("gelu").expect("parse"),
            Activation::Gelu
        );
        assert_eq!(
            Activation::from_str_contract("relu").expect("parse"),
            Activation::Relu
        );
        assert!(Activation::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_norm_type_parsing() {
        assert_eq!(
            NormType::from_str_contract("rmsnorm").expect("parse"),
            NormType::RmsNorm
        );
        assert_eq!(
            NormType::from_str_contract("layernorm").expect("parse"),
            NormType::LayerNorm
        );
        assert!(NormType::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_positional_encoding_parsing() {
        assert_eq!(
            PositionalEncoding::from_str_contract("rope").expect("parse"),
            PositionalEncoding::Rope
        );
        assert_eq!(
            PositionalEncoding::from_str_contract("absolute").expect("parse"),
            PositionalEncoding::Absolute
        );
        assert!(PositionalEncoding::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_mlp_type_parsing() {
        assert_eq!(
            MlpType::from_str_contract("swiglu").expect("parse"),
            MlpType::SwiGlu
        );
        assert_eq!(
            MlpType::from_str_contract("gelu_mlp").expect("parse"),
            MlpType::GeluMlp
        );
        assert!(MlpType::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_validate_tensor_names_rejects_wrong_family() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        let whisper_names = ["encoder.conv1.weight"];
        assert!(family
            .validate_tensor_names(&whisper_names, "0.5b")
            .is_err());
    }

    #[test]
    fn test_contract_error_display() {
        let err = ContractError {
            family: "qwen2".to_string(),
            message: "Missing tensor: lm_head.weight".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Model family contract error [qwen2]: Missing tensor: lm_head.weight"
        );
    }

    #[test]
    fn test_family_registry_empty() {
        let registry = FamilyRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.detect_family(&["foo"]).is_none());
    }

    // PMAT-250: Build-time generated code tests

    #[test]
    fn pmat_250_known_families_populated() {
        assert!(
            !KNOWN_FAMILIES.is_empty(),
            "KNOWN_FAMILIES should be populated by build.rs"
        );
        assert!(
            KNOWN_FAMILIES.contains(&"qwen2"),
            "KNOWN_FAMILIES should contain qwen2"
        );
        assert!(
            KNOWN_FAMILIES.contains(&"llama"),
            "KNOWN_FAMILIES should contain llama"
        );
    }

    #[test]
    fn pmat_250_build_default_registry() {
        let registry = build_default_registry();
        assert!(
            !registry.is_empty(),
            "Default registry should contain families from YAML contracts"
        );
        assert!(
            registry.len() >= 8,
            "Should have at least 8 families (bert, deepseek, gemma, llama, mistral, phi, qwen2, whisper), got {}",
            registry.len()
        );
    }

    #[test]
    fn pmat_250_default_registry_detects_qwen2() {
        let registry = build_default_registry();
        let detected = registry.detect_from_model_type("qwen2");
        assert!(detected.is_some(), "Should detect qwen2 family");
        let family = detected.expect("qwen2 detected");
        assert_eq!(family.family_name(), "qwen2");
    }

    #[test]
    fn pmat_250_default_registry_detects_llama() {
        let registry = build_default_registry();
        let detected = registry.detect_from_model_type("llama");
        assert!(detected.is_some(), "Should detect llama family");
    }

    #[test]
    fn pmat_250_generated_constants_correct() {
        assert_eq!(QWEN2_VENDOR, "Alibaba");
        assert_eq!(LLAMA_VENDOR, "Meta");
        assert_eq!(BERT_VENDOR, "Google");
        assert_eq!(WHISPER_VENDOR, "OpenAI");
        assert_eq!(MISTRAL_VENDOR, "Mistral AI");
        assert_eq!(PHI_VENDOR, "Microsoft");
        assert_eq!(GEMMA_VENDOR, "Google");
        assert_eq!(DEEPSEEK_VENDOR, "DeepSeek");

        // Verify some well-known size constants
        assert_eq!(QWEN2_0_5B_HIDDEN_DIM, 896);
        assert_eq!(QWEN2_0_5B_NUM_LAYERS, 24);
        assert_eq!(LLAMA_8B_HIDDEN_DIM, 4096);
        assert_eq!(LLAMA_8B_NUM_LAYERS, 32);
        assert_eq!(BERT_BASE_HIDDEN_DIM, 768);
        assert_eq!(WHISPER_TINY_HIDDEN_DIM, 384);
        assert_eq!(MISTRAL_7B_HIDDEN_DIM, 4096);
        assert_eq!(PHI_3_8B_HIDDEN_DIM, 3072);
        assert_eq!(GEMMA_2B_HIDDEN_DIM, 2048);
        assert_eq!(DEEPSEEK_7B_HIDDEN_DIM, 4096);
    }

    #[test]
    fn pmat_250_default_registry_size_detection() {
        let registry = build_default_registry();
        let qwen2 = registry
            .detect_from_model_type("qwen2")
            .expect("qwen2 detected");

        // Detect 0.5B by hidden_dim + num_layers
        let size = qwen2.detect_size(896, 24);
        assert_eq!(size.as_deref(), Some("0.5b"));

        // Detect 7B
        let size = qwen2.detect_size(3584, 28);
        assert_eq!(size.as_deref(), Some("7b"));
    }
}
