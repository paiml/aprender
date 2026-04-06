
    #[test]
    fn test_gguf_config_creation() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.architecture, "llama");
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.intermediate_dim, 11008);
        assert_eq!(config.context_length, 4096);
        assert!((config.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!((config.eps - 1e-5).abs() < f32::EPSILON);
        assert_eq!(config.rope_type, 0);
    }

    #[test]
    fn test_gguf_config_clone() {
        let config = GGUFConfig {
            architecture: "qwen2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("qwen2"),
            hidden_dim: 2048,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 5632,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let cloned = config.clone();
        assert_eq!(cloned.architecture, "qwen2");
        assert_eq!(cloned.hidden_dim, config.hidden_dim);
        assert_eq!(cloned.rope_type, 2);
    }

    #[test]
    fn test_gguf_config_debug() {
        let config = GGUFConfig {
            architecture: "phi2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
            hidden_dim: 2560,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            vocab_size: 51200,
            intermediate_dim: 10240,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("phi2"));
        assert!(debug_str.contains("2560"));
    }

    #[test]
    fn test_gguf_config_gqa_ratio() {
        // Test GQA config (4 Q heads per KV head)
        let config = GGUFConfig {
            architecture: "llama3".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama3"),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // 4:1 GQA ratio
            vocab_size: 128256,
            intermediate_dim: 14336,
            context_length: 8192,
            rope_theta: 500000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.num_heads / config.num_kv_heads, 4);
    }

    #[test]
    fn test_gguf_config_mha() {
        // Test MHA config (num_heads == num_kv_heads)
        let config = GGUFConfig {
            architecture: "phi2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("phi2"),
            hidden_dim: 2560,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32, // MHA
            vocab_size: 51200,
            intermediate_dim: 10240,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.num_heads, config.num_kv_heads);
    }

    #[test]
    fn test_gguf_config_neox_rope() {
        // Test NEOX-style RoPE (type 2)
        let config = GGUFConfig {
            architecture: "gpt-neox".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("gpt-neox"),
            hidden_dim: 6144,
            num_layers: 44,
            num_heads: 64,
            num_kv_heads: 64,
            vocab_size: 50432,
            intermediate_dim: 24576,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 2, // NEOX
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert_eq!(config.rope_type, 2);
    }

    #[test]
    fn test_gguf_config_high_rope_theta() {
        // Test Qwen-style high rope_theta
        let config = GGUFConfig {
            architecture: "qwen".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("qwen"),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 8960,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert!(config.rope_theta > 100_000.0);
        assert!(config.context_length > 8192);
    }

    #[test]
    fn test_gguf_config_small_epsilon() {
        // Test different epsilon values
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test"),
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 50257,
            intermediate_dim: 2048,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-6, // Smaller epsilon
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        assert!((config.eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_gguf_config_head_dim() {
        // Verify head_dim calculation
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let head_dim = config.hidden_dim / config.num_heads;
        assert_eq!(head_dim, 128);
    }

    #[test]
    fn test_gguf_config_kv_dim() {
        // Verify kv_dim calculation for GQA
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let head_dim = config.hidden_dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;
        assert_eq!(kv_dim, 1024);
    }

    #[test]
    fn test_gguf_config_qkv_out_dim() {
        // Verify QKV output dimension calculation
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let head_dim = config.hidden_dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;
        let qkv_out_dim = config.hidden_dim + 2 * kv_dim;
        assert_eq!(qkv_out_dim, 4096 + 2 * 1024);
    }

    #[test]
    fn test_gguf_config_debug_contains_all_fields() {
        let config = GGUFConfig {
            architecture: "mistral".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("mistral"),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 14336,
            context_length: 8192,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let debug = format!("{:?}", config);
        assert!(debug.contains("architecture"));
        assert!(debug.contains("hidden_dim"));
        assert!(debug.contains("num_layers"));
        assert!(debug.contains("num_heads"));
        assert!(debug.contains("num_kv_heads"));
        assert!(debug.contains("vocab_size"));
        assert!(debug.contains("intermediate_dim"));
        assert!(debug.contains("context_length"));
        assert!(debug.contains("rope_theta"));
        assert!(debug.contains("eps"));
        assert!(debug.contains("rope_type"));
    }

    #[test]
    fn test_gguf_config_clone_deep() {
        let original = GGUFConfig {
            architecture: "test_arch".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("test_arch"),
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size: 50257,
            intermediate_dim: 3072,
            context_length: 1024,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let cloned = original.clone();

        // Verify all fields are cloned correctly
        assert_eq!(cloned.architecture, original.architecture);
        assert_eq!(cloned.hidden_dim, original.hidden_dim);
        assert_eq!(cloned.num_layers, original.num_layers);
        assert_eq!(cloned.num_heads, original.num_heads);
        assert_eq!(cloned.num_kv_heads, original.num_kv_heads);
        assert_eq!(cloned.vocab_size, original.vocab_size);
        assert_eq!(cloned.intermediate_dim, original.intermediate_dim);
        assert_eq!(cloned.context_length, original.context_length);
        assert!((cloned.rope_theta - original.rope_theta).abs() < f32::EPSILON);
        assert!((cloned.eps - original.eps).abs() < f32::EPSILON);
        assert_eq!(cloned.rope_type, original.rope_type);
    }

    #[test]
    fn test_default_bos_qwen2() {
        assert_eq!(default_bos_for_architecture("qwen2"), Some(151_643));
    }

    #[test]
    fn test_default_bos_llama() {
        assert_eq!(default_bos_for_architecture("llama"), Some(128_000));
    }

    #[test]
    fn test_default_bos_mistral() {
        assert_eq!(default_bos_for_architecture("mistral"), Some(1));
    }

    #[test]
    fn test_default_bos_gemma() {
        assert_eq!(default_bos_for_architecture("gemma"), Some(2));
        assert_eq!(default_bos_for_architecture("gemma2"), Some(2));
    }

    #[test]
    fn test_default_bos_deepseek() {
        assert_eq!(default_bos_for_architecture("deepseek"), Some(0));
        assert_eq!(default_bos_for_architecture("deepseek2"), Some(0));
    }

    #[test]
    fn test_default_bos_phi3() {
        // Phi-3: bos_token_id=1 per HuggingFace config.json
        assert_eq!(default_bos_for_architecture("phi3"), Some(1));
    }

    #[test]
    fn test_default_bos_unknown_returns_none() {
        // phi/phi2/gpt2: no BOS token
        assert_eq!(default_bos_for_architecture("phi"), None);
        assert_eq!(default_bos_for_architecture("phi2"), None);
        assert_eq!(default_bos_for_architecture("unknown_arch"), None);
        assert_eq!(default_bos_for_architecture(""), None);
    }

    // -----------------------------------------------------------------------
    // ValidatedModelConfig tests
    // -----------------------------------------------------------------------

    /// Helper: build a valid LLaMA-style GGUFConfig for testing.
    fn valid_llama_config() -> GGUFConfig {
        GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 11008,
            context_length: 4096,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: Some(128_000),
            eos_token_id: Some(128_001),
        }
    }

    #[test]
    fn test_validated_config_accepts_valid() {
        let validated = ValidatedModelConfig::validate(valid_llama_config());
        assert!(validated.is_ok());
    }

    #[test]
    fn test_validated_config_getters() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");

        assert_eq!(v.architecture(), "llama");
        assert_eq!(v.hidden_dim(), 4096);
        assert_eq!(v.num_layers(), 32);
        assert_eq!(v.num_heads(), 32);
        assert_eq!(v.num_kv_heads(), 8);
        assert_eq!(v.vocab_size(), 32000);
        assert_eq!(v.intermediate_dim(), 11008);
        assert_eq!(v.context_length(), 4096);
        assert!((v.rope_theta() - 10000.0).abs() < f32::EPSILON);
        assert!((v.eps() - 1e-5).abs() < f32::EPSILON);
        assert_eq!(v.rope_type(), 0);
        assert_eq!(v.bos_token_id(), Some(128_000));
    }

    #[test]
    fn test_validated_config_head_dim() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        // 4096 / 32 = 128
        assert_eq!(v.head_dim(), 128);
    }

    #[test]
    fn test_validated_config_kv_dim() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        // 8 * 128 = 1024
        assert_eq!(v.kv_dim(), 1024);
    }

    #[test]
    fn test_validated_config_deref() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        // Access via Deref — should reach GGUFConfig fields directly
        assert_eq!(v.hidden_dim, 4096);
        assert_eq!(v.num_heads, 32);
    }

    #[test]
    fn test_validated_config_inner_ref() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        let inner: &GGUFConfig = v.config();
        assert_eq!(inner.architecture, "llama");
        assert_eq!(inner.hidden_dim, 4096);
    }

    #[test]
    fn test_validated_config_clone() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        let cloned = v.clone();
        assert_eq!(cloned.hidden_dim(), v.hidden_dim());
        assert_eq!(cloned.architecture(), v.architecture());
    }

    #[test]
    fn test_validated_config_debug() {
        let v =
            ValidatedModelConfig::validate(valid_llama_config()).expect("valid config should pass");
        let debug = format!("{v:?}");
        assert!(debug.contains("ValidatedModelConfig"));
        assert!(debug.contains("llama"));
    }
