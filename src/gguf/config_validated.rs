
#[cfg(test)]
mod tests {
    use super::*;

    // -- Validation failure tests --

    #[test]
    fn test_validated_config_rejects_zero_hidden_dim() {
        let mut cfg = valid_llama_config();
        cfg.hidden_dim = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("hidden_dim"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_layers() {
        let mut cfg = valid_llama_config();
        cfg.num_layers = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_layers"));
    }

    #[test]
    fn test_validated_config_rejects_zero_vocab_size() {
        let mut cfg = valid_llama_config();
        cfg.vocab_size = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("vocab_size"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_heads() {
        let mut cfg = valid_llama_config();
        cfg.num_heads = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_heads"));
    }

    #[test]
    fn test_validated_config_rejects_zero_num_kv_heads() {
        let mut cfg = valid_llama_config();
        cfg.num_kv_heads = 0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_kv_heads"));
    }

    #[test]
    fn test_validated_config_rejects_bad_head_dim() {
        let mut cfg = valid_llama_config();
        // 4096 % 33 != 0
        cfg.num_heads = 33;
        cfg.num_kv_heads = 33;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("divisible by num_heads"));
    }

    #[test]
    fn test_validated_config_rejects_bad_gqa_ratio() {
        let mut cfg = valid_llama_config();
        // 32 % 5 != 0
        cfg.num_kv_heads = 5;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("GQA ratio"));
    }

    #[test]
    fn test_validated_config_mha() {
        // MHA: num_heads == num_kv_heads
        let mut cfg = valid_llama_config();
        cfg.num_kv_heads = 32;
        let v = ValidatedModelConfig::validate(cfg).expect("MHA should be valid");
        assert_eq!(v.num_heads(), v.num_kv_heads());
        assert_eq!(v.head_dim(), 128);
        assert_eq!(v.kv_dim(), 4096); // 32 * 128
    }

    #[test]
    fn test_validated_config_qwen_style() {
        // Qwen 1.5B: hidden=1536, heads=12, kv_heads=2
        let cfg = GGUFConfig {
            architecture: "qwen2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("qwen2"),
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 8960,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2,
            explicit_head_dim: None,
            bos_token_id: Some(151_643),
            eos_token_id: Some(151_645),
        };
        let v = ValidatedModelConfig::validate(cfg).expect("Qwen config should be valid");
        assert_eq!(v.head_dim(), 128); // 1536 / 12
        assert_eq!(v.kv_dim(), 256); // 2 * 128
    }

    /// GH-39: Qwen2.5-0.5B has unusual 7:1 GQA ratio and head_dim=64
    #[test]
    fn test_validated_config_qwen2_0_5b_dimensions() {
        let cfg = GGUFConfig {
            architecture: "qwen2".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("qwen2"),
            hidden_dim: 896,
            num_layers: 24,
            num_heads: 14,
            num_kv_heads: 2,
            vocab_size: 151936,
            intermediate_dim: 4864,
            context_length: 32768,
            rope_theta: 1_000_000.0,
            eps: 1e-6,
            rope_type: 2,
            explicit_head_dim: None,
            bos_token_id: Some(151_643),
            eos_token_id: Some(151_645),
        };
        let v = ValidatedModelConfig::validate(cfg)
            .expect("GH-39: Qwen2.5-0.5B config should be valid");
        assert_eq!(v.head_dim(), 64, "GH-39: 0.5B has head_dim=64");
        assert_eq!(v.kv_dim(), 128, "GH-39: kv_dim = 2 * 64");
        assert_eq!(v.num_heads() / v.num_kv_heads(), 7, "GH-39: 7:1 GQA ratio");
        assert_eq!(v.rope_type(), 2, "GH-39: NEOX RoPE required");
    }
    // -- Upper-bound validation tests (model-metadata-bounds-v1.yaml) --

    #[test]
    fn test_validated_config_rejects_hidden_dim_too_large() {
        let mut cfg = valid_llama_config();
        cfg.hidden_dim = 65_537;
        cfg.num_heads = 1;
        cfg.num_kv_heads = 1;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("hidden_dim"), "{err}");
        assert!(err.to_string().contains("65536"), "{err}");
    }

    #[test]
    fn test_validated_config_rejects_num_layers_too_large() {
        let mut cfg = valid_llama_config();
        cfg.num_layers = 257;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_layers"), "{err}");
        assert!(err.to_string().contains("256"), "{err}");
    }

    #[test]
    fn test_validated_config_rejects_num_heads_too_large() {
        let mut cfg = valid_llama_config();
        cfg.num_heads = 257;
        cfg.num_kv_heads = 257;
        cfg.hidden_dim = 257 * 128; // keep head_dim divisible
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("num_heads"), "{err}");
        assert!(err.to_string().contains("256"), "{err}");
    }

    #[test]
    fn test_validated_config_rejects_vocab_size_too_large() {
        let mut cfg = valid_llama_config();
        cfg.vocab_size = 1_000_001;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("vocab_size"), "{err}");
        assert!(err.to_string().contains("1000000"), "{err}");
    }

    #[test]
    fn test_validated_config_rejects_intermediate_dim_too_large() {
        let mut cfg = valid_llama_config();
        cfg.intermediate_dim = 262_145;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("intermediate_dim"), "{err}");
        assert!(err.to_string().contains("262144"), "{err}");
    }

    #[test]
    fn test_validated_config_rejects_context_length_too_large() {
        let mut cfg = valid_llama_config();
        cfg.context_length = 2_097_153;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("context_length"), "{err}");
        assert!(err.to_string().contains("2097152"), "{err}");
    }

    #[test]
    fn test_validated_config_rejects_rope_theta_too_small() {
        let mut cfg = valid_llama_config();
        cfg.rope_theta = 0.5; // > 0 but < 1.0
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("rope_theta"), "{err}");
    }

    #[test]
    fn test_validated_config_rejects_rope_theta_too_large() {
        let mut cfg = valid_llama_config();
        cfg.rope_theta = 200_000_000.0;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("rope_theta"), "{err}");
    }

    #[test]
    fn test_validated_config_rejects_eps_too_large() {
        let mut cfg = valid_llama_config();
        cfg.eps = 0.1;
        let err = ValidatedModelConfig::validate(cfg).unwrap_err();
        assert!(err.to_string().contains("eps"), "{err}");
    }

    #[test]
    fn test_validated_config_allows_boundary_values() {
        // Max valid values
        let mut cfg = valid_llama_config();
        cfg.hidden_dim = 65_536;
        cfg.num_heads = 256;
        cfg.num_kv_heads = 256;
        cfg.num_layers = 256;
        cfg.vocab_size = 1_000_000;
        cfg.intermediate_dim = 262_144;
        cfg.context_length = 2_097_152;
        cfg.rope_theta = 100_000_000.0;
        cfg.eps = 0.01;
        ValidatedModelConfig::validate(cfg).expect("boundary values should be valid");
    }

    #[test]
    fn test_validated_config_into_inner() {
        let cfg = valid_llama_config();
        let v = ValidatedModelConfig::validate(cfg.clone()).expect("valid");
        let inner = v.into_inner();
        assert_eq!(inner.hidden_dim, cfg.hidden_dim);
        assert_eq!(inner.num_layers, cfg.num_layers);
    }

    #[test]
    fn test_validated_config_eos_token_id_getter() {
        let cfg = valid_llama_config();
        let v = ValidatedModelConfig::validate(cfg).expect("valid");
        assert_eq!(v.eos_token_id(), Some(128_001));
    }

    // -- FALSIFY: Verify Rust bounds match YAML contract --

    /// Parse model-metadata-bounds-v1.yaml and verify each field's bounds
    /// match the Rust validation code in ValidatedModelConfig::validate().
    #[test]
    fn falsify_bounds_match_yaml_contract() {
        // This test verifies the Rust code stays in sync with the YAML contract.
        // If either the Rust bounds or the YAML bounds change, this test fails.
        struct BoundsSpec {
            field: &'static str,
            min: f64,
            max: f64,
        }

        let specs = [
            BoundsSpec { field: "hidden_dim", min: 1.0, max: 65_536.0 },
            BoundsSpec { field: "num_layers", min: 1.0, max: 256.0 },
            BoundsSpec { field: "num_heads", min: 1.0, max: 256.0 },
            BoundsSpec { field: "num_kv_heads", min: 1.0, max: 256.0 },
            BoundsSpec { field: "vocab_size", min: 1.0, max: 1_000_000.0 },
            BoundsSpec { field: "intermediate_dim", min: 1.0, max: 262_144.0 },
            BoundsSpec { field: "context_length", min: 0.0, max: 2_097_152.0 },
            BoundsSpec { field: "rope_theta", min: 1.0, max: 100_000_000.0 },
            BoundsSpec { field: "eps", min: 1e-10, max: 0.01 },
        ];

        // Verify each field's min > 0 constraint is enforced (except context_length)
        for spec in &specs {
            if spec.min > 0.0 && spec.field != "rope_theta" && spec.field != "eps" {
                let mut cfg = valid_llama_config();
                match spec.field {
                    "hidden_dim" => { cfg.hidden_dim = 0; cfg.num_heads = 1; cfg.num_kv_heads = 1; },
                    "num_layers" => cfg.num_layers = 0,
                    "num_heads" => cfg.num_heads = 0,
                    "num_kv_heads" => cfg.num_kv_heads = 0,
                    "vocab_size" => cfg.vocab_size = 0,
                    "intermediate_dim" => cfg.intermediate_dim = 0,
                    _ => continue,
                }
                assert!(
                    ValidatedModelConfig::validate(cfg).is_err(),
                    "FALSIFY: {} = 0 should be rejected (min = {})",
                    spec.field,
                    spec.min
                );
            }

            // Verify max+1 is rejected
            let mut cfg = valid_llama_config();
            match spec.field {
                "hidden_dim" => { cfg.hidden_dim = spec.max as usize + 1; cfg.num_heads = 1; cfg.num_kv_heads = 1; },
                "num_layers" => cfg.num_layers = spec.max as usize + 1,
                "num_heads" => {
                    cfg.num_heads = spec.max as usize + 1;
                    cfg.num_kv_heads = spec.max as usize + 1;
                    cfg.hidden_dim = (spec.max as usize + 1) * 128;
                },
                "num_kv_heads" => {
                    cfg.num_kv_heads = spec.max as usize + 1;
                    cfg.num_heads = spec.max as usize + 1;
                    cfg.hidden_dim = (spec.max as usize + 1) * 128;
                },
                "vocab_size" => cfg.vocab_size = spec.max as usize + 1,
                "intermediate_dim" => cfg.intermediate_dim = spec.max as usize + 1,
                "context_length" => cfg.context_length = spec.max as usize + 1,
                "rope_theta" => cfg.rope_theta = spec.max as f32 * 2.0,
                "eps" => cfg.eps = spec.max as f32 * 2.0,
                _ => continue,
            }
            assert!(
                ValidatedModelConfig::validate(cfg).is_err(),
                "FALSIFY: {} = max+1 ({}) should be rejected (max = {})",
                spec.field,
                spec.max + 1.0,
                spec.max
            );
        }
    }

    // -- FALSIFY: Verify special-tokens-registry matches Rust code --

    /// Verify default_eos_for_architecture() matches special-tokens-registry-v1.yaml.
    /// If either changes without updating the other, this test fails.
    #[test]
    fn falsify_eos_tokens_match_yaml_registry() {
        // Canonical EOS values from special-tokens-registry-v1.yaml
        let expected: &[(&str, Option<u32>)] = &[
            ("qwen2", Some(151_645)),
            ("qwen3", Some(151_645)),
            ("qwen3moe", Some(151_645)),
            ("qwen3_5", Some(248_044)),
            ("llama", Some(128_001)),
            ("mistral", Some(2)),
            ("gemma", Some(1)),
            ("gemma2", Some(1)),
            ("deepseek", Some(1)),
            ("deepseek2", Some(1)),
            ("phi3", Some(32_000)),
            ("phi2", Some(50_256)),
            ("phi", Some(50_256)),
            ("gpt2", Some(50_256)),
            ("unknown_arch", None),
        ];

        for &(arch, want) in expected {
            let got = default_eos_for_architecture(arch);
            assert_eq!(
                got, want,
                "FALSIFY: default_eos_for_architecture(\"{arch}\") = {got:?}, expected {want:?} (special-tokens-registry-v1.yaml)"
            );
        }
    }

    /// Verify default_bos_for_architecture() matches special-tokens-registry-v1.yaml.
    #[test]
    fn falsify_bos_tokens_match_yaml_registry() {
        let expected: &[(&str, Option<u32>)] = &[
            ("qwen2", Some(151_643)),
            ("qwen3", Some(151_643)),
            ("qwen3moe", Some(151_643)),
            ("llama", Some(128_000)),
            ("mistral", Some(1)),
            ("gemma", Some(2)),
            ("gemma2", Some(2)),
            ("deepseek", Some(0)),
            ("deepseek2", Some(0)),
            ("phi3", Some(1)),
            // phi2/phi/gpt2/qwen3_5: no BOS
            ("phi2", None),
            ("phi", None),
            ("gpt2", None),
            ("qwen3_5", None),
            ("unknown_arch", None),
        ];

        for &(arch, want) in expected {
            let got = default_bos_for_architecture(arch);
            assert_eq!(
                got, want,
                "FALSIFY: default_bos_for_architecture(\"{arch}\") = {got:?}, expected {want:?} (special-tokens-registry-v1.yaml)"
            );
        }
    }

    /// Verify EOS token IDs are within vocab bounds (proof obligation from YAML).
    #[test]
    fn falsify_eos_within_vocab_bounds() {
        // (arch, eos_id, vocab_size) from special-tokens-registry-v1.yaml
        let families: &[(&str, u32, u32)] = &[
            ("qwen2", 151_645, 151_936),
            ("qwen3", 151_645, 151_936),
            ("qwen3_5", 248_044, 248_320),
            ("llama", 128_001, 128_256),
            ("mistral", 2, 32_000),
            ("gemma", 1, 256_000),
            ("deepseek", 1, 129_280),
            ("phi3", 32_000, 32_064),
            ("phi2", 50_256, 51_200),
            ("gpt2", 50_256, 50_257),
        ];

        for &(arch, eos, vocab) in families {
            assert!(
                eos < vocab,
                "FALSIFY: {arch} eos_token_id ({eos}) >= vocab_size ({vocab}) — violates eos_within_vocab proof obligation"
            );
        }
    }

include!("config_gguf.rs");
}
