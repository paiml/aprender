
    // ========================================================================
    // PMAT-546: Architecture ↔ model-family YAML parity tests
    // Contract: contracts/model-family-parity-v1.yaml
    // ========================================================================

    /// All non-Auto Architecture variants and their expected YAML family key.
    /// This table is the single source of truth for the parity mapping.
    const ARCH_YAML_MAP: &[(Architecture, &str)] = &[
        (Architecture::Whisper, "whisper"),
        (Architecture::Llama, "llama"),
        (Architecture::Bert, "bert"),
        (Architecture::Qwen2, "qwen2"),
        (Architecture::Qwen3, "qwen3"),
        (Architecture::Qwen3_5, "qwen3_5"),
        (Architecture::Gpt2, "gpt2"),
        (Architecture::Phi, "phi"),
        (Architecture::GptNeoX, "gptneox"),
        (Architecture::Opt, "opt"),
        (Architecture::DeepSeek, "deepseek"),
        (Architecture::Gemma, "gemma"),
        (Architecture::Mistral, "mistral"),
        (Architecture::FalconH1, "falcon_h1"),
        (Architecture::Mamba, "mamba"),
        (Architecture::Moonshine, "moonshine"),
        (Architecture::OpenElm, "openelm"),
        (Architecture::Rwkv7, "rwkv7"),
    ];

    /// FALSIFY-PARITY-001: Every non-Auto Architecture variant has a model-family YAML.
    #[test]
    fn test_every_architecture_has_model_family_yaml() {
        let model_families_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates dir")
            .parent()
            .expect("workspace root")
            .join("contracts/model-families");

        let mut missing = Vec::new();
        for (arch, yaml_key) in ARCH_YAML_MAP {
            let yaml_path = model_families_dir.join(format!("{yaml_key}.yaml"));
            if !yaml_path.exists() {
                missing.push(format!(
                    "Architecture::{arch:?} → contracts/model-families/{yaml_key}.yaml (NOT FOUND)"
                ));
            }
        }
        assert!(
            missing.is_empty(),
            "FALSIFY-PARITY-001 FAIL: Architecture variants missing YAML contracts:\n  {}",
            missing.join("\n  ")
        );
    }

    /// FALSIFY-PARITY-002: Every model-family YAML is recognized by from_model_type().
    #[test]
    fn test_every_model_family_yaml_has_architecture() {
        let model_families_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates dir")
            .parent()
            .expect("workspace root")
            .join("contracts/model-families");

        let mut unrecognized = Vec::new();
        for entry in std::fs::read_dir(&model_families_dir).expect("read model-families dir") {
            let entry = entry.expect("dir entry");
            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();

            // Skip schema files
            if filename_str.starts_with('_') || !filename_str.ends_with(".yaml") {
                continue;
            }

            let family_key = filename_str.trim_end_matches(".yaml");

            // Read the YAML to get the family field
            let contents =
                std::fs::read_to_string(entry.path()).expect("read YAML");
            let family_field = contents
                .lines()
                .find(|l| l.starts_with("family:"))
                .map(|l| l.trim_start_matches("family:").trim().to_string())
                .unwrap_or_else(|| family_key.to_string());

            if Architecture::from_model_type(&family_field).is_none() {
                unrecognized.push(format!(
                    "{filename_str} (family: {family_field}) → from_model_type() returns None"
                ));
            }
        }
        assert!(
            unrecognized.is_empty(),
            "FALSIFY-PARITY-002 FAIL: Model-family YAMLs not recognized by from_model_type():\n  {}",
            unrecognized.join("\n  ")
        );
    }

    /// FALSIFY-PARITY-003: from_model_type() returns correct variant for all new architectures.
    #[test]
    fn test_from_model_type_new_variants() {
        // FalconH1
        assert_eq!(Architecture::from_model_type("falcon_h1"), Some(Architecture::FalconH1));
        assert_eq!(Architecture::from_model_type("falcon-h1"), Some(Architecture::FalconH1));
        assert_eq!(Architecture::from_model_type("falcon3"), Some(Architecture::FalconH1));

        // Mamba
        assert_eq!(Architecture::from_model_type("mamba"), Some(Architecture::Mamba));
        assert_eq!(Architecture::from_model_type("mamba2"), Some(Architecture::Mamba));

        // Moonshine
        assert_eq!(Architecture::from_model_type("moonshine"), Some(Architecture::Moonshine));

        // OpenELM
        assert_eq!(Architecture::from_model_type("openelm"), Some(Architecture::OpenElm));

        // RWKV-7
        assert_eq!(Architecture::from_model_type("rwkv"), Some(Architecture::Rwkv7));
        assert_eq!(Architecture::from_model_type("rwkv7"), Some(Architecture::Rwkv7));
        assert_eq!(Architecture::from_model_type("rwkv-7"), Some(Architecture::Rwkv7));
    }

    /// FALSIFY-PARITY-004: display_name() returns a non-empty human-readable string for all variants.
    #[test]
    fn test_display_name_all_variants() {
        let all_variants = [
            Architecture::Auto,
            Architecture::Whisper,
            Architecture::Llama,
            Architecture::Bert,
            Architecture::Qwen2,
            Architecture::Qwen3,
            Architecture::Qwen3_5,
            Architecture::Gpt2,
            Architecture::Phi,
            Architecture::GptNeoX,
            Architecture::Opt,
            Architecture::DeepSeek,
            Architecture::Gemma,
            Architecture::Mistral,
            Architecture::FalconH1,
            Architecture::Mamba,
            Architecture::Moonshine,
            Architecture::OpenElm,
            Architecture::Rwkv7,
        ];

        for variant in &all_variants {
            let name = variant.display_name();
            assert!(
                !name.is_empty(),
                "FALSIFY-PARITY-004 FAIL: Architecture::{variant:?} has empty display_name()"
            );
        }

        // Verify specific display names for new variants
        assert_eq!(Architecture::FalconH1.display_name(), "Falcon-H1");
        assert_eq!(Architecture::Mamba.display_name(), "Mamba");
        assert_eq!(Architecture::Moonshine.display_name(), "Moonshine");
        assert_eq!(Architecture::OpenElm.display_name(), "OpenELM");
        assert_eq!(Architecture::Rwkv7.display_name(), "RWKV-7");
    }

    /// FALSIFY-PARITY-005: is_llm() classification matches expected categories.
    /// Audio models and encoder-only models return false; decoder-only LLMs return true.
    #[test]
    fn test_is_llm_matches_contract() {
        // Non-LLM architectures (audio, encoder-only)
        assert!(!Architecture::Auto.is_llm(), "Auto should not be classified as LLM");
        assert!(!Architecture::Whisper.is_llm(), "Whisper (audio) should not be LLM");
        assert!(!Architecture::Bert.is_llm(), "BERT (encoder-only) should not be LLM");
        assert!(!Architecture::Moonshine.is_llm(), "Moonshine (audio) should not be LLM");

        // LLM architectures (decoder-only text generation)
        assert!(Architecture::Llama.is_llm(), "LLaMA should be LLM");
        assert!(Architecture::Qwen2.is_llm(), "Qwen2 should be LLM");
        assert!(Architecture::Qwen3.is_llm(), "Qwen3 should be LLM");
        assert!(Architecture::Qwen3_5.is_llm(), "Qwen3.5 should be LLM");
        assert!(Architecture::Gpt2.is_llm(), "GPT-2 should be LLM");
        assert!(Architecture::Phi.is_llm(), "Phi should be LLM");
        assert!(Architecture::GptNeoX.is_llm(), "GPT-NeoX should be LLM");
        assert!(Architecture::Opt.is_llm(), "OPT should be LLM");
        assert!(Architecture::DeepSeek.is_llm(), "DeepSeek should be LLM");
        assert!(Architecture::Gemma.is_llm(), "Gemma should be LLM");
        assert!(Architecture::Mistral.is_llm(), "Mistral should be LLM");
        assert!(Architecture::FalconH1.is_llm(), "Falcon-H1 should be LLM");
        assert!(Architecture::Mamba.is_llm(), "Mamba (causal LM) should be LLM");
        assert!(Architecture::OpenElm.is_llm(), "OpenELM should be LLM");
        assert!(Architecture::Rwkv7.is_llm(), "RWKV-7 (causal LM) should be LLM");
    }

    /// FALSIFY-PARITY-006: map_name() handles all variants without panic.
    #[test]
    fn test_map_name_all_variants_no_panic() {
        let test_name = "model.layers.0.self_attn.q_proj.weight";
        let all_variants = [
            Architecture::Auto,
            Architecture::Whisper,
            Architecture::Llama,
            Architecture::Bert,
            Architecture::Qwen2,
            Architecture::Qwen3,
            Architecture::Qwen3_5,
            Architecture::Gpt2,
            Architecture::Phi,
            Architecture::GptNeoX,
            Architecture::Opt,
            Architecture::DeepSeek,
            Architecture::Gemma,
            Architecture::Mistral,
            Architecture::FalconH1,
            Architecture::Mamba,
            Architecture::Moonshine,
            Architecture::OpenElm,
            Architecture::Rwkv7,
        ];

        for variant in &all_variants {
            let mapped = variant.map_name(test_name);
            assert!(
                !mapped.is_empty(),
                "Architecture::{variant:?}.map_name() returned empty string"
            );
        }
    }
