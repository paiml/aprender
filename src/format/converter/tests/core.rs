//! APR Converter Core Tests - Extreme TDD
//! PMAT-197: Split from tests.rs for file size reduction
//!
//! Contains: source parsing, name mapping, tensor expectations,
//! converter builder, import options, conversion, tensor stats,
//! quantization, convert, and sharded import tests.
//!
//! # Harness Policy (Audit Round 3, Item #1)
//!
//! **All new conversion tests MUST use `ConversionTestHarness`** from
//! `crate::format::test_factory::harness`. Direct `BTreeMap::new()` +
//! manual tensor construction is forbidden for new tests. The harness
//! provides: temp directory management, PygmyConfig-driven model
//! generation, and automatic cleanup. Pre-existing tests (349 legacy)
//! are grandfathered but should be migrated when touched.
//!
//! ```rust,ignore
//! // CORRECT: Use harness
//! ConversionTestHarness::assert_import_ok(PygmyConfig::default());
//!
//! // WRONG: Manual tensor construction (legacy only)
//! let mut tensors = BTreeMap::new();
//! tensors.insert("weight".into(), (vec![0.0; 64], vec![8, 8]));
//! ```

#[allow(unused_imports)]
use super::super::*;

#[cfg(test)]
mod tests_source_parsing {
    use super::*;

    #[test]
    fn test_parse_hf_org_repo() {
        let source = Source::parse("hf://openai/whisper-tiny").unwrap();
        assert_eq!(
            source,
            Source::HuggingFace {
                org: "openai".to_string(),
                repo: "whisper-tiny".to_string(),
                file: None,
            }
        );
    }

    #[test]
    fn test_parse_hf_org_repo_file() {
        let source = Source::parse("hf://openai/whisper-tiny/model.safetensors").unwrap();
        assert_eq!(
            source,
            Source::HuggingFace {
                org: "openai".to_string(),
                repo: "whisper-tiny".to_string(),
                file: Some("model.safetensors".to_string()),
            }
        );
    }

    #[test]
    fn test_parse_hf_nested_file() {
        let source =
            Source::parse("hf://meta-llama/Llama-2-7b/pytorch_model-00001-of-00002.bin").unwrap();
        assert_eq!(
            source,
            Source::HuggingFace {
                org: "meta-llama".to_string(),
                repo: "Llama-2-7b".to_string(),
                file: Some("pytorch_model-00001-of-00002.bin".to_string()),
            }
        );
    }

    #[test]
    fn test_parse_local_path() {
        let source = Source::parse("./models/model.safetensors").unwrap();
        assert_eq!(
            source,
            Source::Local(PathBuf::from("./models/model.safetensors"))
        );
    }

    #[test]
    fn test_parse_url() {
        let source = Source::parse("https://example.com/model.safetensors").unwrap();
        assert_eq!(
            source,
            Source::Url("https://example.com/model.safetensors".to_string())
        );
    }

    #[test]
    fn test_parse_hf_invalid() {
        let result = Source::parse("hf://invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_default_file() {
        let hf = Source::HuggingFace {
            org: "openai".to_string(),
            repo: "whisper".to_string(),
            file: None,
        };
        assert_eq!(hf.default_file(), "model.safetensors");

        let hf_with_file = Source::HuggingFace {
            org: "openai".to_string(),
            repo: "whisper".to_string(),
            file: Some("custom.safetensors".to_string()),
        };
        assert_eq!(hf_with_file.default_file(), "custom.safetensors");
    }
}

#[cfg(test)]
mod tests_name_mapping {
    use super::*;

    #[test]
    fn test_whisper_preserve_model_prefix() {
        // PMAT-099: Names are now preserved for AprTransformer compatibility
        let mapped = Architecture::Whisper.map_name("model.encoder.conv1.weight");
        assert_eq!(mapped, "model.encoder.conv1.weight");
    }

    #[test]
    fn test_whisper_no_prefix() {
        let mapped = Architecture::Whisper.map_name("encoder.conv1.weight");
        assert_eq!(mapped, "encoder.conv1.weight");
    }

    #[test]
    fn test_whisper_decoder_layer_norm() {
        // PMAT-099: Names are now preserved for AprTransformer compatibility
        let mapped = Architecture::Whisper.map_name("model.decoder.layer_norm.weight");
        assert_eq!(mapped, "model.decoder.layer_norm.weight");
    }

    #[test]
    fn test_auto_preserves_model_prefix() {
        // PMAT-099: model. prefix preserved for AprTransformer::from_apr_bytes compatibility
        let mapped = Architecture::Auto.map_name("model.encoder.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "model.encoder.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_llama_mapping() {
        // PMAT-099: Preserve original names for inference compatibility
        let mapped = Architecture::Llama.map_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "model.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_bert_mapping() {
        // PMAT-099: Preserve original names
        let mapped =
            Architecture::Bert.map_name("bert.encoder.layer.0.attention.self.query.weight");
        assert_eq!(mapped, "bert.encoder.layer.0.attention.self.query.weight");
    }

    #[test]
    fn test_qwen2_mapping() {
        // PMAT-099: Preserve model. prefix for AprTransformer compatibility
        let mapped = Architecture::Qwen2.map_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "model.layers.0.self_attn.q_proj.weight");
    }
}

#[cfg(test)]
mod tests_tensor_expectations {
    use super::*;

    #[test]
    fn test_layer_norm_weight_expectation() {
        let exp = TensorExpectation::for_tensor("encoder.layer_norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.mean_range, (0.5, 3.0));
    }

    #[test]
    fn test_layer_norm_bias_expectation() {
        let exp = TensorExpectation::for_tensor("decoder.layers.0.self_attn_layer_norm.bias");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.mean_range, (-0.5, 0.5));
    }

    #[test]
    fn test_linear_weight_expectation() {
        let exp = TensorExpectation::for_tensor("encoder.layers.0.fc1.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.mean_range, (-0.1, 0.1));
    }

    #[test]
    fn test_embedding_expectation() {
        let exp = TensorExpectation::for_tensor("decoder.embed_tokens.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_check_layer_norm_valid() {
        let stats = TensorStats {
            name: "encoder.layer_norm.weight".to_string(),
            count: 384,
            min: 0.5,
            max: 2.0,
            mean: 1.0,
            std: 0.3,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };

        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        assert!(exp.check(&stats).is_ok());
    }

    #[test]
    fn test_check_layer_norm_invalid_mean() {
        let stats = TensorStats {
            name: "decoder.layer_norm.weight".to_string(),
            count: 384,
            min: 5.0,
            max: 15.0,
            mean: 11.0, // BUG: should be ~1.0
            std: 2.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };

        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let result = exp.check(&stats);
        assert!(result.is_err());

        let err = result.unwrap_err().to_string();
        assert!(err.contains("mean=11"));
        assert!(err.contains("outside expected range"));
    }

    #[test]
    fn test_rmsnorm_weight_detection() {
        // LLaMA-style input_layernorm
        let exp = TensorExpectation::for_tensor("model.layers.0.input_layernorm.weight");
        assert!(exp.is_some());
        // Issue #46: Widened ranges for Qwen2.5 compatibility (mean=7.23, std=2.11)
        assert_eq!(exp.unwrap().mean_range, (-1.0, 10.0));

        // LLaMA-style post_attention_layernorm
        let exp = TensorExpectation::for_tensor("model.layers.5.post_attention_layernorm.weight");
        assert!(exp.is_some());
        assert_eq!(exp.unwrap().mean_range, (-1.0, 10.0));

        // Final norm
        let exp = TensorExpectation::for_tensor("model.norm.weight");
        assert!(exp.is_some());
        assert_eq!(exp.unwrap().mean_range, (-1.0, 10.0));
    }

    #[test]
    fn test_rmsnorm_accepts_trained_weights() {
        // TinyLlama trained model has means from 0.005 to 0.5
        let stats = TensorStats {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            count: 2048,
            min: -0.2,
            max: 0.8,
            mean: 0.05, // Trained weight, NOT near 1.0
            std: 0.15,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };

        let exp = TensorExpectation::RMSNORM_WEIGHT;
        assert!(exp.check(&stats).is_ok());
    }
}

#[cfg(test)]
mod tests_converter_builder {
    use super::*;

    #[test]
    fn test_converter_builder_chain() {
        let converter = AprConverter::new()
            .source("hf://openai/whisper-tiny")
            .unwrap()
            .architecture(Architecture::Whisper)
            .validate(ValidationConfig::Strict)
            .quantize(QuantizationType::Int8)
            .compress(Compression::Lz4);

        assert_eq!(converter.architecture, Architecture::Whisper);
        assert_eq!(converter.validation, ValidationConfig::Strict);
        assert_eq!(converter.quantize, Some(QuantizationType::Int8));
        assert_eq!(converter.compress, Some(Compression::Lz4));
    }

    #[test]
    fn test_converter_no_source_error() {
        let converter = AprConverter::new();
        let result = converter.convert();
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod tests_import_options {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = ImportOptions::default();
        assert_eq!(opts.architecture, Architecture::Auto);
        assert_eq!(opts.validation, ValidationConfig::Strict);
        assert_eq!(opts.quantize, None);
        assert_eq!(opts.compress, None);
        assert!(!opts.strict);
        assert!(opts.cache);
    }
}

include!("core_conversion.rs");
include!("core_convert.rs");
include!("core_rosetta_gqa.rs");
include!("core_q4k_q6k_roundtrip.rs");
