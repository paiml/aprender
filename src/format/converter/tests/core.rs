//! APR Converter Core Tests - Extreme TDD
//! PMAT-197: Split from tests.rs for file size reduction
//!
//! Contains: source parsing, name mapping, tensor expectations,
//! converter builder, import options, conversion, tensor stats,
//! quantization, convert, and sharded import tests.

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

#[cfg(test)]
mod tests_conversion {
    use super::*;
    use crate::format::test_factory::harness::ConversionTestHarness;
    use crate::format::test_factory::PygmyConfig;

    fn create_test_safetensors(path: &Path, tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) {
        save_safetensors(path, tensors).expect("Failed to create test SafeTensors file");
    }

    /// Harness-based: pygmy LLaMA model imports to APR with read-back verification.
    #[test]
    fn test_convert_valid_safetensors() {
        let h = ConversionTestHarness::new()
            .with_safetensors(PygmyConfig::llama_style())
            .import_to_apr(ImportOptions::default());

        // Verify the output APR has correct tensor data
        h.verify_apr().assert_passed();
    }

    /// Intentionally bad data: invalid LayerNorm mean=11 must fail strict validation.
    /// Uses Architecture::Llama (verified) with strict=true to bypass the unverified-arch
    /// check and reach the LayerNorm tensor validation path.
    #[test]
    fn test_convert_invalid_layernorm_fails_strict() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("invalid_ln.safetensors");
        let output = dir.path().join("output.apr");

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "decoder.layer_norm.weight".to_string(),
            (vec![11.0f32; 384], vec![384]),
        );
        create_test_safetensors(&input, &tensors);

        let options = ImportOptions {
            architecture: Architecture::Llama,
            validation: ValidationConfig::Strict,
            strict: true,
            ..Default::default()
        };
        let result = apr_import(&input.to_string_lossy(), &output, options);

        assert!(
            result.is_err(),
            "Invalid LayerNorm should fail strict validation"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mean=11") || err.contains("LayerNorm") || err.contains("outside expected range"),
            "Error should mention LayerNorm issue: {err}"
        );
    }

    /// Intentionally bad data: invalid LayerNorm passes in default (permissive) mode.
    #[test]
    fn test_convert_invalid_layernorm_force_succeeds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("force_ln.safetensors");
        let output = dir.path().join("output.apr");

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "decoder.layer_norm.weight".to_string(),
            (vec![11.0f32; 384], vec![384]),
        );
        create_test_safetensors(&input, &tensors);

        let options = ImportOptions {
            validation: ValidationConfig::Strict,
            ..Default::default()
        };
        let result = apr_import(&input.to_string_lossy(), &output, options);

        assert!(
            result.is_ok(),
            "Permissive mode should bypass validation: {:?}",
            result.err()
        );
    }

    /// Intentionally bad data: NaN values must fail validation.
    #[test]
    fn test_convert_nan_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("nan.safetensors");
        let output = dir.path().join("output.apr");

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "test.weight".to_string(),
            (vec![1.0, f32::NAN, 3.0], vec![3]),
        );
        create_test_safetensors(&input, &tensors);

        let result = apr_import(&input.to_string_lossy(), &output, ImportOptions::default());

        assert!(result.is_err(), "NaN should fail validation");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN"), "Error should mention NaN: {err}");
    }

    /// Error path: nonexistent file must produce clear error.
    #[test]
    fn test_convert_nonexistent_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("out.apr");
        let result = apr_import(
            "/tmp/nonexistent_model_abc123.safetensors",
            &output,
            ImportOptions::default(),
        );
        assert!(result.is_err(), "Nonexistent file should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found") || err.contains("No such file"),
            "Error should mention file not found: {err}"
        );
    }

    /// Error path: unsupported format (.gguf stub) must fail.
    #[test]
    fn test_convert_unsupported_format() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("bad.gguf");
        let output = dir.path().join("out.apr");
        fs::write(&input, b"test").expect("Failed to create test file");

        let result = apr_import(&input.to_string_lossy(), &output, ImportOptions::default());
        assert!(result.is_err(), "Unsupported format should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("GGUF") || err.contains("not yet"),
            "Error should mention unsupported: {err}"
        );
    }

    /// Harness-based: Whisper name mapping preserves model.* prefix (PMAT-099).
    #[test]
    fn test_name_mapping_whisper() {
        use crate::format::v2::AprV2Reader;

        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("whisper.safetensors");
        let output = dir.path().join("whisper.apr");

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.encoder.conv1.weight".to_string(),
            (vec![0.01f32; 100], vec![10, 10]),
        );
        tensors.insert(
            "model.decoder.layer_norm.weight".to_string(),
            (vec![1.0f32; 384], vec![384]),
        );
        create_test_safetensors(&input, &tensors);

        let options = ImportOptions {
            architecture: Architecture::Whisper,
            ..Default::default()
        };
        let result = apr_import(&input.to_string_lossy(), &output, options);
        assert!(
            result.is_ok(),
            "Whisper mapping should work in permissive mode: {:?}",
            result.err()
        );

        // Read-back verification: check tensor names preserved
        let data = fs::read(&output).expect("Failed to read output");
        let reader = AprV2Reader::from_bytes(&data).expect("Failed to parse APR");
        let tensor_names = reader.tensor_names();

        assert!(
            tensor_names.contains(&"model.encoder.conv1.weight"),
            "Should preserve 'model.' prefix for AprTransformer compatibility, got: {:?}",
            tensor_names
        );
        assert!(
            tensor_names.contains(&"model.decoder.layer_norm.weight"),
            "Should preserve 'model.' prefix for AprTransformer compatibility, got: {:?}",
            tensor_names
        );
    }
}

#[cfg(test)]
mod tests_tensor_stats {
    use super::*;

    #[test]
    fn test_compute_stats_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 0.001, "Mean should be 3.0");
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_compute_stats_with_nan() {
        let data = vec![1.0f32, f32::NAN, 3.0];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.count, 3);
        // Mean computed from valid values only
        assert!(
            (stats.mean - 2.0).abs() < 0.001,
            "Mean should be 2.0 (from valid values)"
        );
    }

    #[test]
    fn test_compute_stats_with_inf() {
        let data = vec![1.0f32, f32::INFINITY, f32::NEG_INFINITY, 3.0];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.inf_count, 2);
        assert!(
            (stats.mean - 2.0).abs() < 0.001,
            "Mean should be 2.0 (from valid values)"
        );
    }

    #[test]
    fn test_compute_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_compute_stats_all_zeros() {
        let data = vec![0.0f32; 100];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.zero_count, 100);
        assert_eq!(stats.mean, 0.0);
    }
}

#[cfg(test)]
mod tests_quantization {
    use super::*;

    #[test]
    fn test_quantize_int8_basic() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0];
        let quantized = quantize_int8(&data);

        assert_eq!(quantized.len(), data.len());
        // Values should be close but not exact due to quantization
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            assert!((orig - quant).abs() < 0.02, "Quantization error too large");
        }
    }

    #[test]
    fn test_quantize_int8_preserves_zeros() {
        let data = vec![0.0f32; 10];
        let quantized = quantize_int8(&data);
        assert!(
            quantized.iter().all(|&v| v == 0.0),
            "Zeros should remain zeros"
        );
    }

    #[test]
    fn test_quantize_int8_empty() {
        let data: Vec<f32> = vec![];
        let quantized = quantize_int8(&data);
        assert!(quantized.is_empty());
    }

    #[test]
    fn test_quantize_int4_basic() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0];
        let quantized = quantize_int4(&data);

        assert_eq!(quantized.len(), data.len());
        // Int4 has more error than int8
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            assert!(
                (orig - quant).abs() < 0.2,
                "Int4 quantization error too large"
            );
        }
    }

    #[test]
    fn test_quantize_fp16_basic() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 0.123456789];
        let quantized = quantize_fp16(&data);

        assert_eq!(quantized.len(), data.len());
        // FP16 should have minimal error for simple values
        assert_eq!(quantized[0], 1.0);
        assert_eq!(quantized[1], -1.0);
        assert_eq!(quantized[4], 0.0);
    }

    #[test]
    fn test_quantize_tensors_int8() {
        let mut tensors = BTreeMap::new();
        tensors.insert("test".to_string(), (vec![1.0f32, -1.0, 0.5], vec![3]));

        let result = quantize_tensors(&tensors, &QuantizationType::Int8).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result.contains_key("test"));
        let (data, shape) = result.get("test").unwrap();
        assert_eq!(shape, &vec![3]);
        assert_eq!(data.len(), 3);
    }
}

#[cfg(test)]
mod tests_convert {
    use super::*;

    fn create_test_model(path: &Path) {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "encoder.weight".to_string(),
            (vec![0.01f32; 1000], vec![100, 10]),
        );
        tensors.insert("encoder.bias".to_string(), (vec![0.0f32; 100], vec![100]));
        tensors.insert(
            "decoder.weight".to_string(),
            (vec![0.02f32; 500], vec![50, 10]),
        );
        save_safetensors(path, &tensors).expect("Failed to create test model");
    }

    #[test]
    fn test_convert_no_quantization() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("convert_input.safetensors");
        let output = dir.path().join("convert_output.apr");

        create_test_model(&input);

        let options = ConvertOptions::default();
        let result = apr_convert(&input, &output, options);

        assert!(
            result.is_ok(),
            "Convert without quantization should work: {:?}",
            result.err()
        );
        let report = result.unwrap();
        assert_eq!(report.tensor_count, 3);
        assert!(report.quantization.is_none());
    }

    #[test]
    fn test_convert_with_int8_quantization() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("convert_int8_input.safetensors");
        let output = dir.path().join("convert_int8_output.apr");

        create_test_model(&input);

        let options = ConvertOptions {
            quantize: Some(QuantizationType::Int8),
            ..Default::default()
        };
        let result = apr_convert(&input, &output, options);

        assert!(
            result.is_ok(),
            "Int8 quantization should work: {:?}",
            result.err()
        );
        let report = result.unwrap();
        assert_eq!(report.quantization, Some(QuantizationType::Int8));
        assert_eq!(report.tensor_count, 3);
    }

    #[test]
    fn test_convert_with_fp16_quantization() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("convert_fp16_input.safetensors");
        let output = dir.path().join("convert_fp16_output.apr");

        create_test_model(&input);

        let options = ConvertOptions {
            quantize: Some(QuantizationType::Fp16),
            ..Default::default()
        };
        let result = apr_convert(&input, &output, options);

        assert!(
            result.is_ok(),
            "FP16 quantization should work: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_convert_nonexistent_file() {
        let options = ConvertOptions::default();
        let result = apr_convert(
            "/tmp/nonexistent_model_abc123.safetensors",
            "/tmp/nonexistent_output_abc123.apr",
            options,
        );

        assert!(result.is_err(), "Nonexistent file should fail");
    }

    #[test]
    fn test_convert_report_reduction_percent() {
        let report = ConvertReport {
            original_size: 1000,
            converted_size: 250,
            tensor_count: 5,
            quantization: Some(QuantizationType::Int8),
            compression: None,
            reduction_ratio: 4.0,
        };

        assert_eq!(report.reduction_percent(), "75.0%");
    }

    #[test]
    fn test_convert_options_default() {
        let options = ConvertOptions::default();
        assert!(options.quantize.is_none());
        assert!(options.compress.is_none());
        assert!(options.validate);
    }
}

// ============================================================================
// GH-127: Multi-tensor (sharded) model import tests
// ============================================================================

#[cfg(test)]
mod tests_sharded_import {
    use super::*;

    #[test]
    fn test_sharded_index_parse_valid() {
        let json = r#"{
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "encoder.conv1.weight": "model-00001-of-00002.safetensors",
                "encoder.conv2.weight": "model-00001-of-00002.safetensors",
                "decoder.fc.weight": "model-00002-of-00002.safetensors"
            }
        }"#;

        let index = ShardedIndex::parse(json).expect("Valid index should parse");
        assert_eq!(index.shard_count(), 2);
        assert_eq!(index.tensor_count(), 3);
        assert!(index.total_size().is_some());
    }

    #[test]
    fn test_sharded_index_shard_for_tensor() {
        let json = r#"{
            "weight_map": {
                "encoder.weight": "shard-00001.safetensors",
                "decoder.weight": "shard-00002.safetensors"
            }
        }"#;

        let index = ShardedIndex::parse(json).unwrap();
        assert_eq!(
            index.shard_for_tensor("encoder.weight"),
            Some("shard-00001.safetensors")
        );
        assert_eq!(
            index.shard_for_tensor("decoder.weight"),
            Some("shard-00002.safetensors")
        );
        assert_eq!(index.shard_for_tensor("unknown"), None);
    }

    #[test]
    fn test_sharded_index_tensors_in_shard() {
        let json = r#"{
            "weight_map": {
                "a": "shard1.safetensors",
                "b": "shard1.safetensors",
                "c": "shard2.safetensors"
            }
        }"#;

        let index = ShardedIndex::parse(json).unwrap();
        let shard1_tensors = index.tensors_in_shard("shard1.safetensors");
        assert_eq!(shard1_tensors.len(), 2);
        assert!(shard1_tensors.contains(&"a"));
        assert!(shard1_tensors.contains(&"b"));
    }

    #[test]
    fn test_sharded_index_parse_invalid_json() {
        let result = ShardedIndex::parse("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_sharded_index_parse_missing_weight_map() {
        let result = ShardedIndex::parse(r#"{"metadata": {}}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_sharded_model_index_exists() {
        // Create a temp dir with index.json
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("model.safetensors.index.json");
        fs::write(&index_path, r#"{"weight_map": {"a": "shard.safetensors"}}"#).unwrap();

        let result = detect_sharded_model(dir.path(), "model.safetensors");
        assert!(result.is_some());
    }

    #[test]
    fn test_detect_sharded_model_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.safetensors");
        fs::write(&model_path, &[0u8; 8]).unwrap(); // Minimal file

        let result = detect_sharded_model(dir.path(), "model.safetensors");
        assert!(result.is_none(), "Single file should not be sharded");
    }

    #[test]
    fn test_sharded_index_shard_files_sorted() {
        let json = r#"{
            "weight_map": {
                "a": "model-00002-of-00003.safetensors",
                "b": "model-00001-of-00003.safetensors",
                "c": "model-00003-of-00003.safetensors"
            }
        }"#;

        let index = ShardedIndex::parse(json).unwrap();
        let shards = index.shard_files();
        assert_eq!(shards[0], "model-00001-of-00003.safetensors");
        assert_eq!(shards[1], "model-00002-of-00003.safetensors");
        assert_eq!(shards[2], "model-00003-of-00003.safetensors");
    }
}

// ============================================================================
// GH-196: Conversion round-trip regression tests
// ============================================================================

#[cfg(test)]
mod tests_gh196_roundtrip {
    use super::*;
    use crate::format::test_factory::harness::ConversionTestHarness;
    use crate::format::test_factory::PygmyConfig;
    use crate::format::v2::AprV2Reader;

    /// GH-196: Auto architecture infers from tensor names (no explicit arch required).
    #[test]
    fn test_gh196_auto_arch_import() {
        let h = ConversionTestHarness::new()
            .with_safetensors(PygmyConfig::llama_style())
            .import_to_apr(ImportOptions {
                architecture: Architecture::Auto,
                ..Default::default()
            });

        // Should succeed and produce valid APR
        let output = h.output_path().expect("output exists");
        let data = fs::read(output).expect("read output");
        let reader = AprV2Reader::from_bytes(&data).expect("parse APR");
        assert!(
            !reader.tensor_names().is_empty(),
            "GH-196: Auto arch should produce tensors"
        );
    }

    /// GH-196: --strict blocks Auto architecture when tensors don't match known patterns.
    /// Uses `embedding_only()` config which lacks `model.layers` tensor names,
    /// so auto-detection yields "unknown" architecture (unverified).
    #[test]
    fn test_gh196_strict_rejects_unverified() {
        let h = ConversionTestHarness::new()
            .with_safetensors(PygmyConfig::embedding_only());

        let result = h.try_import_to_apr(ImportOptions {
            architecture: Architecture::Auto,
            validation: ValidationConfig::Strict,
            strict: true,
            ..Default::default()
        });

        assert!(
            result.is_err(),
            "GH-196: --strict should reject unverified Auto architecture"
        );
    }

    /// GH-196: Default (non-strict) allows Auto architecture.
    #[test]
    fn test_gh196_default_permissive() {
        let h = ConversionTestHarness::new()
            .with_safetensors(PygmyConfig::default());

        let result = h.try_import_to_apr(ImportOptions::default());

        assert!(
            result.is_ok(),
            "GH-196: Default (permissive) should allow Auto: {:?}",
            result.err()
        );
    }

    /// GH-196: Imported tensors match source bit-for-bit (F32 tolerance).
    #[test]
    fn test_gh196_tensor_data_preserved() {
        let h = ConversionTestHarness::new()
            .with_safetensors(PygmyConfig::default())
            .import_to_apr(ImportOptions::default());

        let result = h.verify_apr();
        result.assert_passed();
    }

    /// GH-196: Full round-trip SafeTensors -> APR -> SafeTensors with default config.
    #[test]
    fn test_gh196_full_roundtrip_default() {
        ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::default());
    }

    /// GH-196: Full round-trip with LLaMA-style config (attention + MLP + norms).
    #[test]
    fn test_gh196_full_roundtrip_llama() {
        ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::llama_style());
    }

    /// GH-196: Full round-trip with minimal config (embedding only).
    #[test]
    fn test_gh196_full_roundtrip_minimal() {
        ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::minimal());
    }

    /// GH-196: Architecture field preserved in APR metadata after import.
    #[test]
    fn test_gh196_metadata_architecture() {
        let h = ConversionTestHarness::new()
            .with_safetensors(PygmyConfig::llama_style())
            .import_to_apr(ImportOptions {
                architecture: Architecture::Llama,
                ..Default::default()
            });

        let output = h.output_path().expect("output exists");
        let data = fs::read(output).expect("read output");
        let reader = AprV2Reader::from_bytes(&data).expect("parse APR");
        let metadata = reader.metadata();

        // The architecture field should be set (either from import option or inferred)
        assert!(
            metadata.architecture.is_some(),
            "GH-196: APR metadata should have architecture field, got: {:?}",
            metadata.architecture
        );
    }
}
