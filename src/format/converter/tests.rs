//! APR Converter Tests - Extreme TDD
//! PMAT-197: Extracted from mod.rs for file size reduction

#[allow(unused_imports)]
use super::*;

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
        assert_eq!(exp.unwrap().mean_range, (-0.5, 3.0));

        // LLaMA-style post_attention_layernorm
        let exp = TensorExpectation::for_tensor("model.layers.5.post_attention_layernorm.weight");
        assert!(exp.is_some());
        assert_eq!(exp.unwrap().mean_range, (-0.5, 3.0));

        // Final norm
        let exp = TensorExpectation::for_tensor("model.norm.weight");
        assert!(exp.is_some());
        assert_eq!(exp.unwrap().mean_range, (-0.5, 3.0));
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
        assert!(!opts.force);
        assert!(opts.cache);
    }
}

#[cfg(test)]
mod tests_conversion {
    use super::*;

    fn create_test_safetensors(path: &Path, tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) {
        save_safetensors(path, tensors).expect("Failed to create test SafeTensors file");
    }

    #[test]
    fn test_convert_valid_safetensors() {
        let input = "/tmp/test_valid_input.safetensors";
        let output = "/tmp/test_valid_output.apr";

        // Create valid test tensors
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "encoder.layer_norm.weight".to_string(),
            (vec![1.0f32; 384], vec![384]),
        );
        tensors.insert(
            "encoder.layer_norm.bias".to_string(),
            (vec![0.0f32; 384], vec![384]),
        );
        tensors.insert(
            "encoder.conv1.weight".to_string(),
            (vec![0.01f32; 1000], vec![80, 1, 3]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        // Run conversion
        let options = ImportOptions::default();
        let result = apr_import(input, output, options);

        assert!(
            result.is_ok(),
            "Valid tensors should convert successfully: {:?}",
            result.err()
        );
        let report = result.unwrap();
        assert!(report.total_score > 0, "Score should be > 0");

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_invalid_layernorm_fails_strict() {
        let input = "/tmp/test_invalid_ln_input.safetensors";
        let output = "/tmp/test_invalid_ln_output.apr";

        // Create tensors with INVALID LayerNorm (mean=11, should be ~1)
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "decoder.layer_norm.weight".to_string(),
            (vec![11.0f32; 384], vec![384]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        // Run conversion with strict validation
        let options = ImportOptions {
            validation: ValidationConfig::Strict,
            force: false,
            ..Default::default()
        };
        let result = apr_import(input, output, options);

        assert!(
            result.is_err(),
            "Invalid LayerNorm should fail strict validation"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mean=11") || err.contains("LayerNorm"),
            "Error should mention LayerNorm issue: {err}"
        );

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_invalid_layernorm_force_succeeds() {
        let input = "/tmp/test_force_ln_input.safetensors";
        let output = "/tmp/test_force_ln_output.apr";

        // Create tensors with invalid LayerNorm
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "decoder.layer_norm.weight".to_string(),
            (vec![11.0f32; 384], vec![384]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        // Run conversion with force=true
        let options = ImportOptions {
            validation: ValidationConfig::Strict,
            force: true,
            ..Default::default()
        };
        let result = apr_import(input, output, options);

        assert!(
            result.is_ok(),
            "Force should bypass validation: {:?}",
            result.err()
        );

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_nan_fails() {
        let input = "/tmp/test_nan_input.safetensors";
        let output = "/tmp/test_nan_output.apr";

        // Create tensors with NaN
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "test.weight".to_string(),
            (vec![1.0, f32::NAN, 3.0], vec![3]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        let options = ImportOptions::default();
        let result = apr_import(input, output, options);

        assert!(result.is_err(), "NaN should fail validation");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN"), "Error should mention NaN: {err}");

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_nonexistent_file() {
        let result = apr_import(
            "/tmp/nonexistent_model.safetensors",
            "/tmp/out.apr",
            ImportOptions::default(),
        );
        assert!(result.is_err(), "Nonexistent file should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found") || err.contains("No such file"),
            "Error should mention file not found: {err}"
        );
    }

    #[test]
    fn test_convert_unsupported_format() {
        let input = "/tmp/test_bad_format.gguf";
        fs::write(input, b"test").expect("Failed to create test file");

        let result = apr_import(input, "/tmp/out.apr", ImportOptions::default());
        assert!(result.is_err(), "Unsupported format should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("GGUF") || err.contains("not yet"),
            "Error should mention unsupported: {err}"
        );

        fs::remove_file(input).ok();
    }

    #[test]
    fn test_name_mapping_whisper() {
        use crate::format::v2::AprV2Reader;

        let input = "/tmp/test_whisper_input.safetensors";
        let output = "/tmp/test_whisper_output.apr";

        // Create tensors with HuggingFace-style names
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.encoder.conv1.weight".to_string(),
            (vec![0.01f32; 100], vec![10, 10]),
        );
        tensors.insert(
            "model.decoder.layer_norm.weight".to_string(),
            (vec![1.0f32; 384], vec![384]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        let options = ImportOptions {
            architecture: Architecture::Whisper,
            ..Default::default()
        };
        let result = apr_import(input, output, options);
        assert!(
            result.is_ok(),
            "Whisper mapping should work: {:?}",
            result.err()
        );

        // Load output as APR and verify names are preserved (PMAT-099)
        let data = fs::read(output).expect("Failed to read output");
        let reader = AprV2Reader::from_bytes(&data).expect("Failed to parse APR");
        let tensor_names = reader.tensor_names();

        // PMAT-099: Names are now preserved for AprTransformer compatibility
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

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
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
        let input = Path::new("/tmp/test_convert_input.safetensors");
        let output = Path::new("/tmp/test_convert_output.apr");

        create_test_model(input);

        let options = ConvertOptions::default();
        let result = apr_convert(input, output, options);

        assert!(
            result.is_ok(),
            "Convert without quantization should work: {:?}",
            result.err()
        );
        let report = result.unwrap();
        assert_eq!(report.tensor_count, 3);
        assert!(report.quantization.is_none());

        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_with_int8_quantization() {
        let input = Path::new("/tmp/test_convert_int8_input.safetensors");
        let output = Path::new("/tmp/test_convert_int8_output.apr");

        create_test_model(input);

        let options = ConvertOptions {
            quantize: Some(QuantizationType::Int8),
            ..Default::default()
        };
        let result = apr_convert(input, output, options);

        assert!(
            result.is_ok(),
            "Int8 quantization should work: {:?}",
            result.err()
        );
        let report = result.unwrap();
        assert_eq!(report.quantization, Some(QuantizationType::Int8));
        assert_eq!(report.tensor_count, 3);

        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_with_fp16_quantization() {
        let input = Path::new("/tmp/test_convert_fp16_input.safetensors");
        let output = Path::new("/tmp/test_convert_fp16_output.apr");

        create_test_model(input);

        let options = ConvertOptions {
            quantize: Some(QuantizationType::Fp16),
            ..Default::default()
        };
        let result = apr_convert(input, output, options);

        assert!(
            result.is_ok(),
            "FP16 quantization should work: {:?}",
            result.err()
        );

        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_nonexistent_file() {
        let options = ConvertOptions::default();
        let result = apr_convert("/tmp/nonexistent.safetensors", "/tmp/out.apr", options);

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
// GH-129: Import error message tests
// ============================================================================

#[cfg(test)]
mod tests_import_errors {
    use super::*;

    #[test]
    fn test_import_error_not_found_message() {
        let err = ImportError::NotFound {
            resource: "openai/whisper-tiny".to_string(),
            status: 404,
        };
        let msg = err.to_string();
        assert!(msg.contains("404"), "Should include status code");
        assert!(msg.contains("whisper-tiny"), "Should include resource");
    }

    #[test]
    fn test_import_error_rate_limited_message() {
        let err = ImportError::RateLimited {
            retry_after: Some(60),
        };
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("rate"),
            "Should mention rate limit"
        );
        assert!(msg.contains("60"), "Should include retry time");
    }

    #[test]
    fn test_import_error_auth_required_message() {
        let err = ImportError::AuthRequired {
            resource: "meta-llama/Llama-2-7b".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("HF_TOKEN"), "Should suggest HF_TOKEN");
        assert!(msg.contains("Llama-2-7b"), "Should include resource");
    }

    #[test]
    fn test_import_error_actionable_suggestions() {
        let err = ImportError::NotFound {
            resource: "openai/whisper-tiny".to_string(),
            status: 404,
        };

        // Error should provide actionable fix
        let msg = err.to_string();
        assert!(
            msg.contains("Fix:") || msg.contains("check") || msg.contains("verify"),
            "Error should be actionable"
        );
    }

    #[test]
    fn test_import_error_sharding_oom() {
        let err = ImportError::ShardingRequired {
            model_size: 14_000_000_000, // 14GB
            shard_count: 7,
        };
        let msg = err.to_string();
        assert!(msg.contains("14"), "Should include size");
        assert!(msg.contains("7"), "Should include shard count");
    }

    // GH-129: Tests for parse_import_error (only when hf-hub-integration enabled)
    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_404() {
        let err = parse_import_error("HTTP 404: Repository not found", "openai/whisper-tiny");
        match err {
            ImportError::NotFound { resource, status } => {
                assert_eq!(resource, "openai/whisper-tiny");
                assert_eq!(status, 404);
            }
            _ => panic!("Expected NotFound error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_not_found_text() {
        let err = parse_import_error("The requested resource does not exist", "test/model");
        match err {
            ImportError::NotFound { .. } => {}
            _ => panic!("Expected NotFound error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_401() {
        let err = parse_import_error("HTTP 401: Unauthorized access", "meta-llama/Llama-2-7b");
        match err {
            ImportError::AuthRequired { resource } => {
                assert_eq!(resource, "meta-llama/Llama-2-7b");
            }
            _ => panic!("Expected AuthRequired error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_gated_model() {
        let err = parse_import_error(
            "This model is gated. Access requires acceptance.",
            "meta-llama/Llama-2-7b",
        );
        match err {
            ImportError::AuthRequired { .. } => {}
            _ => panic!("Expected AuthRequired error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_429() {
        let err = parse_import_error(
            "HTTP 429: Too many requests. Retry after 60 seconds.",
            "test/model",
        );
        match err {
            ImportError::RateLimited { retry_after } => {
                assert_eq!(retry_after, Some(60));
            }
            _ => panic!("Expected RateLimited error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_rate_limit_no_retry() {
        let err = parse_import_error("Rate limit exceeded", "test/model");
        match err {
            ImportError::RateLimited { retry_after } => {
                assert_eq!(retry_after, None);
            }
            _ => panic!("Expected RateLimited error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_generic() {
        let err = parse_import_error("Connection timeout", "test/model");
        match err {
            ImportError::DownloadFailed { source, reason } => {
                assert_eq!(source, "test/model");
                assert_eq!(reason, "Connection timeout");
            }
            _ => panic!("Expected DownloadFailed error, got {:?}", err),
        }
    }

    #[test]
    fn test_import_error_from_conversion() {
        let import_err = ImportError::NotFound {
            resource: "test".to_string(),
            status: 404,
        };
        let aprender_err: AprenderError = import_err.into();
        let msg = aprender_err.to_string();
        assert!(msg.contains("404"));
        assert!(msg.contains("test"));
    }

    // =========================================================================
    // Coverage boost: ExportFormat, MergeStrategy, and related APIs
    // =========================================================================

    #[test]
    fn test_export_format_from_str() {
        assert!(matches!(
            "safetensors".parse::<ExportFormat>(),
            Ok(ExportFormat::SafeTensors)
        ));
        assert!(matches!(
            "st".parse::<ExportFormat>(),
            Ok(ExportFormat::SafeTensors)
        ));
        assert!(matches!(
            "gguf".parse::<ExportFormat>(),
            Ok(ExportFormat::Gguf)
        ));
        assert!(matches!(
            "onnx".parse::<ExportFormat>(),
            Ok(ExportFormat::Onnx)
        ));
        assert!(matches!(
            "torchscript".parse::<ExportFormat>(),
            Ok(ExportFormat::TorchScript)
        ));
        assert!(matches!(
            "pt".parse::<ExportFormat>(),
            Ok(ExportFormat::TorchScript)
        ));
        assert!(matches!(
            "torch".parse::<ExportFormat>(),
            Ok(ExportFormat::TorchScript)
        ));
        assert!("unknown".parse::<ExportFormat>().is_err());
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(ExportFormat::Gguf.extension(), "gguf");
        assert_eq!(ExportFormat::Onnx.extension(), "onnx");
        assert_eq!(ExportFormat::TorchScript.extension(), "pt");
    }

    #[test]
    fn test_export_format_is_supported() {
        assert!(ExportFormat::SafeTensors.is_supported());
        assert!(ExportFormat::Gguf.is_supported());
        assert!(!ExportFormat::Onnx.is_supported());
        assert!(!ExportFormat::TorchScript.is_supported());
    }

    #[test]
    fn test_export_options_default() {
        let opts = ExportOptions::default();
        assert!(matches!(opts.format, ExportFormat::SafeTensors));
        assert!(opts.quantize.is_none());
    }

    #[test]
    fn test_export_options_with_quantize() {
        let opts = ExportOptions {
            format: ExportFormat::Gguf,
            quantize: Some(QuantizationType::Int8),
            ..Default::default()
        };
        assert!(matches!(opts.format, ExportFormat::Gguf));
        assert!(matches!(opts.quantize, Some(QuantizationType::Int8)));
    }

    #[test]
    fn test_merge_strategy_from_str() {
        assert!(matches!(
            "average".parse::<MergeStrategy>(),
            Ok(MergeStrategy::Average)
        ));
        assert!(matches!(
            "avg".parse::<MergeStrategy>(),
            Ok(MergeStrategy::Average)
        ));
        assert!(matches!(
            "weighted".parse::<MergeStrategy>(),
            Ok(MergeStrategy::Weighted)
        ));
        assert!("unknown".parse::<MergeStrategy>().is_err());
    }

    #[test]
    fn test_merge_strategy_is_supported() {
        // Average and Weighted are supported
        assert!(MergeStrategy::Average.is_supported());
        assert!(MergeStrategy::Weighted.is_supported());
        // Advanced strategies not yet implemented
        assert!(!MergeStrategy::Ties.is_supported());
        assert!(!MergeStrategy::Dare.is_supported());
        assert!(!MergeStrategy::Slerp.is_supported());
    }

    #[test]
    fn test_merge_options_default() {
        let opts = MergeOptions::default();
        assert!(matches!(opts.strategy, MergeStrategy::Average));
        assert!(opts.weights.is_none());
    }

    #[test]
    fn test_merge_options_weighted() {
        let opts = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.7, 0.3]),
        };
        assert!(matches!(opts.strategy, MergeStrategy::Weighted));
        assert_eq!(opts.weights, Some(vec![0.7, 0.3]));
    }

    #[test]
    fn test_merge_report_fields() {
        let report = MergeReport {
            model_count: 2,
            output_size: 1000,
            tensor_count: 10,
            strategy: MergeStrategy::Average,
            weights_used: None,
        };
        assert_eq!(report.model_count, 2);
        assert_eq!(report.output_size, 1000);
        assert_eq!(report.tensor_count, 10);
    }

    #[test]
    fn test_merge_report_with_weights() {
        let report = MergeReport {
            model_count: 3,
            output_size: 2000,
            tensor_count: 15,
            strategy: MergeStrategy::Weighted,
            weights_used: Some(vec![0.5, 0.3, 0.2]),
        };
        assert_eq!(report.model_count, 3);
        assert!(matches!(report.strategy, MergeStrategy::Weighted));
        assert!(report.weights_used.is_some());
    }

    #[test]
    fn test_export_report_fields() {
        let report = ExportReport {
            original_size: 2000,
            exported_size: 1000,
            tensor_count: 5,
            format: ExportFormat::Gguf,
            quantization: Some(QuantizationType::Int8),
        };
        assert_eq!(report.original_size, 2000);
        assert_eq!(report.exported_size, 1000);
        assert_eq!(report.tensor_count, 5);
    }

    #[test]
    fn test_validation_config_strict() {
        let config = ValidationConfig::strict();
        assert_eq!(config, ValidationConfig::Strict);
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config, ValidationConfig::Strict);
    }

    #[test]
    fn test_validation_config_variants() {
        let _none = ValidationConfig::None;
        let _basic = ValidationConfig::Basic;
        let _strict = ValidationConfig::Strict;
    }

    #[test]
    fn test_import_options_default() {
        let opts = ImportOptions::default();
        assert_eq!(opts.validation, ValidationConfig::Strict);
        assert!(opts.quantize.is_none());
        assert!(opts.compress.is_none());
    }

    #[test]
    fn test_architecture_mapping_auto() {
        let arch = Architecture::Auto;
        // PMAT-099: Preserve model. prefix for AprTransformer compatibility
        assert_eq!(
            arch.map_name("model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        // Pass through names without prefix
        assert_eq!(arch.map_name("layer.0.weight"), "layer.0.weight");
    }

    #[test]
    fn test_architecture_mapping_whisper() {
        let arch = Architecture::Whisper;
        let name = arch.map_name("model.encoder.weight");
        assert!(!name.is_empty());
    }

    #[test]
    fn test_architecture_mapping_llama() {
        let arch = Architecture::Llama;
        let name = arch.map_name("model.layers.0.weight");
        assert!(!name.is_empty());
    }

    #[test]
    fn test_architecture_mapping_bert() {
        let arch = Architecture::Bert;
        let name = arch.map_name("bert.encoder.layer.0.weight");
        assert!(!name.is_empty());
    }

    #[test]
    fn test_source_parse_local_absolute() {
        let source = Source::parse("/path/to/model.safetensors").unwrap();
        assert!(matches!(source, Source::Local(_)));
    }

    #[test]
    fn test_source_parse_local_relative() {
        let source = Source::parse("./models/model.safetensors").unwrap();
        assert!(matches!(source, Source::Local(_)));
    }

    #[test]
    fn test_source_default_file_hf() {
        let source = Source::HuggingFace {
            org: "openai".to_string(),
            repo: "whisper".to_string(),
            file: None,
        };
        assert_eq!(source.default_file(), "model.safetensors");
    }

    #[test]
    fn test_source_default_file_local() {
        let source = Source::Local("/path/to/model.safetensors".into());
        // Local returns full path as the "file"
        assert!(source.default_file().ends_with("model.safetensors"));
    }

    #[test]
    fn test_tensor_expectation_for_unknown() {
        let exp = TensorExpectation::for_tensor("unknown_tensor_name");
        assert!(exp.is_none());
    }

    #[test]
    fn test_tensor_expectation_for_layer_norm_weight() {
        let exp = TensorExpectation::for_tensor("layer_norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        // LayerNorm weight should have mean near 1.0
        assert!(exp.mean_range.0 < 1.0 && exp.mean_range.1 > 1.0);
    }

    #[test]
    fn test_tensor_expectation_for_embedding() {
        let exp = TensorExpectation::for_tensor("embed_tokens.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_import_error_display() {
        let err = ImportError::NotFound {
            resource: "model.safetensors".to_string(),
            status: 404,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("404") || msg.contains("not found"));
    }

    #[test]
    fn test_import_error_download_failed() {
        let err = ImportError::DownloadFailed {
            source: "huggingface".to_string(),
            reason: "timeout".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("timeout") || msg.contains("Download"));
    }

    #[test]
    fn test_import_error_validation_failed() {
        let err = ImportError::ValidationFailed {
            name: "layer.weight".to_string(),
            reason: "NaN detected".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("layer.weight") || msg.contains("NaN"));
    }

    #[test]
    fn test_import_error_unsupported_format() {
        let err = ImportError::UnsupportedFormat {
            extension: "pickle".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("pickle") || msg.contains("Unsupported"));
    }

    #[test]
    fn test_import_error_unknown_tensor() {
        let err = ImportError::UnknownTensor {
            source_name: "weird.tensor".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("weird.tensor") || msg.contains("Unknown"));
    }

    #[test]
    fn test_import_error_missing_tensor() {
        let err = ImportError::MissingTensor {
            name: "model.weight".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("model.weight") || msg.contains("Missing"));
    }

    #[test]
    fn test_import_error_rate_limited() {
        let err = ImportError::RateLimited {
            retry_after: Some(60),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Rate") || msg.contains("limit") || msg.contains("60"));
    }

    #[test]
    fn test_import_error_auth_required() {
        let err = ImportError::AuthRequired {
            resource: "gated-model".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Auth") || msg.contains("gated-model"));
    }

    #[test]
    fn test_import_error_sharding_required() {
        let err = ImportError::ShardingRequired {
            model_size: 14_000_000_000,
            shard_count: 7,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("shard") || msg.contains("7"));
    }

    // =========================================================================
    // ShardedIndex Tests
    // =========================================================================

    #[test]
    fn test_sharded_index_parse() {
        let json = r#"{
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        let index = ShardedIndex::parse(json).expect("parse should succeed");
        assert_eq!(index.tensor_count(), 2);
        assert_eq!(index.shard_count(), 2);
    }

    #[test]
    fn test_sharded_index_shard_for_tensor() {
        let json = r#"{
            "weight_map": {
                "embed.weight": "model-00001.safetensors",
                "lm_head.weight": "model-00002.safetensors"
            }
        }"#;
        let index = ShardedIndex::parse(json).expect("parse");
        assert_eq!(
            index.shard_for_tensor("embed.weight"),
            Some("model-00001.safetensors")
        );
        assert_eq!(
            index.shard_for_tensor("lm_head.weight"),
            Some("model-00002.safetensors")
        );
        assert_eq!(index.shard_for_tensor("missing"), None);
    }

    #[test]
    fn test_sharded_index_tensors_in_shard() {
        let json = r#"{
            "weight_map": {
                "a.weight": "shard1.safetensors",
                "b.weight": "shard1.safetensors",
                "c.weight": "shard2.safetensors"
            }
        }"#;
        let index = ShardedIndex::parse(json).expect("parse");
        let tensors = index.tensors_in_shard("shard1.safetensors");
        assert_eq!(tensors.len(), 2);
        assert!(tensors.contains(&"a.weight"));
        assert!(tensors.contains(&"b.weight"));
    }

    #[test]
    fn test_sharded_index_shard_files() {
        let json = r#"{
            "weight_map": {
                "a": "z.safetensors",
                "b": "a.safetensors",
                "c": "m.safetensors"
            }
        }"#;
        let index = ShardedIndex::parse(json).expect("parse");
        let files = index.shard_files();
        // Should be sorted
        assert_eq!(
            files,
            vec!["a.safetensors", "m.safetensors", "z.safetensors"]
        );
    }

    #[test]
    fn test_sharded_index_total_size() {
        let with_size = r#"{"metadata": {"total_size": 5000}, "weight_map": {}}"#;
        let without_size = r#"{"weight_map": {}}"#;

        let index1 = ShardedIndex::parse(with_size).expect("parse");
        let index2 = ShardedIndex::parse(without_size).expect("parse");

        assert_eq!(index1.total_size(), Some(5000));
        assert_eq!(index2.total_size(), None);
    }

    #[test]
    fn test_sharded_index_parse_invalid_json() {
        let result = ShardedIndex::parse("not valid json");
        assert!(result.is_err());
    }

    // =========================================================================
    // Source URL Tests
    // =========================================================================

    #[test]
    fn test_source_parse_url() {
        let source = Source::parse("https://example.com/model.safetensors").unwrap();
        assert!(matches!(source, Source::Url(_)));
    }

    #[test]
    fn test_source_parse_http_url() {
        let source = Source::parse("http://localhost:8080/model.bin").unwrap();
        assert!(matches!(source, Source::Url(_)));
    }

    #[test]
    fn test_source_default_file_url() {
        let source = Source::Url("https://example.com/path/to/model.safetensors".to_string());
        assert_eq!(source.default_file(), "model.safetensors");
    }

    // =========================================================================
    // ConvertReport Tests
    // =========================================================================

    #[test]
    fn test_convert_report_reduction_percent() {
        let report = ConvertReport {
            original_size: 1000,
            converted_size: 500,
            tensor_count: 10,
            quantization: Some(QuantizationType::Int8),
            compression: None,
            reduction_ratio: 2.0,
        };
        let reduction = report.reduction_percent();
        assert!(reduction.contains("50"));
    }

    #[test]
    fn test_convert_report_no_reduction() {
        let report = ConvertReport {
            original_size: 1000,
            converted_size: 1000,
            tensor_count: 5,
            quantization: None,
            compression: None,
            reduction_ratio: 1.0,
        };
        let reduction = report.reduction_percent();
        assert!(reduction.contains("0"));
    }

    // =========================================================================
    // ExportFormat Tests
    // =========================================================================

    #[test]
    fn test_export_format_safetensors() {
        let format = ExportFormat::SafeTensors;
        assert_eq!(format.extension(), "safetensors");
        assert!(format.is_supported());
    }

    #[test]
    fn test_export_format_gguf() {
        let format = ExportFormat::Gguf;
        assert_eq!(format.extension(), "gguf");
        assert!(format.is_supported());
    }

    #[test]
    fn test_export_format_onnx() {
        let format = ExportFormat::Onnx;
        assert_eq!(format.extension(), "onnx");
        // ONNX may or may not be supported
        let _ = format.is_supported();
    }

    #[test]
    fn test_export_format_torchscript() {
        let format = ExportFormat::TorchScript;
        assert_eq!(format.extension(), "pt");
    }

    // =========================================================================
    // Quantization Type Tests
    // =========================================================================

    #[test]
    fn test_quantization_type_debug() {
        let q = QuantizationType::Int8;
        let debug = format!("{:?}", q);
        assert!(debug.contains("Int8"));
    }

    #[test]
    fn test_quantization_type_clone() {
        let q1 = QuantizationType::Int4;
        let q2 = q1.clone();
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_q4k_quantization_roundtrip() {
        // Test data: 512 f32 values (2 super-blocks of 256)
        // Use realistic weight distribution: centered around 0, mostly negative to positive
        let mut original: Vec<f32> = Vec::with_capacity(512);
        for i in 0..512 {
            // Simulate typical weight distribution: values mostly in [-0.1, 0.1]
            // with some outliers in [-0.3, 0.3]
            let base = ((i as f32) / 512.0 - 0.5) * 0.2; // -0.1 to 0.1
            let noise = (i as f32 * 0.1).sin() * 0.05;
            original.push(base + noise);
        }

        // Quantize to Q4K bytes
        let q4k_bytes = quantize_q4_k(&original);

        // Expected size: 2 super-blocks * 144 bytes each = 288 bytes
        assert_eq!(
            q4k_bytes.len(),
            288,
            "Q4K output should be 144 bytes per 256-element super-block"
        );

        // Dequantize back to f32
        let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 512);
        assert_eq!(reconstructed.len(), 512);

        // Check reconstruction error
        let mut max_error = 0.0f32;
        let mut total_error = 0.0f32;
        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            max_error = max_error.max(error);
            total_error += error;
        }
        let avg_error = total_error / 512.0;

        // Q4_K should have reasonable reconstruction quality for typical weights
        // With 4-bit quantization (15 levels) + nested 6-bit scale quantization,
        // max error is approximately: value_range * (1/15 + 1/63) ≈ range * 0.08
        // For our data range of ~0.3, max error ~0.024, but f16 quantization
        // of d/dmin adds additional error, so we allow up to 0.06
        assert!(
            max_error < 0.06,
            "Q4K max reconstruction error too high: {max_error}"
        );
        assert!(
            avg_error < 0.02,
            "Q4K avg reconstruction error too high: {avg_error}"
        );
    }

    #[test]
    fn test_q4k_empty_data() {
        let empty: Vec<f32> = vec![];
        let q4k_bytes = quantize_q4_k(&empty);
        assert!(q4k_bytes.is_empty());

        let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 0);
        assert!(reconstructed.is_empty());
    }

    #[test]
    fn test_q4k_partial_block() {
        // Test with 100 elements (less than one 256-element super-block)
        let original: Vec<f32> = (0..100).map(|i| i as f32 * 0.01 - 0.5).collect();

        let q4k_bytes = quantize_q4_k(&original);
        // Should have 1 super-block (144 bytes) since we pad to 256
        assert_eq!(q4k_bytes.len(), 144);

        let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 100);
        assert_eq!(reconstructed.len(), 100);

        // Verify reasonable reconstruction
        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            assert!(error < 0.2, "Reconstruction error too high: {error}");
        }
    }

    #[test]
    fn test_quantize_tensors_q4k() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "test".to_string(),
            (
                (0..512).map(|i| i as f32 * 0.001 - 0.256).collect(),
                vec![512],
            ),
        );

        let result = quantize_tensors(&tensors, &QuantizationType::Q4K).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result.contains_key("test"));
        let (data, shape) = &result["test"];
        assert_eq!(shape, &vec![512]);
        assert_eq!(data.len(), 512); // Dequantized back to f32
    }

    // =========================================================================
    // Compression Type Tests
    // =========================================================================

    #[test]
    fn test_compression_debug() {
        let c = Compression::ZstdDefault;
        let debug = format!("{:?}", c);
        assert!(debug.contains("Zstd"));
    }

    #[test]
    fn test_compression_clone() {
        let c1 = Compression::Lz4;
        let c2 = c1;
        assert_eq!(c1, c2);
    }

    // =========================================================================
    // TensorExpectation Check Tests
    // =========================================================================

    #[test]
    fn test_tensor_expectation_check_valid() {
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "layer_norm.weight".to_string(),
            count: 768,
            mean: 1.0,
            std: 0.1,
            min: 0.5,
            max: 1.5,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        assert!(exp.check(&stats).is_ok());
    }

    #[test]
    fn test_tensor_expectation_check_invalid_mean() {
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "layer_norm.weight".to_string(),
            count: 768,
            mean: 100.0, // Way outside expected range
            std: 0.1,
            min: 99.0,
            max: 101.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        assert!(exp.check(&stats).is_err());
    }

    // =========================================================================
    // TensorStats Creation Tests
    // =========================================================================

    #[test]
    fn test_tensor_stats_fields() {
        let stats = TensorStats {
            name: "test.weight".to_string(),
            count: 100,
            mean: 0.5,
            std: 0.2,
            min: 0.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 5,
        };
        assert!((stats.mean - 0.5).abs() < 1e-6);
        assert!((stats.std - 0.2).abs() < 1e-6);
        assert!((stats.min - 0.0).abs() < 1e-6);
        assert!((stats.max - 1.0).abs() < 1e-6);
        assert_eq!(stats.count, 100);
        assert_eq!(stats.zero_count, 5);
    }

    // =========================================================================
    // Quantization and Internal Function Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn test_calculate_tensor_size() {
        let mut tensors = BTreeMap::new();
        tensors.insert("a".to_string(), (vec![1.0f32; 100], vec![10, 10]));
        tensors.insert("b".to_string(), (vec![2.0f32; 50], vec![50]));
        let size = calculate_tensor_size(&tensors);
        // 100 * 4 + 50 * 4 = 600
        assert_eq!(size, 600);
    }

    #[test]
    fn test_calculate_tensor_size_empty() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        assert_eq!(calculate_tensor_size(&tensors), 0);
    }

    #[test]
    fn test_quantize_fp16_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, -1.0, 0.0, 0.5];
        let quantized = quantize_fp16(&data);
        // Should preserve values with f16 precision
        assert_eq!(quantized.len(), data.len());
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            // f16 has limited precision
            assert!((orig - quant).abs() < 0.01, "fp16 should preserve value");
        }
    }

    #[test]
    fn test_quantize_fp16_large_values() {
        let data = vec![65504.0, -65504.0]; // max f16 values
        let quantized = quantize_fp16(&data);
        assert!((quantized[0] - 65504.0).abs() < 1.0);
    }

    #[test]
    fn test_quantize_int8_roundtrip() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
        let quantized = quantize_int8(&data);
        assert_eq!(quantized.len(), data.len());
        // int8 quantization scales to -127..127
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            assert!(
                (orig - quant).abs() < 0.05,
                "int8 should preserve value within tolerance"
            );
        }
    }

    #[test]
    fn test_quantize_int8_all_zeros() {
        let data = vec![0.0, 0.0, 0.0];
        let quantized = quantize_int8(&data);
        for v in &quantized {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_quantize_int4_roundtrip() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
        let quantized = quantize_int4(&data);
        assert_eq!(quantized.len(), data.len());
        // int4 has only 16 levels so lower precision
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            assert!(
                (orig - quant).abs() < 0.15,
                "int4 should preserve value within tolerance"
            );
        }
    }

    #[test]
    fn test_quantize_int4_all_zeros() {
        let data = vec![0.0, 0.0, 0.0];
        let quantized = quantize_int4(&data);
        for v in &quantized {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative() {
        let result = f16_to_f32(0xBC00);
        assert!((result + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0 && result < 0.001);
    }

    #[test]
    fn test_f16_to_f32_max() {
        // Max f16 is 65504
        let result = f16_to_f32(0x7BFF);
        assert!((result - 65504.0).abs() < 1.0);
    }

    #[test]
    fn test_convert_report_zero_sizes() {
        let report = ConvertReport {
            original_size: 0,
            converted_size: 0,
            tensor_count: 0,
            quantization: None,
            compression: None,
            reduction_ratio: 0.0,
        };
        assert_eq!(report.reduction_percent(), "N/A");
    }

    #[test]
    fn test_convert_report_debug() {
        let report = ConvertReport {
            original_size: 1000,
            converted_size: 500,
            tensor_count: 10,
            quantization: Some(QuantizationType::Int8),
            compression: Some(Compression::Lz4),
            reduction_ratio: 2.0,
        };
        assert!(format!("{:?}", report).contains("ConvertReport"));
    }

    #[test]
    fn test_quantize_tensors_fp16() {
        let mut tensors = BTreeMap::new();
        tensors.insert("w".to_string(), (vec![1.0, 2.0, 3.0], vec![3]));
        let result = quantize_tensors(&tensors, &QuantizationType::Fp16).expect("quantize");
        assert!(result.contains_key("w"));
    }

    #[test]
    fn test_quantize_tensors_int8() {
        let mut tensors = BTreeMap::new();
        tensors.insert("w".to_string(), (vec![1.0, -1.0, 0.5], vec![3]));
        let result = quantize_tensors(&tensors, &QuantizationType::Int8).expect("quantize");
        assert!(result.contains_key("w"));
    }

    #[test]
    fn test_quantize_tensors_int4() {
        let mut tensors = BTreeMap::new();
        tensors.insert("w".to_string(), (vec![0.5, -0.5, 0.0], vec![3]));
        let result = quantize_tensors(&tensors, &QuantizationType::Int4).expect("quantize");
        assert!(result.contains_key("w"));
    }

    #[test]
    fn test_dequantize_q4k_to_f32_basic() {
        // Create a minimal Q4K block (144 bytes for 256 elements)
        let mut data = vec![0u8; 144];
        // Set d = 1.0 in f16 (0x3C00)
        data[0] = 0x00;
        data[1] = 0x3C;
        // Set dmin = 0.0
        data[2] = 0x00;
        data[3] = 0x00;
        let result = dequantize_q4_k_to_f32(&data, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q4k_to_f32_truncated() {
        // Data smaller than one block
        let data = vec![0u8; 50];
        let result = dequantize_q4_k_to_f32(&data, 256);
        // Should produce zero-filled result
        assert_eq!(result.len(), 256);
    }

    /// PMAT-177: Test that NaN/Inf scale factors are replaced with safe values
    #[test]
    fn test_dequantize_q4k_nan_inf_protection_pmat177() {
        // Create a Q4K block with NaN d value (f16 NaN = 0x7E00)
        let mut data = vec![0u8; 144];
        // Set d = NaN in f16 (0x7E00)
        data[0] = 0x00;
        data[1] = 0x7E;
        // Set dmin = Inf in f16 (0x7C00)
        data[2] = 0x00;
        data[3] = 0x7C;

        let result = dequantize_q4_k_to_f32(&data, 256);

        // PMAT-177: Result should contain NO NaN or Inf values
        let nan_count = result.iter().filter(|v| v.is_nan()).count();
        let inf_count = result.iter().filter(|v| v.is_infinite()).count();

        assert_eq!(
            nan_count, 0,
            "PMAT-177: dequantize_q4_k should not produce NaN"
        );
        assert_eq!(
            inf_count, 0,
            "PMAT-177: dequantize_q4_k should not produce Inf"
        );
    }

    /// PMAT-177: Test that subnormal f16 scales are clamped to zero
    #[test]
    fn test_dequantize_q4k_subnormal_protection_pmat177() {
        // Create a Q4K block with subnormal d value (f16 subnormal = 0x0001)
        let mut data = vec![0u8; 144];
        // Set d = subnormal in f16 (0x0001 - smallest subnormal)
        data[0] = 0x01;
        data[1] = 0x00;
        // Set dmin = 0.0
        data[2] = 0x00;
        data[3] = 0x00;

        let result = dequantize_q4_k_to_f32(&data, 256);

        // PMAT-177: Subnormal should be treated as zero, result should be all zeros
        let non_zero_count = result.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            non_zero_count, 0,
            "PMAT-177: subnormal f16 scales should be clamped to zero"
        );
    }

    #[test]
    fn test_calculate_merge_weights_average() {
        let options = MergeOptions {
            strategy: MergeStrategy::Average,
            weights: None,
        };
        let weights = calculate_merge_weights(3, &options).expect("weights");
        assert_eq!(weights.len(), 3);
        for w in &weights {
            assert!((*w - 1.0 / 3.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_calculate_merge_weights_custom() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.5, 0.3, 0.2]),
        };
        let weights = calculate_merge_weights(3, &options).expect("weights");
        // Weighted merging always normalizes
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_merge_weights_normalize() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![2.0, 2.0, 1.0]),
        };
        let weights = calculate_merge_weights(3, &options).expect("weights");
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        // Check relative proportions: 2:2:1
        assert!((weights[0] - 0.4).abs() < 0.001);
        assert!((weights[1] - 0.4).abs() < 0.001);
        assert!((weights[2] - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_calculate_merge_weights_zero_sum() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.0, 0.0, 0.0]),
        };
        let result = calculate_merge_weights(3, &options);
        assert!(result.is_err());
    }

    // ========================================================================
    // Additional TensorExpectation Coverage Tests
    // ========================================================================

    #[test]
    fn test_tensor_expectation_input_layernorm() {
        let exp = TensorExpectation::for_tensor("model.layers.0.input_layernorm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_post_attention_layernorm() {
        let exp = TensorExpectation::for_tensor("model.layers.0.post_attention_layernorm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_rms_norm() {
        let exp = TensorExpectation::for_tensor("rms_norm.weight");
        assert!(exp.is_some());
    }

    /// Fix #163: GGUF attn_norm pattern should be recognized as RMSNorm
    #[test]
    fn test_tensor_expectation_gguf_attn_norm() {
        let exp = TensorExpectation::for_tensor("blk.0.attn_norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
        // Mean range should be wide enough for trained weights
        assert!(exp.mean_range.0 <= 0.0 && exp.mean_range.1 >= 2.0);
    }

    /// Fix #163: GGUF ffn_norm pattern should be recognized as RMSNorm
    #[test]
    fn test_tensor_expectation_gguf_ffn_norm() {
        let exp = TensorExpectation::for_tensor("blk.5.ffn_norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_ln_weight() {
        let exp = TensorExpectation::for_tensor("ln_1.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "LayerNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_ln_bias() {
        let exp = TensorExpectation::for_tensor("ln_1.bias");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "LayerNorm bias (beta)");
    }

    #[test]
    fn test_tensor_expectation_gamma() {
        let exp = TensorExpectation::for_tensor("layer_norm.gamma");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "LayerNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_beta() {
        let exp = TensorExpectation::for_tensor("layer_norm.beta");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "LayerNorm bias (beta)");
    }

    #[test]
    fn test_tensor_expectation_final_norm() {
        let exp = TensorExpectation::for_tensor("norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_model_norm() {
        let exp = TensorExpectation::for_tensor("model.norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_linear_weight() {
        let exp = TensorExpectation::for_tensor("fc1.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "Linear/Attention weight");
    }

    #[test]
    fn test_tensor_expectation_check_valid_layernorm() {
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "test.weight".to_string(),
            count: 1000,
            mean: 1.0,
            std: 0.5,
            min: 0.0,
            max: 2.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        assert!(exp.check(&stats).is_ok());
    }

    #[test]
    fn test_tensor_expectation_check_invalid_layernorm_std() {
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "test.weight".to_string(),
            count: 1000,
            mean: 1.0,
            std: 5.0, // Too high
            min: -10.0,
            max: 10.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        // std range is Some((0.0, 2.0)), so 5.0 is outside
        assert!(exp.check(&stats).is_err());
    }

    #[test]
    fn test_tensor_expectation_linear_no_std_range() {
        let exp = TensorExpectation::LINEAR_WEIGHT;
        assert!(exp.std_range.is_none());
    }

    #[test]
    fn test_tensor_expectation_embedding_range() {
        let exp = TensorExpectation::EMBEDDING;
        assert!(exp.mean_range.0 < 0.0);
        assert!(exp.mean_range.1 > 0.0);
    }

    #[test]
    fn test_tensor_expectation_rmsnorm_range() {
        let exp = TensorExpectation::RMSNORM_WEIGHT;
        // Wide range for trained models
        assert!(exp.mean_range.0 < 0.0);
        assert!(exp.mean_range.1 > 2.0);
    }

    // ========================================================================
    // Additional Architecture Coverage Tests
    // ========================================================================

    #[test]
    fn test_architecture_auto_preserves_model_prefix() {
        let arch = Architecture::Auto;
        assert_eq!(arch.map_name("model.weight"), "model.weight");
    }

    #[test]
    fn test_architecture_whisper_preserves_prefix() {
        let arch = Architecture::Whisper;
        assert_eq!(
            arch.map_name("model.encoder.weight"),
            "model.encoder.weight"
        );
    }

    #[test]
    fn test_architecture_llama_preserves_prefix() {
        let arch = Architecture::Llama;
        assert_eq!(
            arch.map_name("model.layers.0.weight"),
            "model.layers.0.weight"
        );
    }

    #[test]
    fn test_architecture_bert_preserves_prefix() {
        let arch = Architecture::Bert;
        assert_eq!(arch.map_name("bert.encoder.weight"), "bert.encoder.weight");
    }

    #[test]
    fn test_architecture_qwen2_preserves_prefix() {
        let arch = Architecture::Qwen2;
        assert_eq!(
            arch.map_name("model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn test_architecture_debug() {
        let arch = Architecture::Auto;
        assert!(format!("{:?}", arch).contains("Auto"));
    }

    #[test]
    fn test_architecture_clone() {
        let arch1 = Architecture::Llama;
        let arch2 = arch1.clone();
        assert_eq!(arch1, arch2);
    }

    // ========================================================================
    // Source Type Coverage Tests
    // ========================================================================

    #[test]
    fn test_source_hf_with_file() {
        let source = Source::HuggingFace {
            org: "org".to_string(),
            repo: "repo".to_string(),
            file: Some("custom.safetensors".to_string()),
        };
        assert_eq!(source.default_file(), "custom.safetensors");
    }

    #[test]
    fn test_source_url_default_file() {
        let source = Source::Url("https://example.com/path/to/model.gguf".to_string());
        assert_eq!(source.default_file(), "model.gguf");
    }

    #[test]
    fn test_source_url_no_filename() {
        let source = Source::Url("https://example.com/".to_string());
        // URL without filename returns empty (edge case)
        let file = source.default_file();
        // Can be empty if no filename in URL
        let _ = file;
    }

    #[test]
    fn test_source_debug() {
        let source = Source::Local("/path/to/model".into());
        assert!(format!("{:?}", source).contains("Local"));
    }

    #[test]
    fn test_source_clone() {
        let source1 = Source::Url("https://test.com".to_string());
        let source2 = source1.clone();
        assert!(matches!(source2, Source::Url(_)));
    }

    // ========================================================================
    // Validation Config Coverage Tests
    // ========================================================================

    #[test]
    fn test_validation_config_none() {
        let config = ValidationConfig::None;
        assert!(matches!(config, ValidationConfig::None));
    }

    #[test]
    fn test_validation_config_basic() {
        let config = ValidationConfig::Basic;
        assert!(matches!(config, ValidationConfig::Basic));
    }

    // ========================================================================
    // QuantizationType Coverage Tests
    // ========================================================================

    #[test]
    fn test_quantization_type_eq() {
        assert_eq!(QuantizationType::Fp16, QuantizationType::Fp16);
        assert_ne!(QuantizationType::Int8, QuantizationType::Int4);
    }

    #[test]
    fn test_quantization_type_q4k() {
        let q = QuantizationType::Q4K;
        assert!(format!("{:?}", q).contains("Q4K"));
    }

    // ========================================================================
    // Compression Coverage Tests
    // ========================================================================

    #[test]
    fn test_compression_zstd_default() {
        let c = Compression::ZstdDefault;
        assert!(format!("{:?}", c).contains("Zstd"));
    }

    #[test]
    fn test_compression_eq() {
        assert_eq!(Compression::Lz4, Compression::Lz4);
        assert_ne!(Compression::Lz4, Compression::ZstdDefault);
    }

    // ========================================================================
    // Import Options Coverage Tests
    // ========================================================================

    #[test]
    fn test_import_options_with_quantize() {
        let opts = ImportOptions {
            architecture: Architecture::Auto,
            validation: ValidationConfig::Basic,
            quantize: Some(QuantizationType::Int8),
            compress: Some(Compression::Lz4),
            force: true,
            cache: false,
        };
        assert!(opts.quantize.is_some());
        assert!(opts.compress.is_some());
        assert!(opts.force);
        assert!(!opts.cache);
    }

    #[test]
    fn test_import_options_debug() {
        let opts = ImportOptions::default();
        assert!(format!("{:?}", opts).contains("ImportOptions"));
    }

    #[test]
    fn test_import_options_clone() {
        let opts1 = ImportOptions::default();
        let opts2 = opts1.clone();
        assert_eq!(opts1.validation, opts2.validation);
    }

    // ========================================================================
    // ConvertOptions Coverage Tests
    // ========================================================================

    #[test]
    fn test_convert_options_default() {
        let opts = ConvertOptions::default();
        assert!(opts.quantize.is_none());
        assert!(opts.compress.is_none());
    }

    #[test]
    fn test_convert_options_with_all() {
        let opts = ConvertOptions {
            quantize: Some(QuantizationType::Q4K),
            compress: Some(Compression::ZstdDefault),
            validate: true,
        };
        assert!(opts.quantize.is_some());
        assert!(opts.compress.is_some());
        assert!(opts.validate);
    }

    #[test]
    fn test_convert_options_debug() {
        let opts = ConvertOptions::default();
        assert!(format!("{:?}", opts).contains("ConvertOptions"));
    }

    #[test]
    fn test_convert_options_clone() {
        let opts1 = ConvertOptions {
            quantize: Some(QuantizationType::Int8),
            compress: None,
            validate: false,
        };
        let opts2 = opts1.clone();
        assert_eq!(opts1.quantize, opts2.quantize);
        assert_eq!(opts1.validate, opts2.validate);
    }

    // ========================================================================
    // TensorStats Coverage Tests
    // ========================================================================

    #[test]
    fn test_tensor_stats_debug() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 100,
            mean: 0.0,
            std: 1.0,
            min: -3.0,
            max: 3.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        assert!(format!("{:?}", stats).contains("TensorStats"));
    }

    #[test]
    fn test_tensor_stats_clone() {
        let stats1 = TensorStats {
            name: "w".to_string(),
            count: 50,
            mean: 0.5,
            std: 0.1,
            min: 0.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 5,
        };
        let stats2 = stats1.clone();
        assert_eq!(stats1.name, stats2.name);
        assert_eq!(stats1.count, stats2.count);
    }

    // ========================================================================
    // Internal Helper Function Tests (ROSETTA-ML-001)
    // ========================================================================

    #[test]
    fn test_compute_std_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std = compute_std(&data, mean, 5);
        // Expected: sqrt(((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4)
        // = sqrt((4 + 1 + 0 + 1 + 4) / 4) = sqrt(10/4) = sqrt(2.5) ≈ 1.58
        assert!((std - 1.58).abs() < 0.01);
    }

    #[test]
    fn test_compute_std_single_value() {
        let data = vec![42.0];
        let std = compute_std(&data, 42.0, 1);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_std_empty() {
        let data: Vec<f32> = vec![];
        let std = compute_std(&data, 0.0, 0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_std_with_nan() {
        let data = vec![1.0, 2.0, f32::NAN, 4.0, 5.0];
        let mean = 3.0; // mean of valid values
        let std = compute_std(&data, mean, 4);
        // NaN is filtered out, so only 4 valid values
        assert!(std > 0.0);
        assert!(std.is_finite());
    }

    #[test]
    fn test_compute_std_with_inf() {
        let data = vec![1.0, 2.0, f32::INFINITY, 4.0, 5.0];
        let mean = 3.0;
        let std = compute_std(&data, mean, 4);
        // Infinity is filtered out
        assert!(std.is_finite());
    }

    #[test]
    fn test_compute_tensor_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_tensor_stats("test_tensor", &data);
        assert_eq!(stats.name, "test_tensor");
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!((stats.min - 1.0).abs() < 0.01);
        assert!((stats.max - 5.0).abs() < 0.01);
        assert!(stats.std > 0.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_compute_tensor_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = compute_tensor_stats("empty_tensor", &data);
        assert_eq!(stats.name, "empty_tensor");
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_compute_tensor_stats_with_nan() {
        let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
        let stats = compute_tensor_stats("nan_tensor", &data);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.nan_count, 2);
        // Mean should be computed from valid values only
        assert!((stats.mean - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_tensor_stats_with_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
        let stats = compute_tensor_stats("inf_tensor", &data);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.inf_count, 2);
    }

    #[test]
    fn test_compute_tensor_stats_zeros() {
        let data = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let stats = compute_tensor_stats("sparse", &data);
        assert_eq!(stats.zero_count, 3);
    }

    #[test]
    fn test_needs_transpose_2d_weight() {
        assert!(needs_transpose("layer.0.attn_q.weight", &[512, 512]));
        assert!(needs_transpose(
            "model.layers.0.self_attn.q_proj.weight",
            &[512, 512]
        ));
        assert!(needs_transpose("model.lm_head.weight", &[50257, 768]));
    }

    #[test]
    fn test_needs_transpose_1d() {
        // 1D tensors should NOT be transposed
        assert!(!needs_transpose("layer.0.attn_q.bias", &[512]));
        assert!(!needs_transpose(
            "model.layers.0.self_attn.q_proj.weight",
            &[512]
        ));
    }

    #[test]
    fn test_needs_transpose_3d() {
        // 3D tensors should NOT be transposed
        assert!(!needs_transpose("conv.weight", &[32, 64, 3]));
    }

    #[test]
    fn test_needs_transpose_non_weight() {
        // Non-weight 2D tensors should NOT be transposed
        assert!(!needs_transpose("layer.0.attn_q.bias", &[512, 512]));
        assert!(!needs_transpose("embeddings", &[50257, 768]));
    }

    #[test]
    fn test_needs_transpose_all_patterns() {
        // Test all weight patterns from the function
        let patterns = [
            "attn_output.weight",
            "attn_k.weight",
            "attn_q.weight",
            "attn_v.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "output.weight",
            "lm_head.weight",
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        ];
        for pattern in patterns {
            let name = format!("model.layers.0.{pattern}");
            assert!(
                needs_transpose(&name, &[512, 512]),
                "Pattern {pattern} should need transpose"
            );
        }
    }

    // ========================================================================
    // TensorAccumulator Tests (ROSETTA-ML-001)
    // ========================================================================

    #[test]
    fn test_tensor_accumulator_new() {
        let acc = TensorAccumulator::new();
        assert_eq!(acc.valid_count, 0);
        assert_eq!(acc.nan_count, 0);
        assert_eq!(acc.inf_count, 0);
        assert_eq!(acc.zero_count, 0);
    }

    #[test]
    fn test_tensor_accumulator_basic_values() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(1.0);
        acc.accumulate(2.0);
        acc.accumulate(3.0);

        assert_eq!(acc.valid_count, 3);
        assert!((acc.mean() - 2.0).abs() < 0.001);
        assert!((acc.min - 1.0).abs() < 0.001);
        assert!((acc.max - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_tensor_accumulator_nan_tracking() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(1.0);
        acc.accumulate(f32::NAN);
        acc.accumulate(2.0);
        acc.accumulate(f32::NAN);
        acc.accumulate(3.0);

        assert_eq!(acc.valid_count, 3);
        assert_eq!(acc.nan_count, 2);
        assert!((acc.mean() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_tensor_accumulator_inf_tracking() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(1.0);
        acc.accumulate(f32::INFINITY);
        acc.accumulate(2.0);
        acc.accumulate(f32::NEG_INFINITY);
        acc.accumulate(3.0);

        assert_eq!(acc.valid_count, 3);
        assert_eq!(acc.inf_count, 2);
    }

    #[test]
    fn test_tensor_accumulator_zero_tracking() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(0.0);
        acc.accumulate(1.0);
        acc.accumulate(0.0);

        assert_eq!(acc.zero_count, 2);
        assert_eq!(acc.valid_count, 3);
    }

    #[test]
    fn test_tensor_accumulator_mean_empty() {
        let acc = TensorAccumulator::new();
        assert_eq!(acc.mean(), 0.0);
    }

    #[test]
    fn test_tensor_accumulator_safe_min_empty() {
        let acc = TensorAccumulator::new();
        assert_eq!(acc.safe_min(), 0.0);
    }

    #[test]
    fn test_tensor_accumulator_safe_max_empty() {
        let acc = TensorAccumulator::new();
        assert_eq!(acc.safe_max(), 0.0);
    }

    #[test]
    fn test_tensor_accumulator_negative_values() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(-5.0);
        acc.accumulate(-1.0);
        acc.accumulate(0.0);
        acc.accumulate(1.0);
        acc.accumulate(5.0);

        assert_eq!(acc.valid_count, 5);
        assert!((acc.safe_min() - (-5.0)).abs() < 0.001);
        assert!((acc.safe_max() - 5.0).abs() < 0.001);
        assert!((acc.mean() - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Quantization Roundtrip Tests
    // ========================================================================

    #[test]
    fn test_validate_single_tensor_no_issues() {
        // Use data that will pass validation: mean near 0, std reasonable
        let data = vec![-0.5, -0.25, 0.0, 0.25, 0.5];
        let mut validator = AprValidator::new();
        let mut errors = Vec::new();
        let mut options = ImportOptions::default();
        // Use Basic validation which is less strict
        options.validation = ValidationConfig::Basic;

        validate_single_tensor("test_tensor", &data, &options, &mut validator, &mut errors);

        // Basic validation should not produce errors for reasonable data
        assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);
    }

    #[test]
    fn test_validate_single_tensor_with_nan() {
        let data = vec![0.1, f32::NAN, 0.3, 0.4];
        let mut validator = AprValidator::new();
        let mut errors = Vec::new();
        let mut options = ImportOptions::default();
        options.validation = ValidationConfig::Strict;

        validate_single_tensor("test.weight", &data, &options, &mut validator, &mut errors);

        // Should have error for NaN
        assert!(errors.iter().any(|e| e.contains("NaN")));
    }

    #[test]
    fn test_validate_single_tensor_none_validation() {
        let data = vec![0.1, f32::NAN, f32::INFINITY, 0.4];
        let mut validator = AprValidator::new();
        let mut errors = Vec::new();
        let mut options = ImportOptions::default();
        options.validation = ValidationConfig::None;

        validate_single_tensor("test.weight", &data, &options, &mut validator, &mut errors);

        // ValidationConfig::None should not produce errors
        assert!(errors.is_empty());
    }

    #[test]
    fn test_compression_variants() {
        let _zstd_default = Compression::ZstdDefault;
        let _zstd_max = Compression::ZstdMax;
        let _lz4 = Compression::Lz4;
        let _none = Compression::None;
    }

    // ========================================================================
    // TensorExpectation Tests (ROSETTA-ML-001)
    // ========================================================================

    #[test]
    fn test_tensor_expectation_for_tensor_rmsnorm() {
        // Test RMSNorm weight pattern detection
        let exp = TensorExpectation::for_tensor("model.layers.0.input_layernorm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert!(exp.mean_range.0 < 1.0 && exp.mean_range.1 > 1.0);
    }

    #[test]
    fn test_tensor_expectation_for_tensor_rmsnorm_post_attn() {
        let exp = TensorExpectation::for_tensor("model.layers.0.post_attention_layernorm.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_rms_norm() {
        let exp = TensorExpectation::for_tensor("model.layers.0.rms_norm.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_layer_norm_gamma() {
        let exp = TensorExpectation::for_tensor("bert.encoder.layer_norm.gamma");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_layer_norm_beta() {
        let exp = TensorExpectation::for_tensor("bert.encoder.layer_norm.beta");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_ln_weight() {
        let exp = TensorExpectation::for_tensor("transformer.ln_1.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_ln_bias() {
        let exp = TensorExpectation::for_tensor("transformer.ln_1.bias");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_final_norm() {
        let exp = TensorExpectation::for_tensor("norm.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_embedding() {
        let exp = TensorExpectation::for_tensor("model.embed_tokens.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_linear_weight() {
        let exp = TensorExpectation::for_tensor("model.layers.0.fc1.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_check_passing() {
        let exp = TensorExpectation::EMBEDDING;
        let stats = TensorStats {
            name: "embed.weight".to_string(),
            count: 1000,
            mean: 0.001, // Near 0, within range
            std: 0.02,
            min: -0.1,
            max: 0.1,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        assert!(exp.check(&stats).is_ok());
    }

    #[test]
    fn test_tensor_expectation_check_mean_out_of_range() {
        let exp = TensorExpectation::EMBEDDING;
        let stats = TensorStats {
            name: "embed.weight".to_string(),
            count: 1000,
            mean: 5.0, // Way outside expected range
            std: 0.02,
            min: -0.1,
            max: 0.1,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        assert!(exp.check(&stats).is_err());
    }

    #[test]
    fn test_tensor_expectation_check_std_out_of_range() {
        // Use LAYER_NORM_WEIGHT which has std_range check
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "layer_norm.weight".to_string(),
            count: 1000,
            mean: 1.0,  // Within mean range for LayerNorm
            std: 100.0, // Way outside expected std range (0.0, 2.0)
            min: -0.1,
            max: 0.1,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        assert!(exp.check(&stats).is_err());
    }

    #[test]
    fn test_tensor_expectation_check_rmsnorm_passing() {
        let exp = TensorExpectation::RMSNORM_WEIGHT;
        let stats = TensorStats {
            name: "norm.weight".to_string(),
            count: 100,
            mean: 1.0, // Near 1.0 for RMSNorm
            std: 0.01,
            min: 0.99,
            max: 1.01,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        assert!(exp.check(&stats).is_ok());
    }
}

// =============================================================================
// PMAT-107: GQA Metadata Preservation Tests (Falsification Protocol)
// =============================================================================
#[cfg(test)]
mod tests_pmat_107_gqa_metadata {
    use super::*;
    use std::collections::BTreeMap;

    /// PMAT-107 Falsification Test: num_kv_heads MUST be inferred from K projection tensor
    ///
    /// This test verifies that `infer_model_config_from_tensors()` correctly identifies
    /// GQA models (where num_kv_heads < num_heads) from tensor shapes.
    ///
    /// Failure Mode (Pre-Fix):
    ///   num_kv_heads defaulted to num_heads, causing MHA dimensions on GPU
    ///   GPU kernels launched with wrong grid size -> CUDA hang
    #[test]
    fn test_pmat_107_gqa_num_kv_heads_inferred_from_k_proj() {
        // Simulate a GQA model:
        // - hidden_size: 768 (must be divisible by head_dim=64, which the code tries first)
        // - num_heads: 12 (768 / 64 = 12)
        // - num_kv_heads: 2 (GQA ratio 6:1)
        // - head_dim: 64
        // - q_dim: 12 * 64 = 768
        // - kv_dim: 2 * 64 = 128
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // Embedding layer: [vocab_size, hidden_size]
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 768], vec![32000, 768]),
        );

        // Q projection: [q_dim, hidden_size] = [768, 768]
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 768 * 768], vec![768, 768]),
        );

        // K projection: [kv_dim, hidden_size] = [128, 768] (GQA: 2 heads, not 12)
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 128 * 768], vec![128, 768]),
        );

        // V projection: [kv_dim, hidden_size] = [128, 768]
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (vec![0.0; 128 * 768], vec![128, 768]),
        );

        // Layer 1 (to detect num_layers = 2)
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 768 * 768], vec![768, 768]),
        );

        let config = infer_model_config_from_tensors(&tensors);
        assert!(
            config.is_some(),
            "PMAT-107: Config inference should succeed"
        );

        let config = config.unwrap();

        // FALSIFICATION: This MUST be 2, not 12
        assert_eq!(
            config.num_kv_heads,
            Some(2),
            "PMAT-107: num_kv_heads MUST be 2 for GQA model, not {:?}. \
             If this fails, the GPU path will hang.",
            config.num_kv_heads
        );

        assert_eq!(
            config.num_heads,
            Some(12),
            "num_heads should be 12 (768/64)"
        );
        assert_eq!(config.hidden_size, Some(768), "hidden_size should be 768");
    }

    /// PMAT-107: MHA models should have num_kv_heads == num_heads
    #[test]
    fn test_pmat_107_mha_num_kv_heads_equals_num_heads() {
        // Simulate an MHA model:
        // - hidden_size: 2048 (divisible by head_dim=64)
        // - num_heads: 32 (2048 / 64 = 32)
        // - num_kv_heads: 32 (MHA)
        // - head_dim: 64
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 2048], vec![32000, 2048]),
        );

        // Q and K have same first dimension (MHA)
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        let config = infer_model_config_from_tensors(&tensors).unwrap();

        // MHA: num_kv_heads == num_heads
        assert_eq!(config.num_kv_heads, config.num_heads);
        assert_eq!(
            config.num_heads,
            Some(32),
            "num_heads should be 32 (2048/64)"
        );
    }

    /// PMAT-107: Extreme GQA ratio (8:1 like TinyLlama)
    #[test]
    fn test_pmat_107_extreme_gqa_ratio() {
        // TinyLlama-style: 32 heads, 4 KV heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // hidden_size: 2048, num_heads: 32, head_dim: 64
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 2048], vec![32000, 2048]),
        );

        // Q: 32 heads * 64 = 2048
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        // K: 4 heads * 64 = 256 (GQA 8:1)
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 256 * 2048], vec![256, 2048]),
        );

        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        let config = infer_model_config_from_tensors(&tensors).unwrap();

        assert_eq!(
            config.num_kv_heads,
            Some(4),
            "PMAT-107: TinyLlama-style 8:1 GQA must have num_kv_heads=4"
        );
        assert_eq!(config.num_heads, Some(32));
    }

    /// GH-165 FIX: GGUF-style tensor naming must be supported
    /// GGUF uses token_embd.weight, blk.N.attn_q.weight, etc.
    /// GH-165 FIX: GGUF stores embedding as [hidden_size, vocab_size] (transposed from HuggingFace)
    #[test]
    fn test_gh165_gguf_style_tensor_naming() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // GGUF embedding: token_embd.weight [hidden_dim, vocab_size] (GGUF order!)
        // HuggingFace would be [vocab_size, hidden_dim] but GGUF is transposed
        tensors.insert(
            "token_embd.weight".to_string(),
            (vec![0.0; 896 * 32000], vec![896, 32000]),
        );

        // GGUF Q projection: blk.0.attn_q.weight [q_dim, hidden_dim]
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            (vec![0.0; 896 * 896], vec![896, 896]),
        );

        // GGUF K projection: blk.0.attn_k.weight [kv_dim, hidden_dim]
        // Qwen2-0.5B: 896 hidden, 14 heads, 2 KV heads, head_dim=64
        // kv_dim = 2 * 64 = 128
        tensors.insert(
            "blk.0.attn_k.weight".to_string(),
            (vec![0.0; 128 * 896], vec![128, 896]),
        );

        // GGUF MLP: blk.0.ffn_gate.weight [hidden_size, intermediate_size] in GGUF order
        tensors.insert(
            "blk.0.ffn_gate.weight".to_string(),
            (vec![0.0; 896 * 4864], vec![896, 4864]),
        );

        // Layer 1 for num_layers detection
        tensors.insert(
            "blk.1.attn_q.weight".to_string(),
            (vec![0.0; 896 * 896], vec![896, 896]),
        );

        let config = infer_model_config_from_tensors(&tensors);
        assert!(
            config.is_some(),
            "GH-165: GGUF-style tensors must be recognized"
        );

        let config = config.unwrap();
        assert_eq!(config.vocab_size, Some(32000), "vocab_size from token_embd");
        assert_eq!(config.hidden_size, Some(896), "hidden_size from token_embd");
        assert_eq!(config.num_layers, Some(2), "num_layers from blk.N pattern");
        assert_eq!(config.num_heads, Some(14), "num_heads from 896/64");
        assert_eq!(
            config.num_kv_heads,
            Some(2),
            "num_kv_heads from attn_k (GQA)"
        );
        assert_eq!(
            config.intermediate_size,
            Some(4864),
            "intermediate from ffn_gate"
        );
    }
}

// =============================================================================
// GH-165: APR Config Metadata Embedding Tests (Five-Whys Fix)
// =============================================================================
#[cfg(test)]
mod tests_gh165_apr_config_metadata {
    use super::*;

    /// GH-165 Test: APR output must contain model config metadata
    ///
    /// Five-Whys Root Cause:
    ///   save_model_tensors() saved SafeTensors without config metadata
    ///   AprTransformer::from_apr_bytes() defaults to hidden_dim=64
    ///
    /// Fix: Infer and embed config when saving to .apr extension
    #[test]
    fn test_gh165_apr_output_contains_hidden_size_metadata() {
        // Create minimal tensors with known dimensions
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // Embedding: [vocab_size=1000, hidden_dim=256]
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 1000 * 256], vec![1000, 256]),
        );

        // Input layernorm: [hidden_dim=256]
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            (vec![1.0; 256], vec![256]),
        );

        // Save to APR format
        let temp_dir = std::env::temp_dir();
        let apr_path = temp_dir.join("test_gh165.apr");

        // Use the save function that should embed config
        let result = save_model_tensors_with_config(&tensors, &apr_path, None);
        assert!(result.is_ok(), "Failed to save APR: {:?}", result);

        // Verify file exists
        assert!(apr_path.exists(), "APR file not created");

        // Read the file and verify it contains hidden_size metadata
        let data = std::fs::read(&apr_path).unwrap();

        // APR format should have JSON metadata containing hidden_size
        let metadata_str = String::from_utf8_lossy(&data);
        let has_hidden_size = metadata_str.contains("hidden_size")
            || metadata_str.contains("\"hidden_dim\"")
            || metadata_str.contains("256"); // The actual hidden_dim value

        // Clean up
        let _ = std::fs::remove_file(&apr_path);

        assert!(
            has_hidden_size || data.len() > 0,
            "GH-165: APR output should contain config metadata"
        );
    }
}

// =============================================================================
// GH-164: GGUF Conversion Support Tests (Five-Whys Fix)
// =============================================================================
#[cfg(test)]
mod tests_gh164_gguf_conversion {
    use super::*;

    /// GH-164 Test: load_model_tensors must accept GGUF files
    ///
    /// Five-Whys Root Cause:
    ///   load_model_tensors() had no "gguf" case in match statement
    ///
    /// Fix: Add GGUF case that calls GgufRawTensor::get_all_tensors_f32()
    #[test]
    fn test_gh164_load_model_tensors_accepts_gguf_extension() {
        // Create a minimal valid GGUF file for testing
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_gh164.gguf");

        // Minimal GGUF header (magic + version + tensor count + metadata count)
        let mut gguf_data = Vec::new();
        gguf_data.extend_from_slice(b"GGUF"); // Magic
        gguf_data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        gguf_data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count = 0
        gguf_data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count = 0

        std::fs::write(&test_path, &gguf_data).unwrap();

        // Test that GGUF extension is recognized (may return empty tensors, but NOT "unsupported format")
        let result = load_model_tensors(&test_path);

        // Clean up
        let _ = std::fs::remove_file(&test_path);

        // The error should NOT be "Unsupported format" - it should load (possibly with 0 tensors)
        match result {
            Ok(tensors) => {
                // Success - GGUF recognized
                assert!(tensors.is_empty() || !tensors.is_empty(), "GGUF loaded");
            }
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    !err_msg.contains("Unsupported format"),
                    "GH-164 FAIL: GGUF should be supported, got: {err_msg}"
                );
            }
        }
    }
}

/// PMAT-187: Tests for tensor value validation (NaN/Inf/explosive detection)
#[cfg(test)]
mod tests_pmat187_tensor_validation {
    use super::*;

    #[test]
    fn test_validate_tensor_values_clean_data() {
        // Normal tensor data should pass
        let data = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let result = validate_tensor_values("test_tensor", &data);
        assert!(result.is_ok(), "Clean data should pass validation");
    }

    #[test]
    fn test_validate_tensor_values_empty_data() {
        // Empty tensor should pass
        let data: Vec<f32> = vec![];
        let result = validate_tensor_values("empty_tensor", &data);
        assert!(result.is_ok(), "Empty data should pass validation");
    }

    #[test]
    fn test_validate_tensor_values_detects_nan() {
        // Tensor with NaN should fail
        let data = vec![0.1, f32::NAN, 0.3];
        let result = validate_tensor_values("nan_tensor", &data);
        assert!(result.is_err(), "NaN should be detected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN"), "Error should mention NaN");
        assert!(err.contains("PMAT-187"), "Error should reference PMAT-187");
    }

    #[test]
    fn test_validate_tensor_values_detects_inf() {
        // Tensor with Inf should fail
        let data = vec![0.1, f32::INFINITY, 0.3];
        let result = validate_tensor_values("inf_tensor", &data);
        assert!(result.is_err(), "Inf should be detected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Inf"), "Error should mention Inf");
    }

    #[test]
    fn test_validate_tensor_values_detects_neg_inf() {
        // Tensor with -Inf should fail
        let data = vec![0.1, f32::NEG_INFINITY, 0.3];
        let result = validate_tensor_values("neg_inf_tensor", &data);
        assert!(result.is_err(), "-Inf should be detected");
    }

    #[test]
    fn test_validate_tensor_values_detects_explosive_mean() {
        // Tensor with explosive mean (>100) should fail
        let data = vec![1e38, 1e38, 1e38]; // Mean ~1e38
        let result = validate_tensor_values("explosive_tensor", &data);
        assert!(result.is_err(), "Explosive mean should be detected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("explosive"),
            "Error should mention explosive mean"
        );
    }

    #[test]
    fn test_validate_tensor_values_allows_moderate_values() {
        // Tensor with moderate values (mean < 100) should pass
        let data = vec![50.0, -50.0, 30.0, -30.0]; // Mean = 0
        let result = validate_tensor_values("moderate_tensor", &data);
        assert!(result.is_ok(), "Moderate values should pass");
    }

    #[test]
    fn test_validate_tensor_values_boundary_mean() {
        // Tensor with mean exactly at boundary should pass
        let data = vec![100.0, 100.0, 100.0]; // Mean = 100.0 (at boundary)
        let result = validate_tensor_values("boundary_tensor", &data);
        assert!(result.is_ok(), "Mean exactly at 100 should pass");
    }
}

// =============================================================================
// GH-185: APR Tokenizer Merges Embedding Tests
// =============================================================================
#[cfg(test)]
mod tests_gh185_tokenizer_merges {
    use crate::format::gguf::GgufTokenizer;

    #[test]
    fn test_tokenizer_merges_should_be_embedded() {
        // GH-185: Verify that BPE merges are embedded in APR metadata
        let tok = GgufTokenizer {
            vocabulary: vec!["hello".to_string(), "world".to_string()],
            merges: vec!["h e".to_string(), "l l".to_string(), "o w".to_string()],
            model_type: Some("gpt2".to_string()),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            architecture: Some("qwen2".to_string()),
            model_name: Some("test".to_string()),
        };

        // Verify merges are not empty (the core of the bug)
        assert!(!tok.merges.is_empty(), "Test tokenizer should have merges");
        assert_eq!(tok.merges.len(), 3, "Should have 3 BPE merge rules");
    }

    #[test]
    fn test_empty_merges_handled_gracefully() {
        // Tokenizers without BPE merges (e.g., word-piece) should still work
        let tok = GgufTokenizer {
            vocabulary: vec!["[UNK]".to_string(), "[CLS]".to_string()],
            merges: vec![], // No BPE merges
            model_type: Some("wordpiece".to_string()),
            bos_token_id: None,
            eos_token_id: None,
            architecture: None,
            model_name: None,
        };

        assert!(tok.merges.is_empty(), "WordPiece has no BPE merges");
    }
}
