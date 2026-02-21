
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
        let dir = tempfile::tempdir().expect("tempdir");
        let nonexistent_input = dir.path().join("nonexistent_model_abc123.safetensors");
        let nonexistent_output = dir.path().join("nonexistent_output_abc123.apr");
        let options = ConvertOptions::default();
        let result = apr_convert(&nonexistent_input, &nonexistent_output, options);

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
                allow_no_config: true,
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
        let h = ConversionTestHarness::new().with_safetensors(PygmyConfig::embedding_only());

        let result = h.try_import_to_apr(ImportOptions {
            architecture: Architecture::Auto,
            validation: ValidationConfig::Strict,
            strict: true,
            allow_no_config: true,
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
        let h = ConversionTestHarness::new().with_safetensors(PygmyConfig::default());

        let result = h.try_import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        });

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
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });

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
                allow_no_config: true,
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
