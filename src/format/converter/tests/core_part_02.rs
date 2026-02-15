
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
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });

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
            allow_no_config: true,
            ..Default::default()
        };
        let result = apr_import(&input.to_string_lossy(), &output, options);

        assert!(
            result.is_err(),
            "Invalid LayerNorm should fail strict validation"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mean=11")
                || err.contains("LayerNorm")
                || err.contains("outside expected range"),
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
            allow_no_config: true,
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

        let result = apr_import(
            &input.to_string_lossy(),
            &output,
            ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            },
        );

        assert!(result.is_err(), "NaN should fail validation");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN"), "Error should mention NaN: {err}");
    }

    /// Error path: nonexistent file must produce clear error.
    #[test]
    fn test_convert_nonexistent_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let nonexistent = dir.path().join("nonexistent_model_abc123.safetensors");
        let output = dir.path().join("out.apr");
        let result = apr_import(
            &nonexistent.to_string_lossy(),
            &output,
            ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            },
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

        let result = apr_import(
            &input.to_string_lossy(),
            &output,
            ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            },
        );
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
            allow_no_config: true,
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

        let native = NativeF32Tensors::new(tensors);
        let result = quantize_tensors(&native, &QuantizationType::Int8).unwrap();
        let result = result.as_ref();

        assert_eq!(result.len(), 1);
        assert!(result.contains_key("test"));
        let (data, shape) = result.get("test").unwrap();
        assert_eq!(shape, &vec![3]);
        assert_eq!(data.len(), 3);
    }
}
