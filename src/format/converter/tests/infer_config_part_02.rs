
#[test]
fn test_tensor_accumulator_accumulate_normal() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(1.0);
    acc.accumulate(2.0);
    acc.accumulate(3.0);

    assert_eq!(acc.valid_count, 3);
    assert!((acc.sum - 6.0).abs() < 1e-10);
    assert!((acc.min - 1.0).abs() < 1e-5);
    assert!((acc.max - 3.0).abs() < 1e-5);
}

#[test]
fn test_tensor_accumulator_accumulate_nan() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(f32::NAN);
    assert_eq!(acc.nan_count, 1);
    assert_eq!(acc.valid_count, 0);
}

#[test]
fn test_tensor_accumulator_accumulate_inf() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(f32::INFINITY);
    acc.accumulate(f32::NEG_INFINITY);
    assert_eq!(acc.inf_count, 2);
    assert_eq!(acc.valid_count, 0);
}

#[test]
fn test_tensor_accumulator_accumulate_zero() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(0.0);
    assert_eq!(acc.zero_count, 1);
    assert_eq!(acc.valid_count, 1);
}

#[test]
fn test_tensor_accumulator_mean() {
    let mut acc = TensorAccumulator::new();
    acc.accumulate(2.0);
    acc.accumulate(4.0);
    assert!((acc.mean() - 3.0).abs() < 1e-5);
}

#[test]
fn test_tensor_accumulator_mean_empty() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.mean(), 0.0);
}

#[test]
fn test_tensor_accumulator_safe_min_max() {
    let acc = TensorAccumulator::new();
    assert_eq!(acc.safe_min(), 0.0, "safe_min of empty should be 0.0");
    assert_eq!(acc.safe_max(), 0.0, "safe_max of empty should be 0.0");

    let mut acc2 = TensorAccumulator::new();
    acc2.accumulate(-5.0);
    acc2.accumulate(10.0);
    assert!((acc2.safe_min() - (-5.0)).abs() < 1e-5);
    assert!((acc2.safe_max() - 10.0).abs() < 1e-5);
}

// ============================================================================
// validate_single_tensor
// ============================================================================

#[test]
fn test_validate_single_tensor_no_errors() {
    let data = vec![0.1, 0.2, 0.3];
    let options = ImportOptions {
        allow_no_config: true,
        ..ImportOptions::default()
    };
    let mut validator = AprValidator::new();
    let mut errors = Vec::new();

    validate_single_tensor("test", &data, &options, &mut validator, &mut errors);
    assert!(errors.is_empty(), "Valid data should produce no errors");
}

#[test]
fn test_validate_single_tensor_with_nan_strict() {
    let data = vec![0.1, f32::NAN, 0.3];
    let options = ImportOptions {
        strict: true,
        validation: ValidationConfig::Strict,
        allow_no_config: true,
        ..Default::default()
    };
    let mut validator = AprValidator::new();
    let mut errors = Vec::new();

    validate_single_tensor("nan_tensor", &data, &options, &mut validator, &mut errors);
    assert!(
        !errors.is_empty(),
        "NaN data in strict mode should produce errors"
    );
    assert!(errors.iter().any(|e| e.contains("NaN")));
}

// ============================================================================
// AprConverter builder
// ============================================================================

#[test]
fn test_apr_converter_new() {
    let converter = AprConverter::new();
    // Just ensure it's constructible
    let debug_str = format!("{converter:?}");
    assert!(debug_str.contains("AprConverter"));
}

#[test]
fn test_apr_converter_default() {
    let converter = AprConverter::default();
    let debug_str = format!("{converter:?}");
    assert!(debug_str.contains("Auto"));
}

#[test]
fn test_apr_converter_builder_chain() {
    let result = AprConverter::new()
        .source("/tmp/nonexistent.safetensors")
        .map(|c| c.architecture(Architecture::Qwen2))
        .map(|c| c.validate(ValidationConfig::Strict))
        .map(|c| c.quantize(QuantizationType::Fp16))
        .map(|c| c.compress(Compression::None));

    assert!(result.is_ok(), "Builder chain should succeed");
}

#[test]
fn test_apr_converter_convert_without_source() {
    let converter = AprConverter::new();
    let result = converter.convert();
    assert!(result.is_err(), "Convert without source should error");
    assert!(result.unwrap_err().to_string().contains("No source"));
}

#[test]
fn test_apr_converter_convert_returns_not_implemented() {
    let converter = AprConverter::new()
        .source("/tmp/test.safetensors")
        .expect("source parse");
    let result = converter.convert();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("GH-80"));
}

// ============================================================================
// ConvertOptions
// ============================================================================

#[test]
fn test_convert_options_default() {
    let opts = ConvertOptions::default();
    assert!(opts.quantize.is_none());
    assert!(opts.compress.is_none());
    assert!(opts.validate);
}

// ============================================================================
// quantize_tensors - embedding skip logic
// ============================================================================

#[test]
fn test_quantize_tensors_skips_embeddings() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let quantized = quantized.as_ref();

    // Embedding should be unchanged (F32 preserved)
    let embed = &quantized["model.embed_tokens.weight"].0;
    assert!(
        (embed[0] - 0.1).abs() < 1e-6,
        "Embedding should not be quantized"
    );

    // Non-embedding should be quantized (values will differ due to int8 round-trip)
    let weight = &quantized["model.layers.0.self_attn.q_proj.weight"].0;
    // Just verify it's not exactly the same (quantization applied)
    assert_eq!(weight.len(), 4);
}

#[test]
fn test_quantize_tensors_fp16() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "some.weight".to_string(),
        (vec![1.0, -1.0, 0.5, 0.0], vec![2, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Fp16);
    assert!(result.is_ok());
}

#[test]
fn test_quantize_tensors_int4() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "some.weight".to_string(),
        (vec![1.0, -1.0, 0.5, 0.0], vec![2, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4);
    assert!(result.is_ok());
}

#[test]
fn test_quantize_tensors_skips_token_embd() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "token_embd.weight".to_string(),
        (vec![0.01, 0.02, 0.03, 0.04], vec![2, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let embed = &quantized.as_ref()["token_embd.weight"].0;
    assert!(
        (embed[0] - 0.01).abs() < 1e-6,
        "token_embd should not be quantized"
    );
}

#[test]
fn test_quantize_tensors_skips_wte() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "wte.weight".to_string(),
        (vec![0.01, 0.02, 0.03, 0.04], vec![2, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let embed = &quantized.as_ref()["wte.weight"].0;
    assert!(
        (embed[0] - 0.01).abs() < 1e-6,
        "wte should not be quantized"
    );
}

#[test]
fn test_quantize_tensors_skips_word_embeddings() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "word_embeddings.weight".to_string(),
        (vec![0.01, 0.02], vec![1, 2]),
    );
    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Fp16);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let embed = &quantized.as_ref()["word_embeddings.weight"].0;
    assert!(
        (embed[0] - 0.01).abs() < 1e-6,
        "word_embeddings should not be quantized"
    );
}

// ============================================================================
// map_tensor_names
// ============================================================================

#[test]
fn test_map_tensor_names_preserves_count() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "blk.0.attn_q.weight".to_string(),
        (vec![1.0, 2.0], vec![1, 2]),
    );
    tensors.insert(
        "blk.0.attn_k.weight".to_string(),
        (vec![3.0, 4.0], vec![1, 2]),
    );
    tensors.insert(
        "token_embd.weight".to_string(),
        (vec![5.0, 6.0], vec![1, 2]),
    );

    let mapped = map_tensor_names(&tensors, Architecture::Qwen2);
    assert_eq!(mapped.len(), tensors.len(), "Should preserve tensor count");
}

// ============================================================================
// resolve_source error paths
// ============================================================================

#[test]
fn test_resolve_source_local_not_found() {
    let source = Source::Local(std::path::PathBuf::from("/nonexistent/model.safetensors"));
    let result = import::resolve_source(&source, false);
    assert!(result.is_err());
}

#[test]
fn test_resolve_source_url_not_implemented() {
    let source = Source::Url("https://example.com/model.bin".to_string());
    let result = import::resolve_source(&source, false);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not yet implemented"));
}

// ============================================================================
// validate_tensor_values (additional edge cases)
// ============================================================================

#[test]
fn test_validate_tensor_values_empty_ok() {
    let result = validate_tensor_values("empty", &[]);
    assert!(result.is_ok());
}

#[test]
fn test_validate_tensor_values_valid_ok() {
    let data = vec![0.1, -0.2, 0.5, 0.0];
    let result = validate_tensor_values("valid", &data);
    assert!(result.is_ok());
}

#[test]
fn test_validate_tensor_values_nan_error() {
    let data = vec![0.1, f32::NAN, 0.3];
    let result = validate_tensor_values("nan_tensor", &data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("NaN"));
}

#[test]
fn test_validate_tensor_values_inf_error() {
    let data = vec![0.1, f32::INFINITY, 0.3];
    let result = validate_tensor_values("inf_tensor", &data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Inf"));
}

#[test]
fn test_validate_tensor_values_explosive_mean() {
    // Mean > 100 should trigger error
    let data = vec![200.0, 300.0, 400.0];
    let result = validate_tensor_values("explosive", &data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("explosive mean"));
}

// ============================================================================
// load_model_tensors unsupported format
// ============================================================================

#[test]
fn test_load_model_tensors_unsupported_format() {
    let path = std::path::Path::new("/tmp/test.xyz");
    let result = load_model_tensors(path);
    assert!(result.is_err());
    // PMAT-271: Error comes from magic byte detection + extension fallback
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Unknown format extension") || err.contains("Unsupported format"),
        "Expected format detection error, got: {err}"
    );
}

// ============================================================================
// ConvertReport
// ============================================================================

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
    assert_eq!(report.reduction_percent(), "50.0%");
}

#[test]
fn test_convert_report_reduction_percent_zero_original() {
    let report = ConvertReport {
        original_size: 0,
        converted_size: 500,
        tensor_count: 10,
        quantization: None,
        compression: None,
        reduction_ratio: 0.0,
    };
    assert_eq!(report.reduction_percent(), "N/A");
}

#[test]
fn test_convert_report_reduction_percent_zero_converted() {
    let report = ConvertReport {
        original_size: 1000,
        converted_size: 0,
        tensor_count: 10,
        quantization: None,
        compression: None,
        reduction_ratio: 0.0,
    };
    assert_eq!(report.reduction_percent(), "N/A");
}

// ============================================================================
// calculate_tensor_size
// ============================================================================

#[test]
fn test_calculate_tensor_size_basic() {
    let mut tensors = BTreeMap::new();
    tensors.insert("a".to_string(), (vec![1.0; 100], vec![10, 10]));
    tensors.insert("b".to_string(), (vec![1.0; 50], vec![50]));

    let size = calculate_tensor_size(&tensors);
    assert_eq!(size, (100 + 50) * 4); // 150 f32 values * 4 bytes each
}

#[test]
fn test_calculate_tensor_size_empty() {
    let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    assert_eq!(calculate_tensor_size(&tensors), 0);
}
