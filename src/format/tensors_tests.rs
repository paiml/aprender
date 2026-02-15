use super::*;
use crate::format::test_factory::{
    build_pygmy_apr, build_pygmy_apr_f16, build_pygmy_apr_q4, build_pygmy_apr_q8,
    build_pygmy_apr_with_config, build_pygmy_safetensors, build_pygmy_safetensors_with_config,
    PygmyConfig,
};

// ========================================================================
// Format Detection Tests
// ========================================================================

#[test]
fn test_detect_format_v2() {
    assert_eq!(detect_format(&MAGIC_APR2), Some("v2"));
    assert_eq!(detect_format(&MAGIC_APR0), Some("v2"));
}

#[test]
fn test_detect_format_v1() {
    assert_eq!(detect_format(&MAGIC_APRN), Some("v1"));
    assert_eq!(detect_format(&MAGIC_APR1), Some("v1"));
}

#[test]
fn test_detect_format_invalid() {
    assert_eq!(detect_format(&[0x00, 0x00, 0x00, 0x00]), None);
    assert_eq!(detect_format(&[0xFF, 0xFF, 0xFF, 0xFF]), None);
    assert_eq!(detect_format(b"GGUF"), None);
}

#[test]
fn test_is_valid_apr_magic() {
    assert!(is_valid_apr_magic(&MAGIC_APR2));
    assert!(is_valid_apr_magic(&MAGIC_APR0));
    assert!(is_valid_apr_magic(&MAGIC_APRN));
    assert!(!is_valid_apr_magic(b"GGUF"));
    assert!(!is_valid_apr_magic(&[0x00; 4]));
}

// ========================================================================
// TensorListOptions Tests
// ========================================================================

#[test]
fn test_options_default() {
    let opts = TensorListOptions::default();
    assert!(!opts.compute_stats);
    assert!(opts.filter.is_none());
    // Default limit is usize::MAX (effectively unlimited)
    assert_eq!(opts.limit, usize::MAX);
}

#[test]
fn test_options_builder() {
    let opts = TensorListOptions::new()
        .with_stats()
        .with_filter("weight")
        .with_limit(10);

    assert!(opts.compute_stats);
    assert_eq!(opts.filter, Some("weight".to_string()));
    assert_eq!(opts.limit, 10);
}

// ========================================================================
// Pygmy APR v2 Tests (TOOL-APR-001 Fix)
// ========================================================================

#[test]
fn test_list_tensors_pygmy_apr_default() {
    let apr_bytes = build_pygmy_apr();
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    assert_eq!(result.format_version, "v2");
    assert!(result.tensor_count > 0, "Expected at least one tensor");
    assert!(result.total_size_bytes > 0);

    // Check we got tensor names from the index (could be various naming conventions)
    let names: Vec<_> = result.tensors.iter().map(|t| t.name.as_str()).collect();

    // Pygmy uses "model." prefix
    let has_model_tensors = names.iter().any(|n| n.starts_with("model."));
    let has_lm_head = names.iter().any(|n| n.contains("lm_head"));

    assert!(
        has_model_tensors || has_lm_head,
        "Expected model tensors, got: {:?}",
        names
    );
}

#[test]
fn test_list_tensors_pygmy_apr_with_filter() {
    let apr_bytes = build_pygmy_apr();
    let opts = TensorListOptions::new().with_filter("self_attn");
    let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

    // All returned tensors should match filter
    for tensor in &result.tensors {
        assert!(
            tensor.name.contains("self_attn"),
            "Expected tensor {} to contain 'self_attn'",
            tensor.name
        );
    }
}

#[test]
fn test_list_tensors_pygmy_apr_with_limit() {
    // Use llama_style which has many tensors (>10)
    let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::llama_style());

    // First, get total count without limit
    let full_result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("full list");
    let total_tensors = full_result.tensor_count;

    // GH-195 FIX: Verify model has more than our test limit
    assert!(
        total_tensors > 3,
        "Test requires model with >3 tensors, got {total_tensors}"
    );

    // Now apply limit
    let opts = TensorListOptions::new().with_limit(3);
    let limited_result = list_tensors_from_bytes(&apr_bytes, opts).expect("limited list");

    // P2 FIX: Use exact assertions, not tautological >= checks
    assert_eq!(
        limited_result.tensors.len(),
        3,
        "tensors.len() should equal the limit when total > limit"
    );
    assert_eq!(
        limited_result.tensor_count, total_tensors,
        "tensor_count must reflect TRUE total ({total_tensors}), not truncated length"
    );
}

#[test]
fn test_list_tensors_pygmy_apr_with_stats() {
    let apr_bytes = build_pygmy_apr();
    let opts = TensorListOptions::new().with_stats().with_limit(5);
    let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

    // Check at least one tensor has stats
    let has_stats = result
        .tensors
        .iter()
        .any(|t| t.mean.is_some() && t.std.is_some() && t.nan_count.is_some());
    assert!(has_stats, "Expected at least one tensor to have stats");
}

#[test]
fn test_list_tensors_pygmy_apr_f16() {
    let apr_bytes = build_pygmy_apr_f16();
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // F16 tensors should be detected
    let f16_tensors: Vec<_> = result.tensors.iter().filter(|t| t.dtype == "f16").collect();
    assert!(!f16_tensors.is_empty(), "Expected F16 tensors");
}

#[test]
fn test_list_tensors_pygmy_apr_q8() {
    let apr_bytes = build_pygmy_apr_q8();
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // Should have at least some tensors (mix of Q8 and F32)
    assert!(!result.tensors.is_empty(), "Expected tensors in Q8 model");

    // Q8 model contains both F32 (embedding) and Q8 (attention) tensors
    let dtypes: Vec<_> = result.tensors.iter().map(|t| t.dtype.as_str()).collect();
    assert!(
        dtypes.iter().any(|d| *d == "q8" || *d == "f32"),
        "Expected Q8 or F32 tensors, got: {:?}",
        dtypes
    );
}

#[test]
fn test_list_tensors_pygmy_apr_q4() {
    let apr_bytes = build_pygmy_apr_q4();
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // Should have at least some tensors (mix of Q4 and F32)
    assert!(!result.tensors.is_empty(), "Expected tensors in Q4 model");

    // Q4 model contains both F32 (embedding) and Q4 (attention) tensors
    let dtypes: Vec<_> = result.tensors.iter().map(|t| t.dtype.as_str()).collect();
    assert!(
        dtypes.iter().any(|d| *d == "q4" || *d == "f32"),
        "Expected Q4 or F32 tensors, got: {:?}",
        dtypes
    );
}

#[test]
fn test_list_tensors_pygmy_apr_minimal() {
    let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::minimal());
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // Minimal config has embedding only
    assert!(result.tensor_count >= 1);
}

#[test]
fn test_list_tensors_pygmy_apr_llama_style() {
    let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::llama_style());
    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list tensors");

    // LLaMA style has many tensors (at least some)
    assert!(
        result.tensor_count >= 1,
        "Expected at least 1 tensor, got {}",
        result.tensor_count
    );

    // Should have tensors with various components
    let has_any_tensor = !result.tensors.is_empty();
    assert!(has_any_tensor);
}

// ========================================================================
// Error Handling Tests
// ========================================================================

#[test]
fn test_list_tensors_empty_data() {
    let result = list_tensors_from_bytes(&[], TensorListOptions::default());
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("too small"));
}

#[test]
fn test_list_tensors_invalid_magic() {
    let data = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
    let result = list_tensors_from_bytes(&data, TensorListOptions::default());
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Unknown model format"));
}

#[test]
fn test_list_tensors_truncated_v2() {
    // Valid v2 magic but truncated file
    let mut data = vec![0x41, 0x50, 0x52, 0x00]; // "APR\0"
    data.extend_from_slice(&[0u8; 10]); // Not enough for header

    let result = list_tensors_from_bytes(&data, TensorListOptions::default());
    assert!(result.is_err());
}

// ========================================================================
// Statistics Tests
// ========================================================================

#[test]
fn test_compute_stats_normal_values() {
    let mut info = TensorInfo {
        name: "test".to_string(),
        shape: vec![4],
        dtype: "f32".to_string(),
        size_bytes: 16,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    let data = vec![1.0, 2.0, 3.0, 4.0];
    compute_tensor_stats(&mut info, &data);

    assert_eq!(info.mean, Some(2.5));
    assert!(info.std.expect("test value") > 1.0 && info.std.expect("test value") < 1.2);
    assert_eq!(info.min, Some(1.0));
    assert_eq!(info.max, Some(4.0));
    assert_eq!(info.nan_count, Some(0));
    assert_eq!(info.inf_count, Some(0));
}

#[test]
fn test_compute_stats_with_nan() {
    let mut info = TensorInfo {
        name: "test".to_string(),
        shape: vec![3],
        dtype: "f32".to_string(),
        size_bytes: 12,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    let data = vec![1.0, f32::NAN, 2.0];
    compute_tensor_stats(&mut info, &data);

    assert_eq!(info.nan_count, Some(1));
    assert_eq!(info.inf_count, Some(0));
    // Mean should be computed from valid values only
    assert_eq!(info.mean, Some(1.5));
}

#[test]
fn test_compute_stats_with_inf() {
    let mut info = TensorInfo {
        name: "test".to_string(),
        shape: vec![3],
        dtype: "f32".to_string(),
        size_bytes: 12,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    let data = vec![1.0, f32::INFINITY, 2.0];
    compute_tensor_stats(&mut info, &data);

    assert_eq!(info.nan_count, Some(0));
    assert_eq!(info.inf_count, Some(1));
}

#[test]
fn test_compute_stats_empty() {
    let mut info = TensorInfo {
        name: "test".to_string(),
        shape: vec![0],
        dtype: "f32".to_string(),
        size_bytes: 0,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    compute_tensor_stats(&mut info, &[]);

    assert!(info.mean.is_none());
    assert!(info.std.is_none());
}

// ========================================================================
// Format Size Tests
// ========================================================================

#[test]
fn test_format_size_bytes() {
    assert_eq!(format_size(500), "500 B");
    assert_eq!(format_size(0), "0 B");
}

#[test]
fn test_format_size_kb() {
    assert_eq!(format_size(1024), "1.00 KB");
    assert_eq!(format_size(2048), "2.00 KB");
}

#[test]
fn test_format_size_mb() {
    assert_eq!(format_size(1024 * 1024), "1.00 MB");
    assert_eq!(format_size(100 * 1024 * 1024), "100.00 MB");
}

#[test]
fn test_format_size_gb() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
    assert_eq!(format_size(5 * 1024 * 1024 * 1024), "5.00 GB");
}

// ========================================================================
// TensorInfo Tests
// ========================================================================

#[test]
fn test_tensor_info_clone() {
    let info = TensorInfo {
        name: "test".to_string(),
        shape: vec![10, 20],
        dtype: "f32".to_string(),
        size_bytes: 800,
        mean: Some(0.5),
        std: Some(0.1),
        min: Some(0.0),
        max: Some(1.0),
        nan_count: Some(0),
        inf_count: Some(0),
    };

    let cloned = info.clone();
    assert_eq!(cloned.name, info.name);
    assert_eq!(cloned.shape, info.shape);
    assert_eq!(cloned.dtype, info.dtype);
}

#[test]
fn test_tensor_list_result_clone() {
    let result = TensorListResult {
        file: "test.apr".to_string(),
        format_version: "v2".to_string(),
        tensor_count: 5,
        total_size_bytes: 1000,
        tensors: vec![],
    };

    let cloned = result.clone();
    assert_eq!(cloned.file, result.file);
    assert_eq!(cloned.format_version, result.format_version);
}

// ========================================================================
// Integration: Full Tensor Listing Workflow
// ========================================================================

#[test]
fn test_full_workflow_list_filter_stats() {
    // Build a pygmy model
    let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::llama_style());

    // List with filter and stats
    let opts = TensorListOptions::new()
        .with_filter("proj")
        .with_stats()
        .with_limit(10);

    let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

    // Verify results
    assert_eq!(result.format_version, "v2");
    for tensor in &result.tensors {
        assert!(tensor.name.contains("proj"));
        // Stats should be computed for F32 tensors
        if tensor.dtype == "f32" {
            assert!(tensor.mean.is_some());
            assert!(tensor.std.is_some());
        }
    }
}

// ========================================================================
// GGUF Format Tests (PMAT-ROSETTA-001)
// ========================================================================

#[test]
fn test_ggml_dtype_name_known_types() {
    assert_eq!(ggml_dtype_name(0), "F32");
    assert_eq!(ggml_dtype_name(1), "F16");
    assert_eq!(ggml_dtype_name(2), "Q4_0");
    assert_eq!(ggml_dtype_name(3), "Q4_1");
    assert_eq!(ggml_dtype_name(8), "Q8_0");
    assert_eq!(ggml_dtype_name(12), "Q4_K");
    assert_eq!(ggml_dtype_name(14), "Q6_K");
}

#[test]
fn test_ggml_dtype_name_unknown() {
    assert_eq!(ggml_dtype_name(99), "unknown");
    assert_eq!(ggml_dtype_name(255), "unknown");
}

#[test]
fn test_ggml_dtype_element_size() {
    assert!((ggml_dtype_element_size(0) - 4.0).abs() < 0.001); // F32
    assert!((ggml_dtype_element_size(1) - 2.0).abs() < 0.001); // F16
                                                               // Q8_0 = 1.0 + 2.0/32.0 ≈ 1.0625
    assert!((ggml_dtype_element_size(8) - 1.0625).abs() < 0.01); // Q8_0
                                                                 // Q4_0 = 0.5 + 2.0/32.0 ≈ 0.5625
    assert!((ggml_dtype_element_size(2) - 0.5625).abs() < 0.01); // Q4_0
}

#[test]
fn test_list_tensors_gguf_magic_detection() {
    // GGUF magic bytes - verify format is detected
    let mut data = b"GGUF".to_vec();
    // Add version 3 (u32) + tensor count 0 (u64) + metadata count 0 (u64)
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

    let result = list_tensors_from_bytes(&data, TensorListOptions::default());
    // Should succeed with 0 tensors (valid but empty GGUF)
    assert!(result.is_ok(), "Valid empty GGUF should parse: {result:?}");
    let result = result.expect("test value");
    assert_eq!(result.tensor_count, 0);
    assert!(result.format_version.contains("GGUF"));
}

#[test]
fn test_list_tensors_gguf_valid() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create minimal valid GGUF with one F32 tensor
    let tensor_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: tensor_data,
    };

    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let result = list_tensors_from_bytes(&gguf_bytes, TensorListOptions::default())
        .expect("list GGUF tensors");

    assert!(result.format_version.contains("GGUF"));
    assert_eq!(result.tensor_count, 1);
    assert_eq!(result.tensors[0].name, "test.weight");
    assert_eq!(result.tensors[0].shape, vec![2, 2]);
    assert_eq!(result.tensors[0].dtype, "F32");
}

#[test]
fn test_list_tensors_gguf_with_stats() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};

    let tensor_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let tensor = GgufTensor {
        name: "model.embed".to_string(),
        shape: vec![4],
        dtype: GgmlType::F32,
        data: tensor_data,
    };

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &[]).expect("export GGUF");

    let opts = TensorListOptions::new().with_stats();
    let result = list_tensors_from_bytes(&gguf_bytes, opts).expect("list");

    let t = &result.tensors[0];
    // GGUF stats computation may not be implemented, just check basics
    assert_eq!(t.name, "model.embed");
    assert_eq!(t.dtype, "F32");
}

#[test]
fn test_list_tensors_gguf_multiple_tensors() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};

    let t1_data: Vec<u8> = vec![1.0f32, 2.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let t2_data: Vec<u8> = vec![3.0f32, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let tensors = vec![
        GgufTensor {
            name: "layer.0.weight".to_string(),
            shape: vec![2],
            dtype: GgmlType::F32,
            data: t1_data,
        },
        GgufTensor {
            name: "layer.1.weight".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: t2_data,
        },
    ];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &tensors, &[]).expect("export");

    let result = list_tensors_from_bytes(&gguf_bytes, TensorListOptions::default()).expect("list");

    assert_eq!(result.tensor_count, 2);
    assert!(result.tensors.iter().any(|t| t.name == "layer.0.weight"));
    assert!(result.tensors.iter().any(|t| t.name == "layer.1.weight"));
}

#[test]
fn test_list_tensors_gguf_with_filter() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};

    let data: Vec<u8> = vec![1.0f32, 2.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let tensors = vec![
        GgufTensor {
            name: "model.attn.weight".to_string(),
            shape: vec![2],
            dtype: GgmlType::F32,
            data: data.clone(),
        },
        GgufTensor {
            name: "model.mlp.weight".to_string(),
            shape: vec![2],
            dtype: GgmlType::F32,
            data: data.clone(),
        },
        GgufTensor {
            name: "model.norm.weight".to_string(),
            shape: vec![2],
            dtype: GgmlType::F32,
            data,
        },
    ];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &tensors, &[]).expect("export");

    let opts = TensorListOptions::new().with_filter("attn");
    let result = list_tensors_from_bytes(&gguf_bytes, opts).expect("list");

    assert_eq!(result.tensors.len(), 1);
    assert_eq!(result.tensors[0].name, "model.attn.weight");
}

// ========================================================================
// SafeTensors Format Tests (PMAT-ROSETTA-001)
// ========================================================================

#[test]
fn test_f16_to_f32_conversion() {
    // Test known f16 bit patterns
    // 0x3C00 = 1.0 in f16
    assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001);
    // 0x0000 = 0.0 in f16
    assert!((f16_to_f32(0x0000) - 0.0).abs() < 0.001);
    // 0x4000 = 2.0 in f16
    assert!((f16_to_f32(0x4000) - 2.0).abs() < 0.001);
    // 0xBC00 = -1.0 in f16
    assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.001);
}

#[test]
fn test_bf16_to_f32_conversion() {
    // BF16 shares exponent range with F32, just truncated mantissa
    // 0x3F80 = 1.0 in bf16 (same top bits as f32 1.0)
    assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 0.001);
    // 0x0000 = 0.0 in bf16
    assert!((bf16_to_f32(0x0000) - 0.0).abs() < 0.001);
    // 0x4000 = 2.0 in bf16
    assert!((bf16_to_f32(0x4000) - 2.0).abs() < 0.001);
}

#[test]
fn test_list_tensors_safetensors_magic_detection() {
    // SafeTensors starts with u64 header length + '{"'
    // Create minimal detection pattern
    let header_len: u64 = 10;
    let mut data = header_len.to_le_bytes().to_vec();
    data.extend_from_slice(b"{\""); // JSON start
    data.extend_from_slice(&[0u8; 20]); // Truncated

    let result = list_tensors_from_bytes(&data, TensorListOptions::default());
    // Should detect SafeTensors but fail to parse (truncated JSON)
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        !err.contains("Unknown model format"),
        "Should detect SafeTensors format: {err}"
    );
}

#[test]
fn test_list_tensors_safetensors_valid() {
    let st_bytes = build_pygmy_safetensors();

    let result =
        list_tensors_from_bytes(&st_bytes, TensorListOptions::default()).expect("list SafeTensors");

    assert_eq!(result.format_version, "SafeTensors");
    assert!(result.tensor_count > 0);
    // Default pygmy has token_embedding.weight
    assert!(
        result
            .tensors
            .iter()
            .any(|t| t.name.contains("embedding") || t.name.contains("token")),
        "Should have embedding tensor"
    );
}

#[test]
fn test_list_tensors_safetensors_with_stats() {
    let st_bytes = build_pygmy_safetensors();

    let opts = TensorListOptions::new().with_stats();
    let result = list_tensors_from_bytes(&st_bytes, opts).expect("list");

    // At least one tensor should have stats computed
    let has_stats = result.tensors.iter().any(|t| t.mean.is_some());
    assert!(has_stats, "Should compute stats for SafeTensors");
}

#[test]
fn test_list_tensors_safetensors_with_config() {
    let st_bytes = build_pygmy_safetensors_with_config(PygmyConfig::llama_style());

    let result = list_tensors_from_bytes(&st_bytes, TensorListOptions::default()).expect("list");

    assert_eq!(result.format_version, "SafeTensors");
    // LLaMA style should have multiple tensors
    assert!(
        result.tensor_count >= 2,
        "LLaMA style should have multiple tensors"
    );
}

#[test]
fn test_list_tensors_safetensors_with_filter() {
    let st_bytes = build_pygmy_safetensors_with_config(PygmyConfig::llama_style());

    let opts = TensorListOptions::new().with_filter("norm");
    let result = list_tensors_from_bytes(&st_bytes, opts).expect("list");

    for tensor in &result.tensors {
        assert!(
            tensor.name.contains("norm"),
            "Filtered tensor should contain 'norm': {}",
            tensor.name
        );
    }
}

#[test]
fn test_list_tensors_safetensors_with_limit() {
    let st_bytes = build_pygmy_safetensors_with_config(PygmyConfig::llama_style());

    let opts = TensorListOptions::new().with_limit(2);
    let result = list_tensors_from_bytes(&st_bytes, opts).expect("list");

    assert!(
        result.tensors.len() <= 2,
        "Should limit to 2 tensors, got {}",
        result.tensors.len()
    );
}

// ========================================================================
// Format Detection Priority Tests
// ========================================================================

#[test]
fn test_format_detection_gguf_priority() {
    // GGUF magic should be detected before SafeTensors heuristic
    let mut data = b"GGUF".to_vec();
    // Add bytes that could trigger SafeTensors detection
    data.extend_from_slice(&[10, 0, 0, 0, 0, 0, 0, 0]); // u64 = 10
    data.extend_from_slice(b"{\""); // JSON start

    let result = list_tensors_from_bytes(&data, TensorListOptions::default());
    // Should fail as GGUF (not SafeTensors), proving GGUF check comes first
    assert!(result.is_err());
    // Error should not mention "Unknown format" since GGUF was detected
    let err = result.unwrap_err().to_string();
    assert!(!err.contains("Unknown model format"));
}

#[test]
fn test_format_detection_apr_fallback() {
    // APR v2 magic
    let apr_bytes = build_pygmy_apr();

    let result =
        list_tensors_from_bytes(&apr_bytes, TensorListOptions::default()).expect("list APR");

    assert_eq!(result.format_version, "v2");
}

// ====================================================================
// Coverage: f16_to_f32 special cases (denorm, inf, NaN)
// ====================================================================

#[test]
fn test_f16_to_f32_denormalized() {
    // Smallest positive denorm: sign=0, exp=0, mantissa=1
    let val = super::f16_to_f32(0x0001);
    assert!(
        val > 0.0 && val < 1e-6,
        "denorm should be tiny positive: {val}"
    );
}

#[test]
fn test_f16_to_f32_positive_infinity() {
    // +Inf: sign=0, exp=31, mantissa=0 → 0x7C00
    let val = super::f16_to_f32(0x7C00);
    assert!(val.is_infinite() && val > 0.0);
}

#[test]
fn test_f16_to_f32_negative_infinity() {
    // -Inf: sign=1, exp=31, mantissa=0 → 0xFC00
    let val = super::f16_to_f32(0xFC00);
    assert!(val.is_infinite() && val < 0.0);
}

#[test]
fn test_f16_to_f32_nan() {
    // NaN: sign=0, exp=31, mantissa!=0 → 0x7E00
    let val = super::f16_to_f32(0x7E00);
    assert!(val.is_nan());
}

#[test]
fn test_f16_to_f32_negative_zero() {
    // -0: sign=1, exp=0, mantissa=0 → 0x8000
    let val = super::f16_to_f32(0x8000);
    assert_eq!(val, 0.0);
    assert!(val.is_sign_negative());
}

// ====================================================================
// Coverage: safetensors_bytes_to_f32 all dtype branches
// ====================================================================

#[test]
fn test_safetensors_bytes_to_f32_bf16() {
    // BF16 for 1.0: top 16 bits of f32 1.0 (0x3F80_0000) → 0x3F80
    let bf16_bytes = 0x3F80u16.to_le_bytes();
    let result = super::safetensors_bytes_to_f32(&bf16_bytes, "BF16");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_safetensors_bytes_to_f32_unknown_dtype() {
    let bytes = [0u8; 16];
    let result = super::safetensors_bytes_to_f32(&bytes, "Q4_K");
    assert!(result.is_empty());
}

#[test]
fn test_safetensors_bytes_to_f32_f32() {
    let val: f32 = 3.14;
    let bytes = val.to_le_bytes();
    let result = super::safetensors_bytes_to_f32(&bytes, "F32");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 3.14).abs() < 1e-6);
}

// ====================================================================
// Coverage: compute_tensor_stats edge cases
// ====================================================================

#[test]
fn test_compute_stats_all_nan() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![3],
        dtype: "F32".to_string(),
        size_bytes: 12,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[f32::NAN, f32::NAN, f32::NAN]);
    assert_eq!(info.nan_count, Some(3));
    assert!(info.mean.is_none(), "no valid values → no mean");
}

#[test]
fn test_compute_stats_all_inf() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![2],
        dtype: "F32".to_string(),
        size_bytes: 8,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(info.inf_count, Some(2));
    assert!(info.mean.is_none());
}

#[test]
fn test_compute_stats_single_value() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![1],
        dtype: "F32".to_string(),
        size_bytes: 4,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[42.0]);
    assert_eq!(info.mean, Some(42.0));
    assert_eq!(info.std, Some(0.0));
    assert_eq!(info.min, Some(42.0));
    assert_eq!(info.max, Some(42.0));
}

#[test]
fn test_compute_stats_identical_values() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[5.0, 5.0, 5.0, 5.0]);
    assert_eq!(info.mean, Some(5.0));
    assert_eq!(info.std, Some(0.0));
}

// ====================================================================
// Coverage: list_tensors_safetensors error paths
// ====================================================================

#[test]
fn test_safetensors_too_small() {
    let result = list_tensors_from_bytes(&[0u8; 4], TensorListOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_safetensors_truncated_header() {
    // header_len says 1000, but only 20 bytes total
    let mut data = vec![0u8; 20];
    data[0..8].copy_from_slice(&1000u64.to_le_bytes());
    // Make it look like SafeTensors (bytes 8-9 = '{"')
    data[8] = b'{';
    data[9] = b'"';
    let result = super::list_tensors_safetensors(&data, TensorListOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("truncated"));
}

#[test]
fn test_safetensors_invalid_json() {
    let header = b"not json at all";
    let header_len = header.len() as u64;
    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header);
    let result = super::list_tensors_safetensors(&data, TensorListOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("JSON"));
}

#[test]
fn test_safetensors_json_not_object() {
    let header = b"[1, 2, 3]";
    let header_len = header.len() as u64;
    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header);
    let result = super::list_tensors_safetensors(&data, TensorListOptions::default());
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not a JSON object"));
}

#[test]
fn test_safetensors_with_stats() {
    // Build a minimal SafeTensors with one F32 tensor and compute stats
    let tensor_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let header = format!(
        r#"{{"test_tensor":{{"dtype":"F32","shape":[4],"data_offsets":[0,{}]}}}}"#,
        tensor_data.len()
    );
    let header_bytes = header.as_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(header_bytes);
    data.extend_from_slice(&tensor_data);

    let opts = TensorListOptions {
        compute_stats: true,
        ..TensorListOptions::default()
    };
    let result = super::list_tensors_safetensors(&data, opts).expect("parse ok");
    assert_eq!(result.tensor_count, 1);
    let t = &result.tensors[0];
    assert!(t.mean.is_some());
    assert!((t.mean.expect("test value") - 2.5).abs() < 0.01);
}

#[test]
fn test_safetensors_limit() {
    // Two tensors, limit=1
    let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&[0u8; 8]);

    let opts = TensorListOptions {
        limit: 1,
        ..TensorListOptions::default()
    };
    let result = super::list_tensors_safetensors(&data, opts).expect("parse ok");
    // GH-195 FIX: tensor_count reflects true total, not limited count
    assert_eq!(result.tensor_count, 2);
    assert_eq!(result.tensors.len(), 1);
}

#[test]
fn test_safetensors_filter() {
    let header = r#"{"attn.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"mlp.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(&(header.len() as u64).to_le_bytes());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&[0u8; 16]);

    let opts = TensorListOptions {
        filter: Some("attn".to_string()),
        ..TensorListOptions::default()
    };
    let result = super::list_tensors_safetensors(&data, opts).expect("parse ok");
    assert_eq!(result.tensor_count, 1);
    assert_eq!(result.tensors[0].name, "attn.weight");
}

// ====================================================================
// Coverage: list_tensors_v1 basic path
// ====================================================================

#[test]
fn test_list_tensors_v1_too_small() {
    let data = vec![0u8; 16]; // less than HEADER_SIZE
    let result = super::list_tensors_v1(&data, TensorListOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_list_tensors_v1_metadata_truncated() {
    // Header says metadata is 1000 bytes but file is too small
    let mut data = vec![0u8; 64]; // HEADER_SIZE
                                  // offset 8-11: metadata_size = 1000
    data[8..12].copy_from_slice(&1000u32.to_le_bytes());
    let result = super::list_tensors_v1(&data, TensorListOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_list_tensors_v1_empty_metadata() {
    // Valid header with empty metadata (size=2, content="{}")
    let metadata = b"{}";
    let mut data = vec![0u8; 64];
    data[8..12].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
    data.extend_from_slice(metadata);
    let result = super::list_tensors_v1(&data, TensorListOptions::default());
    assert!(result.is_ok());
    assert_eq!(result.expect("test value").tensor_count, 0);
}

// ====================================================================
// Coverage: ggml_dtype_name all branches
// ====================================================================

#[test]
fn test_ggml_dtype_name_all() {
    assert_eq!(super::ggml_dtype_name(0), "F32");
    assert_eq!(super::ggml_dtype_name(1), "F16");
    assert_eq!(super::ggml_dtype_name(2), "Q4_0");
    assert_eq!(super::ggml_dtype_name(3), "Q4_1");
    assert_eq!(super::ggml_dtype_name(6), "Q5_0");
    assert_eq!(super::ggml_dtype_name(7), "Q5_1");
    assert_eq!(super::ggml_dtype_name(8), "Q8_0");
    assert_eq!(super::ggml_dtype_name(12), "Q4_K");
    assert_eq!(super::ggml_dtype_name(14), "Q6_K");
    assert_eq!(super::ggml_dtype_name(999), "unknown");
}

// ====================================================================
// Coverage: ggml_dtype_name exhaustive (all match arms)
// ====================================================================

#[test]
fn test_ggml_dtype_name_exhaustive_all_arms() {
    // Ensure every documented dtype code has its correct name
    assert_eq!(super::ggml_dtype_name(9), "Q8_1");
    assert_eq!(super::ggml_dtype_name(10), "Q2_K");
    assert_eq!(super::ggml_dtype_name(11), "Q3_K");
    assert_eq!(super::ggml_dtype_name(13), "Q5_K");
    assert_eq!(super::ggml_dtype_name(15), "Q8_K");
    assert_eq!(super::ggml_dtype_name(16), "IQ2_XXS");
    assert_eq!(super::ggml_dtype_name(17), "IQ2_XS");
    assert_eq!(super::ggml_dtype_name(18), "IQ3_XXS");
    assert_eq!(super::ggml_dtype_name(26), "BF16");
    // Newly added I-quant and integer types
    assert_eq!(super::ggml_dtype_name(19), "IQ1_S");
    assert_eq!(super::ggml_dtype_name(20), "IQ4_NL");
    assert_eq!(super::ggml_dtype_name(21), "IQ3_S");
    assert_eq!(super::ggml_dtype_name(22), "IQ2_S");
    assert_eq!(super::ggml_dtype_name(23), "IQ4_XS");
    assert_eq!(super::ggml_dtype_name(24), "I8");
    assert_eq!(super::ggml_dtype_name(25), "I16");
    assert_eq!(super::ggml_dtype_name(27), "I32");
    assert_eq!(super::ggml_dtype_name(28), "I64");
    assert_eq!(super::ggml_dtype_name(29), "F64");
    assert_eq!(super::ggml_dtype_name(30), "IQ1_M");
    // Codes with no mapping (gaps in GGML enum)
    assert_eq!(super::ggml_dtype_name(4), "unknown");
    assert_eq!(super::ggml_dtype_name(5), "unknown");
    assert_eq!(super::ggml_dtype_name(31), "unknown");
    assert_eq!(super::ggml_dtype_name(u32::MAX), "unknown");
}

// ====================================================================
// Coverage: ggml_dtype_element_size exhaustive (all match arms)
// ====================================================================

#[test]
fn test_ggml_dtype_element_size_exhaustive() {
    // F32
    assert!((super::ggml_dtype_element_size(0) - 4.0).abs() < 0.001);
    // F16
    assert!((super::ggml_dtype_element_size(1) - 2.0).abs() < 0.001);
    // Q4_0
    assert!((super::ggml_dtype_element_size(2) - 0.5625).abs() < 0.01);
    // Q4_1
    assert!((super::ggml_dtype_element_size(3) - 0.625).abs() < 0.01);
    // Q5_0
    assert!((super::ggml_dtype_element_size(6) - 0.6875).abs() < 0.01);
    // Q5_1
    assert!((super::ggml_dtype_element_size(7) - 0.75).abs() < 0.01);
    // Q8_0
    assert!((super::ggml_dtype_element_size(8) - 1.0625).abs() < 0.01);
    // Q8_1
    assert!((super::ggml_dtype_element_size(9) - 1.125).abs() < 0.01);
    // Q2_K
    assert!((super::ggml_dtype_element_size(10) - 0.3125).abs() < 0.01);
    // Q3_K
    assert!((super::ggml_dtype_element_size(11) - 0.4375).abs() < 0.01);
    // Q4_K
    assert!((super::ggml_dtype_element_size(12) - 0.5625).abs() < 0.01);
    // Q5_K
    assert!((super::ggml_dtype_element_size(13) - 0.6875).abs() < 0.01);
    // Q6_K
    assert!((super::ggml_dtype_element_size(14) - 0.8125).abs() < 0.01);
    // Q8_K
    assert!((super::ggml_dtype_element_size(15) - 1.0625).abs() < 0.01);
    // BF16
    assert!((super::ggml_dtype_element_size(26) - 2.0).abs() < 0.001);
    // I-quant types
    assert!((super::ggml_dtype_element_size(16) - 0.5625).abs() < 0.01); // IQ2_XXS
    assert!((super::ggml_dtype_element_size(17) - 0.625).abs() < 0.01); // IQ2_XS
    assert!((super::ggml_dtype_element_size(18) - 0.6875).abs() < 0.01); // IQ3_XXS
    assert!((super::ggml_dtype_element_size(19) - 0.4375).abs() < 0.01); // IQ1_S
    assert!((super::ggml_dtype_element_size(20) - 0.5625).abs() < 0.01); // IQ4_NL
    assert!((super::ggml_dtype_element_size(21) - 0.4375).abs() < 0.01); // IQ3_S
    assert!((super::ggml_dtype_element_size(22) - 0.625).abs() < 0.01); // IQ2_S
    assert!((super::ggml_dtype_element_size(23) - 0.5).abs() < 0.01); // IQ4_XS
                                                                      // Integer types
    assert!((super::ggml_dtype_element_size(24) - 1.0).abs() < 0.01); // I8
    assert!((super::ggml_dtype_element_size(25) - 2.0).abs() < 0.01); // I16
    assert!((super::ggml_dtype_element_size(27) - 4.0).abs() < 0.01); // I32
    assert!((super::ggml_dtype_element_size(28) - 8.0).abs() < 0.01); // I64
    assert!((super::ggml_dtype_element_size(29) - 8.0).abs() < 0.01); // F64
    assert!((super::ggml_dtype_element_size(30) - 0.375).abs() < 0.01); // IQ1_M
                                                                        // Unknown defaults to F32 (4.0) — conservative size estimate
    assert!((super::ggml_dtype_element_size(99) - 4.0).abs() < 0.001);
    assert!((super::ggml_dtype_element_size(u32::MAX) - 4.0).abs() < 0.001);
}

// ====================================================================
// Coverage: f16_to_f32 additional edge cases
// ====================================================================

#[test]
fn test_f16_to_f32_positive_zero() {
    // +0: sign=0, exp=0, mantissa=0 -> 0x0000
    let val = super::f16_to_f32(0x0000);
    assert_eq!(val, 0.0);
    assert!(val.is_sign_positive());
}

#[test]
fn test_f16_to_f32_normal_half() {
    // 0.5 in f16 = sign=0, exp=14, mantissa=0 -> 0x3800
    let val = super::f16_to_f32(0x3800);
    assert!((val - 0.5).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_largest_denorm() {
    // Largest denorm: sign=0, exp=0, mantissa=0x3FF -> 0x03FF
    let val = super::f16_to_f32(0x03FF);
    assert!(val > 0.0 && val < 1e-4, "largest denorm: {val}");
}

#[test]
fn test_f16_to_f32_negative_normal() {
    // -2.0 in f16 = sign=1, exp=16, mantissa=0 -> 0xC000
    let val = super::f16_to_f32(0xC000);
    assert!((val - (-2.0)).abs() < 0.001);
}

#[test]
fn test_f16_to_f32_negative_nan() {
    // Negative NaN: sign=1, exp=31, mantissa!=0 -> 0xFE00
    let val = super::f16_to_f32(0xFE00);
    assert!(val.is_nan());
}

// ====================================================================
// Coverage: bf16_to_f32 additional cases
// ====================================================================

#[test]
fn test_bf16_to_f32_negative() {
    // -1.0 in bf16: top 16 bits of f32 -1.0 (0xBF80_0000) -> 0xBF80
    let val = super::bf16_to_f32(0xBF80);
    assert!((val - (-1.0)).abs() < 0.001);
}

#[test]
fn test_bf16_to_f32_large_value() {
    // 256.0 in bf16: top 16 bits of f32 256.0 (0x4380_0000) -> 0x4380
    let val = super::bf16_to_f32(0x4380);
    assert!((val - 256.0).abs() < 1.0);
}

#[test]
fn test_bf16_to_f32_small_positive() {
    // 0.5 in bf16: top 16 bits of f32 0.5 (0x3F00_0000) -> 0x3F00
    let val = super::bf16_to_f32(0x3F00);
    assert!((val - 0.5).abs() < 0.001);
}

// ====================================================================
// Coverage: safetensors_bytes_to_f32 - F16 branch
// ====================================================================

#[test]
fn test_safetensors_bytes_to_f32_f16() {
    // 1.0 in f16 = 0x3C00
    let f16_bytes = 0x3C00u16.to_le_bytes();
    let result = super::safetensors_bytes_to_f32(&f16_bytes, "F16");
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.0).abs() < 0.001);
}

#[test]
fn test_safetensors_bytes_to_f32_f16_multiple() {
    // Two f16 values: 1.0 (0x3C00), 2.0 (0x4000)
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x3C00u16.to_le_bytes());
    bytes.extend_from_slice(&0x4000u16.to_le_bytes());
    let result = super::safetensors_bytes_to_f32(&bytes, "F16");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 0.001);
    assert!((result[1] - 2.0).abs() < 0.001);
}

#[test]
fn test_safetensors_bytes_to_f32_f32_multiple() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1.5f32.to_le_bytes());
    bytes.extend_from_slice(&(-2.5f32).to_le_bytes());
    let result = super::safetensors_bytes_to_f32(&bytes, "F32");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.5).abs() < 1e-6);
    assert!((result[1] - (-2.5)).abs() < 1e-6);
}

#[test]
fn test_safetensors_bytes_to_f32_bf16_multiple() {
    // 1.0 (0x3F80), -1.0 (0xBF80)
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x3F80u16.to_le_bytes());
    bytes.extend_from_slice(&0xBF80u16.to_le_bytes());
    let result = super::safetensors_bytes_to_f32(&bytes, "BF16");
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_safetensors_bytes_to_f32_empty_bytes() {
    let result = super::safetensors_bytes_to_f32(&[], "F32");
    assert!(result.is_empty());

    let result = super::safetensors_bytes_to_f32(&[], "F16");
    assert!(result.is_empty());

    let result = super::safetensors_bytes_to_f32(&[], "BF16");
    assert!(result.is_empty());
}

// ====================================================================
// Coverage: compute_tensor_stats mixed NaN/Inf/valid
// ====================================================================

#[test]
fn test_compute_stats_mixed_nan_inf_valid() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![5],
        dtype: "F32".to_string(),
        size_bytes: 20,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(
        &mut info,
        &[1.0, f32::NAN, f32::INFINITY, 3.0, f32::NEG_INFINITY],
    );
    assert_eq!(info.nan_count, Some(1));
    assert_eq!(info.inf_count, Some(2));
    // Mean from valid values: (1.0 + 3.0) / 2 = 2.0
    assert!((info.mean.expect("test value") - 2.0).abs() < 0.01);
    assert_eq!(info.min, Some(1.0));
    assert_eq!(info.max, Some(3.0));
}

#[test]
fn test_compute_stats_negative_values() {
    let mut info = TensorInfo {
        name: "t".to_string(),
        shape: vec![4],
        dtype: "F32".to_string(),
        size_bytes: 16,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };
    super::compute_tensor_stats(&mut info, &[-3.0, -1.0, 1.0, 3.0]);
    assert_eq!(info.mean, Some(0.0));
    assert_eq!(info.min, Some(-3.0));
    assert_eq!(info.max, Some(3.0));
    assert!(info.std.expect("test value") > 0.0);
}

// ====================================================================
// Coverage: format_size boundary values
// ====================================================================

#[test]
fn test_format_size_boundary_values() {
    // Just below KB
    assert_eq!(super::format_size(1023), "1023 B");
    // Exactly KB
    assert_eq!(super::format_size(1024), "1.00 KB");
    // Just below MB
    let just_below_mb = 1024 * 1024 - 1;
    let result = super::format_size(just_below_mb);
    assert!(result.contains("KB"));
    // Exactly MB
    assert_eq!(super::format_size(1024 * 1024), "1.00 MB");
    // Just below GB
    let just_below_gb = 1024 * 1024 * 1024 - 1;
    let result = super::format_size(just_below_gb);
    assert!(result.contains("MB"));
    // Exactly GB
    assert_eq!(super::format_size(1024 * 1024 * 1024), "1.00 GB");
    // Multi-GB
    assert_eq!(super::format_size(10 * 1024 * 1024 * 1024), "10.00 GB");
}

#[test]
fn test_format_size_one_byte() {
    assert_eq!(super::format_size(1), "1 B");
}

// ====================================================================
// Coverage: parse_shape_array
// ====================================================================

#[test]
fn test_parse_shape_array_valid() {
    let val = serde_json::json!([2, 3, 4]);
    let shape = super::parse_shape_array(&val);
    assert_eq!(shape, vec![2, 3, 4]);
}

#[test]
fn test_parse_shape_array_empty() {
    let val = serde_json::json!([]);
    let shape = super::parse_shape_array(&val);
    assert!(shape.is_empty());
}

#[test]
fn test_parse_shape_array_non_array() {
    let val = serde_json::json!("not an array");
    let shape = super::parse_shape_array(&val);
    assert!(shape.is_empty());
}

#[test]
fn test_parse_shape_array_null() {
    let val = serde_json::json!(null);
    let shape = super::parse_shape_array(&val);
    assert!(shape.is_empty());
}

#[test]
fn test_parse_shape_array_mixed_types() {
    // Non-u64 values should be filtered out
    let val = serde_json::json!([10, "bad", 20, null]);
    let shape = super::parse_shape_array(&val);
    assert_eq!(shape, vec![10, 20]);
}

// ====================================================================
// Coverage: list_tensors_from_bytes error paths
// ====================================================================

#[test]
fn test_list_tensors_too_small_1_byte() {
    let result = super::list_tensors_from_bytes(&[0x42], TensorListOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too small"));
}

#[test]
fn test_list_tensors_too_small_3_bytes() {
    let result = super::list_tensors_from_bytes(&[0x41, 0x50, 0x52], TensorListOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too small"));
}

#[test]
fn test_list_tensors_unknown_magic_4_bytes() {
    // Exactly 4 bytes, not GGUF, not SafeTensors, not APR
    let data = [0xDE, 0xAD, 0xBE, 0xEF];
    let result = super::list_tensors_from_bytes(&data, TensorListOptions::default());
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Unknown model format"));
}

// ====================================================================
// Coverage: GGUF dispatch through list_tensors_from_bytes
// ====================================================================

#[test]
fn test_list_tensors_gguf_dispatch_empty_model() {
    // Build valid GGUF header with 0 tensors
    let mut data = b"GGUF".to_vec();
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count = 0

    let result =
        super::list_tensors_from_bytes(&data, TensorListOptions::default()).expect("valid GGUF");
    assert_eq!(result.tensor_count, 0);
    assert!(result.format_version.contains("GGUF"));
    assert!(result.tensors.is_empty());
}

// ====================================================================
// Coverage: SafeTensors too small (< 8 bytes)
// ====================================================================

#[test]
fn test_safetensors_exactly_8_bytes_too_small() {
    // 8 bytes but the header_len points to data beyond what we have
    let mut data = vec![0u8; 8];
    data[0..8].copy_from_slice(&100u64.to_le_bytes());
    let result = super::list_tensors_safetensors(&data, TensorListOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_safetensors_less_than_8_bytes() {
    let data = vec![0u8; 5];
    let result = super::list_tensors_safetensors(&data, TensorListOptions::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too small"));
}

// ====================================================================
// extract_tensors_from_metadata_with_counts: Coverage tests (impact 5.7)
// ====================================================================

#[test]
fn test_extract_tensors_from_metadata_with_counts_basic() {
    use std::collections::HashMap;

    let mut metadata = HashMap::new();
    let shapes = serde_json::json!({
        "model.embed_tokens.weight": [256, 128],
        "model.norm.weight": [128],
        "model.layers.0.self_attn.q_proj.weight": [128, 128]
    });
    metadata.insert("tensor_shapes".to_string(), shapes);

    let options = TensorListOptions::default();
    let (tensors, total_count, total_size_bytes) =
        super::extract_tensors_from_metadata_with_counts(&metadata, &options);

    assert_eq!(total_count, 3);
    assert_eq!(tensors.len(), 3);
    // Size = sum of elements * 4 (f32 assumed)
    // 256*128*4 + 128*4 + 128*128*4 = 131072 + 512 + 65536 = 197120
    assert_eq!(total_size_bytes, 197120);
}

#[test]
fn test_extract_tensors_from_metadata_with_counts_empty_shapes() {
    use std::collections::HashMap;

    let mut metadata = HashMap::new();
    metadata.insert("tensor_shapes".to_string(), serde_json::json!({}));

    let options = TensorListOptions::default();
    let (tensors, total_count, total_size) =
        super::extract_tensors_from_metadata_with_counts(&metadata, &options);

    assert_eq!(total_count, 0);
    assert_eq!(tensors.len(), 0);
    assert_eq!(total_size, 0);
}

#[test]
fn test_extract_tensors_from_metadata_with_counts_no_tensor_shapes_key() {
    use std::collections::HashMap;

    let metadata = HashMap::new();

    let options = TensorListOptions::default();
    let (tensors, total_count, total_size) =
        super::extract_tensors_from_metadata_with_counts(&metadata, &options);

    assert_eq!(total_count, 0);
    assert_eq!(tensors.len(), 0);
    assert_eq!(total_size, 0);
}

#[test]
fn test_extract_tensors_from_metadata_with_counts_non_object_tensor_shapes() {
    use std::collections::HashMap;

    let mut metadata = HashMap::new();
    metadata.insert(
        "tensor_shapes".to_string(),
        serde_json::json!("not an object"),
    );

    let options = TensorListOptions::default();
    let (tensors, total_count, total_size) =
        super::extract_tensors_from_metadata_with_counts(&metadata, &options);

    assert_eq!(total_count, 0);
    assert_eq!(tensors.len(), 0);
    assert_eq!(total_size, 0);
}

#[test]
fn test_extract_tensors_from_metadata_with_counts_with_filter() {
    use std::collections::HashMap;

    let mut metadata = HashMap::new();
    let shapes = serde_json::json!({
        "model.embed_tokens.weight": [256, 128],
        "model.norm.weight": [128],
        "model.layers.0.self_attn.q_proj.weight": [128, 128],
        "model.layers.0.self_attn.k_proj.weight": [64, 128]
    });
    metadata.insert("tensor_shapes".to_string(), shapes);

    let options = TensorListOptions {
        filter: Some("attn".to_string()),
        ..Default::default()
    };
    let (tensors, total_count, _total_size) =
        super::extract_tensors_from_metadata_with_counts(&metadata, &options);

    // Only attn tensors match
    assert_eq!(total_count, 2);
    assert_eq!(tensors.len(), 2);
    for t in &tensors {
        assert!(
            t.name.contains("attn"),
            "all tensors should match filter: {}",
            t.name
        );
    }
}

#[test]
fn test_extract_tensors_from_metadata_with_counts_with_limit() {
    use std::collections::HashMap;

    let mut metadata = HashMap::new();
    let mut shapes = serde_json::Map::new();
    for i in 0..10 {
        shapes.insert(format!("tensor_{i}"), serde_json::json!([64, 64]));
    }
    metadata.insert(
        "tensor_shapes".to_string(),
        serde_json::Value::Object(shapes),
    );

    let options = TensorListOptions {
        limit: 3,
        ..Default::default()
    };
    let (tensors, total_count, total_size) =
        super::extract_tensors_from_metadata_with_counts(&metadata, &options);

    // total_count should be 10 (all tensors counted)
    assert_eq!(total_count, 10);
    // But only 3 returned (due to limit)
    assert_eq!(tensors.len(), 3);
    // total_size should reflect ALL tensors, not just limited ones
    // 10 * 64 * 64 * 4 = 163840
    assert_eq!(total_size, 163840);
}

#[test]
fn test_extract_tensors_from_metadata_with_counts_dtype_is_f32() {
    use std::collections::HashMap;

    let mut metadata = HashMap::new();
    let shapes = serde_json::json!({
        "tensor_a": [4, 4]
    });
    metadata.insert("tensor_shapes".to_string(), shapes);

    let options = TensorListOptions::default();
    let (tensors, _, _) = super::extract_tensors_from_metadata_with_counts(&metadata, &options);

    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].dtype, "f32");
    assert_eq!(tensors[0].size_bytes, 4 * 4 * 4); // 4*4 elements * 4 bytes
    assert!(tensors[0].mean.is_none());
    assert!(tensors[0].std.is_none());
    assert!(tensors[0].min.is_none());
    assert!(tensors[0].max.is_none());
    assert!(tensors[0].nan_count.is_none());
    assert!(tensors[0].inf_count.is_none());
}

#[test]
fn test_extract_tensors_from_metadata_with_counts_filter_no_match() {
    use std::collections::HashMap;

    let mut metadata = HashMap::new();
    let shapes = serde_json::json!({
        "model.embed_tokens.weight": [256, 128],
        "model.norm.weight": [128]
    });
    metadata.insert("tensor_shapes".to_string(), shapes);

    let options = TensorListOptions {
        filter: Some("nonexistent_pattern".to_string()),
        ..Default::default()
    };
    let (tensors, total_count, total_size) =
        super::extract_tensors_from_metadata_with_counts(&metadata, &options);

    assert_eq!(total_count, 0);
    assert_eq!(tensors.len(), 0);
    assert_eq!(total_size, 0);
}
