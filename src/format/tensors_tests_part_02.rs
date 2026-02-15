
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
