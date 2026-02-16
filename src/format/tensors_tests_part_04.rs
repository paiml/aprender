use super::*;

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
