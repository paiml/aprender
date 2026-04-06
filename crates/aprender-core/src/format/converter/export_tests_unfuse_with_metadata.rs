use super::*;

#[test]
fn test_unfuse_qkv_tensors_weight_split_with_metadata() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");

    // Build APR with metadata that has the config we need
    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.hidden_size = Some(4);
    metadata.num_heads = Some(2);
    metadata.num_kv_heads = Some(2);

    let mut writer = AprV2Writer::new(metadata);
    // Need at least one tensor to write a valid APR
    writer.add_f32_tensor("dummy", vec![4], &[1.0f32; 4]);
    let bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &bytes).expect("write file");

    // hidden_size=4, num_heads=2, num_kv_heads=2
    // head_dim = 4/2 = 2, kv_dim = 2*2 = 4
    // qkv_dim = hidden_size + 2*kv_dim = 4 + 8 = 12
    // weight shape: [qkv_dim, hidden_dim] = [12, 4]
    let hidden_dim = 4;
    let q_elements = 4 * hidden_dim; // hidden_size * hidden_dim = 16
    let kv_elements = 4 * hidden_dim; // kv_dim * hidden_dim = 16
    let total = q_elements + 2 * kv_elements; // 48

    let mut data = Vec::with_capacity(total);
    for i in 0..total {
        data.push(i as f32);
    }

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.weight".to_string(),
        (data, vec![12, 4]),
    );

    let result = unfuse_qkv_tensors(tensors, &apr_path);

    // Should have 3 separate tensors instead of 1 fused
    assert_eq!(result.len(), 3, "should split into q, k, v");
    assert!(result.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(result.contains_key("model.layers.0.self_attn.k_proj.weight"));
    assert!(result.contains_key("model.layers.0.self_attn.v_proj.weight"));

    // Verify shapes
    let (q_data, q_shape) = result
        .get("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q_shape, &vec![4, 4]); // [hidden_size, hidden_dim]
    assert_eq!(q_data.len(), 16);

    let (k_data, k_shape) = result
        .get("model.layers.0.self_attn.k_proj.weight")
        .unwrap();
    assert_eq!(k_shape, &vec![4, 4]); // [kv_dim, hidden_dim]
    assert_eq!(k_data.len(), 16);

    let (v_data, v_shape) = result
        .get("model.layers.0.self_attn.v_proj.weight")
        .unwrap();
    assert_eq!(v_shape, &vec![4, 4]); // [kv_dim, hidden_dim]
    assert_eq!(v_data.len(), 16);

    // Verify data is correctly split (sequential values)
    assert_eq!(q_data[0], 0.0);
    assert_eq!(k_data[0], 16.0);
    assert_eq!(v_data[0], 32.0);
}

#[test]
fn test_unfuse_qkv_tensors_bias_split_with_metadata() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");

    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.hidden_size = Some(4);
    metadata.num_heads = Some(2);
    metadata.num_kv_heads = Some(2);

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("dummy", vec![4], &[1.0f32; 4]);
    let bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &bytes).expect("write file");

    // kv_dim = 2*2 = 4
    // bias shape: [qkv_dim] = [hidden_size + 2*kv_dim] = [4 + 8] = [12]
    let mut data = Vec::new();
    for i in 0..12 {
        data.push(i as f32);
    }

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.bias".to_string(),
        (data, vec![12]),
    );

    let result = unfuse_qkv_tensors(tensors, &apr_path);

    assert_eq!(result.len(), 3, "bias should split into q, k, v");
    assert!(result.contains_key("model.layers.0.self_attn.q_proj.bias"));
    assert!(result.contains_key("model.layers.0.self_attn.k_proj.bias"));
    assert!(result.contains_key("model.layers.0.self_attn.v_proj.bias"));

    let (q_bias, q_shape) = result.get("model.layers.0.self_attn.q_proj.bias").unwrap();
    assert_eq!(q_shape, &vec![4]); // hidden_size
    assert_eq!(q_bias, &[0.0, 1.0, 2.0, 3.0]);

    let (k_bias, k_shape) = result.get("model.layers.0.self_attn.k_proj.bias").unwrap();
    assert_eq!(k_shape, &vec![4]); // kv_dim
    assert_eq!(k_bias, &[4.0, 5.0, 6.0, 7.0]);

    let (v_bias, v_shape) = result.get("model.layers.0.self_attn.v_proj.bias").unwrap();
    assert_eq!(v_shape, &vec![4]); // kv_dim
    assert_eq!(v_bias, &[8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn test_unfuse_qkv_tensors_weight_too_small_passthrough() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");

    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.hidden_size = Some(4);
    metadata.num_heads = Some(2);
    metadata.num_kv_heads = Some(2);

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("dummy", vec![4], &[1.0f32; 4]);
    let bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &bytes).expect("write file");

    // Data too small for split (needs 48, only give 10)
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.weight".to_string(),
        (vec![0.5f32; 10], vec![5, 2]),
    );

    let result = unfuse_qkv_tensors(tensors, &apr_path);
    // Should keep the original tensor because data is too small to split
    assert_eq!(result.len(), 1);
    assert!(result.contains_key("model.layers.0.self_attn.qkv_proj.weight"));
}

#[test]
fn test_unfuse_qkv_tensors_bias_wrong_size_passthrough() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");

    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.hidden_size = Some(4);
    metadata.num_heads = Some(2);
    metadata.num_kv_heads = Some(2);

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("dummy", vec![4], &[1.0f32; 4]);
    let bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &bytes).expect("write file");

    // Bias data length doesn't match expected qkv_dim=12
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.bias".to_string(),
        (vec![0.5f32; 7], vec![7]),
    );

    let result = unfuse_qkv_tensors(tensors, &apr_path);
    assert_eq!(result.len(), 1);
    assert!(result.contains_key("model.layers.0.self_attn.qkv_proj.bias"));
}

#[test]
fn test_unfuse_qkv_tensors_zero_hidden_returns_original() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");

    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.hidden_size = Some(0); // zero hidden size
    metadata.num_heads = Some(0); // zero heads

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("dummy", vec![4], &[1.0f32; 4]);
    let bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &bytes).expect("write file");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.weight".to_string(),
        (vec![0.5f32; 48], vec![12, 4]),
    );

    let result = unfuse_qkv_tensors(tensors, &apr_path);
    // zero hidden_size should bail early and return original
    assert_eq!(result.len(), 1);
}

// ========================================================================
// push_string_array / push_u32_field / push_i32_array: Helper tests
// ========================================================================

#[test]
fn test_push_string_array_present() {
    use crate::format::gguf::GgufValue;
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert(
        "tokenizer.tokens".to_string(),
        serde_json::json!(["hello", "world", "test"]),
    );

    push_string_array(
        &mut entries,
        &custom,
        "tokenizer.tokens",
        "tokenizer.ggml.tokens",
    );
    assert_eq!(entries.len(), 1);
    match &entries[0].1 {
        GgufValue::ArrayString(v) => assert_eq!(v, &["hello", "world", "test"]),
        other => panic!("Expected ArrayString, got: {other:?}"),
    }
}

#[test]
fn test_push_string_array_missing_key() {
    let mut entries = Vec::new();
    let custom = std::collections::HashMap::new();

    push_string_array(&mut entries, &custom, "nonexistent", "target");
    assert!(entries.is_empty());
}

#[test]
fn test_push_string_array_empty_array() {
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("arr".to_string(), serde_json::json!([]));

    push_string_array(&mut entries, &custom, "arr", "target");
    assert!(entries.is_empty(), "empty array should not be pushed");
}

#[test]
fn test_push_u32_field_present() {
    use crate::format::gguf::GgufValue;
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("vocab_size".to_string(), serde_json::json!(32000));

    push_u32_field(&mut entries, &custom, "vocab_size", "tokenizer.vocab_size");
    assert_eq!(entries.len(), 1);
    match &entries[0].1 {
        GgufValue::Uint32(v) => assert_eq!(*v, 32000),
        other => panic!("Expected Uint32, got: {other:?}"),
    }
}

#[test]
fn test_push_u32_field_missing() {
    let mut entries = Vec::new();
    let custom = std::collections::HashMap::new();

    push_u32_field(&mut entries, &custom, "nonexistent", "target");
    assert!(entries.is_empty());
}

#[test]
fn test_push_i32_array_present() {
    use crate::format::gguf::GgufValue;
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("types".to_string(), serde_json::json!([1, 3, 1, 1, 3]));

    push_i32_array(&mut entries, &custom, "types", "tokenizer.ggml.token_type");
    assert_eq!(entries.len(), 1);
    match &entries[0].1 {
        GgufValue::ArrayInt32(v) => assert_eq!(v, &[1, 3, 1, 1, 3]),
        other => panic!("Expected ArrayInt32, got: {other:?}"),
    }
}

#[test]
fn test_push_i32_array_missing() {
    let mut entries = Vec::new();
    let custom = std::collections::HashMap::new();

    push_i32_array(&mut entries, &custom, "nonexistent", "target");
    assert!(entries.is_empty());
}

#[test]
fn test_push_i32_array_empty() {
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("types".to_string(), serde_json::json!([]));

    push_i32_array(&mut entries, &custom, "types", "target");
    assert!(entries.is_empty(), "empty array should not be pushed");
}
