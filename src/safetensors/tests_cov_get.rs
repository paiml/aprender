
#[test]
fn test_cov_get_tensor_f16_wrong_dtype() {
    let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f16_as_f32("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_f16_data_offset_exceeds() {
    let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,100]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 4]); // Only 4 bytes, offset says 100

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f16_as_f32("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_bf16_as_f32() {
    let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    // Two BF16 values: 1.0 and 2.0
    data.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::bf16::from_f32(2.0).to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_bf16_as_f32("weights").expect("test");

    assert_eq!(weights.len(), 2);
    assert!((weights[0] - 1.0).abs() < 0.01);
    assert!((weights[1] - 2.0).abs() < 0.01);
}

#[test]
fn test_cov_get_tensor_bf16_not_found() {
    let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 4]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_bf16_as_f32("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_bf16_wrong_dtype() {
    let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_bf16_as_f32("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_bf16_data_offset_exceeds() {
    let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,100]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 4]); // Only 4 bytes

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_bf16_as_f32("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_auto_f32() {
    let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_auto("weights").expect("test");
    assert_eq!(weights, vec![1.0, 2.0]);
}

#[test]
fn test_cov_get_tensor_auto_f16() {
    let json = r#"{"weights":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::f16::from_f32(2.0).to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_auto("weights").expect("test");
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_cov_get_tensor_auto_bf16() {
    let json = r#"{"weights":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
    data.extend_from_slice(&half::bf16::from_f32(2.0).to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let weights = model.get_tensor_auto("weights").expect("test");
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_cov_get_tensor_auto_unsupported_dtype() {
    let json = r#"{"weights":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_auto("weights");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_auto_not_found() {
    let json = r#"{"weights":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_auto("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_cov_tensor_names() {
    let json = r#"{
            "weight1":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "weight2":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 16]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let names = model.tensor_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"weight1"));
    assert!(names.contains(&"weight2"));
}

#[test]
fn test_cov_get_tensor_info() {
    let json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 24]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");

    let info = model.get_tensor_info("weight");
    assert!(info.is_some());
    let info = info.expect("operation failed");
    assert_eq!(info.shape, vec![2, 3]);
    assert_eq!(info.dtype, SafetensorsDtype::F32);

    assert!(model.get_tensor_info("nonexistent").is_none());
}

#[test]
fn test_cov_has_tensor() {
    let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert!(model.has_tensor("weight"));
    assert!(!model.has_tensor("nonexistent"));
}

#[test]
fn test_cov_get_tensor_f32_data_offset_exceeds() {
    let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,100]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]); // Only 8 bytes, offset says 100

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f32("weight");
    assert!(result.is_err());
}

#[test]
fn test_cov_get_tensor_f32_not_multiple_of_4() {
    let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,7]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 7]); // 7 bytes, not a multiple of 4

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    let result = model.get_tensor_f32("weight");
    assert!(result.is_err());
}

#[test]
fn test_cov_safetensors_config_num_kv_heads() {
    let config = SafetensorsConfig {
        hidden_size: Some(768),
        num_hidden_layers: Some(12),
        num_attention_heads: Some(12),
        num_key_value_heads: Some(4),
        vocab_size: Some(32000),
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
        ..Default::default()
    };

    assert_eq!(config.num_kv_heads(), 4);
}

#[test]
fn test_cov_safetensors_config_num_kv_heads_default() {
    let config = SafetensorsConfig {
        hidden_size: Some(768),
        num_hidden_layers: Some(12),
        num_attention_heads: Some(12),
        num_key_value_heads: None, // Not set, should fall back to attention heads
        vocab_size: Some(32000),
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
        ..Default::default()
    };

    assert_eq!(config.num_kv_heads(), 12);
}

#[test]
fn test_cov_safetensors_config_num_kv_heads_fallback() {
    let config = SafetensorsConfig {
        hidden_size: Some(768),
        num_hidden_layers: Some(12),
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: Some(32000),
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
        ..Default::default()
    };

    assert_eq!(config.num_kv_heads(), 1); // Fallback to 1
}

#[test]
fn test_cov_safetensors_config_architecture_from_architectures() {
    let config = SafetensorsConfig {
        hidden_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: None,
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: Some(vec!["LlamaForCausalLM".to_string()]),
        model_type: Some("llama".to_string()),
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
        ..Default::default()
    };

    assert_eq!(config.architecture(), "LlamaForCausalLM");
}

#[test]
fn test_cov_safetensors_config_architecture_from_model_type() {
    let config = SafetensorsConfig {
        hidden_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: None,
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: Some("llama".to_string()),
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
        ..Default::default()
    };

    assert_eq!(config.architecture(), "llama");
}

#[test]
fn test_cov_safetensors_config_architecture_unknown() {
    let config = SafetensorsConfig {
        hidden_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        vocab_size: None,
        intermediate_size: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        architectures: None,
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: None,
        ..Default::default()
    };

    assert_eq!(config.architecture(), "unknown");
}

#[test]
fn test_cov_safetensors_config_load_from_sibling_not_found() {
    let path = std::path::Path::new("/nonexistent/model.safetensors");
    let config = SafetensorsConfig::load_from_sibling(path);
    assert!(config.is_none());
}

#[test]
fn test_cov_metadata_key_skipped() {
    // Test that __metadata__ key is skipped
    let json = r#"{
            "__metadata__":{"format":"pt"},
            "weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}
        }"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let model = SafetensorsModel::from_bytes(&data).expect("test");
    assert_eq!(model.tensors.len(), 1);
    assert!(model.tensors.contains_key("weight"));
    assert!(!model.tensors.contains_key("__metadata__"));
}

#[test]
fn test_cov_json_not_object() {
    // JSON is an array, not an object
    let json = r"[]";
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);

    let result = SafetensorsModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_cov_tensor_metadata_parse_error() {
    // Tensor has invalid metadata (missing dtype)
    let json = r#"{"weight":{"shape":[2],"data_offsets":[0,8]}}"#;
    let json_bytes = json.as_bytes();

    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&[0u8; 8]);

    let result = SafetensorsModel::from_bytes(&data);
    assert!(result.is_err());
}
