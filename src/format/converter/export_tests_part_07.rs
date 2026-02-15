
// ========================================================================
// GH-258: end-to-end apr_export → GGUF
// ========================================================================

#[test]
fn test_apr_export_to_gguf_end_to_end() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};

    let dir = tempfile::tempdir().expect("temp dir");

    // Create a real APR file
    let mut metadata = AprV2Metadata::new("e2e-test");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(64);
    metadata.vocab_size = Some(256);
    metadata.num_layers = Some(1);
    metadata.num_heads = Some(4);
    metadata.num_kv_heads = Some(2);

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![256, 64],
        &vec![0.1; 256 * 64],
    );
    writer.add_f32_tensor("model.norm.weight", vec![64], &vec![1.0; 64]);
    writer.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![64],
        &vec![1.0; 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.post_attention_layernorm.weight",
        vec![64],
        &vec![1.0; 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.mlp.up_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );
    writer.add_f32_tensor(
        "model.layers.0.mlp.down_proj.weight",
        vec![64, 64],
        &vec![0.01; 64 * 64],
    );

    let apr_path = dir.path().join("model.apr");
    let bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, bytes).expect("write file");

    let output = dir.path().join("output.gguf");
    let options = ExportOptions {
        format: ExportFormat::Gguf,
        quantize: None,
        include_tokenizer: false,
        include_config: false,
    };

    let report = apr_export(&apr_path, &output, options).expect("export should succeed");

    assert_eq!(report.format, ExportFormat::Gguf);
    assert_eq!(report.tensor_count, 11);
    assert!(report.exported_size > 0);
    assert!(report.original_size > 0);
    assert!(output.exists());
}

// ========================================================================
// GH-258: push_string_array / push_u32_field / push_i32_array helpers
// ========================================================================

#[test]
fn test_push_string_array_with_valid_data() {
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert(
        "my_tokens".to_string(),
        serde_json::json!(["hello", "world"]),
    );

    push_string_array(&mut entries, &custom, "my_tokens", "tokenizer.ggml.tokens");

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, "tokenizer.ggml.tokens");
}

#[test]
fn test_push_string_array_missing_key_no_op() {
    let mut entries = Vec::new();
    let custom = std::collections::HashMap::new();

    push_string_array(&mut entries, &custom, "missing", "tokenizer.ggml.tokens");
    assert!(entries.is_empty());
}

#[test]
fn test_push_string_array_empty_array_no_op() {
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("empty".to_string(), serde_json::json!([]));

    push_string_array(&mut entries, &custom, "empty", "tokenizer.ggml.tokens");
    assert!(entries.is_empty(), "empty array should not add entry");
}

#[test]
fn test_push_u32_field_with_valid_data() {
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("bos".to_string(), serde_json::json!(42));

    push_u32_field(&mut entries, &custom, "bos", "tokenizer.ggml.bos_token_id");

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, "tokenizer.ggml.bos_token_id");
}

#[test]
fn test_push_u32_field_missing_key_no_op() {
    let mut entries = Vec::new();
    let custom = std::collections::HashMap::new();

    push_u32_field(&mut entries, &custom, "missing", "key");
    assert!(entries.is_empty());
}

#[test]
fn test_push_u32_field_non_numeric_no_op() {
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("not_num".to_string(), serde_json::json!("hello"));

    push_u32_field(&mut entries, &custom, "not_num", "key");
    assert!(entries.is_empty());
}

#[test]
fn test_push_i32_array_with_valid_data() {
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("types".to_string(), serde_json::json!([1, 3, 1, 1]));

    push_i32_array(&mut entries, &custom, "types", "tokenizer.ggml.token_type");

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, "tokenizer.ggml.token_type");
}

#[test]
fn test_push_i32_array_empty_array_no_op() {
    let mut entries = Vec::new();
    let mut custom = std::collections::HashMap::new();
    custom.insert("empty".to_string(), serde_json::json!([]));

    push_i32_array(&mut entries, &custom, "empty", "key");
    assert!(entries.is_empty());
}

// ========================================================================
// GH-258: extract_apr_tokenizer_for_gguf edge cases
// ========================================================================

#[test]
fn test_extract_apr_tokenizer_maps_bpe_to_gpt2() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.custom
        .insert("tokenizer.model".to_string(), serde_json::json!("bpe"));
    apr.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["a", "b"]),
    );

    let entries = extract_apr_tokenizer_for_gguf(&apr);
    let model_entry = entries.iter().find(|(k, _)| k == "tokenizer.ggml.model");
    assert!(model_entry.is_some());
    // "bpe" should be mapped to "gpt2"
    if let Some((_, crate::format::gguf::GgufValue::String(val))) = model_entry {
        assert_eq!(val, "gpt2", "bpe should be mapped to gpt2");
    }
}

#[test]
fn test_extract_apr_tokenizer_includes_chat_template() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.chat_template = Some("{% for msg in messages %}...{% endfor %}".to_string());

    let entries = extract_apr_tokenizer_for_gguf(&apr);
    let tmpl = entries.iter().find(|(k, _)| k == "tokenizer.chat_template");
    assert!(tmpl.is_some(), "should include chat template from metadata");
}

#[test]
fn test_extract_apr_tokenizer_chat_template_from_custom() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    // No chat_template field, but has it in custom
    apr.custom.insert(
        "tokenizer.chat_template".to_string(),
        serde_json::json!("template_str"),
    );

    let entries = extract_apr_tokenizer_for_gguf(&apr);
    let tmpl = entries.iter().find(|(k, _)| k == "tokenizer.chat_template");
    assert!(tmpl.is_some(), "should find chat template in custom fields");
}

#[test]
fn test_extract_apr_tokenizer_add_bos_token() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.custom.insert(
        "tokenizer.add_bos_token".to_string(),
        serde_json::json!(true),
    );

    let entries = extract_apr_tokenizer_for_gguf(&apr);
    let bos = entries
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.add_bos_token");
    assert!(bos.is_some(), "should include add_bos_token flag");
}

// ========================================================================
// GH-258: resolve_architecture
// ========================================================================

#[test]
fn test_resolve_architecture_from_architecture_field() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.architecture = Some("llama".to_string());

    assert_eq!(resolve_architecture(&apr), "llama");
}

#[test]
fn test_resolve_architecture_falls_back_to_model_type() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.architecture = None;
    apr.model_type = "phi".to_string();

    assert_eq!(resolve_architecture(&apr), "phi");
}

#[test]
fn test_resolve_architecture_ignores_unknown_model_type() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.architecture = None;
    apr.model_type = "unknown".to_string();

    assert_eq!(
        resolve_architecture(&apr),
        "qwen2",
        "unknown model_type should default to qwen2"
    );
}

#[test]
fn test_resolve_architecture_ignores_empty_model_type() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.architecture = None;
    apr.model_type = String::new();

    assert_eq!(
        resolve_architecture(&apr),
        "qwen2",
        "empty model_type should default to qwen2"
    );
}

// ========================================================================
// build_gguf_arch_metadata: Coverage tests (impact 22.8)
// ========================================================================

#[test]
fn test_build_gguf_arch_metadata_with_all_fields_populated() {
    use crate::format::gguf::GgufValue;
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test-model");
    apr.architecture = Some("qwen2".to_string());
    apr.hidden_size = Some(896);
    apr.num_layers = Some(24);
    apr.num_heads = Some(14);
    apr.num_kv_heads = Some(2);
    apr.vocab_size = Some(151936);
    apr.intermediate_size = Some(4864);
    apr.max_position_embeddings = Some(32768);
    apr.rope_theta = Some(1_000_000.0);
    apr.rms_norm_eps = Some(1e-6);
    apr.name = Some("Qwen2.5-0.5B".to_string());

    let entries = build_gguf_arch_metadata(&apr);

    // Verify all expected keys are present
    let find = |key: &str| entries.iter().find(|(k, _)| k == key).map(|(_, v)| v);

    // general.architecture
    match find("general.architecture") {
        Some(GgufValue::String(s)) => assert_eq!(s, "qwen2"),
        other => panic!("Expected String 'qwen2', got: {other:?}"),
    }

    // general.name
    match find("general.name") {
        Some(GgufValue::String(s)) => assert_eq!(s, "Qwen2.5-0.5B"),
        other => panic!("Expected String 'Qwen2.5-0.5B', got: {other:?}"),
    }

    // general.quantization_version
    match find("general.quantization_version") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 2),
        other => panic!("Expected Uint32(2), got: {other:?}"),
    }

    // qwen2.context_length
    match find("qwen2.context_length") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 32768),
        other => panic!("Expected Uint32(32768), got: {other:?}"),
    }

    // qwen2.embedding_length
    match find("qwen2.embedding_length") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 896),
        other => panic!("Expected Uint32(896), got: {other:?}"),
    }

    // qwen2.block_count
    match find("qwen2.block_count") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 24),
        other => panic!("Expected Uint32(24), got: {other:?}"),
    }

    // qwen2.feed_forward_length
    match find("qwen2.feed_forward_length") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 4864),
        other => panic!("Expected Uint32(4864), got: {other:?}"),
    }

    // qwen2.attention.head_count
    match find("qwen2.attention.head_count") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 14),
        other => panic!("Expected Uint32(14), got: {other:?}"),
    }

    // qwen2.attention.head_count_kv
    match find("qwen2.attention.head_count_kv") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 2),
        other => panic!("Expected Uint32(2), got: {other:?}"),
    }

    // qwen2.attention.layer_norm_rms_epsilon
    match find("qwen2.attention.layer_norm_rms_epsilon") {
        Some(GgufValue::Float32(v)) => assert!((*v - 1e-6).abs() < 1e-10),
        other => panic!("Expected Float32(1e-6), got: {other:?}"),
    }

    // qwen2.rope.freq_base
    match find("qwen2.rope.freq_base") {
        Some(GgufValue::Float32(v)) => assert!((*v - 1_000_000.0).abs() < 1.0),
        other => panic!("Expected Float32(1000000.0), got: {other:?}"),
    }

    // qwen2.rope.dimension_count = hidden_size / num_heads = 896/14 = 64
    match find("qwen2.rope.dimension_count") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 64),
        other => panic!("Expected Uint32(64), got: {other:?}"),
    }

    // qwen2.vocab_size
    match find("qwen2.vocab_size") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 151936),
        other => panic!("Expected Uint32(151936), got: {other:?}"),
    }

    // Verify total count: 14 entries
    assert_eq!(entries.len(), 14);
}
