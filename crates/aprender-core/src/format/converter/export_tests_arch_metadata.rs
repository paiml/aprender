use super::*;

/// #1865: Missing required dimensions return a clean error, not a panic.
/// (Pre-#1865 these used `.expect()` and were tested with `#[should_panic]`.)
#[test]
fn test_build_gguf_arch_metadata_errors_on_missing_hidden_size() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.hidden_size = None;
    apr.num_layers = Some(32);
    apr.num_heads = Some(32);
    apr.vocab_size = Some(32000);
    apr.intermediate_size = Some(11008);

    let err = build_gguf_arch_metadata(&apr).expect_err("missing hidden_size must error");
    assert!(err.to_string().contains("hidden_size"), "got: {err}");
}

#[test]
fn test_build_gguf_arch_metadata_errors_on_missing_num_layers() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.hidden_size = Some(4096);
    apr.num_layers = None;
    apr.num_heads = Some(32);
    apr.vocab_size = Some(32000);
    apr.intermediate_size = Some(11008);

    let err = build_gguf_arch_metadata(&apr).expect_err("missing num_layers must error");
    assert!(err.to_string().contains("num_layers"), "got: {err}");
}

#[test]
fn test_build_gguf_arch_metadata_errors_on_missing_vocab_size() {
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.hidden_size = Some(4096);
    apr.num_layers = Some(32);
    apr.num_heads = Some(32);
    apr.vocab_size = None;
    apr.intermediate_size = Some(11008);

    let err = build_gguf_arch_metadata(&apr).expect_err("missing vocab_size must error");
    assert!(err.to_string().contains("vocab_size"), "got: {err}");
}

/// C-07: With all required fields populated, export succeeds.
#[test]
fn test_build_gguf_arch_metadata_succeeds_with_required_fields() {
    use crate::format::gguf::GgufValue;
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.architecture = Some("qwen2".to_string());
    apr.hidden_size = Some(4096);
    apr.num_layers = Some(32);
    apr.num_heads = Some(32);
    apr.vocab_size = Some(32000);
    apr.intermediate_size = Some(11008);
    apr.name = Some("model".to_string());

    let entries = build_gguf_arch_metadata(&apr).expect("required fields populated");
    let find = |key: &str| entries.iter().find(|(k, _)| k == key).map(|(_, v)| v);

    match find("qwen2.embedding_length") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 4096),
        other => panic!("Expected Uint32(4096), got: {other:?}"),
    }
    match find("qwen2.block_count") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 32),
        other => panic!("Expected Uint32(32), got: {other:?}"),
    }
}

#[test]
fn test_build_gguf_arch_metadata_zero_heads_uses_default_head_dim() {
    use crate::format::gguf::GgufValue;
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("test");
    apr.architecture = Some("llama".to_string());
    apr.num_heads = Some(0);
    apr.hidden_size = Some(512);
    apr.num_layers = Some(1);
    apr.vocab_size = Some(100);
    apr.intermediate_size = Some(256);

    let entries = build_gguf_arch_metadata(&apr).expect("required fields populated");
    let find = |key: &str| entries.iter().find(|(k, _)| k == key).map(|(_, v)| v);

    // N-02: When num_heads=0, head_dim defaults to 0 (no hidden assumption)
    match find("llama.rope.dimension_count") {
        Some(GgufValue::Uint32(v)) => assert_eq!(*v, 0),
        other => panic!("Expected Uint32(0), got: {other:?}"),
    }
}

#[test]
fn test_build_gguf_arch_metadata_llama_architecture() {
    use crate::format::gguf::GgufValue;
    use crate::format::v2::AprV2Metadata;

    let mut apr = AprV2Metadata::new("llama-model");
    apr.architecture = Some("llama".to_string());
    apr.hidden_size = Some(4096);
    apr.num_layers = Some(32);
    apr.num_heads = Some(32);
    apr.num_kv_heads = Some(8);
    apr.vocab_size = Some(32000);
    apr.intermediate_size = Some(11008);
    apr.name = Some("LLaMA-7B".to_string());

    let entries = build_gguf_arch_metadata(&apr).expect("required fields populated");
    let find = |key: &str| entries.iter().find(|(k, _)| k == key).map(|(_, v)| v);

    // All arch-prefixed keys should use "llama" prefix
    match find("general.architecture") {
        Some(GgufValue::String(s)) => assert_eq!(s, "llama"),
        other => panic!("Expected 'llama', got: {other:?}"),
    }
    assert!(find("llama.context_length").is_some());
    assert!(find("llama.embedding_length").is_some());
    assert!(find("llama.block_count").is_some());
    assert!(find("llama.attention.head_count").is_some());
    assert!(find("llama.attention.head_count_kv").is_some());
    assert!(find("llama.rope.dimension_count").is_some());

    // And none with qwen2 prefix
    assert!(find("qwen2.context_length").is_none());
}

// ========================================================================
// export_apr_to_gguf_raw: Full round-trip test (impact 7.6)
// ========================================================================

#[test]
fn test_export_apr_to_gguf_raw_round_trip() {
    use crate::format::gguf::GgufReader;
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    // Build a minimal APR file with F32 tensors
    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(4);
    metadata.num_layers = Some(1);
    metadata.num_heads = Some(2);
    metadata.num_kv_heads = Some(2);
    metadata.vocab_size = Some(8);
    metadata.intermediate_size = Some(16);
    metadata.max_position_embeddings = Some(2048);
    metadata.rope_theta = Some(10000.0);
    metadata.rms_norm_eps = Some(1e-5);
    metadata.name = Some("test-model".to_string());
    // Add tokenizer custom fields so ValidatedGgufMetadata passes
    metadata
        .custom
        .insert("tokenizer.model".to_string(), serde_json::json!("gpt2"));
    metadata.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!([
            "<|endoftext|>",
            "hello",
            "world",
            "the",
            "a",
            "is",
            ".",
            " "
        ]),
    );
    metadata
        .custom
        .insert("tokenizer.vocab_size".to_string(), serde_json::json!(8));

    let mut writer = AprV2Writer::new(metadata);

    // Add tensors
    let embed_data = vec![0.1f32; 8 * 4]; // [8, 4] embedding
    writer.add_f32_tensor("model.embed_tokens.weight", vec![8, 4], &embed_data);

    let norm_data = vec![1.0f32; 4]; // [4] norm
    writer.add_f32_tensor("model.norm.weight", vec![4], &norm_data);

    let lm_head_data = vec![0.05f32; 8 * 4]; // [8, 4] lm_head
    writer.add_f32_tensor("lm_head.weight", vec![8, 4], &lm_head_data);

    let apr_bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &apr_bytes).expect("write APR file");

    // Export APR to GGUF
    let report = export_apr_to_gguf_raw(&apr_path, &gguf_path).expect("export should succeed");

    assert_eq!(report.tensor_count, 3);
    assert_eq!(report.format, ExportFormat::Gguf);
    // Note: exported_size may be 0 due to BufWriter not flushed before fs::metadata
    assert!(gguf_path.exists());

    // Read the GGUF and verify
    let gguf = GgufReader::from_file(&gguf_path).expect("read GGUF");

    // Check tensor count
    assert_eq!(gguf.tensor_count, 3);

    // Check that tensor names are mapped to GGUF convention
    let tensor_names: Vec<String> = gguf.tensors.iter().map(|t| t.name.clone()).collect();
    assert!(
        tensor_names.contains(&"token_embd.weight".to_string()),
        "Expected token_embd.weight, got: {tensor_names:?}"
    );
    assert!(
        tensor_names.contains(&"output_norm.weight".to_string()),
        "Expected output_norm.weight, got: {tensor_names:?}"
    );
    assert!(
        tensor_names.contains(&"output.weight".to_string()),
        "Expected output.weight, got: {tensor_names:?}"
    );

    // Verify architecture metadata is present
    let arch = gguf.metadata.get("general.architecture");
    assert!(arch.is_some(), "general.architecture should be present");
}

#[test]
fn test_export_apr_to_gguf_raw_missing_file() {
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("nonexistent.apr");
    let gguf_path = dir.path().join("output.gguf");

    let result = export_apr_to_gguf_raw(&apr_path, &gguf_path);
    assert!(result.is_err());
}

#[test]
fn test_export_apr_to_gguf_raw_includes_f16_dtype() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    // Build APR with F16 tensor
    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(4);
    metadata.num_layers = Some(1);
    metadata.num_heads = Some(2);
    metadata.vocab_size = Some(8);
    metadata.intermediate_size = Some(16);
    metadata.name = Some("test".to_string());
    metadata
        .custom
        .insert("tokenizer.model".to_string(), serde_json::json!("gpt2"));
    metadata.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["a", "b", "c", "d", "e", "f", "g", "h"]),
    );

    let mut writer = AprV2Writer::new(metadata);

    // Add F16 tensor
    let f32_data = vec![0.5f32; 8 * 4];
    writer.add_f16_tensor("model.embed_tokens.weight", vec![8, 4], &f32_data);

    // Add F32 tensor
    let norm_data = vec![1.0f32; 4];
    writer.add_f32_tensor("model.norm.weight", vec![4], &norm_data);

    let apr_bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &apr_bytes).expect("write APR file");

    let report = export_apr_to_gguf_raw(&apr_path, &gguf_path).expect("export should succeed");
    assert_eq!(report.tensor_count, 2);
    assert!(gguf_path.exists());
}

#[test]
fn test_export_apr_to_gguf_raw_shape_reversal() {
    use crate::format::gguf::GgufReader;
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(4);
    metadata.num_layers = Some(1);
    metadata.num_heads = Some(2);
    metadata.vocab_size = Some(8);
    metadata.intermediate_size = Some(16);
    metadata.name = Some("test".to_string());
    metadata
        .custom
        .insert("tokenizer.model".to_string(), serde_json::json!("gpt2"));
    metadata.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["a", "b", "c", "d", "e", "f", "g", "h"]),
    );

    let mut writer = AprV2Writer::new(metadata);

    // 2D tensor: APR [rows=8, cols=4]
    let data = vec![0.1f32; 8 * 4];
    writer.add_f32_tensor("model.embed_tokens.weight", vec![8, 4], &data);

    // 1D tensor: stays the same
    let data_1d = vec![1.0f32; 4];
    writer.add_f32_tensor("model.norm.weight", vec![4], &data_1d);

    let apr_bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &apr_bytes).expect("write APR file");

    let _report = export_apr_to_gguf_raw(&apr_path, &gguf_path).expect("export");

    let gguf = GgufReader::from_file(&gguf_path).expect("read GGUF");

    // For the 2D tensor, GGUF shape should be [ne0=4, ne1=8] (reversed)
    let embd_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("find embedding tensor");
    assert_eq!(
        embd_tensor.dims,
        vec![4_u64, 8],
        "2D shape should be reversed for GGUF"
    );

    // For the 1D tensor, shape stays the same
    let norm_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "output_norm.weight")
        .expect("find norm tensor");
    assert_eq!(
        norm_tensor.dims,
        vec![4_u64],
        "1D shape should be unchanged"
    );
}

// ========================================================================
// unfuse_qkv_tensors: Coverage tests (impact 6.6)
// ========================================================================

#[test]
fn test_unfuse_qkv_tensors_no_fused_tensors_passthrough() {
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("fake.apr");

    // Tensors without any qkv_proj should pass through unchanged
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![1.0f32; 16], vec![4, 4]),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![2.0f32; 16], vec![4, 4]),
    );

    let result = unfuse_qkv_tensors(tensors.clone(), &apr_path);
    assert_eq!(result.len(), 2, "non-fused tensors should pass through");
    assert!(result.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(result.contains_key("model.layers.0.self_attn.k_proj.weight"));
}

#[test]
fn test_unfuse_qkv_tensors_no_apr_metadata_returns_original() {
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("nonexistent.apr");

    // Has fused tensor but no APR file -> should return original
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.qkv_proj.weight".to_string(),
        (vec![0.5f32; 48], vec![12, 4]),
    );

    let result = unfuse_qkv_tensors(tensors.clone(), &apr_path);
    // No metadata available (file doesn't exist) -> returns original
    assert_eq!(result.len(), 1);
    assert!(result.contains_key("model.layers.0.self_attn.qkv_proj.weight"));
}

// ========================================================================
// #1865: infer_num_layers_from_tensor_names — apr-export-num-layers-v1
// ========================================================================

#[test]
fn test_infer_num_layers_from_blk_n_names() {
    let names = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.5.ffn_down.weight",
        "blk.27.attn_norm.weight",
        "output_norm.weight",
    ];
    assert_eq!(infer_num_layers_from_tensor_names(&names), Some(28));
}

#[test]
fn test_infer_num_layers_from_hf_model_layers_n() {
    let names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.31.mlp.up_proj.weight",
        "model.norm.weight",
    ];
    assert_eq!(infer_num_layers_from_tensor_names(&names), Some(32));
}

#[test]
fn test_infer_num_layers_returns_none_when_no_block_tensors() {
    let names = ["token_embd.weight", "output_norm.weight", "output.weight"];
    assert_eq!(infer_num_layers_from_tensor_names(&names), None);
}

#[test]
fn test_infer_num_layers_ignores_malformed_indices() {
    // blk.foo.x has a non-numeric index and must be skipped without panicking.
    let names = ["blk.0.attn_q", "blk.foo.weird", "blk.7.ffn_down"];
    assert_eq!(infer_num_layers_from_tensor_names(&names), Some(8));
}

/// FALSIFY-EXPORT-NUM-LAYERS-001: integration smoke for the inference path.
/// Builds an APR file with no `num_layers` metadata but well-formed `blk.N.*`
/// tensor names; `apr export` must succeed (or fail cleanly), never panic.
#[test]
fn test_export_apr_to_gguf_raw_infers_num_layers_when_missing() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("create temp dir");
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(4);
    metadata.num_layers = None; // <-- intentionally missing; must be inferred
    metadata.num_heads = Some(2);
    metadata.vocab_size = Some(8);
    metadata.intermediate_size = Some(16);
    metadata.name = Some("test".to_string());
    metadata
        .custom
        .insert("tokenizer.model".to_string(), serde_json::json!("gpt2"));
    metadata.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["a", "b", "c", "d", "e", "f", "g", "h"]),
    );

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("token_embd.weight", vec![8, 4], &vec![0.1f32; 8 * 4]);
    writer.add_f32_tensor("blk.0.attn_norm.weight", vec![4], &[1.0f32; 4]);
    writer.add_f32_tensor("blk.1.attn_norm.weight", vec![4], &[1.0f32; 4]);
    writer.add_f32_tensor("output_norm.weight", vec![4], &[1.0f32; 4]);

    let apr_bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &apr_bytes).expect("write APR file");

    // Critical: this must NOT panic. Pre-#1865 this aborted with exit 101.
    let result = export_apr_to_gguf_raw(&apr_path, &gguf_path);
    assert!(
        result.is_ok(),
        "export must succeed via num_layers inference; got: {result:?}"
    );
}

/// FT-APRQ8-001: APR→GGUF export must REJECT an AprQ8 tensor (APR-native
/// single-whole-tensor-scale 8-bit, 4+N bytes) instead of silently relabeling
/// it as GGML Q8_0 (per-32-block, ceil(N/32)*34 bytes) — which produced a
/// CORRUPT GGUF that any llama.cpp loader misreads. Symmetric with the existing
/// AprQ4 rejection and the import-side Q8_0 refusal.
#[test]
fn ft_aprq8_001_export_rejects_aprq8_not_silent_corruption() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use tempfile::tempdir;

    let dir = tempdir().expect("temp dir");
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(4);
    metadata.num_layers = Some(1);
    metadata.num_heads = Some(2);
    metadata.num_kv_heads = Some(2);
    metadata.vocab_size = Some(8);
    metadata.intermediate_size = Some(16);
    metadata.max_position_embeddings = Some(2048);
    metadata.rope_theta = Some(10000.0);
    metadata.rms_norm_eps = Some(1e-5);
    metadata.name = Some("test-model".to_string());
    metadata
        .custom
        .insert("tokenizer.model".to_string(), serde_json::json!("gpt2"));
    metadata.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["<|endoftext|>", "hello", "world", "the", "a", "is", ".", " "]),
    );
    metadata
        .custom
        .insert("tokenizer.vocab_size".to_string(), serde_json::json!(8));

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("model.embed_tokens.weight", vec![8, 4], &vec![0.1f32; 32]);
    writer.add_f32_tensor("model.norm.weight", vec![4], &vec![1.0f32; 4]);
    // The offending tensor: an APR-native AprQ8 weight (NOT GGML Q8_0).
    writer.add_q8_tensor("lm_head.weight", vec![8, 4], &vec![0.05f32; 32]);

    let apr_bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &apr_bytes).expect("write APR file");

    let result = export_apr_to_gguf_raw(&apr_path, &gguf_path);
    assert!(
        result.is_err(),
        "AprQ8 export must be REJECTED, not silently relabeled as Q8_0 (corrupt GGUF)"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("AprQ8"),
        "rejection must name AprQ8 to disambiguate from Q8_0; got: {msg}"
    );
}
