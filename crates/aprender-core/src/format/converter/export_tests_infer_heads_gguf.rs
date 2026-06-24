//! OBLIG-APR-GGUF-EXPORT-INFER-METADATA (PMAT-920):
//! `apr export --format gguf` on a metadata-light .apr (no explicit `num_heads`,
//! `hidden_size`, `vocab_size`, or `intermediate_size`) must INFER those
//! GGUF-required dimensions from the model's tensor shapes instead of
//! hard-failing with a C-07 missing-dimension error.
//!
//! Before the fix, `export_apr_to_gguf_raw` called
//! `apr_metadata.num_heads.ok_or_else(|| missing_dim_err("num_heads"))?`,
//! so a .apr produced by training / `apr convert` without a fully-populated
//! metadata block could not be exported to GGUF for llama.cpp / ollama at all.
//!
//! The shapes in this fixture are an unambiguous Qwen2-style 0.5B-ish layout:
//! q_proj/o_proj = [hidden, hidden]; k_proj/v_proj = [kv_dim, hidden] (GQA);
//! head_dim is 64 → num_heads = hidden/64, num_kv_heads = kv_dim/64.
use super::*;

/// Build a metadata-light APR (only architecture + tokenizer custom fields)
/// whose tensor shapes imply the attention dimensions. `hidden=128`,
/// `kv_dim=64`, head_dim=64 → num_heads=2, num_kv_heads=1.
fn write_metadata_light_apr(apr_path: &std::path::Path) {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};

    let hidden = 128usize;
    let kv_dim = 64usize; // GQA: 1 kv head of dim 64
    let inter = 256usize;
    // Realistic LLM: vocab >> hidden (embedding shape [vocab, hidden]).
    let vocab = 512usize;

    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.architecture = Some("qwen2".to_string());
    // Intentionally MISSING: num_heads, num_kv_heads, hidden_size,
    // vocab_size, intermediate_size. num_layers stays unset too (already
    // inferred from blk.N.*).
    metadata.num_heads = None;
    metadata.num_kv_heads = None;
    metadata.hidden_size = None;
    metadata.vocab_size = None;
    metadata.intermediate_size = None;
    metadata.num_layers = None;
    metadata.name = Some("metadata-light".to_string());
    metadata
        .custom
        .insert("tokenizer.model".to_string(), serde_json::json!("gpt2"));
    let vocab_tokens: Vec<String> = (0..vocab).map(|i| format!("tok{i}")).collect();
    metadata.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(vocab_tokens),
    );

    let mut writer = AprV2Writer::new(metadata);

    // Embedding [vocab, hidden] → implies vocab_size + hidden_size.
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        &vec![0.01f32; vocab * hidden],
    );
    // Attention projections for layer 0 (HF names).
    writer.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        vec![hidden, hidden],
        &vec![0.01f32; hidden * hidden],
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        vec![kv_dim, hidden],
        &vec![0.01f32; kv_dim * hidden],
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        vec![kv_dim, hidden],
        &vec![0.01f32; kv_dim * hidden],
    );
    writer.add_f32_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        vec![hidden, hidden],
        &vec![0.01f32; hidden * hidden],
    );
    // FFN gate/up → implies intermediate_size.
    writer.add_f32_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        vec![inter, hidden],
        &vec![0.01f32; inter * hidden],
    );
    writer.add_f32_tensor(
        "model.layers.0.mlp.up_proj.weight",
        vec![inter, hidden],
        &vec![0.01f32; inter * hidden],
    );
    writer.add_f32_tensor(
        "model.layers.0.mlp.down_proj.weight",
        vec![hidden, inter],
        &vec![0.01f32; hidden * inter],
    );
    writer.add_f32_tensor("model.norm.weight", vec![hidden], &vec![1.0f32; hidden]);
    writer.add_f32_tensor(
        "lm_head.weight",
        vec![vocab, hidden],
        &vec![0.01f32; vocab * hidden],
    );

    let apr_bytes = writer.write().expect("write APR");
    std::fs::write(apr_path, &apr_bytes).expect("write APR file");
}

/// FALSIFIER (RED on unfixed: hard-fails with C-07 missing num_heads;
/// GREEN on fix: succeeds AND emits correct num_heads/num_kv_heads).
#[test]
fn ft_apr_gguf_export_infers_heads_when_metadata_light() {
    use crate::format::gguf::{GgufReader, GgufValue};
    use tempfile::tempdir;

    let dir = tempdir().expect("temp dir");
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    write_metadata_light_apr(&apr_path);

    // Must NOT hard-fail on missing num_heads — must infer from shapes.
    let report = export_apr_to_gguf_raw(&apr_path, &gguf_path)
        .expect("metadata-light APR must export to GGUF via shape inference, not hard-fail");
    assert_eq!(report.format, ExportFormat::Gguf);
    assert!(gguf_path.exists(), "gguf must be written");

    // Re-read the produced GGUF: the inferred head counts must be correct.
    let gguf = GgufReader::from_file(&gguf_path).expect("read produced GGUF");

    let head_count = gguf
        .metadata
        .get("qwen2.attention.head_count")
        .expect("qwen2.attention.head_count must be present");
    match head_count {
        GgufValue::Uint32(v) => assert_eq!(
            *v, 2,
            "hidden=128, head_dim=64 → num_heads must be inferred as 2, got {v}"
        ),
        other => panic!("head_count should be Uint32, got {other:?}"),
    }

    let kv_count = gguf
        .metadata
        .get("qwen2.attention.head_count_kv")
        .expect("qwen2.attention.head_count_kv must be present");
    match kv_count {
        GgufValue::Uint32(v) => assert_eq!(
            *v, 1,
            "kv_dim=64, head_dim=64 → num_kv_heads must be inferred as 1, got {v}"
        ),
        other => panic!("head_count_kv should be Uint32, got {other:?}"),
    }

    // embedding_length (hidden_size) must also be inferred (not 0).
    let emb = gguf
        .metadata
        .get("qwen2.embedding_length")
        .expect("qwen2.embedding_length must be present");
    match emb {
        GgufValue::Uint32(v) => {
            assert_eq!(*v, 128, "hidden_size must be inferred as 128, got {v}")
        }
        other => panic!("embedding_length should be Uint32, got {other:?}"),
    }
}
