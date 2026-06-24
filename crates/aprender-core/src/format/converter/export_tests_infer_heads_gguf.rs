//! OBLIG-APR-GGUF-EXPORT-HEAD-COUNT-SOUND (PMAT-920):
//! `apr export --format gguf` on a metadata-light .apr must derive `num_heads`
//! / `num_kv_heads` EXACTLY from an EXPLICIT `head_dim`
//! (`num_heads = q_dim / head_dim`), and MUST NOT guess `head_dim` from a
//! hardcoded list when it is absent.
//!
//! The original PMAT-920 fix inferred `num_heads` via the
//! `[64, 128, 96, 80]` first-divisor guess in `infer_head_counts`. That
//! SILENTLY mis-stamps real models: Qwen2-1.5B (q_dim=1536, head_dim=128,
//! 12 heads) → the guess picks head_dim=64 first → `1536/64 = 24` heads
//! written into a valid-looking GGUF, no error. A silently-wrong head count is
//! worse than an honest failure.
//!
//! This file falsifies BOTH directions:
//!   (i)  EXPLICIT head_dim present (num_heads absent) → exact head count
//!        `q_dim / head_dim`, using a head_dim=128 fixture where the old
//!        first-divisor guess would have said 64 (so it PROVES we don't guess).
//!   (ii) head_dim AND num_heads BOTH absent → ACTIONABLE hard-fail (names the
//!        missing dim + a working remedy: stamp head_dim/num_heads or convert
//!        from source), NOT a silently-wrong head count.
use super::*;

/// Shared FFN/embedding/norm tensors for a single-layer Qwen2-style model.
/// Attention q/k/v shapes are parameterized so each test can exercise a
/// specific (q_dim, kv_dim) without re-guessing head_dim.
fn add_common_tensors(
    writer: &mut crate::format::v2::AprV2Writer,
    vocab: usize,
    hidden: usize,
    inter: usize,
) {
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        &vec![0.01f32; vocab * hidden],
    );
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
}

fn add_attn_tensors(
    writer: &mut crate::format::v2::AprV2Writer,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
) {
    writer.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        vec![q_dim, hidden],
        &vec![0.01f32; q_dim * hidden],
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
        vec![hidden, q_dim],
        &vec![0.01f32; hidden * q_dim],
    );
}

fn metadata_light(name: &str, vocab: usize) -> crate::format::v2::AprV2Metadata {
    use crate::format::v2::AprV2Metadata;
    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.architecture = Some("qwen2".to_string());
    // All dims intentionally MISSING; tests set head_dim individually.
    metadata.num_heads = None;
    metadata.num_kv_heads = None;
    metadata.hidden_size = None;
    metadata.vocab_size = None;
    metadata.intermediate_size = None;
    metadata.num_layers = None;
    metadata.head_dim = None;
    metadata.name = Some(name.to_string());
    metadata
        .custom
        .insert("tokenizer.model".to_string(), serde_json::json!("gpt2"));
    let vocab_tokens: Vec<String> = (0..vocab).map(|i| format!("tok{i}")).collect();
    metadata.custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(vocab_tokens),
    );
    metadata
}

/// DIRECTION (i): EXPLICIT head_dim present, num_heads absent → exact head
/// count `q_dim / head_dim`.
///
/// Fixture: hidden=q_dim=256, kv_dim=256, EXPLICIT head_dim=128
/// → num_heads = 256/128 = 2, num_kv_heads = 256/128 = 2.
///
/// CRITICAL: the OLD `[64,128,96,80]` first-divisor guess would have picked
/// head_dim=64 → 256/64 = 4 heads (WRONG). Asserting head_count == 2 proves we
/// use the explicit head_dim and do NOT guess.
#[test]
fn ft_apr_gguf_export_uses_explicit_head_dim_exactly_not_guess() {
    use crate::format::gguf::{GgufReader, GgufValue};
    use crate::format::v2::AprV2Writer;
    use tempfile::tempdir;

    let dir = tempdir().expect("temp dir");
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    let hidden = 256usize;
    let q_dim = 256usize;
    let kv_dim = 256usize;
    let inter = 512usize;
    let vocab = 512usize;

    let mut metadata = metadata_light("explicit-head-dim", vocab);
    metadata.head_dim = Some(128); // EXPLICIT, sound. Old guess would say 64.
    let mut writer = AprV2Writer::new(metadata);
    add_common_tensors(&mut writer, vocab, hidden, inter);
    add_attn_tensors(&mut writer, hidden, q_dim, kv_dim);
    let apr_bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &apr_bytes).expect("write APR file");

    let report = export_apr_to_gguf_raw(&apr_path, &gguf_path)
        .expect("APR with explicit head_dim must export to GGUF");
    assert_eq!(report.format, ExportFormat::Gguf);

    let gguf = GgufReader::from_file(&gguf_path).expect("read produced GGUF");

    let head_count = gguf
        .metadata
        .get("qwen2.attention.head_count")
        .expect("qwen2.attention.head_count must be present");
    match head_count {
        GgufValue::Uint32(v) => assert_eq!(
            *v, 2,
            "q_dim=256 / explicit head_dim=128 → num_heads must be EXACTLY 2 \
             (the old [64,128,96,80] guess would wrongly give 256/64 = 4), got {v}"
        ),
        other => panic!("head_count should be Uint32, got {other:?}"),
    }

    let kv_count = gguf
        .metadata
        .get("qwen2.attention.head_count_kv")
        .expect("qwen2.attention.head_count_kv must be present");
    match kv_count {
        GgufValue::Uint32(v) => assert_eq!(
            *v, 2,
            "kv_dim=256 / explicit head_dim=128 → num_kv_heads must be EXACTLY 2, got {v}"
        ),
        other => panic!("head_count_kv should be Uint32, got {other:?}"),
    }

    // hidden/vocab/intermediate stay unambiguously inferable from shapes.
    let emb = gguf
        .metadata
        .get("qwen2.embedding_length")
        .expect("qwen2.embedding_length must be present");
    match emb {
        GgufValue::Uint32(v) => assert_eq!(*v, 256, "hidden_size inferred from embedding, got {v}"),
        other => panic!("embedding_length should be Uint32, got {other:?}"),
    }
}

/// DIRECTION (ii): head_dim AND num_heads BOTH absent → ACTIONABLE hard-fail,
/// NOT a silently-wrong head count.
///
/// This is the regression that the original fix introduced: with the
/// `[64,128,96,80]` guess, a metadata-light .apr exported "successfully" with a
/// fabricated head count. The correct behavior is to refuse and tell the user
/// how to supply the missing dimension.
#[test]
fn ft_apr_gguf_export_hard_fails_when_head_dim_and_num_heads_absent() {
    use crate::format::v2::AprV2Writer;
    use tempfile::tempdir;

    let dir = tempdir().expect("temp dir");
    let apr_path = dir.path().join("model.apr");
    let gguf_path = dir.path().join("model.gguf");

    // hidden=q_dim=1536 (Qwen2-1.5B), kv_dim=256. NO head_dim, NO num_heads.
    let hidden = 1536usize;
    let q_dim = 1536usize;
    let kv_dim = 256usize;
    let inter = 8960usize;
    let vocab = 512usize;

    let metadata = metadata_light("no-head-dim", vocab); // head_dim stays None
    let mut writer = AprV2Writer::new(metadata);
    add_common_tensors(&mut writer, vocab, hidden, inter);
    add_attn_tensors(&mut writer, hidden, q_dim, kv_dim);
    let apr_bytes = writer.write().expect("write APR");
    std::fs::write(&apr_path, &apr_bytes).expect("write APR file");

    let err = export_apr_to_gguf_raw(&apr_path, &gguf_path).expect_err(
        "metadata-light APR with no head_dim and no num_heads must HARD-FAIL, \
         not silently guess a head count",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("num_heads"),
        "error must name the missing dimension (num_heads), got: {msg}"
    );
    assert!(
        msg.contains("head_dim") && msg.contains("apr stamp"),
        "error must be ACTIONABLE — name head_dim and a remedy (apr stamp / convert), got: {msg}"
    );
    // The GGUF must NOT have been written with a fabricated head count.
    assert!(
        !gguf_path.exists(),
        "no GGUF may be produced when the head count cannot be soundly determined"
    );
}
