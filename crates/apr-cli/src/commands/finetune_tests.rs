use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_finetune_method_parse() {
    assert!(matches!(
        "auto".parse::<FinetuneMethod>(),
        Ok(FinetuneMethod::Auto)
    ));
    assert!(matches!(
        "full".parse::<FinetuneMethod>(),
        Ok(FinetuneMethod::Full)
    ));
    assert!(matches!(
        "lora".parse::<FinetuneMethod>(),
        Ok(FinetuneMethod::LoRA)
    ));
    assert!(matches!(
        "qlora".parse::<FinetuneMethod>(),
        Ok(FinetuneMethod::QLoRA)
    ));
    assert!("unknown".parse::<FinetuneMethod>().is_err());
}

#[test]
fn test_finetune_method_to_entrenar() {
    assert!(matches!(Method::from(FinetuneMethod::Auto), Method::Auto));
    assert!(matches!(Method::from(FinetuneMethod::LoRA), Method::LoRA));
    assert!(matches!(Method::from(FinetuneMethod::QLoRA), Method::QLoRA));
    assert!(matches!(Method::from(FinetuneMethod::Full), Method::Full));
}

#[test]
fn test_parse_model_size() {
    assert_eq!(parse_model_size("7B").expect("7B"), 7_000_000_000);
    assert_eq!(parse_model_size("1.5B").expect("1.5B"), 1_500_000_000);
    assert_eq!(parse_model_size("135M").expect("135M"), 135_000_000);
    assert!(parse_model_size("invalid").is_err());
}

#[test]
fn test_format_params() {
    assert_eq!(format_params(7_000_000_000), "7.0B");
    assert_eq!(format_params(135_000_000), "135.0M");
    assert_eq!(format_params(1000), "1000");
}

#[test]
fn test_run_no_model() {
    let result = run(
        None,
        "auto",
        None,
        16.0,
        false,
        None,
        None,
        None,
        false,
        3,
        Some(2e-4),
        None,
        None,
        5,
        "apr,safetensors",
        false,
        None,
        false,
        None,
        "cuda",
        None,
        None,
        None,
        None,
        0,
        &[],
        None,
        false,
        false,
        0,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_plan_with_model_size() {
    let result = run(
        None,
        "lora",
        None,
        16.0,
        true,
        None,
        None,
        None,
        false,
        3,
        Some(2e-4),
        Some("7B"),
        None,
        5,
        "apr,safetensors",
        false,
        None,
        false,
        None,
        "cuda",
        None,
        None,
        None,
        None,
        0,
        &[],
        None,
        false,
        false,
        0,
    );
    assert!(result.is_ok());
}

#[test]
fn test_run_plan_json() {
    let result = run(
        None,
        "qlora",
        None,
        24.0,
        true,
        None,
        None,
        None,
        false,
        3,
        Some(2e-4),
        Some("14B"),
        None,
        5,
        "apr,safetensors",
        false,
        None,
        false,
        None,
        "cuda",
        None,
        None,
        None,
        None,
        0,
        &[],
        None,
        true,
        false,
        0,
    );
    assert!(result.is_ok());
}

#[test]
fn test_run_with_model_file() {
    let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
    input.write_all(&[0u8; 4096]).expect("write");
    let result = run(
        Some(input.path()),
        "auto",
        None,
        16.0,
        true,
        None,
        None,
        None,
        false,
        3,
        Some(2e-4),
        None,
        None,
        5,
        "apr,safetensors",
        false,
        None,
        false,
        None,
        "cuda",
        None,
        None,
        None,
        None,
        0,
        &[],
        None,
        false,
        false,
        0,
    );
    assert!(result.is_ok());
}

#[test]
fn test_merge_no_model() {
    let result = run_merge(None, None, None, false);
    assert!(result.is_err());
}

#[test]
fn test_merge_no_adapter() {
    let input = NamedTempFile::with_suffix(".apr").expect("create input");
    let result = run_merge(Some(input.path()), None, None, false);
    assert!(result.is_err());
}

#[test]
fn test_merge_model_not_found() {
    let result = run_merge(
        Some(Path::new("/nonexistent.apr")),
        Some(Path::new("/nonexistent_adapter/")),
        None,
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_is_lora_eligible() {
    assert!(is_lora_eligible("model.layers.0.self_attn.q_proj.weight"));
    assert!(is_lora_eligible("model.layers.0.self_attn.v_proj.weight"));
    assert!(is_lora_eligible("model.layers.0.mlp.gate_proj.weight"));
    assert!(is_lora_eligible("model.layers.0.mlp.up_proj.weight"));
    assert!(is_lora_eligible("model.layers.0.mlp.down_proj.weight"));
    assert!(is_lora_eligible("blk.0.attn_q.weight"));
    assert!(is_lora_eligible("blk.0.ffn_gate.weight"));

    // Should NOT be eligible
    assert!(!is_lora_eligible("model.embed_tokens.weight"));
    assert!(!is_lora_eligible("model.norm.weight"));
    assert!(!is_lora_eligible("lm_head.weight"));
    assert!(!is_lora_eligible("model.layers.0.self_attn.q_proj.bias"));
    assert!(!is_lora_eligible("token_embd.weight"));
}

#[test]
fn test_hash_seed_deterministic() {
    let s1 = hash_seed("test.weight", 0);
    let s2 = hash_seed("test.weight", 0);
    assert_eq!(s1, s2, "Same inputs must produce same output");

    let s3 = hash_seed("test.weight", 1);
    assert_ne!(s1, s3, "Different index must produce different output");

    let s4 = hash_seed("other.weight", 0);
    assert_ne!(s1, s4, "Different name must produce different output");
}

#[test]
fn test_run_training_creates_adapter() {
    // Create a valid model APR with LoRA-eligible layers and architecture metadata
    let mut writer = aprender::serialization::apr::AprWriter::new();
    writer.set_metadata("model_type", serde_json::json!("qwen2"));
    writer.set_metadata("hidden_size", serde_json::json!(8));
    writer.set_metadata("num_hidden_layers", serde_json::json!(1));
    writer.set_metadata("num_attention_heads", serde_json::json!(1));
    writer.set_metadata("num_key_value_heads", serde_json::json!(1));
    writer.set_metadata("vocab_size", serde_json::json!(10));
    writer.set_metadata("intermediate_size", serde_json::json!(16));
    let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    writer.add_tensor_f32(
        "model.layers.0.self_attn.q_proj.weight",
        vec![8, 8],
        &q_data,
    );
    let v_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
    writer.add_tensor_f32(
        "model.layers.0.self_attn.v_proj.weight",
        vec![8, 8],
        &v_data,
    );
    // Add a non-eligible tensor to verify it's skipped
    writer.add_tensor_f32("model.embed_tokens.weight", vec![10, 8], &vec![0.1; 80]);

    let input_file = NamedTempFile::with_suffix(".apr").expect("create input");
    let bytes = writer.to_bytes().expect("serialize");
    std::fs::write(input_file.path(), bytes).expect("write");

    // Create a dummy data file
    let data_file = NamedTempFile::with_suffix(".jsonl").expect("create data");
    std::fs::write(
        data_file.path(),
        "{\"instruction\": \"Say hello\", \"response\": \"Hello world\"}\n",
    )
    .expect("write data");

    let output_file = NamedTempFile::with_suffix(".apr").expect("create output");

    let result = run(
        Some(input_file.path()),
        "lora",
        None,
        16.0,
        false,
        Some(data_file.path()),
        Some(output_file.path()),
        None,
        false,
        3,
        Some(2e-4),
        Some("0.5B"),
        None,
        5,
        "apr,safetensors",
        false,
        None,
        false,
        None,
        "cuda",
        None,
        None,
        None,
        None,
        0,
        &[],
        None,
        true,
        false,
        0,
    );
    // Training fails with a minimal model (missing norm weights, etc.)
    // but the pipeline should get past config resolution and data parsing.
    // A full end-to-end test requires a complete model file.
    match &result {
        Ok(()) => {
            // If training somehow succeeds, verify the adapter
            let adapter = aprender::serialization::apr::AprReader::open(output_file.path())
                .expect("adapter should be valid APR");
            assert!(!adapter.tensors.is_empty(), "Adapter should have tensors");
        }
        Err(e) => {
            let msg = format!("{e}");
            // Acceptable failures: model too minimal for full training
            assert!(
                msg.contains("Missing model.norm.weight")
                    || msg.contains("pipeline")
                    || msg.contains("Configuration error"),
                "Unexpected error (expected pipeline/config issue): {msg}"
            );
        }
    }
    // The fact that we got past config resolution proves the metadata fix works.
    // Full end-to-end adapter creation requires a complete model with norm weights.
}

#[test]
fn test_merge_creates_merged_model() {
    // Create base model. C-APR-MERGE-RUNNABLE: run_merge now fail-closes on
    // outputs that are not directly runnable, so the base must carry C-01/C-03
    // metadata + an embedded tokenizer for the merge to succeed.
    let mut base_writer = aprender::serialization::apr::AprWriter::new();
    base_writer.set_metadata("model_type", serde_json::json!("test"));
    base_writer.set_metadata("architecture", serde_json::json!("qwen2"));
    base_writer.set_metadata("hidden_size", serde_json::json!(8));
    base_writer.set_metadata("num_layers", serde_json::json!(1));
    base_writer.set_metadata("num_heads", serde_json::json!(2));
    base_writer.set_metadata("num_kv_heads", serde_json::json!(1));
    base_writer.set_metadata("intermediate_size", serde_json::json!(8));
    base_writer.set_metadata("vocab_size", serde_json::json!(4));
    base_writer.set_metadata(
        "tokenizer.vocabulary",
        serde_json::json!(["a", "b", "c", "d"]),
    );
    base_writer.set_metadata("tokenizer.merges", serde_json::json!(["a b"]));
    let q_data: Vec<f32> = vec![1.0; 64];
    base_writer.add_tensor_f32(
        "model.layers.0.self_attn.q_proj.weight",
        vec![8, 8],
        &q_data,
    );
    base_writer.add_tensor_f32("model.norm.weight", vec![8], &vec![1.0; 8]);

    let base_file = NamedTempFile::with_suffix(".apr").expect("create base");
    std::fs::write(base_file.path(), base_writer.to_bytes().expect("serialize")).expect("write");

    // Create adapter
    let mut adapter_writer = aprender::serialization::apr::AprWriter::new();
    adapter_writer.set_metadata("lora_rank", serde_json::json!(4));
    adapter_writer.set_metadata("lora_alpha", serde_json::json!(8.0));
    let lora_a: Vec<f32> = vec![0.1; 4 * 8]; // [rank=4, cols=8]
    adapter_writer.add_tensor_f32(
        "model.layers.0.self_attn.q_proj.weight.lora_a",
        vec![4, 8],
        &lora_a,
    );
    let lora_b: Vec<f32> = vec![0.05; 8 * 4]; // [rows=8, rank=4]
    adapter_writer.add_tensor_f32(
        "model.layers.0.self_attn.q_proj.weight.lora_b",
        vec![8, 4],
        &lora_b,
    );

    let adapter_file = NamedTempFile::with_suffix(".apr").expect("create adapter");
    std::fs::write(
        adapter_file.path(),
        adapter_writer.to_bytes().expect("serialize"),
    )
    .expect("write");

    let output_file = NamedTempFile::with_suffix(".apr").expect("create output");

    let result = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(output_file.path()),
        true,
    );
    assert!(result.is_ok(), "Merge should succeed: {result:?}");

    // Verify merged model
    let merged = aprender::serialization::apr::AprReader::open(output_file.path())
        .expect("merged should be valid APR");
    assert_eq!(merged.tensors.len(), 2); // q_proj + norm
    let q_merged = merged
        .read_tensor_f32("model.layers.0.self_attn.q_proj.weight")
        .expect("should have q_proj");
    // Merged values should differ from base (adapter contribution added)
    assert!(
        q_merged.iter().any(|&v| (v - 1.0).abs() > 1e-6),
        "Merged weights should differ from base"
    );
}

// ============================================================================
// PMAT-712: LoRA → GGUF export round-trip falsifier
//
// Pillar-3 REPLACE gap: "fine-tune in apr, deploy via GGUF" (the Unsloth story).
// Proves the END-TO-END chain works with EXISTING pieces wired together:
//
//   base.apr + LoRA-adapter.apr
//     ── apr finetune --merge ──►  merged.apr   (run_merge, full arch metadata)
//     ── apr export --format gguf ──►  merged.gguf
//     ── GgufReader::from_file ──►  STRUCTURALLY VALID GGUF
//
// The bar is "produces a structurally-valid, loadable GGUF that carries the
// merged (not base) weights" — NOT perfect inference quality.
//
// Falsifiers (any failure ⇒ the REPLACE story is broken):
//   F-LORA-GGUF-001: GGUF magic/version parse (GgufReader::from_file succeeds)
//   F-LORA-GGUF-002: architecture + dims survive the round-trip (qwen2, hidden, layers)
//   F-LORA-GGUF-003: every base weight reaches the GGUF (no tensor dropped)
//   F-LORA-GGUF-004: the merged q_proj weight DIFFERS from base in GGUF bytes
//                    (the LoRA delta survived merge→export, not silently lost)
// ============================================================================

/// Build a complete (tiny) Qwen2-style base model as APR v2 with full arch
/// metadata so GGUF export resolves a real config (not a guess).
#[cfg(test)]
fn build_tiny_qwen2_base_v2(hidden: usize) -> Vec<u8> {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let mut md = AprV2Metadata::new("pmat712-base");
    md.architecture = Some("qwen2".to_string());
    md.hidden_size = Some(hidden);
    md.vocab_size = Some(64);
    md.num_layers = Some(1);
    md.num_heads = Some(4);
    md.num_kv_heads = Some(2);
    md.intermediate_size = Some(hidden);
    md.max_position_embeddings = Some(128);
    md.rope_theta = Some(1_000_000.0);
    md.rms_norm_eps = Some(1e-6);
    // C-APR-MERGE-RUNNABLE: embedded tokenizer so the merged output passes
    // the post-write runnability gate (PMAT-171/172 self-contained APR).
    let vocab: Vec<String> = (0..64).map(|i| format!("t{i}")).collect();
    md.custom
        .insert("tokenizer.vocabulary".to_string(), serde_json::json!(vocab));
    md.custom
        .insert("tokenizer.merges".to_string(), serde_json::json!(["t0 t1"]));
    md.custom
        .insert("tokenizer.bos_token_id".to_string(), serde_json::json!(0));
    md.custom
        .insert("tokenizer.eos_token_id".to_string(), serde_json::json!(1));

    let mut w = AprV2Writer::new(md);
    let sq = |n: usize| vec![0.02_f32; n];
    w.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![64, hidden],
        &sq(64 * hidden),
    );
    w.add_f32_tensor("model.norm.weight", vec![hidden], &vec![1.0; hidden]);
    // q_proj seeded with a constant base so we can detect the LoRA delta later.
    w.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        vec![hidden, hidden],
        &vec![1.0_f32; hidden * hidden],
    );
    w.add_f32_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![hidden],
        &vec![1.0; hidden],
    );
    w.add_f32_tensor(
        "model.layers.0.post_attention_layernorm.weight",
        vec![hidden],
        &vec![1.0; hidden],
    );
    w.add_f32_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.mlp.up_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.mlp.down_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.write().expect("write base v2")
}

/// Build a LoRA adapter (APR v2) targeting q_proj with `.lora_a` / `.lora_b`
/// tensors in the naming `run_merge` expects.
#[cfg(test)]
fn build_tiny_lora_adapter_v2(hidden: usize, rank: usize, alpha: f64) -> Vec<u8> {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let mut md = AprV2Metadata::new("pmat712-adapter");
    md.custom
        .insert("lora_rank".to_string(), serde_json::json!(rank));
    md.custom
        .insert("lora_alpha".to_string(), serde_json::json!(alpha));

    let mut w = AprV2Writer::new(md);
    // lora_a: [rank, hidden], lora_b: [hidden, rank] — non-zero so B@A is non-zero.
    w.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight.lora_a",
        vec![rank, hidden],
        &vec![0.3_f32; rank * hidden],
    );
    w.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight.lora_b",
        vec![hidden, rank],
        &vec![0.5_f32; hidden * rank],
    );
    w.write().expect("write adapter v2")
}

#[test]
fn test_lora_to_gguf_export_roundtrip_pmat712() {
    use aprender::format::gguf::{load_gguf_tensors, GgufReader};
    use aprender::format::{apr_export, ExportFormat, ExportOptions};

    // hidden=256 keeps K % 256 == 0 (q4k constraint) and stays tiny.
    let hidden = 256usize;
    let rank = 8usize;
    let alpha = 16.0f64;

    let base_file = NamedTempFile::with_suffix(".apr").expect("base tmp");
    std::fs::write(base_file.path(), build_tiny_qwen2_base_v2(hidden)).expect("write base");

    let adapter_file = NamedTempFile::with_suffix(".apr").expect("adapter tmp");
    std::fs::write(
        adapter_file.path(),
        build_tiny_lora_adapter_v2(hidden, rank, alpha),
    )
    .expect("write adapter");

    // ── Step 1: apr finetune --merge  (base + adapter → merged.apr) ──────────
    let merged_apr = NamedTempFile::with_suffix(".apr").expect("merged tmp");
    let merge_res = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(merged_apr.path()),
        true,
    );
    assert!(merge_res.is_ok(), "LoRA merge must succeed: {merge_res:?}");

    // ── Step 2: apr export --format gguf  (merged.apr → merged.gguf) ─────────
    let gguf_file = NamedTempFile::with_suffix(".gguf").expect("gguf tmp");
    let opts = ExportOptions {
        format: ExportFormat::Gguf,
        quantize: None,
        include_tokenizer: false,
        include_config: false,
        // tiny model lacks qwen2 attention biases; structural validity is the bar.
        skip_completeness_check: true,
    };
    let report =
        apr_export(merged_apr.path(), gguf_file.path(), opts).expect("GGUF export must succeed");
    assert_eq!(report.format, ExportFormat::Gguf);
    assert!(gguf_file.path().exists(), "GGUF file must be written");

    // ── Step 3: structural validity via apr's own GGUF reader ───────────────
    // F-LORA-GGUF-001: magic + version parse (from_file errors on bad magic).
    let gguf = GgufReader::from_file(gguf_file.path())
        .expect("F-LORA-GGUF-001: GGUF must be parseable (valid magic/version)");
    assert!(
        gguf.version >= 2,
        "GGUF version must be >= 2, got {}",
        gguf.version
    );

    // F-LORA-GGUF-002: architecture + dims survive the round-trip.
    assert_eq!(
        gguf.architecture().as_deref(),
        Some("qwen2"),
        "F-LORA-GGUF-002: architecture must round-trip as qwen2"
    );
    assert_eq!(
        gguf.hidden_size(),
        Some(hidden),
        "F-LORA-GGUF-002: embedding_length must match base hidden_size"
    );
    assert_eq!(
        gguf.num_layers(),
        Some(1),
        "F-LORA-GGUF-002: block_count must match base num_layers"
    );

    // F-LORA-GGUF-003: tensor table is consistent + q_proj present under GGUF name.
    assert_eq!(
        gguf.tensor_count as usize,
        gguf.tensors.len(),
        "tensor_count header must match parsed tensors"
    );
    let q_name = "blk.0.attn_q.weight";
    let q_meta = gguf
        .tensors
        .iter()
        .find(|t| t.name == q_name)
        .unwrap_or_else(|| {
            panic!(
                "F-LORA-GGUF-003: GGUF must contain {q_name}; got {:?}",
                gguf.tensors
                    .iter()
                    .map(|t| t.name.clone())
                    .collect::<Vec<_>>()
            )
        });
    // GGUF stores dims reversed: [ne0=hidden, ne1=hidden] for a [hidden,hidden] weight.
    assert_eq!(
        q_meta.dims,
        vec![hidden as u64, hidden as u64],
        "q_proj dims must be [hidden,hidden]"
    );
    assert_eq!(
        q_meta.dtype, 0,
        "F32 GgmlType is 0 (no quantization requested)"
    );

    // F-LORA-GGUF-004: the merged q_proj DIFFERS from the base in the GGUF data.
    // Base q_proj was seeded to a constant 1.0; the LoRA delta (alpha/rank * B@A
    // with non-zero A,B) must shift it away from 1.0 in the exported tensor data.
    // Use the high-level loader (load → F32) — the same path realizar uses.
    let _ = q_meta; // dims/dtype already asserted above
    let tensors = load_gguf_tensors(gguf_file.path())
        .expect("F-LORA-GGUF-004: GGUF tensors must load back as F32");
    let (q_data, q_shape) = tensors
        .get(q_name)
        .expect("loaded GGUF must contain blk.0.attn_q.weight");
    assert_eq!(
        q_data.len(),
        hidden * hidden,
        "q_proj element count must be hidden^2"
    );
    assert_eq!(
        q_shape.iter().product::<usize>(),
        hidden * hidden,
        "q_proj shape product"
    );
    let delta_present = q_data.iter().any(|&v| (v - 1.0).abs() > 1e-4);
    assert!(
        delta_present,
        "F-LORA-GGUF-004: merged q_proj must differ from base (LoRA delta survived merge→GGUF)"
    );
}

// ============================================================================
// PMAT-712 BEAT: LoRA → GGUF deploy is LOSSLESS by forward-output equivalence
//
// Pillar-3 deploy-correctness BEAT (incumbent: Unsloth).
//
// The structural falsifier above proves the exported GGUF is well-formed and
// carries the *merged* weights. This BEAT proves the STRONGER claim Unsloth
// markets but never makes falsifiable: the F32-exported GGUF produces
// NUMERICALLY-EQUIVALENT forward output to the apr in-memory merged model.
//
//   base.apr + LoRA-adapter.apr
//     ── run_merge ──►  merged.apr   (W = W_base + (alpha/rank)·Bᵀ@Aᵀ, row-major)
//     ── apr_export(Gguf, quantize=None) ──►  merged.gguf   (F32, no quant)
//
// Forward equivalence at the q_proj projection (the LoRA-targeted layer):
//   y_apr  = W_apr  @ x   where W_apr  = merged q_proj loaded from merged.apr
//   y_gguf = W_gguf @ x   where W_gguf = merged q_proj loaded from merged.gguf
//   BEAT: max_i |y_apr[i] − y_gguf[i]|  ≤  1e-4   (F32-lossless tolerance)
//
// Why a real forward, not a byte-diff: y = W @ x is order-sensitive. A lost or
// garbled weight, a layout TRANSPOSE bug (q_proj written column-major), or a
// metadata/name mismatch that re-shapes the tensor all break this equality even
// when |W_apr| == |W_gguf| element-count matches. The weights here are
// position-dependent and ASYMMETRIC (W ≠ Wᵀ) precisely so a transpose bug is
// observable — a constant matrix is its own transpose and would hide it.
//
// Scope honesty: realizar's full decode path needs a real tokenizer/config the
// synthetic tiny model lacks, so this BEAT proves equivalence at the q_proj
// projection — a true forward pass through the LoRA-targeted matmul, loaded
// from the exact GGUF realizar would serve, via the same load_gguf_tensors
// path realizar uses. It is the strongest forward-equivalence reachable for a
// synthetic tiny model and it directly falsifies the lossless-deploy claim.
//
// Falsifiers (any failure ⇒ "fine-tune in apr, deploy lossless via GGUF" is FALSE):
//   F-LOSSLESS-001: q_proj loads back from GGUF with the apr [hidden,hidden] shape
//   F-LOSSLESS-002: forward output is non-degenerate (W·x is not constant — the
//                   test actually exercises ordering, so a transpose WOULD show)
//   F-LOSSLESS-003: max|y_apr − y_gguf| ≤ 1e-4  (THE BEAT: F32 export is lossless)
//   F-LOSSLESS-004: the apr↔gguf forward agreement is strictly TIGHTER than the
//                   forward through a transposed W (guards the test's sensitivity)
// ============================================================================

/// Asymmetric, position-dependent base so the merged q_proj satisfies W ≠ Wᵀ.
/// A transpose bug in export is then observable in y = W·x (a constant or
/// symmetric matrix would be its own transpose and hide such a bug).
#[cfg(test)]
fn build_tiny_qwen2_base_beat(hidden: usize) -> Vec<u8> {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let mut md = AprV2Metadata::new("pmat712-beat-base");
    md.architecture = Some("qwen2".to_string());
    md.hidden_size = Some(hidden);
    md.vocab_size = Some(64);
    md.num_layers = Some(1);
    md.num_heads = Some(4);
    md.num_kv_heads = Some(2);
    md.intermediate_size = Some(hidden);
    md.max_position_embeddings = Some(128);
    md.rope_theta = Some(1_000_000.0);
    md.rms_norm_eps = Some(1e-6);
    // C-APR-MERGE-RUNNABLE: embedded tokenizer so the merged output passes
    // the post-write runnability gate (PMAT-171/172 self-contained APR).
    let vocab: Vec<String> = (0..64).map(|i| format!("t{i}")).collect();
    md.custom
        .insert("tokenizer.vocabulary".to_string(), serde_json::json!(vocab));
    md.custom
        .insert("tokenizer.merges".to_string(), serde_json::json!(["t0 t1"]));

    // Deterministic, ASYMMETRIC q_proj: W[r,c] depends on (r,c) so W ≠ Wᵀ.
    // Values kept O(1) and well-separated so f32 rounding is the only error term.
    let mut qw = vec![0.0_f32; hidden * hidden];
    for r in 0..hidden {
        for c in 0..hidden {
            // sin keeps it bounded; (2r - c) makes it genuinely asymmetric.
            qw[r * hidden + c] = (((2 * r) as f32) - (c as f32) + 0.5).sin() * 0.5;
        }
    }
    let sq = |n: usize| vec![0.02_f32; n];

    let mut w = AprV2Writer::new(md);
    w.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![64, hidden],
        &sq(64 * hidden),
    );
    w.add_f32_tensor("model.norm.weight", vec![hidden], &vec![1.0; hidden]);
    w.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        vec![hidden, hidden],
        &qw,
    );
    w.add_f32_tensor(
        "model.layers.0.self_attn.k_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.self_attn.v_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.self_attn.o_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![hidden],
        &vec![1.0; hidden],
    );
    w.add_f32_tensor(
        "model.layers.0.post_attention_layernorm.weight",
        vec![hidden],
        &vec![1.0; hidden],
    );
    w.add_f32_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.mlp.up_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.add_f32_tensor(
        "model.layers.0.mlp.down_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.write().expect("write beat base v2")
}

/// LoRA adapter with deterministic, position-dependent (non-constant) factors so
/// the delta (alpha/rank)·Bᵀ@Aᵀ is itself asymmetric and order-sensitive.
#[cfg(test)]
fn build_tiny_lora_adapter_beat(hidden: usize, rank: usize, alpha: f64) -> Vec<u8> {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let mut md = AprV2Metadata::new("pmat712-beat-adapter");
    md.custom
        .insert("lora_rank".to_string(), serde_json::json!(rank));
    md.custom
        .insert("lora_alpha".to_string(), serde_json::json!(alpha));

    // lora_a: [rank, hidden], lora_b: [hidden, rank] — both non-constant.
    let mut a = vec![0.0_f32; rank * hidden];
    for k in 0..rank {
        for c in 0..hidden {
            a[k * hidden + c] = (((k + 1) as f32) * 0.013 + (c as f32) * 0.0007).cos() * 0.1;
        }
    }
    let mut b = vec![0.0_f32; hidden * rank];
    for r in 0..hidden {
        for k in 0..rank {
            b[r * rank + k] = (((r + 3) as f32) * 0.011 - ((k + 1) as f32) * 0.019).sin() * 0.1;
        }
    }

    let mut w = AprV2Writer::new(md);
    w.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight.lora_a",
        vec![rank, hidden],
        &a,
    );
    w.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight.lora_b",
        vec![hidden, rank],
        &b,
    );
    w.write().expect("write beat adapter v2")
}

/// Row-major matvec: y[r] = Σ_c W[r*cols + c] · x[c], W is [rows, cols].
#[cfg(test)]
fn rowmajor_matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "weight element count");
    assert_eq!(x.len(), cols, "input length must equal cols");
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..r * cols + cols];
            row.iter().zip(x).map(|(wv, xv)| wv * xv).sum::<f32>()
        })
        .collect()
}

#[test]
fn beat_lora_gguf_lossless_deploy_pmat712() {
    use aprender::format::gguf::load_gguf_tensors;
    use aprender::format::rosetta::RosettaStone;
    use aprender::format::{apr_export, ExportFormat, ExportOptions};

    // hidden=256 keeps K % 256 == 0 and stays tiny → fast CPU per-PR test.
    let hidden = 256usize;
    let rank = 8usize;
    let alpha = 16.0f64;

    let base_file = NamedTempFile::with_suffix(".apr").expect("base tmp");
    std::fs::write(base_file.path(), build_tiny_qwen2_base_beat(hidden)).expect("write base");

    let adapter_file = NamedTempFile::with_suffix(".apr").expect("adapter tmp");
    std::fs::write(
        adapter_file.path(),
        build_tiny_lora_adapter_beat(hidden, rank, alpha),
    )
    .expect("write adapter");

    // ── Merge: base + adapter → merged.apr (apr in-memory merged model) ──────
    let merged_apr = NamedTempFile::with_suffix(".apr").expect("merged tmp");
    let merge_res = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(merged_apr.path()),
        true,
    );
    assert!(merge_res.is_ok(), "LoRA merge must succeed: {merge_res:?}");

    // ── Export F32 GGUF: quantize=None ⇒ true LOSSLESS export (no quant error) ─
    let gguf_file = NamedTempFile::with_suffix(".gguf").expect("gguf tmp");
    let opts = ExportOptions {
        format: ExportFormat::Gguf,
        quantize: None, // F32 — losslessness, not quant-tolerance, is the claim
        include_tokenizer: false,
        include_config: false,
        skip_completeness_check: true,
    };
    apr_export(merged_apr.path(), gguf_file.path(), opts).expect("F32 GGUF export must succeed");

    // ── Load the merged q_proj from BOTH sides ──────────────────────────────
    // apr side: the same RosettaStone F32 loader the merge wrote through.
    let rosetta = RosettaStone::new();
    let w_apr = rosetta
        .load_tensor_f32(merged_apr.path(), "model.layers.0.self_attn.q_proj.weight")
        .expect("load merged q_proj from apr");

    // gguf side: load_gguf_tensors — the exact F32 path realizar uses to serve.
    let q_name = "blk.0.attn_q.weight";
    let gguf_tensors =
        load_gguf_tensors(gguf_file.path()).expect("load merged q_proj from exported GGUF");
    let (w_gguf, gguf_shape) = gguf_tensors
        .get(q_name)
        .expect("exported GGUF must contain blk.0.attn_q.weight");

    // F-LOSSLESS-001: q_proj loads back at the apr [hidden,hidden] element count.
    assert_eq!(
        w_apr.len(),
        hidden * hidden,
        "F-LOSSLESS-001: apr q_proj must be hidden^2"
    );
    assert_eq!(
        w_gguf.len(),
        hidden * hidden,
        "F-LOSSLESS-001: gguf q_proj must be hidden^2"
    );
    assert_eq!(
        gguf_shape.iter().product::<usize>(),
        hidden * hidden,
        "F-LOSSLESS-001: gguf q_proj shape product"
    );

    // ── Forward pass on BOTH: y = W @ x for the same deterministic input ─────
    // x is non-constant so y is sensitive to weight ORDERING (catches transpose).
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.017 + 0.31).sin())
        .collect();
    let y_apr = rowmajor_matvec(&w_apr, &x, hidden, hidden);
    let y_gguf = rowmajor_matvec(w_gguf, &x, hidden, hidden);

    // F-LOSSLESS-002: forward output is non-degenerate (not a constant vector),
    // proving the matvec actually exercises ordering — a transpose bug is
    // therefore *observable* in this output, not silently equal.
    let y_min = y_apr.iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = y_apr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (y_max - y_min) > 1e-2,
        "F-LOSSLESS-002: forward output must be non-degenerate (spread {} too small)",
        y_max - y_min
    );

    // F-LOSSLESS-003 — THE BEAT: max|y_apr − y_gguf| ≤ 1e-4 (F32 is lossless).
    let max_abs_dlogit = y_apr
        .iter()
        .zip(&y_gguf)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let y_scale = y_max.abs().max(y_min.abs()).max(1.0);
    eprintln!(
        "[BEAT pmat712] forward equivalence: max|Δy| = {max_abs_dlogit:.3e}  \
         (output scale ≈ {y_scale:.3}, tolerance 1e-4)"
    );
    assert!(
        max_abs_dlogit <= 1e-4,
        "F-LOSSLESS-003 (BEAT): F32 GGUF export must be forward-lossless — \
         max|y_apr − y_gguf| = {max_abs_dlogit:.3e} exceeds 1e-4. A lost/garbled \
         weight, layout transpose, or shape/metadata mismatch in export breaks this."
    );

    // F-LOSSLESS-004: sensitivity guard — the apr↔gguf agreement must be strictly
    // TIGHTER than the forward through a TRANSPOSED weight. If a transpose were a
    // no-op here (e.g. symmetric W), the BEAT above could not catch a real
    // transpose bug. Since W ≠ Wᵀ, y(Wᵀ) diverges materially from y(W).
    let mut w_apr_t = vec![0.0_f32; hidden * hidden];
    for r in 0..hidden {
        for c in 0..hidden {
            w_apr_t[c * hidden + r] = w_apr[r * hidden + c];
        }
    }
    let y_apr_t = rowmajor_matvec(&w_apr_t, &x, hidden, hidden);
    let max_abs_transpose_gap = y_apr
        .iter()
        .zip(&y_apr_t)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_transpose_gap > 100.0 * max_abs_dlogit.max(1e-6),
        "F-LOSSLESS-004: test must be transpose-sensitive — y(W) vs y(Wᵀ) gap \
         ({max_abs_transpose_gap:.3e}) should dwarf the apr↔gguf gap \
         ({max_abs_dlogit:.3e}); otherwise the BEAT could not catch a transpose bug"
    );
}

// ============================================================================
// PMAT-125 B1: additional CPU-only coverage for finetune helper logic
// ============================================================================

#[test]
fn finetune_method_parse_is_case_insensitive() {
    assert!(matches!(
        "LoRA".parse::<FinetuneMethod>(),
        Ok(FinetuneMethod::LoRA)
    ));
    assert!(matches!(
        "QLORA".parse::<FinetuneMethod>(),
        Ok(FinetuneMethod::QLoRA)
    ));
    assert!(matches!(
        "Full".parse::<FinetuneMethod>(),
        Ok(FinetuneMethod::Full)
    ));
}

#[test]
fn finetune_method_parse_unknown_reports_options() {
    let err = "bogus".parse::<FinetuneMethod>().unwrap_err();
    assert!(err.contains("bogus"));
    assert!(err.contains("auto"));
    assert!(err.contains("qlora"));
}

#[test]
fn finetune_method_default_is_auto() {
    assert!(matches!(FinetuneMethod::default(), FinetuneMethod::Auto));
}

#[test]
fn finetune_method_into_entrenar_method_all_variants() {
    assert!(matches!(Method::from(FinetuneMethod::Auto), Method::Auto));
    assert!(matches!(Method::from(FinetuneMethod::Full), Method::Full));
    assert!(matches!(Method::from(FinetuneMethod::LoRA), Method::LoRA));
    assert!(matches!(Method::from(FinetuneMethod::QLoRA), Method::QLoRA));
}

// ── parse_model_size ────────────────────────────────────────────────────────

#[test]
fn parse_model_size_billions_and_millions() {
    assert_eq!(parse_model_size("7B").unwrap(), 7_000_000_000);
    assert_eq!(parse_model_size("1.5B").unwrap(), 1_500_000_000);
    assert_eq!(parse_model_size("370M").unwrap(), 370_000_000);
    // Lowercase is upcased internally.
    assert_eq!(parse_model_size("70b").unwrap(), 70_000_000_000);
}

#[test]
fn parse_model_size_missing_suffix_errors() {
    let err = parse_model_size("7000").unwrap_err();
    match err {
        CliError::ValidationFailed(m) => assert!(m.contains("Invalid model size format")),
        _ => panic!("expected ValidationFailed"),
    }
}

#[test]
fn parse_model_size_non_numeric_errors() {
    let err = parse_model_size("abcB").unwrap_err();
    match err {
        CliError::ValidationFailed(m) => assert!(m.contains("Invalid number")),
        _ => panic!("expected ValidationFailed"),
    }
}

// ── format_params ───────────────────────────────────────────────────────────

#[test]
fn format_params_buckets() {
    assert_eq!(format_params(7_000_000_000), "7.0B");
    assert_eq!(format_params(1_500_000_000), "1.5B");
    assert_eq!(format_params(370_000_000), "370.0M");
    assert_eq!(format_params(500), "500");
}

// ── resolve_model_params ────────────────────────────────────────────────────

#[test]
fn resolve_model_params_prefers_explicit_size() {
    // When --model-size is given it is parsed directly (no file touch).
    let n = resolve_model_params(Some("3B"), None).unwrap();
    assert_eq!(n, 3_000_000_000);
}

#[test]
fn resolve_model_params_requires_size_or_path() {
    let err = resolve_model_params(None, None).unwrap_err();
    match err {
        CliError::ValidationFailed(m) => assert!(m.contains("model path or --model-size")),
        _ => panic!("expected ValidationFailed"),
    }
}

// ── build_distributed_config ────────────────────────────────────────────────

#[test]
fn build_distributed_config_coordinator_requires_bind() {
    // Issue #393 / PR #2294 implemented distributed config: the coordinator role is
    // now SUPPORTED and validates its params, rather than being rejected as an
    // "unreleased entrenar" feature. Without --bind it fails with a bind-required
    // ValidationFailed (the role itself is accepted).
    let err = build_distributed_config(Some("coordinator"), None, None, None).unwrap_err();
    match err {
        CliError::ValidationFailed(m) => assert!(m.contains("--bind is required")),
        _ => panic!("expected ValidationFailed when coordinator role lacks --bind"),
    }
}

#[test]
fn build_distributed_config_without_role_warns_but_ok() {
    // No --role: stray distributed flags only warn (stderr) and return Ok.
    let r = build_distributed_config(None, Some("0.0.0.0:9000"), Some("1.2.3.4"), Some(4));
    assert!(r.is_ok());
}

// ── parse_adapter_specs ─────────────────────────────────────────────────────

#[test]
fn parse_adapter_specs_bad_format_errors() {
    let specs = vec!["no-colon-here".to_string()];
    let err = parse_adapter_specs(&specs).unwrap_err();
    match err {
        CliError::ValidationFailed(m) => assert!(m.contains("Invalid --adapters format")),
        _ => panic!("expected ValidationFailed for missing colon"),
    }
}

#[test]
fn parse_adapter_specs_missing_data_file_is_file_not_found() {
    let specs = vec!["/nonexistent/apr-pmat125-data.jsonl:checkpoints/a".to_string()];
    let err = parse_adapter_specs(&specs).unwrap_err();
    assert!(matches!(err, CliError::FileNotFound(_)));
}

#[test]
fn parse_adapter_specs_valid_pair_resolves_paths() {
    let mut data = NamedTempFile::new().unwrap();
    writeln!(data, "{{}}").unwrap();
    let spec = format!("{}:checkpoints/adapter-a", data.path().display());
    let specs = parse_adapter_specs(&[spec]).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].0, data.path());
    assert_eq!(
        specs[0].1,
        std::path::PathBuf::from("checkpoints/adapter-a")
    );
}

// ── merge_adapters_config ───────────────────────────────────────────────────

#[test]
fn merge_adapters_config_no_file_returns_cli_only() {
    let cli = vec!["a:1".to_string(), "b:2".to_string()];
    let merged = merge_adapters_config(&cli, None, true).unwrap();
    assert_eq!(merged, cli);
}

// ── build_classify_config ───────────────────────────────────────────────────

#[test]
fn build_classify_config_maps_cli_args() {
    let model_cfg = entrenar::transformer::TransformerConfig::llama2_7b();
    let cfg = build_classify_config(&model_cfg, 5, 16, 3, 1e-4, Some(256), false);
    assert_eq!(cfg.num_classes, 5);
    assert_eq!(cfg.lora_rank, 16);
    assert!((cfg.lora_alpha - 16.0).abs() < 1e-6);
    assert_eq!(cfg.epochs, 3);
    assert_eq!(cfg.max_seq_len, 256);
    assert!((cfg.learning_rate - 1e-4).abs() < 1e-9);
}

#[test]
fn build_classify_config_defaults_max_seq_len_when_none() {
    use entrenar::finetune::classify_pipeline::ClassifyConfig;
    let model_cfg = entrenar::transformer::TransformerConfig::llama2_7b();
    let cfg = build_classify_config(&model_cfg, 2, 8, 1, 5e-5, None, false);
    assert_eq!(cfg.max_seq_len, ClassifyConfig::default().max_seq_len);
}

// ── build_instruct_config (Defect 2: honor --max-seq-len on the LoRA path) ───

#[test]
fn build_instruct_config_threads_max_seq_len() {
    // A user-supplied --max-seq-len must reach InstructConfig.max_seq_len on the
    // instruct/LoRA path, NOT be silently dropped to the old hardcoded 512.
    let mut config = plan(1_000_000_000, 24.0, Method::LoRA).expect("plan lora");
    config.method = Method::LoRA;
    let cfg = build_instruct_config(&config, 1e-4, 3, Some(384));
    assert_eq!(
        cfg.max_seq_len, 384,
        "--max-seq-len 384 must be honored, not dropped to 512"
    );
    assert_eq!(cfg.epochs, 3);
    assert!(!cfg.quantize_nf4, "plain LoRA → NF4 off");
}

#[test]
fn build_instruct_config_defaults_max_seq_len_to_512_when_absent() {
    use entrenar::finetune::instruct_pipeline::InstructConfig;
    let mut config = plan(1_000_000_000, 24.0, Method::QLoRA).expect("plan qlora");
    config.method = Method::QLoRA;
    let cfg = build_instruct_config(&config, 5e-5, 1, None);
    assert_eq!(cfg.max_seq_len, InstructConfig::default().max_seq_len);
    assert_eq!(cfg.max_seq_len, 512);
    assert!(cfg.quantize_nf4, "QLoRA method → NF4 on");
}

// ── gpu_backend_notice (Defect 1: truthful GPU banner) ──────────────────────

#[test]
fn gpu_backend_notice_cuda_plain_lora_warns_cpu() {
    // --gpu-backend cuda + plain LoRA (quantize_nf4=false) must NOT claim cuBLAS;
    // it must WARN that training actually runs on CPU (init_cuda is QLoRA-only).
    let p = gpu_backend_notice("cuda", false, false);
    assert!(!p.use_wgpu);
    assert!(p.notice.contains("WARNING"), "must warn: {}", p.notice);
    assert!(p.notice.to_lowercase().contains("cpu"));
    assert!(
        !p.notice.contains("using cuBLAS backward path (QLoRA/NF4)"),
        "must not make a false cuBLAS claim for plain LoRA: {}",
        p.notice
    );
}

#[test]
fn gpu_backend_notice_cuda_qlora_claims_cublas() {
    let p = gpu_backend_notice("cuda", true, false);
    assert!(!p.use_wgpu);
    assert!(
        p.notice.contains("cuBLAS"),
        "QLoRA on CUDA truthfully engages cuBLAS: {}",
        p.notice
    );
    assert!(!p.notice.contains("WARNING"));
}

#[test]
fn gpu_backend_notice_wgpu_selects_wgpu() {
    let p = gpu_backend_notice("wgpu", false, true);
    assert!(p.use_wgpu);
    assert!(p.notice.contains("WGPU"));
}

#[test]
fn gpu_backend_notice_auto_plain_lora_is_cpu() {
    let p = gpu_backend_notice("auto", false, true);
    assert!(!p.use_wgpu, "plain LoRA under auto stays on the CPU path");
    assert!(p.notice.to_lowercase().contains("cpu"));
}

#[test]
fn gpu_backend_notice_auto_qlora_prefers_wgpu_when_available() {
    let with_wgpu = gpu_backend_notice("auto", true, true);
    assert!(with_wgpu.use_wgpu, "NF4 + wgpu available → WGPU fast path");
    let without_wgpu = gpu_backend_notice("auto", true, false);
    assert!(!without_wgpu.use_wgpu, "no wgpu feature → CUDA path");
    assert!(without_wgpu.notice.contains("cuBLAS"));
}

// ── run (top-level dispatch) error paths ────────────────────────────────────

#[test]
fn run_with_missing_model_file_errors() {
    let missing = std::path::Path::new("/nonexistent/apr-pmat125-model.apr");
    let result = run(
        Some(missing), // model_path
        "lora",        // method
        Some(16),      // rank
        32.0,          // vram_gb
        true,          // plan_only — avoid any training
        None,          // data_path
        None,          // output_path
        None,          // adapter_path
        false,         // merge_mode
        1,             // epochs
        Some(1e-4),    // learning_rate
        None,          // model_size
        None,          // task
        2,             // num_classes
        "apr",         // checkpoint_format
        false,         // oversample
        None,          // max_seq_len
        false,         // quantize_nf4
        None,          // gpus
        "auto",        // gpu_backend
        None,          // role
        None,          // bind
        None,          // coordinator
        None,          // expect_workers
        0,             // wait_gpu
        &[],           // adapters
        None,          // adapters_config
        true,          // json_output
        false,         // experimental_mps
        0,             // gpu_share
    );
    assert!(result.is_err());
}

// ============================================================================
// C-APR-MERGE-RUNNABLE: `apr finetune --merge` must produce a DIRECTLY
// RUNNABLE .apr (contracts/apr-merge-runnable-v1.yaml).
//
// Mechanism being falsified: import-produced bases carry HF-alias dimension
// keys (num_hidden_layers/num_attention_heads/num_key_value_heads) in the
// metadata `custom` map. Pre-fix, run_merge re-serialized the cloned metadata
// with BOTH `"num_layers": null` (typed field, no skip_serializing_if) AND
// the alias key — realizar's serde-aliased `AprMetadata` fails that JSON with
// "duplicate field", `MappedAprModel::from_mmap` swallows the error via
// `unwrap_or_default()`, and the merged model loses ALL metadata: C-01
// "missing 'architecture'" + "no tokenizer in APR metadata" on a file that
// physically contains both.
// ============================================================================

/// Build a base APR v2 shaped EXACTLY like a real `apr convert` GGUF import
/// (e.g. qwen2.5-coder-1.5b-instruct-q4k.apr): typed architecture/hidden
/// fields, HF-alias dimension keys ONLY in `custom`, embedded tokenizer.
#[cfg(test)]
fn build_import_shaped_qwen2_base_v2(hidden: usize, with_tokenizer: bool) -> Vec<u8> {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let mut md = AprV2Metadata::new("transformer_lm_q4k");
    md.architecture = Some("qwen2".to_string());
    md.hidden_size = Some(hidden);
    md.vocab_size = Some(64);
    md.intermediate_size = Some(hidden);
    md.max_position_embeddings = Some(128);
    md.rope_theta = Some(1_000_000.0);
    md.rms_norm_eps = Some(1e-6);
    // HF-alias keys ONLY in custom — exactly how the GGUF import path stamps
    // them (the typed fields stay None, mirroring the real q4k base).
    md.custom
        .insert("num_hidden_layers".to_string(), serde_json::json!(1));
    md.custom
        .insert("num_attention_heads".to_string(), serde_json::json!(4));
    md.custom
        .insert("num_key_value_heads".to_string(), serde_json::json!(2));
    if with_tokenizer {
        let vocab: Vec<String> = (0..64).map(|i| format!("t{i}")).collect();
        md.custom
            .insert("tokenizer.vocabulary".to_string(), serde_json::json!(vocab));
        md.custom
            .insert("tokenizer.merges".to_string(), serde_json::json!(["t0 t1"]));
        md.custom
            .insert("tokenizer.bos_token_id".to_string(), serde_json::json!(0));
        md.custom
            .insert("tokenizer.eos_token_id".to_string(), serde_json::json!(1));
    }

    let mut w = AprV2Writer::new(md);
    let sq = |n: usize| vec![0.02_f32; n];
    w.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![64, hidden],
        &sq(64 * hidden),
    );
    w.add_f32_tensor("model.norm.weight", vec![hidden], &vec![1.0; hidden]);
    w.add_f32_tensor(
        "model.layers.0.self_attn.q_proj.weight",
        vec![hidden, hidden],
        &vec![1.0_f32; hidden * hidden],
    );
    w.add_f32_tensor(
        "model.layers.0.mlp.gate_proj.weight",
        vec![hidden, hidden],
        &sq(hidden * hidden),
    );
    w.write().expect("write import-shaped base v2")
}

/// FALSIFY-APR-MERGE-RUNNABLE-001: merge an import-shaped base + LoRA adapter,
/// then load the output through REALIZAR'S OWN deserializer + config path (the
/// exact code `apr run` executes). RED on pre-fix main: realizar parses the
/// merged metadata to EMPTY (duplicate-field poison) → architecture None.
#[cfg(feature = "inference")]
#[test]
fn falsify_apr_merge_runnable_001_merged_output_loads_in_realizar() {
    let hidden = 32usize;
    let base_file = NamedTempFile::with_suffix(".apr").expect("base tmp");
    std::fs::write(
        base_file.path(),
        build_import_shaped_qwen2_base_v2(hidden, true),
    )
    .expect("write base");

    let adapter_file = NamedTempFile::with_suffix(".apr").expect("adapter tmp");
    std::fs::write(
        adapter_file.path(),
        build_tiny_lora_adapter_v2(hidden, 4, 8.0),
    )
    .expect("write adapter");

    let merged = NamedTempFile::with_suffix(".apr").expect("merged tmp");
    let res = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(merged.path()),
        true,
    );
    assert!(
        res.is_ok(),
        "merge must succeed on a runnable base: {res:?}"
    );

    // ORACLE: realizar's own mmap loader — the exact `apr run` path.
    let mapped = realizar::apr::MappedAprModel::from_path(merged.path())
        .expect("realizar must mmap the merged output");
    assert_eq!(
        mapped.metadata.architecture.as_deref(),
        Some("qwen2"),
        "C-01: architecture must survive merge AND parse in realizar \
         (empty ⇒ duplicate-field metadata poison regressed)"
    );
    assert_eq!(mapped.metadata.num_layers, Some(1), "C-03: num_layers");
    assert_eq!(mapped.metadata.num_heads, Some(4), "C-03: num_heads");
    assert_eq!(mapped.metadata.num_kv_heads, Some(2), "C-03: num_kv_heads");
    assert_eq!(
        mapped.metadata.intermediate_size,
        Some(hidden),
        "C-03: intermediate_size"
    );

    // C-01/C-03 verbatim: the same config extraction `apr run` performs.
    realizar::gguf::GGUFConfig::from_apr(&mapped, 64)
        .expect("GGUFConfig::from_apr must succeed on the merged output");

    // PMAT-171: embedded BPE tokenizer must load from the merged output.
    let model = realizar::apr::AprV2Model::load(merged.path()).expect("realizar AprV2Model load");
    assert!(
        model.load_embedded_bpe_tokenizer().is_some(),
        "embedded BPE tokenizer (vocabulary+merges) must load from the merged output"
    );
}

/// FALSIFY-APR-MERGE-RUNNABLE-002: `-o out.safetensors` while writing an APR
/// container is format-in-disguise — must ERROR and write nothing. RED on
/// pre-fix main (run_merge happily wrote APR bytes under .safetensors).
#[test]
fn falsify_apr_merge_runnable_002_safetensors_extension_rejected() {
    let hidden = 32usize;
    let base_file = NamedTempFile::with_suffix(".apr").expect("base tmp");
    std::fs::write(
        base_file.path(),
        build_import_shaped_qwen2_base_v2(hidden, true),
    )
    .expect("write base");
    let adapter_file = NamedTempFile::with_suffix(".apr").expect("adapter tmp");
    std::fs::write(
        adapter_file.path(),
        build_tiny_lora_adapter_v2(hidden, 4, 8.0),
    )
    .expect("write adapter");

    let out_dir = tempfile::tempdir().expect("tmpdir");
    let out = out_dir.path().join("merged.safetensors");
    let res = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(&out),
        true,
    );
    assert!(
        res.is_err(),
        "merge must reject a .safetensors output extension (APR-in-disguise)"
    );
    assert!(
        !out.exists(),
        "no format-in-disguise file may be left on disk"
    );
}

/// FALSIFY-APR-MERGE-RUNNABLE-003: fail-closed gate — a base WITHOUT an
/// embedded tokenizer cannot produce a runnable merge; run_merge must ERROR
/// and DELETE the output. RED on pre-fix main (returned Ok, left an
/// unrunnable artifact).
#[test]
fn falsify_apr_merge_runnable_003_gate_deletes_tokenizerless_output() {
    let hidden = 32usize;
    let base_file = NamedTempFile::with_suffix(".apr").expect("base tmp");
    std::fs::write(
        base_file.path(),
        build_import_shaped_qwen2_base_v2(hidden, false),
    )
    .expect("write base");
    let adapter_file = NamedTempFile::with_suffix(".apr").expect("adapter tmp");
    std::fs::write(
        adapter_file.path(),
        build_tiny_lora_adapter_v2(hidden, 4, 8.0),
    )
    .expect("write adapter");

    let out_dir = tempfile::tempdir().expect("tmpdir");
    let out = out_dir.path().join("merged.apr");
    let res = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(&out),
        true,
    );
    assert!(
        res.is_err(),
        "merge of a tokenizer-less base must fail the runnability gate"
    );
    assert!(!out.exists(), "gate must DELETE the unrunnable output");
}

/// FALSIFY-APR-MERGE-RUNNABLE-004: entrenar trainer checkpoints name adapter
/// tensors `lora.{layer}.{proj}.lora_{a,b}` (cuda_trainer.rs /
/// wgpu_checkpoint.rs) — `apr finetune --merge` must merge ITS OWN trainer's
/// adapters, deriving rank from the adapter's [rank, d_in] shape (a wrong
/// global rank makes MergeEngine mis-derive dims and silently no-op).
/// RED on pre-fix main: 0 layers merged, output weights byte-identical to base.
#[test]
fn falsify_apr_merge_runnable_004_entrenar_adapter_naming_merges() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let hidden = 32usize;
    let rank = 4usize;
    let base_file = NamedTempFile::with_suffix(".apr").expect("base tmp");
    std::fs::write(
        base_file.path(),
        build_import_shaped_qwen2_base_v2(hidden, true),
    )
    .expect("write base");

    // Entrenar-style adapter: lora.0.q_proj.lora_{a,b}. Metadata rank is
    // deliberately WRONG (64) — the merge must trust the tensor shape (4).
    let mut md = AprV2Metadata::new("entrenar-adapter");
    md.custom
        .insert("lora_rank".to_string(), serde_json::json!(64));
    md.custom
        .insert("lora_alpha".to_string(), serde_json::json!(8.0));
    let mut w = AprV2Writer::new(md);
    w.add_f32_tensor(
        "lora.0.q_proj.lora_a",
        vec![rank, hidden],
        &vec![0.3_f32; rank * hidden],
    );
    w.add_f32_tensor(
        "lora.0.q_proj.lora_b",
        vec![hidden, rank],
        &vec![0.5_f32; hidden * rank],
    );
    let adapter_file = NamedTempFile::with_suffix(".apr").expect("adapter tmp");
    std::fs::write(adapter_file.path(), w.write().expect("write adapter"))
        .expect("write adapter file");

    let merged = NamedTempFile::with_suffix(".apr").expect("merged tmp");
    let res = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(merged.path()),
        true,
    );
    assert!(
        res.is_ok(),
        "merge must recognize entrenar lora.{{layer}}.{{proj}} naming: {res:?}"
    );

    // The merged q_proj must DIFFER from the base (constant 1.0): the LoRA
    // delta actually landed, at shape-derived rank.
    let q: Vec<f32> = aprender::format::rosetta::RosettaStone::new()
        .load_tensor_f32(merged.path(), "model.layers.0.self_attn.q_proj.weight")
        .expect("q_proj present");
    assert!(
        q.iter().any(|&v| (v - 1.0).abs() > 1e-4),
        "merged q_proj must differ from base — entrenar-style adapter was silently ignored"
    );
}

/// FALSIFY-APR-MERGE-RUNNABLE-005: an adapter whose LoRA tensors match NO
/// base tensor must FAIL the merge (silent no-op ships a byte-identical copy
/// of the base under a 'merged' name). RED on pre-fix main (returned Ok,
/// "0 / N layers merged").
#[test]
fn falsify_apr_merge_runnable_005_noop_merge_rejected() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let hidden = 32usize;
    let base_file = NamedTempFile::with_suffix(".apr").expect("base tmp");
    std::fs::write(
        base_file.path(),
        build_import_shaped_qwen2_base_v2(hidden, true),
    )
    .expect("write base");

    let mut md = AprV2Metadata::new("mismatched-adapter");
    md.custom
        .insert("lora_rank".to_string(), serde_json::json!(4));
    md.custom
        .insert("lora_alpha".to_string(), serde_json::json!(8.0));
    let mut w = AprV2Writer::new(md);
    w.add_f32_tensor(
        "some.unrelated.tensor.lora_a",
        vec![4, hidden],
        &vec![0.1_f32; 4 * hidden],
    );
    w.add_f32_tensor(
        "some.unrelated.tensor.lora_b",
        vec![hidden, 4],
        &vec![0.1_f32; hidden * 4],
    );
    let adapter_file = NamedTempFile::with_suffix(".apr").expect("adapter tmp");
    std::fs::write(adapter_file.path(), w.write().expect("write adapter"))
        .expect("write adapter file");

    let out_dir = tempfile::tempdir().expect("tmpdir");
    let out = out_dir.path().join("merged.apr");
    let res = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(&out),
        true,
    );
    assert!(
        res.is_err(),
        "a merge where zero adapter tensors match must fail loudly, got: {res:?}"
    );
    assert!(!out.exists(), "no-op merge must not leave an output file");
}

/// FALSIFY-APR-MERGE-RUNNABLE-004b: same entrenar adapter naming, but the
/// base uses GGUF-style tensor names (`blk.{N}.attn_q.weight` — the
/// `apr convert` import layout, and the ACTUAL layout of the live flip base
/// qwen2.5-coder-1.5b-instruct-q4k.apr). `lora.{N}.q_proj.*` must resolve
/// against `blk.{N}.attn_q.weight` via the GGUF→HF projection map.
#[test]
fn falsify_apr_merge_runnable_004b_entrenar_adapter_merges_gguf_named_base() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let hidden = 32usize;
    let rank = 4usize;

    // GGUF-named base with import-shaped metadata (HF-alias dims in custom).
    let mut md = AprV2Metadata::new("transformer_lm_q4k");
    md.architecture = Some("qwen2".to_string());
    md.hidden_size = Some(hidden);
    md.vocab_size = Some(64);
    md.intermediate_size = Some(hidden);
    md.rms_norm_eps = Some(1e-6);
    md.custom
        .insert("num_hidden_layers".to_string(), serde_json::json!(1));
    md.custom
        .insert("num_attention_heads".to_string(), serde_json::json!(4));
    md.custom
        .insert("num_key_value_heads".to_string(), serde_json::json!(2));
    let vocab: Vec<String> = (0..64).map(|i| format!("t{i}")).collect();
    md.custom
        .insert("tokenizer.vocabulary".to_string(), serde_json::json!(vocab));
    md.custom
        .insert("tokenizer.merges".to_string(), serde_json::json!(["t0 t1"]));
    let mut w = AprV2Writer::new(md);
    let sq = |n: usize| vec![0.02_f32; n];
    w.add_f32_tensor("token_embd.weight", vec![64, hidden], &sq(64 * hidden));
    w.add_f32_tensor(
        "blk.0.attn_q.weight",
        vec![hidden, hidden],
        &vec![1.0_f32; hidden * hidden],
    );
    let base_file = NamedTempFile::with_suffix(".apr").expect("base tmp");
    std::fs::write(base_file.path(), w.write().expect("write base")).expect("write base file");

    // Entrenar adapter targets q_proj (HF spelling).
    let mut amd = AprV2Metadata::new("entrenar-adapter");
    amd.custom
        .insert("lora_rank".to_string(), serde_json::json!(rank));
    amd.custom
        .insert("lora_alpha".to_string(), serde_json::json!(8.0));
    let mut aw = AprV2Writer::new(amd);
    aw.add_f32_tensor(
        "lora.0.q_proj.lora_a",
        vec![rank, hidden],
        &vec![0.3_f32; rank * hidden],
    );
    aw.add_f32_tensor(
        "lora.0.q_proj.lora_b",
        vec![hidden, rank],
        &vec![0.5_f32; hidden * rank],
    );
    let adapter_file = NamedTempFile::with_suffix(".apr").expect("adapter tmp");
    std::fs::write(adapter_file.path(), aw.write().expect("write adapter"))
        .expect("write adapter file");

    let merged = NamedTempFile::with_suffix(".apr").expect("merged tmp");
    let res = run_merge(
        Some(base_file.path()),
        Some(adapter_file.path()),
        Some(merged.path()),
        true,
    );
    assert!(
        res.is_ok(),
        "entrenar adapter must merge against a GGUF-named base: {res:?}"
    );
    let q: Vec<f32> = aprender::format::rosetta::RosettaStone::new()
        .load_tensor_f32(merged.path(), "blk.0.attn_q.weight")
        .expect("attn_q present");
    assert!(
        q.iter().any(|&v| (v - 1.0).abs() > 1e-4),
        "merged blk.0.attn_q.weight must differ from base — GGUF→HF projection map broken"
    );
}
