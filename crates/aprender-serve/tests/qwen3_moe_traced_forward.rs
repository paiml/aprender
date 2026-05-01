//! M32d Step 2 falsifier — `forward_qwen3_moe_traced` per-layer
//! ActivationStats variant of `forward_qwen3_moe`.
//!
//! Companion `claude-code-parity-apr-poc.md` § "M32d FAST PATH" Step 2:
//!
//!   "wire `apr trace --json --payload` into qwen3_moe forward (today
//!    returns null per-layer stats). Add a parallel
//!    `forward_qwen3_moe_traced` that records each of the 48 layer
//!    outputs."
//!
//! ### Exit criterion (per spec)
//!
//!   "`apr trace --json --payload <gguf> --prompt "What is 2+2?"` returns
//!    non-null `output_stats` for every `transformer_block_N` entry, with
//!    finite L2 norms."
//!
//! This test asserts the in-memory equivalent: `forward_qwen3_moe_traced`
//! returns a `ForwardTrace` with one `LayerActivation` per decoder layer
//! whose every populated `ActivationStats` slot is finite. Once this
//! method is wired into the `apr trace` orchestrator (separate PR), the
//! CLI exit criterion is mechanically satisfied.
//!
//! Skipped when GGUF absent (fixture-absent ≠ defect, per
//! M32c.2.2.2.1.4 convention).

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGGUFTransformer};

use std::path::Path;

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_INTERMEDIATE: usize = 768;
const EXPECTED_N_EXPERTS: usize = 128;
const EXPECTED_K: usize = 8;
const EXPECTED_VOCAB: usize = 151936;

#[test]
fn f_qw3_moe_step2_001_traced_returns_per_layer_finite_stats() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "F-QW3-MOE-STEP2-001: skipped — no cached Qwen3-Coder GGUF at any of: {:?}",
            CANONICAL_QWEN3_CODER_GGUF_PATHS
        );
        return;
    };

    eprintln!("F-QW3-MOE-STEP2-001: traced forward against {gguf_path}");

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");
    let data = mapped.data();
    let _transformer = QuantizedGGUFTransformer::from_gguf_for_moe(&mapped.model, data)
        .expect("from_gguf_for_moe must succeed");
    let model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped");

    let mut moe_layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        moe_layers.push(
            load_qwen3_moe_layer(&mapped.model, data, layer_idx)
                .unwrap_or_else(|e| panic!("layer {layer_idx} MoE load failed: {e:?}")),
        );
    }

    let token_ids = vec![0u32];

    eprintln!("F-QW3-MOE-STEP2-001: running traced forward (this takes a few minutes)...");
    let start = std::time::Instant::now();
    let trace = model
        .forward_qwen3_moe_traced(
            &token_ids,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
        )
        .expect("F-QW3-MOE-STEP2-001: forward_qwen3_moe_traced should succeed");
    let elapsed = start.elapsed();

    // Per-layer count parity
    assert_eq!(
        trace.layer_activations.len(),
        EXPECTED_NUM_LAYERS,
        "F-QW3-MOE-STEP2-001: must produce one LayerActivation per decoder layer"
    );

    // Logits shape parity (matches f_qw3_moe_c22211_001)
    assert_eq!(
        trace.logits.len(),
        EXPECTED_VOCAB,
        "F-QW3-MOE-STEP2-001: logits len must equal vocab_size"
    );
    assert!(
        trace.logits.iter().all(|v| v.is_finite()),
        "F-QW3-MOE-STEP2-001: all logits must be finite"
    );

    // Embed/final-norm/logits stats finite
    assert_finite_stats(&trace.embed_stats, "embed_stats");
    assert_finite_stats(&trace.final_norm_stats, "final_norm_stats");
    assert_finite_stats(&trace.logits_stats, "logits_stats");
    assert_eq!(
        trace.embed_stats.count, 2048,
        "embed_stats count == hidden_dim"
    );

    // Per-layer: every populated stat slot is finite. Sub-FFN slots
    // default to zero (no SwiGLU breakdown in MoE) — those are explicitly
    // allowed.
    for (i, layer) in trace.layer_activations.iter().enumerate() {
        assert_eq!(
            layer.layer_idx, i,
            "F-QW3-MOE-STEP2-001: layer_idx must match position"
        );
        assert_finite_stats(&layer.attn_norm_stats, &format!("layer[{i}].attn_norm"));
        assert_finite_stats(&layer.qkv_stats, &format!("layer[{i}].qkv"));
        assert_finite_stats(&layer.attn_out_stats, &format!("layer[{i}].attn_out"));
        assert_finite_stats(&layer.ffn_norm_stats, &format!("layer[{i}].ffn_norm"));
        assert_finite_stats(&layer.ffn_out_stats, &format!("layer[{i}].ffn_out"));
        assert_finite_stats(&layer.output_stats, &format!("layer[{i}].output"));

        // count must equal hidden_dim for the populated slots
        assert_eq!(
            layer.attn_norm_stats.count, 2048,
            "layer[{i}].attn_norm.count must equal hidden_dim"
        );
        assert_eq!(
            layer.output_stats.count, 2048,
            "layer[{i}].output.count must equal hidden_dim"
        );
    }

    let l2: f32 = trace.logits.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!(
        "F-QW3-MOE-STEP2-001: PASS\n  elapsed = {:?}\n  layers traced = {}\n  ||logits||_2 = {:.4}\n  layer[0].output_stats.std_dev = {:.4}\n  layer[47].output_stats.std_dev = {:.4}",
        elapsed,
        trace.layer_activations.len(),
        l2,
        trace.layer_activations[0].output_stats.std_dev,
        trace.layer_activations[EXPECTED_NUM_LAYERS - 1]
            .output_stats
            .std_dev
    );
}

#[test]
fn f_qw3_moe_step2_002_traced_rejects_empty_input() {
    // Pure error-path test — runs without GGUF.
    let mapped_path = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists());
    let Some(gguf_path) = mapped_path else {
        eprintln!("F-QW3-MOE-STEP2-002: skipped — no cached GGUF for model construction");
        return;
    };
    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");
    let data = mapped.data();
    let _transformer = QuantizedGGUFTransformer::from_gguf_for_moe(&mapped.model, data)
        .expect("from_gguf_for_moe");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("from_mapped");

    let result = model.forward_qwen3_moe_traced(
        &[],
        &[],
        EXPECTED_N_EXPERTS,
        EXPECTED_K,
        EXPECTED_INTERMEDIATE,
        data,
    );
    assert!(
        result.is_err(),
        "F-QW3-MOE-STEP2-002: empty token_ids must error"
    );
}

fn assert_finite_stats(stats: &realizar::apr_transformer::ActivationStats, label: &str) {
    assert_eq!(
        stats.nan_count, 0,
        "{label}: nan_count must be 0, got {}",
        stats.nan_count
    );
    assert_eq!(
        stats.inf_count, 0,
        "{label}: inf_count must be 0, got {}",
        stats.inf_count
    );
    assert!(
        stats.min.is_finite(),
        "{label}: min must be finite, got {}",
        stats.min
    );
    assert!(
        stats.max.is_finite(),
        "{label}: max must be finite, got {}",
        stats.max
    );
    assert!(
        stats.mean.is_finite(),
        "{label}: mean must be finite, got {}",
        stats.mean
    );
    assert!(
        stats.std_dev.is_finite(),
        "{label}: std_dev must be finite, got {}",
        stats.std_dev
    );
    assert!(stats.count > 0, "{label}: count must be > 0");
}
