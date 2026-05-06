//! M-FFN-GGUF-3 — heavy APR-vs-GGUF layer-3 ffn_swigl std comparison harness.
//!
//! Per `contracts/trace-ffn-sub-block-gguf-v1.yaml` step M-FFN-GGUF-3, this
//! harness compares APR-side and GGUF-side `ffn_swiglu_inner_stats.std_dev` at
//! layer 3 on the same canonical 7B teacher prompt to distinguish two
//! competing hypotheses for the SHIP-007 §21 layer-3 ffn_swigl anomaly:
//!
//! - **H1**: Token-position-dependent correlation — NORMAL model behavior;
//!   SHIP-007 root cause is ELSEWHERE. Predicts: GGUF layer-3 ffn_swigl std
//!   ≈ APR layer-3 ffn_swigl std (ratio within `[1/RATIO_TOL, RATIO_TOL]`).
//! - **H2**: APR-side bug — APR forward path produces different VALUES than
//!   GGUF (despite SHIP-003 PR #1059 proving weights are byte-equivalent at
//!   cos≥0.9999999). Predicts: ratio outside `[1/RATIO_TOL, RATIO_TOL]`.
//!
//! Discharges FALSIFY-FFN-GGUF-003. The H1/H2 outcome dictates the next
//! cascade step (M-FFN-GGUF-4 — SHIP-007 root-cause fix PR cites the
//! bisected hypothesis).
//!
//! ## How to run
//!
//! Skip-if-not-present pattern (mirrors M80
//! `qwen3_moe_gpu_per_stage_diff::falsify_moe_sub_002_*`):
//!
//! ```bash
//! cargo test -p aprender-serve --test ffn_gguf_apr_layer_3_swigl_diff \
//!     -- --include-ignored --nocapture
//! ```
//!
//! `#[ignore]`-gated so normal CI doesn't run the heavy load. Cleanly
//! skips with a clear message if either the canonical 7B APR teacher
//! file or the canonical 7B GGUF file is missing on the host.
//!
//! ## Required files (any one of each list)
//!
//! Canonical 7B APR teacher (paiml/qwen2.5-coder-7b-apache-q4k-v1):
//!   - `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-apache-q4k-v1.apr`
//!   - `/home/noah/.apr/models/qwen2.5-coder-7b-apache-q4k-v1.apr`
//!   - `/mnt/nvme-raid0/cache/apr-home/models/qwen2.5-coder-7b-apache-q4k-v1.apr`
//!
//! Canonical 7B GGUF (qwen2.5-coder-7b-instruct-q4k.gguf):
//!   - `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.gguf`
//!   - `/home/noah/.cache/huggingface/hub/.../qwen2.5-coder-7b-instruct-q4k.gguf`
//!
//! ## Reference
//!
//! - `contracts/trace-ffn-sub-block-gguf-v1.yaml` (this contract)
//! - SHIP-007 §21 (aprender PR #1072 squash 211edeafc) — the layer-3
//!   anomaly site narrowing
//! - `crates/aprender-serve/tests/qwen3_moe_gpu_per_stage_diff.rs`
//!   (M80 — sibling pattern for CPU-vs-GPU MoE diff harness)
//!
//! ## Ratio interpretation
//!
//! At PR-author time (2026-05-06), known APR layer-3 ffn_swigl std
//! ≈ 1.222 from the §21 evidence. The harness reports the ratio
//! `apr_std / gguf_std`. Outcome interpretation:
//!
//! - ratio in [0.5, 2.0] → **H1 confirmed** (normal model behavior);
//!   SHIP-007 root cause is ELSEWHERE (lm_head, post-FFN residual,
//!   token-position correlation, ...).
//! - ratio > 2.0 (or < 0.5) → **H2 confirmed** (APR-side bug);
//!   fix at `inference.rs:160-164` swigl elementwise multiply.

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

use std::path::Path;

const CANONICAL_QWEN25_CODER_7B_APR_PATHS: &[&str] = &[
    "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-apache-q4k-v1.apr",
    "/home/noah/.apr/models/qwen2.5-coder-7b-apache-q4k-v1.apr",
    "/mnt/nvme-raid0/cache/apr-home/models/qwen2.5-coder-7b-apache-q4k-v1.apr",
];

const CANONICAL_QWEN25_CODER_7B_GGUF_PATHS: &[&str] = &[
    "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.gguf",
];

/// Layer 3 is the §21 narrowed anomaly site (ffn_swigl std = 1.222 vs
/// layer-2 baseline 0.071 — 17.2× spike).
const ANOMALY_LAYER: usize = 3;

/// Ratio outside this band → H2 confirmed (APR-side bug). Within band → H1
/// confirmed (normal model behavior). 2.0× tolerance is generous —
/// quantization difference between APR and GGUF Q4_K is typically < 5%
/// per element-level cosine of weights (SHIP-003 PR #1059 cos≥0.9999999).
const RATIO_TOL: f32 = 2.0;

/// Canonical SHIP-007 prompt. Same as the §21 evidence file
/// `evidence/ship-007-layer-3-anomaly/sub-ffn-bisection-2026-04-26.txt`.
const CANONICAL_PROMPT: &str = "What is 2+2?";

#[test]
#[ignore]
fn falsify_ffn_gguf_003_layer_3_swigl_h1_h2_bisection() {
    let Some(apr_path) = CANONICAL_QWEN25_CODER_7B_APR_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "M-FFN-GGUF-3 layer-3 swigl diff: skipped — no canonical 7B APR teacher \
             in {CANONICAL_QWEN25_CODER_7B_APR_PATHS:?}"
        );
        return;
    };
    let Some(gguf_path) = CANONICAL_QWEN25_CODER_7B_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "M-FFN-GGUF-3 layer-3 swigl diff: skipped — no canonical 7B GGUF \
             in {CANONICAL_QWEN25_CODER_7B_GGUF_PATHS:?}"
        );
        return;
    };

    eprintln!("M-FFN-GGUF-3: APR vs GGUF layer-3 ffn_swigl std H1/H2 bisection");
    eprintln!("  apr_path:    {apr_path}");
    eprintln!("  gguf_path:   {gguf_path}");
    eprintln!("  prompt:      {CANONICAL_PROMPT:?}");
    eprintln!("  layer:       {ANOMALY_LAYER}");
    eprintln!("  ratio_tol:   {RATIO_TOL}");
    eprintln!();

    // ---- APR side ----
    eprintln!("Loading APR teacher...");
    let apr_transformer =
        AprTransformer::from_apr_file(apr_path).expect("AprTransformer::from_apr_file failed");

    // GGUF tokenizer is consistent with APR for this teacher pair —
    // re-tokenize once and use the same tokens for both forwards.
    let mapped = MappedGGUFModel::from_path(gguf_path).expect("MappedGGUFModel::from_path failed");
    let tokens = mapped
        .model
        .encode(CANONICAL_PROMPT)
        .unwrap_or_else(|| vec![1u32]);
    eprintln!("  tokens:    {tokens:?} ({} tokens)", tokens.len());
    eprintln!();

    eprintln!("APR forward_traced...");
    let apr_trace = apr_transformer
        .forward_traced(&tokens)
        .expect("APR forward_traced failed");

    // ---- GGUF side ----
    eprintln!("GGUF forward_traced...");
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped).expect("from_mapped failed");
    let gguf_trace = gguf_model
        .forward_traced(&tokens)
        .expect("GGUF forward_traced failed");

    // ---- Per-layer comparison ----
    let apr_layers = &apr_trace.layer_activations;
    let gguf_layers = &gguf_trace.layer_activations;
    let n_layers = apr_layers.len().min(gguf_layers.len());
    assert!(
        n_layers > ANOMALY_LAYER,
        "model has fewer layers than ANOMALY_LAYER ({ANOMALY_LAYER}); got {n_layers}"
    );

    eprintln!();
    eprintln!("layer | apr.ffn_swigl.std    | gguf.ffn_swigl.std   | ratio (apr/gguf)");
    eprintln!("------|----------------------|----------------------|------------------");
    let mut anomaly_ratio: Option<f32> = None;
    for layer_idx in 0..n_layers {
        let apr_std = apr_layers[layer_idx].ffn_swiglu_inner_stats.std_dev;
        let gguf_std = gguf_layers[layer_idx].ffn_swiglu_inner_stats.std_dev;
        let ratio = if gguf_std.abs() > 1e-9 {
            apr_std / gguf_std
        } else {
            f32::NAN
        };
        eprintln!(
            "L{layer_idx:02}   | {apr_std:>20.6} | {gguf_std:>20.6} | {ratio:>16.4}"
        );
        if layer_idx == ANOMALY_LAYER {
            anomaly_ratio = Some(ratio);
        }
    }

    // ---- H1/H2 verdict ----
    eprintln!();
    let ratio = anomaly_ratio.expect("ANOMALY_LAYER index always set in the loop");
    let in_h1_band = ratio.abs() >= 1.0 / RATIO_TOL && ratio.abs() <= RATIO_TOL;
    if in_h1_band {
        eprintln!(
            "M-FFN-GGUF-3 verdict: **H1 CONFIRMED** at layer {ANOMALY_LAYER} (ratio {ratio:.4} in [{:.4}, {RATIO_TOL:.4}])",
            1.0 / RATIO_TOL
        );
        eprintln!("  → APR layer-3 ffn_swigl is normal model behavior (matches GGUF).");
        eprintln!("  → SHIP-007 root cause is ELSEWHERE — pivot to lm_head /");
        eprintln!("    post-FFN residual / token-position correlation surface.");
    } else {
        eprintln!(
            "M-FFN-GGUF-3 verdict: **H2 CONFIRMED** at layer {ANOMALY_LAYER} (ratio {ratio:.4} outside [{:.4}, {RATIO_TOL:.4}])",
            1.0 / RATIO_TOL
        );
        eprintln!("  → APR-side bug confirmed — fix at `inference.rs:160-164`");
        eprintln!("    swigl elementwise multiply.");
    }
    eprintln!();
    eprintln!("Per FALSIFY-FFN-GGUF-003: this harness DISCHARGES the rule (it");
    eprintln!("reports a single H1 or H2 verdict). Per FALSIFY-FFN-GGUF-004:");
    eprintln!("the SHIP-007 root-cause fix PR title/body MUST cite either H1");
    eprintln!("or H2 and one of:");
    eprintln!("  {{ffn_swigl, swigl_elementwise_multiply, lm_head,");
    eprintln!("    post_ffn_residual, token_position_correlation}}");

    // The harness's job is to PRODUCE the H1/H2 verdict, not to assert.
    // Both verdicts are valid outcomes — only fix-PR-cites-stage
    // (FALSIFY-FFN-GGUF-004) requires the operator to do something with
    // the result.
}
