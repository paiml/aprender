//! M32d Step 5 regression test — context-awareness post Q/K RMSNorm fix.
//!
//! Pre-fix (forward_qwen3_moe missing per-head Q/K RMSNorm), `apr run`
//! produced "%%%%%%%%" gibberish — greedy argmax repeating one token
//! regardless of prompt. The forward path was producing context-INVARIANT
//! logits because attention scores compounded unboundedly through 48
//! layers (no Q/K norm to gate them).
//!
//! Post-fix (this PR adds per-head Q/K RMSNorm to forward_qwen3_moe per
//! GH-279, mirroring the dense path's adaptive_ffn.rs:174-179):
//! `apr run` produces coherent English ("Human: What is 2+", "Human:
//! What is the difference between a function and a method in Python?")
//! and the argmax varies with prompt.
//!
//! This regression test asserts the **context-awareness invariant**:
//! two distinct prompts must produce distinct argmax tokens. If this
//! test fails again, the per-head Q/K RMSNorm has regressed.
//!
//! Skipped when GGUF absent (fixture-absent ≠ defect, per
//! M32c.2.2.2.1.4 convention).
//!
//! References:
//!   - companion `claude-code-parity-apr` § "M32d FAST PATH" (M34, 2026-05-01)
//!   - GH-279 (Qwen3 per-head Q/K RMSNorm)
//!   - aprender adaptive_ffn.rs:174-179 (dense-path reference impl)

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

#[test]
fn f_qw3_moe_step5_001_context_aware_argmax() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!("F-QW3-MOE-STEP5-001: skipped — no cached Qwen3-Coder GGUF");
        return;
    };

    eprintln!("F-QW3-MOE-STEP5-001: context-aware argmax against {gguf_path}");

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");
    let data = mapped.data();
    let _transformer = QuantizedGGUFTransformer::from_gguf_for_moe(&mapped.model, data)
        .expect("from_gguf_for_moe");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("from_mapped");

    let mut moe_layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        moe_layers.push(
            load_qwen3_moe_layer(&mapped.model, data, layer_idx)
                .unwrap_or_else(|e| panic!("layer {layer_idx} load: {e:?}")),
        );
    }

    // Two distinct prompts encoded via the model's tokenizer fall back
    // to single-token synthetic IDs if encoding fails. The point is
    // that the input differs, so the output argmax should differ.
    let prompt_a = mapped
        .model
        .encode("What is 2+2?")
        .unwrap_or_else(|| vec![100u32]);
    let prompt_b = mapped
        .model
        .encode("Hello world")
        .unwrap_or_else(|| vec![200u32]);

    assert_ne!(
        prompt_a, prompt_b,
        "prompts must encode to distinct token sequences"
    );

    let logits_a = model
        .forward_qwen3_moe(
            &prompt_a,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
        )
        .expect("forward A succeeds");
    let logits_b = model
        .forward_qwen3_moe(
            &prompt_b,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
        )
        .expect("forward B succeeds");

    let argmax_a = argmax(&logits_a);
    let argmax_b = argmax(&logits_b);

    eprintln!(
        "F-QW3-MOE-STEP5-001:\n  prompt_a → argmax={argmax_a}\n  prompt_b → argmax={argmax_b}"
    );

    // Pre-fix: argmax_a == argmax_b (and == some degenerate '%' token).
    // Post-fix: argmax depends on context, so they should differ.
    //
    // Note: a healthy model could in some rare cases produce the same
    // argmax for different prompts (e.g., both ending with the same
    // BOS-fallback path), but for these two clearly-different prompts
    // — one math, one greeting — the argmax should differ.
    assert_ne!(
        argmax_a, argmax_b,
        "F-QW3-MOE-STEP5-001 FAIL: argmax is context-invariant. \
         This is the same symptom as the pre-fix gibberish — \
         per-head Q/K RMSNorm has regressed (forward_qwen3_moe.rs)."
    );

    // Stronger invariant: the top-1 logit value should not be
    // pathologically larger than top-2 (pre-fix it was, by ~10x or
    // more).
    let top_two_gap_a = top_two_gap(&logits_a);
    let top_two_gap_b = top_two_gap(&logits_b);
    eprintln!("  top-2 gap A: {top_two_gap_a:.4}\n  top-2 gap B: {top_two_gap_b:.4}");
    assert!(
        top_two_gap_a < 50.0 && top_two_gap_b < 50.0,
        "F-QW3-MOE-STEP5-001 FAIL: top-1 dominates logits with gap > 50 \
         (top-2 gap A={top_two_gap_a:.4}, B={top_two_gap_b:.4}). \
         Indicator of degenerate forward — Q/K norm or related fix has regressed."
    );
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .expect("logits non-empty")
}

fn top_two_gap(logits: &[f32]) -> f32 {
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &v in logits {
        if v > top1 {
            top2 = top1;
            top1 = v;
        } else if v > top2 {
            top2 = v;
        }
    }
    top1 - top2
}
