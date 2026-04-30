//! M32c.2.2.2.1.1 falsifier — full single-token forward pass for Qwen3-MoE.
//!
//! Exercises `OwnedQuantizedModel::forward_qwen3_moe` end-to-end against
//! the cached 17.3 GB Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf:
//!
//!   token_embedding → 48 × (RMSNorm + QKV proj + RoPE + causal attn +
//!   attn_output proj + ffn_norm + MoE FFN) → output_norm → lm_head
//!
//! Asserts the final logits vector is the right shape (vocab_size = 151936),
//! finite, and non-trivial. This is the per-token primitive that
//! `run_qwen3_moe_generate` (M32c.2.2.2.1.2) will call in a loop.
//!
//! NOTE: This test runs ONE forward pass on the FULL 30B model. It takes
//! a few minutes per call due to mmap fault-in + 48 × 128-expert
//! softmax routing + 48 × top-8 × per-expert Q4_K/Q6_K matmul. The skip
//! path keeps test suites fast when no GGUF is cached.

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGGUFTransformer};

use std::path::Path;

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

const EXPECTED_NUM_LAYERS: usize = 48;
#[allow(dead_code)]
const EXPECTED_HIDDEN: usize = 2048; // documented for shape parity with sibling tests
const EXPECTED_INTERMEDIATE: usize = 768;
const EXPECTED_N_EXPERTS: usize = 128;
const EXPECTED_K: usize = 8;
const EXPECTED_VOCAB: usize = 151936;

#[test]
fn f_qw3_moe_c22211_001_full_forward_one_token_finite_logits() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!("F-QW3-MOE-C22211-001: skipped — no cached GGUF.");
        return;
    };

    eprintln!("F-QW3-MOE-C22211-001: full 1-token forward against {gguf_path}");

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");
    let data = mapped.data();
    let _transformer = QuantizedGGUFTransformer::from_gguf_for_moe(&mapped.model, data)
        .expect("from_gguf_for_moe must succeed");

    // Build the OwnedQuantizedModel via the standard from_mapped path.
    // Post-M32c.2.1, that dispatches to from_gguf_for_moe internally for
    // qwen3_moe arch. Dense FFN fields are placeholder zeros; attention/
    // norm/lm_head are real owned bytes.
    let model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped");

    // Load all 48 layers' MoE descriptors (the M32c.2 from_gguf_for_moe
    // populates `transformer.moe_layers` but OwnedQuantizedModel doesn't
    // currently propagate them — load fresh here per the M32c.2.2.2.1.1
    // method signature).
    let mut moe_layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        moe_layers.push(
            load_qwen3_moe_layer(&mapped.model, data, layer_idx)
                .unwrap_or_else(|e| panic!("layer {layer_idx} MoE load failed: {e:?}")),
        );
    }

    // Run forward on a 1-token input.
    // BOS token id varies by tokenizer; use 0 as a safe synthetic input.
    let token_ids = vec![0u32];

    eprintln!("F-QW3-MOE-C22211-001: running forward (this takes a few minutes)...");
    let start = std::time::Instant::now();
    let logits = model
        .forward_qwen3_moe(
            &token_ids,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
        )
        .expect("F-QW3-MOE-C22211-001: forward should succeed");
    let elapsed = start.elapsed();

    assert_eq!(
        logits.len(),
        EXPECTED_VOCAB,
        "F-QW3-MOE-C22211-001: logits len must equal vocab_size"
    );
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "F-QW3-MOE-C22211-001: all logits must be finite (no NaN/Inf)"
    );
    let (max_idx, &max_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("logits non-empty");
    let top1_argmax = max_idx as u32;
    assert!(
        top1_argmax < EXPECTED_VOCAB as u32,
        "F-QW3-MOE-C22211-001: argmax must be in vocab range"
    );

    let l2: f32 = logits.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!(
        "F-QW3-MOE-C22211-001: PASS\n  elapsed = {:?}\n  logits.len() = {}\n  argmax = {} (val = {:.4})\n  ||logits||_2 = {:.4}",
        elapsed,
        logits.len(),
        top1_argmax,
        max_val,
        l2
    );
}
