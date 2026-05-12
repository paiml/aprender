//! M-GPU-MOE-2.3 — wgpu sibling of `qwen3_moe_gpu_parity.rs`.
//!
//! Asserts cosine ≥0.99 between APR's CPU `forward_qwen3_moe` reference
//! and the wgpu `OwnedQuantizedModelWgpu::forward_qwen3_moe_wgpu`
//! integration (M-GPU-MOE-2.2, pending).
//!
//! Contract: [`contracts/qwen3-moe-forward-gpu-v1.yaml`] v1.2.0 —
//! `FALSIFY-QW3-MOE-GPU-PARITY-001` (formal:
//! `cosine_similarity(apr_wgpu_logits, apr_cpu_logits) ≥ 0.99` on a
//! fixed prompt against the cached 17.3 GB Qwen3-Coder GGUF). Same
//! threshold as the CUDA sibling (`qwen3_moe_gpu_parity.rs`); same
//! falsifier ID — wgpu is a second backend implementing the same
//! contract, not a different gate.
//!
//! ## Heavy-test layout
//!
//! 1. Same 17.3 GB Qwen3-Coder GGUF as the CUDA sibling.
//! 2. Requires a wgpu-capable adapter (Apple Silicon Metal, AMD via
//!    Vulkan, Intel ARC via Vulkan, or NVIDIA via Vulkan/DX12 —
//!    NOT NVIDIA via CUDA, which is the cuda sibling's job).
//! 3. Gated behind `#[cfg(feature = "gpu")]` (the wgpu feature flag,
//!    matching the gate on `OwnedQuantizedModelWgpu` itself per
//!    `crates/aprender-serve/src/gguf/mod.rs`) and `#[ignore]`.
//!
//! Run with:
//!
//!     cargo test -p aprender-serve --test qwen3_moe_wgpu_parity \
//!         --features gpu -- --include-ignored
//!
//! ## What the test does (when --include-ignored)
//!
//! 1. Loads the GGUF once (single mmap).
//! 2. Builds CPU `OwnedQuantizedModel` #1 → runs `forward_qwen3_moe`
//!    on a fixed prompt → `cpu_logits` (the LAZY-FUSED-MATVEC
//!    ground truth — same reference the cuda sibling test uses).
//! 3. Builds CPU `OwnedQuantizedModel` #2 → wraps into
//!    `OwnedQuantizedModelWgpu` → runs `forward_qwen3_moe_wgpu` on
//!    the same prompt → `wgpu_logits`.
//! 4. Computes cosine similarity over the full 151936-dim vocab.
//! 5. Asserts `cos_sim ≥ 0.99`.
//!
//! ## When the test passes
//!
//! - M-GPU-MOE-2.0 stub returns `UnsupportedOperation` so this test
//!   currently panics in step 3 (correct behaviour for a falsifier
//!   against an incomplete implementation).
//! - M-GPU-MOE-2.1 (per-expert wgpu helpers) + M-GPU-MOE-2.2 (full
//!   forward integration analog of `forward_qwen3_moe_cuda`) must
//!   both land before this test passes on hardware.
//! - On hardware with wgpu support, run with `--include-ignored` to
//!   exercise the falsifier. PASS discharges
//!   `FALSIFY-QW3-MOE-GPU-PARITY-001` for the wgpu backend (the
//!   cuda backend is discharged by the sibling test
//!   `qwen3_moe_gpu_parity.rs`).

#![cfg(feature = "gpu")]

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelWgpu};

use std::path::Path;

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/cache/apr-home/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_INTERMEDIATE: usize = 768;
const EXPECTED_N_EXPERTS: usize = 128;
const EXPECTED_K: usize = 8;
const EXPECTED_VOCAB: usize = 151936;

const COSINE_THRESHOLD: f32 = 0.99;

const CANONICAL_PROMPT_TOKENS: &[u32] = &[785, 9217, 308];

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "cosine_similarity: vectors must be same length"
    );
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[test]
#[ignore]
fn falsify_qw3_moe_wgpu_parity_001_cosine_vs_cpu() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "FALSIFY-QW3-MOE-GPU-PARITY-001 (wgpu): skipped — no cached Qwen3-Coder GGUF in {CANONICAL_QWEN3_CODER_GGUF_PATHS:?}"
        );
        return;
    };

    eprintln!("FALSIFY-QW3-MOE-GPU-PARITY-001 (wgpu): cosine vs CPU LAZY-FUSED-MATVEC");
    eprintln!("  gguf:    {gguf_path}");
    eprintln!("  prompt:  {CANONICAL_PROMPT_TOKENS:?}");

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");
    let data = mapped.data();

    let mut moe_layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        moe_layers.push(
            load_qwen3_moe_layer(&mapped.model, data, layer_idx)
                .unwrap_or_else(|e| panic!("layer {layer_idx} MoE load failed: {e:?}")),
        );
    }

    let cpu_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #1");

    eprintln!(
        "FALSIFY-QW3-MOE-GPU-PARITY-001 (wgpu): running CPU forward on {} prompt tokens (this takes a few minutes)...",
        CANONICAL_PROMPT_TOKENS.len()
    );
    let start = std::time::Instant::now();
    let cpu_logits = cpu_model
        .forward_qwen3_moe(
            CANONICAL_PROMPT_TOKENS,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
        )
        .expect("FALSIFY-QW3-MOE-GPU-PARITY-001 (wgpu): CPU forward should succeed");
    let cpu_elapsed = start.elapsed();

    assert_eq!(
        cpu_logits.len(),
        EXPECTED_VOCAB,
        "CPU logits len must equal vocab_size"
    );
    assert!(
        cpu_logits.iter().all(|v| v.is_finite()),
        "all CPU logits must be finite (no NaN/Inf)"
    );

    let wgpu_inner_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #2");
    let wgpu_model = OwnedQuantizedModelWgpu::new(wgpu_inner_model)
        .expect("OwnedQuantizedModelWgpu::new should succeed on a wgpu-capable adapter");

    eprintln!(
        "FALSIFY-QW3-MOE-GPU-PARITY-001 (wgpu): running wgpu forward on {} prompt tokens...",
        CANONICAL_PROMPT_TOKENS.len()
    );
    let start = std::time::Instant::now();
    let wgpu_logits = wgpu_model
        .forward_qwen3_moe_wgpu(
            CANONICAL_PROMPT_TOKENS,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
        )
        .expect("FALSIFY-QW3-MOE-GPU-PARITY-001 (wgpu): wgpu forward should succeed");
    let wgpu_elapsed = start.elapsed();

    assert_eq!(
        wgpu_logits.len(),
        EXPECTED_VOCAB,
        "wgpu logits len must equal vocab_size"
    );
    assert!(
        wgpu_logits.iter().all(|v| v.is_finite()),
        "all wgpu logits must be finite (no NaN/Inf)"
    );

    let cos = cosine_similarity(&cpu_logits, &wgpu_logits);

    let (cpu_argmax, &cpu_max_val) = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("CPU logits non-empty");
    let cpu_argmax = cpu_argmax as u32;

    let (wgpu_argmax, &wgpu_max_val) = wgpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("wgpu logits non-empty");
    let wgpu_argmax = wgpu_argmax as u32;

    eprintln!(
        "FALSIFY-QW3-MOE-GPU-PARITY-001 (wgpu):\n  cpu_elapsed   = {cpu_elapsed:?}\n  wgpu_elapsed  = {wgpu_elapsed:?}\n  cos_sim       = {cos:.6}\n  threshold     = {COSINE_THRESHOLD}\n  cpu_argmax    = {cpu_argmax} (val = {cpu_max_val:.4})\n  wgpu_argmax   = {wgpu_argmax} (val = {wgpu_max_val:.4})"
    );

    assert!(
        cos >= COSINE_THRESHOLD,
        "FALSIFY-QW3-MOE-GPU-PARITY-001 (wgpu): \
         cosine_similarity(apr_wgpu_logits, apr_cpu_logits) = {cos:.6} \
         is NOT ≥ {COSINE_THRESHOLD}. Per contract `if_fails`: \
         wgpu kernel diverges from CPU LAZY-FUSED-MATVEC reference. \
         Bisect via `apr trace --json --payload` (M32d Step 2 surface) \
         on both paths, layer-by-layer; first divergent stage is the \
         root cause."
    );
}

#[test]
fn cosine_similarity_unit_vectors() {
    let a = vec![1.0_f32, 0.0, 0.0];
    let b = vec![1.0_f32, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

    let c = vec![1.0_f32, 0.0, 0.0];
    let d = vec![0.0_f32, 1.0, 0.0];
    assert!(cosine_similarity(&c, &d).abs() < 1e-6);

    let e = vec![1.0_f32, 0.0, 0.0];
    let f = vec![-1.0_f32, 0.0, 0.0];
    assert!((cosine_similarity(&e, &f) - (-1.0)).abs() < 1e-6);
}

#[test]
fn cosine_similarity_handles_zero_vector() {
    let a = vec![0.0_f32; 8];
    let b = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(cosine_similarity(&a, &b), 0.0);
    assert_eq!(cosine_similarity(&b, &a), 0.0);
}
