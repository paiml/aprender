//! M-GPU-MOE-1.2 — `FALSIFY-QW3-MOE-GPU-PARITY-001`: cosine-similarity
//! parity gate between APR's CPU `forward_qwen3_moe` reference and the
//! GPU `OwnedQuantizedModelCuda::forward_qwen3_moe_cuda` integration
//! (M-GPU-MOE-1.1.2, aprender PR #1477).
//!
//! Contract: [`contracts/qwen3-moe-forward-gpu-v1.yaml`] v1.1.0 —
//! `FALSIFY-QW3-MOE-GPU-PARITY-001` (formal:
//! `cosine_similarity(apr_gpu_logits, apr_cpu_logits) ≥ 0.99` on a
//! fixed prompt against the cached 17.3 GB Qwen3-Coder GGUF).
//!
//! ## Heavy-test layout
//!
//! 1. The 17.3 GB `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` weights,
//!    mmap'd under `MappedGGUFModel::from_path`. Cached on
//!    lambda-vector at the paths in `CANONICAL_QWEN3_CODER_GGUF_PATHS`.
//! 2. Requires CUDA (RTX 4090 on lambda-vector). The test is gated
//!    behind `#[cfg(feature = "cuda")]` and `#[ignore]`; CI invokes it
//!    explicitly via:
//!
//!        cargo test -p aprender-serve --test qwen3_moe_gpu_parity \
//!            --features cuda -- --include-ignored
//!
//! If the cached GGUF is not present, the test prints a skip line and
//! returns Ok (matches the sibling CPU-vs-HF-FP16 test pattern).
//!
//! ## What the test does
//!
//! 1. Loads the GGUF once (single mmap).
//! 2. Builds `moe_layers: Vec<Qwen3MoeQuantizedLayer>` once.
//! 3. Builds CPU `OwnedQuantizedModel` #1 → runs `forward_qwen3_moe`
//!    on a fixed prompt → `cpu_logits` (this is the ground-truth
//!    LAZY-FUSED-MATVEC reference).
//! 4. Builds CPU `OwnedQuantizedModel` #2 → wraps into
//!    `OwnedQuantizedModelCuda` → runs `forward_qwen3_moe_cuda` on the
//!    same prompt → `gpu_logits`.
//! 5. Computes cosine similarity over the full 151936-dim vocab.
//! 6. Asserts `cos_sim ≥ 0.99` per
//!    `qwen3-moe-forward-gpu-v1` v1.1.0 ::
//!    `FALSIFY-QW3-MOE-GPU-PARITY-001`.

#![cfg(feature = "cuda")]

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

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
fn falsify_qw3_moe_gpu_parity_001_cosine_vs_cpu() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "FALSIFY-QW3-MOE-GPU-PARITY-001: skipped — no cached Qwen3-Coder GGUF in {CANONICAL_QWEN3_CODER_GGUF_PATHS:?}"
        );
        return;
    };

    eprintln!("FALSIFY-QW3-MOE-GPU-PARITY-001: cosine vs CPU LAZY-FUSED-MATVEC");
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
        "FALSIFY-QW3-MOE-GPU-PARITY-001: running CPU forward on {} prompt tokens (this takes a few minutes)...",
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
        .expect("FALSIFY-QW3-MOE-GPU-PARITY-001: CPU forward should succeed");
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

    let gpu_inner_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #2");
    // PR #1477 (M-GPU-MOE-1.1.2) flips forward_qwen3_moe_cuda's receiver to
    // `&mut self` (kernel cache mutation); the `mut` here is forward-looking.
    #[allow(unused_mut)]
    let mut gpu_model = OwnedQuantizedModelCuda::new(gpu_inner_model, 0)
        .expect("OwnedQuantizedModelCuda::new(model, 0) should succeed on RTX 4090");

    eprintln!(
        "FALSIFY-QW3-MOE-GPU-PARITY-001: running GPU forward on {} prompt tokens...",
        CANONICAL_PROMPT_TOKENS.len()
    );
    let start = std::time::Instant::now();
    let gpu_logits = gpu_model
        .forward_qwen3_moe_cuda(
            CANONICAL_PROMPT_TOKENS,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
        )
        .expect("FALSIFY-QW3-MOE-GPU-PARITY-001: GPU forward should succeed");
    let gpu_elapsed = start.elapsed();

    assert_eq!(
        gpu_logits.len(),
        EXPECTED_VOCAB,
        "GPU logits len must equal vocab_size"
    );

    // M-GPU-MOE-1.4 (per qwen3-moe-forward-gpu-v1 v1.4.0 amendment):
    // diagnostic stats printed BEFORE the finiteness assertion to
    // give bisection data when the assertion fires. This is
    // load-bearing for the M-GPU-MOE-1.4 NaN/Inf bisection
    // (evidence file pending).
    let nan_count = gpu_logits.iter().filter(|v| v.is_nan()).count();
    let inf_count = gpu_logits.iter().filter(|v| v.is_infinite()).count();
    let finite_count = gpu_logits.iter().filter(|v| v.is_finite()).count();
    let first_nan_idx = gpu_logits.iter().position(|v| v.is_nan());
    let first_inf_idx = gpu_logits.iter().position(|v| v.is_infinite());
    let finite_min = gpu_logits.iter().filter(|v| v.is_finite()).cloned()
        .fold(f32::INFINITY, f32::min);
    let finite_max = gpu_logits.iter().filter(|v| v.is_finite()).cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "FALSIFY-QW3-MOE-GPU-PARITY-001 finiteness diagnostic:\n  \
         total      = {}\n  \
         finite     = {}\n  \
         nan        = {} (first idx: {:?})\n  \
         inf        = {} (first idx: {:?})\n  \
         finite_min = {:.6}\n  \
         finite_max = {:.6}",
        gpu_logits.len(), finite_count, nan_count, first_nan_idx,
        inf_count, first_inf_idx, finite_min, finite_max,
    );

    assert!(
        gpu_logits.iter().all(|v| v.is_finite()),
        "all GPU logits must be finite (no NaN/Inf) — see diagnostic above. \
         Per qwen3-moe-forward-gpu-v1 v1.4.0 M-GPU-MOE-1.4 bisection plan, \
         next step: extend `apr trace` to capture per-stage MoE GPU tensors \
         and diff CPU-vs-GPU per-stage to find first NaN/Inf-producing stage."
    );

    let cos = cosine_similarity(&cpu_logits, &gpu_logits);

    let (cpu_argmax, &cpu_max_val) = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("CPU logits non-empty");
    let cpu_argmax = cpu_argmax as u32;

    let (gpu_argmax, &gpu_max_val) = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("GPU logits non-empty");
    let gpu_argmax = gpu_argmax as u32;

    eprintln!(
        "FALSIFY-QW3-MOE-GPU-PARITY-001:\n  cpu_elapsed   = {cpu_elapsed:?}\n  gpu_elapsed   = {gpu_elapsed:?}\n  cos_sim       = {cos:.6}\n  threshold     = {COSINE_THRESHOLD}\n  cpu_argmax    = {cpu_argmax} (val = {cpu_max_val:.4})\n  gpu_argmax    = {gpu_argmax} (val = {gpu_max_val:.4})"
    );

    assert!(
        cos >= COSINE_THRESHOLD,
        "FALSIFY-QW3-MOE-GPU-PARITY-001: \
         cosine_similarity(apr_gpu_logits, apr_cpu_logits) = {cos:.6} \
         is NOT ≥ {COSINE_THRESHOLD}. Per contract `if_fails`: \
         GPU kernel diverges from CPU LAZY-FUSED-MATVEC reference. \
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

#[test]
fn cosine_similarity_within_threshold() {
    let a = vec![0.99_f32, 0.01, 0.0, 0.0];
    let b = vec![1.00_f32, 0.00, 0.0, 0.0];
    let cos = cosine_similarity(&a, &b);
    assert!(
        cos >= 0.99,
        "near-parallel vectors should have cosine ≥ 0.99 (got {cos})"
    );
}
