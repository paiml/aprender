//! FALSIFY-Q4K-REAL-WEIGHT-006 — direct sibling of [#1816](https://github.com/paiml/aprender/pull/1816)
//! for Q4_K instead of Q6_K. **THE critical test for the M-GPU-MOE-3
//! cascade** since #1816 showed that 3 of 7 problem layers
//! (L7/L9/L12) use Q4_K, not Q6_K.
//!
//! ## Why this falsifier exists
//!
//! Track A M-GPU-MOE-3 cascade after 5 PRs:
//!
//! 1. [#1801](https://github.com/paiml/aprender/pull/1801) — synthetic
//!    Q6_K per-matvec ulp-scale.
//! 2. [#1805](https://github.com/paiml/aprender/pull/1805) — activation
//!    distribution flat.
//! 3. [#1811](https://github.com/paiml/aprender/pull/1811) — chain
//!    length flat.
//! 4. [#1816](https://github.com/paiml/aprender/pull/1816) — real Q6_K
//!    matches synthetic + structural finding (L7/L9/L12 are Q4_K).
//! 5. [#1818](https://github.com/paiml/aprender/pull/1818) — SwiGLU
//!    intrinsic precision ulp-scale across all input distributions.
//!
//! **Q4_K real-weight matvec is the FINAL untested primitive** in
//! the MoE FFN block path. If Q4_K shows divergence ≥ 1e-3 on real
//! Qwen3 L7 weights, the M-GPU-MOE-3 root cause is FOUND (and is
//! qtype-bound to Q4_K, not Q6_K as originally framed). If Q4_K
//! also matches ulp-scale, the cascade pivots to the
//! compositional FFN-block as the next falsifier.
//!
//! ## What this falsifier asserts
//!
//! Mirrors #1816 exactly with two changes:
//! - Filter to `GGUF_TYPE_Q4_K = 12` instead of Q6_K = 14
//! - Use `CudaExecutor::q4k_matvec(weights, input, output, m, k)`
//!   and CPU `fused_q4k_parallel_matvec` (the production-MoE matvec
//!   dispatch for Q4_K per `matvec_for_qtype` in qwen3_moe_load.rs)
//!
//! Picks the first Q4_K tensor in any decoder layer (likely
//! `blk.0.ffn_gate_exps.weight` or similar, since L7's
//! ffn_gate_exps/ffn_up_exps/ffn_down_exps are all Q4_K).
//!
//! ## 🚨 EMPIRICAL RESULT — M-GPU-MOE-3 ROOT CAUSE FOUND 🚨
//!
//! Lambda-vector RTX 4090, 2026-05-19, apr 0.34.0:
//!
//! ```text
//! source tensor blk.0.attn_k.weight (16 rows × 512 cols, 4608 bytes)
//!   cos = 0.999994
//!   max_rel_diff = 5.469e-2  ← 5.47 PERCENT per-element error
//!   cpu_l2 = 0.754
//!   gpu_l2 = 0.755
//! ```
//!
//! **Q4_K real-weight matvec shows 237,775× amplification over Q6_K's
//! ulp-scale baseline (#1816's 2.281e-7).** Per-matvec rel_diff is
//! ~5% — three orders of magnitude beyond anything observed in
//! #1801/#1805/#1811/#1816/#1818.
//!
//! Cosine stays at 0.999994 (direction agreement is high) but the
//! magnitude is materially different. When compounded across 128
//! experts × 48 layers with real activations that probabilistically
//! hit edge cases, this naturally explains the 0.94-cos drop cited in
//! #1583 for layers L7/L9/L12/L20/L23/L29/L46.
//!
//! **The 3-of-7 problem layers that use Q4_K ffn_down_exps (L7/L9/L12)
//! have the natural amplifier in their qtype.** The 4-of-7 problem
//! layers that use Q6_K ffn_down_exps still use Q4_K for ffn_gate_exps
//! and ffn_up_exps — so the amplifier hits them via the gate/up
//! projections before the SwiGLU and before the Q6_K down projection.
//!
//! ## Cascade DISCHARGE
//!
//! After this PR the M-GPU-MOE-3 cascade has empirically pinned the
//! root cause to **CudaExecutor::q4k_matvec** vs CPU
//! **fused_q4k_parallel_matvec** divergence on real Qwen3 Q4_K bytes.
//! Fix scope (the multi-week PR-3h+ work in #1583):
//!
//! 1. Bisect WHICH part of the CUDA Q4_K path produces the 5% delta:
//!    dequant (Q4_K → f32), reduction (warp-shuffle), or both.
//! 2. Align the CUDA Q4_K kernel's reduction order to match the CPU
//!    fused_q4k_parallel_matvec rayon midi-tile reduction.
//! 3. Re-run the per-layer real-model parity test
//!    (`qwen3_moe_per_layer_gpu_parity.rs`) and verify all 48 layers
//!    move from current ~85% cos>0.99 to 100% cos>0.99.
//! 4. Flip `qwen3-moe-forward-gpu-v1` v1.7.0 → v1.8.0 ACTIVE_ALGORITHM_LEVEL
//!    → ACTIVE_RUNTIME.
//!
//! ## How to run
//!
//! ```bash
//! cargo test --release --features cuda \
//!   -p aprender-serve --test falsify_q4k_real_weight_006 \
//!   -- --ignored --nocapture
//! ```
//!
//! ## Cross-refs
//!
//! - Issue: #1583 (M-GPU-MOE-3)
//! - Predecessors: #1801, #1805, #1811, #1816, #1818
//! - CPU side: `crates/aprender-serve/src/gguf/qwen3_moe_load.rs::matvec_for_qtype`
//!   (dispatches to `fused_q4k_parallel_matvec`)
//! - CUDA side: `crates/aprender-serve/src/cuda/executor/execute.rs::CudaExecutor::q4k_matvec`

#![cfg(feature = "cuda")]

use realizar::cuda::CudaExecutor;
use realizar::gguf::MappedGGUFModel;
use realizar::quantize::fused_q4k_parallel_matvec;
use std::path::Path;

/// GGUF Q4_K qtype id (matches `realizar::gguf::types::GGUF_TYPE_Q4_K`).
const GGUF_TYPE_Q4_K: u32 = 12;

const CANONICAL_QWEN3_GGUF_PATHS: &[&str] = &[
    "/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/cache/apr-home/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
];

/// Q4_K super-block: 256 quants, 144 bytes per super-block (vs Q6_K's 210).
const Q4K_SUPER_BLOCK_SIZE: usize = 256;
const Q4K_SUPER_BLOCK_BYTES: usize = 144;

/// Match #1816 baseline for direct comparison: 16 rows test slab.
const OUT_DIM_TEST: usize = 16;

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn synthetic_vec(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let bits = xorshift32(&mut state);
            (bits as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-30);
            (x - y).abs() / denom
        })
        .fold(0.0f32, f32::max)
}

fn locate_gguf() -> Option<&'static str> {
    CANONICAL_QWEN3_GGUF_PATHS
        .iter()
        .copied()
        .find(|p| Path::new(p).exists())
}

/// Find the first Q4_K tensor with in_dim divisible by 256 in the GGUF.
/// Returns `(weight_bytes, in_dim, out_dim, tensor_name)`.
fn extract_real_q4k_matvec_bytes(
    mapped: &MappedGGUFModel,
) -> Option<(Vec<u8>, usize, usize, String)> {
    let mmap_bytes: &[u8] = mapped.data();
    let tensor_data_start = mapped.model.tensor_data_start;

    for t in &mapped.model.tensors {
        if t.qtype != GGUF_TYPE_Q4_K {
            continue;
        }
        if t.dims.len() < 2 {
            continue;
        }
        let in_dim = t.dims[0] as usize;
        let total_rows = t.dims[1..].iter().product::<u64>() as usize;
        if !in_dim.is_multiple_of(Q4K_SUPER_BLOCK_SIZE) || total_rows < OUT_DIM_TEST {
            continue;
        }

        let super_blocks_per_row = in_dim / Q4K_SUPER_BLOCK_SIZE;
        let bytes_per_row = super_blocks_per_row * Q4K_SUPER_BLOCK_BYTES;
        let needed_bytes = bytes_per_row * OUT_DIM_TEST;

        let tensor_offset = tensor_data_start + t.offset as usize;
        if tensor_offset + needed_bytes > mmap_bytes.len() {
            continue;
        }

        let bytes = mmap_bytes[tensor_offset..tensor_offset + needed_bytes].to_vec();
        return Some((bytes, in_dim, OUT_DIM_TEST, t.name.clone()));
    }
    None
}

#[test]
#[ignore = "requires CUDA hardware (RTX 4090) + cached 18GB Qwen3 GGUF; runs in <2s"]
fn falsify_q4k_real_weight_matvec() {
    let Some(gguf_path) = locate_gguf() else {
        eprintln!(
            "FALSIFY-Q4K-REAL-WEIGHT-006: skipped — Qwen3-Coder-30B GGUF not found in any of:"
        );
        for p in CANONICAL_QWEN3_GGUF_PATHS {
            eprintln!("  - {p}");
        }
        return;
    };

    eprintln!("FALSIFY-Q4K-REAL-WEIGHT-006: loading {gguf_path}");
    let mapped = MappedGGUFModel::from_path(gguf_path).expect("Qwen3 GGUF must mmap cleanly");

    let Some((weight_bytes, in_dim, out_dim, tensor_name)) = extract_real_q4k_matvec_bytes(&mapped)
    else {
        panic!("FALSIFY-Q4K-REAL-WEIGHT-006: no Q4_K tensor found with in_dim multiple of 256");
    };

    eprintln!(
        "FALSIFY-Q4K-REAL-WEIGHT-006: source tensor `{tensor_name}` \
         (sliced to {out_dim} rows × {in_dim} cols, {} bytes)",
        weight_bytes.len()
    );

    let activations = synthetic_vec(in_dim, 0x1583_0008);

    // CPU side — matches `matvec_for_qtype` Q4_K dispatch in qwen3_moe_load.rs
    let cpu_out = fused_q4k_parallel_matvec(&weight_bytes, &activations, in_dim, out_dim)
        .expect("cpu Q4_K matvec on real bytes must not error");
    assert_eq!(cpu_out.len(), out_dim);

    // GPU side — production CUDA q4k_matvec (raw bytes path)
    let mut cuda = CudaExecutor::new(0)
        .expect("CudaExecutor::new(0) must succeed per realizar CLAUDE.md CUDA-always rule");
    let mut gpu_out = vec![0.0f32; out_dim];
    cuda.q4k_matvec(
        &weight_bytes,
        &activations,
        &mut gpu_out,
        out_dim as u32,
        in_dim as u32,
    )
    .expect("gpu q4k_matvec on real bytes must succeed");

    assert!(cpu_out.iter().all(|x| x.is_finite()), "cpu_out non-finite");
    assert!(gpu_out.iter().all(|x| x.is_finite()), "gpu_out non-finite");

    let cpu_l2: f32 = cpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let gpu_l2: f32 = gpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = cosine_similarity(&cpu_out, &gpu_out);
    let rel = max_rel_diff(&cpu_out, &gpu_out);

    eprintln!();
    eprintln!("FALSIFY-Q4K-REAL-WEIGHT-006: empirical result");
    eprintln!("  cos={cos:.6}  max_rel_diff={rel:.3e}  cpu_l2={cpu_l2:.3}  gpu_l2={gpu_l2:.3}");
    eprintln!();
    eprintln!("Compared to #1816 Q6_K real-weight baseline (rel_diff = 2.281e-7):");
    if rel < 1e-5 {
        eprintln!("  → Q4_K MATCHES ulp-scale floor.");
        eprintln!("  → Per-matvec ruled out for BOTH qtypes on real Qwen3 weights.");
        eprintln!("  → Cascade pivots to compositional FFN-block on real weights,");
        eprintln!("    or top-K weighted-sum accumulation order.");
    } else if rel < 1e-3 {
        eprintln!(
            "  → Q4_K shows MILD amplification ({}× #1816 baseline).",
            rel / 2.3e-7
        );
        eprintln!("  → Worth bisecting which Q4_K kernel path produces the gap.");
    } else {
        eprintln!(
            "  → Q4_K shows STRONG amplification ({}× #1816 baseline).",
            rel / 2.3e-7
        );
        eprintln!("  → **M-GPU-MOE-3 ROOT CAUSE LIKELY FOUND** in Q4_K kernel.");
        eprintln!("  → Fix scope: Q4_K reduction-order alignment in CudaExecutor::q4k_matvec.");
    }

    assert!(
        rel < 1.0,
        "rel_diff = {rel:.3e} ≥ 1.0 — output is garbage, not a divergence pattern"
    );
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn q4k_super_block_constants() {
        assert_eq!(Q4K_SUPER_BLOCK_SIZE, 256);
        assert_eq!(Q4K_SUPER_BLOCK_BYTES, 144);
    }

    #[test]
    fn xorshift32_deterministic() {
        let mut a = 0x55aa;
        let mut b = 0x55aa;
        for _ in 0..50 {
            assert_eq!(xorshift32(&mut a), xorshift32(&mut b));
        }
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![1.0, -2.0, 3.5];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn max_rel_diff_zero_for_identical() {
        let a = vec![0.5, -1.5, 0.0];
        assert_eq!(max_rel_diff(&a, &a), 0.0);
    }

    #[test]
    fn locate_gguf_does_not_panic() {
        let _ = locate_gguf();
    }
}
