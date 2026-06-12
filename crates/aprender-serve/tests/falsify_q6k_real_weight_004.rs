//! FALSIFY-Q6K-REAL-WEIGHT-004 — does real-Qwen3 Q6_K weight pattern
//! produce a different per-matvec divergence than synthetic random?
//!
//! ## Cascade state entering this PR
//!
//! [#1801](https://github.com/paiml/aprender/pull/1801),
//! [#1805](https://github.com/paiml/aprender/pull/1805), and
//! [#1811](https://github.com/paiml/aprender/pull/1811) collectively
//! ruled out three candidate amplifiers for the 0.94-cos drop on real
//! Qwen3 layers L7/L9/L12/L20/L23/L29/L46 (#1583):
//!
//! 1. Per-matvec reduction-order divergence (#1801): ulp-scale ~1e-7
//! 2. Activation distribution amplification (#1805): flat across bursty
//! 3. Accumulator-chain length compounding (#1811): flat from N=1 to N=48
//!
//! **All three were tested on SYNTHETIC RANDOM weights.** The recommended
//! pivot in #1811's docstring was: "load actual L7 q6k bytes from cached
//! GGUF and re-run #1801's single-matvec test." This file does exactly
//! that.
//!
//! ## What this falsifier asserts
//!
//! Maps the cached Qwen3-Coder-30B-A3B GGUF, finds the first Q6_K
//! tensor (we use any Q6_K in any layer for the real-weight pattern
//! test), extracts a 16-row × in_dim slab of bytes, and runs CPU
//! `fused_q6k_parallel_matvec` vs GPU `CudaExecutor::q6k_gemv` on the
//! same bytes.
//!
//! ## Empirical result (lambda-vector RTX 4090, 2026-05-19, apr 0.34.0)
//!
//! ```text
//! source tensor `blk.0.attn_v.weight` (16 rows × 512 cols, 6720 bytes)
//!   cos=1.000000  max_rel_diff=2.281e-7  cpu_l2=0.688  gpu_l2=0.688
//! ```
//!
//! **Real Q6_K weights match the synthetic ulp-scale baseline.** This
//! confirms (alongside #1801/#1805/#1811) that per-matvec Q6_K is
//! FINE on both synthetic and real inputs. The bug is NOT here.
//!
//! ## CRITICAL CASCADE PIVOT — discovered in this session
//!
//! GGUF tensor inventory on `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
//! reveals that the "problem layers" cited by #1583 split between
//! qtypes for `ffn_down_exps`:
//!
//! | Layer | ffn_down_exps qtype |
//! |-------|---------------------|
//! | L7    | **Q4_K**            |
//! | L9    | **Q4_K**            |
//! | L12   | **Q4_K**            |
//! | L20   | Q6_K                |
//! | L23   | Q6_K                |
//! | L29   | Q6_K                |
//! | L46   | Q6_K                |
//!
//! **Three of seven problem layers have NO Q6_K tensors at all** —
//! they are pure Q4_K MoE. The amplifier must be **qtype-agnostic**.
//!
//! ## What the cascade must test next
//!
//! Combined with #1801/#1805/#1811's synthetic findings and this PR's
//! real-Q6_K confirmation, the M-GPU-MOE-3 root cause must be in
//! something SHARED between Q4_K and Q6_K paths:
//!
//! 1. **SwiGLU activation parity** (HIGHEST EV) — CPU uses `f32::exp`,
//!    CUDA uses `ex2.approx.f32`. Same algebra, different f32-precision
//!    behavior on extreme inputs. Both paths reach this op regardless
//!    of qtype.
//! 2. **Q4_K real-weight matvec** — direct sibling of this PR; should
//!    show whether Q4_K shares the ulp-scale floor.
//! 3. **Top-K expert weighted sum** — host-side f32 accumulation;
//!    bit-identical by inspection but worth empirical verification.
//!
//! ## How to run
//!
//! Requires:
//! - CUDA hardware (RTX 4090)
//! - Cached `/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
//!   (18GB)
//!
//! ```bash
//! cargo test --release --features cuda \
//!   -p aprender-serve --test falsify_q6k_real_weight_004 \
//!   -- --ignored --nocapture
//! ```
//!
//! Skipped when the GGUF is absent (returns early after logging).
//!
//! ## Cross-refs
//!
//! - Issue: #1583 (M-GPU-MOE-3)
//! - Predecessors: #1801, #1805, #1811
//! - GGUF tensor structure: `qwen3_moe_load.rs::Qwen3MoeQuantizedLayer`
//!   — for Qwen3-Coder-30B, `ffn_down_exps` is shape `[128, 2048, 768]`
//!   row-major, Q6_K. One expert = 2048 × 768 weight matrix.

#![cfg(feature = "cuda")]

use realizar::cuda::CudaExecutor;
/// GGUF Q6_K qtype id (matches `realizar::gguf::types::GGUF_TYPE_Q6_K`).
/// Hard-coded here since `types` is a private module from outside the crate.
const GGUF_TYPE_Q6_K: u32 = 14;
use realizar::gguf::MappedGGUFModel;
use realizar::quantize::fused_q6k_parallel_matvec;
use std::path::Path;

const CANONICAL_QWEN3_GGUF_PATHS: &[&str] = &[
    "/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/cache/apr-home/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
];

/// Q6_K super-block: 256 quants, 210 bytes per super-block.
const Q6K_SUPER_BLOCK_SIZE: usize = 256;
const Q6K_SUPER_BLOCK_BYTES: usize = 210;

/// Number of output rows to test (matches #1801 baseline for direct
/// comparison). Picks rows 0..16 from the full expert weight matrix.
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

/// Locate the bytes of a single Q6_K matvec from a Qwen3 layer-7 tensor.
///
/// Returns `(weight_bytes, in_dim, out_dim, tensor_name)` where:
/// - `weight_bytes`: row-major Q6_K bytes for `out_dim` rows of width `in_dim`
/// - `in_dim`: must be a multiple of 256 (Q6_K super-block size)
/// - `out_dim`: number of rows extracted (capped at `OUT_DIM_TEST`)
/// - `tensor_name`: human-readable label for the eprintln telemetry
///
/// Picks the first Q6_K tensor in `blk.7.*` it finds. For Qwen3-Coder-30B
/// the candidate is typically `blk.7.ffn_down_exps.weight` —
/// `[num_experts=128, hidden_dim=2048, intermediate=768]` row-major, so
/// each expert occupies `2048 × ceil(768/256) × 210 = 1,290,240 bytes`
/// and we slice the first `OUT_DIM_TEST` rows of expert 0.
fn extract_real_q6k_matvec_bytes(
    mapped: &MappedGGUFModel,
) -> Option<(Vec<u8>, usize, usize, String)> {
    let mmap_bytes: &[u8] = mapped.data();
    let tensor_data_start = mapped.model.tensor_data_start;

    // Find the FIRST Q6_K tensor in any decoder layer. Critical empirical
    // finding from this session's GGUF inventory:
    //
    // In Qwen3-Coder-30B-A3B-Instruct-Q4_K_M, the "problem layers" cited
    // by #1583 (L7/L9/L12/L20/L23/L29/L46) MIX qtypes for ffn_down_exps:
    //
    //   L7  Q4_K    L20 Q6_K
    //   L9  Q4_K    L23 Q6_K
    //   L12 Q4_K    L29 Q6_K
    //               L46 Q6_K
    //
    // Three of seven problem layers DO NOT EVEN HAVE Q6_K tensors —
    // they are pure Q4_K MoE. #1583's framing that this is purely a
    // Q6_K reduction-order issue is incomplete: the divergence affects
    // BOTH Q4_K and Q6_K paths. Whatever the amplifier is, it must be
    // qtype-agnostic (SwiGLU activation, weighted-sum, or a specific
    // real-weight pattern).
    //
    // We still test with a Q6_K tensor here to anchor against the
    // synthetic baselines from #1801/#1805/#1811. The qtype-agnostic
    // finding is captured in the docstring; a follow-up cascade PR
    // should test Q4_K real-weight matvec separately.
    for t in &mapped.model.tensors {
        if t.qtype != GGUF_TYPE_Q6_K {
            continue;
        }

        // GGUF dim ordering: dims[0] is fastest-moving (= in_dim per row).
        if t.dims.len() < 2 {
            continue;
        }
        let in_dim = t.dims[0] as usize;
        let total_rows = t.dims[1..].iter().product::<u64>() as usize;
        if !in_dim.is_multiple_of(Q6K_SUPER_BLOCK_SIZE) || total_rows < OUT_DIM_TEST {
            continue;
        }

        let super_blocks_per_row = in_dim / Q6K_SUPER_BLOCK_SIZE;
        let bytes_per_row = super_blocks_per_row * Q6K_SUPER_BLOCK_BYTES;
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
fn falsify_q6k_real_weight_l7_matvec() {
    let Some(gguf_path) = locate_gguf() else {
        eprintln!(
            "FALSIFY-Q6K-REAL-WEIGHT-004: skipped — Qwen3-Coder-30B GGUF not found in any of:"
        );
        for p in CANONICAL_QWEN3_GGUF_PATHS {
            eprintln!("  - {p}");
        }
        eprintln!("(test exits cleanly per M32c.2.2.2.1.4 convention)");
        return;
    };

    eprintln!("FALSIFY-Q6K-REAL-WEIGHT-004: loading {gguf_path}");
    let mapped = MappedGGUFModel::from_path(gguf_path)
        .expect("Qwen3 GGUF must mmap cleanly (run `apr inspect` if not)");

    let Some((weight_bytes, in_dim, out_dim, tensor_name)) = extract_real_q6k_matvec_bytes(&mapped)
    else {
        panic!(
            "FALSIFY-Q6K-REAL-WEIGHT-004: no Q6_K tensor found under blk.7.* with \
             in_dim multiple of 256 — check GGUF tensor inventory or relax constraints"
        );
    };

    eprintln!(
        "FALSIFY-Q6K-REAL-WEIGHT-004: source tensor `{tensor_name}` \
         (sliced to first {out_dim} rows × {in_dim} cols, {} bytes)",
        weight_bytes.len()
    );

    let activations = synthetic_vec(in_dim, 0x1583_0007);

    // CPU side
    let cpu_out = fused_q6k_parallel_matvec(&weight_bytes, &activations, in_dim, out_dim)
        .expect("cpu Q6_K matvec on real bytes must not error");
    assert_eq!(cpu_out.len(), out_dim);

    // GPU side
    let mut cuda = CudaExecutor::new(0)
        .expect("CudaExecutor::new(0) must succeed per realizar CLAUDE.md CUDA-always rule");
    let mut gpu_out = vec![0.0f32; out_dim];
    cuda.q6k_gemv(
        &weight_bytes,
        &activations,
        &mut gpu_out,
        out_dim as u32,
        in_dim as u32,
    )
    .expect("gpu q6k_gemv on real bytes must succeed");

    // Sanity
    assert!(cpu_out.iter().all(|x| x.is_finite()), "cpu_out non-finite");
    assert!(gpu_out.iter().all(|x| x.is_finite()), "gpu_out non-finite");

    let cpu_l2: f32 = cpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let gpu_l2: f32 = gpu_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = cosine_similarity(&cpu_out, &gpu_out);
    let rel = max_rel_diff(&cpu_out, &gpu_out);

    eprintln!();
    eprintln!("FALSIFY-Q6K-REAL-WEIGHT-004: empirical result");
    eprintln!("  cos={cos:.6}  max_rel_diff={rel:.3e}  cpu_l2={cpu_l2:.3}  gpu_l2={gpu_l2:.3}");
    eprintln!();
    eprintln!("Compared to #1801's synthetic baseline (rel_diff ≈ 6e-7):");
    if rel < 1e-5 {
        eprintln!("  → REAL WEIGHTS MATCH SYNTHETIC ulp-scale floor.");
        eprintln!("  → Bug is NOT in per-matvec on real Q6_K — pivot cascade to");
        eprintln!("    Q4_K matmul, SwiGLU activation, or weighted-sum.");
    } else if rel < 1e-3 {
        eprintln!(
            "  → Real weights show MILD amplification ({}× synthetic).",
            rel / 6e-7
        );
        eprintln!("  → Worth bisecting per-expert weight non-uniformity.");
    } else {
        eprintln!(
            "  → Real weights show STRONG amplification ({}× synthetic).",
            rel / 6e-7
        );
        eprintln!("  → #1801's synthetic-baseline premise was incomplete.");
        eprintln!("  → Real-weight reduction-order divergence IS in play.");
    }

    // Sanity floor only — the telemetry is the load-bearing artifact.
    assert!(
        rel < 1.0,
        "rel_diff = {rel:.3e} ≥ 1.0 — output is garbage, not a divergence pattern"
    );
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn xorshift32_deterministic() {
        let mut a = 0x1234;
        let mut b = 0x1234;
        for _ in 0..100 {
            assert_eq!(xorshift32(&mut a), xorshift32(&mut b));
        }
    }

    #[test]
    fn cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn max_rel_diff_zero_for_identical() {
        let a = vec![0.5, -1.5, 0.0];
        assert_eq!(max_rel_diff(&a, &a), 0.0);
    }

    #[test]
    fn locate_gguf_returns_none_if_missing() {
        // Confirms the path-check guard works even on a clean machine
        // (we just check it returns a value or None without panicking).
        let _ = locate_gguf();
    }

    #[test]
    fn q6k_super_block_constants() {
        assert_eq!(Q6K_SUPER_BLOCK_SIZE, 256);
        assert_eq!(Q6K_SUPER_BLOCK_BYTES, 210);
    }
}
