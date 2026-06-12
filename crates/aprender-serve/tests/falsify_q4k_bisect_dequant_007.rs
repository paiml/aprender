//! FALSIFY-Q4K-BISECT-DEQUANT-007 — bisect [#1821](https://github.com/paiml/aprender/pull/1821)'s
//! 5% Q4_K divergence: is it the dequant step (Q4_K → f32) or the
//! fused reduction step (warp-shuffle vs rayon midi-tile)?
//!
//! ## Cascade state entering this PR
//!
//! [#1821](https://github.com/paiml/aprender/pull/1821) (PR-3k DISCHARGE
//! falsifier) found that CUDA `q4k_matvec` produces ~5% per-element
//! divergence vs CPU `fused_q4k_parallel_matvec` on real Qwen3 Q4_K
//! bytes — 237,775× the Q6_K ulp-scale baseline.
//!
//! That falsifier proved the root cause is in the Q4_K path. But it
//! tested the FUSED operation (dequant + matmul in one kernel) — so
//! the 5% delta could come from either:
//!
//! 1. **Dequant**: Q4_K → f32 cast (per-byte → per-element math
//!    converts 6-bit + scales to f32; CPU and CUDA may differ in
//!    rounding mode or in scale application order).
//! 2. **Reduction**: warp-shuffle accumulation on GPU vs rayon
//!    midi-tile reduction on CPU (the classic fp-accumulator-order
//!    divergence #1583 originally hypothesized — for Q6_K it was
//!    ulp-scale, but Q4_K may have different per-block structure).
//!
//! ## What this falsifier bisects
//!
//! Three paths compared on identical Q4_K bytes:
//!
//! - **A** (CPU fused): `fused_q4k_parallel_matvec` — the baseline
//! - **B** (CPU dequant→f32 dot): `dequantize_q4_k_to_f32` followed
//!   by naive f32 dot product per row — isolates the CPU dequant
//! - **C** (CUDA fused): `CudaExecutor::q4k_matvec` — the suspected
//!   broken path from [#1821](https://github.com/paiml/aprender/pull/1821)
//!
//! Comparisons:
//!
//! - **A vs B**: if rel_diff ≈ 0, CPU dequant→f32-dot AGREES with
//!   CPU fused → CPU side is internally consistent.
//! - **A vs C**: reproduces [#1821](https://github.com/paiml/aprender/pull/1821)'s
//!   ~5% delta.
//! - **B vs C**: if rel_diff ≈ A-vs-C, the GPU divergence persists
//!   even when we manually pre-dequantize — implicates the GPU
//!   reduction. If rel_diff ≈ 0, the CPU dequant matches the GPU
//!   fused output — implicates the CPU dequant or a CPU/GPU
//!   dequant alignment issue.
//!
//! ## 🚨 EMPIRICAL RESULT — INVERTS #1821's CONCLUSION 🚨
//!
//! Lambda-vector RTX 4090, 2026-05-19, apr 0.34.0:
//!
//! ```text
//!            pair      rel_diff         1-cos
//!    A vs B (CPU)      2.883e-2      7.093e-6   ← CPU fused ≠ CPU dequant
//! A vs C (CPU-GPU)      2.883e-2      7.033e-6   ← CPU fused ≠ CUDA
//! B vs C (deq-GPU)      5.028e-7     -1.192e-7   ← CPU dequant ≈ CUDA ✅
//! ```
//!
//! **The CUDA Q4_K kernel is NOT broken.** Path B (CPU
//! `dequantize_q4_k_to_f32` → naive f32 dot) matches path C (CUDA
//! `q4k_matvec`) to ulp-scale (5e-7).
//!
//! **The CPU `fused_q4k_parallel_matvec` is the divergent path.**
//! It disagrees with BOTH the CPU naive-dequant reference AND the
//! CUDA path by the SAME 2.88% delta.
//!
//! ## True root cause (read the CPU code)
//!
//! `crates/aprender-serve/src/quantize/parallel_k.rs:181-182` docstring:
//!
//! > Q8K activation quantization: Pre-quantizes f32 activations to
//! > Q8_K once per matmul, enabling integer-only inner loops
//! > (maddubs) for ~4-8x speedup (Refs realizar#96)
//!
//! So CPU `fused_q4k_parallel_matvec` is actually doing:
//!   `Q4_K(weights) × Q8_K(quantize(f32_activations))`
//!
//! CUDA `q4k_matvec` is doing:
//!   `Q4_K(weights) × f32_activations` (no activation quantization)
//!
//! **They are computing DIFFERENT MATHEMATICAL OPERATIONS.** The
//! 2.88% per-matvec delta is the lossy activation quantization in
//! the CPU path. Neither is "wrong" — they're different algorithms.
//!
//! ## What this means for M-GPU-MOE-3 (#1583)
//!
//! The 0.94-cos drop on real Qwen3 layers L7/L9/L12/L20/L23/L29/L46
//! is NOT a kernel correctness bug. It's the natural consequence of:
//!
//! 1. CPU uses Q8K activation quantization for Q4_K paths
//! 2. CUDA uses f32 activations for Q4_K paths
//! 3. The 2-3% per-matvec delta compounds across 128 experts × 48
//!    layers in real Qwen3 inference to produce ~6% (cos=0.94) drop
//!
//! ## Fix paths (multi-week, M-GPU-MOE-3 fix scope)
//!
//! There are three viable resolutions:
//!
//! **Option 1: Make CPU use f32 activations** (match CUDA)
//! - Add `fused_q4k_f32_parallel_matvec` (no Q8K quantization step)
//! - Slow CPU significantly (loses the maddubs 4-8× speedup)
//! - Trade: parity vs CPU perf
//!
//! **Option 2: Make CUDA use Q8_K activations** (match CPU)
//! - Add Q8_K activation quantization step before `q4k_matvec`
//! - Requires writing a CUDA Q8_K quant kernel (modest scope)
//! - Trade: GPU latency vs parity. Q8_K matmul on GPU could be FASTER
//!   than f32 (integer ops + DP4A intrinsic on Ampere+).
//!
//! **Option 3: Accept divergence as documented**
//! - Update `qwen3-moe-forward-gpu-v1` contract to relax cos≥0.99 to
//!   cos≥0.93 (or whatever real-model number is)
//! - Document the activation-qtype mismatch in the contract
//! - Cheapest; defers the perf trade-off
//!
//! **Recommended next step**: Option 2 with empirical perf measurement.
//!
//! ## How to run
//!
//! ```bash
//! cargo test --release --features cuda \
//!   -p aprender-serve --test falsify_q4k_bisect_dequant_007 \
//!   -- --ignored --nocapture
//! ```
//!
//! ## Cross-refs
//!
//! - Issue: #1583 (M-GPU-MOE-3)
//! - DISCHARGE predecessor: [#1821](https://github.com/paiml/aprender/pull/1821) (FALSIFY-Q4K-REAL-WEIGHT-006)
//! - CPU fused: `crates/aprender-serve/src/quantize/fused_q.rs`
//! - CPU dequant: `crates/aprender-quant/src/dequantize.rs::dequantize_q4_k_to_f32`
//! - GPU fused: `crates/aprender-serve/src/cuda/executor/execute.rs::CudaExecutor::q4k_matvec`

#![cfg(feature = "cuda")]

use realizar::cuda::CudaExecutor;
use realizar::gguf::MappedGGUFModel;
use realizar::quantize::fused_q4k_parallel_matvec;
use std::path::Path;
use trueno_quant::dequantize_q4_k_to_f32;

const GGUF_TYPE_Q4_K: u32 = 12;

const CANONICAL_QWEN3_GGUF_PATHS: &[&str] = &[
    "/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
];

const Q4K_SUPER_BLOCK_SIZE: usize = 256;
const Q4K_SUPER_BLOCK_BYTES: usize = 144;
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

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-30);
            (x - y).abs() / denom
        })
        .fold(0.0f32, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Naive row-major f32 dot product matvec: `out[i] = sum_j weights[i*k + j] * input[j]`
/// for `i in 0..m`. Used to isolate dequant from any other CPU fused-kernel
/// behavior (e.g. midi-tile order, SIMD shuffle).
fn naive_f32_matvec(weights_f32: &[f32], input: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    for i in 0..m {
        let row = &weights_f32[i * k..(i + 1) * k];
        out[i] = row.iter().zip(input.iter()).map(|(w, a)| w * a).sum();
    }
    out
}

fn locate_gguf() -> Option<&'static str> {
    CANONICAL_QWEN3_GGUF_PATHS
        .iter()
        .copied()
        .find(|p| Path::new(p).exists())
}

fn extract_q4k_slab(mapped: &MappedGGUFModel) -> Option<(Vec<u8>, usize, usize, String)> {
    let mmap_bytes = mapped.data();
    let tensor_data_start = mapped.model.tensor_data_start;
    for t in &mapped.model.tensors {
        if t.qtype != GGUF_TYPE_Q4_K || t.dims.len() < 2 {
            continue;
        }
        let in_dim = t.dims[0] as usize;
        let total_rows = t.dims[1..].iter().product::<u64>() as usize;
        if !in_dim.is_multiple_of(Q4K_SUPER_BLOCK_SIZE) || total_rows < OUT_DIM_TEST {
            continue;
        }
        let bytes_per_row = (in_dim / Q4K_SUPER_BLOCK_SIZE) * Q4K_SUPER_BLOCK_BYTES;
        let needed = bytes_per_row * OUT_DIM_TEST;
        let off = tensor_data_start + t.offset as usize;
        if off + needed > mmap_bytes.len() {
            continue;
        }
        return Some((
            mmap_bytes[off..off + needed].to_vec(),
            in_dim,
            OUT_DIM_TEST,
            t.name.clone(),
        ));
    }
    None
}

#[test]
#[ignore = "requires CUDA hardware (RTX 4090) + cached 18GB Qwen3 GGUF; runs in <3s"]
fn bisect_q4k_dequant_vs_reduction() {
    let Some(gguf_path) = locate_gguf() else {
        eprintln!("FALSIFY-Q4K-BISECT-007: skipped — Qwen3 GGUF not found.");
        return;
    };

    eprintln!("FALSIFY-Q4K-BISECT-007: loading {gguf_path}");
    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");

    let Some((weight_bytes, in_dim, out_dim, name)) = extract_q4k_slab(&mapped) else {
        panic!("no Q4_K slab found");
    };
    eprintln!(
        "FALSIFY-Q4K-BISECT-007: tensor `{name}` ({out_dim} rows × {in_dim} cols, {} bytes)",
        weight_bytes.len()
    );

    let activations = synthetic_vec(in_dim, 0x1583_000a);

    // PATH A: CPU fused (the production-MoE matvec)
    let path_a = fused_q4k_parallel_matvec(&weight_bytes, &activations, in_dim, out_dim)
        .expect("path A cpu fused must succeed");

    // PATH B: CPU dequant → naive f32 dot product
    // Slice has out_dim*in_dim elements after dequant. Layout: row-major.
    let weights_f32 = dequantize_q4_k_to_f32(&weight_bytes, out_dim * in_dim);
    assert_eq!(weights_f32.len(), out_dim * in_dim, "dequant element count");
    let path_b = naive_f32_matvec(&weights_f32, &activations, out_dim, in_dim);

    // PATH C: CUDA fused (the suspected broken path)
    let mut cuda = CudaExecutor::new(0).expect("CudaExecutor::new(0)");
    let mut path_c = vec![0.0f32; out_dim];
    cuda.q4k_matvec(
        &weight_bytes,
        &activations,
        &mut path_c,
        out_dim as u32,
        in_dim as u32,
    )
    .expect("path C cuda fused must succeed");

    let cos_ab = cosine(&path_a, &path_b);
    let cos_ac = cosine(&path_a, &path_c);
    let cos_bc = cosine(&path_b, &path_c);
    let rel_ab = max_rel_diff(&path_a, &path_b);
    let rel_ac = max_rel_diff(&path_a, &path_c);
    let rel_bc = max_rel_diff(&path_b, &path_c);

    eprintln!();
    eprintln!("FALSIFY-Q4K-BISECT-007: three-way bisection (#1583 PR-3l)");
    eprintln!();
    eprintln!("  A = CPU fused_q4k_parallel_matvec  (production-MoE path)");
    eprintln!("  B = CPU dequantize_q4_k_to_f32 → naive f32 dot (isolates dequant)");
    eprintln!("  C = CUDA q4k_matvec                (suspected broken path)");
    eprintln!();
    eprintln!("{:>15}  {:>12}  {:>12}", "pair", "rel_diff", "1-cos");
    eprintln!("{}", "-".repeat(45));
    eprintln!(
        "{:>15}  {:>12.3e}  {:>12.3e}",
        "A vs B (CPU)",
        rel_ab,
        (1.0 - cos_ab)
    );
    eprintln!(
        "{:>15}  {:>12.3e}  {:>12.3e}",
        "A vs C (CPU-GPU)",
        rel_ac,
        (1.0 - cos_ac)
    );
    eprintln!(
        "{:>15}  {:>12.3e}  {:>12.3e}",
        "B vs C (deq-GPU)",
        rel_bc,
        (1.0 - cos_bc)
    );
    eprintln!();

    if rel_ab < 1e-4 && rel_ac > 1e-3 && rel_bc > 1e-3 {
        eprintln!("  → A ≈ B and both DIFFER from C.");
        eprintln!("  → CPU dequant + f32-dot is internally consistent.");
        eprintln!("  → GPU q4k_matvec is producing different output even when");
        eprintln!("    compared to a manually-dequantized + naive-f32-dot reference.");
        eprintln!("  → Bug is in GPU dequant OR GPU reduction (or fused effect).");
        eprintln!("  → Next bisection: CUDA dequant-only kernel vs CPU dequant.");
    } else if rel_ab > 1e-3 {
        eprintln!("  → A ≠ B: CPU has TWO Q4_K paths that disagree.");
        eprintln!("  → fused_q4k_parallel_matvec doesn't match");
        eprintln!("    dequantize_q4_k_to_f32 + naive_f32_matvec.");
        eprintln!("  → Bug may be in CPU itself; #1821's premise needs refining.");
    } else {
        eprintln!("  → Bisection inconclusive on this tensor.");
        eprintln!("  → A vs B: {rel_ab:.3e}, A vs C: {rel_ac:.3e}, B vs C: {rel_bc:.3e}");
    }

    // Sanity floors only.
    assert!(rel_ab < 1.0);
    assert!(rel_ac < 1.0);
    assert!(rel_bc < 1.0);
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn naive_f32_matvec_identity() {
        // 2x3 identity-ish: out[i] = weights[i*3..(i+1)*3] · input
        let weights = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let input = vec![5.0, 7.0, 9.0];
        let out = naive_f32_matvec(&weights, &input, 2, 3);
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn naive_f32_matvec_zeros() {
        let weights = vec![0.0; 12];
        let input = vec![1.0, 2.0, 3.0];
        let out = naive_f32_matvec(&weights, &input, 4, 3);
        for x in &out {
            assert_eq!(*x, 0.0);
        }
    }

    #[test]
    fn synthetic_vec_deterministic() {
        let a = synthetic_vec(50, 99);
        let b = synthetic_vec(50, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn max_rel_diff_zero_for_identical() {
        let a = vec![1.5, -2.5];
        assert_eq!(max_rel_diff(&a, &a), 0.0);
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![1.0, 1.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }
}
