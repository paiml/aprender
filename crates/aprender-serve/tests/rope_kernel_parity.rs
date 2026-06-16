//! PMAT-792: RoPE GPU kernel CPU↔GPU parity (NORM + NEOX pairing)
//!
//! Falsifiable, tolerance-bearing parity test for the GPU RoPE kernels,
//! filling the gap left by `rope_parity.rs` (which is end-to-end argmax-only
//! and never `assert!`s on divergence).
//!
//! This pins the two distinct RoPE pairing conventions on hardware:
//!   - NORM  (`rope_type == 0`, LLaMA): adjacent pairs `(2*i, 2*i+1)`
//!   - NEOX  (`rope_type == 2`, Qwen/GPT-NeoX): split halves `(i, i+head_dim/2)`
//!
//! The CPU reference below is byte-for-byte the math in
//! `OwnedQuantizedModel::apply_rope` (crates/aprender-serve/src/gguf/inference/rope.rs):
//!   freq = 1 / theta^(2*i / head_dim);  angle = pos * freq
//!   x0' = x0*cos - x1*sin;  x1' = x0*sin + x1*cos
//!
//! A divergence here is the PMAT-216 / #2749 bug class: a GPU kernel that
//! pairs elements differently from CPU silently corrupts every rotated vector.

#![cfg(feature = "cuda")]

/// CPU reference RoPE — mirrors `OwnedQuantizedModel::apply_rope`.
/// `rope_neox = true` selects the NEOX split-half pairing; `false` is NORM.
fn cpu_rope(
    x: &[f32],
    position: u32,
    num_heads: u32,
    head_dim: u32,
    theta: f32,
    rope_neox: bool,
) -> Vec<f32> {
    let head_dim = head_dim as usize;
    let half_dim = head_dim / 2;
    let pos_f32 = position as f32;
    let head_dim_f32 = head_dim as f32;

    // Pre-compute cos/sin for this position (reused across all heads).
    let mut cos_vals = vec![0.0f32; half_dim];
    let mut sin_vals = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        let freq = 1.0f32 / theta.powf(2.0 * i as f32 / head_dim_f32);
        let angle = pos_f32 * freq;
        let (sin_v, cos_v) = angle.sin_cos();
        cos_vals[i] = cos_v;
        sin_vals[i] = sin_v;
    }

    let mut out = x.to_vec();
    for h in 0..num_heads as usize {
        let head_start = h * head_dim;
        if head_start + head_dim > out.len() {
            continue;
        }
        for i in 0..half_dim {
            let (idx0, idx1) = if rope_neox {
                // NEOX: split halves (i, i + half_dim)
                (head_start + i, head_start + i + half_dim)
            } else {
                // NORM: adjacent pairs (2*i, 2*i+1)
                (head_start + 2 * i, head_start + 2 * i + 1)
            };
            let x0 = x[idx0];
            let x1 = x[idx1];
            let cos_v = cos_vals[i];
            let sin_v = sin_vals[i];
            out[idx0] = x0 * cos_v - x1 * sin_v;
            out[idx1] = x0 * sin_v + x1 * cos_v;
        }
    }
    out
}

/// Deterministic pseudo-random-ish input so the test is reproducible.
fn make_input(num_heads: u32, head_dim: u32) -> Vec<f32> {
    let n = (num_heads * head_dim) as usize;
    (0..n)
        .map(|i| ((i as f32 * 0.137).sin() * 1.7) + ((i % 7) as f32 - 3.0) * 0.11)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    (max_diff, max_idx)
}

/// PMAT-792: NORM-style RoPE GPU parity vs CPU `apply_rope` (rope_type 0).
#[test]
#[ignore] // Run with --features cuda -- --ignored on a CUDA host.
fn test_cpu_gpu_rope_norm_parity() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed, cannot run RoPE parity: {e:?}");
            panic!("PMAT-792: CUDA required for RoPE parity test");
        },
    };

    // head_dim=128 (typical), theta=10000 (LLaMA), several positions + heads.
    let head_dim = 128u32;
    let theta = 10000.0f32;
    let tolerance = 1e-3f32;

    for &num_heads in &[1u32, 4, 8] {
        let input = make_input(num_heads, head_dim);
        for &position in &[0u32, 1, 7, 64, 511] {
            let cpu = cpu_rope(&input, position, num_heads, head_dim, theta, false);

            let mut gpu = vec![0.0f32; input.len()];
            executor
                .rope_host(&input, &mut gpu, position, num_heads, head_dim, theta)
                .expect("rope_host (NORM) failed");

            let (max_diff, idx) = max_abs_diff(&cpu, &gpu);
            eprintln!(
                "NORM  heads={num_heads:>2} pos={position:>4}  max_diff={max_diff:.3e} @ {idx}"
            );
            assert!(
                max_diff < tolerance,
                "PMAT-792 FAIL: NORM RoPE GPU diverged {max_diff:.3e} (> {tolerance:.0e}) \
                 from CPU at idx {idx} (heads={num_heads}, pos={position}); CPU={:.6} GPU={:.6}",
                cpu[idx],
                gpu[idx],
            );
        }
    }
}

/// PMAT-792: NEOX-style RoPE GPU parity vs CPU `apply_rope` (rope_type 2).
#[test]
#[ignore] // Run with --features cuda -- --ignored on a CUDA host.
fn test_cpu_gpu_rope_neox_parity() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("CUDA init failed, cannot run RoPE parity: {e:?}");
            panic!("PMAT-792: CUDA required for RoPE parity test");
        },
    };

    let head_dim = 128u32;
    let theta = 1_000_000.0f32; // Qwen-style high theta stresses trig precision.
    let tolerance = 1e-3f32;

    for &num_heads in &[1u32, 4, 8] {
        let input = make_input(num_heads, head_dim);
        for &position in &[0u32, 1, 7, 64, 511] {
            let cpu = cpu_rope(&input, position, num_heads, head_dim, theta, true);

            let mut gpu = vec![0.0f32; input.len()];
            executor
                .rope_neox_host(&input, &mut gpu, position, num_heads, head_dim, theta)
                .expect("rope_neox_host (NEOX) failed");

            let (max_diff, idx) = max_abs_diff(&cpu, &gpu);
            eprintln!(
                "NEOX  heads={num_heads:>2} pos={position:>4}  max_diff={max_diff:.3e} @ {idx}"
            );
            assert!(
                max_diff < tolerance,
                "PMAT-792 FAIL: NEOX RoPE GPU diverged {max_diff:.3e} (> {tolerance:.0e}) \
                 from CPU at idx {idx} (heads={num_heads}, pos={position}); CPU={:.6} GPU={:.6}",
                cpu[idx],
                gpu[idx],
            );
        }
    }
}

/// Cross-check: NORM and NEOX produce DIFFERENT results for the same input
/// (guards against the two kernels collapsing to the same pairing — which
/// would mask a pairing bug). This runs CPU-only, no GPU required.
#[test]
fn test_norm_vs_neox_differ() {
    let head_dim = 128u32;
    let num_heads = 4u32;
    let theta = 10000.0f32;
    let input = make_input(num_heads, head_dim);
    let position = 7u32;

    let norm = cpu_rope(&input, position, num_heads, head_dim, theta, false);
    let neox = cpu_rope(&input, position, num_heads, head_dim, theta, true);

    let (max_diff, _) = max_abs_diff(&norm, &neox);
    assert!(
        max_diff > 1e-2,
        "NORM and NEOX pairings should differ substantially; max_diff={max_diff:.3e}"
    );
}
