//! PMAT-793: Q5_K / Q6_K GPU GEMM (matvec) per-ROW value parity.
//!
//! Falsifiable, value-asserting parity test for the `q5k_gemm_ggml` /
//! `q6k_gemm_ggml` PTX kernels driven through `CudaExecutor::{q5k,q6k}_matvec`.
//!
//! ## KNOWN-BROKEN (verified on NVIDIA GB10, sm_121) -- `#[ignore]`d.
//!
//! These two tests CURRENTLY FAIL: both `q{5,6}k_matvec` produce a non-zero,
//! correct value ONLY in output row 0; every other output row is left 0.0,
//! regardless of `m`. The full output for m=8 with byte-identical per-row
//! weights is `[v, 0, 0, 0, 0, 0, 0, 0]` instead of `[v, v, v, v, v, v, v, v]`.
//! This is the PMAT-792 RoPE-NORM dimension-collapse CLASS, but the root cause
//! is deeper than the host launch grid, so a launch-config change alone does
//! NOT fix it (both the pre-fix `LaunchConfig::linear(m, 256)` and a corrected
//! per-tile `grid_2d(1, ceil(m/32), 1024, 1)` produce the SAME row-0-only
//! result on hardware):
//!
//!   1. LAUNCH: `linear(m, 256)` gives `grid_y == 1`, so `blockIdx.y == 0`
//!      (`out_row == 0`) -- needs `grid_y == ceil(m/32)`.
//!   2. KERNEL CONTRACT (the load-bearing defect): `q{5,6}k_gemm_ggml` is a
//!      GEMM that computes `C[row,col] = A[row,:] . B[col,:]` -- activations
//!      `A` indexed by `clamped_row` (q6k/gemm.rs:211), weights `B` by
//!      `clamped_col` (q6k/gemm.rs:127). The `q6k_matvec` caller, however,
//!      passes the single shared input vector as `A` and the per-output-row
//!      weights as `B`, with `n == 1`. So for output row i>0 the kernel reads
//!      `A[i*k ..]` (OUT OF BOUNDS of the one-row input) and `B[col==0]` (the
//!      wrong, first weight row). Only `C[0,0]` is fed valid `A[0,:]`/`B[0,:]`.
//!   3. REDUCTION MODEL: the kernel reduces over 16 warp lanes (`lane=tid%16`,
//!      `shfl_down` 8/4/2/1) to cooperatively sum ONE output element over k,
//!      which contradicts the one-output-element-per-thread row/col indexing.
//!
//! A correct fix is a kernel/caller redesign (or routing matvec through the
//! proven `q6k_gemv_indexed` GEMV path), not a launch-config tweak -- out of
//! scope for the launch-grid audit. These call sites are TEST-ONLY (no
//! production decode/prefill path calls `q{5,6}k_matvec`; real Q5K/Q6K
//! inference uses the GEMV kernels), so the defect ships no active regression.
//!
//! ## Why the prior tests missed it
//! `tests_cov023_rmsnorm.rs::test_cov024_q{5,6}k_matvec_dimension_basic` use
//! ALL-ZERO weights and assert only `result.is_ok()`. With zero weights every
//! row legitimately produces 0.0, so a stale (0.0) row is indistinguishable
//! from a correct one. This test uses NON-ZERO weights that are byte-identical
//! across all `m` rows, so every row MUST produce the same non-zero value; the
//! row-0-only collapse is caught. Run on a CUDA host with:
//!   `cargo test -p aprender-serve --features cuda --test
//!    q5k_q6k_gemm_row_parity -- --ignored`

#![cfg(feature = "cuda")]

/// Build a single valid Q6_K super-block (210 bytes, 256 values) with a
/// non-zero, deterministic dequantized result. Layout (GGML Q6_K):
///   [0..128)   ql    (low 4 bits, 2 vals/byte)
///   [128..192) qh    (high 2 bits, 4 vals/byte)
///   [192..208) scales(16 x int8)
///   [208..210) d     (f16 super-block scale)
fn q6k_superblock_nonzero() -> Vec<u8> {
    let mut b = vec![0u8; 210];
    // ql: every nibble = 0x5 -> low bits non-zero across all 256 values.
    for x in b.iter_mut().take(128) {
        *x = 0x55;
    }
    // qh: high bits = 0b01 pattern -> 0x55 packs four 2-bit lanes of 0b01.
    for x in b.iter_mut().take(192).skip(128) {
        *x = 0x55;
    }
    // scales: all = 8 (positive, non-zero) for every 16-value sub-block.
    for x in b.iter_mut().take(208).skip(192) {
        *x = 8;
    }
    // d = 1.0 in f16.
    b[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    b
}

/// Build a single valid Q5_K super-block (176 bytes, 256 values), non-zero.
/// Layout (GGML Q5_K):
///   [0..2)    d      (f16 super-block scale)
///   [2..4)    dmin   (f16 super-block min)
///   [4..16)   scales (6-bit packed)
///   [16..48)  qh     (high bit, 1 bit/val)
///   [48..176) qs     (low 4 bits, 2 vals/byte)
fn q5k_superblock_nonzero() -> Vec<u8> {
    let mut b = vec![0u8; 176];
    b[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
    b[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes()); // dmin
    for x in b.iter_mut().take(16).skip(4) {
        *x = 0x21; // packed 6-bit scales, non-zero
    }
    for x in b.iter_mut().take(48).skip(16) {
        *x = 0x00; // qh = 0 (keep values in low 4-bit range)
    }
    for x in b.iter_mut().take(176).skip(48) {
        *x = 0x55; // qs low nibbles non-zero
    }
    b
}

fn max_abs_dev_from_row0(out: &[f32]) -> (f32, usize) {
    let r0 = out[0];
    let mut md = 0.0f32;
    let mut mi = 0;
    for (i, &v) in out.iter().enumerate() {
        let d = (v - r0).abs();
        if d > md {
            md = d;
            mi = i;
        }
    }
    (md, mi)
}

/// PMAT-793: every Q6_K output row gets identical non-zero weights, so every
/// row must produce the identical non-zero value. Catches the row-collapse.
#[test]
#[ignore] // Run with --features cuda -- --ignored on a CUDA host.
fn test_q6k_matvec_all_rows_written() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => panic!("PMAT-793: CUDA required for Q6_K row-parity test: {e:?}"),
    };

    let k = 256u32; // one super-block per row
    let sb = q6k_superblock_nonzero();
    let input = vec![0.1f32; k as usize];

    // Test several m values straddling the old collapse boundary (8 rows).
    for &m in &[8u32, 9, 16, 32, 64, 127] {
        let mut weights = Vec::with_capacity(m as usize * sb.len());
        for _ in 0..m {
            weights.extend_from_slice(&sb);
        }
        let mut out = vec![f32::NAN; m as usize];
        executor
            .q6k_matvec(&weights, &input, &mut out, m, k)
            .expect("q6k_matvec failed");

        // Row 0 must be the genuine (non-zero) dequant-matvec result.
        assert!(
            out[0].is_finite() && out[0].abs() > 1e-6,
            "PMAT-793 (m={m}): row 0 should be non-zero, got {}",
            out[0]
        );
        // Every row used identical weights+input -> identical output.
        let (dev, idx) = max_abs_dev_from_row0(&out);
        assert!(
            dev < 1e-4,
            "PMAT-793 (m={m}): row {idx}={} deviates from row0={} by {dev} \
             (stale-row collapse: rows >=8 were left 0.0). full={:?}",
            out[idx],
            out[0],
            out
        );
    }
}

/// PMAT-793: same falsifier for the Q5_K GEMM kernel.
#[test]
#[ignore]
fn test_q5k_matvec_all_rows_written() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(e) => panic!("PMAT-793: CUDA required for Q5_K row-parity test: {e:?}"),
    };

    let k = 256u32;
    let sb = q5k_superblock_nonzero();
    let input = vec![0.1f32; k as usize];

    for &m in &[8u32, 9, 16, 32, 64, 127] {
        let mut weights = Vec::with_capacity(m as usize * sb.len());
        for _ in 0..m {
            weights.extend_from_slice(&sb);
        }
        let mut out = vec![f32::NAN; m as usize];
        executor
            .q5k_matvec(&weights, &input, &mut out, m, k)
            .expect("q5k_matvec failed");

        assert!(
            out[0].is_finite() && out[0].abs() > 1e-6,
            "PMAT-793 (m={m}): Q5_K row 0 should be non-zero, got {}",
            out[0]
        );
        let (dev, idx) = max_abs_dev_from_row0(&out);
        assert!(
            dev < 1e-4,
            "PMAT-793 (m={m}): Q5_K row {idx}={} deviates from row0={} by {dev}. full={:?}",
            out[idx],
            out[0],
            out
        );
    }
}
