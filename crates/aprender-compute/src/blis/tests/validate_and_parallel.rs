//! Coverage tests for `validate_gemm_dims` (compute.rs:143) and
//! `gemm_blis_parallel` (parallel.rs:67).
//!
//! Targets: 15 uncovered lines in validate_gemm_dims, 19 in gemm_blis_parallel.

use super::super::*;

// ========================================================================
// validate_gemm_dims — exercised via gemm_blis error paths
// ========================================================================

/// A-size mismatch: a.len() != m * k
#[test]
fn test_validate_gemm_dims_a_mismatch() {
    let a = vec![1.0f32; 10]; // expected m*k = 3*4 = 12
    let b = vec![1.0f32; 20]; // k*n = 4*5 = 20
    let mut c = vec![0.0f32; 15]; // m*n = 3*5 = 15

    let result = gemm_blis(3, 5, 4, &a, &b, &mut c, None);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("A size mismatch"), "Got: {}", msg);
}

/// B-size mismatch: b.len() != k * n
#[test]
fn test_validate_gemm_dims_b_mismatch() {
    let a = vec![1.0f32; 12]; // m*k = 3*4 = 12 OK
    let b = vec![1.0f32; 19]; // expected k*n = 4*5 = 20
    let mut c = vec![0.0f32; 15]; // m*n = 3*5 = 15

    let result = gemm_blis(3, 5, 4, &a, &b, &mut c, None);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("B size mismatch"), "Got: {}", msg);
}

/// C-size mismatch: c.len() != m * n
#[test]
fn test_validate_gemm_dims_c_mismatch() {
    let a = vec![1.0f32; 12]; // m*k = 3*4 = 12 OK
    let b = vec![1.0f32; 20]; // k*n = 4*5 = 20 OK
    let mut c = vec![0.0f32; 14]; // expected m*n = 3*5 = 15

    let result = gemm_blis(3, 5, 4, &a, &b, &mut c, None);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("C size mismatch"), "Got: {}", msg);
}

/// All dimensions correct — should succeed.
#[test]
fn test_validate_gemm_dims_all_correct() {
    let m = 4;
    let n = 3;
    let k = 5;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    let result = gemm_blis(m, n, k, &a, &b, &mut c, None);
    assert!(result.is_ok());
}

/// Zero-dimension m: should return Ok immediately (early exit).
#[test]
fn test_gemm_blis_zero_m() {
    let a: Vec<f32> = vec![];
    let b = vec![1.0f32; 20]; // k*n = 4*5
    let mut c: Vec<f32> = vec![];

    let result = gemm_blis(0, 5, 4, &a, &b, &mut c, None);
    assert!(result.is_ok());
}

/// Zero-dimension n: should return Ok immediately.
#[test]
fn test_gemm_blis_zero_n() {
    let a = vec![1.0f32; 12]; // m*k = 3*4
    let b: Vec<f32> = vec![];
    let mut c: Vec<f32> = vec![];

    let result = gemm_blis(3, 0, 4, &a, &b, &mut c, None);
    assert!(result.is_ok());
}

/// Zero-dimension k: should return Ok immediately.
#[test]
fn test_gemm_blis_zero_k() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let mut c = vec![0.0f32; 15]; // m*n = 3*5

    let result = gemm_blis(3, 5, 0, &a, &b, &mut c, None);
    assert!(result.is_ok());
}

/// Small matrix (m*n*k < 4096) falls through to reference GEMM.
#[test]
fn test_gemm_blis_falls_to_reference() {
    // 4*4*4 = 64 < 4096 => reference path
    let a =
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let b = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut c = vec![0.0f32; 16];

    gemm_blis(4, 4, 4, &a, &b, &mut c, None).unwrap();
    // Multiplying by identity should give A back
    for i in 0..16 {
        assert!((c[i] - a[i]).abs() < 1e-5, "c[{}] = {}, expected {}", i, c[i], a[i]);
    }
}

// ========================================================================
// gemm_blis_parallel — exercises dimension mismatch and large-matrix path
// ========================================================================

/// Parallel GEMM: a-size mismatch.
#[test]
fn test_gemm_blis_parallel_a_mismatch() {
    let a = vec![1.0f32; 5]; // wrong: expected m*k = 2*3 = 6
    let b = vec![1.0f32; 12]; // k*n = 3*4 = 12
    let mut c = vec![0.0f32; 8]; // m*n = 2*4 = 8

    let result = gemm_blis_parallel(2, 4, 3, &a, &b, &mut c);
    assert!(result.is_err());
}

/// Parallel GEMM: b-size mismatch.
#[test]
fn test_gemm_blis_parallel_b_mismatch() {
    let a = vec![1.0f32; 6]; // m*k = 2*3 = 6 OK
    let b = vec![1.0f32; 11]; // wrong: expected k*n = 3*4 = 12
    let mut c = vec![0.0f32; 8]; // m*n = 2*4 = 8

    let result = gemm_blis_parallel(2, 4, 3, &a, &b, &mut c);
    assert!(result.is_err());
}

/// Parallel GEMM: c-size mismatch.
#[test]
fn test_gemm_blis_parallel_c_mismatch() {
    let a = vec![1.0f32; 6]; // m*k = 2*3 = 6 OK
    let b = vec![1.0f32; 12]; // k*n = 3*4 = 12 OK
    let mut c = vec![0.0f32; 7]; // wrong: expected m*n = 2*4 = 8

    let result = gemm_blis_parallel(2, 4, 3, &a, &b, &mut c);
    assert!(result.is_err());
}

/// Parallel GEMM: small matrix falls through to single-threaded gemm_blis.
#[test]
fn test_gemm_blis_parallel_small_falls_to_sequential() {
    // m*n*k = 8*8*8 = 512 < 1_000_000 => sequential path
    let n = 8;
    let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i + 3) % 5) as f32).collect();
    let mut c_par = vec![0.0f32; n * n];
    let mut c_ref = vec![0.0f32; n * n];

    gemm_blis_parallel(n, n, n, &a, &b, &mut c_par).unwrap();
    gemm_reference(n, n, n, &a, &b, &mut c_ref).unwrap();

    for i in 0..n * n {
        assert!(
            (c_par[i] - c_ref[i]).abs() < 1e-3,
            "Mismatch at {}: par={}, ref={}",
            i,
            c_par[i],
            c_ref[i]
        );
    }
}

/// Parallel GEMM: large matrix exercises the parallel Rayon path.
/// m*n*k = 128*128*128 = 2_097_152 > 1_000_000 => parallel path (when `parallel` feature is on).
#[test]
fn test_gemm_blis_parallel_large_matrix() {
    let n = 128;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let mut c_par = vec![0.0f32; n * n];
    let mut c_ref = vec![0.0f32; n * n];

    gemm_blis_parallel(n, n, n, &a, &b, &mut c_par).unwrap();
    gemm_reference(n, n, n, &a, &b, &mut c_ref).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..n * n {
        let diff = (c_par[i] - c_ref[i]).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(max_diff < 1e-1, "Max diff: {}", max_diff);
}

/// Parallel GEMM: rectangular matrix (M > N).
#[test]
fn test_gemm_blis_parallel_rectangular_tall() {
    let m = 200;
    let n = 50;
    let k = 60;
    // m*n*k = 600_000 < 1M, sequential path; but tests validation logic
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; m * n];

    let result = gemm_blis_parallel(m, n, k, &a, &b, &mut c);
    assert!(result.is_ok());
    // Verify at least one non-zero output
    assert!(c.iter().any(|&v| v != 0.0));
}

// ========================================================================
// Shared-B parallel GEMM tests
// ========================================================================

/// FALSIFY-SHARED-B-001: shared-B parallel matches reference for 256×256.
#[cfg(feature = "parallel")]
#[test]
fn test_gemm_parallel_shared_b_256() {
    let n = 256;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let mut c_shared = vec![0.0f32; n * n];
    let mut c_ref = vec![0.0f32; n * n];

    super::super::parallel::gemm_blis_parallel_shared_b(n, n, n, &a, &b, &mut c_shared).unwrap();
    gemm_reference(n, n, n, &a, &b, &mut c_ref).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..n * n {
        let diff = (c_shared[i] - c_ref[i]).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(max_diff < 1e-1, "FALSIFY-SHARED-B-001: max diff {max_diff} >= 1e-1 at 256");
}

/// FALSIFY-SHARED-B-002: shared-B parallel matches reference for non-aligned 100×96.
#[cfg(feature = "parallel")]
#[test]
fn test_gemm_parallel_shared_b_non_aligned() {
    let m = 100;
    let n = 96;
    let k = 128;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let mut c_shared = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];

    super::super::parallel::gemm_blis_parallel_shared_b(m, n, k, &a, &b, &mut c_shared).unwrap();
    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        let diff = (c_shared[i] - c_ref[i]).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(max_diff < 1e-1, "FALSIFY-SHARED-B-002: max diff {max_diff} >= 1e-1");
}

/// Falsifier for [`shared_b_has_microkernel`]: on a target with no 8x32 microkernel the
/// shared-B entry point must FALL BACK and still produce the reference answer, rather than
/// entering a hot loop whose full-tile branch is an empty `#[cfg]` block and returning an
/// all-zero C.
///
/// The bug this guards reported "max diff 39.2" against a 1e-1 tolerance, which read as an
/// accuracy problem; the output was in fact 0 of 65536 cells written, and the 39.2 was
/// simply max(reference). So this asserts the thing that actually distinguishes the two:
/// C must not come back untouched. It is deliberately NOT gated on `target_arch`, so it is
/// meaningful on x86_64 (where it proves the microkernel path writes) and on aarch64 (where
/// it proves the fallback writes).
/// Refs `.planning/debug/gemm-shared-b-parallel.md` (FALSIFY-SHARED-B-001).
#[cfg(feature = "parallel")]
#[test]
fn shared_b_falls_back_without_the_8x32_microkernel() {
    use super::super::parallel::shared_b_has_microkernel;

    // 256^3 = 16.7M flops, above the 8M single-thread bailout, and mr/nr-aligned — the
    // exact shape that was returning all zeros.
    let n = 256;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; n * n];
    super::super::parallel::gemm_blis_parallel_shared_b(n, n, n, &a, &b, &mut c).unwrap();

    let nonzero = c.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero > 0,
        "shared-B wrote 0 of {} cells: the full-tile branch computed nothing and C was \
         returned at its zero initial value (has_microkernel = {})",
        c.len(),
        shared_b_has_microkernel()
    );

    // The predicate must agree with the architecture it describes: there is no 8x32
    // microkernel anywhere but x86_64, so a `true` off x86_64 means the guard has been
    // widened without a kernel behind it.
    assert!(
        !shared_b_has_microkernel() || cfg!(target_arch = "x86_64"),
        "shared_b_has_microkernel() must never be true off x86_64"
    );
}

/// Falsifier for the 2026-06-13 thin-NN-GEMM serial routing fix: the dispatch
/// MUST run thin NN-scale GEMMs serially (rayon was measured 2.2x slower) while
/// still parallelizing square sub-64M and large GEMMs. Guards
/// contracts/gemm-parallel-dispatch-v1.yaml (obligation THIN-SERIAL).
#[cfg(feature = "parallel")]
#[test]
fn nn_thin_gemm_prefers_serial() {
    use super::super::parallel::gemm_should_run_serial;
    // thin MLP-layer GEMM [1024x256]@[256x128] = 33.6M FLOP, n=128 < 192 -> serial
    assert!(gemm_should_run_serial(1024, 128, 256), "thin NN GEMM must run serial");
    // square 256^3 = 16.7M FLOP, n=256 -> still parallel (~1.24x benefit)
    assert!(!gemm_should_run_serial(256, 256, 256), "square sub-64M still parallelizes");
    // tiny -> serial
    assert!(gemm_should_run_serial(8, 8, 8), "tiny GEMM serial");
    // large square 512^3 = 134M -> parallel
    assert!(!gemm_should_run_serial(512, 512, 512), "large GEMM parallelizes");
}
