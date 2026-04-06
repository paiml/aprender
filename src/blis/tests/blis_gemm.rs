use super::super::*;

// ========================================================================
// Phase 4: BLIS GEMM Tests
// ========================================================================

#[test]
fn test_gemm_blis_small() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];

    gemm_blis(2, 2, 2, &a, &b, &mut c, None).unwrap();

    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_gemm_blis_medium() {
    let n = 64;
    let a: Vec<f32> = (0..n * n).map(|i| (i % 10) as f32).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i + 3) % 10) as f32).collect();
    let mut c_ref = vec![0.0; n * n];
    let mut c_blis = vec![0.0; n * n];

    gemm_reference(n, n, n, &a, &b, &mut c_ref).unwrap();
    gemm_blis(n, n, n, &a, &b, &mut c_blis, None).unwrap();

    for i in 0..n * n {
        let diff = (c_ref[i] - c_blis[i]).abs();
        assert!(diff < 1e-3, "Mismatch at {}: ref={}, blis={}", i, c_ref[i], c_blis[i]);
    }
}

#[test]
fn test_gemm_blis_large() {
    let n = 256;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; n * n];
    let mut c_blis = vec![0.0; n * n];

    gemm_reference(n, n, n, &a, &b, &mut c_ref).unwrap();
    gemm_blis(n, n, n, &a, &b, &mut c_blis, None).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..n * n {
        let diff = (c_ref[i] - c_blis[i]).abs();
        max_diff = max_diff.max(diff);
    }

    assert!(max_diff < 1e-2, "Max diff: {}", max_diff);
}

#[test]
fn test_gemm_blis_rectangular() {
    // Common ML shape: 32 x 4096 @ 4096 x 11008
    let m = 32;
    let k = 128;
    let n = 256;

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        let diff = (c_ref[i] - c_blis[i]).abs();
        max_diff = max_diff.max(diff);
    }

    assert!(max_diff < 1e-3, "Max diff: {}", max_diff);
}

#[test]
fn test_gemm_blis_edge_m_not_divisible_by_mr() {
    let m = 13; // Not divisible by MR=8
    let n = 16;
    let k = 16;

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    for i in 0..m * n {
        let diff = (c_ref[i] - c_blis[i]).abs();
        assert!(diff < 1e-3, "Mismatch at {}: {} vs {}", i, c_ref[i], c_blis[i]);
    }
}

// ========================================================================
// Broadcast-B GEMM Tests (faer-style MR=64, NR=6)
// ========================================================================

/// FALSIFY-BCAST-B-001: broadcast-B GEMM matches reference for 256×256.
#[cfg(target_arch = "x86_64")]
#[test]
fn test_gemm_broadcast_b_256() {
    let n = 256;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; n * n];
    let mut c_bcast = vec![0.0; n * n];

    gemm_reference(n, n, n, &a, &b, &mut c_ref).unwrap();
    super::super::compute::gemm_blis_broadcast_b(n, n, n, &a, &b, &mut c_bcast).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..n * n {
        let diff = (c_ref[i] - c_bcast[i]).abs();
        max_diff = max_diff.max(diff);
    }

    assert!(max_diff < 1e-1, "FALSIFY-BCAST-B-001: max diff {max_diff} >= 1e-1 at 256");
}

/// FALSIFY-BCAST-B-002: broadcast-B GEMM matches reference for non-MR-aligned M.
#[cfg(target_arch = "x86_64")]
#[test]
fn test_gemm_broadcast_b_non_aligned() {
    let m = 100; // Not a multiple of MR=64
    let n = 96;
    let k = 128;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_bcast = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    super::super::compute::gemm_blis_broadcast_b(m, n, k, &a, &b, &mut c_bcast).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        let diff = (c_ref[i] - c_bcast[i]).abs();
        max_diff = max_diff.max(diff);
    }

    assert!(max_diff < 1e-1, "FALSIFY-BCAST-B-002: max diff {max_diff} >= 1e-1 at {m}x{n}x{k}");
}

#[test]
fn test_gemm_blis_edge_n_not_divisible_by_nr() {
    let m = 16;
    let n = 17; // Not divisible by NR=6
    let k = 16;

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    for i in 0..m * n {
        let diff = (c_ref[i] - c_blis[i]).abs();
        assert!(diff < 1e-3, "Mismatch at {}: {} vs {}", i, c_ref[i], c_blis[i]);
    }
}
