use super::super::*;

// ========================================================================
// Falsification Tests (Popperian)
// ========================================================================

#[test]
fn test_falsification_01_scalar_matches_numpy_2x2() {
    // Falsifiable: If this fails, our reference is wrong
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    gemm_reference(2, 2, 2, &a, &b, &mut c).unwrap();
    // numpy.dot([[1,2],[3,4]], [[5,6],[7,8]]) = [[19,22],[43,50]]
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_falsification_02_microkernel_k1() {
    // Falsifiable: Microkernel with k=1 must match outer product
    let a = vec![1.0; MR];
    let b = vec![2.0; NR];
    let mut c = vec![0.0; MR * NR];
    microkernel_scalar(1, &a, &b, &mut c, MR);
    for val in &c {
        assert_eq!(*val, 2.0);
    }
}

#[test]
fn test_falsification_09_edge_m_not_mr() {
    // M=13, not divisible by MR=8
    let m = 13;
    let n = 8;
    let k = 8;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];
    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();
    for i in 0..m * n {
        assert!((c_ref[i] - c_blis[i]).abs() < 1.0);
    }
}

#[test]
fn test_falsification_10_edge_n_not_nr() {
    // N=17, not divisible by NR=6
    let m = 8;
    let n = 17;
    let k = 8;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];
    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();
    for i in 0..m * n {
        assert!((c_ref[i] - c_blis[i]).abs() < 1.0);
    }
}

#[test]
fn test_falsification_18_zero_matrix_a() {
    let m = 16;
    let n = 16;
    let k = 16;
    let a = vec![0.0; m * k];
    let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
    let mut c = vec![1.0; m * n];
    let c_orig = c.clone();
    gemm_blis(m, n, k, &a, &b, &mut c, None).unwrap();
    // C should be unchanged (0 * B = 0, C += 0)
    assert_eq!(c, c_orig);
}

#[test]
fn test_falsification_19_identity() {
    let n = 16;
    let mut identity = vec![0.0; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    let a: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
    let mut c = vec![0.0; n * n];
    gemm_blis(n, n, n, &a, &identity, &mut c, None).unwrap();
    for i in 0..n * n {
        assert!((c[i] - a[i]).abs() < 1e-3);
    }
}

// F3: Microkernel matches reference for k=64
#[test]
fn test_falsification_03_microkernel_k64() {
    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| ((i % 10) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| ((i % 10) as f32) * 0.1).collect();
    let mut c_ref = [0.0f32; MR * NR];
    let mut c_scalar = [0.0f32; MR * NR];

    // Reference: simple accumulation
    for p in 0..k {
        for j in 0..NR {
            for i in 0..MR {
                c_ref[j * MR + i] += a[p * MR + i] * b[p * NR + j];
            }
        }
    }

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    for i in 0..MR * NR {
        assert!((c_ref[i] - c_scalar[i]).abs() < 1e-4, "F3: k=64 mismatch at {}", i);
    }
}

// F4: Microkernel matches reference for k=256
#[test]
fn test_falsification_04_microkernel_k256() {
    let k = 256;
    let a: Vec<f32> = (0..MR * k).map(|i| ((i % 50) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| ((i % 50) as f32) * 0.01).collect();
    let mut c_ref = [0.0f32; MR * NR];
    let mut c_scalar = [0.0f32; MR * NR];

    for p in 0..k {
        for j in 0..NR {
            for i in 0..MR {
                c_ref[j * MR + i] += a[p * MR + i] * b[p * NR + j];
            }
        }
    }

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    for i in 0..MR * NR {
        assert!((c_ref[i] - c_scalar[i]).abs() < 1e-3, "F4: k=256 mismatch at {}", i);
    }
}

// F5: Pack A produces correct layout
#[test]
fn test_falsification_05_pack_a_layout() {
    let mc = 16;
    let kc = 8;
    let a: Vec<f32> = (0..mc * kc).map(|i| i as f32).collect();
    let mut packed = vec![0.0f32; packed_a_size(mc, kc)];

    pack_a(&a, kc, mc, kc, &mut packed);

    // Verify first panel (MR=8 rows)
    for col in 0..kc {
        for row in 0..MR {
            let expected = a[row * kc + col];
            let actual = packed[col * MR + row];
            assert_eq!(expected, actual, "F5: Pack A mismatch at row={}, col={}", row, col);
        }
    }
}

// F6: Pack B produces correct layout
#[test]
fn test_falsification_06_pack_b_layout() {
    let kc = 8;
    let nc = 12;
    let b: Vec<f32> = (0..kc * nc).map(|i| i as f32).collect();
    let mut packed = vec![0.0f32; packed_b_size(kc, nc)];

    pack_b(&b, nc, kc, nc, &mut packed);

    // Verify first panel (NR=6 columns)
    for row in 0..kc {
        for col in 0..NR {
            let expected = b[row * nc + col];
            let actual = packed[row * NR + col];
            assert_eq!(expected, actual, "F6: Pack B mismatch at row={}, col={}", row, col);
        }
    }
}

// F7: L2 blocking produces correct result (MC boundary)
#[test]
fn test_falsification_07_l2_blocking_mc_boundary() {
    // Test with M = MC + partial = 72 + 16 = 88
    let m = MC + 16;
    let n = 32;
    let k = 64;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    let max_diff: f32 =
        c_ref.iter().zip(c_blis.iter()).map(|(r, b)| (r - b).abs()).fold(0.0, f32::max);

    assert!(max_diff < 1e-2, "F7: L2 blocking MC boundary max_diff={}", max_diff);
}

// F8: L3 blocking produces correct result (NC boundary)
#[test]
fn test_falsification_08_l3_blocking_nc_boundary() {
    // Test with N that triggers NC blocking (smaller for test speed)
    let m = 32;
    let n = 256; // Would trigger NC blocking if NC < 256
    let k = 64;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    let max_diff: f32 =
        c_ref.iter().zip(c_blis.iter()).map(|(r, b)| (r - b).abs()).fold(0.0, f32::max);

    assert!(max_diff < 1e-2, "F8: L3 blocking NC boundary max_diff={}", max_diff);
}

// F11: Edge case: K not divisible by KC
#[test]
fn test_falsification_11_k_not_divisible_by_kc() {
    let m = 32;
    let n = 32;
    let k = 300; // KC=256, so 300 = 256 + 44
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    let max_diff: f32 =
        c_ref.iter().zip(c_blis.iter()).map(|(r, b)| (r - b).abs()).fold(0.0, f32::max);

    assert!(max_diff < 1e-1, "F11: K not divisible by KC max_diff={}", max_diff);
}

// F12: Edge case: M=1 (vector-matrix multiplication)
#[test]
fn test_falsification_12_vector_matrix() {
    let m = 1;
    let n = 64;
    let k = 64;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 10) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    let max_diff: f32 =
        c_ref.iter().zip(c_blis.iter()).map(|(r, b)| (r - b).abs()).fold(0.0, f32::max);

    assert!(max_diff < 1e-3, "F12: Vector-matrix max_diff={}", max_diff);
}

// F13: Edge case: N=1 (matrix-vector multiplication)
#[test]
fn test_falsification_13_matrix_vector() {
    let m = 64;
    let n = 1;
    let k = 64;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 10) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    let max_diff: f32 =
        c_ref.iter().zip(c_blis.iter()).map(|(r, b)| (r - b).abs()).fold(0.0, f32::max);

    assert!(max_diff < 1e-3, "F13: Matrix-vector max_diff={}", max_diff);
}

// F14: Edge case: K=1 (outer product)
#[test]
fn test_falsification_14_outer_product() {
    let m = 32;
    let n = 32;
    let k = 1;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_blis = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    // Outer product: c[i,j] = a[i] * b[j]
    for i in 0..m * n {
        assert!((c_ref[i] - c_blis[i]).abs() < 1e-5, "F14: Outer product mismatch at {}", i);
    }
}

// F15: Subnormal inputs handled
#[test]
fn test_falsification_15_subnormal_inputs() {
    let m = 8;
    let n = 8;
    let k = 8;
    // Use very small (subnormal) values
    let subnormal = f32::MIN_POSITIVE / 2.0;
    let a: Vec<f32> = vec![subnormal; m * k];
    let b: Vec<f32> = vec![1.0; k * n];
    let mut c = vec![0.0; m * n];

    gemm_blis(m, n, k, &a, &b, &mut c, None).unwrap();

    // Should not produce NaN or Inf
    for val in &c {
        assert!(!val.is_nan(), "F15: NaN produced from subnormal inputs");
        assert!(!val.is_infinite(), "F15: Inf produced from subnormal inputs");
    }
}

// F16: Large values handled (no overflow check, just correctness)
#[test]
fn test_falsification_16_large_values() {
    let m = 8;
    let n = 8;
    let k = 4; // Small k to avoid overflow
    let large = 1e10f32;
    let a: Vec<f32> = vec![large; m * k];
    let b: Vec<f32> = vec![1e-10; k * n]; // Counter-balance to avoid overflow
    let mut c = vec![0.0; m * n];

    gemm_blis(m, n, k, &a, &b, &mut c, None).unwrap();

    // Should produce finite values around k * large * 1e-10 = k
    for val in &c {
        assert!(!val.is_nan(), "F16: NaN from large values");
        assert!(val.is_finite(), "F16: Infinite from large values");
    }
}

// F17: Negative values handled correctly
#[test]
fn test_falsification_17_negative_values() {
    let a = vec![-1.0, -2.0, -3.0, -4.0];
    let b = vec![5.0, -6.0, 7.0, -8.0];
    let mut c = vec![0.0; 4];

    gemm_reference(2, 2, 2, &a, &b, &mut c).unwrap();

    // [-1 -2] * [ 5 -6] = [-1*5-2*7  -1*(-6)-2*(-8)] = [-19  22]
    // [-3 -4]   [ 7 -8]   [-3*5-4*7  -3*(-6)-4*(-8)]   [-43  50]
    assert_eq!(c, vec![-19.0, 22.0, -43.0, 50.0], "F17: Negative values incorrect");
}

// F20: Associativity (approximate)
#[test]
fn test_falsification_20_associativity() {
    let n = 16;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 5) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let c: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();

    // Compute (A * B) * C
    let mut ab = vec![0.0; n * n];
    let mut abc_left = vec![0.0; n * n];
    gemm_reference(n, n, n, &a, &b, &mut ab).unwrap();
    gemm_reference(n, n, n, &ab, &c, &mut abc_left).unwrap();

    // Compute A * (B * C)
    let mut bc = vec![0.0; n * n];
    let mut abc_right = vec![0.0; n * n];
    gemm_reference(n, n, n, &b, &c, &mut bc).unwrap();
    gemm_reference(n, n, n, &a, &bc, &mut abc_right).unwrap();

    // Should be approximately equal (floating-point associativity)
    let max_rel_diff: f32 = abc_left
        .iter()
        .zip(abc_right.iter())
        .map(|(l, r)| (l - r).abs() / l.abs().max(1e-10))
        .fold(0.0, f32::max);

    assert!(max_rel_diff < 1e-4, "F20: Associativity max_rel_diff={}", max_rel_diff);
}

// ========================================================================
// Memory Criteria Tests (F31-F37)
// ========================================================================

// F34: Workspace allocation is bounded by cache hierarchy constants
#[test]
fn test_falsification_34_workspace_allocation() {
    // BLIS workspace is fixed-size for cache hierarchy, not proportional to matrix
    // Pack A: MC × KC for L2 cache (rounded to MR panels)
    // Pack B: KC × NC for L3 cache (rounded to NR panels)
    let packed_a = packed_a_size(MC, KC);
    let packed_b = packed_b_size(KC, NC);

    // Verify sizes are at least the minimum required
    assert!(packed_a >= MC * KC, "F34: Pack A too small");
    assert!(packed_b >= KC * NC, "F34: Pack B too small");

    // Verify padding overhead is minimal (< 1% for typical sizes)
    let a_overhead = (packed_a as f64 / (MC * KC) as f64) - 1.0;
    let b_overhead = (packed_b as f64 / (KC * NC) as f64) - 1.0;
    assert!(a_overhead < 0.01, "F34: Pack A overhead {} > 1%", a_overhead);
    assert!(b_overhead < 0.01, "F34: Pack B overhead {} > 1%", b_overhead);

    // Total workspace should be < 8 MB (reasonable for modern CPUs)
    let total_bytes = (packed_a + packed_b) * 4; // f32 = 4 bytes
    assert!(total_bytes < 8 * 1024 * 1024, "F34: Workspace {} bytes > 8MB", total_bytes);
}
