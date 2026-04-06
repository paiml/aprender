use crate::blis::*;

// ========================================================================
// Property-Based Tests (Fast, Deterministic)
// ========================================================================

/// Property: GEMM with zero matrix A produces unchanged C
#[test]
fn prop_zero_a_unchanged_c() {
    for n in [8, 16, 32, 64] {
        let a = vec![0.0f32; n * n];
        let b: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let mut c = vec![1.0f32; n * n];
        let c_orig = c.clone();

        gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();

        assert_eq!(c, c_orig, "C should be unchanged when A=0 for n={}", n);
    }
}

/// Property: GEMM with zero matrix B produces unchanged C
#[test]
fn prop_zero_b_unchanged_c() {
    for n in [8, 16, 32, 64] {
        let a: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let b = vec![0.0f32; n * n];
        let mut c = vec![1.0f32; n * n];
        let c_orig = c.clone();

        gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();

        assert_eq!(c, c_orig, "C should be unchanged when B=0 for n={}", n);
    }
}

/// Property: GEMM is consistent across multiple calls
#[test]
fn prop_deterministic() {
    let n = 64;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();

    let mut c1 = vec![0.0f32; n * n];
    let mut c2 = vec![0.0f32; n * n];

    gemm_blis(n, n, n, &a, &b, &mut c1, None).unwrap();
    gemm_blis(n, n, n, &a, &b, &mut c2, None).unwrap();

    assert_eq!(c1, c2, "GEMM should be deterministic");
}

/// Property: BLIS matches reference for various dimensions
#[test]
fn prop_blis_matches_reference() {
    let test_cases = [
        (8, 8, 8),
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (13, 17, 19), // Primes (not divisible by MR/NR)
        (1, 64, 64),  // Vector-matrix
        (64, 1, 64),  // Matrix-vector
        (64, 64, 1),  // Outer product
    ];

    for (m, n, k) in test_cases {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.1).collect();

        let mut c_ref = vec![0.0f32; m * n];
        let mut c_blis = vec![0.0f32; m * n];

        gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
        gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

        let max_diff: f32 =
            c_ref.iter().zip(c_blis.iter()).map(|(r, b)| (r - b).abs()).fold(0.0, f32::max);

        assert!(
            max_diff < 1e-3,
            "BLIS should match reference for {}x{}x{}, max_diff={}",
            m,
            n,
            k,
            max_diff
        );
    }
}

/// Property: Accumulation works correctly (C += A*B)
#[test]
fn prop_accumulation() {
    let n = 32;
    let a: Vec<f32> = vec![1.0; n * n];
    let b: Vec<f32> = vec![1.0; n * n];

    let mut c = vec![0.0f32; n * n];

    gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();
    let c_first = c.clone();

    gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();

    for i in 0..n * n {
        let expected = c_first[i] * 2.0;
        assert!(
            (c[i] - expected).abs() < 1e-3,
            "Accumulation failed at {}: {} vs {}",
            i,
            c[i],
            expected
        );
    }
}

/// Property: Scaling works (alpha * A * B)
#[test]
fn prop_scaling() {
    let n = 32;
    let a: Vec<f32> = (0..n * n).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = vec![1.0; n * n];

    let mut c1 = vec![0.0f32; n * n];
    gemm_blis(n, n, n, &a, &b, &mut c1, None).unwrap();

    let a_scaled: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
    let mut c2 = vec![0.0f32; n * n];
    gemm_blis(n, n, n, &a_scaled, &b, &mut c2, None).unwrap();

    for i in 0..n * n {
        let expected = c1[i] * 2.0;
        assert!(
            (c2[i] - expected).abs() < 1e-2,
            "Scaling property failed at {}: {} vs {}",
            i,
            c2[i],
            expected
        );
    }
}

/// Property: Microkernel produces correct output dimensions
#[test]
fn prop_microkernel_dimensions() {
    for k in [1, 4, 16, 64, 256] {
        let a = vec![1.0f32; MR * k];
        let b = vec![1.0f32; k * NR];
        let mut c = vec![0.0f32; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c, MR);

        for val in &c {
            assert!(
                (*val - k as f32).abs() < 1e-5,
                "Microkernel output wrong for k={}: {} vs {}",
                k,
                val,
                k
            );
        }
    }
}

/// Property: Packing preserves all elements
#[test]
fn prop_pack_preserves_elements() {
    let mc = 32;
    let kc = 64;

    let a: Vec<f32> = (0..mc * kc).map(|i| i as f32).collect();
    let mut packed = vec![0.0f32; packed_a_size(mc, kc)];

    pack_a(&a, kc, mc, kc, &mut packed);

    let _orig_sum: f32 = a.iter().sum();
    let _packed_sum: f32 = packed.iter().sum();

    let mut found = vec![false; mc * kc];
    for val in &packed {
        let idx = *val as usize;
        if idx < mc * kc {
            found[idx] = true;
        }
    }

    let all_found = found.iter().all(|&f| f);
    assert!(all_found, "Packing should preserve all unique values");
}
