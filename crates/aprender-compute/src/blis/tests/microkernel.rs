use super::super::microkernels::*;
use super::super::*;

// ========================================================================
// Phase 2: Microkernel Tests
// ========================================================================

#[test]
fn test_microkernel_scalar_single_k() {
    // MR=8, NR=6, K=1
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 8x1
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 1x6
    let mut c = vec![0.0; MR * NR]; // 8x6 column-major

    microkernel_scalar(1, &a, &b, &mut c, MR);

    // c[j,i] = a[i] * b[j]
    for j in 0..NR {
        for i in 0..MR {
            let expected = a[i] * b[j];
            assert!(
                (c[j * MR + i] - expected).abs() < 1e-6,
                "Mismatch at ({}, {}): {} vs {}",
                i,
                j,
                c[j * MR + i],
                expected
            );
        }
    }
}

#[test]
fn test_microkernel_scalar_accumulation() {
    let a = vec![1.0; MR * 4]; // 8x4
    let b = vec![1.0; 4 * NR]; // 4x6
    let mut c = vec![0.0; MR * NR];

    microkernel_scalar(4, &a, &b, &mut c, MR);

    // Each output should be 4.0 (sum of 4 ones)
    for val in &c {
        assert!((val - 4.0).abs() < 1e-6);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

    let mut c_scalar = vec![0.0; MR * NR];
    let mut c_avx2 = vec![0.0; MR * NR];

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_avx2(k, a.as_ptr(), b.as_ptr(), c_avx2.as_mut_ptr(), MR);
    }

    for i in 0..MR * NR {
        let diff = (c_scalar[i] - c_avx2[i]).abs();
        let rel_diff = diff / c_scalar[i].abs().max(1e-10);
        assert!(
            rel_diff < 1e-5,
            "Mismatch at {}: scalar={}, avx2={}, rel_diff={}",
            i,
            c_scalar[i],
            c_avx2[i],
            rel_diff
        );
    }
}

// ========================================================================
// AVX2 ASM Microkernel Tests (coverage for microkernel_8x6_avx2_asm)
// ========================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_asm_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

    let mut c_scalar = vec![0.0; MR * NR];
    let mut c_asm = vec![0.0; MR * NR];

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
    }

    for i in 0..MR * NR {
        let diff = (c_scalar[i] - c_asm[i]).abs();
        let rel_diff = diff / c_scalar[i].abs().max(1e-10);
        assert!(
            rel_diff < 1e-4,
            "avx2_asm mismatch at {}: scalar={}, asm={}, rel_diff={}",
            i,
            c_scalar[i],
            c_asm[i],
            rel_diff
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_asm_k_less_than_4() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    for k in [1, 2, 3] {
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            assert!(
                diff < 1e-4,
                "avx2_asm k={} mismatch at {}: scalar={}, asm={}, diff={}",
                k,
                i,
                c_scalar[i],
                c_asm[i],
                diff
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_asm_k_with_remainder() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    // k=5 → 1 unrolled iteration + 1 remainder
    // k=7 → 1 unrolled iteration + 3 remainder
    for k in [5, 6, 7, 9, 13, 17] {
        let a: Vec<f32> = (0..MR * k).map(|i| ((i % 10) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| ((i % 7) as f32) * 0.2).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-4,
                "avx2_asm k={} mismatch at {}: scalar={}, asm={}, rel_diff={}",
                k,
                i,
                c_scalar[i],
                c_asm[i],
                rel_diff
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_asm_k_exact_multiple_of_4() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    for k in [4, 8, 16, 32] {
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.02).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-4,
                "avx2_asm k={} mismatch at {}: scalar={}, asm={}",
                k,
                i,
                c_scalar[i],
                c_asm[i]
            );
        }
    }
}

// ========================================================================
// True ASM Microkernel Tests (coverage for microkernel_8x6_true_asm)
// ========================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

    let mut c_scalar = vec![0.0; MR * NR];
    let mut c_asm = vec![0.0; MR * NR];

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
    }

    for i in 0..MR * NR {
        let diff = (c_scalar[i] - c_asm[i]).abs();
        let rel_diff = diff / c_scalar[i].abs().max(1e-10);
        assert!(
            rel_diff < 1e-4,
            "true_asm mismatch at {}: scalar={}, asm={}, rel_diff={}",
            i,
            c_scalar[i],
            c_asm[i],
            rel_diff
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_k_less_than_4() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    for k in [1, 2, 3] {
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            assert!(
                diff < 1e-4,
                "true_asm k={} mismatch at {}: scalar={}, asm={}, diff={}",
                k,
                i,
                c_scalar[i],
                c_asm[i],
                diff
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_k_with_remainder() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    // Test k values that exercise both main loop and remainder
    for k in [5, 6, 7, 9, 11, 15] {
        let a: Vec<f32> = (0..MR * k).map(|i| ((i % 10) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| ((i % 7) as f32) * 0.2).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-4,
                "true_asm k={} mismatch at {}: scalar={}, asm={}, rel_diff={}",
                k,
                i,
                c_scalar[i],
                c_asm[i],
                rel_diff
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_k_exact_multiple_of_4() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    for k in [4, 8, 12, 16, 32, 64] {
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.02).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-4,
                "true_asm k={} mismatch at {}: scalar={}, asm={}",
                k,
                i,
                c_scalar[i],
                c_asm[i]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_accumulates_into_c() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 8;
    let a: Vec<f32> = vec![1.0; MR * k];
    let b: Vec<f32> = vec![1.0; k * NR];

    // Pre-fill C with non-zero values
    let mut c = vec![10.0f32; MR * NR];

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), MR);
    }

    // Each output should be 10.0 + k = 18.0
    for (i, val) in c.iter().enumerate() {
        assert!(
            (*val - 18.0).abs() < 1e-3,
            "true_asm accumulation at {}: expected 18.0, got {}",
            i,
            val
        );
    }
}

// ========================================================================
// NEON 8x6 Microkernel Tests (aarch64) — contract neon-blis-v1.yaml
// ========================================================================

/// Deterministic pseudo-random values in [-1, 1) (LCG; no dev-dependency needed).
#[cfg(target_arch = "aarch64")]
fn neon_test_values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// FALSIFY-NEON-BLIS-001: the NEON 8×6 kernel matches the scalar reference kernel.
#[test]
#[cfg(target_arch = "aarch64")]
fn test_microkernel_neon_8x6_matches_scalar() {
    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

    let mut c_scalar = vec![0.0; MR * NR];
    let mut c_neon = vec![0.0; MR * NR];

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_neon(k, a.as_ptr(), b.as_ptr(), c_neon.as_mut_ptr(), MR);
    }

    for i in 0..MR * NR {
        let diff = (c_scalar[i] - c_neon[i]).abs();
        let rel_diff = diff / c_scalar[i].abs().max(1e-10);
        assert!(
            rel_diff < 1e-5,
            "Mismatch at {}: scalar={}, neon={}, rel_diff={}",
            i,
            c_scalar[i],
            c_neon[i],
            rel_diff
        );
    }
}

/// FALSIFY-NEON-BLIS-001 (edge K): every K including 0, the unrolled-remainder sizes and
/// the KC block size, accumulating into a non-zero C exactly like the scalar kernel.
#[test]
#[cfg(target_arch = "aarch64")]
fn test_microkernel_neon_8x6_short_k_and_accumulate() {
    for k in [0usize, 1, 2, 3, 4, 5, 7, 9, 255, 256, 257] {
        let a = neon_test_values(MR * k, 11 + k as u64);
        let b = neon_test_values(k * NR, 29 + k as u64);
        let c0 = neon_test_values(MR * NR, 47 + k as u64);

        let mut c_scalar = c0.clone();
        let mut c_neon = c0;

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);
        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_neon(k, a.as_ptr(), b.as_ptr(), c_neon.as_mut_ptr(), MR);
        }

        let scale = c_scalar.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_neon[i]).abs();
            assert!(
                diff <= 1e-5 * scale,
                "k={k}: mismatch at {i}: scalar={}, neon={}, diff={diff}",
                c_scalar[i],
                c_neon[i]
            );
        }
    }
}

/// FALSIFY-NEON-BLIS-002: through `gemm_blis`, shapes with remainder tiles in M, N and K,
/// K below the unroll width, and a transformer-shaped GEMM all match the naive reference.
#[test]
#[cfg(target_arch = "aarch64")]
fn test_gemm_blis_neon_odd_shapes_match_reference() {
    // All above the `m * n * k < 4096` reference cutoff so the BLIS path is exercised.
    for (m, n, k) in [
        (129usize, 131usize, 67usize),
        (33, 50, 100),
        (7, 5, 300),
        (64, 64, 1),
        (64, 48, 3),
        (8, 6, 256),
        (17, 9, 257),
        (129, 256, 256),
        (200, 300, 17),
    ] {
        let a = neon_test_values(m * k, 3 + m as u64);
        let b = neon_test_values(k * n, 5 + n as u64);
        let mut c_blis = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];

        gemm_blis(m, n, k, &a, &b, &mut c_blis, None).expect("gemm_blis");
        gemm_reference(m, n, k, &a, &b, &mut c_ref).expect("gemm_reference");

        let scale = c_ref.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
        for i in 0..m * n {
            let diff = (c_blis[i] - c_ref[i]).abs();
            assert!(
                diff <= 1e-5 * scale,
                "{m}x{n}x{k}: mismatch at {i}: blis={}, ref={}, diff={diff}",
                c_blis[i],
                c_ref[i]
            );
        }
    }
}
