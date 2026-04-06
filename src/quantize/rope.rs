
/// AVX2 SIMD dequantization for a single Q8_0 block
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn dequantize_q8_0_block_avx2(block_data: &[u8]) -> Vec<f32> {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let mut result = vec![0.0f32; 32];

    // Read scale (f16 -> f32)
    let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
    let scale = f16_to_f32(scale_bits);

    // SAFETY: AVX2 availability verified by caller's target_feature
    unsafe {
        let scale_vec = _mm256_set1_ps(scale);

        // Process 32 i8 values in 4 iterations of 8
        for chunk in 0..4 {
            let byte_start = 2 + chunk * 8; // Start at offset 2 (after f16 scale)

            // Load 8 i8 values and sign-extend to i32
            let q0 = block_data[byte_start] as i8 as i32;
            let q1 = block_data[byte_start + 1] as i8 as i32;
            let q2 = block_data[byte_start + 2] as i8 as i32;
            let q3 = block_data[byte_start + 3] as i8 as i32;
            let q4 = block_data[byte_start + 4] as i8 as i32;
            let q5 = block_data[byte_start + 5] as i8 as i32;
            let q6 = block_data[byte_start + 6] as i8 as i32;
            let q7 = block_data[byte_start + 7] as i8 as i32;

            let q_vec = _mm256_setr_epi32(q0, q1, q2, q3, q4, q5, q6, q7);
            let q_f32 = _mm256_cvtepi32_ps(q_vec);

            // Multiply by scale
            let dequant = _mm256_mul_ps(scale_vec, q_f32);

            // Store 8 results
            _mm256_storeu_ps(result.as_mut_ptr().add(chunk * 8), dequant);
        }
    }

    result
}

// DequantStats, SimdBackend, detect_simd_backend moved to types.rs (PMAT-802)

/// SIMD-optimized RoPE rotation for a single head
///
/// Applies rotary position embedding rotation to a single attention head:
/// x1[i] = x1[i] * cos[i] - x2[i] * sin[i]
/// x2[i] = x1[i] * sin[i] + x2[i] * cos[i]
///
/// # Arguments
/// * `x1` - First half of head (will be modified in-place)
/// * `x2` - Second half of head (will be modified in-place)
/// * `cos_vals` - Precomputed cosine values
/// * `sin_vals` - Precomputed sine values
#[inline]
pub fn apply_rope_rotation_simd(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    contract_pre_rotation_matrix!();
    debug_assert_eq!(x1.len(), x2.len());
    debug_assert_eq!(x1.len(), cos_vals.len());
    debug_assert_eq!(x1.len(), sin_vals.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                apply_rope_rotation_avx512(x1, x2, cos_vals, sin_vals);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                apply_rope_rotation_avx2(x1, x2, cos_vals, sin_vals);
            }
            return;
        }
    }

    // Scalar fallback
    apply_rope_rotation_scalar(x1, x2, cos_vals, sin_vals);
    contract_post_rotation_matrix!(&x1);
}

/// Scalar fallback for RoPE rotation
///
/// Exposed as `pub(crate)` for direct testing on AVX2 machines.
#[inline]
pub(crate) fn apply_rope_rotation_scalar(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    for i in 0..x1.len() {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn apply_rope_rotation_avx2(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
    };

    let n = x1.len();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= n {
        let v1 = _mm256_loadu_ps(x1.as_ptr().add(i));
        let v2 = _mm256_loadu_ps(x2.as_ptr().add(i));
        let cos_v = _mm256_loadu_ps(cos_vals.as_ptr().add(i));
        let sin_v = _mm256_loadu_ps(sin_vals.as_ptr().add(i));

        // r1 = v1 * cos - v2 * sin
        let v1_cos = _mm256_mul_ps(v1, cos_v);
        let r1 = _mm256_fnmadd_ps(v2, sin_v, v1_cos);

        // r2 = v1 * sin + v2 * cos
        let v1_sin = _mm256_mul_ps(v1, sin_v);
        let r2 = _mm256_fmadd_ps(v2, cos_v, v1_sin);

        _mm256_storeu_ps(x1.as_mut_ptr().add(i), r1);
        _mm256_storeu_ps(x2.as_mut_ptr().add(i), r2);

        i += 8;
    }

    // Handle remainder
    while i < n {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn apply_rope_rotation_avx512(
    x1: &mut [f32],
    x2: &mut [f32],
    cos_vals: &[f32],
    sin_vals: &[f32],
) {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_storeu_ps,
    };

    let n = x1.len();
    let mut i = 0;

    // Process 16 elements at a time with AVX-512
    while i + 16 <= n {
        let v1 = _mm512_loadu_ps(x1.as_ptr().add(i));
        let v2 = _mm512_loadu_ps(x2.as_ptr().add(i));
        let cos_v = _mm512_loadu_ps(cos_vals.as_ptr().add(i));
        let sin_v = _mm512_loadu_ps(sin_vals.as_ptr().add(i));

        // r1 = v1 * cos - v2 * sin
        let v1_cos = _mm512_mul_ps(v1, cos_v);
        let r1 = _mm512_fnmadd_ps(v2, sin_v, v1_cos);

        // r2 = v1 * sin + v2 * cos
        let v1_sin = _mm512_mul_ps(v1, sin_v);
        let r2 = _mm512_fmadd_ps(v2, cos_v, v1_sin);

        _mm512_storeu_ps(x1.as_mut_ptr().add(i), r1);
        _mm512_storeu_ps(x2.as_mut_ptr().add(i), r2);

        i += 16;
    }

    // Handle remainder with AVX2 or scalar
    while i < n {
        let v1 = x1[i];
        let v2 = x2[i];
        let cos_v = cos_vals[i];
        let sin_v = sin_vals[i];
        x1[i] = v1 * cos_v - v2 * sin_v;
        x2[i] = v1 * sin_v + v2 * cos_v;
        i += 1;
    }
}

#[cfg(test)]
mod rope_contract_tests {
    use super::*;

    // =========================================================================
    // FALSIFY-RP: rope-kernel-v1.yaml contract (realizar RoPE SIMD)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: realizar had 15+ RoPE functions but zero FALSIFY-RP-* tests
    //   Why 2: RoPE tested only via end-to-end model inference, not unit contracts
    //   Why 3: no mapping from rope-kernel-v1.yaml to realizar test names
    //   Why 4: realizar predates the provable-contracts YAML convention
    //   Why 5: SIMD RoPE was benchmarked for speed, not contract-tested for correctness
    //
    // References:
    //   - provable-contracts/contracts/rope-kernel-v1.yaml
    //   - Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    // =========================================================================

    /// FALSIFY-RP-001: Norm preservation — ‖RoPE(x)‖ ≈ ‖x‖
    ///
    /// 2D rotation preserves vector norm for each (x1, x2) pair.
    #[test]
    fn falsify_rp_001_norm_preservation() {
        let dim = 32;
        let half = dim / 2;

        let mut x1: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut x2: Vec<f32> = (0..half).map(|i| (i as f32 * 0.73).cos()).collect();

        let input_norm: f32 = x1.iter().chain(x2.iter()).map(|v| v * v).sum::<f32>().sqrt();

        // Precompute cos/sin for position 42 with base 10000
        let cos_vals: Vec<f32> = (0..half)
            .map(|i| (42.0 * (10000.0_f32).powf(-2.0 * i as f32 / dim as f32)).cos())
            .collect();
        let sin_vals: Vec<f32> = (0..half)
            .map(|i| (42.0 * (10000.0_f32).powf(-2.0 * i as f32 / dim as f32)).sin())
            .collect();

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        let output_norm: f32 = x1.iter().chain(x2.iter()).map(|v| v * v).sum::<f32>().sqrt();

        let diff = (output_norm - input_norm).abs();
        assert!(
            diff < 1e-4,
            "FALSIFIED RP-001: ‖RoPE(x)‖ = {output_norm}, ‖x‖ = {input_norm}, diff = {diff}"
        );
    }

    /// FALSIFY-RP-003: SIMD equivalence — SIMD matches scalar within tolerance
    ///
    /// Contract: |rope_simd(x) - rope_scalar(x)| < 4 ULP
    #[test]
    fn falsify_rp_003_simd_vs_scalar_equivalence() {
        let dim = 64;
        let half = dim / 2;

        let x1_orig: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).sin()).collect();
        let x2_orig: Vec<f32> = (0..half).map(|i| (i as f32 * 0.73).cos()).collect();

        let cos_vals: Vec<f32> = (0..half)
            .map(|i| (10.0 * (10000.0_f32).powf(-2.0 * i as f32 / dim as f32)).cos())
            .collect();
        let sin_vals: Vec<f32> = (0..half)
            .map(|i| (10.0 * (10000.0_f32).powf(-2.0 * i as f32 / dim as f32)).sin())
            .collect();

        // Scalar path
        let mut x1_scalar = x1_orig.clone();
        let mut x2_scalar = x2_orig.clone();
        apply_rope_rotation_scalar(&mut x1_scalar, &mut x2_scalar, &cos_vals, &sin_vals);

        // SIMD path (dispatches to AVX2/AVX512 if available)
        let mut x1_simd = x1_orig;
        let mut x2_simd = x2_orig;
        apply_rope_rotation_simd(&mut x1_simd, &mut x2_simd, &cos_vals, &sin_vals);

        for i in 0..half {
            let diff1 = (x1_simd[i] - x1_scalar[i]).abs();
            let diff2 = (x2_simd[i] - x2_scalar[i]).abs();
            assert!(
                diff1 < 1e-5,
                "FALSIFIED RP-003: x1 SIMD vs scalar mismatch at [{i}]: {} vs {} (diff={diff1})",
                x1_simd[i], x1_scalar[i]
            );
            assert!(
                diff2 < 1e-5,
                "FALSIFIED RP-003: x2 SIMD vs scalar mismatch at [{i}]: {} vs {} (diff={diff2})",
                x2_simd[i], x2_scalar[i]
            );
        }
    }

    /// FALSIFY-RP-004: Zero position — RoPE(x, 0) = x (identity)
    ///
    /// At position 0, all angles are 0: cos(0)=1, sin(0)=0
    #[test]
    fn falsify_rp_004_zero_position_identity() {
        let dim = 16;
        let half = dim / 2;

        let x1_orig: Vec<f32> = vec![1.0, -2.0, 3.0, -0.5, 4.0, -1.0, 2.5, -3.0];
        let x2_orig: Vec<f32> = vec![0.5, 1.5, -1.0, 2.0, -3.0, 0.0, 1.0, -0.5];

        let mut x1 = x1_orig.clone();
        let mut x2 = x2_orig.clone();

        // Position 0: all angles = 0 * freq = 0
        let cos_vals = vec![1.0f32; half]; // cos(0) = 1
        let sin_vals = vec![0.0f32; half]; // sin(0) = 0

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        for i in 0..half {
            let diff1 = (x1[i] - x1_orig[i]).abs();
            let diff2 = (x2[i] - x2_orig[i]).abs();
            assert!(
                diff1 < 1e-6,
                "FALSIFIED RP-004: x1[{i}] changed from {} to {} at position 0",
                x1_orig[i], x1[i]
            );
            assert!(
                diff2 < 1e-6,
                "FALSIFIED RP-004: x2[{i}] changed from {} to {} at position 0",
                x2_orig[i], x2[i]
            );
        }
    }

    /// FALSIFY-RP-001b: Norm preservation per-pair
    ///
    /// Each (x1[i], x2[i]) pair should have preserved L2 norm after rotation.
    #[test]
    fn falsify_rp_001_per_pair_norm() {
        let half = 8;
        let dim = half * 2;

        let mut x1: Vec<f32> = (0..half).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut x2: Vec<f32> = (0..half).map(|i| (i as f32 + 1.0) * -0.3).collect();

        let pair_norms_before: Vec<f32> = (0..half)
            .map(|i| (x1[i] * x1[i] + x2[i] * x2[i]).sqrt())
            .collect();

        let cos_vals: Vec<f32> = (0..half)
            .map(|i| (7.0 * (10000.0_f32).powf(-2.0 * i as f32 / dim as f32)).cos())
            .collect();
        let sin_vals: Vec<f32> = (0..half)
            .map(|i| (7.0 * (10000.0_f32).powf(-2.0 * i as f32 / dim as f32)).sin())
            .collect();

        apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

        for i in 0..half {
            let norm_after = (x1[i] * x1[i] + x2[i] * x2[i]).sqrt();
            let diff = (norm_after - pair_norms_before[i]).abs();
            assert!(
                diff < 1e-5,
                "FALSIFIED RP-001: pair[{i}] norm changed: {} → {} (diff={diff})",
                pair_norms_before[i], norm_after
            );
        }
    }

    /// FALSIFY-RP-002: Relative position — dot product depends only on m-n
    ///
    /// Contract: ⟨RoPE(q,m), RoPE(k,n)⟩ = ⟨RoPE(q,0), RoPE(k,n-m)⟩
    #[test]
    fn falsify_rp_002_relative_position() {
        let dim = 16;
        let half = dim / 2;
        let base = 10000.0_f32;

        let q1_orig: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).sin()).collect();
        let q2_orig: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).cos()).collect();
        let k1_orig: Vec<f32> = (0..half).map(|i| (i as f32 * 0.73).cos()).collect();
        let k2_orig: Vec<f32> = (0..half).map(|i| (i as f32 * 0.73).sin()).collect();

        fn make_cos_sin(pos: f32, half: usize, dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
            let cos_vals: Vec<f32> = (0..half)
                .map(|i| (pos * base.powf(-2.0 * i as f32 / dim as f32)).cos())
                .collect();
            let sin_vals: Vec<f32> = (0..half)
                .map(|i| (pos * base.powf(-2.0 * i as f32 / dim as f32)).sin())
                .collect();
            (cos_vals, sin_vals)
        }

        // Test (m, n) pairs with same relative offset of 5
        let offsets = [(10, 15), (20, 25), (50, 55)];
        let mut dots = Vec::new();

        for &(m, n) in &offsets {
            let mut q1 = q1_orig.clone();
            let mut q2 = q2_orig.clone();
            let mut k1 = k1_orig.clone();
            let mut k2 = k2_orig.clone();

            let (cos_m, sin_m) = make_cos_sin(m as f32, half, dim, base);
            let (cos_n, sin_n) = make_cos_sin(n as f32, half, dim, base);

            apply_rope_rotation_simd(&mut q1, &mut q2, &cos_m, &sin_m);
            apply_rope_rotation_simd(&mut k1, &mut k2, &cos_n, &sin_n);

            let dot: f32 = q1.iter().zip(k1.iter()).map(|(&a, &b)| a * b).sum::<f32>()
                + q2.iter().zip(k2.iter()).map(|(&a, &b)| a * b).sum::<f32>();
            dots.push(dot);
        }

        for i in 1..dots.len() {
            let diff = (dots[i] - dots[0]).abs();
            assert!(
                diff < 1e-3,
                "FALSIFIED RP-002: dot products for same relative offset differ: {:?}",
                dots
            );
        }
    }

    mod rp_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // RP-001-prop: norm preservation for random dimensions and positions
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_rp_001_prop_norm_preservation(
                half in prop::sample::select(vec![4_usize, 8, 16, 32]),
                pos in 0.0_f32..1000.0,
            ) {
                let dim = half * 2;
                let base = 10000.0_f32;
                let mut x1: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37 * pos.sin()).cos()).collect();
                let mut x2: Vec<f32> = (0..half).map(|i| (i as f32 * 0.73 * pos.cos()).sin()).collect();

                let norm_before: f32 = x1.iter().chain(x2.iter()).map(|v| v * v).sum::<f32>().sqrt();

                let cos_vals: Vec<f32> = (0..half)
                    .map(|i| (pos * base.powf(-2.0 * i as f32 / dim as f32)).cos())
                    .collect();
                let sin_vals: Vec<f32> = (0..half)
                    .map(|i| (pos * base.powf(-2.0 * i as f32 / dim as f32)).sin())
                    .collect();

                apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

                let norm_after: f32 = x1.iter().chain(x2.iter()).map(|v| v * v).sum::<f32>().sqrt();
                prop_assert!(
                    (norm_after - norm_before).abs() < 1e-3,
                    "FALSIFIED RP-001-prop: norm {} → {} (d={}, pos={})",
                    norm_before, norm_after, dim, pos
                );
            }
        }

        // RP-004-prop: zero position identity for random vectors
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_rp_004_prop_zero_identity(
                half in prop::sample::select(vec![4_usize, 8, 16, 32]),
            ) {
                let mut x1: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
                let mut x2: Vec<f32> = (0..half).map(|i| (i as f32 * 0.73).cos() * 5.0).collect();
                let x1_orig = x1.clone();
                let x2_orig = x2.clone();

                let cos_vals = vec![1.0f32; half]; // cos(0) = 1
                let sin_vals = vec![0.0f32; half]; // sin(0) = 0

                apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

                for i in 0..half {
                    prop_assert!(
                        (x1[i] - x1_orig[i]).abs() < 1e-5,
                        "FALSIFIED RP-004-prop: x1[{i}] changed at pos 0"
                    );
                    prop_assert!(
                        (x2[i] - x2_orig[i]).abs() < 1e-5,
                        "FALSIFIED RP-004-prop: x2[{i}] changed at pos 0"
                    );
                }
            }
        }
    }
}
