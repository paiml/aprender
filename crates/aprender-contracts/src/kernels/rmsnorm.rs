//! RMSNorm kernel: root mean square layer normalization.
//!
//! Matches `rmsnorm-kernel-v1.yaml`.
//! `RMS(x)` = sqrt(sum(x^2)/n + eps), output = x/`RMS(x)` * gamma

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Scalar reference implementation of RMSNorm.
///
/// Computes `output_i = (x_i / sqrt(sum(x_j^2)/n + eps)) * gamma_i`.
///
/// # Panics
///
/// Panics if `input`, `gamma`, and `output` have different lengths, or if
/// `input` is empty.
pub fn rmsnorm_scalar(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(n, gamma.len(), "input/gamma length mismatch");
    assert_eq!(n, output.len(), "input/output length mismatch");
    assert!(n > 0, "rmsnorm requires non-empty input");

    // Phase 1: sum of squares
    let mut sum_sq = 0.0_f32;
    for &x in input {
        sum_sq += x * x;
    }

    // Phase 2: compute RMS = sqrt(sum_sq / n + eps)
    let rms = (sum_sq / n as f32 + eps).sqrt();

    // Phase 3: normalize and scale by gamma
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        output[i] = input[i] * inv_rms * gamma[i];
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 SIMD implementation of RMSNorm.
///
/// Uses `_mm256_mul_ps` + `_mm256_add_ps` for sum-of-squares accumulation,
/// then vectorized normalization with `_mm256_mul_ps`.
///
/// # Safety
///
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
///
/// Panics if `input`, `gamma`, and `output` have different lengths, or if
/// `input` is empty.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn rmsnorm_avx2(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(n, gamma.len(), "input/gamma length mismatch");
    assert_eq!(n, output.len(), "input/output length mismatch");
    assert!(n > 0, "rmsnorm requires non-empty input");

    let chunks = n / 8;
    let remainder_start = chunks * 8;

    // SAFETY: caller guarantees AVX2 is available; target_feature gate enforces it.
    unsafe {
        // Phase 1: sum of squares using AVX2
        let mut sum_vec = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(v, v));
        }

        // Horizontal sum of the 8-wide accumulator
        let mut tmp = [0.0_f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), sum_vec);
        let mut sum_sq = 0.0_f32;
        for &v in &tmp {
            sum_sq += v;
        }
        // Remainder
        for i in remainder_start..n {
            sum_sq += input[i] * input[i];
        }

        // Phase 2: compute inverse RMS
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Phase 3: normalize and scale using AVX2
        let inv_rms_vec = _mm256_set1_ps(inv_rms);
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let g = _mm256_loadu_ps(gamma.as_ptr().add(i * 8));
            let normed = _mm256_mul_ps(x, inv_rms_vec);
            let scaled = _mm256_mul_ps(normed, g);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), scaled);
        }
        // Scalar tail
        for i in remainder_start..n {
            output[i] = input[i] * inv_rms * gamma[i];
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for RMSNorm kernel.
///
/// Reduction pattern: shared memory for sum of squares, then `rsqrt.approx.f32`,
/// then normalize and scale by gamma. 1 block per vector, 256 threads.
pub fn rmsnorm_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// RMSNorm kernel: 1 block per vector, 256 threads per block.
// Two-pass: sum-of-squares reduction, then normalize * gamma.
.visible .entry rmsnorm_kernel(
    .param .u64 input_ptr,
    .param .u64 gamma_ptr,
    .param .u64 output_ptr,
    .param .u32 n,
    .param .f32 eps
)
{
    .reg .u32 %tid, %n, %i, %lane, %warp_id, %mask;
    .reg .u64 %in_base, %g_base, %out_base, %addr;
    .reg .f32 %val, %sq, %sum_local, %sum_warp, %sum_global;
    .reg .f32 %rms_inv, %eps, %nf, %normed, %gamma_val, %result;
    .reg .pred %p;
    .shared .f32 smem[32];

    ld.param.u64 %in_base, [input_ptr];
    ld.param.u64 %g_base, [gamma_ptr];
    ld.param.u64 %out_base, [output_ptr];
    ld.param.u32 %n, [n];
    ld.param.f32 %eps, [eps];

    mov.u32 %tid, %tid.x;
    mov.u32 %mask, 0xFFFFFFFF;

    // --- Pass 1: sum of squares ---
    mov.f32 %sum_local, 0f00000000;
    mov.u32 %i, %tid;
sum_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra sum_done;
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    mul.f32 %sq, %val, %val;
    add.f32 %sum_local, %sum_local, %sq;
    add.u32 %i, %i, 256;
    bra sum_loop;
sum_done:

    // Warp-level sum reduction via shuffle
    shfl.sync.down.b32 %sum_warp, %sum_local, 16, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 8, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 4, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 2, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 1, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;

    and.b32 %lane, %tid, 31;
    shr.b32 %warp_id, %tid, 5;
    setp.eq.u32 %p, %lane, 0;
    @%p st.shared.f32 [smem + %warp_id * 4], %sum_local;

    bar.sync 0;

    // First warp reduces across warps (8 warps for 256 threads)
    setp.lt.u32 %p, %tid, 8;
    @%p ld.shared.f32 %sum_local, [smem + %tid * 4];
    @!%p mov.f32 %sum_local, 0f00000000;
    shfl.sync.down.b32 %sum_warp, %sum_local, 4, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 2, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 1, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;

    // Compute rsqrt(sum/n + eps) and broadcast
    setp.eq.u32 %p, %tid, 0;
    cvt.rn.f32.u32 %nf, %n;
    div.approx.f32 %sum_global, %sum_local, %nf;
    add.f32 %sum_global, %sum_global, %eps;
    rsqrt.approx.f32 %rms_inv, %sum_global;
    @%p st.shared.f32 [smem], %rms_inv;
    bar.sync 0;
    ld.shared.f32 %rms_inv, [smem];

    // --- Pass 2: normalize and scale ---
    mov.u32 %i, %tid;
norm_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra norm_done;
    // Load input[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    // Load gamma[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %g_base, %addr;
    ld.global.f32 %gamma_val, [%addr];
    // output[i] = input[i] * inv_rms * gamma[i]
    mul.f32 %normed, %val, %rms_inv;
    mul.f32 %result, %normed, %gamma_val;
    // Store output[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out_base, %addr;
    st.global.f32 [%addr], %result;
    add.u32 %i, %i, 256;
    bra norm_loop;
norm_done:

    bar.sync 0;
    ret;
}
"#
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    // ── Scalar known-answer tests ────────────────────────────────────────

    #[test]
    fn test_rmsnorm_known_unit() {
        // input = [1,1,1,1], gamma = [1,1,1,1], eps = 0
        // sum_sq = 4, rms = sqrt(4/4 + 0) = 1.0
        // output = [1, 1, 1, 1]
        let input = [1.0_f32, 1.0, 1.0, 1.0];
        let gamma = [1.0_f32, 1.0, 1.0, 1.0];
        let mut output = [0.0_f32; 4];
        rmsnorm_scalar(&input, &gamma, 0.0, &mut output);
        for (i, &o) in output.iter().enumerate() {
            assert!((o - 1.0).abs() < 1e-6, "output[{i}] = {o}, expected 1.0");
        }
    }

    #[test]
    fn test_rmsnorm_with_gamma() {
        // input = [3, 4], gamma = [2, 0.5], eps = 0
        // sum_sq = 25, rms = sqrt(25/2) = 3.535533...
        let input = [3.0_f32, 4.0];
        let gamma = [2.0_f32, 0.5];
        let mut output = [0.0_f32; 2];
        rmsnorm_scalar(&input, &gamma, 0.0, &mut output);
        let rms = (25.0_f32 / 2.0).sqrt();
        assert!((output[0] - 3.0 / rms * 2.0).abs() < 1e-5);
        assert!((output[1] - 4.0 / rms * 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rmsnorm_zeros_with_eps() {
        // All-zero input with eps > 0 should not produce NaN
        let input = [0.0_f32, 0.0, 0.0, 0.0];
        let gamma = [1.0_f32, 1.0, 1.0, 1.0];
        let mut output = [0.0_f32; 4];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut output);
        for (i, &o) in output.iter().enumerate() {
            assert!(o.is_finite(), "output[{i}] must be finite, got {o}");
            assert!((o - 0.0).abs() < 1e-3, "output[{i}] should be ~0, got {o}");
        }
    }

    #[test]
    #[should_panic(expected = "input/gamma length mismatch")]
    fn test_rmsnorm_gamma_length_mismatch() {
        let input = [1.0_f32, 2.0];
        let gamma = [1.0_f32];
        let mut output = [0.0_f32; 2];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut output);
    }

    #[test]
    #[should_panic(expected = "input/output length mismatch")]
    fn test_rmsnorm_output_length_mismatch() {
        let input = [1.0_f32, 2.0];
        let gamma = [1.0_f32, 1.0];
        let mut output = [0.0_f32; 3];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut output);
    }

    #[test]
    #[should_panic(expected = "rmsnorm requires non-empty input")]
    fn test_rmsnorm_empty_input() {
        let input: [f32; 0] = [];
        let gamma: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut output);
    }

    // ── Property-based tests ─────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_rmsnorm_finite_output(
            v in proptest::collection::vec(-10.0_f32..10.0, 1..64)
        ) {
            let gamma = vec![1.0_f32; v.len()];
            let mut output = vec![0.0_f32; v.len()];
            rmsnorm_scalar(&v, &gamma, 1e-5, &mut output);
            for (i, &o) in output.iter().enumerate() {
                prop_assert!(o.is_finite(), "output[{i}] = {o} is not finite");
            }
        }

        #[test]
        fn prop_rmsnorm_scale_invariance(
            v in proptest::collection::vec(0.1_f32..10.0, 2..32),
            alpha in 0.1_f32..10.0
        ) {
            let gamma = vec![1.0_f32; v.len()];
            let mut out1 = vec![0.0_f32; v.len()];
            rmsnorm_scalar(&v, &gamma, 1e-8, &mut out1);

            let scaled: Vec<f32> = v.iter().map(|&x| x * alpha).collect();
            let mut out2 = vec![0.0_f32; v.len()];
            rmsnorm_scalar(&scaled, &gamma, 1e-8, &mut out2);

            // RMSNorm(alpha*x) = sign(alpha) * RMSNorm(x) for alpha > 0
            for i in 0..v.len() {
                prop_assert!(
                    (out1[i] - out2[i]).abs() < 1e-4,
                    "scale invariance violated at {i}: {} vs {}",
                    out1[i], out2[i]
                );
            }
        }

        #[test]
        fn prop_rmsnorm_unit_gamma_normalized_rms(
            v in proptest::collection::vec(0.1_f32..10.0, 2..32)
        ) {
            let gamma = vec![1.0_f32; v.len()];
            let mut output = vec![0.0_f32; v.len()];
            rmsnorm_scalar(&v, &gamma, 1e-8, &mut output);

            // RMS of output (with unit gamma) should be approximately 1
            let sum_sq: f32 = output.iter().map(|x| x * x).sum();
            let rms_out = (sum_sq / output.len() as f32).sqrt();
            prop_assert!(
                (rms_out - 1.0).abs() < 1e-3,
                "RMS of output = {rms_out}, expected ~1.0"
            );
        }
    }

    // ── AVX2 parity tests ────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_rmsnorm_avx2_basic() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let gamma = vec![1.0_f32; 16];
        let mut scalar_out = vec![0.0_f32; 16];
        let mut avx2_out = vec![0.0_f32; 16];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut scalar_out);
        unsafe { rmsnorm_avx2(&input, &gamma, 1e-5, &mut avx2_out) };
        assert_ulp_eq(&scalar_out, &avx2_out, 4);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_rmsnorm_avx2_non_multiple_of_8() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let gamma = [1.0_f32, 1.0, 1.0, 1.0, 1.0];
        let mut scalar_out = [0.0_f32; 5];
        let mut avx2_out = [0.0_f32; 5];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut scalar_out);
        unsafe { rmsnorm_avx2(&input, &gamma, 1e-5, &mut avx2_out) };
        assert_ulp_eq(&scalar_out, &avx2_out, 4);
    }

    #[cfg(target_arch = "x86_64")]
    proptest! {
        #[test]
        fn prop_rmsnorm_avx2_parity(
            v in proptest::collection::vec(-10.0_f32..10.0, 1..64)
        ) {
            if !is_x86_feature_detected!("avx2") {
                return Ok(());
            }
            let gamma = vec![1.0_f32; v.len()];
            let mut scalar_out = vec![0.0_f32; v.len()];
            let mut avx2_out = vec![0.0_f32; v.len()];
            rmsnorm_scalar(&v, &gamma, 1e-5, &mut scalar_out);
            unsafe { rmsnorm_avx2(&v, &gamma, 1e-5, &mut avx2_out) };
            assert_ulp_eq(&scalar_out, &avx2_out, 8);
        }
    }

    // ── PTX structural tests ─────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_ptx_version() {
        let ptx = rmsnorm_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
    }

    #[test]
    fn test_rmsnorm_ptx_target() {
        let ptx = rmsnorm_ptx();
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
    }

    #[test]
    fn test_rmsnorm_ptx_entry() {
        let ptx = rmsnorm_ptx();
        assert!(ptx.contains(".entry rmsnorm_kernel"), "missing entry point");
    }

    #[test]
    fn test_rmsnorm_ptx_ret() {
        let ptx = rmsnorm_ptx();
        assert!(ptx.contains("ret;"), "missing ret instruction");
    }

    #[test]
    fn test_rmsnorm_ptx_shared_memory() {
        let ptx = rmsnorm_ptx();
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
    }

    #[test]
    fn test_rmsnorm_ptx_warp_shuffle() {
        let ptx = rmsnorm_ptx();
        assert!(
            ptx.contains("shfl.sync"),
            "missing warp shuffle instructions"
        );
    }

    #[test]
    fn test_rmsnorm_ptx_rsqrt() {
        let ptx = rmsnorm_ptx();
        assert!(ptx.contains("rsqrt.approx.f32"), "missing rsqrt.approx.f32");
    }

    #[test]
    fn test_rmsnorm_ptx_bar_sync() {
        let ptx = rmsnorm_ptx();
        assert!(
            ptx.contains("bar.sync"),
            "missing bar.sync for block synchronization"
        );
    }

    #[test]
    fn test_rmsnorm_ptx_balanced_braces() {
        let ptx = rmsnorm_ptx();
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }
}
