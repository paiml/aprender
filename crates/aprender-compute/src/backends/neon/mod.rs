//! ARM NEON backend implementation (ARM 128-bit SIMD)
//!
//! This backend uses ARM NEON intrinsics for 128-bit SIMD operations.
//! NEON is available on ARMv7 (32-bit) and ARMv8/AArch64 (64-bit) CPUs.
//!
//! # Performance
//!
//! Expected speedup: 4x for operations on f32 vectors (4 elements per register)
//! Similar performance characteristics to SSE2 on x86_64.

mod ops;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use std::arch::arm::*;

use super::VectorBackend;

/// ARM NEON backend (128-bit SIMD)
pub struct NeonBackend;

impl VectorBackend for NeonBackend {
    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::add(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::sub(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::mul(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::div(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        ops::reductions::dot(a, b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum(a: &[f32]) -> f32 {
        ops::reductions::sum(a)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn max(a: &[f32]) -> f32 {
        ops::reductions::max(a)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn min(a: &[f32]) -> f32 {
        ops::reductions::min(a)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmax(a: &[f32]) -> usize {
        ops::reductions::argmax(a)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmin(a: &[f32]) -> usize {
        ops::reductions::argmin(a)
    }

    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        ops::reductions::sum_kahan(a)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        let len = a.len();
        let mut i = 0;
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            acc = vaddq_f32(acc, vmulq_f32(va, va));
            i += 4;
        }
        #[cfg(target_arch = "aarch64")]
        let mut sum_sq = vaddvq_f32(acc);
        #[cfg(target_arch = "arm")]
        let mut sum_sq = {
            let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            let pair = vpadd_f32(pair, pair);
            vget_lane_f32::<0>(pair)
        };
        for j in i..len {
            sum_sq += a[j] * a[j];
        }
        sum_sq.sqrt()
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= len {
            acc = vaddq_f32(acc, vabsq_f32(vld1q_f32(a.as_ptr().add(i))));
            i += 4;
        }
        #[cfg(target_arch = "aarch64")]
        let mut result = vaddvq_f32(acc);
        #[cfg(target_arch = "arm")]
        let mut result = {
            let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            let pair = vpadd_f32(pair, pair);
            vget_lane_f32::<0>(pair)
        };
        for j in i..len {
            result += a[j].abs();
        }
        result
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let mut vmax = vdupq_n_f32(0.0);
        while i + 4 <= len {
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(a.as_ptr().add(i))));
            i += 4;
        }
        let mut result = vmaxvq_f32(vmax);
        for j in i..len {
            let abs_val = a[j].abs();
            if abs_val > result {
                result = abs_val;
            }
        }
        result
    }

    #[cfg(target_arch = "arm")]
    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let mut vmax = vdupq_n_f32(0.0);
        while i + 4 <= len {
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(a.as_ptr().add(i))));
            i += 4;
        }
        let pair = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        let pair = vpmax_f32(pair, pair);
        let mut result = vget_lane_f32::<0>(pair);
        for j in i..len {
            let abs_val = a[j].abs();
            if abs_val > result {
                result = abs_val;
            }
        }
        result
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let scalar_vec = vdupq_n_f32(scalar);
        while i + 4 <= len {
            vst1q_f32(
                result.as_mut_ptr().add(i),
                vmulq_f32(vld1q_f32(a.as_ptr().add(i)), scalar_vec),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * scalar;
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            vst1q_f32(result.as_mut_ptr().add(i), vabsq_f32(vld1q_f32(a.as_ptr().add(i))));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].abs();
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let min_vec = vdupq_n_f32(min_val);
        let max_vec = vdupq_n_f32(max_val);
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            vst1q_f32(result.as_mut_ptr().add(i), vminq_f32(vmaxq_f32(va, min_vec), max_vec));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].max(min_val).min(max_val);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let t_vec = vdupq_n_f32(t);
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            vst1q_f32(result.as_mut_ptr().add(i), vfmaq_f32(va, t_vec, vsubq_f32(vb, va)));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] + t * (b[j] - a[j]);
        }
    }

    #[cfg(target_arch = "arm")]
    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let t_vec = vdupq_n_f32(t);
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            vst1q_f32(result.as_mut_ptr().add(i), vmlaq_f32(va, t_vec, vsubq_f32(vb, va)));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] + t * (b[j] - a[j]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vld1q_f32(c.as_ptr().add(i));
            vst1q_f32(result.as_mut_ptr().add(i), vfmaq_f32(vc, va, vb));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * b[j] + c[j];
        }
    }

    #[cfg(target_arch = "arm")]
    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vld1q_f32(c.as_ptr().add(i));
            vst1q_f32(result.as_mut_ptr().add(i), vmlaq_f32(vc, va, vb));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * b[j] + c[j];
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let zero = vdupq_n_f32(0.0);
        while i + 4 <= len {
            vst1q_f32(result.as_mut_ptr().add(i), vmaxq_f32(vld1q_f32(a.as_ptr().add(i)), zero));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].max(0.0);
        }
    }

    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::exp(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() {
            result[j] = 1.0 / (1.0 + (-a[j]).exp());
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() {
            let x = a[j];
            let inner = 0.797_884_56 * (x + 0.044_715 * x * x * x);
            result[j] = 0.5 * x * (1.0 + inner.tanh());
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() {
            result[j] = a[j] / (1.0 + (-a[j]).exp());
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() {
            result[j] = a[j].tanh();
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::sqrt(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::recip(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn ln(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::ln(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn log2(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::log2(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn log10(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::log10(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sin(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::sin(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn cos(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::cos(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn tan(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::tan(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn floor(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::floor(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn ceil(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::ceil(a, result);
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn round(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::round(a, result);
    }
}

#[cfg(all(test, any(target_arch = "aarch64", target_arch = "arm")))]
mod tests {
    use super::*;

    #[test]
    fn test_neon_add() {
        let a = vec![1.0; 16];
        let b = vec![2.0; 16];
        let mut result = vec![0.0; 16];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            NeonBackend::add(&a, &b, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_neon_sub() {
        let a = vec![5.0; 16];
        let b = vec![2.0; 16];
        let mut result = vec![0.0; 16];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            NeonBackend::sub(&a, &b, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_neon_mul() {
        let a = vec![2.0; 16];
        let b = vec![3.0; 16];
        let mut result = vec![0.0; 16];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            NeonBackend::mul(&a, &b, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_neon_div() {
        let a = vec![6.0; 16];
        let b = vec![2.0; 16];
        let mut result = vec![0.0; 16];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            NeonBackend::div(&a, &b, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-4));
    }

    #[test]
    fn test_neon_dot() {
        let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let result = unsafe { NeonBackend::dot(&a, &b) };
        let expected: f32 = (1..=16).map(|i| (i * i) as f32).sum();
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn test_neon_sum() {
        let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let result = unsafe { NeonBackend::sum(&a) };
        assert!((result - 136.0).abs() < 1e-3);
    }
}
