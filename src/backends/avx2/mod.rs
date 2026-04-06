//! AVX2 backend implementation (x86_64 advanced SIMD)
//!
//! This backend uses AVX2 intrinsics for 256-bit SIMD operations with FMA.
//! AVX2 is available on Intel Haswell (2013+) and AMD Excavator (2015+) CPUs.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::VectorBackend;

mod ops;

/// AVX2 backend (256-bit SIMD for x86_64)
pub struct Avx2Backend;

impl VectorBackend for Avx2Backend {
    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe { ops::arithmetic::add(a, b, result) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe { ops::arithmetic::sub(a, b, result) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe { ops::arithmetic::mul(a, b, result) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe { ops::arithmetic::div(a, b, result) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        unsafe { ops::reductions::dot(a, b) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum(a: &[f32]) -> f32 {
        unsafe { ops::reductions::sum(a) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn max(a: &[f32]) -> f32 {
        unsafe { ops::reductions::max(a) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn min(a: &[f32]) -> f32 {
        unsafe { ops::reductions::min(a) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmax(a: &[f32]) -> usize {
        unsafe { ops::reductions::argmax(a) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmin(a: &[f32]) -> usize {
        unsafe { ops::reductions::argmin(a) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        unsafe { ops::reductions::sum_kahan(a) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        unsafe {
            if a.is_empty() {
                return 0.0;
            }
            Self::dot(a, a).sqrt()
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        unsafe {
            if a.is_empty() {
                return 0.0;
            }
            let len = a.len();
            let mut i = 0;
            let mut acc = _mm256_setzero_ps();
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let abs_va = _mm256_and_ps(va, sign_mask);
                acc = _mm256_add_ps(acc, abs_va);
                i += 8;
            }

            let mut result = {
                let sum_halves =
                    _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
                let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
                let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
                _mm_cvtss_f32(temp)
            };

            for &val in &a[i..] {
                result += val.abs();
            }
            result
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        unsafe {
            if a.is_empty() {
                return 0.0;
            }
            let len = a.len();
            let mut i = 0;
            let mut max_vec = _mm256_setzero_ps();
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let abs_va = _mm256_and_ps(va, sign_mask);
                max_vec = _mm256_max_ps(max_vec, abs_va);
                i += 8;
            }

            let mut result = {
                let max_halves =
                    _mm_max_ps(_mm256_castps256_ps128(max_vec), _mm256_extractf128_ps(max_vec, 1));
                let temp = _mm_max_ps(max_halves, _mm_movehl_ps(max_halves, max_halves));
                let temp = _mm_max_ss(temp, _mm_shuffle_ps(temp, temp, 1));
                _mm_cvtss_f32(temp)
            };

            for &val in &a[i..] {
                let abs_val = val.abs();
                if abs_val > result {
                    result = abs_val;
                }
            }
            result
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let scalar_vec = _mm256_set1_ps(scalar);

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vresult = _mm256_mul_ps(va, scalar_vec);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            while i < len {
                result[i] = a[i] * scalar;
                i += 1;
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vresult = _mm256_and_ps(va, sign_mask);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j].abs();
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let vmin = _mm256_set1_ps(min_val);
            let vmax = _mm256_set1_ps(max_val);

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vresult = _mm256_min_ps(_mm256_max_ps(va, vmin), vmax);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j].clamp(min_val, max_val);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let vt = _mm256_set1_ps(t);
            let v1_minus_t = _mm256_set1_ps(1.0 - t);

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vresult = _mm256_fmadd_ps(vb, vt, _mm256_mul_ps(va, v1_minus_t));
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j] * (1.0 - t) + b[j] * t;
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vc = _mm256_loadu_ps(c.as_ptr().add(i));
                let vresult = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j] * b[j] + c[j];
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let vzero = _mm256_setzero_ps();

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vresult = _mm256_max_ps(va, vzero);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j].max(0.0);
            }
        }
    }

    // Delegate transcendental functions to scalar backend
    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::exp(a, result) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::sigmoid(a, result) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::gelu(a, result) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::swish(a, result) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::tanh(a, result) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vresult = _mm256_sqrt_ps(va);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j].sqrt();
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let vone = _mm256_set1_ps(1.0);

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vresult = _mm256_div_ps(vone, va);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = 1.0 / a[j];
            }
        }
    }

    // Delegate log functions to scalar backend
    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn ln(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::ln(a, result) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn log2(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::log2(a, result) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn log10(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::log10(a, result) }
    }

    // Delegate trig functions to scalar backend
    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sin(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::sin(a, result) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn cos(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::cos(a, result) }
    }

    #[inline]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn tan(a: &[f32], result: &mut [f32]) {
        unsafe { super::scalar::ScalarBackend::tan(a, result) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn floor(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vresult = _mm256_floor_ps(va);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j].floor();
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn ceil(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vresult = _mm256_ceil_ps(va);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j].ceil();
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn round(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;

            // Round ties away from zero to match Rust's f32::round()
            let half = _mm256_set1_ps(0.5);
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x8000_0000));
            let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let sign = _mm256_and_ps(va, sign_mask);
                let abs_val = _mm256_and_ps(va, abs_mask);
                let shifted = _mm256_add_ps(abs_val, half);
                let rounded_abs = _mm256_floor_ps(shifted);
                let vresult = _mm256_or_ps(rounded_abs, sign);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
                i += 8;
            }

            for j in i..len {
                result[j] = a[j].round();
            }
        }
    }
}
