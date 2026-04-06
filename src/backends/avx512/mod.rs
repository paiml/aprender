//! AVX-512 backend implementation (x86_64 advanced SIMD)
//!
//! This backend uses AVX-512 intrinsics for 512-bit SIMD operations.
//! AVX-512 is available on Intel Skylake-X/Sapphire Rapids (2017+) and AMD Zen 4 (2022+) CPUs.
//!
//! # Performance
//!
//! Expected speedup: 16x for operations on f32 vectors (16 elements per register)
//! This provides 2x improvement over AVX2 (8 elements) and ~16x over scalar.
//!
//! # Safety
//!
//! All AVX-512 intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

mod ops;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::VectorBackend;

/// AVX-512 backend (512-bit SIMD for x86_64)
pub struct Avx512Backend;

impl VectorBackend for Avx512Backend {
    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            ops::arithmetic::add(a, b, result);
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            ops::arithmetic::sub(a, b, result);
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            ops::arithmetic::mul(a, b, result);
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            ops::arithmetic::div(a, b, result);
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        unsafe { ops::reductions::dot(a, b) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum(a: &[f32]) -> f32 {
        unsafe { ops::reductions::sum(a) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn max(a: &[f32]) -> f32 {
        unsafe { ops::reductions::max(a) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn min(a: &[f32]) -> f32 {
        unsafe { ops::reductions::min(a) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmax(a: &[f32]) -> usize {
        unsafe { ops::reductions::argmax(a) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmin(a: &[f32]) -> usize {
        unsafe { ops::reductions::argmin(a) }
    }

    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        unsafe { ops::reductions::sum_kahan(a) }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        unsafe {
            if a.is_empty() {
                return 0.0;
            }
            let len = a.len();
            let mut i = 0;
            let mut acc = _mm512_setzero_ps();
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                acc = _mm512_add_ps(acc, _mm512_mul_ps(va, va));
                i += 16;
            }
            let mut sum_sq = _mm512_reduce_add_ps(acc);
            for &val in &a[i..] {
                sum_sq += val * val;
            }
            sum_sq.sqrt()
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
            let mut acc = _mm512_setzero_ps();
            while i + 16 <= len {
                acc = _mm512_add_ps(
                    acc,
                    _mm512_and_ps(_mm512_loadu_ps(a.as_ptr().add(i)), sign_mask),
                );
                i += 16;
            }
            let mut result = _mm512_reduce_add_ps(acc);
            for &val in &a[i..] {
                result += val.abs();
            }
            result
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
            let mut max_vec = _mm512_setzero_ps();
            while i + 16 <= len {
                max_vec = _mm512_max_ps(
                    max_vec,
                    _mm512_and_ps(_mm512_loadu_ps(a.as_ptr().add(i)), sign_mask),
                );
                i += 16;
            }
            let mut result = _mm512_reduce_max_ps(max_vec);
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
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let scalar_vec = _mm512_set1_ps(scalar);
            while i + 16 <= len {
                _mm512_storeu_ps(
                    result.as_mut_ptr().add(i),
                    _mm512_mul_ps(_mm512_loadu_ps(a.as_ptr().add(i)), scalar_vec),
                );
                i += 16;
            }
            for j in i..len {
                result[j] = a[j] * scalar;
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
            while i + 16 <= len {
                _mm512_storeu_ps(
                    result.as_mut_ptr().add(i),
                    _mm512_and_ps(_mm512_loadu_ps(a.as_ptr().add(i)), sign_mask),
                );
                i += 16;
            }
            for j in i..len {
                result[j] = a[j].abs();
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let min_vec = _mm512_set1_ps(min_val);
            let max_vec = _mm512_set1_ps(max_val);
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                _mm512_storeu_ps(
                    result.as_mut_ptr().add(i),
                    _mm512_min_ps(_mm512_max_ps(va, min_vec), max_vec),
                );
                i += 16;
            }
            for j in i..len {
                result[j] = a[j].max(min_val).min(max_val);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let t_vec = _mm512_set1_ps(t);
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                _mm512_storeu_ps(
                    result.as_mut_ptr().add(i),
                    _mm512_fmadd_ps(t_vec, _mm512_sub_ps(vb, va), va),
                );
                i += 16;
            }
            for j in i..len {
                result[j] = a[j] + t * (b[j] - a[j]);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                let vc = _mm512_loadu_ps(c.as_ptr().add(i));
                _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_fmadd_ps(va, vb, vc));
                i += 16;
            }
            for j in i..len {
                result[j] = a[j] * b[j] + c[j];
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let ap = a.as_ptr();
            let rp = result.as_mut_ptr();
            let mut i = 0;
            let zero = _mm512_setzero_ps();

            if len >= 8192 {
                // Non-temporal path: 4-way unrolled with prefetch
                while i + 64 <= len {
                    _mm_prefetch(ap.add(i + 128).cast::<i8>(), _MM_HINT_T0);

                    _mm512_stream_ps(rp.add(i), _mm512_max_ps(_mm512_loadu_ps(ap.add(i)), zero));
                    _mm512_stream_ps(
                        rp.add(i + 16),
                        _mm512_max_ps(_mm512_loadu_ps(ap.add(i + 16)), zero),
                    );
                    _mm512_stream_ps(
                        rp.add(i + 32),
                        _mm512_max_ps(_mm512_loadu_ps(ap.add(i + 32)), zero),
                    );
                    _mm512_stream_ps(
                        rp.add(i + 48),
                        _mm512_max_ps(_mm512_loadu_ps(ap.add(i + 48)), zero),
                    );

                    i += 64;
                }
                while i + 16 <= len {
                    _mm512_stream_ps(rp.add(i), _mm512_max_ps(_mm512_loadu_ps(ap.add(i)), zero));
                    i += 16;
                }
                _mm_sfence();
            } else {
                while i + 16 <= len {
                    _mm512_storeu_ps(rp.add(i), _mm512_max_ps(_mm512_loadu_ps(ap.add(i)), zero));
                    i += 16;
                }
            }
            for j in i..len {
                result[j] = a[j].max(0.0);
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
            let inv_ln2 = _mm512_set1_ps(1.0 / std::f32::consts::LN_2);
            let one = _mm512_set1_ps(1.0);
            let c2 = _mm512_set1_ps(0.5);
            let c3 = _mm512_set1_ps(0.166_666_67);
            let c4 = _mm512_set1_ps(0.041_666_668);
            let c5 = _mm512_set1_ps(0.008_333_334);
            while i + 16 <= len {
                let x = _mm512_loadu_ps(a.as_ptr().add(i));
                let k = _mm512_cvtps_epi32(_mm512_mul_ps(x, inv_ln2));
                let kf = _mm512_cvtepi32_ps(k);
                let r = _mm512_sub_ps(x, _mm512_mul_ps(kf, ln2));
                let mut poly = _mm512_fmadd_ps(r, c5, one);
                poly = _mm512_fmadd_ps(r, _mm512_fmadd_ps(r, poly, c4), one);
                poly = _mm512_fmadd_ps(r, _mm512_fmadd_ps(r, poly, c3), one);
                poly = _mm512_fmadd_ps(r, _mm512_fmadd_ps(r, poly, c2), one);
                poly = _mm512_fmadd_ps(r, poly, one);
                let exp_k = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_add_epi32(k, _mm512_set1_epi32(127)),
                    23,
                ));
                _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_mul_ps(poly, exp_k));
                i += 16;
            }
            for j in i..len {
                result[j] = a[j].exp();
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let len = a.len();
        for j in 0..len {
            result[j] = 1.0 / (1.0 + (-a[j]).exp());
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() {
            let x = a[j];
            let inner = 0.797_884_56 * (x + 0.044_715 * x * x * x);
            result[j] = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() {
            result[j] = a[j] / (1.0 + (-a[j]).exp());
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() {
            result[j] = a[j].tanh();
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            while i + 16 <= len {
                _mm512_storeu_ps(
                    result.as_mut_ptr().add(i),
                    _mm512_sqrt_ps(_mm512_loadu_ps(a.as_ptr().add(i))),
                );
                i += 16;
            }
            for j in i..len {
                result[j] = a[j].sqrt();
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let mut i = 0;
            let one = _mm512_set1_ps(1.0);
            while i + 16 <= len {
                _mm512_storeu_ps(
                    result.as_mut_ptr().add(i),
                    _mm512_div_ps(one, _mm512_loadu_ps(a.as_ptr().add(i))),
                );
                i += 16;
            }
            for j in i..len {
                result[j] = a[j].recip();
            }
        }
    }

    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn ln(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::ln(a, result);
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn log2(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::log2(a, result);
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn log10(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::log10(a, result);
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sin(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::sin(a, result);
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn cos(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::cos(a, result);
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn tan(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::tan(a, result);
        }
    }

    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn floor(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::floor(a, result);
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn ceil(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::ceil(a, result);
        }
    }
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn round(a: &[f32], result: &mut [f32]) {
        unsafe {
            super::scalar::ScalarBackend::round(a, result);
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
