//! SSE2 backend implementation (x86_64 baseline SIMD)
//!
//! This backend uses SSE2 intrinsics for 128-bit SIMD operations.
//! SSE2 is available on all x86_64 CPUs as a baseline requirement.
//!
//! # Performance
//!
//! Expected speedup: 4x for operations on aligned f32 vectors (4 elements per register)
//!
//! # Safety
//!
//! All SSE2 intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

mod ops;

use super::VectorBackend;

/// SSE2 backend (128-bit SIMD for x86_64)
pub struct Sse2Backend;

impl VectorBackend for Sse2Backend {
    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            ops::arithmetic::add(a, b, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            ops::arithmetic::sub(a, b, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            ops::arithmetic::mul(a, b, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            ops::arithmetic::div(a, b, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        unsafe { ops::reductions::dot(a, b) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum(a: &[f32]) -> f32 {
        unsafe { ops::reductions::sum(a) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn max(a: &[f32]) -> f32 {
        unsafe { ops::reductions::max(a) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn min(a: &[f32]) -> f32 {
        unsafe { ops::reductions::min(a) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmax(a: &[f32]) -> usize {
        unsafe { ops::reductions::argmax(a) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmin(a: &[f32]) -> usize {
        unsafe { ops::reductions::argmin(a) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        unsafe { ops::reductions::sum_kahan(a) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
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
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        unsafe { ops::elementwise::norm_l1(a) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        unsafe { ops::elementwise::norm_linf(a) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        unsafe {
            ops::elementwise::scale(a, scalar, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::elementwise::abs(a, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        unsafe {
            ops::elementwise::clamp(a, min_val, max_val, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        unsafe {
            ops::elementwise::lerp(a, b, t, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        unsafe {
            ops::elementwise::fma(a, b, c, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::elementwise::relu(a, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::activations::exp(a, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::activations::sigmoid(a, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::activations::gelu(a, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::activations::swish(a, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::activations::tanh(a, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::elementwise::sqrt(a, result);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        unsafe {
            ops::elementwise::recip(a, result);
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

#[cfg(test)]
mod tests;
