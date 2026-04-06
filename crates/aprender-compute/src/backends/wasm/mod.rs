//! WebAssembly SIMD128 backend implementation
//!
//! This backend uses WebAssembly SIMD128 intrinsics for 128-bit SIMD operations.
//! SIMD128 is supported in modern browsers and wasm runtimes.
//!
//! # Performance
//!
//! Expected speedup: 4x for operations on f32 vectors (4 elements per register)
//! Similar performance characteristics to SSE2 and NEON.
//!
//! # Safety
//!
//! All WASM SIMD intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

mod ops;

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

use super::VectorBackend;

/// WebAssembly SIMD128 backend (128-bit SIMD)
pub struct WasmBackend;

impl VectorBackend for WasmBackend {
    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::add(a, b, result);
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::sub(a, b, result);
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::mul(a, b, result);
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::div(a, b, result);
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        ops::reductions::dot(a, b)
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum(a: &[f32]) -> f32 {
        ops::reductions::sum(a)
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn max(a: &[f32]) -> f32 {
        ops::reductions::max(a)
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn min(a: &[f32]) -> f32 {
        ops::reductions::min(a)
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmax(a: &[f32]) -> usize {
        ops::reductions::argmax(a)
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn argmin(a: &[f32]) -> usize {
        ops::reductions::argmin(a)
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        ops::reductions::sum_kahan(a)
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        Self::dot(a, a).sqrt()
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        let len = a.len();
        let mut i = 0;
        let mut acc = f32x4_splat(0.0);
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            acc = f32x4_add(acc, f32x4_abs(va));
            i += 4;
        }
        let mut result = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        for &val in &a[i..] {
            result += val.abs();
        }
        result
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        let len = a.len();
        let mut i = 0;
        let mut vmax = f32x4_splat(0.0);
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            vmax = f32x4_pmax(vmax, f32x4_abs(va));
            i += 4;
        }
        let mut result = f32x4_extract_lane::<0>(vmax)
            .max(f32x4_extract_lane::<1>(vmax))
            .max(f32x4_extract_lane::<2>(vmax))
            .max(f32x4_extract_lane::<3>(vmax));
        for &val in &a[i..] {
            let abs_val = val.abs();
            if abs_val > result {
                result = abs_val;
            }
        }
        result
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let scalar_vec = f32x4_splat(scalar);
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            v128_store(result.as_mut_ptr().add(i) as *mut v128, f32x4_mul(va, scalar_vec));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * scalar;
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            v128_store(result.as_mut_ptr().add(i) as *mut v128, f32x4_abs(va));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].abs();
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let min_vec = f32x4_splat(min_val);
        let max_vec = f32x4_splat(max_val);
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            v128_store(
                result.as_mut_ptr().add(i) as *mut v128,
                f32x4_pmin(f32x4_pmax(va, min_vec), max_vec),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].max(min_val).min(max_val);
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let t_vec = f32x4_splat(t);
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            let diff = f32x4_sub(vb, va);
            v128_store(
                result.as_mut_ptr().add(i) as *mut v128,
                f32x4_add(va, f32x4_mul(t_vec, diff)),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] + t * (b[j] - a[j]);
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            let vc = v128_load(c.as_ptr().add(i) as *const v128);
            v128_store(result.as_mut_ptr().add(i) as *mut v128, f32x4_add(f32x4_mul(va, vb), vc));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * b[j] + c[j];
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let zero = f32x4_splat(0.0);
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            v128_store(result.as_mut_ptr().add(i) as *mut v128, f32x4_pmax(va, zero));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].max(0.0);
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        // Polynomial approximation for exp
        let len = a.len();
        let mut i = 0;
        let ln2 = f32x4_splat(std::f32::consts::LN_2);
        let inv_ln2 = f32x4_splat(1.0 / std::f32::consts::LN_2);
        let one = f32x4_splat(1.0);
        let c2 = f32x4_splat(0.5);
        let c3 = f32x4_splat(0.166_666_67);
        let c4 = f32x4_splat(0.041_666_668);
        let c5 = f32x4_splat(0.008_333_334);
        while i + 4 <= len {
            let x = v128_load(a.as_ptr().add(i) as *const v128);
            let k = i32x4_trunc_sat_f32x4(f32x4_mul(x, inv_ln2));
            let kf = f32x4_convert_i32x4(k);
            let r = f32x4_sub(x, f32x4_mul(kf, ln2));
            let mut poly = f32x4_add(one, f32x4_mul(r, c5));
            poly = f32x4_add(one, f32x4_mul(r, f32x4_add(c4, f32x4_mul(r, poly))));
            poly = f32x4_add(one, f32x4_mul(r, f32x4_add(c3, f32x4_mul(r, poly))));
            poly = f32x4_add(one, f32x4_mul(r, f32x4_add(c2, f32x4_mul(r, poly))));
            poly = f32x4_add(one, f32x4_mul(r, poly));
            let exp_k = i32x4_shl(i32x4_add(k, i32x4_splat(127)), 23);
            v128_store(
                result.as_mut_ptr().add(i) as *mut v128,
                f32x4_mul(poly, v128_bitselect(exp_k, exp_k, exp_k)),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].exp();
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let len = a.len();
        for j in 0..len {
            result[j] = 1.0 / (1.0 + (-a[j]).exp());
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        for j in 0..a.len() {
            let x = a[j];
            let inner = 0.797_884_56 * (x + 0.044_715 * x * x * x);
            result[j] = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    #[target_feature(enable = "simd128")]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // swish(x) = x * sigmoid(x)
        for j in 0..a.len() {
            result[j] = a[j] / (1.0 + (-a[j]).exp());
        }
    }

    #[target_feature(enable = "simd128")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 9];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            WasmBackend::add(&a, &b, &mut result);
        }
        assert_eq!(result, vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_mul() {
        let a = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 9];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        unsafe {
            WasmBackend::mul(&a, &b, &mut result);
        }
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0]);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let result = unsafe { WasmBackend::dot(&a, &b) };
        assert!((result - 165.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_sum() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        // SAFETY: SIMD intrinsic call with valid inputs, target feature verified by caller
        let result = unsafe { WasmBackend::sum(&a) };
        assert!((result - 45.0).abs() < 1e-5);
    }
}
