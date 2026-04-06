//! WebAssembly SIMD128 arithmetic operations (add, sub, mul, div).

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// WASM SIMD128 vector addition.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        let va = v128_load(a.as_ptr().add(i) as *const v128);
        let vb = v128_load(b.as_ptr().add(i) as *const v128);
        v128_store(result.as_mut_ptr().add(i) as *mut v128, f32x4_add(va, vb));
        i += 4;
    }
    for j in i..len {
        result[j] = a[j] + b[j];
    }
}

/// WASM SIMD128 vector subtraction.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        let va = v128_load(a.as_ptr().add(i) as *const v128);
        let vb = v128_load(b.as_ptr().add(i) as *const v128);
        v128_store(result.as_mut_ptr().add(i) as *mut v128, f32x4_sub(va, vb));
        i += 4;
    }
    for j in i..len {
        result[j] = a[j] - b[j];
    }
}

/// WASM SIMD128 vector multiplication.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        let va = v128_load(a.as_ptr().add(i) as *const v128);
        let vb = v128_load(b.as_ptr().add(i) as *const v128);
        v128_store(result.as_mut_ptr().add(i) as *mut v128, f32x4_mul(va, vb));
        i += 4;
    }
    for j in i..len {
        result[j] = a[j] * b[j];
    }
}

/// WASM SIMD128 vector division.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        let va = v128_load(a.as_ptr().add(i) as *const v128);
        let vb = v128_load(b.as_ptr().add(i) as *const v128);
        v128_store(result.as_mut_ptr().add(i) as *mut v128, f32x4_div(va, vb));
        i += 4;
    }
    for j in i..len {
        result[j] = a[j] / b[j];
    }
}
