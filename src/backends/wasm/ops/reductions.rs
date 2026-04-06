//! WebAssembly SIMD128 reduction operations (dot, sum, max, min, argmax, argmin).

use crate::backends::VectorBackend;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// WASM SIMD128 dot product.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = f32x4_splat(0.0);

    while i + 4 <= len {
        let va = v128_load(a.as_ptr().add(i) as *const v128);
        let vb = v128_load(b.as_ptr().add(i) as *const v128);
        acc = f32x4_add(acc, f32x4_mul(va, vb));
        i += 4;
    }

    let mut result = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();
    result
}

/// WASM SIMD128 vector sum.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn sum(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = f32x4_splat(0.0);

    while i + 4 <= len {
        acc = f32x4_add(acc, v128_load(a.as_ptr().add(i) as *const v128));
        i += 4;
    }

    let mut result = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    result += a[i..].iter().sum::<f32>();
    result
}

/// WASM SIMD128 vector max.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn max(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut vmax = f32x4_splat(a[0]);

    while i + 4 <= len {
        vmax = f32x4_pmax(vmax, v128_load(a.as_ptr().add(i) as *const v128));
        i += 4;
    }

    let mut result = f32x4_extract_lane::<0>(vmax)
        .max(f32x4_extract_lane::<1>(vmax))
        .max(f32x4_extract_lane::<2>(vmax))
        .max(f32x4_extract_lane::<3>(vmax));
    for &val in &a[i..] {
        if val > result {
            result = val;
        }
    }
    result
}

/// WASM SIMD128 vector min.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn min(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut vmin = f32x4_splat(a[0]);

    while i + 4 <= len {
        vmin = f32x4_pmin(vmin, v128_load(a.as_ptr().add(i) as *const v128));
        i += 4;
    }

    let mut result = f32x4_extract_lane::<0>(vmin)
        .min(f32x4_extract_lane::<1>(vmin))
        .min(f32x4_extract_lane::<2>(vmin))
        .min(f32x4_extract_lane::<3>(vmin));
    for &val in &a[i..] {
        if val < result {
            result = val;
        }
    }
    result
}

/// WASM SIMD128 argmax.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn argmax(a: &[f32]) -> usize {
    let mut max_idx: usize = 0;
    let mut max_val = a[0];
    for (i, &val) in a.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

/// WASM SIMD128 argmin.
#[target_feature(enable = "simd128")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn argmin(a: &[f32]) -> usize {
    let mut min_idx: usize = 0;
    let mut min_val = a[0];
    for (i, &val) in a.iter().enumerate() {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }
    min_idx
}

/// Kahan sum (delegates to scalar).
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn sum_kahan(a: &[f32]) -> f32 {
    crate::backends::scalar::ScalarBackend::sum_kahan(a)
}
