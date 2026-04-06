//! SSE2 reduction operations (dot, sum, max, min, argmax, argmin).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::backends::VectorBackend;

/// SSE2 dot product.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm_setzero_ps();

        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
            i += 4;
        }

        let mut result = horizontal_sum_ps(acc);
        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();
        result
    }
}

/// SSE2 vector sum.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sum(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm_setzero_ps();

        while i + 4 <= len {
            acc = _mm_add_ps(acc, _mm_loadu_ps(a.as_ptr().add(i)));
            i += 4;
        }

        let mut result = horizontal_sum_ps(acc);
        result += a[i..].iter().sum::<f32>();
        result
    }
}

/// SSE2 vector max.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn max(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut vmax = _mm_set1_ps(a[0]);

        while i + 4 <= len {
            vmax = _mm_max_ps(vmax, _mm_loadu_ps(a.as_ptr().add(i)));
            i += 4;
        }

        let mut result = horizontal_max_ps(vmax);
        for &val in &a[i..] {
            if val > result {
                result = val;
            }
        }
        result
    }
}

/// SSE2 vector min.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn min(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut vmin = _mm_set1_ps(a[0]);

        while i + 4 <= len {
            vmin = _mm_min_ps(vmin, _mm_loadu_ps(a.as_ptr().add(i)));
            i += 4;
        }

        let mut result = horizontal_min_ps(vmin);
        for &val in &a[i..] {
            if val < result {
                result = val;
            }
        }
        result
    }
}

/// SSE2 argmax.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn argmax(a: &[f32]) -> usize {
    let _len = a.len();
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

/// SSE2 argmin.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn argmin(a: &[f32]) -> usize {
    let _len = a.len();
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
#[inline]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sum_kahan(a: &[f32]) -> f32 {
    unsafe { crate::backends::scalar::ScalarBackend::sum_kahan(a) }
}

// Helper: horizontal sum of 4 floats
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
unsafe fn horizontal_sum_ps(v: __m128) -> f32 {
    let temp = _mm_add_ps(v, _mm_movehl_ps(v, v));
    let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    _mm_cvtss_f32(temp)
}

// Helper: horizontal max of 4 floats
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
unsafe fn horizontal_max_ps(v: __m128) -> f32 {
    let temp = _mm_max_ps(v, _mm_movehl_ps(v, v));
    let temp = _mm_max_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    _mm_cvtss_f32(temp)
}

// Helper: horizontal min of 4 floats
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
unsafe fn horizontal_min_ps(v: __m128) -> f32 {
    let temp = _mm_min_ps(v, _mm_movehl_ps(v, v));
    let temp = _mm_min_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    _mm_cvtss_f32(temp)
}
