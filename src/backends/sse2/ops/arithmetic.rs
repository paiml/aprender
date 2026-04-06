//! SSE2 arithmetic operations (add, sub, mul, div).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE2 vector addition.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_add_ps(va, vb));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }
}

/// SSE2 vector subtraction.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_sub_ps(va, vb));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }
}

/// SSE2 vector multiplication.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_mul_ps(va, vb));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }
}

/// SSE2 vector division.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_div_ps(va, vb));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] / b[j];
        }
    }
}
