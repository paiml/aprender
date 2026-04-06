//! AVX-512 reduction operations (dot, sum, max, min, argmax, argmin).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX-512 dot product.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm512_setzero_ps();

        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            acc = _mm512_fmadd_ps(va, vb, acc);
            i += 16;
        }

        let mut result = _mm512_reduce_add_ps(acc);
        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();
        result
    }
}

/// AVX-512 vector sum.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sum(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm512_setzero_ps();

        while i + 16 <= len {
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(a.as_ptr().add(i)));
            i += 16;
        }

        let mut result = _mm512_reduce_add_ps(acc);
        result += a[i..].iter().sum::<f32>();
        result
    }
}

/// AVX-512 vector max.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn max(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut vmax = _mm512_set1_ps(a[0]);

        while i + 16 <= len {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(a.as_ptr().add(i)));
            i += 16;
        }

        let mut result = _mm512_reduce_max_ps(vmax);
        for &val in &a[i..] {
            if val > result {
                result = val;
            }
        }
        result
    }
}

/// AVX-512 vector min.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn min(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut vmin = _mm512_set1_ps(a[0]);

        while i + 16 <= len {
            vmin = _mm512_min_ps(vmin, _mm512_loadu_ps(a.as_ptr().add(i)));
            i += 16;
        }

        let mut result = _mm512_reduce_min_ps(vmin);
        for &val in &a[i..] {
            if val < result {
                result = val;
            }
        }
        result
    }
}

/// AVX-512 argmax.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn argmax(a: &[f32]) -> usize {
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

/// AVX-512 argmin.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn argmin(a: &[f32]) -> usize {
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

/// Kahan sum (scalar implementation).
#[inline]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sum_kahan(a: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for &x in a {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}
