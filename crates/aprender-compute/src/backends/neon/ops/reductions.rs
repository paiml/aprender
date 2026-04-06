//! NEON reduction operations (dot, sum, max, min, argmax, argmin).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use std::arch::arm::*;

/// NEON dot product.
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0);
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        #[cfg(target_arch = "aarch64")]
        {
            acc = vfmaq_f32(acc, va, vb);
        }
        #[cfg(target_arch = "arm")]
        {
            acc = vmlaq_f32(acc, va, vb);
        }
        i += 4;
    }
    let mut result = horizontal_sum(acc);
    for j in i..len {
        result += a[j] * b[j];
    }
    result
}

/// NEON vector sum.
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn sum(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = vdupq_n_f32(0.0);
    while i + 4 <= len {
        acc = vaddq_f32(acc, vld1q_f32(a.as_ptr().add(i)));
        i += 4;
    }
    let mut result = horizontal_sum(acc);
    for j in i..len {
        result += a[j];
    }
    result
}

/// NEON vector max.
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn max(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut vmax = vdupq_n_f32(a[0]);
    while i + 4 <= len {
        vmax = vmaxq_f32(vmax, vld1q_f32(a.as_ptr().add(i)));
        i += 4;
    }
    let mut result = horizontal_max(vmax);
    for j in i..len {
        if a[j] > result {
            result = a[j];
        }
    }
    result
}

/// NEON vector min.
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn min(a: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut vmin = vdupq_n_f32(a[0]);
    while i + 4 <= len {
        vmin = vminq_f32(vmin, vld1q_f32(a.as_ptr().add(i)));
        i += 4;
    }
    let mut result = horizontal_min(vmin);
    for j in i..len {
        if a[j] < result {
            result = a[j];
        }
    }
    result
}

/// NEON argmax (scalar fallback).
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn argmax(a: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = a[0];
    for (i, &v) in a.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

/// NEON argmin (scalar fallback).
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn argmin(a: &[f32]) -> usize {
    let mut min_idx = 0;
    let mut min_val = a[0];
    for (i, &v) in a.iter().enumerate() {
        if v < min_val {
            min_val = v;
            min_idx = i;
        }
    }
    min_idx
}

/// Kahan sum (scalar implementation).
#[inline]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn sum_kahan(a: &[f32]) -> f32 {
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

// Helper: horizontal sum of float32x4
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
unsafe fn horizontal_sum(v: float32x4_t) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        vaddvq_f32(v)
    }
    #[cfg(target_arch = "arm")]
    {
        let pair = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
        let pair = vpadd_f32(pair, pair);
        vget_lane_f32::<0>(pair)
    }
}

// Helper: horizontal max of float32x4
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
unsafe fn horizontal_max(v: float32x4_t) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        vmaxvq_f32(v)
    }
    #[cfg(target_arch = "arm")]
    {
        let pair = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
        let pair = vpmax_f32(pair, pair);
        vget_lane_f32::<0>(pair)
    }
}

// Helper: horizontal min of float32x4
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
unsafe fn horizontal_min(v: float32x4_t) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        vminvq_f32(v)
    }
    #[cfg(target_arch = "arm")]
    {
        let pair = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
        let pair = vpmin_f32(pair, pair);
        vget_lane_f32::<0>(pair)
    }
}
