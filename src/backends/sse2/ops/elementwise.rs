//! SSE2 elementwise operations (scale, abs, clamp, lerp, fma, relu, sqrt, recip, norms).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE2 L1 norm.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn norm_l1(a: &[f32]) -> f32 {
    unsafe {
        if a.is_empty() {
            return 0.0;
        }
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm_setzero_ps();
        let sign_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
        while i + 4 <= len {
            acc = _mm_add_ps(acc, _mm_and_ps(_mm_loadu_ps(a.as_ptr().add(i)), sign_mask));
            i += 4;
        }
        let mut result = {
            let temp = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };
        for &val in &a[i..] {
            result += val.abs();
        }
        result
    }
}

/// SSE2 L-infinity norm.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn norm_linf(a: &[f32]) -> f32 {
    unsafe {
        if a.is_empty() {
            return 0.0;
        }
        let len = a.len();
        let mut i = 0;
        let mut max_vec = _mm_setzero_ps();
        let sign_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            max_vec = _mm_max_ps(max_vec, _mm_and_ps(va, sign_mask));
            i += 4;
        }
        let mut result = {
            let temp = _mm_max_ps(max_vec, _mm_movehl_ps(max_vec, max_vec));
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

/// SSE2 scalar multiply.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let scalar_vec = _mm_set1_ps(scalar);
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(i)), scalar_vec),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * scalar;
        }
    }
}

/// SSE2 absolute value.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn abs(a: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let sign_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_and_ps(_mm_loadu_ps(a.as_ptr().add(i)), sign_mask),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].abs();
        }
    }
}

/// SSE2 clamp.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let min_vec = _mm_set1_ps(min_val);
        let max_vec = _mm_set1_ps(max_val);
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_min_ps(_mm_max_ps(va, min_vec), max_vec));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].max(min_val).min(max_val);
        }
    }
}

/// SSE2 linear interpolation.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let t_vec = _mm_set1_ps(t);
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_add_ps(va, _mm_mul_ps(t_vec, _mm_sub_ps(vb, va))),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] + t * (b[j] - a[j]);
        }
    }
}

/// SSE2 fused multiply-add (emulated, no FMA instruction set).
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vc = _mm_loadu_ps(c.as_ptr().add(i));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_add_ps(_mm_mul_ps(va, vb), vc));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * b[j] + c[j];
        }
    }
}

/// SSE2 ReLU activation.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn relu(a: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let zero = _mm_setzero_ps();
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_max_ps(_mm_loadu_ps(a.as_ptr().add(i)), zero),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].max(0.0);
        }
    }
}

/// SSE2 square root.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_sqrt_ps(_mm_loadu_ps(a.as_ptr().add(i))));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].sqrt();
        }
    }
}

/// SSE2 reciprocal.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn recip(a: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let one = _mm_set1_ps(1.0);
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_div_ps(one, _mm_loadu_ps(a.as_ptr().add(i))),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].recip();
        }
    }
}
