//! NEON arithmetic operations (add, sub, mul, div).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use std::arch::arm::*;

/// NEON vector addition.
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        vst1q_f32(
            result.as_mut_ptr().add(i),
            vaddq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i))),
        );
        i += 4;
    }
    for j in i..len {
        result[j] = a[j] + b[j];
    }
}

/// NEON vector subtraction.
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        vst1q_f32(
            result.as_mut_ptr().add(i),
            vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i))),
        );
        i += 4;
    }
    for j in i..len {
        result[j] = a[j] - b[j];
    }
}

/// NEON vector multiplication.
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        vst1q_f32(
            result.as_mut_ptr().add(i),
            vmulq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i))),
        );
        i += 4;
    }
    for j in i..len {
        result[j] = a[j] * b[j];
    }
}

/// NEON vector division.
#[inline]
#[target_feature(enable = "neon")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        #[cfg(target_arch = "aarch64")]
        {
            vst1q_f32(
                result.as_mut_ptr().add(i),
                vdivq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i))),
            );
        }
        #[cfg(target_arch = "arm")]
        {
            // ARM32 doesn't have vdivq_f32, use reciprocal approximation
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let recip = vrecpeq_f32(vb);
            let recip = vmulq_f32(recip, vrecpsq_f32(vb, recip));
            vst1q_f32(result.as_mut_ptr().add(i), vmulq_f32(va, recip));
        }
        i += 4;
    }
    for j in i..len {
        result[j] = a[j] / b[j];
    }
}
