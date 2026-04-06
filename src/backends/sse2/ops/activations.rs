//! SSE2 activation functions (exp, sigmoid, gelu, swish, tanh).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SSE2 exp approximation helper (polynomial range reduction).
///
/// # Safety
///
/// Caller must ensure SSE2 is available on the current CPU.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller verifies SSE2 support, input slices meet alignment/length requirements
pub(crate) unsafe fn exp_approx_sse2(x: __m128) -> __m128 {
    let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
    let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
    let one = _mm_set1_ps(1.0);
    let c2 = _mm_set1_ps(0.5);
    let c3 = _mm_set1_ps(0.166_666_67);
    let c4 = _mm_set1_ps(0.041_666_668);
    let c5 = _mm_set1_ps(0.008_333_334);
    let k = _mm_cvtps_epi32(_mm_mul_ps(x, inv_ln2));
    let kf = _mm_cvtepi32_ps(k);
    let r = _mm_sub_ps(x, _mm_mul_ps(kf, ln2));
    let mut poly = _mm_add_ps(one, _mm_mul_ps(r, c5));
    poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c4, _mm_mul_ps(r, poly))));
    poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, poly))));
    poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, poly))));
    poly = _mm_add_ps(one, _mm_mul_ps(r, poly));
    let exp_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23));
    _mm_mul_ps(poly, exp_k)
}

/// SSE2 exp (element-wise).
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn exp(a: &[f32], result: &mut [f32]) {
    unsafe {
        // Polynomial approximation for exp - range reduction + polynomial
        let len = a.len();
        let mut i = 0;
        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
        let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
        let c1 = _mm_set1_ps(1.0);
        let c2 = _mm_set1_ps(0.5);
        let c3 = _mm_set1_ps(0.166_666_67);
        let c4 = _mm_set1_ps(0.041_666_668);
        let c5 = _mm_set1_ps(0.008_333_334);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let k = _mm_cvtps_epi32(_mm_mul_ps(x, inv_ln2));
            let kf = _mm_cvtepi32_ps(k);
            let r = _mm_sub_ps(x, _mm_mul_ps(kf, ln2));
            let mut poly = _mm_add_ps(c1, _mm_mul_ps(r, c5));
            poly = _mm_add_ps(c1, _mm_mul_ps(r, _mm_add_ps(c4, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(c1, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(c1, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(c1, _mm_mul_ps(r, poly));
            let exp_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_mul_ps(poly, exp_k));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].exp();
        }
    }
}

/// SSE2 sigmoid activation.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
    unsafe {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let len = a.len();
        let mut i = 0;
        let one = _mm_set1_ps(1.0);
        let neg_one = _mm_set1_ps(-1.0);
        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
        let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
        let c2 = _mm_set1_ps(0.5);
        let c3 = _mm_set1_ps(0.166_666_67);
        let c4 = _mm_set1_ps(0.041_666_668);
        let c5 = _mm_set1_ps(0.008_333_334);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let neg_x = _mm_mul_ps(x, neg_one);
            let k = _mm_cvtps_epi32(_mm_mul_ps(neg_x, inv_ln2));
            let kf = _mm_cvtepi32_ps(k);
            let r = _mm_sub_ps(neg_x, _mm_mul_ps(kf, ln2));
            let mut poly = _mm_add_ps(one, _mm_mul_ps(r, c5));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c4, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, poly));
            let exp_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23));
            let exp_neg_x = _mm_mul_ps(poly, exp_k);
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_div_ps(one, _mm_add_ps(one, exp_neg_x)));
            i += 4;
        }
        for j in i..len {
            result[j] = 1.0 / (1.0 + (-a[j]).exp());
        }
    }
}

/// SSE2 GELU activation.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn gelu(a: &[f32], result: &mut [f32]) {
    unsafe {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let len = a.len();
        let mut i = 0;
        let half = _mm_set1_ps(0.5);
        let one = _mm_set1_ps(1.0);
        let sqrt_2_pi = _mm_set1_ps(0.797_884_56);
        let coeff = _mm_set1_ps(0.044_715);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let x3 = _mm_mul_ps(_mm_mul_ps(x, x), x);
            let inner = _mm_mul_ps(sqrt_2_pi, _mm_add_ps(x, _mm_mul_ps(coeff, x3)));
            // tanh approximation: (e^2x - 1) / (e^2x + 1)
            let two_inner = _mm_add_ps(inner, inner);
            let exp_2x = exp_approx_sse2(two_inner);
            let tanh_val = _mm_div_ps(_mm_sub_ps(exp_2x, one), _mm_add_ps(exp_2x, one));
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_mul_ps(half, _mm_mul_ps(x, _mm_add_ps(one, tanh_val))),
            );
            i += 4;
        }
        for j in i..len {
            let x = a[j];
            result[j] = 0.5
                * x
                * (1.0 + ((0.797_884_56 * (x + 0.044_715 * x * x * x)) as f64).tanh() as f32);
        }
    }
}

/// SSE2 swish activation.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn swish(a: &[f32], result: &mut [f32]) {
    unsafe {
        // swish(x) = x * sigmoid(x)
        let len = a.len();
        let mut i = 0;
        let one = _mm_set1_ps(1.0);
        let neg_one = _mm_set1_ps(-1.0);
        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
        let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
        let c2 = _mm_set1_ps(0.5);
        let c3 = _mm_set1_ps(0.166_666_67);
        let c4 = _mm_set1_ps(0.041_666_668);
        let c5 = _mm_set1_ps(0.008_333_334);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let neg_x = _mm_mul_ps(x, neg_one);
            let k = _mm_cvtps_epi32(_mm_mul_ps(neg_x, inv_ln2));
            let kf = _mm_cvtepi32_ps(k);
            let r = _mm_sub_ps(neg_x, _mm_mul_ps(kf, ln2));
            let mut poly = _mm_add_ps(one, _mm_mul_ps(r, c5));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c4, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, poly));
            let exp_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23));
            let exp_neg_x = _mm_mul_ps(poly, exp_k);
            let sigmoid = _mm_div_ps(one, _mm_add_ps(one, exp_neg_x));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_mul_ps(x, sigmoid));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] / (1.0 + (-a[j]).exp());
        }
    }
}

/// SSE2 tanh activation.
#[inline]
#[target_feature(enable = "sse2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn tanh(a: &[f32], result: &mut [f32]) {
    unsafe {
        // tanh(x) = (e^2x - 1) / (e^2x + 1)
        let len = a.len();
        let mut i = 0;
        let one = _mm_set1_ps(1.0);
        let two = _mm_set1_ps(2.0);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let exp_2x = exp_approx_sse2(_mm_mul_ps(two, x));
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_div_ps(_mm_sub_ps(exp_2x, one), _mm_add_ps(exp_2x, one)),
            );
            i += 4;
        }
        for j in i..len {
            let exp_2x = (2.0 * a[j]).exp();
            result[j] = (exp_2x - 1.0) / (exp_2x + 1.0);
        }
    }
}
