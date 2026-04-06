//! AVX2 reduction operations (dot, sum, max, min, argmax, argmin).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::backends::VectorBackend;

/// AVX2 dot product with 4-accumulator unrolling for ILP.
#[inline]
#[target_feature(enable = "avx2,fma")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        while i + 32 <= len {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
            let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);

            i += 32;
        }

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
            i += 8;
        }

        let acc01 = _mm256_add_ps(acc0, acc1);
        let acc23 = _mm256_add_ps(acc2, acc3);
        let acc = _mm256_add_ps(acc01, acc23);

        let mut result = {
            let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();
        result
    }
}

/// AVX2 vector sum.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sum(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm256_setzero_ps();

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            acc = _mm256_add_ps(acc, va);
            i += 8;
        }

        let mut result = {
            let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        result += a[i..].iter().sum::<f32>();
        result
    }
}

/// AVX2 vector max.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn max(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut vmax = _mm256_set1_ps(a[0]);

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            vmax = _mm256_max_ps(vmax, va);
            i += 8;
        }

        let mut result = {
            let max_halves =
                _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
            let temp = _mm_max_ps(max_halves, _mm_movehl_ps(max_halves, max_halves));
            let temp = _mm_max_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        for &val in &a[i..] {
            if val > result {
                result = val;
            }
        }
        result
    }
}

/// AVX2 vector min.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn min(a: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let mut i = 0;
        let mut vmin = _mm256_set1_ps(a[0]);

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            vmin = _mm256_min_ps(vmin, va);
            i += 8;
        }

        let mut result = {
            let min_halves =
                _mm_min_ps(_mm256_castps256_ps128(vmin), _mm256_extractf128_ps(vmin, 1));
            let temp = _mm_min_ps(min_halves, _mm_movehl_ps(min_halves, min_halves));
            let temp = _mm_min_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        for &val in &a[i..] {
            if val < result {
                result = val;
            }
        }
        result
    }
}

/// AVX2 argmax.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn argmax(a: &[f32]) -> usize {
    unsafe {
        let len = a.len();
        let mut max_idx: usize = 0;
        let mut max_val = a[0];
        let mut i = 0;

        let mut vmax = _mm256_set1_ps(a[0]);
        let mut vidx_max = _mm256_setzero_ps();
        let vidx_inc = _mm256_set1_ps(8.0);
        let mut vcurrent_idx = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let mask = _mm256_cmp_ps(va, vmax, _CMP_GT_OQ);
            vmax = _mm256_blendv_ps(vmax, va, mask);
            vidx_max = _mm256_blendv_ps(vidx_max, vcurrent_idx, mask);
            vcurrent_idx = _mm256_add_ps(vcurrent_idx, vidx_inc);
            i += 8;
        }

        // Extract max from vector
        let mut vals = [0.0f32; 8];
        let mut idxs = [0.0f32; 8];
        _mm256_storeu_ps(vals.as_mut_ptr(), vmax);
        _mm256_storeu_ps(idxs.as_mut_ptr(), vidx_max);

        for j in 0..8 {
            if vals[j] > max_val {
                max_val = vals[j];
                max_idx = idxs[j] as usize;
            }
        }

        // Check remaining elements
        for (j, &val) in a[i..].iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i + j;
            }
        }

        max_idx
    }
}

/// AVX2 argmin.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn argmin(a: &[f32]) -> usize {
    unsafe {
        let len = a.len();
        let mut min_idx: usize = 0;
        let mut min_val = a[0];
        let mut i = 0;

        let mut vmin = _mm256_set1_ps(a[0]);
        let mut vidx_min = _mm256_setzero_ps();
        let vidx_inc = _mm256_set1_ps(8.0);
        let mut vcurrent_idx = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let mask = _mm256_cmp_ps(va, vmin, _CMP_LT_OQ);
            vmin = _mm256_blendv_ps(vmin, va, mask);
            vidx_min = _mm256_blendv_ps(vidx_min, vcurrent_idx, mask);
            vcurrent_idx = _mm256_add_ps(vcurrent_idx, vidx_inc);
            i += 8;
        }

        let mut vals = [0.0f32; 8];
        let mut idxs = [0.0f32; 8];
        _mm256_storeu_ps(vals.as_mut_ptr(), vmin);
        _mm256_storeu_ps(idxs.as_mut_ptr(), vidx_min);

        for j in 0..8 {
            if vals[j] < min_val {
                min_val = vals[j];
                min_idx = idxs[j] as usize;
            }
        }

        for (j, &val) in a[i..].iter().enumerate() {
            if val < min_val {
                min_val = val;
                min_idx = i + j;
            }
        }

        min_idx
    }
}

/// Kahan sum for numerical stability (delegates to scalar).
#[inline]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sum_kahan(a: &[f32]) -> f32 {
    unsafe { crate::backends::scalar::ScalarBackend::sum_kahan(a) }
}
