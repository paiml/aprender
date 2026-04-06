//! AVX-512 arithmetic operations (add, sub, mul, div).
//!
//! For large vectors (≥8192 elements), uses non-temporal stores to bypass
//! cache pollution and 4-way unrolling for instruction-level parallelism.
//! Based on: Drepper (2007) "What Every Programmer Should Know About Memory"
//! Section 6.1: non-temporal stores eliminate read-for-ownership traffic.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Threshold above which non-temporal stores are beneficial.
/// Below this, data fits in L2 and cache-through stores are faster.
const NT_THRESHOLD: usize = 8192;

/// AVX-512 vector addition.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        if len >= NT_THRESHOLD {
            add_nt(a, b, result);
        } else {
            add_cached(a, b, result);
        }
    }
}

/// Cached-store path for small vectors (fits in L2).
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn add_cached(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_add_ps(va, vb));
            i += 16;
        }
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }
}

/// Non-temporal store path for large vectors.
/// 4-way unrolled (64 f32 = 256 bytes per iteration = 4 cache lines).
/// Prefetches 8 cache lines ahead (~512 bytes).
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn add_nt(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let rp = result.as_mut_ptr();
        let mut i = 0;

        // 4-way unrolled non-temporal loop
        while i + 64 <= len {
            // Prefetch 8 cache lines ahead (512 bytes = 128 f32)
            _mm_prefetch(ap.add(i + 128).cast::<i8>(), _MM_HINT_T0);
            _mm_prefetch(bp.add(i + 128).cast::<i8>(), _MM_HINT_T0);

            let va0 = _mm512_loadu_ps(ap.add(i));
            let vb0 = _mm512_loadu_ps(bp.add(i));
            let va1 = _mm512_loadu_ps(ap.add(i + 16));
            let vb1 = _mm512_loadu_ps(bp.add(i + 16));
            let va2 = _mm512_loadu_ps(ap.add(i + 32));
            let vb2 = _mm512_loadu_ps(bp.add(i + 32));
            let va3 = _mm512_loadu_ps(ap.add(i + 48));
            let vb3 = _mm512_loadu_ps(bp.add(i + 48));

            _mm512_stream_ps(rp.add(i), _mm512_add_ps(va0, vb0));
            _mm512_stream_ps(rp.add(i + 16), _mm512_add_ps(va1, vb1));
            _mm512_stream_ps(rp.add(i + 32), _mm512_add_ps(va2, vb2));
            _mm512_stream_ps(rp.add(i + 48), _mm512_add_ps(va3, vb3));

            i += 64;
        }

        // Cleanup: remaining full SIMD widths
        while i + 16 <= len {
            let va = _mm512_loadu_ps(ap.add(i));
            let vb = _mm512_loadu_ps(bp.add(i));
            _mm512_stream_ps(rp.add(i), _mm512_add_ps(va, vb));
            i += 16;
        }

        // Memory fence after non-temporal stores
        _mm_sfence();

        // Scalar remainder
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }
}

/// AVX-512 vector subtraction.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        if len >= NT_THRESHOLD {
            sub_nt(a, b, result);
        } else {
            sub_cached(a, b, result);
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn sub_cached(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_sub_ps(va, vb));
            i += 16;
        }
        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn sub_nt(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let rp = result.as_mut_ptr();
        let mut i = 0;

        while i + 64 <= len {
            _mm_prefetch(ap.add(i + 128).cast::<i8>(), _MM_HINT_T0);
            _mm_prefetch(bp.add(i + 128).cast::<i8>(), _MM_HINT_T0);

            let va0 = _mm512_loadu_ps(ap.add(i));
            let vb0 = _mm512_loadu_ps(bp.add(i));
            let va1 = _mm512_loadu_ps(ap.add(i + 16));
            let vb1 = _mm512_loadu_ps(bp.add(i + 16));
            let va2 = _mm512_loadu_ps(ap.add(i + 32));
            let vb2 = _mm512_loadu_ps(bp.add(i + 32));
            let va3 = _mm512_loadu_ps(ap.add(i + 48));
            let vb3 = _mm512_loadu_ps(bp.add(i + 48));

            _mm512_stream_ps(rp.add(i), _mm512_sub_ps(va0, vb0));
            _mm512_stream_ps(rp.add(i + 16), _mm512_sub_ps(va1, vb1));
            _mm512_stream_ps(rp.add(i + 32), _mm512_sub_ps(va2, vb2));
            _mm512_stream_ps(rp.add(i + 48), _mm512_sub_ps(va3, vb3));

            i += 64;
        }

        while i + 16 <= len {
            let va = _mm512_loadu_ps(ap.add(i));
            let vb = _mm512_loadu_ps(bp.add(i));
            _mm512_stream_ps(rp.add(i), _mm512_sub_ps(va, vb));
            i += 16;
        }

        _mm_sfence();

        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }
}

/// AVX-512 vector multiplication.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        if len >= NT_THRESHOLD {
            mul_nt(a, b, result);
        } else {
            mul_cached(a, b, result);
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn mul_cached(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_mul_ps(va, vb));
            i += 16;
        }
        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn mul_nt(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let rp = result.as_mut_ptr();
        let mut i = 0;

        while i + 64 <= len {
            _mm_prefetch(ap.add(i + 128).cast::<i8>(), _MM_HINT_T0);
            _mm_prefetch(bp.add(i + 128).cast::<i8>(), _MM_HINT_T0);

            let va0 = _mm512_loadu_ps(ap.add(i));
            let vb0 = _mm512_loadu_ps(bp.add(i));
            let va1 = _mm512_loadu_ps(ap.add(i + 16));
            let vb1 = _mm512_loadu_ps(bp.add(i + 16));
            let va2 = _mm512_loadu_ps(ap.add(i + 32));
            let vb2 = _mm512_loadu_ps(bp.add(i + 32));
            let va3 = _mm512_loadu_ps(ap.add(i + 48));
            let vb3 = _mm512_loadu_ps(bp.add(i + 48));

            _mm512_stream_ps(rp.add(i), _mm512_mul_ps(va0, vb0));
            _mm512_stream_ps(rp.add(i + 16), _mm512_mul_ps(va1, vb1));
            _mm512_stream_ps(rp.add(i + 32), _mm512_mul_ps(va2, vb2));
            _mm512_stream_ps(rp.add(i + 48), _mm512_mul_ps(va3, vb3));

            i += 64;
        }

        while i + 16 <= len {
            let va = _mm512_loadu_ps(ap.add(i));
            let vb = _mm512_loadu_ps(bp.add(i));
            _mm512_stream_ps(rp.add(i), _mm512_mul_ps(va, vb));
            i += 16;
        }

        _mm_sfence();

        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }
}

/// AVX-512 vector division.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_div_ps(va, vb));
            i += 16;
        }
        for j in i..len {
            result[j] = a[j] / b[j];
        }
    }
}
