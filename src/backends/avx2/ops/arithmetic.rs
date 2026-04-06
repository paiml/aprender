//! AVX2 arithmetic operations (add, sub, mul, div).
//!
//! For large vectors (≥8192 elements), uses non-temporal stores (`_mm256_stream_ps`)
//! to bypass cache and 4-way unrolling for ILP.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const NT_THRESHOLD: usize = 8192;

/// AVX2 vector addition.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        // NT stores require 32-byte alignment (#242 SIGSEGV fix)
        let rp_aligned = (result.as_ptr() as usize) % 32 == 0;
        if len >= NT_THRESHOLD && rp_aligned {
            add_nt(a, b, result);
        } else {
            add_cached(a, b, result);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_cached(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(result.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
            i += 8;
        }
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn add_nt(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let rp = result.as_mut_ptr();
        let mut i = 0;

        // 4-way unrolled NT loop (32 f32 = 128 bytes = 2 cache lines per iter)
        while i + 32 <= len {
            _mm_prefetch(ap.add(i + 64).cast::<i8>(), _MM_HINT_T0);
            _mm_prefetch(bp.add(i + 64).cast::<i8>(), _MM_HINT_T0);

            let va0 = _mm256_loadu_ps(ap.add(i));
            let vb0 = _mm256_loadu_ps(bp.add(i));
            let va1 = _mm256_loadu_ps(ap.add(i + 8));
            let vb1 = _mm256_loadu_ps(bp.add(i + 8));
            let va2 = _mm256_loadu_ps(ap.add(i + 16));
            let vb2 = _mm256_loadu_ps(bp.add(i + 16));
            let va3 = _mm256_loadu_ps(ap.add(i + 24));
            let vb3 = _mm256_loadu_ps(bp.add(i + 24));

            _mm256_stream_ps(rp.add(i), _mm256_add_ps(va0, vb0));
            _mm256_stream_ps(rp.add(i + 8), _mm256_add_ps(va1, vb1));
            _mm256_stream_ps(rp.add(i + 16), _mm256_add_ps(va2, vb2));
            _mm256_stream_ps(rp.add(i + 24), _mm256_add_ps(va3, vb3));

            i += 32;
        }

        while i + 8 <= len {
            let va = _mm256_loadu_ps(ap.add(i));
            let vb = _mm256_loadu_ps(bp.add(i));
            _mm256_stream_ps(rp.add(i), _mm256_add_ps(va, vb));
            i += 8;
        }

        _mm_sfence();

        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }
}

/// AVX2 vector subtraction.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let rp = result.as_mut_ptr();
        let mut i = 0;

        // NT stores require 32-byte alignment (#242 SIGSEGV fix)
        let rp_aligned = (rp as usize) % 32 == 0;
        if len >= NT_THRESHOLD && rp_aligned {
            while i + 32 <= len {
                _mm_prefetch(ap.add(i + 64).cast::<i8>(), _MM_HINT_T0);
                _mm_prefetch(bp.add(i + 64).cast::<i8>(), _MM_HINT_T0);

                _mm256_stream_ps(
                    rp.add(i),
                    _mm256_sub_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i))),
                );
                _mm256_stream_ps(
                    rp.add(i + 8),
                    _mm256_sub_ps(_mm256_loadu_ps(ap.add(i + 8)), _mm256_loadu_ps(bp.add(i + 8))),
                );
                _mm256_stream_ps(
                    rp.add(i + 16),
                    _mm256_sub_ps(_mm256_loadu_ps(ap.add(i + 16)), _mm256_loadu_ps(bp.add(i + 16))),
                );
                _mm256_stream_ps(
                    rp.add(i + 24),
                    _mm256_sub_ps(_mm256_loadu_ps(ap.add(i + 24)), _mm256_loadu_ps(bp.add(i + 24))),
                );
                i += 32;
            }
            while i + 8 <= len {
                _mm256_stream_ps(
                    rp.add(i),
                    _mm256_sub_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i))),
                );
                i += 8;
            }
            _mm_sfence();
        } else {
            while i + 8 <= len {
                _mm256_storeu_ps(
                    rp.add(i),
                    _mm256_sub_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i))),
                );
                i += 8;
            }
        }

        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }
}

/// AVX2 vector multiplication.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let rp = result.as_mut_ptr();
        let mut i = 0;

        // NT stores (_mm256_stream_ps) require 32-byte aligned output.
        // Vec<f32> default alignment is 4 bytes — only use NT path if aligned.
        // Fix for #242 SIGSEGV: General Protection Fault from unaligned stream_ps.
        let rp_aligned = (rp as usize) % 32 == 0;
        if len >= NT_THRESHOLD && rp_aligned {
            while i + 32 <= len {
                _mm_prefetch(ap.add(i + 64).cast::<i8>(), _MM_HINT_T0);
                _mm_prefetch(bp.add(i + 64).cast::<i8>(), _MM_HINT_T0);

                _mm256_stream_ps(
                    rp.add(i),
                    _mm256_mul_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i))),
                );
                _mm256_stream_ps(
                    rp.add(i + 8),
                    _mm256_mul_ps(_mm256_loadu_ps(ap.add(i + 8)), _mm256_loadu_ps(bp.add(i + 8))),
                );
                _mm256_stream_ps(
                    rp.add(i + 16),
                    _mm256_mul_ps(_mm256_loadu_ps(ap.add(i + 16)), _mm256_loadu_ps(bp.add(i + 16))),
                );
                _mm256_stream_ps(
                    rp.add(i + 24),
                    _mm256_mul_ps(_mm256_loadu_ps(ap.add(i + 24)), _mm256_loadu_ps(bp.add(i + 24))),
                );
                i += 32;
            }
            while i + 8 <= len {
                _mm256_stream_ps(
                    rp.add(i),
                    _mm256_mul_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i))),
                );
                i += 8;
            }
            _mm_sfence();
        } else {
            while i + 8 <= len {
                _mm256_storeu_ps(
                    rp.add(i),
                    _mm256_mul_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i))),
                );
                i += 8;
            }
        }

        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }
}

/// AVX2 vector division.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(result.as_mut_ptr().add(i), _mm256_div_ps(va, vb));
            i += 8;
        }
        for j in i..len {
            result[j] = a[j] / b[j];
        }
    }
}
