//! SIMD backend selection and dot product implementations
//!
//! Provides hardware-accelerated dot products for int8 embeddings.
//! Supports AVX2, AVX-512, ARM NEON, and scalar fallback.
//!
//! # Safety
//!
//! This module uses `unsafe` exclusively for CPU SIMD intrinsics
//! (AVX2, AVX-512, NEON). All unsafe calls are guarded by runtime
//! feature detection (`is_x86_feature_detected!`) or compile-time
//! target gates (`#[cfg(target_arch)]`). No raw pointer arithmetic,
//! no transmutes, no FFI — only vendor-provided intrinsics.

// Library code - usage from examples and integration tests
/// SIMD backend selection (Jidoka auto-detection)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// AVX2: 256-bit vectors, 32 int8 ops/cycle
    Avx2,
    /// AVX-512: 512-bit vectors, 64 int8 ops/cycle
    Avx512,
    /// ARM NEON: 128-bit vectors, 16 int8 ops/cycle
    Neon,
    /// Scalar fallback (Jidoka degradation)
    Scalar,
}

impl SimdBackend {
    /// Auto-detect best available SIMD backend (Jidoka)
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let has_avx512 =
                is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
            let has_avx2 = is_x86_feature_detected!("avx2");
            return Self::from_x86_features(has_avx512, has_avx2);
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            return Self::Neon;
        }
        #[allow(unreachable_code)]
        Self::Scalar
    }

    /// Select backend from x86 feature flags (testable)
    #[cfg(target_arch = "x86_64")]
    pub fn from_x86_features(has_avx512: bool, has_avx2: bool) -> Self {
        if has_avx512 {
            Self::Avx512
        } else if has_avx2 {
            Self::Avx2
        } else {
            Self::Scalar
        }
    }

    /// Compute dot product of two i8 vectors
    ///
    /// Returns i32 to prevent overflow (127^2 x 4096 < i32::MAX)
    ///
    /// # Safety rationale
    ///
    /// Calls unsafe SIMD intrinsics guarded by `is_x86_feature_detected!`.
    #[allow(unsafe_code)]
    pub fn dot_i8(&self, a: &[i8], b: &[i8]) -> i32 {
        debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                if is_x86_feature_detected!("avx2") {
                    // Safety: AVX2 feature check above
                    return unsafe { dot_i8_avx2(a, b) };
                }
                dot_i8_scalar(a, b)
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                    // Safety: AVX-512 feature check above
                    return unsafe { dot_i8_avx512(a, b) };
                }
                dot_i8_scalar(a, b)
            }
            #[cfg(target_arch = "aarch64")]
            Self::Neon => {
                // SAFETY: NEON is always available on aarch64 (ARMv8+).
                // The dot_i8_neon function uses only NEON intrinsics which are
                // guaranteed to be supported on all aarch64 targets.
                unsafe { dot_i8_neon(a, b) }
            }
            _ => dot_i8_scalar(a, b),
        }
    }

    /// Compute dot product of f32 query with i8 document (rescoring)
    ///
    /// Used in stage 2 for 99%+ accuracy retention.
    pub fn dot_f32_i8(&self, query: &[f32], doc: &[i8], scale: f32) -> f32 {
        debug_assert_eq!(query.len(), doc.len(), "Vectors must have same length");

        // For rescoring, we use f32 accumulation for precision
        let mut sum: f32 = 0.0;
        for (&q, &d) in query.iter().zip(doc.iter()) {
            sum += q * (d as f32 * scale);
        }
        sum
    }
}

/// Scalar dot product fallback
pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x as i32) * (y as i32)).sum()
}

/// Scalar tail computation for SIMD remainder elements
///
/// Computes i8 dot product for elements starting at `start` index,
/// used by AVX2/AVX-512/NEON functions for their remainder loops.
#[inline]
fn dot_i8_scalar_tail(a: &[i8], b: &[i8], start: usize) -> i32 {
    a[start..].iter().zip(b[start..].iter()).map(|(&x, &y)| (x as i32) * (y as i32)).sum()
}

/// AVX2 SIMD dot product (x86_64)
// SAFETY: caller verifies AVX2 support via is_x86_feature_detected!, slices have equal length
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut sum = _mm256_setzero_si256();

    // Process 32 elements at a time
    let mut i = 0;
    while i + 32 <= n {
        let va = _mm256_loadu_si256(a[i..].as_ptr().cast::<__m256i>());
        let vb = _mm256_loadu_si256(b[i..].as_ptr().cast::<__m256i>());

        // Unpack to i16 and multiply
        let lo_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
        let lo_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
        let hi_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let hi_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        // madd: multiply adjacent pairs and sum to i32
        let prod_lo = _mm256_madd_epi16(lo_a, lo_b);
        let prod_hi = _mm256_madd_epi16(hi_a, hi_b);

        sum = _mm256_add_epi32(sum, prod_lo);
        sum = _mm256_add_epi32(sum, prod_hi);

        i += 32;
    }

    // Horizontal sum
    let sum128 = _mm_add_epi32(_mm256_extracti128_si256(sum, 0), _mm256_extracti128_si256(sum, 1));
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let result = _mm_cvtsi128_si32(sum32);

    // Handle remaining elements
    result + dot_i8_scalar_tail(a, b, i)
}

/// AVX-512 SIMD dot product (x86_64)
/// Note: AVX-512 support varies by CPU; falls back to scalar for remaining elements
// SAFETY: caller verifies AVX-512F+BW support via is_x86_feature_detected!, equal-length slices
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_i8_avx512(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let mut sum = _mm512_setzero_si512();

    // Process 64 elements at a time
    let mut i = 0;
    while i + 64 <= n {
        let va = _mm512_loadu_si512(a[i..].as_ptr().cast::<__m512i>());
        let vb = _mm512_loadu_si512(b[i..].as_ptr().cast::<__m512i>());

        // Extract 256-bit halves and process
        let lo_a = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 0));
        let lo_b = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 0));
        let hi_a = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1));
        let hi_b = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1));

        let prod_lo = _mm512_madd_epi16(lo_a, lo_b);
        let prod_hi = _mm512_madd_epi16(hi_a, hi_b);

        sum = _mm512_add_epi32(sum, prod_lo);
        sum = _mm512_add_epi32(sum, prod_hi);

        i += 64;
    }

    // Reduce 512-bit to scalar
    let result = _mm512_reduce_add_epi32(sum);

    // Handle remaining elements with scalar
    result + dot_i8_scalar_tail(a, b, i)
}

/// ARM NEON SIMD dot product (aarch64)
// SAFETY: NEON is always available on aarch64, slices have equal length (debug_assert)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let mut sum = vdupq_n_s32(0);

    // Process 16 elements at a time
    let mut i = 0;
    while i + 16 <= n {
        let va = vld1q_s8(a[i..].as_ptr());
        let vb = vld1q_s8(b[i..].as_ptr());

        // Multiply-accumulate low and high halves
        let lo_a = vmovl_s8(vget_low_s8(va));
        let lo_b = vmovl_s8(vget_low_s8(vb));
        let hi_a = vmovl_s8(vget_high_s8(va));
        let hi_b = vmovl_s8(vget_high_s8(vb));

        let prod_lo = vmull_s16(vget_low_s16(lo_a), vget_low_s16(lo_b));
        let prod_lo2 = vmull_s16(vget_high_s16(lo_a), vget_high_s16(lo_b));
        let prod_hi = vmull_s16(vget_low_s16(hi_a), vget_low_s16(hi_b));
        let prod_hi2 = vmull_s16(vget_high_s16(hi_a), vget_high_s16(hi_b));

        sum = vaddq_s32(sum, prod_lo);
        sum = vaddq_s32(sum, prod_lo2);
        sum = vaddq_s32(sum, prod_hi);
        sum = vaddq_s32(sum, prod_hi2);

        i += 16;
    }

    // Horizontal sum
    let result = vaddvq_s32(sum);

    // Handle remaining elements
    result + dot_i8_scalar_tail(a, b, i)
}
