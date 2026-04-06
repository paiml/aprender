
/// Helper: Compute Q4×Q8 dot products for 8 blocks using pre-loaded Q8K
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn compute_q4_q8_dots_8blocks(
    qs_ptr: *const u8,
    q8_c0_lo: std::arch::x86_64::__m256i,
    q8_c0_hi: std::arch::x86_64::__m256i,
    q8_c1_lo: std::arch::x86_64::__m256i,
    q8_c1_hi: std::arch::x86_64::__m256i,
    q8_c2_lo: std::arch::x86_64::__m256i,
    q8_c2_hi: std::arch::x86_64::__m256i,
    q8_c3_lo: std::arch::x86_64::__m256i,
    q8_c3_hi: std::arch::x86_64::__m256i,
    nibble_mask: std::arch::x86_64::__m256i,
    ones_16: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // SAFETY: Caller must satisfy the documented preconditions
    // SAFETY: All intrinsics and pointer ops are unsafe, within already-unsafe fn
    unsafe {
        // Load Q4K data (128 bytes total for 256 values)
        let q4_c0 = _mm256_loadu_si256(qs_ptr.cast::<__m256i>());
        let q4_c1 = _mm256_loadu_si256(qs_ptr.add(32).cast::<__m256i>());
        let q4_c2 = _mm256_loadu_si256(qs_ptr.add(64).cast::<__m256i>());
        let q4_c3 = _mm256_loadu_si256(qs_ptr.add(96).cast::<__m256i>());

        // Extract nibbles and compute Q4×Q8 for each chunk
        let dot0 = q4_q8_chunk_dot(q4_c0, q8_c0_lo, q8_c0_hi, nibble_mask, ones_16);
        let dot1 = q4_q8_chunk_dot(q4_c1, q8_c1_lo, q8_c1_hi, nibble_mask, ones_16);
        let dot2 = q4_q8_chunk_dot(q4_c2, q8_c2_lo, q8_c2_hi, nibble_mask, ones_16);
        let dot3 = q4_q8_chunk_dot(q4_c3, q8_c3_lo, q8_c3_hi, nibble_mask, ones_16);

        // Pack 8 dot products into result (2 per chunk: lo nibble block, hi nibble block)
        let mut result = _mm256_setzero_si256();
        result = _mm256_insert_epi32(result, _mm_cvtsi128_si32(_mm256_castsi256_si128(dot0)), 0);
        result = _mm256_insert_epi32(
            result,
            _mm_extract_epi32(_mm256_castsi256_si128(dot0), 1),
            1,
        );
        result = _mm256_insert_epi32(result, _mm_cvtsi128_si32(_mm256_castsi256_si128(dot1)), 2);
        result = _mm256_insert_epi32(
            result,
            _mm_extract_epi32(_mm256_castsi256_si128(dot1), 1),
            3,
        );
        result = _mm256_insert_epi32(result, _mm_cvtsi128_si32(_mm256_castsi256_si128(dot2)), 4);
        result = _mm256_insert_epi32(
            result,
            _mm_extract_epi32(_mm256_castsi256_si128(dot2), 1),
            5,
        );
        result = _mm256_insert_epi32(result, _mm_cvtsi128_si32(_mm256_castsi256_si128(dot3)), 6);
        result = _mm256_insert_epi32(
            result,
            _mm_extract_epi32(_mm256_castsi256_si128(dot3), 1),
            7,
        );
        result
    }
}

/// Helper: Q4×Q8 dot product for one 64-value chunk (32 lo nibbles + 32 hi nibbles)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn q4_q8_chunk_dot(
    q4_packed: std::arch::x86_64::__m256i,
    q8_lo: std::arch::x86_64::__m256i,
    q8_hi: std::arch::x86_64::__m256i,
    nibble_mask: std::arch::x86_64::__m256i,
    ones_16: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // SAFETY: Caller must satisfy the documented preconditions
    // SAFETY: All intrinsics are unsafe, within already-unsafe fn
    unsafe {
        // Extract nibbles
        let q4_lo = _mm256_and_si256(q4_packed, nibble_mask);
        let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4_packed, 4), nibble_mask);

        // Q4 × Q8 products
        let prod_lo_i16 = _mm256_maddubs_epi16(q4_lo, q8_lo);
        let prod_hi_i16 = _mm256_maddubs_epi16(q4_hi, q8_hi);
        let prod_lo_i32 = _mm256_madd_epi16(prod_lo_i16, ones_16);
        let prod_hi_i32 = _mm256_madd_epi16(prod_hi_i16, ones_16);

        // Reduce each to single sum
        let sum_lo = hsum_epi32(prod_lo_i32);
        let sum_hi = hsum_epi32(prod_hi_i32);

        // Return [sum_lo, sum_hi, 0, 0, 0, 0, 0, 0]
        let mut result = _mm256_setzero_si256();
        result = _mm256_insert_epi32(result, sum_lo, 0);
        result = _mm256_insert_epi32(result, sum_hi, 1);
        result
    }
}

/// Helper: Horizontal sum of 8 i32 values to single i32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: Caller must satisfy the documented preconditions
unsafe fn hsum_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    // All intrinsics are unsafe and we're in an unsafe fn with target_feature
    let sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_10_10));
    _mm_cvtsi128_si32(sum32)
}

#[cfg(test)]
#[path = "fused_k_tests.rs"]
mod fused_k_tests;
