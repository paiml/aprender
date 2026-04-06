// PMAT-298: True AVX-512 Q4K x Q8K dot product.
// Uses 512-bit _mm512_maddubs_epi16 to process 2 chunks at once.
//
// Q4K data layout per super-block (128 qs bytes, 4 chunks of 32):
//   chunk c (32 bytes): lo nibbles -> Q8 block 2c, hi nibbles -> Q8 block 2c+1
//
// Q8K data layout (256 i8 values, 8 blocks of 32):
//   block b: q8[b*32 .. b*32+31]
//
// AVX-512 processes 2 chunks (64 bytes Q4K, 128 bytes Q8K) per iteration.

/// Compute Q4xQ8 dot products for 8 blocks using AVX-512 (512-bit).
///
/// Returns 8 i32 dot products, one per Q8 block (32-value sub-block).
/// Processes 2 Q4K chunks per 512-bit operation (2x vs AVX2).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn compute_q4_q8_dots_8blocks_avx512(
    qs_ptr: *const u8,
    q8_ptr: *const i8,
) -> [i32; 8] {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let nibble_mask = _mm512_set1_epi8(0x0F_i8);
    let ones_16 = _mm512_set1_epi16(1);

    let mut dots = [0i32; 8];

    // Process chunks 0+1 together (64 Q4K bytes, Q8 blocks 0-3)
    {
        // Load 64 Q4K packed bytes (chunks 0 and 1)
        let q4_01 = _mm512_loadu_si512(qs_ptr.cast::<__m512i>());

        // Extract lo nibbles (64 unsigned bytes)
        let q4_01_lo = _mm512_and_si512(q4_01, nibble_mask);
        // Extract hi nibbles
        let q4_01_hi = _mm512_and_si512(_mm512_srli_epi16(q4_01, 4), nibble_mask);

        // Load Q8K data for lo nibbles: blocks 0 and 2 (64 signed bytes)
        // chunk 0 lo -> Q8 block 0 (q8[0..31]), chunk 1 lo -> Q8 block 2 (q8[64..95])
        // These are NOT contiguous! Block 0 is at offset 0, block 2 is at offset 64.
        // Load as two 256-bit halves into one 512-bit register.
        let q8_b0 = _mm256_loadu_si256(q8_ptr.cast::<__m256i>()); // block 0
        let q8_b2 = _mm256_loadu_si256(q8_ptr.add(64).cast::<__m256i>()); // block 2
        let q8_lo = _mm512_inserti64x4(_mm512_castsi256_si512(q8_b0), q8_b2, 1);

        // Load Q8K for hi nibbles: blocks 1 and 3
        let q8_b1 = _mm256_loadu_si256(q8_ptr.add(32).cast::<__m256i>()); // block 1
        let q8_b3 = _mm256_loadu_si256(q8_ptr.add(96).cast::<__m256i>()); // block 3
        let q8_hi = _mm512_inserti64x4(_mm512_castsi256_si512(q8_b1), q8_b3, 1);

        // Q4 unsigned x Q8 signed: maddubs then madd to get i32 partial sums
        let prod_lo = _mm512_maddubs_epi16(q4_01_lo, q8_lo);
        let prod_hi = _mm512_maddubs_epi16(q4_01_hi, q8_hi);
        let sum_lo = _mm512_madd_epi16(prod_lo, ones_16);
        let sum_hi = _mm512_madd_epi16(prod_hi, ones_16);

        // sum_lo has 16 i32 values: lo[0..7] from chunk 0, lo[8..15] from chunk 1
        // Need horizontal sum of each 8-element half
        let lo_256_a = _mm512_castsi512_si256(sum_lo);
        let lo_256_b = _mm512_extracti64x4_epi64(sum_lo, 1);
        dots[0] = hsum_256(lo_256_a); // chunk 0 lo = block 0
        dots[2] = hsum_256(lo_256_b); // chunk 1 lo = block 2

        let hi_256_a = _mm512_castsi512_si256(sum_hi);
        let hi_256_b = _mm512_extracti64x4_epi64(sum_hi, 1);
        dots[1] = hsum_256(hi_256_a); // chunk 0 hi = block 1
        dots[3] = hsum_256(hi_256_b); // chunk 1 hi = block 3
    }

    // Process chunks 2+3 together (Q8 blocks 4-7)
    {
        let q4_23 = _mm512_loadu_si512(qs_ptr.add(64).cast::<__m512i>());

        let q4_23_lo = _mm512_and_si512(q4_23, nibble_mask);
        let q4_23_hi = _mm512_and_si512(_mm512_srli_epi16(q4_23, 4), nibble_mask);

        let q8_b4 = _mm256_loadu_si256(q8_ptr.add(128).cast::<__m256i>());
        let q8_b6 = _mm256_loadu_si256(q8_ptr.add(192).cast::<__m256i>());
        let q8_lo = _mm512_inserti64x4(_mm512_castsi256_si512(q8_b4), q8_b6, 1);

        let q8_b5 = _mm256_loadu_si256(q8_ptr.add(160).cast::<__m256i>());
        let q8_b7 = _mm256_loadu_si256(q8_ptr.add(224).cast::<__m256i>());
        let q8_hi = _mm512_inserti64x4(_mm512_castsi256_si512(q8_b5), q8_b7, 1);

        let prod_lo = _mm512_maddubs_epi16(q4_23_lo, q8_lo);
        let prod_hi = _mm512_maddubs_epi16(q4_23_hi, q8_hi);
        let sum_lo = _mm512_madd_epi16(prod_lo, ones_16);
        let sum_hi = _mm512_madd_epi16(prod_hi, ones_16);

        let lo_256_a = _mm512_castsi512_si256(sum_lo);
        let lo_256_b = _mm512_extracti64x4_epi64(sum_lo, 1);
        dots[4] = hsum_256(lo_256_a);
        dots[6] = hsum_256(lo_256_b);

        let hi_256_a = _mm512_castsi512_si256(sum_hi);
        let hi_256_b = _mm512_extracti64x4_epi64(sum_hi, 1);
        dots[5] = hsum_256(hi_256_a);
        dots[7] = hsum_256(hi_256_b);
    }

    dots
}

/// Compute Q8K block sums using 512-bit loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn compute_q8_sums_avx512(q8_ptr: *const i8) -> [i32; 8] {
    use std::arch::x86_64::*;

    let ones = _mm256_set1_epi16(1);
    let mut sums = [0i32; 8];

    // Each Q8 block is 32 signed bytes. Sum via maddubs(ones_u8, q8_signed).
    // maddubs treats first arg as unsigned, second as signed.
    let ones_u8 = _mm256_set1_epi8(1);

    for block in 0..8 {
        let q8 = _mm256_loadu_si256(q8_ptr.add(block * 32).cast::<__m256i>());
        // maddubs(1, q8) = sum of pairs as i16
        let pair_sums = _mm256_maddubs_epi16(ones_u8, q8);
        // madd(pair_sums, 1) = sum of pairs of i16 as i32
        let quad_sums = _mm256_madd_epi16(pair_sums, ones);
        sums[block] = hsum_256(quad_sums);
    }

    sums
}

/// Horizontal sum of 8 i32 values in a 256-bit register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn hsum_256(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b00_00_10_10));
    _mm_cvtsi128_si32(sum32)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_detection() {
        let has = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
        eprintln!("AVX-512 F+BW: {has}");
    }
}
