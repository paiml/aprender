//! AVX2-accelerated LZ4 implementation.
//!
//! This module provides AVX2 (256-bit SIMD) optimized LZ4 compression
//! and decompression for `x86_64` CPUs.

use crate::{Result, PAGE_SIZE};

/// AVX2-accelerated LZ4 compression.
///
/// # Safety
///
/// Caller must ensure AVX2 is available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compress_avx2(input: &[u8; PAGE_SIZE]) -> Result<Vec<u8>> {
    // Compression uses scalar - AVX2 doesn't help much for hash lookups
    super::compress::compress(input)
}

/// AVX2-accelerated LZ4 decompression.
///
/// # Safety
///
/// Caller must ensure AVX2 is available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn decompress_avx2(input: &[u8], output: &mut [u8; PAGE_SIZE]) -> Result<usize> {
    unsafe { decompress_avx2_impl(input, output) }
}

/// Fast decompression - just call scalar with `target_feature` for auto-vectorization.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decompress_avx2_impl(input: &[u8], output: &mut [u8; PAGE_SIZE]) -> Result<usize> {
    // The scalar implementation is already well-optimized.
    // With target_feature(enable = "avx2"), the compiler will use AVX2
    // instructions for memcpy-like operations automatically.
    super::decompress::decompress(input, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_available() {
        if std::arch::is_x86_feature_detected!("avx2") {
            let input = [0u8; PAGE_SIZE];
            let result = unsafe { compress_avx2(&input) };
            assert!(result.is_ok());
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_roundtrip() {
        if std::arch::is_x86_feature_detected!("avx2") {
            let mut input = [0u8; PAGE_SIZE];
            for (i, b) in input.iter_mut().enumerate() {
                *b = (i % 16) as u8;
            }

            let compressed = unsafe { compress_avx2(&input) }.unwrap();
            let mut output = [0u8; PAGE_SIZE];
            let len = unsafe { decompress_avx2(&compressed, &mut output) }.unwrap();

            assert_eq!(len, PAGE_SIZE);
            assert_eq!(input, output);
        }
    }
}
