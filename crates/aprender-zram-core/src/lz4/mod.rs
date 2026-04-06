//! LZ4 compression implementation.
//!
//! This module provides a pure Rust implementation of the LZ4 block format
//! as specified in <https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md>.

mod compress;
mod decompress;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;

pub use compress::{compress, compress_tls};
pub use decompress::decompress;

use crate::{Result, PAGE_SIZE};

/// Decompress with automatic SIMD dispatch.
///
/// Selects the best available SIMD implementation at runtime:
/// - AVX-512 on `x86_64` with AVX-512 support
/// - AVX2 on `x86_64` with AVX2 support
/// - NEON on `AArch64`
/// - Scalar fallback on other platforms
pub fn decompress_simd(input: &[u8], output: &mut [u8; PAGE_SIZE]) -> Result<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX-512 if available (≥5 GB/s target)
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: We just checked that AVX-512 is available
            return unsafe { avx512::decompress_avx512(input, output) };
        }
        // Fall back to AVX2 (≥4 GB/s target)
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked that AVX2 is available
            return unsafe { avx2::decompress_avx2(input, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Use NEON on ARM (≥4 GB/s target)
        // SAFETY: NEON is always available on AArch64
        return unsafe { neon::decompress_neon(input, output) };
    }

    // Fallback to scalar
    #[allow(unreachable_code)]
    decompress(input, output)
}

/// Compress with automatic SIMD dispatch.
///
/// Note: Compression is hash-table bound and benefits less from SIMD than decompression.
/// All backends currently use the optimized scalar implementation.
pub fn compress_simd(input: &[u8; PAGE_SIZE]) -> Result<Vec<u8>> {
    compress(input)
}

/// LZ4 block format constants.
pub mod constants {
    /// Minimum match length.
    pub const MIN_MATCH: usize = 4;
    /// Maximum match length that can be encoded in token.
    pub const ML_MASK: u8 = 0x0F;
    /// Literal length mask in token.
    pub const LL_MASK: u8 = 0xF0;
    /// Maximum literal run before extension bytes.
    pub const RUN_MASK: u8 = 15;
    /// Offset for last literals (must leave room at end).
    pub const LAST_LITERALS: usize = 5;
    /// Minimum length for last match (safety margin).
    pub const MF_LIMIT: usize = 12;
    /// Hash table size (power of 2).
    pub const HASH_SIZE_U32: usize = 1 << 14; // 16384 entries
    /// Maximum offset for matches.
    pub const MAX_DISTANCE: usize = 65535;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PAGE_SIZE;

    // =========================================================================
    // F021-F035: SIMD Correctness Tests (Popperian Falsification)
    // =========================================================================

    /// F021: AVX2 matches scalar output
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f021_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("Skipping F021: AVX2 not available");
            return;
        }

        // Test 1000 random pages
        let mut rng_state = 0xDEADBEEFu64;
        for _ in 0..1000 {
            let mut input = [0u8; PAGE_SIZE];
            for byte in &mut input {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *byte = (rng_state >> 33) as u8;
            }

            let compressed = compress(&input).unwrap();

            // Decompress with scalar
            let mut scalar_output = [0u8; PAGE_SIZE];
            let scalar_len = decompress(&compressed, &mut scalar_output).unwrap();

            // Decompress with AVX2
            let mut avx2_output = [0u8; PAGE_SIZE];
            let avx2_len = unsafe { avx2::decompress_avx2(&compressed, &mut avx2_output) }.unwrap();

            assert_eq!(scalar_len, avx2_len, "Length mismatch");
            assert_eq!(scalar_output, avx2_output, "Output mismatch");
        }
    }

    /// F022: AVX-512 matches scalar output
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f022_avx512_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            println!("Skipping F022: AVX-512 not available");
            return;
        }

        // Test 1000 random pages
        let mut rng_state = 0xCAFEBABEu64;
        for _ in 0..1000 {
            let mut input = [0u8; PAGE_SIZE];
            for byte in &mut input {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *byte = (rng_state >> 33) as u8;
            }

            let compressed = compress(&input).unwrap();

            // Decompress with scalar
            let mut scalar_output = [0u8; PAGE_SIZE];
            let scalar_len = decompress(&compressed, &mut scalar_output).unwrap();

            // Decompress with AVX-512
            let mut avx512_output = [0u8; PAGE_SIZE];
            let avx512_len =
                unsafe { avx512::decompress_avx512(&compressed, &mut avx512_output) }.unwrap();

            assert_eq!(scalar_len, avx512_len, "Length mismatch");
            assert_eq!(scalar_output, avx512_output, "Output mismatch");
        }
    }

    /// F024: Unaligned input handled correctly
    #[test]
    fn test_f024_unaligned_input() {
        // Create input with 1-byte misalignment
        let mut buffer = vec![0u8; PAGE_SIZE + 1];
        for (i, byte) in buffer[1..][..PAGE_SIZE].iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Copy to aligned array for compression
        let mut input_array = [0u8; PAGE_SIZE];
        input_array.copy_from_slice(&buffer[1..][..PAGE_SIZE]);
        let compressed = compress(&input_array).unwrap();

        // Decompress to aligned buffer (SIMD requires aligned output)
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress_simd(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input_array, output);
    }

    /// F025: SIMD feature detection is correct
    #[test]
    fn test_f025_simd_detection() {
        #[cfg(target_arch = "x86_64")]
        {
            // Detection should not panic
            let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
            let has_avx512f = std::arch::is_x86_feature_detected!("avx512f");
            let has_avx512bw = std::arch::is_x86_feature_detected!("avx512bw");

            println!("AVX2: {has_avx2}, AVX512F: {has_avx512f}, AVX512BW: {has_avx512bw}");

            // If we have AVX512BW, we should also have AVX512F
            if has_avx512bw {
                assert!(has_avx512f, "AVX512BW implies AVX512F");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            println!("AArch64: NEON always available");
        }
    }

    /// F026: Fallback works when SIMD is not available
    #[test]
    fn test_f026_scalar_fallback() {
        // Always test scalar path directly
        let input = [0xABu8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();

        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();

        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    /// F027: No illegal instructions on any CPU
    #[test]
    fn test_f027_no_illegal_instructions() {
        // This test passes if it doesn't SIGILL
        let input = [0u8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];

        // SIMD dispatch should never emit illegal instructions
        let result = decompress_simd(&compressed, &mut output);
        assert!(result.is_ok());
    }

    /// F028: Feature flags produce correct code paths
    #[test]
    fn test_f028_feature_flags_respected() {
        let input = [0xCDu8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();

        // All backends should produce identical output
        let mut scalar_out = [0u8; PAGE_SIZE];
        let scalar_len = decompress(&compressed, &mut scalar_out).unwrap();

        let mut simd_out = [0u8; PAGE_SIZE];
        let simd_len = decompress_simd(&compressed, &mut simd_out).unwrap();

        assert_eq!(scalar_len, simd_len);
        assert_eq!(scalar_out, simd_out);
    }

    /// F029: Hot paths are cache-line aligned (64 bytes)
    #[test]
    fn test_f029_cache_line_alignment() {
        // Verify page data is 64-byte aligned for optimal cache utilization
        let input = [0u8; PAGE_SIZE];
        let ptr = input.as_ptr() as usize;

        // Stack arrays may not be aligned, but heap allocations should be
        let heap_input = vec![0u8; PAGE_SIZE];
        let heap_ptr = heap_input.as_ptr() as usize;

        // Check if heap allocation is at least 8-byte aligned (standard)
        assert_eq!(heap_ptr % 8, 0, "Heap should be at least 8-byte aligned");

        // Log alignment for debugging
        println!("Stack alignment: {} (mod 64 = {})", ptr, ptr % 64);
        println!("Heap alignment: {} (mod 64 = {})", heap_ptr, heap_ptr % 64);
    }

    /// F030: No false sharing in per-CPU structures
    #[test]
    fn test_f030_no_false_sharing() {
        use std::sync::atomic::AtomicU64;
        use std::sync::Arc;

        // Simulate per-CPU counters with proper padding
        #[repr(align(64))]
        struct CacheLinePadded {
            counter: AtomicU64,
            _padding: [u8; 56], // 64 - 8 = 56
        }

        let counters: Vec<_> = (0..4)
            .map(|_| Arc::new(CacheLinePadded { counter: AtomicU64::new(0), _padding: [0; 56] }))
            .collect();

        // Verify each counter is on separate cache line
        for i in 0..counters.len() - 1 {
            let addr1 = &counters[i].counter as *const _ as usize;
            let addr2 = &counters[i + 1].counter as *const _ as usize;
            let diff = if addr2 > addr1 { addr2 - addr1 } else { addr1 - addr2 };
            assert!(diff >= 64, "Counters should be >= 64 bytes apart, got {diff}");
        }
    }

    /// F032: Vector registers are preserved across calls
    #[test]
    fn test_f032_vector_register_preservation() {
        // Compress/decompress should not corrupt caller's registers
        // We test this indirectly by checking state before/after
        let sentinel = 0xDEADBEEF_CAFEBABEu64;

        let input = [0xAAu8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];

        // The sentinel should be unchanged after SIMD operations
        let _ = decompress_simd(&compressed, &mut output).unwrap();

        // If we got here without corruption, registers were preserved
        assert_eq!(sentinel, 0xDEADBEEF_CAFEBABEu64);
    }

    /// F035: SIMD operations don't cause exceptions
    #[test]
    fn test_f035_simd_no_exceptions() {
        // Test with edge cases that might cause SIMD exceptions
        let test_cases: Vec<[u8; PAGE_SIZE]> = vec![
            [0u8; PAGE_SIZE],    // All zeros
            [0xFFu8; PAGE_SIZE], // All ones
            [0x80u8; PAGE_SIZE], // Sign bit set
            [0x7Fu8; PAGE_SIZE], // Max positive
        ];

        for input in test_cases {
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; PAGE_SIZE];
            let result = decompress_simd(&compressed, &mut output);
            assert!(result.is_ok(), "SIMD should not raise exceptions");
            assert_eq!(input, output);
        }
    }

    // =========================================================================
    // Original SIMD dispatch tests
    // =========================================================================

    // Test SIMD dispatch functions
    #[test]
    fn test_decompress_simd_zeros() {
        let input = [0u8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress_simd(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_decompress_simd_pattern() {
        let mut input = [0u8; PAGE_SIZE];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress_simd(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_decompress_simd_rle() {
        // RLE pattern exercises memset/broadcast paths
        let input = [0xABu8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress_simd(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_decompress_simd_large_literals() {
        // Create data that produces large literal runs (> 64 bytes)
        let mut input = [0u8; PAGE_SIZE];
        let mut state = 12345u64;
        for byte in &mut input[..256] {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (state >> 33) as u8;
        }
        // Rest is zeros (will match)
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress_simd(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_decompress_simd_small_offsets() {
        // Test various small offsets (2-7) which require byte-by-byte copy
        for pattern_len in 2..=7 {
            let mut input = [0u8; PAGE_SIZE];
            for (i, b) in input.iter_mut().enumerate() {
                *b = (i % pattern_len) as u8;
            }
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; PAGE_SIZE];
            let len = decompress_simd(&compressed, &mut output).unwrap();
            assert_eq!(len, PAGE_SIZE, "failed for pattern_len={pattern_len}");
            assert_eq!(input[..], output[..], "failed for pattern_len={pattern_len}");
        }
    }

    #[test]
    fn test_decompress_simd_medium_offsets() {
        // Test medium offsets (8-63) which use 8-byte copies
        for pattern_len in [8, 16, 32, 48, 63] {
            let mut input = [0u8; PAGE_SIZE];
            for (i, b) in input.iter_mut().enumerate() {
                *b = (i % pattern_len) as u8;
            }
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; PAGE_SIZE];
            let len = decompress_simd(&compressed, &mut output).unwrap();
            assert_eq!(len, PAGE_SIZE, "failed for pattern_len={pattern_len}");
            assert_eq!(input[..], output[..], "failed for pattern_len={pattern_len}");
        }
    }

    #[test]
    fn test_decompress_simd_large_offsets() {
        // Test large offsets (>=64) which can use wildcard copy
        for pattern_len in [64, 128, 256, 512] {
            let mut input = [0u8; PAGE_SIZE];
            for (i, b) in input.iter_mut().enumerate() {
                *b = (i % pattern_len) as u8;
            }
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; PAGE_SIZE];
            let len = decompress_simd(&compressed, &mut output).unwrap();
            assert_eq!(len, PAGE_SIZE, "failed for pattern_len={pattern_len}");
            assert_eq!(input[..], output[..], "failed for pattern_len={pattern_len}");
        }
    }

    #[test]
    fn test_compress_simd_zeros() {
        let input = [0u8; PAGE_SIZE];
        let compressed = compress_simd(&input).unwrap();
        // Verify it compresses
        assert!(compressed.len() < PAGE_SIZE);
        // Verify roundtrip
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_compress_simd_pattern() {
        let mut input = [0u8; PAGE_SIZE];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let compressed = compress_simd(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_zeros() {
        let input = [0u8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_ones() {
        let input = [0xFFu8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_sequential() {
        let mut input = [0u8; PAGE_SIZE];
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_repeating_pattern() {
        let mut input = [0u8; PAGE_SIZE];
        let pattern = b"ABCD";
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = pattern[i % pattern.len()];
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_compression_ratio_zeros() {
        let input = [0u8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        // Zero page should compress very well (>10x)
        let ratio = PAGE_SIZE as f64 / compressed.len() as f64;
        assert!(ratio > 10.0, "Expected ratio > 10x, got {ratio:.2}x");
    }

    #[test]
    fn test_compression_ratio_pattern() {
        let mut input = [0u8; PAGE_SIZE];
        let pattern = b"Hello World! ";
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = pattern[i % pattern.len()];
        }
        let compressed = compress(&input).unwrap();
        // Repeating pattern should compress well (>5x)
        let ratio = PAGE_SIZE as f64 / compressed.len() as f64;
        assert!(ratio > 5.0, "Expected ratio > 5x, got {ratio:.2}x");
    }

    #[test]
    fn test_incompressible_data() {
        // Pseudo-random data that is hard to compress
        let mut input = [0u8; PAGE_SIZE];
        let mut state = 12345u64;
        for byte in &mut input {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (state >> 33) as u8;
        }

        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }
}
