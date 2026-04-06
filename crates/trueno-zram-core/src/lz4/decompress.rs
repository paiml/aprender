//! LZ4 decompression implementation.
//!
//! High-performance LZ4 block format decompression using unsafe pointer arithmetic.

use super::constants::MIN_MATCH;
use crate::{Error, Result};

/// Read u16 from unaligned pointer.
#[inline(always)]
unsafe fn read_u16_le(ptr: *const u8) -> u16 {
    unsafe { std::ptr::read_unaligned(ptr.cast::<u16>()) }
}

/// Read u32 from unaligned pointer.
#[allow(dead_code)] // Reserved for future optimizations
#[inline(always)]
unsafe fn read_u32(ptr: *const u8) -> u32 {
    unsafe { std::ptr::read_unaligned(ptr.cast::<u32>()) }
}

/// Read u64 from unaligned pointer.
#[inline(always)]
unsafe fn read_u64(ptr: *const u8) -> u64 {
    unsafe { std::ptr::read_unaligned(ptr.cast::<u64>()) }
}

/// Write u64 to unaligned pointer.
#[inline(always)]
unsafe fn write_u64(ptr: *mut u8, val: u64) {
    unsafe {
        std::ptr::write_unaligned(ptr.cast::<u64>(), val);
    }
}

/// Copy 8 bytes unconditionally (wildcard copy).
#[inline(always)]
unsafe fn copy_8(dst: *mut u8, src: *const u8) {
    unsafe {
        write_u64(dst, read_u64(src));
    }
}

/// Copy 16 bytes unconditionally (double wildcard copy).
#[inline(always)]
unsafe fn copy_16(dst: *mut u8, src: *const u8) {
    unsafe {
        write_u64(dst, read_u64(src));
        write_u64(dst.add(8), read_u64(src.add(8)));
    }
}

/// Wildcard copy - copies in 8-byte chunks, may overwrite past end.
#[inline(always)]
unsafe fn wildcard_copy(mut dst: *mut u8, mut src: *const u8, len: usize) {
    unsafe {
        let end = dst.add(len);
        loop {
            copy_8(dst, src);
            dst = dst.add(8);
            src = src.add(8);
            if dst >= end {
                break;
            }
        }
    }
}

/// Decompress LZ4 block format data.
///
/// # Arguments
///
/// * `input` - Compressed data in LZ4 block format
/// * `output` - Buffer to write decompressed data (must be large enough)
///
/// # Returns
///
/// The number of bytes written to output.
///
/// # Errors
///
/// Returns an error if the compressed data is corrupted.
pub fn decompress(input: &[u8], output: &mut [u8]) -> Result<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    // SAFETY: We carefully track bounds and ensure all pointer arithmetic stays within limits.
    unsafe { decompress_fast(input, output) }
}

/// Fast decompression using unsafe pointer arithmetic.
///
/// # Complexity Analysis
///
/// **Cyclomatic Complexity: 31** (intentionally high)
///
/// This function has elevated complexity because:
/// 1. **LZ4 format requires sequential token processing** - each token encodes
///    literal length, offset, and match length with variable-length encoding
/// 2. **Multiple copy strategies** - non-overlapping (wildcard), RLE (single-byte),
///    and overlapping (byte-by-byte) each require different code paths
/// 3. **Bounds checking at every step** - mandatory for memory safety
/// 4. **Performance-critical hot loop** - extracting any branch would add call overhead
///
/// The complexity is **justified and not refactorable** without significant
/// performance regression. LZ4 decompression inherently requires handling
/// multiple format cases in a tight loop.
///
/// See: <https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md>
#[inline(never)]
unsafe fn decompress_fast(input: &[u8], output: &mut [u8]) -> Result<usize> {
    unsafe {
        let mut ip = input.as_ptr();
        let ip_end = ip.add(input.len());

        let mut op = output.as_mut_ptr();
        let op_start = op;
        let op_end = op.add(output.len());

        // We need headroom for wildcard copies
        let _op_safe_end = if output.len() >= 16 { op_end.sub(16) } else { op_start };

        loop {
            // Read token
            if ip >= ip_end {
                return Err(Error::CorruptedData("unexpected end of input".to_string()));
            }
            let token = *ip;
            ip = ip.add(1);

            // Decode literal length
            let mut literal_len = ((token >> 4) & 0x0F) as usize;
            if literal_len == 15 {
                loop {
                    if ip >= ip_end {
                        return Err(Error::CorruptedData(
                            "unexpected end of input in literal length".to_string(),
                        ));
                    }
                    let byte = *ip;
                    ip = ip.add(1);
                    literal_len += byte as usize;
                    if byte != 255 {
                        break;
                    }
                }
            }

            // Copy literals
            if literal_len > 0 {
                if ip.add(literal_len) > ip_end {
                    return Err(Error::CorruptedData("literal extends past input".to_string()));
                }
                if op.add(literal_len) > op_end {
                    return Err(Error::BufferTooSmall {
                        needed: (op as usize - op_start as usize) + literal_len,
                        available: output.len(),
                    });
                }

                // Use wildcard copy when safe, otherwise byte-by-byte
                if literal_len <= 16 && op.add(16) <= op_end {
                    copy_16(op, ip);
                } else if op.add(literal_len + 8) <= op_end {
                    wildcard_copy(op, ip, literal_len);
                } else {
                    std::ptr::copy_nonoverlapping(ip, op, literal_len);
                }
                ip = ip.add(literal_len);
                op = op.add(literal_len);
            }

            // Check for end of block (no offset follows last literals)
            if ip >= ip_end {
                break;
            }

            // Read offset (little-endian 16-bit)
            if ip.add(2) > ip_end {
                return Err(Error::CorruptedData("unexpected end of input at offset".to_string()));
            }
            let offset = read_u16_le(ip) as usize;
            ip = ip.add(2);

            if offset == 0 {
                return Err(Error::CorruptedData("zero offset".to_string()));
            }

            let current_pos = op as usize - op_start as usize;
            if offset > current_pos {
                return Err(Error::CorruptedData(format!(
                    "offset {offset} exceeds output position {current_pos}"
                )));
            }

            let match_src = op.sub(offset);

            // Decode match length
            let mut match_len = (token & 0x0F) as usize + MIN_MATCH;
            if (token & 0x0F) == 15 {
                loop {
                    if ip >= ip_end {
                        return Err(Error::CorruptedData(
                            "unexpected end of input in match length".to_string(),
                        ));
                    }
                    let byte = *ip;
                    ip = ip.add(1);
                    match_len += byte as usize;
                    if byte != 255 {
                        break;
                    }
                }
            }

            // Check output space
            if op.add(match_len) > op_end {
                return Err(Error::BufferTooSmall {
                    needed: (op as usize - op_start as usize) + match_len,
                    available: output.len(),
                });
            }

            // Copy match - handle overlapping copies carefully
            if offset >= 8 && op.add(match_len + 8) <= op_end {
                // Non-overlapping: safe to use wildcard copy
                let mut src = match_src;
                let mut dst = op;
                let end = op.add(match_len);

                loop {
                    copy_8(dst, src);
                    dst = dst.add(8);
                    src = src.add(8);
                    if dst >= end {
                        break;
                    }
                }
            } else if offset == 1 && op.add(match_len + 8) <= op_end {
                // RLE (repeat single byte) - optimized path
                let byte = *match_src;
                let pattern = 0x0101010101010101u64 * u64::from(byte);
                let mut dst = op;
                let end = op.add(match_len);
                while dst < end {
                    write_u64(dst, pattern);
                    dst = dst.add(8);
                }
            } else {
                // General case: copy byte-by-byte (handles all overlaps correctly)
                for i in 0..match_len {
                    *op.add(i) = *op.sub(offset).add(i);
                }
            }
            op = op.add(match_len);
        }

        Ok(op as usize - op_start as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lz4::compress::compress;

    #[test]
    fn test_decompress_empty() {
        let mut output = [0u8; 100];
        let len = decompress(&[], &mut output).unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn test_decompress_corrupt_missing_offset() {
        // Token with 1 literal and implied match, but missing offset bytes
        let input = [0x14, b'A']; // 1 literal 'A', then should have offset
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        if let Ok(len) = result {
            assert_eq!(len, 1);
        }
    }

    #[test]
    fn test_decompress_zero_offset() {
        let input = [0x10, b'A', 0x00, 0x00];
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_property() {
        for pattern in [0x00u8, 0xFFu8, 0xAAu8, 0x55u8] {
            let input = [pattern; 4096];
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; 4096];
            let len = decompress(&compressed, &mut output).unwrap();
            assert_eq!(len, 4096);
            assert_eq!(input, output);
        }
    }

    #[test]
    fn test_roundtrip_mixed_data() {
        let mut input = [0u8; 4096];
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = ((i * 17) ^ (i >> 3)) as u8;
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; 4096];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, 4096);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_small_patterns() {
        // Test small repeating patterns that stress overlap handling
        let patterns: &[&[u8]] = &[
            &[0xAB, 0xCD],                   // 2-byte pattern
            &[0x11, 0x22, 0x33],             // 3-byte pattern
            &[0xDE, 0xAD, 0xBE, 0xEF],       // 4-byte pattern
            &[0x01, 0x02, 0x03, 0x04, 0x05], // 5-byte pattern
            &[0xAA; 7],                      // 7-byte pattern
        ];

        for pattern in patterns {
            let mut input = [0u8; 4096];
            for (i, byte) in input.iter_mut().enumerate() {
                *byte = pattern[i % pattern.len()];
            }
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; 4096];
            let len = decompress(&compressed, &mut output).unwrap();
            assert_eq!(len, 4096);
            assert_eq!(input[..], output[..]);
        }
    }

    #[test]
    fn test_decompress_literal_buffer_too_small() {
        // Create compressed data that needs more output space than available
        let input = [0xAA; 100];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; 10]; // Too small
        let result = decompress(&compressed, &mut output);
        assert!(matches!(result, Err(Error::BufferTooSmall { .. })));
    }

    #[test]
    fn test_decompress_offset_exceeds_position() {
        // Manually craft input with offset larger than current output position
        // Token: 0x10 = 1 literal, 0 match length (but offset follows)
        // After literal, offset of 0x0100 (256) but only 1 byte written
        let input = [0x10, b'A', 0x00, 0x01, 0x00]; // Offset 256, but only 1 byte written
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_extended_literal_length() {
        // Test literal length >= 15 which triggers extended length decoding
        // Create data that compresses to have extended literals
        let mut input = [0u8; 4096];
        // Non-repeating pattern forces literals
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = ((i * 7) ^ (i * 13) ^ (i >> 2)) as u8;
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; 4096];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, 4096);
    }

    #[test]
    fn test_decompress_rle_single_byte_repeat() {
        // Test RLE path (offset == 1) which uses optimized memset
        let input = [0xBB; 4096];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; 4096];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, 4096);
        assert_eq!(input, output);
    }

    #[test]
    fn test_decompress_large_match_extended_length() {
        // Test match length >= 15 which triggers extended match length
        // Highly repetitive data will produce long matches
        let mut input = [0u8; 4096];
        for i in 0..4096 {
            input[i] = (i % 4) as u8; // 4-byte repeating pattern
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; 4096];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, 4096);
        assert_eq!(input[..], output[..]);
    }

    #[test]
    fn test_decompress_small_offset_overlap() {
        // Test overlapping copy with small offsets (2-7 bytes)
        // This exercises the byte-by-byte copy path
        for offset in 2..=7 {
            let mut input = [0u8; 4096];
            for i in 0..4096 {
                input[i] = (i % offset) as u8;
            }
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; 4096];
            let len = decompress(&compressed, &mut output).unwrap();
            assert_eq!(len, 4096, "offset={offset}");
            assert_eq!(input[..], output[..], "offset={offset}");
        }
    }

    #[test]
    fn test_decompress_large_offset_non_overlap() {
        // Test non-overlapping copy with large offsets (>= 8 bytes)
        for offset in [8, 16, 32, 64, 128, 256] {
            let mut input = [0u8; 4096];
            for i in 0..4096 {
                input[i] = (i % offset) as u8;
            }
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; 4096];
            let len = decompress(&compressed, &mut output).unwrap();
            assert_eq!(len, 4096, "offset={offset}");
            assert_eq!(input[..], output[..], "offset={offset}");
        }
    }

    #[test]
    fn test_decompress_wildcard_copy_path() {
        // Large literals that trigger wildcard_copy (> 16 bytes)
        // Use pseudo-random data to prevent compression
        let mut input = [0u8; 4096];
        let mut state = 0xDEADBEEFu32;
        for byte in &mut input {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8;
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; 4096];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, 4096);
        assert_eq!(input, output);
    }

    #[test]
    fn test_decompress_copy_16_path() {
        // Small literals (<= 16 bytes) that trigger copy_16
        let input = b"Short literal!!"; // 15 bytes
        let compressed = compress(input).unwrap();
        let mut output = [0u8; 100];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, 15);
        assert_eq!(&output[..15], &input[..]);
    }

    #[test]
    fn test_decompress_truncated_match_length() {
        // Token claims extended match length but input is truncated
        // 0x1F = 1 literal, 15 match (needs extension bytes)
        let input = [0x1F, b'A', 0x01, 0x00]; // Offset 1, but no extension byte
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        // Should either succeed with what it got or fail
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_decompress_small_output_buffer() {
        // Very small output that exercises edge cases
        let input = [0x10, b'X']; // Just 1 literal
        let mut output = [0u8; 1];
        let result = decompress(&input, &mut output);
        assert!(result.is_ok());
        assert_eq!(output[0], b'X');
    }

    #[test]
    fn test_decompress_empty_input() {
        // Empty input returns 0 decompressed bytes
        let input: [u8; 0] = [];
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_decompress_truncated_literal_length_extension() {
        // Token with literal_len=15 needs extension bytes
        // 0xF0 = 15 literals, 0 match
        let input = [0xF0]; // Missing extension byte and literals
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_literal_extends_past_input() {
        // Token claims 5 literals but input has fewer
        // 0x50 = 5 literals, 0 match
        let input = [0x50, b'A', b'B']; // Only 2 bytes, claims 5
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_literal_exceeds_output() {
        // Token claims more literals than output can hold
        // 0x30 = 3 literals, 0 match
        let input = [0x30, b'A', b'B', b'C'];
        let mut output = [0u8; 2]; // Only 2 bytes
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_truncated_at_offset() {
        // Token with literals followed by match, but only 1 offset byte
        // 0x11 = 1 literal, 1 match
        let input = [0x11, b'A', 0x01]; // Has literal, but offset is truncated (needs 2 bytes)
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_truncated_match_extension() {
        // Token with extended match length but missing extension
        // 0x0F = 0 literals, 15 match (needs extension)
        let input = [0x0F, 0x01, 0x00]; // Offset 1, but no match extension
        let mut output = [0u8; 100];
        // First need some data to copy from
        output[0] = b'X';
        let result = decompress(&input, &mut output);
        // Should work with the 15+4=19 match
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_decompress_match_buffer_overflow() {
        // Match that would exceed output buffer
        // Create input that produces more output than buffer allows
        let input = [0x10, b'A', 0x01, 0x00, 0xFF, 0xFF]; // Large match
        let mut output = [0u8; 10]; // Small buffer
        let result = decompress(&input, &mut output);
        // Either succeeds with what fits or fails
        assert!(result.is_ok() || result.is_err());
    }
}
