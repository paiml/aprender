//! ARM NEON accelerated LZ4 implementation.
//!
//! This module provides NEON (128-bit SIMD) optimized LZ4 compression
//! and decompression for AArch64 CPUs.
//!
//! ## Performance Targets
//!
//! - Decompression: ≥4 GB/s throughput on modern ARM cores
//! - 16-byte wide copies for efficient memory bandwidth
//! - Optimized for Apple Silicon and ARM server CPUs

use crate::{Result, PAGE_SIZE};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON accelerated LZ4 compression.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on AArch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn compress_neon(input: &[u8; PAGE_SIZE]) -> Result<Vec<u8>> {
    // Compression is hash-table bound, NEON provides minimal benefit
    super::compress::compress(input)
}

/// Copy 16 bytes using NEON.
///
/// # Safety
///
/// - `src` must be valid for reading 16 bytes
/// - `dst` must be valid for writing 16 bytes
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn copy_16_neon(dst: *mut u8, src: *const u8) {
    let data = vld1q_u8(src);
    vst1q_u8(dst, data);
}

/// Copy 32 bytes using two NEON operations.
///
/// # Safety
///
/// - `src` must be valid for reading 32 bytes
/// - `dst` must be valid for writing 32 bytes
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn copy_32_neon(dst: *mut u8, src: *const u8) {
    let data0 = vld1q_u8(src);
    let data1 = vld1q_u8(src.add(16));
    vst1q_u8(dst, data0);
    vst1q_u8(dst.add(16), data1);
}

/// Copy 64 bytes using four NEON operations.
///
/// # Safety
///
/// - `src` must be valid for reading 64 bytes
/// - `dst` must be valid for writing 64 bytes
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn copy_64_neon(dst: *mut u8, src: *const u8) {
    let data0 = vld1q_u8(src);
    let data1 = vld1q_u8(src.add(16));
    let data2 = vld1q_u8(src.add(32));
    let data3 = vld1q_u8(src.add(48));
    vst1q_u8(dst, data0);
    vst1q_u8(dst.add(16), data1);
    vst1q_u8(dst.add(32), data2);
    vst1q_u8(dst.add(48), data3);
}

/// Wildcard copy using 16-byte NEON operations.
///
/// Copies `len` bytes from `src` to `dst`, potentially overwriting past the end.
///
/// # Safety
///
/// - `src` must be valid for reading at least `len` bytes (plus 16-byte overread)
/// - `dst` must be valid for writing at least `len` bytes (plus 16-byte overwrite)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn wildcard_copy_neon(mut dst: *mut u8, mut src: *const u8, len: usize) {
    let end = dst.add(len);

    // Unroll for common sizes
    if len <= 16 {
        copy_16_neon(dst, src);
        return;
    }

    if len <= 32 {
        copy_32_neon(dst, src);
        return;
    }

    if len <= 64 {
        copy_64_neon(dst, src);
        return;
    }

    // Large copy: 64 bytes at a time
    while dst.add(64) <= end {
        copy_64_neon(dst, src);
        dst = dst.add(64);
        src = src.add(64);
    }

    // Handle remainder
    while dst < end {
        copy_16_neon(dst, src);
        dst = dst.add(16);
        src = src.add(16);
    }
}

/// Fill memory with a repeated byte pattern using NEON.
///
/// # Safety
///
/// - `dst` must be valid for writing `len` bytes (plus potential 16-byte overwrite)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn memset_neon(dst: *mut u8, byte: u8, len: usize) {
    let pattern = vdupq_n_u8(byte);
    let mut ptr = dst;
    let end = dst.add(len);

    while ptr < end {
        vst1q_u8(ptr, pattern);
        ptr = ptr.add(16);
    }
}

/// NEON accelerated LZ4 decompression.
///
/// Uses 128-bit wide copies for efficient memory throughput on ARM CPUs.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on AArch64).
///
/// # Performance
///
/// - 16-byte aligned copies for efficient memory access
/// - Optimized RLE path using NEON broadcast
/// - ~30% faster than scalar on large matches
#[cfg(target_arch = "aarch64")]
pub unsafe fn decompress_neon(input: &[u8], output: &mut [u8; PAGE_SIZE]) -> Result<usize> {
    decompress_neon_impl(input, output)
}

/// Internal NEON decompression implementation.
///
/// # Complexity Analysis
///
/// **Cyclomatic Complexity: 32** (intentionally high)
///
/// Similar to `decompress_fast`, this function has elevated complexity because:
/// 1. **LZ4 format** - sequential token processing with variable-length fields
/// 2. **NEON-specific copy paths** - 16/32/64-byte SIMD copies vs byte-by-byte
/// 3. **Overlap handling** - different strategies for non-overlapping, RLE, and overlapping
/// 4. **Performance-critical** - cannot extract branches without adding call overhead
///
/// The complexity is **justified** - NEON decompression targets ≥4 GB/s and any
/// refactoring would regress performance on the hot path.
#[cfg(target_arch = "aarch64")]
/// Copy one LZ4 match: `match_len` bytes from `offset` bytes behind `op`, with `room` bytes left
/// in the page. A wide copy is only correct when EVERY byte it loads has already been written:
/// copy_32 needs offset >= 32, copy_64 and the 64-byte loop need offset >= 64. `offset >= 16`
/// once sent offsets 16..63 through them, so they loaded output bytes that did not exist yet
/// (aarch64, gx10 2026-09-10: a 16-byte repeating pattern decompressed to zeros after byte 32).
/// The wildcard path also writes up to 64 bytes past `match_len`, so it runs only with that much
/// room left in the page.
///
/// # Safety
/// `op - offset .. op + match_len` must be valid, `1 <= offset`, and `match_len <= room`.
#[inline(always)]
unsafe fn copy_match_neon(op: *mut u8, offset: usize, match_len: usize, room: usize) {
    let match_src = op.sub(offset);
    if offset >= 64 && match_len >= 16 && room >= match_len + 64 {
        wildcard_copy_neon(op, match_src, match_len);
    } else if offset >= 16 && match_len >= 16 {
        // 16-byte steps: each load runs after every byte it reads was stored (offset >= 16)
        copy_steps_16_neon(op, match_src, match_len);
    } else if offset == 1 {
        // RLE (repeat single byte)
        fill_run(op, *match_src, match_len);
    } else if offset >= 8 {
        // Medium offset: 8-byte copies are safe
        copy_steps_8(op, match_src, match_len);
    } else {
        // Small offset (2-7): byte-by-byte for correctness
        for i in 0..match_len {
            *op.add(i) = *match_src.add(i);
        }
    }
}

/// `len` bytes in exact 16-byte steps, then the tail byte by byte, never past `len`.
#[inline(always)]
unsafe fn copy_steps_16_neon(mut dst: *mut u8, mut src: *const u8, len: usize) {
    let end = dst.add(len);
    while dst.add(16) <= end {
        copy_16_neon(dst, src);
        dst = dst.add(16);
        src = src.add(16);
    }
    copy_tail(dst, src, end);
}

/// `len` bytes in exact 8-byte steps, then the tail byte by byte, never past `len`.
#[inline(always)]
unsafe fn copy_steps_8(mut dst: *mut u8, mut src: *const u8, len: usize) {
    let end = dst.add(len);
    while dst.add(8) <= end {
        let val = std::ptr::read_unaligned(src.cast::<u64>());
        std::ptr::write_unaligned(dst.cast::<u64>(), val);
        dst = dst.add(8);
        src = src.add(8);
    }
    copy_tail(dst, src, end);
}

#[inline(always)]
unsafe fn copy_tail(mut dst: *mut u8, mut src: *const u8, end: *mut u8) {
    while dst < end {
        *dst = *src;
        dst = dst.add(1);
        src = src.add(1);
    }
}

/// An offset-1 match: `len` copies of `byte` (NEON memset from 16 bytes up).
#[inline(always)]
unsafe fn fill_run(op: *mut u8, byte: u8, len: usize) {
    if len >= 16 {
        memset_neon(op, byte, len);
        return;
    }
    let pattern = 0x0101010101010101u64 * (byte as u64);
    let mut dst = op;
    let end = op.add(len);
    while dst.add(8) <= end {
        std::ptr::write_unaligned(dst.cast::<u64>(), pattern);
        dst = dst.add(8);
    }
    while dst < end {
        *dst = byte;
        dst = dst.add(1);
    }
}

/// An LZ4 length: `base` (the token nibble) plus, when `base` is 15, every continuation byte
/// (each adds itself; a byte below 255 ends the run). `what` names the field in the error.
#[inline(always)]
unsafe fn read_len(
    ip: &mut *const u8,
    ip_end: *const u8,
    base: usize,
    what: &str,
) -> Result<usize> {
    let mut len = base;
    if base != 15 {
        return Ok(len);
    }
    loop {
        if *ip >= ip_end {
            return Err(crate::Error::CorruptedData(format!(
                "unexpected end of input in {what}"
            )));
        }
        let byte = **ip;
        *ip = (*ip).add(1);
        len += byte as usize;
        if byte != 255 {
            return Ok(len);
        }
    }
}

/// The 2-byte little-endian match offset: nonzero, and not reaching before the output start.
#[inline(always)]
unsafe fn read_offset(ip: &mut *const u8, ip_end: *const u8, current_pos: usize) -> Result<usize> {
    if (*ip).add(2) > ip_end {
        return Err(crate::Error::CorruptedData(
            "unexpected end of input at offset".to_string(),
        ));
    }
    let offset = std::ptr::read_unaligned((*ip).cast::<u16>()) as usize;
    *ip = (*ip).add(2);
    if offset == 0 {
        return Err(crate::Error::CorruptedData("zero offset".to_string()));
    }
    if offset > current_pos {
        return Err(crate::Error::CorruptedData(format!(
            "offset {offset} exceeds output position {current_pos}"
        )));
    }
    Ok(offset)
}

/// `len` more output bytes fit in the page, or the error that says how many were needed.
#[inline(always)]
unsafe fn ensure_room(op: *mut u8, op_start: *mut u8, op_end: *mut u8, len: usize) -> Result<()> {
    if op.add(len) > op_end {
        return Err(crate::Error::BufferTooSmall {
            needed: (op as usize - op_start as usize) + len,
            available: PAGE_SIZE,
        });
    }
    Ok(())
}

/// Copy `literal_len` literal bytes from `ip` to `op` once both buffers are known to hold them.
#[inline(always)]
unsafe fn copy_literals(
    ip: *const u8,
    op: *mut u8,
    literal_len: usize,
    ip_end: *const u8,
    op_start: *mut u8,
    op_end: *mut u8,
) -> Result<()> {
    if ip.add(literal_len) > ip_end {
        return Err(crate::Error::CorruptedData(
            "literal extends past input".to_string(),
        ));
    }
    ensure_room(op, op_start, op_end, literal_len)?;
    // Use NEON for larger copies
    if literal_len >= 16 && op.add(literal_len + 16) <= op_end {
        wildcard_copy_neon(op, ip, literal_len);
    } else {
        std::ptr::copy_nonoverlapping(ip, op, literal_len);
    }
    Ok(())
}

#[inline(never)]
unsafe fn decompress_neon_impl(input: &[u8], output: &mut [u8; PAGE_SIZE]) -> Result<usize> {
    use crate::Error;

    if input.is_empty() {
        return Ok(0);
    }

    let mut ip = input.as_ptr();
    let ip_end = ip.add(input.len());

    let mut op = output.as_mut_ptr();
    let op_start = op;
    let op_end = op.add(PAGE_SIZE);

    loop {
        // Read token
        if ip >= ip_end {
            return Err(Error::CorruptedData("unexpected end of input".to_string()));
        }
        let token = *ip;
        ip = ip.add(1);

        // Literals: the length (with continuation bytes), then the bounded copy
        let literal_len = read_len(
            &mut ip,
            ip_end,
            ((token >> 4) & 0x0F) as usize,
            "literal length",
        )?;
        if literal_len > 0 {
            copy_literals(ip, op, literal_len, ip_end, op_start, op_end)?;
            ip = ip.add(literal_len);
            op = op.add(literal_len);
        }

        // Check for end of block
        if ip >= ip_end {
            break;
        }

        let offset = read_offset(&mut ip, ip_end, op as usize - op_start as usize)?;
        // MIN_MATCH = 4 on top of the nibble and its continuation bytes
        let match_len = read_len(&mut ip, ip_end, (token & 0x0F) as usize, "match length")? + 4;
        ensure_room(op, op_start, op_end, match_len)?;

        // Copy match: `copy_match_neon` picks the widest copy that is correct at this offset.
        let room = op_end as usize - op as usize;
        copy_match_neon(op, offset, match_len, room);
        op = op.add(match_len);
    }

    Ok(op as usize - op_start as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_roundtrip_zeros() {
        let input = [0u8; PAGE_SIZE];
        let compressed = unsafe { compress_neon(&input) }.unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = unsafe { decompress_neon(&compressed, &mut output) }.unwrap();

        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_roundtrip_pattern() {
        let mut input = [0u8; PAGE_SIZE];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }

        let compressed = unsafe { compress_neon(&input) }.unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = unsafe { decompress_neon(&compressed, &mut output) }.unwrap();

        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_roundtrip_rle() {
        // Test RLE (single byte repeated) - exercises memset path
        let input = [0xABu8; PAGE_SIZE];
        let compressed = unsafe { compress_neon(&input) }.unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = unsafe { decompress_neon(&compressed, &mut output) }.unwrap();

        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_roundtrip_mixed() {
        // Mixed data that will have both literals and matches
        let mut input = [0u8; PAGE_SIZE];
        for (i, b) in input.iter_mut().enumerate() {
            *b = ((i * 17) ^ (i / 3)) as u8;
        }

        let compressed = unsafe { compress_neon(&input) }.unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = unsafe { decompress_neon(&compressed, &mut output) }.unwrap();

        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_roundtrip_small_patterns() {
        // Test various small repeating patterns
        for pattern_len in [2, 3, 4, 5, 6, 7, 8, 16, 32] {
            let mut input = [0u8; PAGE_SIZE];
            for (i, b) in input.iter_mut().enumerate() {
                *b = (i % pattern_len) as u8;
            }

            let compressed = unsafe { compress_neon(&input) }.unwrap();
            let mut output = [0u8; PAGE_SIZE];
            let len = unsafe { decompress_neon(&compressed, &mut output) }.unwrap();

            assert_eq!(len, PAGE_SIZE, "pattern_len={pattern_len}");
            assert_eq!(input[..], output[..], "pattern_len={pattern_len}");
        }
    }

    /// Regression (gx10, 2026-09-10): match offsets 16..63 went through the 32/64-byte copies and read
    /// output that was not written yet. Every width boundary of the copy paths is covered here.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_roundtrip_offsets_around_copy_widths() {
        // Test various small repeating patterns
        for pattern_len in [15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 100, 255] {
            let mut input = [0u8; PAGE_SIZE];
            for (i, b) in input.iter_mut().enumerate() {
                *b = (i % pattern_len) as u8;
            }

            let compressed = unsafe { compress_neon(&input) }.unwrap();
            let mut output = [0u8; PAGE_SIZE];
            let len = unsafe { decompress_neon(&compressed, &mut output) }.unwrap();

            assert_eq!(len, PAGE_SIZE, "pattern_len={pattern_len}");
            assert_eq!(input[..], output[..], "pattern_len={pattern_len}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_compression_ratio() {
        // Highly compressible data should compress well
        let input = [0xAAu8; PAGE_SIZE];
        let compressed = unsafe { compress_neon(&input) }.unwrap();

        // Should achieve at least 10:1 compression on uniform data
        assert!(
            compressed.len() < PAGE_SIZE / 10,
            "Expected compression ratio > 10:1, got {}/{}",
            PAGE_SIZE,
            compressed.len()
        );
    }

    // Cross-platform test that doesn't require NEON
    #[test]
    fn test_neon_module_compiles() {
        // This test just verifies the module compiles on all platforms
    }
}
