//! Same-fill page detection and optimization.
//!
//! This module implements the kernel zram same-fill optimization that stores
//! pages consisting of a repeated value (e.g., all zeros) as a flag and value,
//! consuming no memory for compressed storage.
//!
//! Reference: Linux kernel drivers/block/zram/zram_drv.c
//!
//! ## Kernel Algorithm (page_same_filled)
//!
//! The kernel stores a `u64` fill value (not just `u8`), allowing patterns like
//! 0xDEADBEEF repeated to also benefit from same-fill optimization. This is
//! critical for performance: same-fill pages skip compression entirely.
//!
//! ## PERF-017: AVX-512 Optimization
//!
//! On CPUs with AVX-512, we process 64 bytes (8 u64s) at once, reducing
//! iterations from 512 to 64 for 8x theoretical speedup in the detection loop.

use crate::PAGE_SIZE;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Result of same-fill detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameFillResult {
    /// Page is not same-fill (contains varying data).
    NotSameFill,
    /// Page is same-fill with the given repeated value.
    SameFill {
        /// The repeated byte value.
        value: u8,
    },
}

/// Detect if a page consists entirely of the same u64 word value.
///
/// This follows the kernel zram `page_same_filled()` algorithm exactly:
/// 1. Quick check: compare first vs last word (fast rejection)
/// 2. Full scan: only if first/last match
///
/// Returns `Some(fill_value)` if all words are the same, `None` otherwise.
/// The returned `u64` can be stored directly in the handle (no allocation).
///
/// # Performance
///
/// This is the CRITICAL hot path. Same-fill pages (especially zeros) are
/// 30-40% of typical swap workloads. Detecting them BEFORE compression
/// is what allows kernel ZRAM to hit 171 GB/s on zero pages.
///
/// PERF-017: On AVX-512 CPUs, uses vectorized comparison (8x fewer iterations).
///
/// # Reference
///
/// Linux kernel drivers/block/zram/zram_drv.c lines 344-358
#[inline]
#[must_use]
pub fn page_same_filled(page: &[u8; PAGE_SIZE]) -> Option<u64> {
    // PERF-017: Use AVX-512 when available
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return unsafe { page_same_filled_avx512(page) };
        }
    }

    page_same_filled_scalar(page)
}

/// Scalar implementation of page_same_filled.
#[inline]
fn page_same_filled_scalar(page: &[u8; PAGE_SIZE]) -> Option<u64> {
    // Use unaligned reads since page data may not be 8-byte aligned
    let ptr = page.as_ptr().cast::<u64>();
    let num_words = PAGE_SIZE / 8;

    // Safety: ptr is valid for num_words u64 reads
    let val = unsafe { ptr.read_unaligned() };
    let last_pos = num_words - 1;

    // Quick check: first vs last word (kernel optimization)
    if val != unsafe { ptr.add(last_pos).read_unaligned() } {
        return None;
    }

    // Full scan only if first/last match
    for i in 1..last_pos {
        if unsafe { ptr.add(i).read_unaligned() } != val {
            return None;
        }
    }

    Some(val)
}

/// AVX-512 optimized page_same_filled.
///
/// Processes 64 bytes (8 u64s) per iteration instead of 8 bytes.
/// This reduces loop iterations from 512 to 64 for ~8x speedup.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available (checked via is_x86_feature_detected).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn page_same_filled_avx512(page: &[u8; PAGE_SIZE]) -> Option<u64> {
    unsafe {
        // Use unaligned reads since page data may not be 8-byte aligned
        let ptr = page.as_ptr().cast::<u64>();
        let val = ptr.read_unaligned();
        let last_pos = PAGE_SIZE / 8 - 1;

        // Quick check: first vs last word (kernel optimization)
        if val != ptr.add(last_pos).read_unaligned() {
            return None;
        }

        // Broadcast the fill value to all 8 lanes of a 512-bit register
        let fill = _mm512_set1_epi64(val as i64);

        // Process 64 bytes (8 u64s) at a time
        // PAGE_SIZE / 64 = 64 iterations instead of 512
        let ptr = page.as_ptr().cast::<__m512i>();

        for i in 0..(PAGE_SIZE / 64) {
            let chunk = _mm512_loadu_si512(ptr.add(i));
            let mask = _mm512_cmpeq_epi64_mask(chunk, fill);

            // All 8 lanes must match (mask = 0xFF = 255)
            if mask != 0xFF {
                return None;
            }
        }

        Some(val)
    }
}

/// Fill a page with a u64 word value (kernel memset_l equivalent).
///
/// Uses byte copying which the compiler optimizes to memset/rep stosq.
/// For zero fill: ~171 GB/s (memset), For non-zero: ~25 GB/s (rep stosq).
#[inline]
pub fn fill_page_word(page: &mut [u8; PAGE_SIZE], value: u64) {
    // Convert u64 to bytes and fill - safe regardless of alignment
    let bytes = value.to_ne_bytes();
    for chunk in page.chunks_exact_mut(8) {
        chunk.copy_from_slice(&bytes);
    }
}

/// Check if a page consists entirely of the same byte value.
///
/// This is an optimized check that uses word-level comparisons for speed.
/// Following the kernel implementation, we check 8 bytes at a time.
///
/// # Arguments
///
/// * `page` - The 4KB page to check.
///
/// # Returns
///
/// `SameFill { value }` if all bytes are the same, `NotSameFill` otherwise.
#[must_use]
pub fn detect_same_fill(page: &[u8; PAGE_SIZE]) -> SameFillResult {
    if page.is_empty() {
        return SameFillResult::NotSameFill;
    }

    let first_byte = page[0];

    // Create a word filled with the first byte for fast comparison
    let fill_word = u64::from_ne_bytes([first_byte; 8]);

    // Check 8 bytes at a time
    let chunks = page.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // chunks_exact guarantees exactly 8-byte chunks, so try_into always succeeds
        let word = u64::from_ne_bytes(chunk.try_into().expect("chunks_exact guarantees 8 bytes"));
        if word != fill_word {
            return SameFillResult::NotSameFill;
        }
    }

    // Check any remaining bytes
    for &byte in remainder {
        if byte != first_byte {
            return SameFillResult::NotSameFill;
        }
    }

    SameFillResult::SameFill { value: first_byte }
}

/// Check if a page is a zero page (all zeros).
///
/// Zero pages are the most common same-fill case and are given
/// special treatment in zram for maximum efficiency.
#[must_use]
pub fn is_zero_page(page: &[u8; PAGE_SIZE]) -> bool {
    matches!(detect_same_fill(page), SameFillResult::SameFill { value: 0 })
}

/// Expand a same-fill value back to a full page.
///
/// # Arguments
///
/// * `value` - The repeated byte value.
///
/// # Returns
///
/// A 4KB page filled with the specified value.
#[must_use]
pub fn expand_same_fill(value: u8) -> [u8; PAGE_SIZE] {
    [value; PAGE_SIZE]
}

/// Compact representation of a same-fill page.
///
/// This uses only 2 bytes to represent a 4KB same-fill page:
/// - 1 byte: flags (bit 0 = is_same_fill)
/// - 1 byte: fill value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactSameFill {
    /// Flags byte (bit 0 indicates same-fill).
    pub flags: u8,
    /// The fill value.
    pub value: u8,
}

impl CompactSameFill {
    /// Flag indicating this is a same-fill page.
    pub const FLAG_SAME_FILL: u8 = 0x01;

    /// Create a compact representation for a same-fill page.
    #[must_use]
    pub fn new(value: u8) -> Self {
        Self { flags: Self::FLAG_SAME_FILL, value }
    }

    /// Check if this represents a same-fill page.
    #[must_use]
    pub fn is_same_fill(&self) -> bool {
        self.flags & Self::FLAG_SAME_FILL != 0
    }

    /// Expand back to a full page.
    #[must_use]
    pub fn expand(&self) -> [u8; PAGE_SIZE] {
        expand_same_fill(self.value)
    }

    /// Serialize to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 2] {
        [self.flags, self.value]
    }

    /// Deserialize from bytes.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 2]) -> Self {
        Self { flags: bytes[0], value: bytes[1] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Falsification Tests F003, F006, F007
    // ============================================================

    #[test]
    fn test_zero_page_detected() {
        // F003: Zero pages compress correctly
        let page = [0u8; PAGE_SIZE];
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::SameFill { value: 0 });
        assert!(is_zero_page(&page));
    }

    #[test]
    fn test_same_fill_detected() {
        // F007: Repeated patterns detected
        let page = [0xAAu8; PAGE_SIZE];
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::SameFill { value: 0xAA });
        assert!(!is_zero_page(&page));
    }

    #[test]
    fn test_non_same_fill_detected() {
        let mut page = [0u8; PAGE_SIZE];
        page[100] = 1; // Single different byte
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    #[test]
    fn test_last_byte_different() {
        let mut page = [0u8; PAGE_SIZE];
        page[PAGE_SIZE - 1] = 1;
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    #[test]
    fn test_random_page_not_same_fill() {
        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 12345u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    #[test]
    fn test_expand_same_fill_zeros() {
        let page = expand_same_fill(0);
        assert!(page.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_expand_same_fill_value() {
        let page = expand_same_fill(0xFF);
        assert!(page.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_compact_same_fill_new() {
        let compact = CompactSameFill::new(0xBB);
        assert!(compact.is_same_fill());
        assert_eq!(compact.value, 0xBB);
    }

    #[test]
    fn test_compact_same_fill_expand() {
        let compact = CompactSameFill::new(0xCC);
        let page = compact.expand();
        assert!(page.iter().all(|&b| b == 0xCC));
    }

    #[test]
    fn test_compact_same_fill_roundtrip() {
        let compact = CompactSameFill::new(0xDD);
        let bytes = compact.to_bytes();
        let restored = CompactSameFill::from_bytes(&bytes);
        assert_eq!(compact, restored);
    }

    #[test]
    fn test_compact_same_fill_serialization() {
        let compact = CompactSameFill::new(0xEE);
        let bytes = compact.to_bytes();
        assert_eq!(bytes[0], CompactSameFill::FLAG_SAME_FILL);
        assert_eq!(bytes[1], 0xEE);
    }

    #[test]
    fn test_same_fill_all_values() {
        // Test all 256 possible fill values
        for value in 0..=255u8 {
            let page = [value; PAGE_SIZE];
            let result = detect_same_fill(&page);
            assert_eq!(result, SameFillResult::SameFill { value });
        }
    }

    #[test]
    fn test_same_fill_detection_speed() {
        // Ensure detection is fast (should be O(n) with early exit)
        // Note: Debug builds are ~10x slower than release builds
        let page = [0u8; PAGE_SIZE];
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = detect_same_fill(&page);
        }
        let elapsed = start.elapsed();
        // Should process 1K pages in under 1 second (debug build)
        assert!(elapsed.as_millis() < 1000);
    }

    #[test]
    fn test_detect_same_fill_efficiency() {
        // F006: Zero page has minimal storage (2 bytes)
        let compact = CompactSameFill::new(0);
        assert_eq!(compact.to_bytes().len(), 2);
        // Compression ratio: 4096 / 2 = 2048:1
    }

    #[test]
    fn test_different_at_word_boundary() {
        let mut page = [0xAAu8; PAGE_SIZE];
        // Different at 8-byte boundary
        page[8] = 0xBB;
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    #[test]
    fn test_different_in_remainder() {
        // PAGE_SIZE is 4096 which is divisible by 8, so test a specific position
        let mut page = [0xCCu8; PAGE_SIZE];
        page[PAGE_SIZE - 1] = 0xDD;
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    // ============================================================
    // Tests for page_same_filled (kernel-style u64 detection)
    // ============================================================

    #[test]
    fn test_page_same_filled_zeros() {
        let page = [0u8; PAGE_SIZE];
        let result = page_same_filled(&page);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_page_same_filled_pattern() {
        // Create a page with repeated 0xDEADBEEF pattern
        let mut page = [0u8; PAGE_SIZE];
        let pattern: u64 = 0xDEADBEEFDEADBEEF;
        let bytes = pattern.to_ne_bytes();
        for chunk in page.chunks_exact_mut(8) {
            chunk.copy_from_slice(&bytes);
        }
        let result = page_same_filled(&page);
        assert_eq!(result, Some(pattern));
    }

    #[test]
    fn test_page_same_filled_first_last_differ() {
        let mut page = [0u8; PAGE_SIZE];
        // Set last u64 to different value - should fail fast
        let last_word_start = PAGE_SIZE - 8;
        page[last_word_start] = 0xFF;
        let result = page_same_filled(&page);
        assert_eq!(result, None);
    }

    #[test]
    fn test_page_same_filled_middle_differs() {
        let mut page = [0xAAu8; PAGE_SIZE];
        // Change a byte in the middle
        page[PAGE_SIZE / 2] = 0xBB;
        let result = page_same_filled(&page);
        assert_eq!(result, None);
    }

    #[test]
    fn test_page_same_filled_all_values() {
        // Test with a few representative byte values
        for &byte in &[0x00, 0x55, 0xAA, 0xFF] {
            let page = [byte; PAGE_SIZE];
            let expected = u64::from_ne_bytes([byte; 8]);
            let result = page_same_filled(&page);
            assert_eq!(result, Some(expected), "Failed for byte 0x{:02X}", byte);
        }
    }

    // ============================================================
    // Tests for fill_page_word
    // ============================================================

    #[test]
    fn test_fill_page_word_zeros() {
        let mut page = [0xFFu8; PAGE_SIZE];
        fill_page_word(&mut page, 0);
        assert!(page.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_fill_page_word_pattern() {
        let mut page = [0u8; PAGE_SIZE];
        let pattern: u64 = 0xCAFEBABE12345678;
        fill_page_word(&mut page, pattern);

        // Verify all bytes match the pattern
        let expected = pattern.to_ne_bytes();
        for chunk in page.chunks_exact(8) {
            assert_eq!(chunk, &expected);
        }
    }

    #[test]
    fn test_fill_page_word_roundtrip() {
        // Fill with a pattern and verify page_same_filled detects it
        let mut page = [0u8; PAGE_SIZE];
        let pattern: u64 = 0x0102030405060708;
        fill_page_word(&mut page, pattern);

        let detected = page_same_filled(&page);
        assert_eq!(detected, Some(pattern));
    }

    #[test]
    fn test_fill_page_word_all_ones() {
        let mut page = [0u8; PAGE_SIZE];
        fill_page_word(&mut page, u64::MAX);
        assert!(page.iter().all(|&b| b == 0xFF));
    }
}
