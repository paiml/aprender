//! Helper Functions for Tuner
//!
//! Utility functions: CRC32, timestamp, string formatting.

// ============================================================================
// CRC32 Implementation
// ============================================================================

/// Generate CRC32 lookup table at compile time.
const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB8_8320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Simple CRC32 implementation (IEEE polynomial).
/// Used for .apr file checksum verification.
pub(crate) fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    const CRC32_TABLE: [u32; 256] = crc32_table();
    let mut crc = !crc;
    for &byte in data {
        crc = CRC32_TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Compute CRC32 hash for given data (convenience wrapper)
pub(crate) fn crc32_hash(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

// ============================================================================
// Timestamp
// ============================================================================

/// Simple timestamp (avoids chrono dependency)
pub(crate) fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}", duration.as_secs())
}

// ============================================================================
// String Formatting
// ============================================================================

/// Pad string to fixed width
pub(crate) fn pad_right(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - s.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CRC32 Table Tests
    // =========================================================================

    #[test]
    fn test_crc32_table_length() {
        let table = crc32_table();
        assert_eq!(table.len(), 256, "CRC32 table must have exactly 256 entries");
    }

    #[test]
    fn test_crc32_table_first_entry_is_zero() {
        let table = crc32_table();
        assert_eq!(table[0], 0x0000_0000, "CRC32 table[0] must be 0 (identity)");
    }

    #[test]
    fn test_crc32_table_known_entries() {
        // Well-known CRC32 IEEE (0xEDB88320 reflected polynomial) table values
        let table = crc32_table();
        assert_eq!(table[1], 0x7707_3096, "table[1] mismatch for IEEE CRC32");
        assert_eq!(table[2], 0xEE0E_612C, "table[2] mismatch for IEEE CRC32");
        assert_eq!(table[3], 0x9909_51BA, "table[3] mismatch for IEEE CRC32");
        assert_eq!(table[4], 0x076D_C419, "table[4] mismatch for IEEE CRC32");
        assert_eq!(table[128], 0xEDB8_8320, "table[128] must equal the polynomial");
        assert_eq!(table[255], 0x2D02_EF8D, "table[255] mismatch for IEEE CRC32");
    }

    #[test]
    fn test_crc32_table_no_duplicate_nonzero_entries() {
        let table = crc32_table();
        // Every nonzero entry should be unique (CRC32 table is a permutation-like mapping)
        let mut seen = std::collections::HashSet::new();
        for (i, &val) in table.iter().enumerate() {
            if val != 0 {
                assert!(seen.insert(val), "Duplicate nonzero entry 0x{:08X} at index {}", val, i);
            }
        }
    }

    #[test]
    fn test_crc32_table_polynomial_property() {
        // For the IEEE polynomial 0xEDB88320, table[128] == polynomial
        // because 128 == 0x80 (only MSB set), after 8 iterations of the
        // reflected algorithm, the result is the polynomial itself.
        let table = crc32_table();
        assert_eq!(table[128], 0xEDB8_8320, "table[128] must equal the reflected CRC32 polynomial");
    }

    #[test]
    fn test_crc32_table_is_const_deterministic() {
        // Calling crc32_table() twice must produce identical results
        let table1 = crc32_table();
        let table2 = crc32_table();
        assert_eq!(table1, table2, "crc32_table() must be deterministic");
    }

    // =========================================================================
    // CRC32 Hash Tests
    // =========================================================================

    #[test]
    fn test_crc32_hash_empty() {
        // CRC32 of empty data should be 0x00000000
        let result = crc32_hash(&[]);
        assert_eq!(result, 0x0000_0000);
    }

    #[test]
    fn test_crc32_hash_known_value() {
        // CRC32 of "123456789" is a well-known test vector: 0xCBF43926
        let result = crc32_hash(b"123456789");
        assert_eq!(result, 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_hash_single_byte() {
        // CRC32 of a single byte should produce a non-zero result
        let result = crc32_hash(&[0x00]);
        assert_ne!(result, 0);
        // CRC32(0x00) = 0xD202EF8D
        assert_eq!(result, 0xD202_EF8D);
    }

    #[test]
    fn test_crc32_hash_different_inputs_differ() {
        let hash_a = crc32_hash(b"hello");
        let hash_b = crc32_hash(b"world");
        assert_ne!(hash_a, hash_b, "Different inputs should produce different hashes");
    }

    #[test]
    fn test_crc32_hash_deterministic() {
        let hash1 = crc32_hash(b"test data");
        let hash2 = crc32_hash(b"test data");
        assert_eq!(hash1, hash2, "Same input should produce same hash");
    }

    #[test]
    fn test_crc32_update_incremental() {
        // Incremental update should match single-pass hash
        let data = b"hello world";
        let single_pass = crc32_hash(data);

        let part1 = crc32_update(0, b"hello ");
        let incremental = crc32_update(part1, b"world");
        assert_eq!(single_pass, incremental, "Incremental CRC32 must match single-pass");
    }

    #[test]
    fn test_crc32_update_from_nonzero_initial() {
        // Starting from a non-zero CRC should differ from starting at 0
        let from_zero = crc32_update(0, b"test");
        let from_nonzero = crc32_update(0x1234_5678, b"test");
        assert_ne!(
            from_zero, from_nonzero,
            "Different initial CRC should produce different result"
        );
    }

    #[test]
    fn test_crc32_hash_all_zeros() {
        let result = crc32_hash(&[0u8; 16]);
        assert_ne!(result, 0, "CRC32 of all-zero data should not be zero");
    }

    #[test]
    fn test_crc32_hash_all_ones() {
        let result = crc32_hash(&[0xFFu8; 16]);
        assert_ne!(result, 0, "CRC32 of all-0xFF data should not be zero");
    }

    #[test]
    fn test_crc32_hash_large_data() {
        // Test with 1KB of data
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let result = crc32_hash(&data);
        assert_ne!(result, 0);
        // Should be deterministic
        let result2 = crc32_hash(&data);
        assert_eq!(result, result2);
    }

    // =========================================================================
    // chrono_lite_now Tests
    // =========================================================================

    #[test]
    fn test_chrono_lite_now_is_numeric() {
        let ts = chrono_lite_now();
        let parsed: u64 = ts.parse().expect("Timestamp should be a parseable number");
        assert!(parsed > 0, "Timestamp should be positive");
    }

    #[test]
    fn test_chrono_lite_now_is_reasonable() {
        let ts = chrono_lite_now();
        let secs: u64 = ts.parse().expect("Should be parseable");
        // Should be after Jan 1, 2020 (1577836800) and before Jan 1, 2100 (4102444800)
        assert!(secs > 1_577_836_800, "Timestamp should be after 2020");
        assert!(secs < 4_102_444_800, "Timestamp should be before 2100");
    }

    #[test]
    fn test_chrono_lite_now_monotonic() {
        let ts1 = chrono_lite_now();
        let ts2 = chrono_lite_now();
        let secs1: u64 = ts1.parse().expect("Should be parseable");
        let secs2: u64 = ts2.parse().expect("Should be parseable");
        assert!(secs2 >= secs1, "Timestamps should be monotonically non-decreasing");
    }

    // =========================================================================
    // pad_right Tests
    // =========================================================================

    #[test]
    fn test_pad_right_shorter_than_width() {
        let result = pad_right("hi", 10);
        assert_eq!(result, "hi        ");
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_pad_right_exact_width() {
        let result = pad_right("exact", 5);
        assert_eq!(result, "exact");
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_pad_right_longer_than_width() {
        let result = pad_right("toolong", 4);
        assert_eq!(result, "tool");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_pad_right_empty_string() {
        let result = pad_right("", 5);
        assert_eq!(result, "     ");
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_pad_right_zero_width() {
        let result = pad_right("test", 0);
        assert_eq!(result, "");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_pad_right_width_one() {
        let result = pad_right("abc", 1);
        assert_eq!(result, "a");

        let result2 = pad_right("x", 1);
        assert_eq!(result2, "x");
    }

    #[test]
    fn test_pad_right_single_char() {
        let result = pad_right("x", 5);
        assert_eq!(result, "x    ");
        assert_eq!(result.len(), 5);
    }
}
