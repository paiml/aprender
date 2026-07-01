//! Single, deduplicated CRC32 implementation for the `.apr` container.
//!
//! Vendored, zero-dependency IEEE CRC32 (polynomial `0xEDB88320`). This is the
//! ONE source of truth for the integrity trailer, replacing the two byte-identical
//! copies previously living in `aprender-core`:
//!   - `format/core_io.rs::crc32` (v1 APRN trailer)
//!   - `format/v2/mod.rs::crc32`  (v2 APR\0 header/footer checksum)
//!
//! Both originals built the same const lookup table at compile time and folded
//! it with the same `init = 0xFFFFFFFF`, `final = !crc` convention, so this
//! function is byte-for-byte identical to both. The known-answer test below
//! mirrors `format/test_model.rs::test_crc32_known_values`.

/// IEEE CRC32 lookup table (polynomial `0xEDB88320`), built at compile time.
const TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Compute the IEEE CRC32 checksum of `data`.
///
/// Byte-identical to the legacy `aprender-core` v1 and v2 implementations.
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE[idx];
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::crc32;

    #[test]
    fn test_crc32_empty() {
        // Empty input — IEEE convention yields 0.
        assert_eq!(crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn test_crc32_known_values() {
        // Mirrors format/test_model.rs::test_crc32_known_values.
        // "123456789" is the canonical CRC32 check vector → 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_single_byte() {
        // Mirrors format/test_model.rs::test_crc32_single_byte.
        assert_eq!(crc32(&[0x00]), 0xD202_EF8D);
        assert_eq!(crc32(&[0xFF]), 0xFF00_0000);
    }

    #[test]
    fn test_crc32_deterministic_and_distinct() {
        let crc = crc32(b"Hello, World!");
        assert_eq!(crc, crc32(b"Hello, World!"));
        assert_ne!(crc, crc32(b"Hello, World"));
    }
}
