//! Hash function and match length tests

use super::*;

// --- Hash Function Tests ---

#[test]
fn test_lz4_hash_produces_12bit_output() {
    // Hash output must always be < 4096 (12 bits)
    for val in [0u32, 1, 0x12345678, 0xFFFFFFFF, 0xDEADBEEF] {
        let h = lz4_hash(val);
        assert!(h < LZ4_HASH_SIZE, "Hash {} >= 4096 for input {}", h, val);
    }
}

#[test]
fn test_lz4_hash_deterministic() {
    // Same input must produce same hash
    let val = 0x12345678u32;
    assert_eq!(lz4_hash(val), lz4_hash(val));
}

#[test]
fn test_lz4_hash_distribution() {
    // Different inputs should produce different hashes (mostly)
    let h1 = lz4_hash(0x00000000);
    let h2 = lz4_hash(0x00000001);
    let h3 = lz4_hash(0x00010000);
    // Not all different, but collision rate should be low
    assert!(h1 != h2 || h2 != h3, "Too many collisions");
}

#[test]
fn test_lz4_hash_at_from_slice() {
    let data = [0x12u8, 0x34, 0x56, 0x78, 0x9A];
    let expected_val = 0x78563412u32; // Little-endian
    assert_eq!(lz4_hash_at(&data, 0), lz4_hash(expected_val));
}

#[test]
fn test_read_u32_le() {
    assert_eq!(read_u32_le(&[0x01, 0x02, 0x03, 0x04], 0), 0x04030201);
    assert_eq!(read_u32_le(&[0xFF, 0xFF, 0xFF, 0xFF], 0), 0xFFFFFFFF);
    assert_eq!(read_u32_le(&[0x00, 0x00, 0x01, 0x02, 0x03, 0x04], 2), 0x04030201);
}

// --- Match Length Tests ---

#[test]
fn test_lz4_match_length_identical() {
    let data = b"AAAAAAAA";
    let len = lz4_match_length(data, 0, 4, 4);
    assert_eq!(len, 4, "Should match 4 bytes");
}

#[test]
fn test_lz4_match_length_partial() {
    let data = b"AAABAAAC";
    let len = lz4_match_length(data, 0, 4, 8);
    assert_eq!(len, 3, "Should match 3 bytes (AAA vs AAA)");
}

#[test]
fn test_lz4_match_length_no_match() {
    let data = b"ABCDWXYZ";
    let len = lz4_match_length(data, 0, 4, 4);
    assert_eq!(len, 0, "Should match 0 bytes");
}

#[test]
fn test_lz4_match_length_limit_respected() {
    let data = b"AAAAAAAAAAAA";
    let len = lz4_match_length(data, 0, 4, 3);
    assert_eq!(len, 3, "Should be limited to 3 bytes");
}
