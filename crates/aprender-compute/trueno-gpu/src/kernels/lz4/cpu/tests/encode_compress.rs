//! Encode sequence and compress block tests

use super::*;

// --- Encode Sequence Tests ---

#[test]
fn test_lz4_encode_literals_only() {
    let mut output = [0u8; 32];
    let mut pos = 0;
    let literals = b"HELLO";

    lz4_encode_sequence(&mut output, &mut pos, literals, 0, 0).expect("test");

    // Token: 5 literals, 0 match = 0x50
    assert_eq!(output[0], 0x50);
    assert_eq!(&output[1..6], b"HELLO");
    assert_eq!(pos, 6);
}

#[test]
fn test_lz4_encode_match_only() {
    let mut output = [0u8; 32];
    let mut pos = 0;

    // Match of 4 bytes at offset 10
    lz4_encode_sequence(&mut output, &mut pos, &[], 10, 4).expect("test");

    // Token: 0 literals, 0 match (4 - 4 = 0)
    assert_eq!(output[0], 0x00);
    // Offset: 10 little-endian
    assert_eq!(output[1], 10);
    assert_eq!(output[2], 0);
    assert_eq!(pos, 3);
}

#[test]
fn test_lz4_encode_literals_and_match() {
    let mut output = [0u8; 32];
    let mut pos = 0;

    // 3 literals, match of 5 bytes at offset 20
    lz4_encode_sequence(&mut output, &mut pos, b"ABC", 20, 5).expect("test");

    // Token: 3 literals, 1 match (5 - 4 = 1)
    assert_eq!(output[0], 0x31);
    assert_eq!(&output[1..4], b"ABC");
    assert_eq!(output[4], 20); // offset low
    assert_eq!(output[5], 0); // offset high
    assert_eq!(pos, 6);
}

#[test]
fn test_lz4_encode_extended_literal_length() {
    let mut output = [0u8; 64];
    let mut pos = 0;

    // 20 literals (> 15, needs extension)
    let literals = b"12345678901234567890";
    lz4_encode_sequence(&mut output, &mut pos, literals, 0, 0).expect("test");

    // Token: 15 literals (max), 0 match
    assert_eq!(output[0], 0xF0);
    // Extended length: 20 - 15 = 5
    assert_eq!(output[1], 5);
    // Literals start at output[2]
    assert_eq!(&output[2..22], literals.as_slice());
    assert_eq!(pos, 22);
}

// --- Compress Block Tests (F001 equivalent) ---

#[test]
fn test_lz4_compress_empty() {
    let mut output = [0u8; 32];
    let size = lz4_compress_block(&[], &mut output).expect("test");
    assert_eq!(size, 0);
}

#[test]
fn test_lz4_compress_small() {
    let input = b"HELLO";
    let mut output = [0u8; 32];
    let size = lz4_compress_block(input, &mut output).expect("test");

    // Small input should be stored as literals
    assert!(size > 0);
    assert_eq!(output[0] >> 4, 5); // 5 literals in token
}

#[test]
fn test_lz4_compress_repeated_pattern() {
    // Pattern that should compress well
    let mut input = [0u8; 64];
    for i in 0..64 {
        input[i] = (i % 4) as u8; // Repeating 0,1,2,3,0,1,2,3...
    }
    let mut output = [0u8; 128];
    let size = lz4_compress_block(&input, &mut output).expect("test");

    // Should compress (matches found)
    assert!(size < 64, "Should compress, got {} bytes", size);
}

#[test]
fn test_lz4_compress_zeros() {
    // Zero page should compress extremely well
    let input = [0u8; 256];
    let mut output = [0u8; 512];
    let size = lz4_compress_block(&input, &mut output).expect("test");

    // Should achieve good compression
    assert!(size < 128, "Zeros should compress well, got {} bytes", size);
}

#[test]
fn test_lz4_compress_all_same_byte() {
    // F007: Repeated patterns compress well
    let input = [b'A'; 512];
    let mut output = [0u8; 1024];
    let size = lz4_compress_block(&input, &mut output).expect("test");

    // Should achieve >10:1 ratio
    assert!(size < 52, "Repeated pattern should achieve >10:1 ratio, got {} bytes", size);
}
