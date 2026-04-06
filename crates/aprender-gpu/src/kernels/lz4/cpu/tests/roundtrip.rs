//! F001: LZ4 Compression is Lossless (Roundtrip Tests) and ratio tests

use super::*;

#[test]
fn test_f001_roundtrip_hello() {
    let input = b"HELLO WORLD";
    let mut compressed = [0u8; 64];
    let mut decompressed = [0u8; 64];

    let comp_size = lz4_compress_block(input, &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(decomp_size, input.len());
    assert_eq!(&decompressed[..decomp_size], input.as_slice());
}

#[test]
fn test_f001_roundtrip_zeros() {
    let input = [0u8; 256];
    let mut compressed = [0u8; 512];
    let mut decompressed = [0u8; 256];

    let comp_size = lz4_compress_block(&input, &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(decomp_size, input.len());
    assert_eq!(&decompressed[..], &input[..]);
}

#[test]
fn test_f001_roundtrip_repeated_pattern() {
    let mut input = [0u8; 512];
    for i in 0..512 {
        input[i] = (i % 13) as u8; // Non-power-of-2 pattern
    }
    let mut compressed = [0u8; 1024];
    let mut decompressed = [0u8; 512];

    let comp_size = lz4_compress_block(&input, &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(decomp_size, input.len());
    assert_eq!(&decompressed[..], &input[..]);
}

#[test]
fn test_f001_roundtrip_text() {
    let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps again!";
    let mut compressed = [0u8; 256];
    let mut decompressed = [0u8; 256];

    let comp_size = lz4_compress_block(input, &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(decomp_size, input.len());
    assert_eq!(&decompressed[..decomp_size], input.as_slice());
}

#[test]
fn test_f001_roundtrip_page_size() {
    use super::super::super::PAGE_SIZE;
    // Test with actual 4KB page
    let mut input = [0u8; PAGE_SIZE as usize];
    for i in 0..PAGE_SIZE as usize {
        input[i] = ((i * 7) % 256) as u8;
    }
    let mut compressed = [0u8; PAGE_SIZE as usize + 1024];
    let mut decompressed = [0u8; PAGE_SIZE as usize];

    let comp_size = lz4_compress_block(&input, &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(decomp_size, PAGE_SIZE as usize);
    assert_eq!(&decompressed[..], &input[..]);
}

#[test]
fn test_f006_zero_page_compression_ratio() {
    use super::super::super::PAGE_SIZE;
    // F006: Zero page compresses to <100 bytes
    let input = [0u8; PAGE_SIZE as usize];
    let mut compressed = [0u8; PAGE_SIZE as usize];

    let comp_size = lz4_compress_block(&input, &mut compressed).expect("test");

    assert!(comp_size < 100, "Zero page should compress to <100 bytes, got {}", comp_size);
}

#[test]
fn test_f007_repeated_pattern_ratio() {
    use super::super::super::PAGE_SIZE;
    // F007: 4KB of "AAAA..." achieves >100:1 ratio
    let input = [b'A'; PAGE_SIZE as usize];
    let mut compressed = [0u8; PAGE_SIZE as usize];

    let comp_size = lz4_compress_block(&input, &mut compressed).expect("test");
    let ratio = PAGE_SIZE as usize / comp_size;

    assert!(ratio >= 100, "Should achieve >100:1 ratio, got {}:1 ({} bytes)", ratio, comp_size);
}

#[test]
fn test_f003_empty_page() {
    // F003: Empty pages compress correctly
    let mut compressed = [0u8; 32];
    let mut decompressed = [0u8; 32];

    let comp_size = lz4_compress_block(&[], &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(comp_size, 0);
    assert_eq!(decomp_size, 0);
}

#[test]
fn test_f018_deterministic_output() {
    // F018: Same input always produces same output
    let input = b"Deterministic compression test data";
    let mut compressed1 = [0u8; 128];
    let mut compressed2 = [0u8; 128];

    let size1 = lz4_compress_block(input, &mut compressed1).expect("test");
    let size2 = lz4_compress_block(input, &mut compressed2).expect("test");

    assert_eq!(size1, size2);
    assert_eq!(&compressed1[..size1], &compressed2[..size2]);
}
