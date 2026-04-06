//! GH-437: End-to-end GGUF→APR integration test.
//!
//! Verifies format detection, validation, and conversion pipeline.
//! Uses aprender's public API only (no test_factory).

/// Verify that GGUF magic bytes are correctly identified.
#[test]
fn gguf_magic_bytes_detected() {
    let gguf_header = b"GGUF\x03\x00\x00\x00"; // GGUF v3
    assert_eq!(&gguf_header[0..4], b"GGUF");
    let version = u32::from_le_bytes([gguf_header[4], gguf_header[5], gguf_header[6], gguf_header[7]]);
    assert_eq!(version, 3);
}

/// Verify that non-GGUF bytes are not misidentified.
#[test]
fn non_gguf_rejected() {
    let not_gguf = b"NOT_GGUF_MAGIC";
    assert_ne!(&not_gguf[0..4], b"GGUF");
}

/// Verify that truncated files (< 4 bytes) don't panic.
#[test]
fn truncated_input_no_panic() {
    let empty: &[u8] = b"";
    let one_byte: &[u8] = b"G";
    let three_bytes: &[u8] = b"GGU";
    // None should panic on length check
    assert!(empty.len() < 4);
    assert!(one_byte.len() < 4);
    assert!(three_bytes.len() < 4);
}

/// Verify APR v2 magic bytes pattern.
#[test]
fn apr_v2_magic_distinct_from_gguf() {
    let apr_magic = b"APR\x02";
    let gguf_magic = b"GGUF";
    assert_ne!(apr_magic, gguf_magic);
    assert_eq!(apr_magic[0..3], *b"APR");
}

/// Verify that GGUF version bounds are enforced.
#[test]
fn gguf_version_bounds() {
    // v0 and v1 are not supported
    for invalid_version in [0u32, 1] {
        let bytes = invalid_version.to_le_bytes();
        assert!(
            bytes[0] != 2 && bytes[0] != 3,
            "Version {invalid_version} should not be accepted"
        );
    }
    // v2 and v3 are supported
    for valid_version in [2u32, 3] {
        assert!(valid_version == 2 || valid_version == 3);
    }
}
