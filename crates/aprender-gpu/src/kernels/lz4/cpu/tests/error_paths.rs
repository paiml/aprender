//! Additional Coverage Tests: Error Paths and Edge Cases

use super::*;

#[test]
fn test_lz4_encode_output_buffer_too_small() {
    // Cover line 89: "Output buffer too small" error
    let mut output = [0u8; 2]; // Too small for any meaningful output
    let mut pos = 0;
    let literals = b"HELLO WORLD"; // 11 bytes + 1 token = needs at least 12

    let result = lz4_encode_sequence(&mut output, &mut pos, literals, 0, 0);
    assert_eq!(result, Err("Output buffer too small"));
}

#[test]
fn test_lz4_encode_extended_literal_length_over_255() {
    // Cover lines 99-103: literal length >= 15 + 255 = 270 bytes
    // This exercises the while loop that writes 255 bytes at a time
    let mut output = [0u8; 1024];
    let mut pos = 0;
    let literals = vec![b'X'; 300]; // 300 bytes (15 + 255 + 30)

    lz4_encode_sequence(&mut output, &mut pos, &literals, 0, 0).expect("test");

    // Token should have 0xF0 (15 literals, 0 match)
    assert_eq!(output[0] & 0xF0, 0xF0);
    // Extended length: first byte should be 255, second byte should be 30
    assert_eq!(output[1], 255);
    assert_eq!(output[2], 30); // 300 - 15 - 255 = 30
}

#[test]
fn test_lz4_encode_very_long_literal_multiple_255s() {
    // Cover multiple iterations of while remaining >= 255
    let mut output = [0u8; 2048];
    let mut pos = 0;
    let literals = vec![b'Y'; 600]; // 600 bytes (15 + 255 + 255 + 75)

    lz4_encode_sequence(&mut output, &mut pos, &literals, 0, 0).expect("test");

    // Extended length bytes
    assert_eq!(output[1], 255);
    assert_eq!(output[2], 255);
    assert_eq!(output[3], 75); // 600 - 15 - 255 - 255 = 75
}

#[test]
fn test_lz4_decompress_truncated_literal_length() {
    // Cover line 158: "Truncated literal length" error
    // Token 0xF0 = 15 literals, needs extended length byte but none provided
    let input = [0xF0u8]; // Token with literal_len=15, but no extension byte
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Truncated extended length"));
}

#[test]
fn test_lz4_decompress_extended_literal_with_255() {
    // Cover line 165: byte == 255 branch in literal length loop
    // Token 0xF0 = 15 literals, extension byte 255 means continue reading
    let mut input = Vec::new();
    input.push(0xF0); // Token: 15 literals, 0 match
    input.push(255); // Extended: +255, continue reading
    input.push(10); // Extended: +10, stop (total = 15 + 255 + 10 = 280)
                    // Now we need 280 literal bytes
    input.extend(std::iter::repeat_n(b'A', 280));

    let mut output = [0u8; 512];
    let result = lz4_decompress_block(&input, &mut output).expect("test");
    assert_eq!(result, 280);
    assert!(output[..280].iter().all(|&b| b == b'A'));
}

#[test]
fn test_lz4_decompress_truncated_literals() {
    // Cover line 172: "Truncated literals" error
    // Token says 5 literals, but only 3 bytes follow
    let input = [0x50u8, b'A', b'B', b'C']; // Token for 5 literals, only 3 provided
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Truncated literals"));
}

#[test]
fn test_lz4_decompress_output_overflow_literals() {
    // Cover line 175: "Output buffer overflow (literals)" error
    let mut input = Vec::new();
    input.push(0x50); // Token: 5 literals
    input.extend(b"HELLO");

    let mut output = [0u8; 3]; // Too small for 5 literals

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Output buffer overflow (literals)"));
}

#[test]
fn test_lz4_decompress_truncated_match_offset() {
    // Cover line 190: "Truncated match offset" error
    // Token has match (non-zero lower nibble), but only 1 byte of offset provided
    let input = [0x11u8, b'A', 0x01]; // Token: 1 literal, 1 match; 1 literal byte, only 1 offset byte
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Truncated match offset"));
}

#[test]
fn test_lz4_decompress_zero_offset() {
    // Cover line 196: "Invalid zero offset" error
    let input = [0x11u8, b'A', 0x00, 0x00]; // Token: 1 literal, 1 match; offset=0
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Invalid zero offset"));
}

#[test]
fn test_lz4_decompress_offset_exceeds_output() {
    // Cover line 199: "Invalid offset (exceeds output)" error
    // Offset points before beginning of output
    let input = [0x11u8, b'A', 0x10, 0x00]; // Token: 1 literal, 1 match; offset=16
    let mut output = [0u8; 64];

    let result = lz4_decompress_block(&input, &mut output);
    // At this point, out_pos is 1 (1 literal written), but offset is 16
    assert_eq!(result, Err("Invalid offset (exceeds output)"));
}

#[test]
fn test_lz4_decompress_truncated_match_length() {
    // Cover line 209: "Truncated match length" error
    // Token has match_len_base=15, needs extension but none provided
    let input = vec![
        0x1F, // Token: 1 literal, 15 match (needs extension)
        b'A', // 1 literal
        0x01, // offset low
        0x00, // offset high (offset=1, valid since out_pos will be 1)
              // No match length extension byte
    ];

    let mut output = [0u8; 64];
    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Truncated extended length"));
}

#[test]
fn test_lz4_decompress_output_overflow_match() {
    // Cover line 222: "Output buffer overflow (match)" error
    // Valid match that would overflow output buffer
    let input = vec![
        0x10, // Token: 1 literal, 0 match len (actual = 4)
        b'A', // 1 literal
        0x01, // offset low
        0x00, // offset high (offset=1)
              // match_len = 0 + 4 = 4 bytes to copy
    ];

    let mut output = [0u8; 3]; // Too small: 1 literal + 4 match = 5 needed

    let result = lz4_decompress_block(&input, &mut output);
    assert_eq!(result, Err("Output buffer overflow (match)"));
}

#[test]
fn test_lz4_compress_input_smaller_than_minmatch() {
    // Cover lines 249-251: input.len() < LZ4_MIN_MATCH
    let input = [b'A', b'B', b'C']; // 3 bytes, less than MIN_MATCH (4)
    let mut output = [0u8; 32];

    let size = lz4_compress_block(&input, &mut output).expect("test");
    assert!(size > 0);
    // Should emit all 3 bytes as literals
    assert_eq!(output[0] >> 4, 3); // Token: 3 literals
}

#[test]
fn test_lz4_compress_single_byte() {
    // Edge case: single byte input
    let input = [b'X'];
    let mut output = [0u8; 32];

    let size = lz4_compress_block(&input, &mut output).expect("test");
    assert!(size > 0);
    assert_eq!(output[0] >> 4, 1); // Token: 1 literal
}

#[test]
fn test_lz4_compress_two_bytes() {
    // Edge case: two byte input
    let input = [b'A', b'B'];
    let mut output = [0u8; 32];

    let size = lz4_compress_block(&input, &mut output).expect("test");
    assert!(size > 0);
    assert_eq!(output[0] >> 4, 2); // Token: 2 literals
}
