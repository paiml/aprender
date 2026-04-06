//! Zstandard compression.
//!
//! Implements a simplified Zstandard compressor suitable for page compression.

use super::{BlockType, ZSTD_MAGIC};
use crate::Result;

/// Compress data using Zstandard format.
///
/// # Arguments
///
/// * `input` - Data to compress
/// * `level` - Compression level (1-22, typically 1-9 for fast compression)
///
/// # Errors
///
/// Returns an error if compression fails.
pub fn compress(input: &[u8], level: i32) -> Result<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let level = level.clamp(1, 19);

    // Estimate output size
    let max_output = input.len() + (input.len() / 128) + 32;
    let mut output = Vec::with_capacity(max_output);

    // Write frame header
    write_frame_header(&mut output, input.len())?;

    // Compress data as blocks
    compress_blocks(&mut output, input, level)?;

    // Write empty checksum (optional in zstd)

    Ok(output)
}

fn write_frame_header(output: &mut Vec<u8>, content_size: usize) -> Result<()> {
    // Magic number
    output.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());

    // Frame header descriptor
    // Bits: Frame_Content_Size_flag (2) | Single_Segment_flag (1) | unused (1) |
    //       Reserved (1) | Content_Checksum_flag (1) | Dictionary_ID_flag (2)
    let fcs_flag = if content_size <= 255 {
        0 // 1 byte FCS
    } else if content_size <= 65535 + 256 {
        1 // 2 bytes FCS
    } else {
        2 // 4 bytes FCS
    };

    let descriptor = (fcs_flag << 6) | (1 << 5); // Single segment flag set
    output.push(descriptor);

    // Frame content size
    match fcs_flag {
        0 => output.push(content_size as u8),
        1 => {
            let size = (content_size - 256) as u16;
            output.extend_from_slice(&size.to_le_bytes());
        }
        2 => {
            output.extend_from_slice(&(content_size as u32).to_le_bytes());
        }
        3 => {
            output.extend_from_slice(&(content_size as u64).to_le_bytes());
        }
        _ => unreachable!(),
    }

    Ok(())
}

fn compress_blocks(output: &mut Vec<u8>, input: &[u8], level: i32) -> Result<()> {
    // For simplicity, we'll use a single block strategy
    // Real implementation would split into 128KB blocks

    let max_block_size = 131072; // 128KB

    for (i, chunk) in input.chunks(max_block_size).enumerate() {
        let is_last = (i + 1) * max_block_size >= input.len();
        compress_block(output, chunk, is_last, level)?;
    }

    Ok(())
}

fn compress_block(output: &mut Vec<u8>, input: &[u8], is_last: bool, level: i32) -> Result<()> {
    // Try to compress the block
    let compressed = compress_sequences(input, level)?;

    // Decide whether to use compressed or raw block
    let (block_type, data) = if compressed.len() < input.len() {
        (BlockType::Compressed, compressed)
    } else {
        // Use raw block if compression doesn't help
        (BlockType::Raw, input.to_vec())
    };

    // Check for RLE opportunity
    let (block_type, data) = if input.iter().all(|&b| b == input[0]) && input.len() > 3 {
        (BlockType::Rle, vec![input[0]])
    } else {
        (block_type, data)
    };

    // Write block header (3 bytes)
    let block_size = match block_type {
        BlockType::Rle => input.len(), // Original size for RLE
        _ => data.len(),
    };

    let header = u32::from(is_last) | ((block_type as u32) << 1) | ((block_size as u32) << 3);

    output.push((header & 0xFF) as u8);
    output.push(((header >> 8) & 0xFF) as u8);
    output.push(((header >> 16) & 0xFF) as u8);

    // Write block data
    output.extend_from_slice(&data);

    Ok(())
}

fn compress_sequences(input: &[u8], _level: i32) -> Result<Vec<u8>> {
    // Simplified sequence compression
    // A full implementation would use:
    // 1. LZ77-style match finding
    // 2. Literals encoding (raw, RLE, Huffman, or 4-stream Huffman)
    // 3. Sequences encoding (FSE for literals length, match length, offset)

    // For now, implement a simple literals-only compressed block
    let mut output = Vec::with_capacity(input.len());

    // Literals section header
    // Type 0 = raw literals
    let lit_size = input.len();

    if lit_size < 32 {
        // Size format 0: 5 bits size
        output.push((lit_size as u8) << 3);
    } else if lit_size < 4096 {
        // Size format 1: 12 bits size
        let header = ((lit_size as u16) << 4) | 0x01;
        output.push((header & 0xFF) as u8);
        output.push((header >> 8) as u8);
    } else {
        // Size format 2: 20 bits size
        let header = ((lit_size as u32) << 4) | 0x02;
        output.push((header & 0xFF) as u8);
        output.push(((header >> 8) & 0xFF) as u8);
        output.push(((header >> 16) & 0xFF) as u8);
    }

    // Raw literals
    output.extend_from_slice(input);

    // Sequences section header (0 sequences)
    output.push(0);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_empty() {
        let result = compress(&[], 3).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_compress_produces_valid_header() {
        let input = [0u8; 100];
        let compressed = compress(&input, 3).unwrap();

        // Check magic number
        assert!(compressed.len() >= 4);
        let magic =
            u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
        assert_eq!(magic, ZSTD_MAGIC);
    }

    #[test]
    fn test_compress_rle_detection() {
        let input = [0xAAu8; 4096];
        let compressed = compress(&input, 3).unwrap();
        // RLE should be very efficient
        assert!(compressed.len() < 50);
    }

    #[test]
    fn test_compress_small_content() {
        // Content size <= 255: uses 1 byte FCS
        let input = [0x12u8; 100];
        let compressed = compress(&input, 1).unwrap();
        assert!(compressed.len() > 4); // At least magic + header
    }

    #[test]
    fn test_compress_medium_content() {
        // Content size 256-65791: uses 2 byte FCS
        let input = vec![0x34u8; 1000];
        let compressed = compress(&input, 1).unwrap();
        assert!(compressed.len() > 6);
    }

    #[test]
    fn test_compress_large_content() {
        // Content size > 65791: uses 4 byte FCS
        let input = vec![0x56u8; 100_000];
        let compressed = compress(&input, 1).unwrap();
        // Should be much smaller than input (RLE)
        assert!(compressed.len() < 100);
    }

    #[test]
    fn test_compress_level_clamp_low() {
        // Level < 1 should be clamped to 1
        let input = [0x78u8; 64];
        let _ = compress(&input, 0).unwrap();
    }

    #[test]
    fn test_compress_level_clamp_high() {
        // Level > 19 should be clamped to 19
        let input = [0x9Au8; 64];
        let _ = compress(&input, 25).unwrap();
    }

    #[test]
    fn test_compress_small_literals() {
        // < 32 bytes: uses 5-bit size format
        let input = [0xBCu8; 16];
        let compressed = compress(&input, 3).unwrap();
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_compress_medium_literals() {
        // 32-4095 bytes: uses 12-bit size format
        let input = vec![0xDEu8; 500];
        let compressed = compress(&input, 3).unwrap();
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_compress_large_literals() {
        // >= 4096 bytes: uses 20-bit size format
        let input = vec![0xF0u8; 5000];
        let compressed = compress(&input, 3).unwrap();
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_compress_mixed_content() {
        // Non-RLE content that will use compressed or raw block
        let mut input = Vec::with_capacity(256);
        for i in 0..256 {
            input.push(i as u8);
        }
        let compressed = compress(&input, 3).unwrap();
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_compress_multiple_blocks() {
        // Create input larger than 128KB block size
        let input = vec![0x11u8; 200_000];
        let compressed = compress(&input, 1).unwrap();
        assert!(!compressed.is_empty());
    }
}
