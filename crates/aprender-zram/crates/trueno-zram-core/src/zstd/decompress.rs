//! Zstandard decompression.

use super::{BlockHeader, BlockType, FrameHeader, ZSTD_MAGIC};
use crate::{Error, Result};

/// Decompress Zstandard data.
///
/// # Arguments
///
/// * `input` - Compressed data in Zstandard format
/// * `output` - Buffer for decompressed data
///
/// # Returns
///
/// Number of bytes written to output.
///
/// # Errors
///
/// Returns an error if decompression fails.
pub fn decompress(input: &[u8], output: &mut [u8]) -> Result<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut pos = 0usize;
    let mut out_pos = 0usize;

    // Read frame header
    let header = read_frame_header(input, &mut pos)?;

    // Decompress blocks
    loop {
        let block = read_block_header(input, &mut pos)?;

        match block.block_type {
            BlockType::Raw => {
                // Copy raw data
                if pos + block.block_size as usize > input.len() {
                    return Err(Error::CorruptedData("block extends past input".to_string()));
                }
                if out_pos + block.block_size as usize > output.len() {
                    return Err(Error::BufferTooSmall {
                        needed: out_pos + block.block_size as usize,
                        available: output.len(),
                    });
                }

                output[out_pos..out_pos + block.block_size as usize]
                    .copy_from_slice(&input[pos..pos + block.block_size as usize]);
                pos += block.block_size as usize;
                out_pos += block.block_size as usize;
            }
            BlockType::Rle => {
                // Repeat single byte
                if pos >= input.len() {
                    return Err(Error::CorruptedData("missing RLE byte".to_string()));
                }
                let byte = input[pos];
                pos += 1;

                if out_pos + block.block_size as usize > output.len() {
                    return Err(Error::BufferTooSmall {
                        needed: out_pos + block.block_size as usize,
                        available: output.len(),
                    });
                }

                for i in 0..block.block_size as usize {
                    output[out_pos + i] = byte;
                }
                out_pos += block.block_size as usize;
            }
            BlockType::Compressed => {
                let block_end = pos + block.block_size as usize;
                if block_end > input.len() {
                    return Err(Error::CorruptedData("block extends past input".to_string()));
                }

                out_pos += decompress_block(&input[pos..block_end], &mut output[out_pos..])?;
                pos = block_end;
            }
            BlockType::Reserved => {
                return Err(Error::CorruptedData("reserved block type".to_string()));
            }
        }

        if block.last_block {
            break;
        }
    }

    // Verify size if known
    if let Some(expected) = header.frame_content_size {
        if out_pos as u64 != expected {
            return Err(Error::CorruptedData(format!(
                "size mismatch: expected {expected}, got {out_pos}"
            )));
        }
    }

    Ok(out_pos)
}

fn read_frame_header(input: &[u8], pos: &mut usize) -> Result<FrameHeader> {
    if input.len() < 5 {
        return Err(Error::CorruptedData("input too short for frame header".to_string()));
    }

    // Check magic
    let magic = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if magic != ZSTD_MAGIC {
        return Err(Error::CorruptedData(format!(
            "invalid magic: expected {ZSTD_MAGIC:#X}, got {magic:#X}"
        )));
    }
    *pos = 4;

    // Read descriptor
    let descriptor = input[*pos];
    *pos += 1;

    let fcs_flag = (descriptor >> 6) & 0x03;
    let single_segment = (descriptor >> 5) & 0x01 != 0;
    let checksum = (descriptor >> 2) & 0x01 != 0;
    let dict_id_flag = descriptor & 0x03;

    // Window descriptor (if not single segment)
    let window_size = if single_segment {
        0 // Will be determined by frame content size
    } else {
        if *pos >= input.len() {
            return Err(Error::CorruptedData("missing window descriptor".to_string()));
        }
        let wd = input[*pos];
        *pos += 1;
        let exponent = u64::from(wd >> 3);
        let mantissa = u64::from(wd & 0x07);
        (1u64 << (10 + exponent)) + (mantissa << (7 + exponent))
    };

    // Dictionary ID
    let dictionary_id = match dict_id_flag {
        0 => None,
        1 => {
            if *pos >= input.len() {
                return Err(Error::CorruptedData("missing dict id".to_string()));
            }
            let id = u32::from(input[*pos]);
            *pos += 1;
            Some(id)
        }
        2 => {
            if *pos + 2 > input.len() {
                return Err(Error::CorruptedData("missing dict id".to_string()));
            }
            let id = u32::from(u16::from_le_bytes([input[*pos], input[*pos + 1]]));
            *pos += 2;
            Some(id)
        }
        3 => {
            if *pos + 4 > input.len() {
                return Err(Error::CorruptedData("missing dict id".to_string()));
            }
            let id = u32::from_le_bytes([
                input[*pos],
                input[*pos + 1],
                input[*pos + 2],
                input[*pos + 3],
            ]);
            *pos += 4;
            Some(id)
        }
        _ => unreachable!(),
    };

    // Frame content size
    let frame_content_size = match fcs_flag {
        0 if single_segment => {
            if *pos >= input.len() {
                return Err(Error::CorruptedData("missing frame content size".to_string()));
            }
            let size = u64::from(input[*pos]);
            *pos += 1;
            Some(size)
        }
        0 => None,
        1 => {
            if *pos + 2 > input.len() {
                return Err(Error::CorruptedData("missing frame content size".to_string()));
            }
            let size = u64::from(u16::from_le_bytes([input[*pos], input[*pos + 1]])) + 256;
            *pos += 2;
            Some(size)
        }
        2 => {
            if *pos + 4 > input.len() {
                return Err(Error::CorruptedData("missing frame content size".to_string()));
            }
            let size = u64::from(u32::from_le_bytes([
                input[*pos],
                input[*pos + 1],
                input[*pos + 2],
                input[*pos + 3],
            ]));
            *pos += 4;
            Some(size)
        }
        3 => {
            if *pos + 8 > input.len() {
                return Err(Error::CorruptedData("missing frame content size".to_string()));
            }
            let size = u64::from_le_bytes([
                input[*pos],
                input[*pos + 1],
                input[*pos + 2],
                input[*pos + 3],
                input[*pos + 4],
                input[*pos + 5],
                input[*pos + 6],
                input[*pos + 7],
            ]);
            *pos += 8;
            Some(size)
        }
        _ => unreachable!(),
    };

    Ok(FrameHeader { window_size, frame_content_size, dictionary_id, checksum, single_segment })
}

fn read_block_header(input: &[u8], pos: &mut usize) -> Result<BlockHeader> {
    if *pos + 3 > input.len() {
        return Err(Error::CorruptedData("missing block header".to_string()));
    }

    let header = u32::from(input[*pos])
        | (u32::from(input[*pos + 1]) << 8)
        | (u32::from(input[*pos + 2]) << 16);
    *pos += 3;

    let last_block = (header & 0x01) != 0;
    let block_type = BlockType::from(((header >> 1) & 0x03) as u8);
    let block_size = header >> 3;

    Ok(BlockHeader { block_type, last_block, block_size })
}

fn decompress_block(input: &[u8], output: &mut [u8]) -> Result<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    // Read literals section header
    let (lit_type, lit_size, lit_header_size) = read_literals_header(input)?;

    if lit_header_size >= input.len() {
        return Err(Error::CorruptedData("literals extend past block".to_string()));
    }

    let lit_start = lit_header_size;
    let lit_end = lit_start
        + match lit_type {
            0 => lit_size, // Raw literals
            1 => 1,        // RLE
            _ => lit_size, // Compressed (simplified)
        };

    if lit_end > input.len() {
        return Err(Error::CorruptedData("literals extend past input".to_string()));
    }

    // Handle literals
    match lit_type {
        0 => {
            // Raw literals
            if lit_size > output.len() {
                return Err(Error::BufferTooSmall { needed: lit_size, available: output.len() });
            }
            output[..lit_size].copy_from_slice(&input[lit_start..lit_end]);
        }
        1 => {
            // RLE literals
            let byte = input[lit_start];
            if lit_size > output.len() {
                return Err(Error::BufferTooSmall { needed: lit_size, available: output.len() });
            }
            for i in 0..lit_size {
                output[i] = byte;
            }
        }
        _ => {
            // Compressed literals - simplified handling
            return Err(Error::Unsupported("compressed literals".to_string()));
        }
    }

    // Read sequences section
    let seq_start = lit_end;
    if seq_start >= input.len() {
        // No sequences, just literals
        return Ok(lit_size);
    }

    let num_sequences = input[seq_start] as usize;
    if num_sequences == 0 {
        return Ok(lit_size);
    }

    // Full sequence decoding would go here
    // For now, return error for non-zero sequences
    Err(Error::Unsupported("sequence decoding".to_string()))
}

fn read_literals_header(input: &[u8]) -> Result<(u8, usize, usize)> {
    if input.is_empty() {
        return Err(Error::CorruptedData("missing literals header".to_string()));
    }

    let header = input[0];
    let lit_type = header & 0x03;
    let size_format = (header >> 2) & 0x03;

    match size_format {
        0 | 1 => {
            // 1 byte header
            let size = (header >> 3) as usize;
            Ok((lit_type, size, 1))
        }
        2 => {
            // 2 byte header
            if input.len() < 2 {
                return Err(Error::CorruptedData("truncated literals header".to_string()));
            }
            let size = ((header >> 4) as usize) | ((input[1] as usize) << 4);
            Ok((lit_type, size, 2))
        }
        3 => {
            // 3 byte header
            if input.len() < 3 {
                return Err(Error::CorruptedData("truncated literals header".to_string()));
            }
            let size =
                ((header >> 4) as usize) | ((input[1] as usize) << 4) | ((input[2] as usize) << 12);
            Ok((lit_type, size, 3))
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let mut output = [0u8; 100];
        let result = decompress(&[], &mut output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_invalid_magic() {
        let input = [0x00, 0x00, 0x00, 0x00, 0x00];
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
        match result {
            Err(Error::CorruptedData(msg)) => assert!(msg.contains("magic")),
            _ => unreachable!("expected corrupted data error"),
        }
    }

    #[test]
    fn test_input_too_short() {
        let input = [0x28, 0xB5, 0x2F]; // Only 3 bytes of magic
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_block_header() {
        // Last block, raw type, size 100
        let input = [0x21, 0x03, 0x00]; // 1 | (0 << 1) | (100 << 3)
        let mut pos = 0;
        let header = read_block_header(&input, &mut pos).unwrap();
        assert!(header.last_block);
        assert_eq!(header.block_type, BlockType::Raw);
        assert_eq!(header.block_size, 100);
    }

    #[test]
    fn test_read_block_header_rle() {
        // Not last block, RLE type, size 50
        let input = [0x92, 0x01, 0x00]; // 0 | (1 << 1) | (50 << 3)
        let mut pos = 0;
        let header = read_block_header(&input, &mut pos).unwrap();
        assert!(!header.last_block);
        assert_eq!(header.block_type, BlockType::Rle);
        assert_eq!(header.block_size, 50);
    }

    #[test]
    fn test_read_block_header_compressed() {
        // Last block, compressed type, size 200
        let input = [0x45, 0x06, 0x00]; // 1 | (2 << 1) | (200 << 3)
        let mut pos = 0;
        let header = read_block_header(&input, &mut pos).unwrap();
        assert!(header.last_block);
        assert_eq!(header.block_type, BlockType::Compressed);
        assert_eq!(header.block_size, 200);
    }

    #[test]
    fn test_read_block_header_reserved() {
        // Last block, reserved type, size 10
        let input = [0x57, 0x00, 0x00]; // 1 | (3 << 1) | (10 << 3)
        let mut pos = 0;
        let header = read_block_header(&input, &mut pos).unwrap();
        assert!(header.last_block);
        assert_eq!(header.block_type, BlockType::Reserved);
    }

    #[test]
    fn test_read_block_header_truncated() {
        let input = [0x21, 0x03]; // Only 2 bytes
        let mut pos = 0;
        let result = read_block_header(&input, &mut pos);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_literals_header_1byte() {
        // size_format=0 or 1 means 1-byte header
        // Byte format: [size(5 bits)][size_format(2 bits)][lit_type(2 bits)]
        // For size=8, size_format=0, lit_type=0: (8 << 3) | (0 << 2) | 0 = 0x40
        let input = [0x40]; // size=8, size_format=0, type=0
        let result = read_literals_header(&input);
        assert!(result.is_ok());
        let (lit_type, size, header_size) = result.unwrap();
        assert_eq!(lit_type, 0);
        assert_eq!(size, 8);
        assert_eq!(header_size, 1);
    }

    #[test]
    fn test_read_literals_header_2byte() {
        // size_format = 2 (2-byte header)
        // First byte: [size_high(4 bits)][size_format(2 bits)][lit_type(2 bits)]
        // size_format = 2 = 0b10
        let input = [0b00001000, 0x10]; // size_format=2, type=0
        let result = read_literals_header(&input);
        assert!(result.is_ok());
        let (lit_type, _size, header_size) = result.unwrap();
        assert_eq!(lit_type, 0);
        assert_eq!(header_size, 2);
    }

    #[test]
    fn test_read_literals_header_3byte() {
        // size_format = 3 (3-byte header)
        let input = [0b00001100, 0x10, 0x00]; // size_format=3, type=0
        let result = read_literals_header(&input);
        assert!(result.is_ok());
        let (_, _, header_size) = result.unwrap();
        assert_eq!(header_size, 3);
    }

    #[test]
    fn test_read_literals_header_truncated_2byte() {
        let input = [0b00001000]; // 2-byte header but only 1 byte
        let result = read_literals_header(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_literals_header_truncated_3byte() {
        let input = [0b00001100, 0x10]; // 3-byte header but only 2 bytes
        let result = read_literals_header(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_literals_header_empty() {
        let input: [u8; 0] = [];
        let result = read_literals_header(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_decompress_block_empty() {
        let input: [u8; 0] = [];
        let mut output = [0u8; 100];
        let result = decompress_block(&input, &mut output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_decompress_block_raw_literals() {
        // Raw literals (type=0), size=4
        let input = [0b00100000, b'T', b'E', b'S', b'T'];
        let mut output = [0u8; 100];
        let result = decompress_block(&input, &mut output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4);
        assert_eq!(&output[..4], b"TEST");
    }

    #[test]
    fn test_decompress_block_rle_literals() {
        // RLE literals (type=1), size=4, byte=0xAA
        let input = [0b00100001, 0xAA];
        let mut output = [0u8; 100];
        let result = decompress_block(&input, &mut output);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 4);
        assert_eq!(output[..4], [0xAA, 0xAA, 0xAA, 0xAA]);
    }

    #[test]
    fn test_decompress_block_buffer_too_small() {
        // Raw literals, size=10
        let input = [0b01010000, b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J'];
        let mut output = [0u8; 5]; // Too small
        let result = decompress_block(&input, &mut output);
        assert!(matches!(result, Err(Error::BufferTooSmall { .. })));
    }

    #[test]
    fn test_decompress_block_literals_past_input() {
        // Raw literals, claims size=100 but input is short
        let header = (100usize << 3) as u8;
        let input = [header, b'A', b'B'];
        let mut output = [0u8; 200];
        let result = decompress_block(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_frame_header_with_window_descriptor() {
        // Valid magic + descriptor with window
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0x00); // Descriptor: fcs_flag=0, single_segment=0, checksum=0, dict_id_flag=0
        input.push(0x10); // Window descriptor
        input.push(0x01); // Block header (minimal)
        input.push(0x00);
        input.push(0x00);

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert!(!header.single_segment);
        assert!(!header.checksum);
        assert!(header.window_size > 0);
    }

    #[test]
    fn test_read_frame_header_single_segment_with_fcs() {
        // Valid magic + descriptor with single segment and FCS
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0x20); // Descriptor: fcs_flag=0, single_segment=1
        input.push(0x10); // FCS (1 byte when single_segment && fcs_flag==0)

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert!(header.single_segment);
        assert_eq!(header.frame_content_size, Some(0x10));
    }

    #[test]
    fn test_read_frame_header_fcs_2bytes() {
        // Valid magic + descriptor with 2-byte FCS
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0x40); // Descriptor: fcs_flag=1 (2 bytes)
        input.push(0x10); // Window descriptor
        input.push(0x00); // FCS low byte
        input.push(0x01); // FCS high byte

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        // FCS = 0x0100 + 256 = 512
        assert_eq!(header.frame_content_size, Some(512));
    }

    #[test]
    fn test_read_frame_header_dict_id_1byte() {
        // Valid magic + descriptor with 1-byte dict ID
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0x01); // Descriptor: dict_id_flag=1
        input.push(0x10); // Window descriptor
        input.push(0xAB); // Dict ID (1 byte)

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert_eq!(header.dictionary_id, Some(0xAB));
    }

    #[test]
    fn test_read_frame_header_dict_id_2bytes() {
        // Valid magic + descriptor with 2-byte dict ID
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0x02); // Descriptor: dict_id_flag=2
        input.push(0x10); // Window descriptor
        input.push(0xCD); // Dict ID low
        input.push(0xAB); // Dict ID high

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert_eq!(header.dictionary_id, Some(0xABCD));
    }

    #[test]
    fn test_read_frame_header_dict_id_4bytes() {
        // Valid magic + descriptor with 4-byte dict ID
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0x03); // Descriptor: dict_id_flag=3
        input.push(0x10); // Window descriptor
        input.push(0x78); // Dict ID bytes
        input.push(0x56);
        input.push(0x34);
        input.push(0x12);

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert_eq!(header.dictionary_id, Some(0x12345678));
    }

    #[test]
    fn test_read_frame_header_fcs_4bytes() {
        // Valid magic + descriptor with 4-byte FCS
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0x80); // Descriptor: fcs_flag=2 (4 bytes)
        input.push(0x10); // Window descriptor
        input.push(0x00); // FCS bytes
        input.push(0x10);
        input.push(0x00);
        input.push(0x00);

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert_eq!(header.frame_content_size, Some(0x00001000));
    }

    #[test]
    fn test_read_frame_header_fcs_8bytes() {
        // Valid magic + descriptor with 8-byte FCS
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0xC0); // Descriptor: fcs_flag=3 (8 bytes)
        input.push(0x10); // Window descriptor
                          // 8 bytes of FCS
        input.extend_from_slice(&[0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert_eq!(header.frame_content_size, Some(0x1000));
    }

    #[test]
    fn test_read_frame_header_missing_window() {
        // Valid magic + descriptor but missing window
        let input = vec![0x28, 0xB5, 0x2F, 0xFD, 0x00]; // Magic + descriptor only
        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_frame_header_missing_fcs() {
        // Valid magic + single_segment descriptor but missing FCS
        let input = vec![0x28, 0xB5, 0x2F, 0xFD, 0x20]; // Magic + single_segment=1
        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_frame_header_checksum_flag() {
        // Valid magic + descriptor with checksum flag
        let mut input = vec![0x28, 0xB5, 0x2F, 0xFD]; // Magic
        input.push(0x24); // Descriptor: checksum=1, single_segment=1
        input.push(0x10); // FCS (1 byte)

        let mut pos = 0;
        let result = read_frame_header(&input, &mut pos);
        assert!(result.is_ok());
        let header = result.unwrap();
        assert!(header.checksum);
    }
}
