//! Zstandard compression implementation.
//!
//! This module provides a pure Rust implementation of Zstandard (zstd) compression
//! as specified in RFC 8878.

mod compress;
mod decompress;

// FSE and Huffman are public for advanced use and testing
pub mod fse;
pub mod huffman;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

pub use compress::compress;
pub use decompress::decompress;

/// Zstd magic number (little-endian).
pub const ZSTD_MAGIC: u32 = 0xFD2FB528;

/// Frame header descriptor.
#[derive(Debug, Clone, Copy)]
pub struct FrameHeader {
    /// Window size in bytes.
    pub window_size: u64,
    /// Original (uncompressed) size if known.
    pub frame_content_size: Option<u64>,
    /// Dictionary ID if present.
    pub dictionary_id: Option<u32>,
    /// Whether checksum is present.
    pub checksum: bool,
    /// Whether this is a single segment.
    pub single_segment: bool,
}

/// Block type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    /// Raw block (uncompressed).
    Raw,
    /// RLE block (single byte repeated).
    Rle,
    /// Compressed block.
    Compressed,
    /// Reserved (invalid).
    Reserved,
}

impl From<u8> for BlockType {
    fn from(value: u8) -> Self {
        match value & 0x03 {
            0 => Self::Raw,
            1 => Self::Rle,
            2 => Self::Compressed,
            _ => Self::Reserved,
        }
    }
}

/// Block header.
#[derive(Debug, Clone, Copy)]
pub struct BlockHeader {
    /// Block type.
    pub block_type: BlockType,
    /// Whether this is the last block.
    pub last_block: bool,
    /// Block size in bytes.
    pub block_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PAGE_SIZE;

    #[test]
    fn test_roundtrip_zeros() {
        let input = [0u8; PAGE_SIZE];
        let compressed = compress(&input, 3).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_pattern() {
        let mut input = [0u8; PAGE_SIZE];
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let compressed = compress(&input, 3).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_text() {
        let mut input = [0u8; PAGE_SIZE];
        let text = b"The quick brown fox jumps over the lazy dog. ";
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = text[i % text.len()];
        }
        let compressed = compress(&input, 3).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_compression_level() {
        let input = [0u8; PAGE_SIZE];

        let c1 = compress(&input, 1).unwrap();
        let c3 = compress(&input, 3).unwrap();

        // Both should decompress correctly
        let mut output = [0u8; PAGE_SIZE];
        decompress(&c1, &mut output).unwrap();
        assert_eq!(input, output);

        decompress(&c3, &mut output).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_block_type_conversion() {
        assert_eq!(BlockType::from(0), BlockType::Raw);
        assert_eq!(BlockType::from(1), BlockType::Rle);
        assert_eq!(BlockType::from(2), BlockType::Compressed);
        assert_eq!(BlockType::from(3), BlockType::Reserved);
    }
}
