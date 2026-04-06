//! GPU LZ4 Compression Kernel (Pure Rust PTX Generation)
//!
//! Implements Warp-per-Page architecture for high-throughput LZ4 compression.
//! Each 4KB page is processed by a single warp (32 threads) cooperatively.
//!
//! ## Algorithm Overview (from LZ4 Block Format)
//!
//! LZ4 encodes data as sequences of:
//! - **Literals**: Raw uncompressed bytes
//! - **Matches**: Back-references to previously seen data (offset + length)
//!
//! Token format: `[4-bit literal length][4-bit match length]`
//! - Minimum match length is 4 bytes (MINMATCH)
//!
//! ## Warp-Cooperative Strategy
//!
//! 1. **Shared Memory Load**: All 32 threads load 128 bytes each (4KB total)
//! 2. **Hash Table**: Hash table in shared memory for match finding
//! 3. **Parallel Match Search**: Each thread checks different positions
//! 4. **Leader Encoding**: Lane 0 encodes tokens sequentially

mod compress;
mod cpu;
mod decompress;

pub use compress::Lz4WarpCompressKernel;
pub use cpu::{
    lz4_compress_block, lz4_decompress_block, lz4_encode_sequence, lz4_hash, lz4_hash_at,
    lz4_match_length, read_u32_le,
};
pub use decompress::Lz4WarpDecompressKernel;

/// LZ4 minimum match length (per LZ4 block format spec)
pub const LZ4_MIN_MATCH: u32 = 4;
/// LZ4 maximum match length: 255 + 15 + 4 = 274 bytes
pub const LZ4_MAX_MATCH: u32 = 255 + 15 + 4;
/// Number of bits for hash table indexing (4096 entries)
pub const LZ4_HASH_BITS: u32 = 12;
/// Hash table size in entries (1 << 12 = 4096)
pub const LZ4_HASH_SIZE: u32 = 1 << LZ4_HASH_BITS;
/// Page size for ZRAM compression (4KB)
pub const PAGE_SIZE: u32 = 4096;
/// LZ4 hash multiplier (Knuth multiplicative hash constant)
pub const LZ4_HASH_MULT: u32 = 2_654_435_761;
/// Maximum offset for LZ4 match (64KB - 1)
pub const LZ4_MAX_OFFSET: u32 = 65535;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f058_lz4_constants() {
        assert_eq!(LZ4_MIN_MATCH, 4);
        assert_eq!(LZ4_HASH_BITS, 12);
        assert_eq!(LZ4_HASH_SIZE, 4096);
        assert_eq!(PAGE_SIZE, 4096);
    }

    #[test]
    fn test_lz4_compress_constants() {
        // Verify constants are correct per LZ4 spec
        assert_eq!(LZ4_MIN_MATCH, 4);
        assert_eq!(LZ4_HASH_SIZE, 4096);
        assert_eq!(LZ4_MAX_OFFSET, 65535);
    }
}
