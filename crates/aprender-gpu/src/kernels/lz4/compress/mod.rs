//! GPU LZ4 Warp-Cooperative compression kernel
//!
//! Each warp (32 threads) processes one 4KB page cooperatively.
//! Block size is 128 threads = 4 warps = 4 pages per block.

mod ptx;
mod wgsl;

use super::{
    LZ4_HASH_BITS, LZ4_HASH_MULT, LZ4_HASH_SIZE, LZ4_MAX_OFFSET, LZ4_MIN_MATCH, PAGE_SIZE,
};

/// GPU LZ4 Warp-Cooperative compression kernel
///
/// Each warp (32 threads) processes one 4KB page cooperatively.
/// Block size is 128 threads = 4 warps = 4 pages per block.
#[derive(Debug, Clone)]
pub struct Lz4WarpCompressKernel {
    /// Number of pages in the batch
    batch_size: u32,
}

impl Lz4WarpCompressKernel {
    /// Create a new LZ4 warp-cooperative compression kernel
    #[must_use]
    pub fn new(batch_size: u32) -> Self {
        Self { batch_size }
    }

    /// Get the batch size
    #[must_use]
    pub fn batch_size(&self) -> u32 {
        self.batch_size
    }

    /// Calculate grid dimensions for the kernel launch
    #[must_use]
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        // 4 warps per block = 4 pages per block
        let pages_per_block = 4;
        let num_blocks = (self.batch_size + pages_per_block - 1) / pages_per_block;
        (num_blocks, 1, 1)
    }

    /// Calculate block dimensions
    #[must_use]
    pub fn block_dim(&self) -> (u32, u32, u32) {
        // 128 threads = 4 warps
        (128, 1, 1)
    }

    /// Calculate shared memory requirement per block
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        // 4 warps x (4KB page buffer + 8KB hash table) = 48KB
        4 * (PAGE_SIZE as usize + LZ4_HASH_SIZE as usize * 2)
    }
}

#[cfg(test)]
mod tests;
