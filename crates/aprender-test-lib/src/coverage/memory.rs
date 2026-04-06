//! Zero-Copy WASM Memory View (Muda Elimination)
//!
//! Per spec §5.3: Zero-copy coverage collection eliminates serialization waste.
//!
//! Direct access to WASM linear memory for reading coverage counters
//! without any copying or serialization overhead.

use super::BlockId;

/// Zero-copy WASM memory view for coverage (Muda elimination)
///
/// Provides direct access to coverage counters stored in WASM linear memory.
/// Counters are stored as little-endian u64 values.
#[derive(Debug)]
pub struct CoverageMemoryView<'a> {
    /// Direct pointer to WASM linear memory
    memory: &'a [u8],
    /// Counter array offset in bytes
    counter_base: usize,
    /// Number of blocks (counters)
    block_count: usize,
}

impl<'a> CoverageMemoryView<'a> {
    /// Create a new memory view
    ///
    /// # Arguments
    ///
    /// * `memory` - Reference to WASM linear memory
    /// * `counter_base` - Byte offset where counters start
    /// * `block_count` - Number of blocks/counters
    #[must_use]
    pub fn new(memory: &'a [u8], counter_base: usize, block_count: usize) -> Self {
        Self {
            memory,
            counter_base,
            block_count,
        }
    }

    /// Read counter without copy (Genchi Genbutsu)
    ///
    /// Directly reads the u64 counter value for the given block.
    #[inline]
    #[must_use]
    pub fn read_counter(&self, block: BlockId) -> u64 {
        let idx = block.as_u32() as usize;
        if idx >= self.block_count {
            return 0;
        }

        let offset = self.counter_base + idx * 8;
        if offset + 8 > self.memory.len() {
            return 0;
        }

        let bytes = &self.memory[offset..offset + 8];
        u64::from_le_bytes(bytes.try_into().unwrap_or([0; 8]))
    }

    /// SIMD batch read all counters
    ///
    /// Reads all counters at once. In a full implementation, this would
    /// use Trueno's SIMD operations for acceleration.
    #[must_use]
    pub fn read_all_counters(&self) -> Vec<u64> {
        let slice = &self.memory[self.counter_base..];
        slice
            .chunks_exact(8)
            .take(self.block_count)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap_or([0; 8])))
            .collect()
    }

    /// Get the number of blocks being tracked
    #[inline]
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    /// Get the counter base offset
    #[inline]
    #[must_use]
    pub fn counter_base(&self) -> usize {
        self.counter_base
    }

    /// Get the total memory size
    #[inline]
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.memory.len()
    }

    /// Check if a block is covered (counter > 0)
    #[inline]
    #[must_use]
    pub fn is_covered(&self, block: BlockId) -> bool {
        self.read_counter(block) > 0
    }

    /// Count the number of covered blocks
    #[must_use]
    pub fn covered_count(&self) -> usize {
        self.read_all_counters().iter().filter(|&&c| c > 0).count()
    }

    /// Calculate coverage percentage
    #[must_use]
    pub fn coverage_percent(&self) -> f64 {
        if self.block_count == 0 {
            return 100.0; // Vacuously true
        }
        (self.covered_count() as f64 / self.block_count as f64) * 100.0
    }
}
