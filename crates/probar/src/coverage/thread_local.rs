//! Thread-Local Counter Buffering (Muda Elimination)
//!
//! Per spec §5.3.1: Workers increment local registers and flush to global
//! counters only upon block exit or checkpoint. This reduces bus contention
//! from O(N) to O(B) where N is instructions and B is block transitions.

use super::BlockId;

/// Thread-local counter buffer (Kaizen: eliminates atomic contention)
///
/// Instead of atomic increments on every block hit, counters are buffered
/// locally and flushed periodically. This dramatically reduces cache
/// coherence traffic in parallel coverage collection.
#[derive(Debug)]
pub struct ThreadLocalCounters {
    /// Local counter buffer (one per block)
    local: Vec<u64>,
    /// Flush threshold (blocks executed before sync)
    flush_threshold: usize,
    /// Blocks since last flush
    blocks_since_flush: usize,
    /// Number of flushes performed
    flush_count: usize,
}

impl ThreadLocalCounters {
    /// Create new thread-local counters for the given number of blocks
    ///
    /// Uses a default flush threshold of 1000 blocks.
    #[must_use]
    pub fn new(block_count: usize) -> Self {
        Self {
            local: vec![0; block_count],
            flush_threshold: 1000,
            blocks_since_flush: 0,
            flush_count: 0,
        }
    }

    /// Create with a custom flush threshold
    ///
    /// Lower thresholds mean more frequent flushes (more accurate but slower).
    /// Higher thresholds mean fewer flushes (faster but less accurate intermediate state).
    #[must_use]
    pub fn with_flush_threshold(block_count: usize, threshold: usize) -> Self {
        Self {
            local: vec![0; block_count],
            flush_threshold: threshold,
            blocks_since_flush: 0,
            flush_count: 0,
        }
    }

    /// Increment counter locally (no atomic, no cache coherence traffic)
    ///
    /// This is the hot path - must be as fast as possible.
    #[inline(always)]
    pub fn increment(&mut self, block: BlockId) {
        let idx = block.as_u32() as usize;
        if idx < self.local.len() {
            self.local[idx] += 1;
            self.blocks_since_flush += 1;

            // Amortize flush cost over many increments
            if self.blocks_since_flush >= self.flush_threshold {
                self.internal_flush();
            }
        }
    }

    /// Get the current local count for a block (before flush)
    #[inline]
    #[must_use]
    pub fn get(&self, block: BlockId) -> u64 {
        let idx = block.as_u32() as usize;
        self.local.get(idx).copied().unwrap_or(0)
    }

    /// Flush local counters and return the counts
    ///
    /// Returns the accumulated counts and resets the local buffer.
    #[must_use]
    pub fn flush(&mut self) -> Vec<u64> {
        let result = self.local.clone();
        self.local.fill(0);
        self.blocks_since_flush = 0;
        self.flush_count += 1;
        result
    }

    /// Internal flush that just resets state (for automatic threshold flush)
    fn internal_flush(&mut self) {
        // In a real implementation, this would atomically add to global counters
        // For now, we just track that a flush occurred
        self.blocks_since_flush = 0;
        self.flush_count += 1;
    }

    /// Get the number of times flush has been called
    #[must_use]
    pub fn flush_count(&self) -> usize {
        self.flush_count
    }

    /// Get the number of blocks being tracked
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.local.len()
    }

    /// Check if any blocks have been hit since last flush
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.blocks_since_flush > 0
    }

    /// Get the flush threshold
    #[must_use]
    pub fn flush_threshold(&self) -> usize {
        self.flush_threshold
    }
}

impl Default for ThreadLocalCounters {
    fn default() -> Self {
        Self::new(0)
    }
}
