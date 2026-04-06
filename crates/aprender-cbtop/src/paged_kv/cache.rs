//! PagedKvCache implementation with allocation, eviction, and copy-on-write.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use super::types::{
    BlockId, CacheStats, EvictionStrategy, PagedKvError, PagedKvResult, SeqId, SequenceInfo,
};

/// Paged KV cache for efficient memory management.
///
/// Based on vLLM's PagedAttention algorithm. Manages KV cache memory
/// using fixed-size blocks to prevent fragmentation and enable
/// efficient memory sharing.
#[derive(Debug)]
pub struct PagedKvCache {
    /// Block size (tokens per block)
    block_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Total number of physical blocks
    num_blocks: usize,
    /// Free block indices
    free_blocks: VecDeque<BlockId>,
    /// Sequence -> info mapping
    sequences: HashMap<SeqId, SequenceInfo>,
    /// Block reference counts (for COW)
    block_refs: HashMap<BlockId, u32>,
    /// Eviction strategy
    eviction_strategy: EvictionStrategy,
    /// Memory threshold for eviction (0.0-1.0)
    eviction_threshold: f64,
    /// Cache statistics
    stats: CacheStats,
}

impl PagedKvCache {
    /// Create a new PagedKvCache.
    ///
    /// # Arguments
    /// - `num_blocks`: Total number of physical blocks
    /// - `block_size`: Tokens per block
    /// - `num_heads`: Number of attention heads
    /// - `head_dim`: Dimension of each head
    pub fn new(num_blocks: usize, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        // Initialize free blocks
        let free_blocks: VecDeque<BlockId> = (0..num_blocks as u32).map(BlockId).collect();

        Self {
            block_size,
            num_heads,
            head_dim,
            num_blocks,
            free_blocks,
            sequences: HashMap::new(),
            block_refs: HashMap::new(),
            eviction_strategy: EvictionStrategy::default(),
            eviction_threshold: 0.9,
            stats: CacheStats::default(),
        }
    }

    /// Set eviction strategy.
    pub fn with_eviction_strategy(mut self, strategy: EvictionStrategy) -> Self {
        self.eviction_strategy = strategy;
        self
    }

    /// Set eviction threshold (0.0-1.0).
    pub fn with_eviction_threshold(mut self, threshold: f64) -> Self {
        self.eviction_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Get block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Get number of free blocks.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get number of used blocks.
    pub fn used_block_count(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Memory utilization percentage (0.0-1.0).
    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.used_block_count() as f64 / self.num_blocks as f64
    }

    /// Calculate memory for a block in bytes.
    pub fn block_memory_bytes(&self) -> usize {
        // KV cache: 2 (K+V) * block_size * num_heads * head_dim * 2 (f16)
        2 * self.block_size * self.num_heads * self.head_dim * 2
    }

    /// Total memory capacity in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.num_blocks * self.block_memory_bytes()
    }

    /// Used memory in bytes.
    pub fn used_memory_bytes(&self) -> usize {
        self.used_block_count() * self.block_memory_bytes()
    }

    /// Check if eviction is needed.
    pub fn needs_eviction(&self) -> bool {
        self.utilization() >= self.eviction_threshold
    }

    /// Get number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Get sequence info.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&SequenceInfo> {
        self.sequences.get(&seq_id)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get eviction strategy.
    pub fn eviction_strategy(&self) -> &EvictionStrategy {
        &self.eviction_strategy
    }

    /// Calculate blocks needed for tokens.
    fn blocks_needed(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    /// Allocate a single block.
    fn allocate_block(&mut self) -> PagedKvResult<BlockId> {
        if let Some(block_id) = self.free_blocks.pop_front() {
            self.block_refs.insert(block_id, 1);
            self.stats.total_allocations += 1;

            // Track peak usage
            let used = self.used_block_count();
            if used > self.stats.peak_blocks_used {
                self.stats.peak_blocks_used = used;
            }

            Ok(block_id)
        } else {
            Err(PagedKvError::OutOfMemory {
                requested: 1,
                available: 0,
            })
        }
    }

    /// Free a single block.
    fn free_block(&mut self, block_id: BlockId) -> PagedKvResult<()> {
        if let Some(refs) = self.block_refs.get_mut(&block_id) {
            *refs -= 1;
            if *refs == 0 {
                self.block_refs.remove(&block_id);
                self.free_blocks.push_back(block_id);
                self.stats.total_frees += 1;
            }
            Ok(())
        } else {
            Err(PagedKvError::BlockNotFound(block_id))
        }
    }

    /// Allocate blocks for a new sequence.
    pub fn allocate(&mut self, seq_id: SeqId, num_tokens: usize) -> PagedKvResult<()> {
        if self.sequences.contains_key(&seq_id) {
            return Err(PagedKvError::InvalidOperation(format!(
                "Sequence {} already exists",
                seq_id
            )));
        }

        let blocks_needed = self.blocks_needed(num_tokens);

        // Check if we have enough blocks
        if blocks_needed > self.free_blocks.len() {
            return Err(PagedKvError::OutOfMemory {
                requested: blocks_needed,
                available: self.free_blocks.len(),
            });
        }

        // Allocate blocks
        let mut block_ids = Vec::with_capacity(blocks_needed);
        for _ in 0..blocks_needed {
            block_ids.push(self.allocate_block()?);
        }

        // Create sequence info
        let mut seq_info = SequenceInfo::new(seq_id);
        seq_info.num_tokens = num_tokens;
        seq_info.block_ids = block_ids;
        seq_info.touch();

        self.sequences.insert(seq_id, seq_info);
        Ok(())
    }

    /// Append tokens to an existing sequence.
    pub fn append(&mut self, seq_id: SeqId, num_new_tokens: usize) -> PagedKvResult<()> {
        // First, calculate how many blocks we need (immutably)
        let (old_tokens, additional_blocks) = {
            let seq_info = self
                .sequences
                .get(&seq_id)
                .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

            let old_tokens = seq_info.num_tokens;
            let new_tokens = old_tokens + num_new_tokens;
            let old_blocks = self.blocks_needed(old_tokens);
            let new_blocks = self.blocks_needed(new_tokens);
            let additional = new_blocks.saturating_sub(old_blocks);

            (old_tokens, additional)
        };

        // Check if we have enough blocks
        if additional_blocks > self.free_blocks.len() {
            return Err(PagedKvError::OutOfMemory {
                requested: additional_blocks,
                available: self.free_blocks.len(),
            });
        }

        // Allocate the blocks
        let mut new_block_ids = Vec::with_capacity(additional_blocks);
        for _ in 0..additional_blocks {
            new_block_ids.push(self.allocate_block()?);
        }

        // Update sequence info
        let seq_info = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

        seq_info.block_ids.extend(new_block_ids);
        seq_info.num_tokens = old_tokens + num_new_tokens;
        seq_info.touch();
        Ok(())
    }

    /// Free all blocks for a sequence.
    pub fn free(&mut self, seq_id: SeqId) -> PagedKvResult<()> {
        let seq_info = self
            .sequences
            .remove(&seq_id)
            .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

        for block_id in seq_info.block_ids {
            self.free_block(block_id)?;
        }

        Ok(())
    }

    /// Copy-on-write fork for beam search.
    ///
    /// Creates a new sequence that shares blocks with the source sequence.
    /// Blocks are only copied when modified (copy-on-write).
    pub fn fork(&mut self, src_seq: SeqId, dst_seq: SeqId) -> PagedKvResult<()> {
        if self.sequences.contains_key(&dst_seq) {
            return Err(PagedKvError::InvalidOperation(format!(
                "Destination sequence {} already exists",
                dst_seq
            )));
        }

        let src_info = self
            .sequences
            .get(&src_seq)
            .ok_or(PagedKvError::SequenceNotFound(src_seq))?
            .clone();

        // Increment reference counts for shared blocks
        for block_id in &src_info.block_ids {
            if let Some(refs) = self.block_refs.get_mut(block_id) {
                *refs += 1;
            }
        }

        // Create new sequence with shared blocks
        let mut dst_info = SequenceInfo::new(dst_seq);
        dst_info.num_tokens = src_info.num_tokens;
        dst_info.block_ids = src_info.block_ids.clone();
        dst_info.touch();

        self.sequences.insert(dst_seq, dst_info);
        self.stats.total_forks += 1;
        Ok(())
    }

    /// Select sequence to evict based on strategy.
    pub fn select_eviction_target(&self) -> Option<SeqId> {
        if self.sequences.is_empty() {
            return None;
        }

        match &self.eviction_strategy {
            EvictionStrategy::LRU => {
                // Evict least recently used
                self.sequences
                    .values()
                    .min_by_key(|s| s.last_access)
                    .map(|s| s.seq_id)
            }
            EvictionStrategy::LFU => {
                // Evict least frequently used
                self.sequences
                    .values()
                    .min_by_key(|s| s.access_count)
                    .map(|s| s.seq_id)
            }
            EvictionStrategy::LongestFirst => {
                // Evict longest sequence (most blocks)
                self.sequences
                    .values()
                    .max_by_key(|s| s.num_tokens)
                    .map(|s| s.seq_id)
            }
            EvictionStrategy::Priority { .. } => {
                // Evict lowest priority
                self.sequences
                    .values()
                    .min_by_key(|s| s.priority)
                    .map(|s| s.seq_id)
            }
            EvictionStrategy::StreamingLLM { .. } => {
                // StreamingLLM doesn't evict sequences, it evicts tokens
                // For simplicity, fall back to LRU for sequence eviction
                self.sequences
                    .values()
                    .min_by_key(|s| s.last_access)
                    .map(|s| s.seq_id)
            }
        }
    }

    /// Evict a sequence to free memory.
    pub fn evict(&mut self) -> PagedKvResult<SeqId> {
        let target = self
            .select_eviction_target()
            .ok_or(PagedKvError::InvalidOperation(
                "No sequences to evict".to_string(),
            ))?;

        self.free(target)?;
        self.stats.total_evictions += 1;
        Ok(target)
    }

    /// Evict until memory utilization is below threshold.
    pub fn evict_to_threshold(&mut self, target_util: f64) -> PagedKvResult<Vec<SeqId>> {
        let mut evicted = Vec::new();
        while self.utilization() > target_util && !self.sequences.is_empty() {
            evicted.push(self.evict()?);
        }
        Ok(evicted)
    }

    /// Apply StreamingLLM eviction to a sequence.
    ///
    /// Keeps sink tokens at the beginning and a recent window at the end,
    /// evicting middle tokens.
    pub fn apply_streaming_llm(
        &mut self,
        seq_id: SeqId,
        sink_tokens: usize,
        window_tokens: usize,
    ) -> PagedKvResult<usize> {
        // Get sequence info immutably first to compute values
        let (num_tokens, blocks_to_remove) = {
            let seq_info = self
                .sequences
                .get(&seq_id)
                .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

            let keep_tokens = sink_tokens + window_tokens;
            if seq_info.num_tokens <= keep_tokens {
                return Ok(0); // Nothing to evict
            }

            let old_blocks = self.blocks_needed(seq_info.num_tokens);
            let new_blocks = self.blocks_needed(keep_tokens);
            let blocks_to_free = old_blocks.saturating_sub(new_blocks);

            // Collect blocks to remove
            let blocks: Vec<BlockId> = seq_info
                .block_ids
                .iter()
                .skip(sink_tokens / self.block_size + 1)
                .take(blocks_to_free)
                .cloned()
                .collect();

            (seq_info.num_tokens, blocks)
        };

        let keep_tokens = sink_tokens + window_tokens;
        let evict_tokens = num_tokens - keep_tokens;

        // Free the blocks
        for block_id in &blocks_to_remove {
            self.free_block(*block_id)?;
        }

        // Update sequence info
        if let Some(seq_info) = self.sequences.get_mut(&seq_id) {
            for block_id in blocks_to_remove {
                seq_info.block_ids.retain(|&id| id != block_id);
            }
            seq_info.num_tokens = keep_tokens;
        }

        Ok(evict_tokens)
    }

    /// Get all sequence IDs.
    pub fn sequence_ids(&self) -> Vec<SeqId> {
        self.sequences.keys().cloned().collect()
    }
}

impl fmt::Display for PagedKvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PagedKvCache")?;
        writeln!(
            f,
            "  Strategy: {} (block_size={})",
            self.eviction_strategy, self.block_size
        )?;
        writeln!(
            f,
            "  Blocks: {}/{} ({:.1}% used)",
            self.used_block_count(),
            self.num_blocks,
            self.utilization() * 100.0
        )?;
        writeln!(
            f,
            "  Memory: {:.2} MB / {:.2} MB",
            self.used_memory_bytes() as f64 / 1_000_000.0,
            self.total_memory_bytes() as f64 / 1_000_000.0
        )?;
        writeln!(f, "  Sequences: {} active", self.num_sequences())?;
        writeln!(
            f,
            "  Stats: allocs={}, frees={}, evictions={}, forks={}",
            self.stats.total_allocations,
            self.stats.total_frees,
            self.stats.total_evictions,
            self.stats.total_forks
        )?;
        Ok(())
    }
}
