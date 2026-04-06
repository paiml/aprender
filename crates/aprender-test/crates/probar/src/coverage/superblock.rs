//! Superblock Tiling (Heijunka)
//!
//! Per spec §5.4.1: The Granularity Problem
//!
//! If a coverage block represents a single basic block (~3 instructions),
//! the overhead of work-stealing scheduler operations exceeds the work itself.
//!
//! Solution: Group localized basic blocks into a single schedulable unit
//! (Superblock) to amortize scheduling overhead.

use super::{BlockId, FunctionId};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Superblock ID (distinct from BlockId for type safety)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SuperblockId(u32);

impl SuperblockId {
    /// Create a new superblock ID
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the inner value
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl Hash for SuperblockId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Superblock: A tile of related basic blocks (Kaizen: amortize scheduling)
///
/// Inspired by RenderMan's bucket system - group spatially local work
/// to reduce coordination overhead.
#[derive(Debug, Clone)]
pub struct Superblock {
    /// Superblock ID
    id: SuperblockId,
    /// Contained basic blocks
    blocks: Vec<BlockId>,
    /// Block set for O(1) lookup
    block_set: HashSet<BlockId>,
    /// Estimated execution cost (for load balancing)
    cost: u64,
    /// Parent function (for locality)
    function: FunctionId,
}

impl Superblock {
    /// Create a new superblock
    #[must_use]
    pub fn new(id: SuperblockId, blocks: Vec<BlockId>, function: FunctionId) -> Self {
        let block_set: HashSet<BlockId> = blocks.iter().copied().collect();
        let cost = blocks.len() as u64;
        Self {
            id,
            blocks,
            block_set,
            cost,
            function,
        }
    }

    /// Get the superblock ID
    #[inline]
    #[must_use]
    pub fn id(&self) -> SuperblockId {
        self.id
    }

    /// Get the number of blocks in this superblock
    #[inline]
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Check if this superblock contains a specific block
    #[inline]
    #[must_use]
    pub fn contains(&self, block: BlockId) -> bool {
        self.block_set.contains(&block)
    }

    /// Get all blocks in this superblock
    #[must_use]
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Get the estimated execution cost
    #[inline]
    #[must_use]
    pub fn cost_estimate(&self) -> u64 {
        self.cost
    }

    /// Set a custom cost estimate (e.g., from profiling)
    pub fn set_cost_estimate(&mut self, cost: u64) {
        self.cost = cost;
    }

    /// Get the parent function
    #[inline]
    #[must_use]
    pub fn function(&self) -> FunctionId {
        self.function
    }

    /// Iterate over blocks
    pub fn iter(&self) -> impl Iterator<Item = &BlockId> {
        self.blocks.iter()
    }
}

/// Superblock builder: Groups blocks by function or loop
///
/// Default configuration: 64 blocks per superblock (empirically optimal
/// for amortizing work-stealing overhead).
#[derive(Debug, Clone)]
pub struct SuperblockBuilder {
    /// Target blocks per superblock (amortization factor)
    target_size: usize,
    /// Maximum blocks per superblock (memory bound)
    max_size: usize,
    /// Next superblock ID
    next_id: u32,
}

impl SuperblockBuilder {
    /// Create a new builder with default settings
    ///
    /// Default: 64 blocks per superblock
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_size: 64,
            max_size: 256,
            next_id: 0,
        }
    }

    /// Set the target number of blocks per superblock
    #[must_use]
    pub fn with_target_size(mut self, size: usize) -> Self {
        self.target_size = size;
        self
    }

    /// Set the maximum number of blocks per superblock
    #[must_use]
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    /// Build superblocks from a list of blocks
    ///
    /// Groups blocks into superblocks of target_size (capped at max_size).
    #[must_use]
    pub fn build_from_blocks(&self, blocks: &[BlockId], function: FunctionId) -> Vec<Superblock> {
        if blocks.is_empty() {
            return Vec::new();
        }

        let effective_size = self.target_size.min(self.max_size);
        let mut superblocks = Vec::new();
        let mut next_id = self.next_id;

        for chunk in blocks.chunks(effective_size) {
            let id = SuperblockId::new(next_id);
            next_id += 1;
            superblocks.push(Superblock::new(id, chunk.to_vec(), function));
        }

        superblocks
    }

    /// Build superblocks from blocks grouped by function
    ///
    /// Each function's blocks are grouped separately, respecting function boundaries.
    #[must_use]
    pub fn build_from_function_blocks(
        &self,
        function_blocks: &[(FunctionId, Vec<BlockId>)],
    ) -> Vec<Superblock> {
        let mut all_superblocks = Vec::new();

        for (function, blocks) in function_blocks {
            let superblocks = self.build_from_blocks(blocks, *function);
            all_superblocks.extend(superblocks);
        }

        all_superblocks
    }

    /// Get the next superblock ID that would be assigned
    #[must_use]
    pub fn next_id(&self) -> u32 {
        self.next_id
    }
}

impl Default for SuperblockBuilder {
    fn default() -> Self {
        Self::new()
    }
}
