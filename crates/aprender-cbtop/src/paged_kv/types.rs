//! Core types for paged KV cache: identifiers, blocks, sequences, errors, and strategies.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

/// Block identifier (index into physical_blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B{}", self.0)
    }
}

/// Sequence identifier (request ID).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub u64);

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

/// KV cache eviction strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Evict longest sequences first
    LongestFirst,
    /// Evict by priority (preempt low-priority requests)
    Priority { levels: usize },
    /// StreamingLLM: keep sink tokens + recent window
    StreamingLLM {
        sink_tokens: usize,
        window_tokens: usize,
    },
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        EvictionStrategy::LRU
    }
}

impl fmt::Display for EvictionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvictionStrategy::LRU => write!(f, "LRU"),
            EvictionStrategy::LFU => write!(f, "LFU"),
            EvictionStrategy::LongestFirst => write!(f, "LongestFirst"),
            EvictionStrategy::Priority { levels } => write!(f, "Priority({})", levels),
            EvictionStrategy::StreamingLLM {
                sink_tokens,
                window_tokens,
            } => {
                write!(
                    f,
                    "StreamingLLM(sink={}, window={})",
                    sink_tokens, window_tokens
                )
            }
        }
    }
}

/// Single KV cache block (logical representation).
///
/// In a real implementation, this would contain GPU buffers:
/// ```ignore
/// pub keys: DeviceBuffer<f16>,   // [block_size, num_heads, head_dim]
/// pub values: DeviceBuffer<f16>, // [block_size, num_heads, head_dim]
/// ```
#[derive(Debug)]
pub struct KvBlock {
    /// Block ID
    pub id: BlockId,
    /// Number of tokens stored in this block (0 to block_size)
    pub num_tokens: usize,
    /// Reference count (for copy-on-write)
    pub ref_count: AtomicU32,
    /// Block size (max tokens)
    pub capacity: usize,
}

impl KvBlock {
    /// Create a new empty block.
    pub fn new(id: BlockId, capacity: usize) -> Self {
        Self {
            id,
            num_tokens: 0,
            ref_count: AtomicU32::new(1),
            capacity,
        }
    }

    /// Check if block is full.
    pub fn is_full(&self) -> bool {
        self.num_tokens >= self.capacity
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.num_tokens)
    }

    /// Get reference count.
    pub fn refs(&self) -> u32 {
        self.ref_count.load(Ordering::Acquire)
    }

    /// Increment reference count.
    pub fn inc_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Decrement reference count, returns true if count reached zero.
    pub fn dec_ref(&self) -> bool {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

/// Sequence metadata for tracking access patterns.
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    /// Sequence ID
    pub seq_id: SeqId,
    /// Total tokens in sequence
    pub num_tokens: usize,
    /// Blocks allocated to this sequence
    pub block_ids: Vec<BlockId>,
    /// Last access timestamp
    pub last_access: Instant,
    /// Access count (for LFU)
    pub access_count: u64,
    /// Priority level (for priority-based eviction)
    pub priority: u32,
}

impl SequenceInfo {
    /// Create new sequence info.
    pub fn new(seq_id: SeqId) -> Self {
        Self {
            seq_id,
            num_tokens: 0,
            block_ids: Vec::new(),
            last_access: Instant::now(),
            access_count: 0,
            priority: 0,
        }
    }

    /// Update access timestamp.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }

    /// Memory usage in blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_ids.len()
    }
}

/// PagedKvCache error types.
#[derive(Debug, Clone)]
pub enum PagedKvError {
    /// Out of memory (no free blocks)
    OutOfMemory { requested: usize, available: usize },
    /// Sequence not found
    SequenceNotFound(SeqId),
    /// Block not found
    BlockNotFound(BlockId),
    /// Invalid operation
    InvalidOperation(String),
}

impl fmt::Display for PagedKvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PagedKvError::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Out of memory: requested {} blocks, {} available",
                    requested, available
                )
            }
            PagedKvError::SequenceNotFound(seq_id) => {
                write!(f, "Sequence not found: {}", seq_id)
            }
            PagedKvError::BlockNotFound(block_id) => {
                write!(f, "Block not found: {}", block_id)
            }
            PagedKvError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for PagedKvError {}

/// Result type for PagedKvCache operations.
pub type PagedKvResult<T> = Result<T, PagedKvError>;

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total allocations
    pub total_allocations: u64,
    /// Total frees
    pub total_frees: u64,
    /// Total evictions
    pub total_evictions: u64,
    /// Total forks (copy-on-write)
    pub total_forks: u64,
    /// Peak blocks used
    pub peak_blocks_used: usize,
}
