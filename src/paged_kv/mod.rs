//! PagedAttention KV Cache Management
//!
//! Per spec §8.1: Efficient KV cache management based on vLLM's PagedAttention.
//! Reference: [4] Kwon et al. (2023) "Efficient Memory Management for LLM Serving"
//!
//! ## Key Features
//!
//! - **Physical Pages**: Fixed-size memory blocks for KV cache storage
//! - **Page Tables**: Logical to physical page mapping per sequence
//! - **Copy-on-Write**: Efficient prefix sharing between sequences
//! - **Dynamic Allocation**: Pages allocated on-demand during generation
//!
//! ## Memory Layout
//!
//! ```text
//! Physical Page (block_size tokens):
//! ┌─────────────────────────────────────────┐
//! │  K: [block_size, num_heads, head_dim]   │
//! │  V: [block_size, num_heads, head_dim]   │
//! └─────────────────────────────────────────┘
//! ```

// Module-level clippy allows
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Error type for PagedKvCache operations
#[derive(Debug, Error)]
pub enum PagedCacheError {
    /// Out of memory - no free pages available
    #[error("Out of memory: need {needed} pages, have {available}")]
    OutOfMemory {
        /// Number of pages needed
        needed: usize,
        /// Number of pages available
        available: usize,
    },

    /// Sequence not found in page table
    #[error("Sequence not found: {0}")]
    SequenceNotFound(u64),

    /// Invalid page access
    #[error("Invalid page access: page {page_id} at offset {offset}")]
    InvalidPageAccess {
        /// Page ID accessed
        page_id: u64,
        /// Offset within page
        offset: usize,
    },

    /// Page table corruption
    #[error("Page table corruption for sequence {seq_id}")]
    PageTableCorruption {
        /// Sequence ID
        seq_id: u64,
    },

    /// Unknown sequence ID
    #[error("Unknown sequence: {0:?}")]
    UnknownSequence(SeqId),

    /// Position out of bounds for sequence
    #[error("Out of bounds: seq {seq_id:?} pos {position}, allocated {allocated}")]
    OutOfBounds {
        /// Sequence ID
        seq_id: SeqId,
        /// Requested position
        position: usize,
        /// Allocated capacity
        allocated: usize,
    },

    /// Not enough free pages
    #[error("Out of pages: need {requested}, have {available}")]
    OutOfPages {
        /// Pages requested
        requested: usize,
        /// Pages available
        available: usize,
    },
}

/// Unique sequence identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SeqId(u64);

impl SeqId {
    /// Create a new unique sequence ID
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    pub fn value(&self) -> u64 {
        self.0
    }
}

impl Default for SeqId {
    fn default() -> Self {
        Self::new()
    }
}

/// Physical page identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageId(u64);

impl PageId {
    /// Create a new page ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// KV cache data for a single page
#[derive(Debug, Clone)]
pub struct KvPage {
    /// Page identifier
    pub id: PageId,
    /// Key cache: [block_size, num_heads, head_dim]
    pub keys: Vec<f32>,
    /// Value cache: [block_size, num_heads, head_dim]
    pub values: Vec<f32>,
    /// Number of tokens currently stored in this page
    pub num_tokens: usize,
    /// Reference count for copy-on-write
    pub ref_count: usize,
}

impl KvPage {
    /// Create a new empty KV page
    pub fn new(id: PageId, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let page_size = block_size * num_heads * head_dim;
        Self {
            id,
            keys: vec![0.0; page_size],
            values: vec![0.0; page_size],
            num_tokens: 0,
            ref_count: 1,
        }
    }

    /// Check if page is full
    pub fn is_full(&self, block_size: usize) -> bool {
        self.num_tokens >= block_size
    }

    /// Check if page is shared (copy-on-write)
    pub fn is_shared(&self) -> bool {
        self.ref_count > 1
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self, block_size: usize) -> usize {
        block_size.saturating_sub(self.num_tokens)
    }
}

/// PagedAttention KV cache manager
/// Reference: [4] Kwon et al. (2023) "Efficient Memory Management for LLM Serving"
pub struct PagedKvCache {
    /// Physical pages (fixed-size blocks)
    physical_pages: Vec<KvPage>,
    /// Logical to physical page mapping (per sequence)
    page_tables: HashMap<SeqId, Vec<PageId>>,
    /// Free page list
    free_pages: VecDeque<PageId>,
    /// Tokens per page (block size)
    block_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Total pages allocated
    total_pages: usize,
    /// Statistics
    stats: PagedCacheStats,
}

/// Statistics for PagedKvCache
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PagedCacheStats {
    /// Total sequences allocated
    pub sequences_allocated: u64,
    /// Total sequences freed
    pub sequences_freed: u64,
    /// Total pages allocated
    pub pages_allocated: u64,
    /// Total pages freed
    pub pages_freed: u64,
    /// Current active sequences
    pub active_sequences: u64,
    /// Current used pages
    pub used_pages: u64,
    /// Copy-on-write operations
    pub cow_operations: u64,
    /// Defragmentation operations performed
    pub defrag_operations: u64,
    /// Pages moved during defragmentation
    pub pages_moved: u64,
}

/// Fragmentation statistics for KV cache
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FragmentationStats {
    /// Number of free page "holes" between used pages
    pub holes: usize,
    /// Total wasted capacity in partially-filled pages (tokens)
    pub wasted_capacity: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented)
    pub fragmentation_ratio: f32,
    /// Largest contiguous free region (in pages)
    pub largest_free_region: usize,
    /// Average tokens per page (efficiency metric)
    pub avg_tokens_per_page: f32,
}

include!("contiguous.rs");
include!("mod_compute_prefix.rs");
include!("mod_quantized.rs");
include!("mod_quantized_paged.rs");
