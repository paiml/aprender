//! PagedKvCache Implementation (PMAT-014)
//!
//! Implements PagedAttention-style KV cache management per cbtop spec S18.
//!
//! # Overview
//!
//! PagedAttention manages KV cache memory using fixed-size blocks, enabling:
//! - Dynamic memory allocation without fragmentation
//! - Copy-on-write for beam search
//! - Efficient eviction under memory pressure
//!
//! # Citations
//!
//! - [Kwon et al. 2023] "Efficient Memory Management for LLM Serving with PagedAttention" SOSP
//! - [Xiao et al. 2023] "StreamingLLM: Efficient Streaming with Attention Sinks" arXiv
//! - [Yu et al. 2022] "ORCA: A Distributed Serving System for Transformer-Based Models" OSDI

mod cache;
mod types;

pub use cache::PagedKvCache;
pub use types::{
    BlockId, CacheStats, EvictionStrategy, KvBlock, PagedKvError, PagedKvResult, SeqId,
    SequenceInfo,
};

#[cfg(test)]
mod tests;
