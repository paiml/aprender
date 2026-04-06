//! Daemon module - ublk I/O processing
//!
//! Handles block device I/O using direct ublk kernel interface.
//!
//! ## Batched GPU Compression
//!
//! This module implements batched compression to achieve >10 GB/s throughput:
//! - Pages are buffered until batch threshold (default 1000) is reached
//! - Large batches use GPU parallel compression via `GpuBatchCompressor`
//! - Small batches use SIMD parallel compression via rayon
//! - Background flush thread handles timeout-based flushes

#![allow(dead_code)]

mod batched;
mod entropy;
mod io;
mod page_store;
mod tiered;

// Re-export all public types to maintain the same external API
pub use batched::{spawn_flush_thread, BatchConfig, BatchedPageStore, BatchedPageStoreStats};
pub use io::{process_io, IoRequest, IoType};
pub use page_store::{PageStore, PageStoreStats, PageStoreTrait};
pub use tiered::{
    spawn_tiered_flush_thread, TieredConfig, TieredPageStore, TieredPageStoreStats,
};

#[cfg(test)]
mod tests;
