//! GPU Memory Management
//!
//! Provides safe RAII wrappers for GPU memory allocation and transfer.
//!
//! # Design Philosophy
//!
//! - **RAII**: Memory automatically freed on drop
//! - **Type Safety**: Generic over element type with size tracking
//! - **Async Support**: Both sync and async transfer methods
//!
//! # Citation
//!
//! [4] Oden & Fröning (HiPC 2013) analyzes cudaMalloc latency (1-10ms),
//!     motivating our pool allocator design in memory/pool.rs.

mod buffer;
mod transfer;

#[allow(unused_imports, unreachable_pub)] // Re-exported by driver/mod.rs
pub use buffer::{
    device_bytes_outstanding, device_memory_exclusive, DeviceMemoryExclusive, GpuBuffer,
    GpuBufferView,
};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
