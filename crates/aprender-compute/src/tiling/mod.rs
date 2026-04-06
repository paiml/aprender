//! Tiling Compute Blocks (TCB) - Work Partitioning for High-Performance Kernels
//!
//! TCBs represent the fundamental unit of work partitioning within `ComputeBrick` kernels.
//! While a `ComputeBrick` defines a logical operation (e.g., Q4_K MatMul), a TCB defines
//! the physical execution strategy—how data is partitioned across the memory hierarchy.
//!
//! # Architecture
//!
//! Tiling occurs at three levels:
//! 1. **Macro-Tile (L3/Global Memory)**: Partitioning across CPU sockets or GPU SMs
//! 2. **Midi-Tile (L2/Shared Memory)**: Partitioning within a thread block or Rayon task
//! 3. **Micro-Tile (Registers)**: Smallest unit processed by SIMD or CUDA warps
//!
//! # Modules
//!
//! - `geometry` - TcbGeometry dimensions and level definitions
//! - `config` - TilingConfig and backend selection
//! - `calculator` - TcbIndexCalculator for index computation
//! - `packing` - Memory layout packing utilities
//! - `prefetch` - Prefetch locality hints
//! - `q4k_matvec` - Q4_K quantized matrix-vector tiling
//! - `error` - TilingError types

mod calculator;
mod config;
mod error;
mod geometry;
mod packing;
mod prefetch;
mod q4k_matvec;

pub use calculator::TcbIndexCalculator;
pub use config::{TilingBackend, TilingConfig};
pub use error::TilingError;
pub use geometry::{TcbGeometry, TcbLevel};
pub use packing::{pack_a_index, pack_b_index, swizzle_index, PackingLayout};
pub use prefetch::{optimal_prefetch_distance, PrefetchLocality};
pub use q4k_matvec::{
    extract_scale_min_6bit, f16_to_f32, TiledQ4KMatvec, TilingStats, Q4K_SUPERBLOCK_BYTES,
    Q4K_SUPERBLOCK_SIZE,
};

#[cfg(test)]
mod tests;
