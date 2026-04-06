//! GPU acceleration for graph algorithms
//!
//! Based on research from:
//! - **Gunrock** (Wang et al., ACM `ToPC` 2017) - GPU graph traversal primitives
//! - **`cuGraph`** (Bader et al., 2022) - GPU-accelerated graph analytics
//! - **`GraphBLAST`** (Yang et al., 2022) - GPU linear algebra for graphs
//! - **Umbra** (Neumann & Freitag, CIDR 2020) - Morsel-driven parallelism
//! - **Funke et al.** (SIGMOD 2018) - GPU memory paging
//!
//! # Architecture
//!
//! - `device`: GPU device initialization and management
//! - `buffer`: GPU buffer management for CSR data
//! - `memory`: VRAM detection and memory limits
//! - `cache`: LRU cache for graph tiles
//! - `paging`: Graph tiling and paging coordinator
//! - `kernels`: WGSL compute shaders for BFS, `PageRank`, etc.
//!
//! # Feature Flag
//!
//! This module is only available with the `gpu` feature flag:
//! ```bash
//! cargo build --features gpu
//! ```

mod bfs;
mod buffer;
mod cache;
mod device;
mod memory;
mod paged_bfs;
mod pagerank;
mod paging;

pub use bfs::{gpu_bfs, GpuBfsResult};
pub use buffer::GpuCsrBuffers;
pub use cache::LruTileCache;
pub use device::{GpuDevice, GpuDeviceError};
pub use memory::{GpuMemoryLimits, DEFAULT_MORSEL_SIZE};
pub use paged_bfs::gpu_bfs_paged;
pub use pagerank::{gpu_pagerank, GpuPageRankResult};
pub use paging::{GraphTile, PagingCoordinator};
