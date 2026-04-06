//! Tiled Q4_K GEMV Kernels with Shared Memory Input Caching
//!
//! - `TiledQ4KGemvKernel`: Input vector cached in shared memory
//! - `ChunkedTiledQ4KGemvKernel`: Handles K > 8K with fixed 32KB chunks

mod chunked;
mod shared_memory;

pub use chunked::ChunkedTiledQ4KGemvKernel;
pub use shared_memory::TiledQ4KGemvKernel;
