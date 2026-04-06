//! Paged/Incremental Attention Kernels for VRAM-bound block management.
//!
//! This module implements incremental attention kernels optimized for
//! autoregressive LLM decoding. Unlike FlashAttention which tiles SRAM,
//! these kernels manage GPU-resident KV caches with efficient block access.
//!
//! ## Kernels
//!
//! - **IncrementalAttentionKernel**: Single-query (M=1) autoregressive attention
//! - **MultiWarpIncrementalAttentionKernel**: Multi-warp version for larger sequences
//! - **BatchedIncrementalAttentionKernel**: Batched incremental attention
//! - **FlashDecodingChunkKernel**: Split-K parallel decoding chunks
//! - **FlashDecodingReduceKernel**: Reduction kernel for Flash Decoding
//!
//! ## References
//!
//! - [Kwon2023] PagedAttention for LLM Serving with vLLM
//! - Flash Decoding (Split-K) for parallel sequence processing

mod batched;
mod flash_decoding;
mod incremental;
mod multi_warp;

pub use batched::BatchedIncrementalAttentionKernel;
pub use flash_decoding::{
    FlashDecodingChunkKernel, FlashDecodingChunkKernel2Warp, FlashDecodingReduceKernel,
    FLASH_DECODE_CHUNK_SIZE,
};
pub use incremental::IncrementalAttentionKernel;
pub use multi_warp::MultiWarpIncrementalAttentionKernel;
