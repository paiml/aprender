//! GPU-resident attention operations for transformer architectures.
//!
//! This module contains batched and incremental multi-head attention implementations
//! that operate entirely on GPU with zero intermediate host transfers.
//!
//! # Implementations
//!
//! - `batched_multihead_attention` - Standard per-head attention processing
//! - `batched_multihead_attention_optimized` - Optimized batched attention (WAPR-PERF-008)
//! - `incremental_attention_gpu` - Autoregressive decoder attention (WAPR-PERF-013)
//! - `kv_cache_scatter_gpu` - KV cache scatter operation

mod batched;
mod helpers;
mod incremental;

#[cfg(feature = "cuda")]
pub use batched::{batched_multihead_attention, batched_multihead_attention_optimized};
#[cfg(feature = "cuda")]
pub use incremental::{
    incremental_attention_gpu, incremental_attention_gpu_async,
    incremental_attention_gpu_with_stream, kv_cache_scatter_gpu,
};
