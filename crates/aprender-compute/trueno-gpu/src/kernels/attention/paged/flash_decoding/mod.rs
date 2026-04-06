//! Flash Decoding - Split-K Attention for 2X Ollama Performance (PAR-118)
//!
//! Flash Decoding splits the KV cache into chunks processed in parallel,
//! then reduces partial results. This amortizes memory bandwidth across
//! multiple thread blocks, achieving higher throughput for long sequences.
//!
//! Algorithm:
//! 1. Split sequence into K chunks of CHUNK_SIZE positions
//! 2. Each chunk computes partial attention: (max_score, sum_exp, weighted_out)
//! 3. Reduction combines partials with proper softmax rescaling:
//!    - new_max = max(chunk_max[0], chunk_max[1], ...)
//!    - For each chunk: scale = exp(chunk_max - new_max)
//!    - new_sum = sum(chunk_sum[i] * scale[i])
//!    - output = sum(chunk_out[i] * chunk_sum[i] * scale[i]) / new_sum
//!
//! Performance:
//! - Current: Sequential loop over seq_len (memory-bandwidth limited)
//! - Flash Decoding: K parallel blocks (K = ceil(seq_len / CHUNK_SIZE))
//! - Expected speedup: ~1.5-2x for typical seq_len (512-2048)

mod chunk_kernel;
mod chunk_kernel_2warp;
mod reduce_kernel;

pub use chunk_kernel::FlashDecodingChunkKernel;
pub use chunk_kernel_2warp::FlashDecodingChunkKernel2Warp;
pub use reduce_kernel::FlashDecodingReduceKernel;

/// Chunk size for Flash Decoding split-K attention.
///
/// PMAT-040: Reduced from 128 to 32 to enable actual parallelism at typical
/// decode sequence lengths (32-256 tokens).
///
/// trueno#246: Further reduced from 32 to 16 after NCU profiling showed 2.15%
/// occupancy on RTX 4090 at M=1 decode. Doubling block count improves
/// inter-SM parallelism:
/// - short ctx (~160): 329 → 354 tok/s (+7.4%)
/// - long ctx (~420): 232 → 339 tok/s (+45.8%)
///
/// Trade-offs at chunk_size=16:
/// - seq_len=32: 2 chunks (vs 1 with 32)
/// - seq_len=128: 8 chunks × 28 heads = 224 blocks → 175% SM util on 4090
/// - seq_len=500: 32 chunks × 28 heads = 896 blocks (oversubscribed, good)
/// - max_seq_len=4096: 256 chunks → partials buffer ~1.9MB (still negligible)
/// - Reduction overhead: max 256 iterations in reduce kernel
/// - chunk_size=8 tested: no better than 16 (overhead dominates)
///
/// Reference: candle-vs-apr spec v14.x §Phase 2b context scaling
pub const FLASH_DECODE_CHUNK_SIZE: u32 = 16;
