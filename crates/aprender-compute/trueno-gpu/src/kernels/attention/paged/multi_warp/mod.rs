//! Multi-Warp Incremental Attention Kernel (PAR-070)
//!
//! Uses multiple warps per head to parallelize across KV cache positions.
//! Each warp processes a chunk of positions, then cross-warp reduction combines
//! the partial softmax states.
//!
//! Performance target: 8x speedup over single-warp (from 81us to ~10us)
//!
//! # Algorithm
//!
//! 1. Launch `num_heads x num_warps_per_head` blocks
//! 2. Each warp handles positions [warp_idx * chunk, (warp_idx + 1) * chunk)
//! 3. Compute local max_score, sum_exp, weighted_output
//! 4. Cross-warp reduction in shared memory to get global max
//! 5. Correction pass to align all warps to global max
//! 6. Final sum and normalization

mod build_ptx;

/// PAR-070: Multi-warp incremental attention for decode phase
///
/// Uses multiple warps per head to parallelize across KV cache positions.
/// Each warp processes a chunk of positions, then cross-warp reduction combines
/// the partial softmax states.
///
/// Performance target: 8x speedup over single-warp (from 81us to ~10us)
#[derive(Debug, Clone)]
pub struct MultiWarpIncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension (must be <= 128)
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA, <= num_heads)
    pub num_kv_heads: u32,
    /// Number of warps per head (parallelism factor)
    pub num_warps_per_head: u32,
    /// Scaling factor for attention scores (1/sqrt(head_dim))
    pub scale: f32,
    /// PAR-061: Read seq_len from device memory (for CUDA graph compatibility)
    pub indirect_seq_len: bool,
}

impl MultiWarpIncrementalAttentionKernel {
    /// Create new multi-warp incremental attention kernel
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head (must be <= 128)
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `num_warps` - Number of warps per head (4-8 recommended)
    #[must_use]
    pub fn new(
        max_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        num_warps: u32,
    ) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            num_warps_per_head: num_warps,
            scale: 1.0 / (head_dim as f32).sqrt(),
            indirect_seq_len: false,
        }
    }

    /// Enable indirect seq_len mode (reads from device memory)
    #[must_use]
    pub fn with_indirect_seq_len(mut self, indirect: bool) -> Self {
        self.indirect_seq_len = indirect;
        self
    }
}
