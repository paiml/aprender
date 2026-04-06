//! FlashAttention-Style Tiled Attention Kernel
//!
//! Implements IO-aware attention per Dao et al. [16]
//! Never materializes the full N×N attention matrix.
//!
//! Standard Attention: O(N²) memory for S = Q × K^T
//! FlashAttention: O(N × d) memory using online softmax
//!
//! ## Variants
//!
//! - **Standard**: FP32 serial dot product (baseline, ~79ms/token)
//! - **Tensor Core**: FP16 WMMA for Q×K^T (target: <2ms/token, ~40x speedup)
//!
//! ## Performance (RTX 4090, seq_len=2048, head_dim=128)
//!
//! | Variant     | Time/token | Throughput |
//! |-------------|------------|------------|
//! | Standard    | 79ms       | 12.6 tok/s |
//! | Tensor Core | ~2ms       | ~500 tok/s |

//! FlashAttention kernel for SRAM-bound tiled attention.
//!
//! This module implements the original FlashAttention algorithm (Dao et al.)
//! which tiles computation to fit in GPU SRAM, avoiding materialization
//! of the full N×N attention matrix.

#![allow(clippy::similar_names)]

mod prefill_fused;
mod standard;
mod tensor_core;

pub use prefill_fused::PrefillAttentionKernel;

use crate::kernels::Kernel;
use crate::ptx::PtxKernel;

/// FlashAttention-style kernel configuration
#[derive(Debug, Clone)]
pub struct AttentionKernel {
    /// Sequence length (N)
    pub seq_len: u32,
    /// Head dimension (d)
    pub head_dim: u32,
    /// Tile size for Q (B_r)
    pub tile_q: u32,
    /// Tile size for KV (B_c)
    pub tile_kv: u32,
    /// Scaling factor for attention scores (1/sqrt(d))
    pub scale: f32,
    /// Use causal masking (for autoregressive models)
    pub causal: bool,
    /// Use Tensor Cores for Q×K^T (FP16 WMMA, requires sm_70+)
    pub use_tensor_cores: bool,
}

impl AttentionKernel {
    /// Create a new attention kernel
    ///
    /// Tile sizes are auto-clamped to not exceed seq_len and head_dim
    /// to handle small inputs gracefully.
    #[must_use]
    pub fn new(seq_len: u32, head_dim: u32) -> Self {
        // Auto-clamp tile sizes to input dimensions
        // Default tiles: 64, but reduce if inputs are smaller
        let tile_q = seq_len.min(64);
        // GH-5 FIX: Ensure tile_kv >= head_dim to prevent shared memory overflow
        // The K dot product loop accesses K[local_col * head_dim + d_idx], which requires
        // head_dim rows in K tile. Without this, we overflow when tile_kv < head_dim.
        let tile_kv = seq_len.min(64).max(head_dim);

        Self {
            seq_len,
            head_dim,
            tile_q,
            tile_kv,
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: false,
            use_tensor_cores: false,
        }
    }

    /// Create Tensor Core attention kernel (highest performance)
    ///
    /// Uses FP16 WMMA for Q×K^T computation, achieving ~40x speedup over FP32.
    /// Requires sm_70+ (Volta or later). Dimensions should be multiples of 16.
    ///
    /// # Performance
    ///
    /// - Standard FP32: ~79ms/token (12.6 tok/s)
    /// - Tensor Core FP16: ~2ms/token (500+ tok/s)
    #[must_use]
    pub fn tensor_core(seq_len: u32, head_dim: u32) -> Self {
        // For Tensor Cores, use tile sizes that are multiples of 16
        let tile_q = seq_len.clamp(16, 64);
        // GH-5 FIX: Ensure tile_kv >= head_dim to prevent shared memory overflow
        // Same issue as new() - K dot product requires head_dim rows in K tile.
        let tile_kv = seq_len.clamp(16, 64).max(head_dim);

        Self {
            seq_len,
            head_dim,
            tile_q,
            tile_kv,
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: false,
            use_tensor_cores: true,
        }
    }

    /// Set tile sizes for Q and KV
    ///
    /// `tile_kv` is clamped to `max(tile_kv, head_dim)` to prevent shared memory
    /// OOB in the K dot product loop (GH-32).
    #[must_use]
    pub const fn with_tiles(mut self, tile_q: u32, tile_kv: u32) -> Self {
        self.tile_q = tile_q;
        // GH-32 FIX: Enforce tile_kv >= head_dim to prevent shared memory overflow
        self.tile_kv = if tile_kv >= self.head_dim { tile_kv } else { self.head_dim };
        self
    }

    /// Enable causal masking for autoregressive attention
    #[must_use]
    pub const fn with_causal(mut self) -> Self {
        self.causal = true;
        self
    }

    /// Set custom scale factor
    #[must_use]
    pub const fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Enable Tensor Core acceleration for Q×K^T computation
    ///
    /// Uses FP16 WMMA instructions for ~40x speedup on sm_70+ GPUs.
    #[must_use]
    pub const fn with_tensor_cores(mut self) -> Self {
        self.use_tensor_cores = true;
        self
    }
}

impl Kernel for AttentionKernel {
    fn name(&self) -> &str {
        match (self.use_tensor_cores, self.causal) {
            (true, true) => "flash_attention_tensor_core_causal",
            (true, false) => "flash_attention_tensor_core",
            (false, true) => "flash_attention_causal",
            (false, false) => "flash_attention",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        if self.use_tensor_cores {
            self.build_tensor_core_attention()
        } else {
            self.build_flash_attention()
        }
    }
}
