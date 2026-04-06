//! Attention Kernel Module
//!
//! Shattered into domain-specific submodules per PMAT-018:
//!
//! - **flash**: FlashAttention-style SRAM-bound tiled attention
//! - **paged**: Incremental/Paged attention for VRAM-bound KV cache management
//!
//! ## Domain Separation
//!
//! FlashAttention and PagedAttention share a name but belong to different
//! mechanical domains:
//!
//! - **FlashAttention**: SRAM-bound tiling algorithm that never materializes
//!   the full N×N attention matrix. Optimized for training and prefill.
//!
//! - **PagedAttention**: VRAM-bound block management algorithm for efficient
//!   KV cache access during autoregressive decoding. Optimized for inference.

mod flash;
mod paged;

#[cfg(test)]
mod flash_tests;
#[cfg(test)]
mod paged_tests;

// Re-export FlashAttention kernels
pub use flash::AttentionKernel;
pub use flash::PrefillAttentionKernel;

// Re-export Paged/Incremental attention kernels
pub use paged::{
    BatchedIncrementalAttentionKernel, FlashDecodingChunkKernel, FlashDecodingChunkKernel2Warp,
    FlashDecodingReduceKernel, IncrementalAttentionKernel, MultiWarpIncrementalAttentionKernel,
    FLASH_DECODE_CHUNK_SIZE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Kernel;

    // =========================================================================
    // FlashAttention Tests (SRAM-bound tiling)
    // =========================================================================

    #[test]
    fn test_attention_kernel_name() {
        let kernel = AttentionKernel::new(2048, 64);
        assert_eq!(kernel.name(), "flash_attention");

        let kernel_causal = AttentionKernel::new(2048, 64).with_causal();
        assert_eq!(kernel_causal.name(), "flash_attention_causal");

        let kernel_tc = AttentionKernel::tensor_core(2048, 64);
        assert_eq!(kernel_tc.name(), "flash_attention_tensor_core");

        let kernel_tc_causal = AttentionKernel::tensor_core(2048, 64).with_causal();
        assert_eq!(kernel_tc_causal.name(), "flash_attention_tensor_core_causal");
    }

    #[test]
    fn test_tensor_core_attention_config() {
        let kernel = AttentionKernel::tensor_core(2048, 128);
        assert_eq!(kernel.seq_len, 2048);
        assert_eq!(kernel.head_dim, 128);
        assert!(kernel.use_tensor_cores);
        assert!(kernel.tile_q >= 16);
        assert!(kernel.tile_kv >= 16);
    }

    #[test]
    fn test_attention_default_config() {
        let kernel = AttentionKernel::new(2048, 64);
        assert_eq!(kernel.seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert!(!kernel.causal);
        assert!((kernel.scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_attention_ptx_generation() {
        let kernel = AttentionKernel::new(2048, 64);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".param .u64 q_ptr"));
        assert!(ptx.contains(".param .u64 k_ptr"));
        assert!(ptx.contains(".param .u64 v_ptr"));
        assert!(ptx.contains(".param .u64 o_ptr"));
    }

    // =========================================================================
    // Incremental Attention Tests (VRAM-bound paged)
    // =========================================================================

    #[test]
    fn test_incremental_attention_kernel_new() {
        let kernel = IncrementalAttentionKernel::new(2048, 64, 22);
        assert_eq!(kernel.max_seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.num_heads, 22);
        assert_eq!(kernel.num_kv_heads, 22); // Default: num_kv_heads = num_heads
        assert!((kernel.scale - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_incremental_attention_kernel_name() {
        let kernel = IncrementalAttentionKernel::new(1024, 64, 22);
        assert_eq!(kernel.name(), "incremental_attention");
    }

    #[test]
    fn test_batched_incremental_attention_kernel_new() {
        let kernel = BatchedIncrementalAttentionKernel::new(2048, 64, 22, 22, 4);
        assert_eq!(kernel.max_seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.batch_size, 4);
    }

    #[test]
    fn test_flash_decoding_chunk_kernel_new() {
        let kernel = FlashDecodingChunkKernel::new(2048, 64, 32, 8, 4);
        assert_eq!(kernel.max_seq_len, 2048);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.num_kv_heads, 8);
        assert_eq!(kernel.batch_size, 4);
    }

    #[test]
    fn test_flash_decoding_reduce_kernel_new() {
        let kernel = FlashDecodingReduceKernel::new(64, 32, 4);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.batch_size, 4);
    }

    // =========================================================================
    // Domain Boundary Verification
    // =========================================================================

    /// Verify flash attention is distinct from paged attention
    #[test]
    fn test_domain_separation() {
        // Flash: Uses SRAM tiling (shared_memory_bytes > 0)
        let flash = AttentionKernel::new(512, 64);
        let flash_ptx = flash.build_ptx();
        assert!(flash_ptx.shared_memory_bytes() > 0, "Flash should use shared memory");

        // Paged: Single-query, no SRAM tiling needed
        let paged = IncrementalAttentionKernel::new(512, 64, 8);
        let paged_ptx = paged.build_ptx();
        // Incremental attention uses registers, not shared memory
        assert!(
            paged_ptx.shared_memory_bytes() == 0,
            "Incremental attention should not use shared memory"
        );
    }
}
