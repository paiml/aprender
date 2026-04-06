//! Paged Attention Kernel Tests (PMAT-018: Coverage Killer Remediation)
//!
//! Tests for all paged/incremental attention kernels to achieve coverage.

use super::paged::{
    BatchedIncrementalAttentionKernel, FlashDecodingChunkKernel, FlashDecodingReduceKernel,
    IncrementalAttentionKernel, MultiWarpIncrementalAttentionKernel, FLASH_DECODE_CHUNK_SIZE,
};
use crate::kernels::Kernel;

// ============================================================================
// IncrementalAttentionKernel Tests
// ============================================================================

#[test]
fn test_incremental_attention_kernel_mha() {
    let kernel = IncrementalAttentionKernel::new(2048, 64, 8);

    assert_eq!(kernel.name(), "incremental_attention");
    assert!(!kernel.is_gqa());
    assert_eq!(kernel.head_dim, 64);
    assert_eq!(kernel.num_heads, 8);
    assert_eq!(kernel.num_kv_heads, 8);

    // Generate PTX
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".entry incremental_attention"));
    assert!(ptx.contains("q_ptr"));
    assert!(ptx.contains("k_ptr"));
    assert!(ptx.contains("v_ptr"));
}

#[test]
fn test_incremental_attention_kernel_gqa() {
    // GQA with 8 query heads and 2 KV heads (4:1 ratio)
    let kernel = IncrementalAttentionKernel::with_gqa(2048, 128, 8, 2);

    assert!(kernel.is_gqa());
    assert_eq!(kernel.num_heads, 8);
    assert_eq!(kernel.num_kv_heads, 2);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry incremental_attention"));
}

#[test]
fn test_incremental_attention_kernel_indirect_seq_len() {
    // PAR-061: Indirect seq_len mode for CUDA graph compatibility
    let kernel = IncrementalAttentionKernel::new(4096, 64, 16).with_indirect_seq_len(true);

    assert_eq!(kernel.name(), "incremental_attention_indirect");
    assert!(kernel.indirect_seq_len);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry incremental_attention_indirect"));
    assert!(ptx.contains("seq_len_ptr")); // Should have ptr instead of u32
}

#[test]
fn test_incremental_attention_kernel_small_head_dim() {
    let kernel = IncrementalAttentionKernel::new(512, 32, 4);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_incremental_attention_kernel_large_config() {
    // Llama-70B style: 80 heads, GQA 8:1
    let kernel = IncrementalAttentionKernel::with_gqa(8192, 128, 80, 10);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry incremental_attention"));
}

// ============================================================================
// MultiWarpIncrementalAttentionKernel Tests
// ============================================================================

#[test]
fn test_multi_warp_incremental_attention() {
    // new(max_seq_len, head_dim, num_heads, num_kv_heads, num_warps)
    let kernel = MultiWarpIncrementalAttentionKernel::new(2048, 64, 8, 8, 4);

    // Note: actual name is "multi_warp_attention" not "multi_warp_incremental_attention"
    assert!(kernel.name().contains("multi_warp"));
    assert_eq!(kernel.head_dim, 64);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"));
    // Multi-warp kernel should have shared memory for cross-warp reduction
    assert!(ptx.contains(".shared"));
}

#[test]
fn test_multi_warp_incremental_attention_gqa() {
    // GQA: 32 query heads, 8 KV heads
    let kernel = MultiWarpIncrementalAttentionKernel::new(4096, 128, 32, 8, 4);

    // GQA check
    assert_ne!(kernel.num_heads, kernel.num_kv_heads);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_multi_warp_incremental_attention_indirect() {
    let kernel =
        MultiWarpIncrementalAttentionKernel::new(2048, 64, 16, 16, 4).with_indirect_seq_len(true);

    assert!(kernel.name().contains("indirect"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("seq_len_ptr"));
}

#[test]
fn test_multi_warp_8_warps() {
    // Test with 8 warps for better parallelism
    let kernel = MultiWarpIncrementalAttentionKernel::new(2048, 64, 8, 8, 8);

    assert_eq!(kernel.num_warps_per_head, 8);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// BatchedIncrementalAttentionKernel Tests
// ============================================================================

#[test]
fn test_batched_incremental_attention() {
    // new(max_seq_len, head_dim, num_heads, num_kv_heads, batch_size)
    let kernel = BatchedIncrementalAttentionKernel::new(2048, 64, 8, 8, 4);

    assert!(kernel.name().contains("batched"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_batched_incremental_attention_gqa() {
    // GQA batched
    let kernel = BatchedIncrementalAttentionKernel::new(4096, 128, 32, 8, 8);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_batched_incremental_attention_large_batch() {
    let kernel = BatchedIncrementalAttentionKernel::new(2048, 64, 16, 4, 32);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// FlashDecodingChunkKernel Tests
// ============================================================================

#[test]
fn test_flash_decoding_chunk_kernel() {
    // new(max_seq_len, head_dim, num_heads, num_kv_heads, batch_size)
    let kernel = FlashDecodingChunkKernel::new(4096, 64, 8, 8, 4);

    assert!(kernel.name().contains("flash_decoding"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_flash_decoding_chunk_kernel_gqa() {
    let kernel = FlashDecodingChunkKernel::new(8192, 128, 32, 8, 8);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_flash_decoding_chunk_kernel_large_seq() {
    let kernel = FlashDecodingChunkKernel::new(16384, 64, 16, 4, 2);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_flash_decode_chunk_size_constant() {
    // Verify the chunk size constant is a reasonable value
    let chunk_size = FLASH_DECODE_CHUNK_SIZE;
    assert!(chunk_size > 0);
    assert!(chunk_size <= 2048);
}

// ============================================================================
// FlashDecodingReduceKernel Tests
// ============================================================================

#[test]
fn test_flash_decoding_reduce_kernel() {
    // new(head_dim, num_heads, batch_size)
    let kernel = FlashDecodingReduceKernel::new(64, 8, 4);

    assert!(kernel.name().contains("flash_decoding"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_flash_decoding_reduce_kernel_large() {
    let kernel = FlashDecodingReduceKernel::new(128, 32, 16);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_flash_decoding_reduce_kernel_small() {
    let kernel = FlashDecodingReduceKernel::new(32, 4, 1);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// Integration Tests: Kernel Chain
// ============================================================================

#[test]
fn test_flash_decoding_kernel_pair() {
    // Flash decoding uses a chunk kernel + reduce kernel
    let max_seq = 8192;
    let head_dim = 64;
    let num_heads = 32;
    let batch_size = 4;

    let chunk_kernel =
        FlashDecodingChunkKernel::new(max_seq, head_dim, num_heads, num_heads, batch_size);
    let reduce_kernel = FlashDecodingReduceKernel::new(head_dim, num_heads, batch_size);

    let chunk_ptx = chunk_kernel.emit_ptx();
    let reduce_ptx = reduce_kernel.emit_ptx();

    // Both kernels should generate valid PTX
    assert!(chunk_ptx.contains(".version"));
    assert!(reduce_ptx.contains(".version"));
}

#[test]
fn test_all_attention_kernel_variants() {
    // Comprehensive test that exercises all kernel types
    let configs = vec![
        (2048, 64, 8, 8, 4),    // MHA small
        (4096, 128, 32, 8, 8),  // GQA medium
        (8192, 128, 64, 8, 16), // GQA large
    ];

    for (max_seq, head_dim, num_heads, num_kv_heads, batch) in configs {
        // IncrementalAttentionKernel
        let k1 = IncrementalAttentionKernel::with_gqa(max_seq, head_dim, num_heads, num_kv_heads);
        assert!(k1.emit_ptx().contains(".version"));

        // MultiWarpIncrementalAttentionKernel (num_warps=4)
        let k2 =
            MultiWarpIncrementalAttentionKernel::new(max_seq, head_dim, num_heads, num_kv_heads, 4);
        assert!(k2.emit_ptx().contains(".version"));

        // BatchedIncrementalAttentionKernel
        let k3 = BatchedIncrementalAttentionKernel::new(
            max_seq,
            head_dim,
            num_heads,
            num_kv_heads,
            batch,
        );
        assert!(k3.emit_ptx().contains(".version"));

        // FlashDecodingChunkKernel
        let k4 = FlashDecodingChunkKernel::new(max_seq, head_dim, num_heads, num_kv_heads, batch);
        assert!(k4.emit_ptx().contains(".version"));

        // FlashDecodingReduceKernel
        let k5 = FlashDecodingReduceKernel::new(head_dim, num_heads, batch);
        assert!(k5.emit_ptx().contains(".version"));
    }
}

// ============================================================================
// Scale Factor Tests
// ============================================================================

#[test]
fn test_attention_scale_factor() {
    // Scale should be 1/sqrt(head_dim)
    let kernel = IncrementalAttentionKernel::new(1024, 64, 8);
    let expected_scale = 1.0 / (64.0_f32).sqrt();
    assert!((kernel.scale - expected_scale).abs() < 1e-6);

    let kernel2 = IncrementalAttentionKernel::new(1024, 128, 8);
    let expected_scale2 = 1.0 / (128.0_f32).sqrt();
    assert!((kernel2.scale - expected_scale2).abs() < 1e-6);
}
