//! PTX Generator Coverage Tests
//!
//! Tests that exercise all `build_ptx()` code paths for kernel generators
//! that were identified as coverage gaps by `pmat query --coverage-gaps`.
//!
//! These are pure Rust PTX string generators — no GPU hardware required.
//!
//! Requires `cuda` feature: `cargo test -p trueno-gpu --test coverage_ptx_generators --features cuda`

#![cfg(feature = "cuda")]

use trueno_gpu::kernels::{
    AttentionKernel, Batched4DGemmKernel, BatchedGemmKernel, BatchedIncrementalAttentionKernel,
    BatchedQ4KGemvKernel, BatchedQ6KGemvKernel, ChunkedTiledQ4KGemvKernel, CoalescedGemvKernel,
    CoalescedQ4KGemvKernel, CoalescedQ6KGemvKernel, Dp4aQ4KGemvKernel, FlashDecodingChunkKernel,
    FlashDecodingReduceKernel, Fp16Q4KGemvKernel, FusedGateUpQ4KGemvKernel,
    FusedRmsNormQ4KGemvKernel, GemmKernel, GemvKernel, IncrementalAttentionKernel, Kernel,
    LongRowSoftmaxKernel, Lz4WarpCompressKernel, MultiWarpIncrementalAttentionKernel,
    PackedDp4aQ4KQ8Kernel, Q4KGemvKernel, Q4KQ8DotKernel, Q5KGemvKernel, Q5KKernel, Q6KGemvKernel,
    Q6KKernel, TiledQ4KGemvKernel, VectorizedQ4KGemvKernel,
};

/// Helper: verify basic PTX structure
fn assert_valid_ptx(ptx: &str, kernel_name: &str) {
    assert!(
        ptx.contains(".version"),
        "{}: Missing .version directive\nPTX (first 200):\n{}",
        kernel_name,
        &ptx[..ptx.len().min(200)]
    );
    assert!(ptx.contains(".entry"), "{}: Missing .entry directive", kernel_name);
    assert!(ptx.contains(".target"), "{}: Missing .target directive", kernel_name);
}

// ============================================================================
// LZ4 Compress (rank 1: 386 uncov)
// ============================================================================

#[test]
fn test_lz4_warp_compress_build_ptx() {
    let kernel = Lz4WarpCompressKernel::new(4);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "lz4_warp_compress");
    assert!(ptx.contains("ld.global"), "Missing global loads");
    assert!(ptx.contains("st.global"), "Missing global stores");
}

// ============================================================================
// Q4K Coalesced (rank 2: 342 uncov + rank 13: 228 uncov)
// ============================================================================

#[test]
fn test_coalesced_q4k_gemv_build_ptx() {
    let kernel = CoalescedQ4KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "coalesced_q4k_gemv");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

#[test]
fn test_vectorized_q4k_gemv_build_ptx() {
    let kernel = VectorizedQ4KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "vectorized_q4k_gemv");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

// ============================================================================
// Paged Attention (rank 3: 331, rank 20: 203, rank 29: 170, rank 34: 156, rank 41: 141)
// ============================================================================

#[test]
fn test_incremental_attention_build_ptx() {
    let kernel = IncrementalAttentionKernel::new(2048, 128, 32);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "incremental_attention");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

#[test]
fn test_multi_warp_incremental_attention_build_ptx() {
    let kernel = MultiWarpIncrementalAttentionKernel::new(2048, 128, 32, 8, 4);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "multi_warp_incremental_attention");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

#[test]
fn test_batched_incremental_attention_build_ptx() {
    let kernel = BatchedIncrementalAttentionKernel::new(2048, 128, 32, 8, 4);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "batched_incremental_attention");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

#[test]
fn test_flash_decoding_chunk_build_ptx() {
    let kernel = FlashDecodingChunkKernel::new(2048, 128, 32, 8, 4);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "flash_decoding_chunk");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

#[test]
fn test_flash_decoding_reduce_build_ptx() {
    let kernel = FlashDecodingReduceKernel::new(128, 32, 4);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "flash_decoding_reduce");
}

// ============================================================================
// Fused Quantized Kernels (rank 4: 293, rank 16: 218)
// ============================================================================

#[test]
fn test_fused_rmsnorm_q4k_gemv_build_ptx() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "fused_rmsnorm_q4k_gemv");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

#[test]
fn test_fused_gate_up_q4k_gemv_build_ptx() {
    let kernel = FusedGateUpQ4KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "fused_gate_up_q4k_gemv");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

// ============================================================================
// Q6K Kernels (rank 5: 278, rank 30: 165, rank 40: 146, rank 42: 137)
// ============================================================================

#[test]
fn test_q6k_gemv_build_ptx() {
    let kernel = Q6KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "q6k_gemv");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

#[test]
fn test_coalesced_q6k_gemv_build_ptx() {
    let kernel = CoalescedQ6KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "coalesced_q6k_gemv");
}

#[test]
fn test_batched_q6k_gemv_build_ptx() {
    let kernel = BatchedQ6KGemvKernel::new(4, 1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "batched_q6k_gemv");
}

#[test]
fn test_q6k_kernel_build_ptx() {
    let kernel = Q6KKernel::new(64, 64, 64);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "q6k_kernel");
}

// ============================================================================
// GEMM Basic (rank 6: 265, rank 48: 128)
// ============================================================================

#[test]
fn test_gemm_tensor_core_build_ptx() {
    let kernel = GemmKernel::tensor_core(64, 64, 64);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "gemm_tensor_core");
    assert!(ptx.contains("fma"), "Missing fma instructions");
}

#[test]
fn test_gemm_tiled_unrolled_build_ptx() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "gemm_tiled_unrolled");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

// ============================================================================
// Q4K Batched (rank 8: 242)
// ============================================================================

#[test]
fn test_batched_q4k_gemv_build_ptx() {
    let kernel = BatchedQ4KGemvKernel::new(4, 1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "batched_q4k_gemv");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

// ============================================================================
// FP16 Tensor (rank 10: 236)
// ============================================================================

#[test]
fn test_fp16_q4k_gemv_build_ptx() {
    let kernel = Fp16Q4KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "fp16_q4k_gemv");
}

// ============================================================================
// Q4K Basic (rank 11: 234)
// ============================================================================

#[test]
fn test_q4k_gemv_build_ptx() {
    let kernel = Q4KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "q4k_gemv");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

// ============================================================================
// Q4K DP4A (rank 12: 228, rank 17: 218)
// ============================================================================

#[test]
fn test_dp4a_q4k_gemv_build_ptx() {
    let kernel = Dp4aQ4KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "dp4a_q4k_gemv");
}

// ============================================================================
// Q4K Tiled (rank 14: 224, rank 22: 200)
// ============================================================================

#[test]
fn test_tiled_q4k_gemv_build_ptx() {
    let kernel = TiledQ4KGemvKernel::new(1536, 4);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "tiled_q4k_gemv");
}

#[test]
fn test_chunked_tiled_q4k_gemv_build_ptx() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(1536, 4);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "chunked_tiled_q4k_gemv");
}

// ============================================================================
// FlashAttention (rank 15: 221, rank 36: 149)
// ============================================================================

#[test]
fn test_flash_attention_build_ptx() {
    let kernel = AttentionKernel::new(128, 64);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "flash_attention");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

#[test]
fn test_flash_attention_causal_build_ptx() {
    let mut kernel = AttentionKernel::new(128, 64);
    kernel.causal = true;
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "flash_attention_causal");
}

#[test]
fn test_flash_attention_tensor_core_build_ptx() {
    let mut kernel = AttentionKernel::new(128, 64);
    kernel.use_tensor_cores = true;
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "flash_attention_tensor_core");
}

// ============================================================================
// Q4K Q8 Dot (rank 23: 196, rank 25: 186)
// ============================================================================

#[test]
fn test_q4k_q8_dot_build_ptx() {
    let kernel = Q4KQ8DotKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "q4k_q8_dot");
}

#[test]
fn test_packed_dp4a_q4k_q8_build_ptx() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "packed_dp4a_q4k_q8");
}

// ============================================================================
// Q5K (rank 26: 184, rank 31: 164)
// ============================================================================

#[test]
fn test_q5k_kernel_build_ptx() {
    let kernel = Q5KKernel::new(64, 64, 64);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "q5k_kernel");
}

#[test]
fn test_q5k_gemv_build_ptx() {
    let kernel = Q5KGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "q5k_gemv");
}

// ============================================================================
// Softmax Long Row (rank 35: 156)
// ============================================================================

#[test]
fn test_long_row_softmax_build_ptx() {
    let kernel = LongRowSoftmaxKernel::new(32768);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "long_row_softmax");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

// ============================================================================
// GEMV Coalesced (rank 49: 127)
// ============================================================================

#[test]
fn test_coalesced_gemv_build_ptx() {
    let kernel = CoalescedGemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "coalesced_gemv");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

// ============================================================================
// Batched GEMM (rank 38: 147, rank 43: 137)
// ============================================================================

#[test]
fn test_batched_gemm_tiled_unrolled_build_ptx() {
    let kernel = BatchedGemmKernel::tiled_unrolled(4, 64, 64, 64, 16);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "batched_gemm_tiled_unrolled");
}

// ============================================================================
// Batched 4D GEMM (rank 50: 127)
// ============================================================================

#[test]
fn test_batched_4d_gemm_build_ptx() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 64);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "batched_4d_gemm");
    assert!(ptx.contains("ld.global"), "Missing global loads");
}

// ============================================================================
// GEMV basic (for completeness)
// ============================================================================

#[test]
fn test_gemv_basic_build_ptx() {
    let kernel = GemvKernel::new(1536, 8960);
    let ptx = kernel.emit_ptx();
    assert_valid_ptx(&ptx, "gemv_basic");
}
