use super::super::*;

// ==========================================================================
// PAR-032: FP16 Q4K GEMV KERNEL TESTS
// ==========================================================================

#[test]
fn test_fp16_q4k_gemv_kernel_name() {
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "fp16_q4k_gemv");
}

#[test]
fn test_fp16_q4k_gemv_generates_ptx() {
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry fp16_q4k_gemv"));
}

#[test]
fn test_fp16_q4k_gemv_has_fp16_loads() {
    // Verify FP16 input loads
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should use ld.global.b16 for FP16 input
    assert!(ptx.contains("ld.global"));
    // Should have cvt.f32.f16 for conversion
    assert!(ptx.contains("cvt.f32.f16"));
}

#[test]
fn test_fp16_q4k_gemv_has_fp16_stores() {
    // Verify FP16 output stores
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should use st.global.b16 for FP16 output
    assert!(ptx.contains("st.global"));
    // Should have cvt.f16.f32 for conversion
    assert!(ptx.contains("cvt.rn.f16.f32"));
}

#[test]
fn test_fp16_q4k_gemv_has_warp_shuffle() {
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should use warp shuffle for reduction
    assert!(ptx.contains("shfl.sync.down"));
}

#[test]
fn test_fp16_q4k_gemv_qwen3b_dimensions() {
    // Qwen 3B typical dimensions
    let kernel = Fp16Q4KGemvKernel::new(3584, 3584);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry"));
}

#[test]
fn test_fp16_q4k_gemv_ffn_dimensions() {
    // Qwen 3B FFN dimensions (hidden_size -> intermediate_size)
    let kernel = Fp16Q4KGemvKernel::new(3584, 18944);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
}

#[test]
fn test_fp16_q4k_gemv_structure() {
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

// ==========================================================================
// PAR-034: TENSOR CORE Q4K GEMM KERNEL TESTS
// ==========================================================================

#[test]
fn test_tensor_core_q4k_gemm_kernel_name() {
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    assert_eq!(kernel.name(), "tensor_core_q4k_gemm");
}

#[test]
fn test_tensor_core_q4k_gemm_generates_ptx() {
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry tensor_core_q4k_gemm"));
}

#[test]
fn test_tensor_core_q4k_gemm_has_wmma_ops() {
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    let ptx = kernel.emit_ptx();
    // PMAT-064: Should have WMMA tensor core operations
    assert!(ptx.contains("wmma.load.a"), "missing wmma.load.a");
    assert!(ptx.contains("wmma.load.b"), "missing wmma.load.b");
    assert!(ptx.contains("wmma.mma"), "missing wmma.mma");
    assert!(ptx.contains("wmma.store.d"), "missing wmma.store.d");
    // Should have Q4K dequant (FP16 conversions)
    assert!(ptx.contains("cvt.f32.f16"), "missing Q4K dequant f32←f16");
    assert!(ptx.contains("cvt.rn.f16.f32"), "missing f16←f32 for SHMEM");
}

#[test]
fn test_tensor_core_q4k_gemm_batched_dimensions() {
    // Speculative decode with K=8 draft tokens
    let kernel = TensorCoreQ4KGemmKernel::new(8, 3584, 4096);
    assert_eq!(kernel.m, 8);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert_eq!(kernel.num_super_blocks(), 14); // 3584 / 256 = 14
}

#[test]
fn test_tensor_core_q4k_gemm_qwen3b_ffn() {
    // Qwen 3B FFN: [batch, 3584] x [3584, 18944]
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 18944);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry"));
}

#[test]
fn test_tensor_core_q4k_gemm_has_barrier() {
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should have barrier for shared memory synchronization
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_tensor_core_q4k_gemm_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(result.is_safe, "Tensor Core Q4K GEMM should be barrier-safe: {:?}", result.violations);
}

// =========================================================================
// Q8_0 GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_q8_0_gemv_kernel_name() {
    let kernel = Q8_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q8_0_gemv_warp_reduce");
}

#[test]
fn test_q8_0_gemv_config() {
    let kernel = Q8_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q8_0_gemv_num_blocks() {
    let kernel = Q8_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.num_blocks_per_row(), 112); // ceil(3584/32)
}

#[test]
fn test_q8_0_gemv_ptx_generation() {
    let kernel = Q8_0GemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q8_0_gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64"));
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global"));
}

// =========================================================================
// Q4_0 GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_q4_0_gemv_kernel_name() {
    let kernel = Q4_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q4_0_gemv_warp_reduce");
}

#[test]
fn test_q4_0_gemv_config() {
    let kernel = Q4_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q4_0_gemv_ptx_generation() {
    let kernel = Q4_0GemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q4_0_gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64"));
    assert!(ptx.contains("ld.global"));
}

// =========================================================================
// Q4_1 GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_q4_1_gemv_kernel_name() {
    let kernel = Q4_1GemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q4_1_gemv_warp_reduce");
}

#[test]
fn test_q4_1_gemv_config() {
    let kernel = Q4_1GemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q4_1_gemv_ptx_generation() {
    let kernel = Q4_1GemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q4_1_gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64"));
    assert!(ptx.contains("ld.global"));
}

// =========================================================================
// Q5_0 GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_q5_0_gemv_kernel_name() {
    let kernel = Q5_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q5_0_gemv_warp_reduce");
}

#[test]
fn test_q5_0_gemv_config() {
    let kernel = Q5_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q5_0_gemv_ptx_generation() {
    let kernel = Q5_0GemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q5_0_gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64"));
    assert!(ptx.contains("ld.global"));
}

// =========================================================================
// CHUNKED TILED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_chunked_tiled_q4k_gemv_kernel_name() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "chunked_tiled_q4k_gemv");
}

#[test]
fn test_chunked_tiled_q4k_gemv_config() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_chunked_tiled_q4k_gemv_ptx_generation() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry chunked_tiled_q4k_gemv"));
    assert!(ptx.contains("bar.sync")); // Shared memory sync
}

#[test]
fn test_chunked_tiled_q4k_gemv_shared_memory() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    let ptx_kernel = kernel.build_ptx();
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_chunked_tiled_q4k_gemv_with_outputs_per_block() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096).with_outputs_per_block(8);
    assert_eq!(kernel.outputs_per_block, 8);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_chunked_tiled_q4k_gemv_with_outputs_per_block_default() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.outputs_per_block, 4); // Default value
}

#[test]
fn test_chunked_tiled_q4k_gemv_with_outputs_per_block_chained() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096)
        .with_outputs_per_block(2)
        .with_outputs_per_block(16);
    assert_eq!(kernel.outputs_per_block, 16); // Last value wins
}

#[test]
fn test_chunked_tiled_q4k_gemv_needs_chunking_small_k() {
    // K = 3584 < 8192 (CHUNK_SIZE), so no chunking needed
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    assert!(!kernel.needs_chunking());
}

#[test]
fn test_chunked_tiled_q4k_gemv_needs_chunking_large_k() {
    // K = 16384 > 8192 (CHUNK_SIZE), so chunking is needed
    let kernel = ChunkedTiledQ4KGemvKernel::new(16384, 4096);
    assert!(kernel.needs_chunking());
}

#[test]
fn test_chunked_tiled_q4k_gemv_needs_chunking_boundary() {
    // K = 8192 = CHUNK_SIZE exactly, no chunking needed
    let kernel_exact = ChunkedTiledQ4KGemvKernel::new(8192, 4096);
    assert!(!kernel_exact.needs_chunking());

    // K = 8193 > CHUNK_SIZE, chunking needed
    let kernel_over = ChunkedTiledQ4KGemvKernel::new(8193, 4096);
    assert!(kernel_over.needs_chunking());
}

#[test]
fn test_chunked_tiled_q4k_gemv_needs_chunking_very_large_k() {
    // K = 32768, definitely needs chunking
    let kernel = ChunkedTiledQ4KGemvKernel::new(32768, 4096);
    assert!(kernel.needs_chunking());
}
