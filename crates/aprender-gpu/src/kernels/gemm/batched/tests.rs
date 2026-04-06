use super::*;

#[test]
fn test_batched_gemm_config_default() {
    let config = BatchedGemmConfig::default();
    assert_eq!(config.batch, 1);
    assert_eq!(config.m, 1024);
    assert_eq!(config.n, 1024);
    assert_eq!(config.k, 1024);
    assert_eq!(config.tile_size, 16);
}

#[test]
fn test_batched_gemm_naive_ptx_gen() {
    let kernel = BatchedGemmKernel::naive(4, 32, 32, 32);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry batched_gemm_naive"));
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 batch"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_batched_gemm_tiled_ptx_gen() {
    let kernel = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry batched_gemm_tiled"));
    assert!(ptx.contains(".shared")); // Should use shared memory
    assert!(ptx.contains("bar.sync")); // Should have barrier
}

#[test]
fn test_batched_gemm_tiled_unrolled_ptx_gen() {
    let kernel = BatchedGemmKernel::tiled_unrolled(4, 64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry batched_gemm_tiled_unrolled"));
    assert!(ptx.contains(".shared"));
    // Unrolled should have more FMA instructions
    assert!(ptx.contains("fma"));
}

#[test]
fn test_batched_gemm_wmma_fp16_ptx_gen() {
    let kernel = BatchedGemmKernel::wmma_fp16(4, 64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry batched_gemm_wmma_fp16"));
    assert!(ptx.contains("wmma.load")); // WMMA loads
    assert!(ptx.contains("wmma.mma")); // WMMA MMA
    assert!(ptx.contains("wmma.store")); // WMMA store
}

#[test]
fn test_batched_gemm_kernel_names() {
    assert_eq!(BatchedGemmKernel::naive(1, 32, 32, 32).name(), "batched_gemm_naive");
    assert_eq!(BatchedGemmKernel::tiled(1, 32, 32, 32, 16).name(), "batched_gemm_tiled");
    assert_eq!(
        BatchedGemmKernel::tiled_unrolled(1, 32, 32, 32, 16).name(),
        "batched_gemm_tiled_unrolled"
    );
    assert_eq!(BatchedGemmKernel::wmma_fp16(1, 32, 32, 32).name(), "batched_gemm_wmma_fp16");
}

#[test]
fn test_batched_gemm_config_clone() {
    let config = BatchedGemmConfig { batch: 8, m: 256, n: 128, k: 64, tile_size: 32 };
    let cloned = config.clone();
    assert_eq!(cloned.batch, 8);
    assert_eq!(cloned.m, 256);
    assert_eq!(cloned.n, 128);
    assert_eq!(cloned.k, 64);
    assert_eq!(cloned.tile_size, 32);
}

#[test]
fn test_batched_gemm_kernel_clone() {
    let kernel = BatchedGemmKernel::naive(2, 16, 16, 16);
    let cloned = kernel.clone();
    assert_eq!(cloned.name(), "batched_gemm_naive");
    assert_eq!(cloned.config.batch, 2);
}

#[test]
fn test_batched_gemm_debug_format() {
    let config = BatchedGemmConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("BatchedGemmConfig"));

    let kernel = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
    let kernel_debug = format!("{:?}", kernel);
    assert!(debug.contains("BatchedGemmConfig") || kernel_debug.contains("BatchedGemmKernel"));
}

#[test]
fn test_batched_gemm_small_dimensions() {
    // Test with minimal dimensions
    let kernel = BatchedGemmKernel::naive(1, 1, 1, 1);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry batched_gemm_naive"));
}

#[test]
fn test_batched_gemm_large_batch() {
    // Test with large batch size
    let kernel = BatchedGemmKernel::naive(128, 16, 16, 16);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry batched_gemm_naive"));
}

#[test]
fn test_batched_gemm_non_square_dims() {
    // Test non-square dimensions
    let kernel = BatchedGemmKernel::tiled(4, 128, 64, 32, 16);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry batched_gemm_tiled"));
    assert_eq!(kernel.config.m, 128);
    assert_eq!(kernel.config.n, 64);
    assert_eq!(kernel.config.k, 32);
}
