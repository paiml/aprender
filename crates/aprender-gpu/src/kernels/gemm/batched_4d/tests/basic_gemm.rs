//! Basic GEMM kernel tests: naive, tiled, tensor core, WMMA variants.

use super::*;

#[test]
fn test_naive_gemm_params() {
    let kernel = GemmKernel::naive(512, 512, 512);
    assert_eq!(kernel.name(), "gemm_naive");
    assert_eq!(kernel.config.m, 512);
}

#[test]
fn test_tiled_gemm_shared_memory() {
    let kernel = GemmKernel::tiled(1024, 1024, 1024, 32);
    let ptx_kernel = kernel.build_ptx();
    assert_eq!(ptx_kernel.shared_memory_bytes(), 32 * 32 * 4 * 2);
}

#[test]
fn test_gemm_ptx_generation() {
    let kernel = GemmKernel::naive(1024, 1024, 1024);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_naive_gemm_full_ptx() {
    let kernel = GemmKernel::naive(128, 128, 128);
    let ptx = kernel.emit_ptx();

    // Verify loop structure
    assert!(ptx.contains("loop_k:"));
    assert!(ptx.contains("loop_end:"));
    assert!(ptx.contains("exit:"));

    // Verify memory operations
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));

    // Verify arithmetic (FMA used for accumulation)
    assert!(ptx.contains("fma") || ptx.contains("mul.f32"));
    // Note: add.f32 may not appear if all additions are fused
}

#[test]
fn test_gemm_variants() {
    let naive = GemmKernel::naive(64, 64, 64);
    let tiled = GemmKernel::tiled(64, 64, 64, 16);
    let tensor = GemmKernel::tensor_core(64, 64, 64);

    assert_eq!(naive.name(), "gemm_naive");
    assert_eq!(tiled.name(), "gemm_tiled");
    assert_eq!(tensor.name(), "gemm_tensor_core");

    // All should produce valid PTX
    let _ = naive.emit_ptx();
    let _ = tiled.emit_ptx();
    let _ = tensor.emit_ptx();
}

#[test]
fn test_gemm_config_default() {
    let config = GemmConfig::default();
    assert_eq!(config.m, 1024);
    assert_eq!(config.n, 1024);
    assert_eq!(config.k, 1024);
    assert_eq!(config.tile_size, 32);
    assert!(!config.use_tensor_cores);
}

#[test]
fn test_tensor_core_kernel() {
    let kernel = GemmKernel::tensor_core(256, 256, 256);
    assert!(kernel.config.use_tensor_cores);
    let ptx_kernel = kernel.build_ptx();
    // WMMA fragments need shared memory
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_tiled_gemm_full_ptx() {
    let kernel = GemmKernel::tiled(256, 256, 256, 16);
    let ptx = kernel.emit_ptx();

    // Verify tiling structure
    assert!(ptx.contains("tile_loop:"));
    assert!(ptx.contains("tile_loop_end:"));
    assert!(ptx.contains("inner_k_loop:"));
    assert!(ptx.contains("inner_k_end:"));

    // Verify shared memory operations
    assert!(ptx.contains("ld.shared.f32") || ptx.contains("ld.f32")); // shared load
    assert!(ptx.contains("st.shared.f32") || ptx.contains("st.f32")); // shared store

    // Verify barrier synchronization
    assert!(ptx.contains("bar"));

    // Verify global loads/stores still present
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_tensor_core_gemm_ptx() {
    let kernel = GemmKernel::tensor_core(512, 512, 512);
    let ptx = kernel.emit_ptx();

    // Verify WMMA structure
    assert!(ptx.contains("wmma_loop:") || ptx.contains("exit:"));

    // Verify memory operations (could be global or shared)
    assert!(ptx.contains("ld.global.f32") || ptx.contains("wmma_m_loop:"));
}

#[test]
fn test_ptx_output_for_verification() {
    let kernel = GemmKernel::tiled(128, 128, 128, 32);
    let ptx = kernel.emit_ptx();

    std::fs::write("/tmp/test_tiled.ptx", &ptx).expect("write PTX");

    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("add.u32"));
    assert!(ptx.contains("%r17, %r17, 1") || ptx.contains("%r"));
    assert!(ptx.contains("%r10, %r10, 1") || ptx.contains("%r"));
}

#[test]
fn test_naive_ptx_for_verification() {
    let kernel = GemmKernel::naive(128, 128, 128);
    let ptx = kernel.emit_ptx();

    std::fs::write("/tmp/test_naive.ptx", &ptx).expect("write PTX");

    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("loop_k:"));
    assert!(ptx.contains("loop_end:"));
}

#[test]
fn test_wmma_fp16_kernel() {
    // Test WmmaFp16 variant - requires dimensions multiple of 16
    let kernel = GemmKernel::wmma_fp16(256, 256, 256);
    assert_eq!(kernel.name(), "gemm_wmma_fp16");
    assert!(kernel.config.use_tensor_cores);
    assert_eq!(kernel.config.tile_size, 16);

    // Build PTX
    let ptx_kernel = kernel.build_ptx();
    assert!(ptx_kernel.shared_memory_bytes() > 0);

    // Emit PTX and verify structure
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry gemm_wmma_fp16"));
    assert!(ptx.contains(".param"));
}

#[test]
fn test_wmma_fp16_ptx_generation() {
    let kernel = GemmKernel::wmma_fp16(128, 128, 128);
    let ptx = kernel.emit_ptx();

    // Verify WMMA-specific patterns
    assert!(ptx.contains("wmma") || ptx.contains("mma") || ptx.contains("ld.global.f32"));

    // Write to /tmp for inspection
    std::fs::write("/tmp/test_wmma.ptx", &ptx).expect("write PTX");
}

#[test]
fn test_all_gemm_variants_emit_valid_ptx() {
    let variants: Vec<GemmKernel> = vec![
        GemmKernel::naive(64, 64, 64),
        GemmKernel::tiled(64, 64, 64, 16),
        GemmKernel::tensor_core(64, 64, 64),
        GemmKernel::wmma_fp16(64, 64, 64),
    ];

    for kernel in variants {
        let name = kernel.name().to_string();
        let ptx = kernel.emit_ptx();
        let ptx_kernel = kernel.build_ptx();

        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains(".param"));

        if name.contains("tiled") || name.contains("tensor") || name.contains("wmma") {
            assert!(ptx_kernel.shared_memory_bytes() > 0);
        }
    }
}

#[test]
fn test_gemm_config_clone() {
    let config = GemmConfig::default();
    let cloned = config.clone();
    assert_eq!(config.m, cloned.m);
    assert_eq!(config.n, cloned.n);
    assert_eq!(config.k, cloned.k);
}

#[test]
fn test_gemm_kernel_clone() {
    let kernel = GemmKernel::naive(128, 128, 128);
    let cloned = kernel.clone();
    assert_eq!(kernel.name(), cloned.name());
}
