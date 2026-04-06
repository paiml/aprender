//! Golden GEMM kernel generator tests.

use trueno_gpu::kernels::{Batched4DGemmKernel, BatchedGemmKernel, GemmKernel, Kernel};

#[test]
fn golden_gemm_naive_kernel_structure() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in GEMM naive\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("ld.global"),
        "GOLDEN FAIL: Missing global loads in GEMM naive\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("st.global"),
        "GOLDEN FAIL: Missing global store in GEMM naive\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("fma") || (ptx.contains("mul") && ptx.contains("add")),
        "GOLDEN FAIL: Missing multiply-accumulate in GEMM naive\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_gemm_tiled_kernel_structure() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in GEMM tiled\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared") || ptx.contains("st.shared"),
        "GOLDEN FAIL: Missing shared memory in GEMM tiled\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier sync in GEMM tiled\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_gemm_naive_kernel_structure() {
    let kernel = BatchedGemmKernel::naive(4, 64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in batched GEMM naive\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("%ctaid") || ptx.contains("batch"),
        "GOLDEN FAIL: Missing batch index in batched GEMM\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_gemm_tiled_kernel_structure() {
    let kernel = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared"),
        "GOLDEN FAIL: Missing shared memory in batched GEMM tiled\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in batched GEMM tiled\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_gemm_tiled_unrolled_kernel_structure() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry gemm_tiled_unrolled"),
        "GOLDEN FAIL: Missing gemm_tiled_unrolled entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared"),
        "GOLDEN FAIL: Missing shared memory in GEMM tiled_unrolled\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in GEMM tiled_unrolled\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("fma"),
        "GOLDEN FAIL: Missing fma in GEMM tiled_unrolled\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_gemm_tensor_core_kernel_structure() {
    let kernel = GemmKernel::tensor_core(64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry gemm_tensor_core"),
        "GOLDEN FAIL: Missing gemm_tensor_core entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".shared"),
        "GOLDEN FAIL: Missing shared memory in GEMM tensor_core\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in GEMM tensor_core\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("fma"),
        "GOLDEN FAIL: Missing fma in GEMM tensor_core\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_gemm_wmma_fp16_kernel_structure() {
    let kernel = GemmKernel::wmma_fp16(64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry gemm_wmma_fp16"),
        "GOLDEN FAIL: Missing gemm_wmma_fp16 entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("wmma"),
        "GOLDEN FAIL: Missing WMMA ops in GEMM wmma_fp16\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("load") && ptx.contains("store"),
        "GOLDEN FAIL: Missing WMMA load/store in GEMM wmma_fp16\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_gemm_tiled_unrolled_kernel_structure() {
    let kernel = BatchedGemmKernel::tiled_unrolled(4, 64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry batched_gemm_tiled_unrolled"),
        "GOLDEN FAIL: Missing batched_gemm_tiled_unrolled entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("%ctaid"),
        "GOLDEN FAIL: Missing block index in batched GEMM tiled_unrolled\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_gemm_wmma_fp16_kernel_structure() {
    let kernel = BatchedGemmKernel::wmma_fp16(4, 64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry batched_gemm_wmma_fp16"),
        "GOLDEN FAIL: Missing batched_gemm_wmma_fp16 entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("wmma"),
        "GOLDEN FAIL: Missing WMMA ops in batched GEMM wmma_fp16\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_4d_gemm_kernel_structure() {
    let kernel = Batched4DGemmKernel::new(2, 8, 32, 32, 64);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry batched_4d_gemm"),
        "GOLDEN FAIL: Missing batched_4d_gemm entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("%ctaid"),
        "GOLDEN FAIL: Missing block index in batched_4d_gemm\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared"),
        "GOLDEN FAIL: Missing shared memory in batched_4d_gemm\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_4d_gemm_with_tile_size() {
    let kernel = Batched4DGemmKernel::with_tile_size(2, 8, 32, 32, 64, 8);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry batched_4d_gemm"),
        "Custom tile size should still produce valid kernel"
    );
}

#[test]
fn golden_gemm_kernel_names_complete() {
    assert_eq!(GemmKernel::naive(64, 64, 64).name(), "gemm_naive");
    assert_eq!(GemmKernel::tiled(64, 64, 64, 16).name(), "gemm_tiled");
    assert_eq!(
        GemmKernel::tiled_unrolled(64, 64, 64, 16).name(),
        "gemm_tiled_unrolled"
    );
    assert_eq!(
        GemmKernel::tensor_core(64, 64, 64).name(),
        "gemm_tensor_core"
    );
    assert_eq!(GemmKernel::wmma_fp16(64, 64, 64).name(), "gemm_wmma_fp16");

    assert_eq!(
        BatchedGemmKernel::naive(4, 64, 64, 64).name(),
        "batched_gemm_naive"
    );
    assert_eq!(
        BatchedGemmKernel::tiled(4, 64, 64, 64, 16).name(),
        "batched_gemm_tiled"
    );
    assert_eq!(
        BatchedGemmKernel::tiled_unrolled(4, 64, 64, 64, 16).name(),
        "batched_gemm_tiled_unrolled"
    );
    assert_eq!(
        BatchedGemmKernel::wmma_fp16(4, 64, 64, 64).name(),
        "batched_gemm_wmma_fp16"
    );

    assert_eq!(
        Batched4DGemmKernel::new(2, 8, 32, 32, 64).name(),
        "batched_4d_gemm"
    );
}
