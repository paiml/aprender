//! Transform Kernel Tests (PMAT-018: Coverage Killer Remediation)
//!
//! Tests for all transform/layout conversion kernels to achieve coverage.

use super::transform::{
    BatchedScaleKernel, BatchedSoftmaxKernel, BatchedToInterleavedKernel, BatchedTransposeKernel,
    CopySingleHeadKernel, ExtractSingleHeadKernel, InterleavedToBatchedKernel, TransposeKernel,
};
use crate::kernels::Kernel;

// ============================================================================
// TransposeKernel Tests
// ============================================================================

#[test]
fn test_transpose_kernel() {
    let kernel = TransposeKernel::new(64, 128);

    assert!(kernel.name().contains("transpose"));
    assert_eq!(kernel.rows, 64);
    assert_eq!(kernel.cols, 128);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_transpose_kernel_square() {
    let kernel = TransposeKernel::new(256, 256);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_transpose_kernel_small() {
    let kernel = TransposeKernel::new(16, 32);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// InterleavedToBatchedKernel Tests
// Signature: new(seq_len: u32, n_heads: u32, head_dim: u32)
// ============================================================================

#[test]
fn test_interleaved_to_batched_kernel() {
    let kernel = InterleavedToBatchedKernel::new(128, 8, 64);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_interleaved_to_batched_kernel_large() {
    let kernel = InterleavedToBatchedKernel::new(2048, 32, 128);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// ExtractSingleHeadKernel Tests
// Signature: new(seq_len: u32, n_heads: u32, head_dim: u32)
// ============================================================================

#[test]
fn test_extract_single_head_kernel() {
    let kernel = ExtractSingleHeadKernel::new(128, 8, 64);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_extract_single_head_kernel_large() {
    let kernel = ExtractSingleHeadKernel::new(4096, 32, 128);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// CopySingleHeadKernel Tests
// Signature: new(seq_len: u32, n_heads: u32, head_dim: u32)
// ============================================================================

#[test]
fn test_copy_single_head_kernel() {
    let kernel = CopySingleHeadKernel::new(128, 8, 64);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// BatchedToInterleavedKernel Tests
// Signature: new(seq_len: u32, n_heads: u32, head_dim: u32)
// ============================================================================

#[test]
fn test_batched_to_interleaved_kernel() {
    let kernel = BatchedToInterleavedKernel::new(128, 8, 64);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_batched_to_interleaved_kernel_large() {
    let kernel = BatchedToInterleavedKernel::new(2048, 32, 128);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// BatchedTransposeKernel Tests
// Signature: new(batch: u32, rows: u32, cols: u32)
// ============================================================================

#[test]
fn test_batched_transpose_kernel() {
    let kernel = BatchedTransposeKernel::new(4, 64, 128);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_batched_transpose_kernel_large() {
    let kernel = BatchedTransposeKernel::new(16, 256, 512);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// BatchedScaleKernel Tests
// Signature: new(n: u32)
// ============================================================================

#[test]
fn test_batched_scale_kernel() {
    let kernel = BatchedScaleKernel::new(1024);

    assert!(kernel.name().contains("scale"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_batched_scale_kernel_large() {
    let kernel = BatchedScaleKernel::new(8192);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// BatchedSoftmaxKernel Tests
// Signature: new(total_rows: u32, row_size: u32)
// ============================================================================

#[test]
fn test_batched_softmax_kernel() {
    let kernel = BatchedSoftmaxKernel::new(4, 1024);

    assert!(kernel.name().contains("softmax"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_batched_softmax_kernel_large() {
    let kernel = BatchedSoftmaxKernel::new(32, 4096);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// Comprehensive Configuration Matrix
// ============================================================================

#[test]
fn test_all_transform_kernel_variants() {
    let configs = vec![
        (128, 8, 64), // seq_len, n_heads, head_dim
        (256, 16, 128),
        (512, 32, 64),
    ];

    for (seq_len, num_heads, head_dim) in configs {
        // TransposeKernel (rows, cols)
        let k1 = TransposeKernel::new(seq_len, num_heads * head_dim);
        assert!(k1.emit_ptx().contains(".version"));

        // InterleavedToBatchedKernel (seq_len, n_heads, head_dim)
        let k2 = InterleavedToBatchedKernel::new(seq_len, num_heads, head_dim);
        assert!(k2.emit_ptx().contains(".version"));

        // ExtractSingleHeadKernel (seq_len, n_heads, head_dim)
        let k3 = ExtractSingleHeadKernel::new(seq_len, num_heads, head_dim);
        assert!(k3.emit_ptx().contains(".version"));

        // CopySingleHeadKernel (seq_len, n_heads, head_dim)
        let k4 = CopySingleHeadKernel::new(seq_len, num_heads, head_dim);
        assert!(k4.emit_ptx().contains(".version"));

        // BatchedToInterleavedKernel (seq_len, n_heads, head_dim)
        let k5 = BatchedToInterleavedKernel::new(seq_len, num_heads, head_dim);
        assert!(k5.emit_ptx().contains(".version"));

        // BatchedTransposeKernel (batch, rows, cols)
        let k6 = BatchedTransposeKernel::new(4, head_dim, seq_len);
        assert!(k6.emit_ptx().contains(".version"));

        // BatchedScaleKernel (n)
        let k7 = BatchedScaleKernel::new(seq_len * head_dim);
        assert!(k7.emit_ptx().contains(".version"));

        // BatchedSoftmaxKernel (total_rows, row_size)
        let k8 = BatchedSoftmaxKernel::new(num_heads, seq_len);
        assert!(k8.emit_ptx().contains(".version"));
    }
}

// ============================================================================
// Name and PTX Generation Tests (moved from inline transform.rs tests)
// ============================================================================

#[test]
fn test_transpose_kernel_name() {
    let kernel = TransposeKernel::new(64, 128);
    assert_eq!(kernel.name(), "transpose");
}

#[test]
fn test_transpose_ptx_generation() {
    let kernel = TransposeKernel::new(64, 128);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry transpose"));
    assert!(ptx.contains(".param .u32 rows"));
    assert!(ptx.contains(".param .u32 cols"));
}

#[test]
fn test_interleaved_to_batched_kernel_name() {
    let kernel = InterleavedToBatchedKernel::new(512, 32, 64);
    assert_eq!(kernel.name(), "interleaved_to_batched");
}

#[test]
fn test_batched_to_interleaved_kernel_name() {
    let kernel = BatchedToInterleavedKernel::new(512, 32, 64);
    assert_eq!(kernel.name(), "batched_to_interleaved");
}

#[test]
fn test_extract_single_head_kernel_name() {
    let kernel = ExtractSingleHeadKernel::new(512, 32, 64);
    assert_eq!(kernel.name(), "extract_single_head");
}

#[test]
fn test_copy_single_head_kernel_name() {
    let kernel = CopySingleHeadKernel::new(512, 32, 64);
    assert_eq!(kernel.name(), "copy_single_head");
}

#[test]
fn test_batched_transpose_kernel_name() {
    let kernel = BatchedTransposeKernel::new(32, 64, 64);
    assert_eq!(kernel.name(), "batched_transpose");
}

#[test]
fn test_batched_scale_kernel_name() {
    let kernel = BatchedScaleKernel::new(65536);
    assert_eq!(kernel.name(), "batched_scale");
}

#[test]
fn test_batched_softmax_kernel_name() {
    let kernel = BatchedSoftmaxKernel::new(1024, 64);
    assert_eq!(kernel.name(), "batched_softmax");
}

#[test]
fn test_batched_softmax_ptx_generation() {
    let kernel = BatchedSoftmaxKernel::new(1024, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry batched_softmax"));
    assert!(ptx.contains("shfl.sync.down"));
    assert!(ptx.contains("ex2.approx.f32"));
}
