//! WGSL Backend Tests (Dual-Backend Support)

use super::*;

#[test]
fn test_f062_wgsl_generation_valid() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(wgsl.contains("@compute"), "Missing @compute attribute");
    assert!(wgsl.contains("@workgroup_size"), "Missing workgroup_size");
    assert!(wgsl.contains("workgroupBarrier"), "Missing workgroup barrier");
}

#[test]
fn test_f062_wgsl_has_bindings() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(wgsl.contains("@group(0) @binding(0)"), "Missing input binding");
    assert!(wgsl.contains("@group(0) @binding(1)"), "Missing output binding");
    assert!(wgsl.contains("@group(0) @binding(2)"), "Missing sizes binding");
}

#[test]
fn test_f062_wgsl_has_shared_memory() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(wgsl.contains("var<workgroup>"), "Missing workgroup shared memory");
}

#[test]
fn test_f063_wgsl_batch_size_embedded() {
    let kernel = Lz4WarpCompressKernel::new(500);
    let wgsl = kernel.emit_wgsl();
    assert!(wgsl.contains("500u"), "Batch size should be embedded in WGSL");
}

#[test]
fn test_f063_wgsl_has_entry_point() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(wgsl.contains("fn lz4_compress_warp"), "Missing entry point function");
}

#[test]
fn test_f064_wgsl_has_builtins() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(wgsl.contains("@builtin(workgroup_id)"), "Missing workgroup_id builtin");
    assert!(wgsl.contains("@builtin(local_invocation_id)"), "Missing local_invocation_id builtin");
}

#[test]
fn test_f064_dual_backend_consistency() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    let wgsl = kernel.emit_wgsl();

    // Both should have the same logical structure
    assert!(ptx.contains("bar.sync") || ptx.contains("barrier"), "PTX missing barrier");
    assert!(wgsl.contains("workgroupBarrier"), "WGSL missing barrier");

    // Both should have the same entry point name
    assert!(ptx.contains("lz4_compress_warp"));
    assert!(wgsl.contains("lz4_compress_warp"));
}

#[test]
fn test_f046_wgsl_zero_page_detection() {
    // F046: WGSL shader also has zero-page detection
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();

    // Should have OR operations for zero detection
    assert!(wgsl.contains("thread_or = thread_or |"), "Missing thread OR reduction");
    // Should have conditional for zero page
    assert!(wgsl.contains("if (page_or == 0u)"), "Missing zero page check");
    // Should output minimal size for zero pages
    assert!(wgsl.contains("20u"), "Missing compressed zero page size");
}

#[test]
fn test_f047_wgsl_reduction_barrier() {
    // F047: WGSL has proper barriers for reduction
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();

    // Should have multiple workgroup barriers
    let barrier_count = wgsl.matches("workgroupBarrier()").count();
    assert!(barrier_count >= 3, "Should have at least 3 barriers, found {}", barrier_count);
}
