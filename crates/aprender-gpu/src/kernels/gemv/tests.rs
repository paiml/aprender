use super::*;
use crate::kernels::Kernel;

#[test]
fn test_gemv_kernel_config() {
    let kernel = GemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
}

#[test]
fn test_gemv_kernel_name() {
    let kernel = GemvKernel::new(4096, 4096);
    assert_eq!(kernel.name(), "gemv_warp_reduce");
}

#[test]
fn test_gemv_ptx_generation() {
    let kernel = GemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains("gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64 y_ptr"));
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 x_ptr"));
}

#[test]
fn test_gemv_has_warp_shuffle() {
    let kernel = GemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Should use warp shuffle for reduction
    assert!(
        ptx.contains("shfl.sync.down") || ptx.contains("shfl.down"),
        "GEMV should use warp shuffle for reduction"
    );
}

#[test]
fn test_gemv_has_fma() {
    let kernel = GemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Should use FMA for dot product
    assert!(
        ptx.contains("fma.rn.f32") || ptx.contains("mad.f32"),
        "GEMV should use FMA for accumulation"
    );
}

// =========================================================================
// COALESCED GEMV TESTS - DECODER THROUGHPUT SPEC
// =========================================================================

#[test]
fn test_coalesced_gemv_kernel_config() {
    let kernel = CoalescedGemvKernel::new(4096, 4096);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_coalesced_gemv_kernel_name() {
    let kernel = CoalescedGemvKernel::new(4096, 4096);
    assert_eq!(kernel.name(), "gemv_coalesced");
}

#[test]
fn test_coalesced_gemv_ptx_generation() {
    let kernel = CoalescedGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".version 8.0"), "Missing PTX version");
    assert!(ptx.contains("gemv_coalesced"), "Missing kernel name");
    assert!(ptx.contains(".param .u64 y_ptr"), "Missing y_ptr param");
    assert!(ptx.contains(".param .u64 a_ptr"), "Missing a_ptr param");
    assert!(ptx.contains(".param .u64 x_ptr"), "Missing x_ptr param");
    assert!(ptx.contains(".param .u32 k_dim"), "Missing k_dim param");
    assert!(ptx.contains(".param .u32 n_dim"), "Missing n_dim param");
}

#[test]
fn test_coalesced_gemv_has_shared_memory() {
    let kernel = CoalescedGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Must declare shared memory for x vector caching
    assert!(
        ptx.contains(".shared"),
        "Coalesced GEMV must use shared memory for x caching"
    );
}

#[test]
fn test_coalesced_gemv_has_barrier() {
    let kernel = CoalescedGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Must have barrier sync for cooperative loading
    assert!(
        ptx.contains("bar.sync"),
        "Coalesced GEMV must have barrier for cooperative loading"
    );
}

#[test]
fn test_coalesced_gemv_has_predicated_load() {
    let kernel = CoalescedGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Must have predicated load for bounds checking
    assert!(
        ptx.contains("@%p"),
        "Coalesced GEMV must use predicated loads for bounds checking"
    );
}

#[test]
fn test_coalesced_gemv_has_fma() {
    let kernel = CoalescedGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Must use FMA for accumulation
    assert!(
        ptx.contains("fma.rn.f32"),
        "Coalesced GEMV must use FMA for accumulation"
    );
}
