use super::*;

#[test]
fn test_gemm_kernel_builds() {
    let kernel = GemmKernel::naive(1024, 1024, 1024);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".visible .entry"));
    assert!(ptx.contains("gemm"));
}

#[test]
fn test_softmax_kernel_builds() {
    let kernel = SoftmaxKernel::new(4096);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".visible .entry"));
    assert!(ptx.contains("softmax"));
}

mod barrier_safety;
mod coverage_tests;
mod dimension_independence;
mod fusion_contract_falsify;
mod property_tests;
