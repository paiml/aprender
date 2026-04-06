use super::*;
use crate::kernels::Kernel;

#[test]
fn test_rope_kernel_name() {
    let kernel = RopeKernel::new(32, 64, 10000.0);
    assert_eq!(kernel.name(), "rope");
}

#[test]
fn test_rope_ptx_generation() {
    let kernel = RopeKernel::new(32, 64, 10000.0);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry rope"));
    assert!(ptx.contains("sin.approx.f32"));
    assert!(ptx.contains("cos.approx.f32"));
}

#[test]
fn test_rope_indirect_kernel_name() {
    let kernel = RopeIndirectKernel::new(32, 64, 10000.0);
    assert_eq!(kernel.name(), "rope_indirect");
}

#[test]
fn test_rope_indirect_ptx_generation() {
    let kernel = RopeIndirectKernel::new(32, 64, 10000.0);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry rope_indirect"));
    assert!(ptx.contains(".param .u64 pos_ptr"));
    assert!(ptx.contains("ld.global.u32"));
}

#[test]
fn test_rope_neox_kernel_name() {
    let kernel = RopeNeoxKernel::new(32, 64, 1_000_000.0);
    assert_eq!(kernel.name(), "rope_neox");
}

#[test]
fn test_rope_neox_indirect_kernel_name() {
    let kernel = RopeNeoxIndirectKernel::new(32, 64, 1_000_000.0);
    assert_eq!(kernel.name(), "rope_neox_indirect");
}

#[test]
fn test_batched_rope_kernel_name() {
    let kernel = BatchedRopeKernel::new(32, 64, 4, 10000.0);
    assert_eq!(kernel.name(), "batched_rope");
}

#[test]
fn test_batched_rope_ptx_generation() {
    let kernel = BatchedRopeKernel::new(32, 64, 4, 10000.0);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry batched_rope"));
    assert!(ptx.contains(".param .u64 positions_ptr"));
}

#[test]
fn test_precise_rope_kernel_name() {
    let kernel = PreciseRopeKernel::new(32, 64, 1_000_000.0);
    assert_eq!(kernel.name(), "rope_precise");
}

#[test]
fn test_precise_rope_indirect_kernel_name() {
    let kernel = PreciseRopeIndirectKernel::new(32, 64, 1_000_000.0);
    assert_eq!(kernel.name(), "rope_precise_indirect");
}
