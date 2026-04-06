//! COVERAGE-BOOST: In-Place Register Operations, Constant Wrappers, and Warp Shuffle

use super::*;

// =========================================================================
// In-Place Register Operations
// =========================================================================

#[test]
fn test_fma_f32_inplace() {
    let kernel = PtxKernel::new("test_fma_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(0.0);
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        ctx.fma_f32_inplace(acc, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("fma") || ptx.contains("mad"), "Expected fma/mad in: {}", ptx);
}

#[test]
fn test_max_f32_inplace() {
    let kernel = PtxKernel::new("test_max_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(1.0);
        let val = ctx.mov_f32_imm(5.0);
        ctx.max_f32_inplace(acc, val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("max"), "Expected max in: {}", ptx);
}

#[test]
fn test_mul_f32_inplace() {
    let kernel = PtxKernel::new("test_mul_inplace").build(|ctx| {
        let val = ctx.mov_f32_imm(2.0);
        let factor = ctx.mov_f32_imm(3.0);
        ctx.mul_f32_inplace(val, factor);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul"), "Expected mul in: {}", ptx);
}

#[test]
fn test_shr_u32_inplace() {
    let kernel = PtxKernel::new("test_shr_inplace").build(|ctx| {
        let val = ctx.mov_u32_imm(256);
        ctx.shr_u32_inplace(val, 2);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shr"), "Expected shr in: {}", ptx);
}

#[test]
fn test_mov_f32_reg() {
    let kernel = PtxKernel::new("test_mov_f32").build(|ctx| {
        let src = ctx.mov_f32_imm(1.5);
        let dest = ctx.mov_f32_imm(0.0);
        ctx.mov_f32_reg(dest, src);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov"), "Expected mov in: {}", ptx);
}

#[test]
fn test_mov_u32_reg() {
    let kernel = PtxKernel::new("test_mov_u32_reg").build(|ctx| {
        let src = ctx.mov_u32_imm(42);
        let dest = ctx.mov_u32_imm(0);
        ctx.mov_u32_reg(dest, src);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov"), "Expected mov in: {}", ptx);
}

#[test]
fn test_mov_u64_reg() {
    let kernel = PtxKernel::new("test_mov_u64_reg").build(|ctx| {
        let src = ctx.mov_u64_imm(0x123456789ABCDEF0);
        let dest = ctx.mov_u64_imm(0);
        ctx.mov_u64_reg(dest, src);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov"), "Expected mov in: {}", ptx);
}

#[test]
fn test_mov_u32_inplace() {
    let kernel = PtxKernel::new("test_mov_u32_inplace").build(|ctx| {
        let dest = ctx.mov_u32_imm(0);
        ctx.mov_u32_inplace(dest, 999);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov") && ptx.contains("999"), "Expected mov with 999 in: {}", ptx);
}

// =========================================================================
// Constant Wrappers
// =========================================================================

#[test]
fn test_const_f32_wrapper() {
    let kernel = PtxKernel::new("test_const_f32").build(|ctx| {
        let _val = ctx.const_f32(std::f32::consts::PI);
        ctx.ret();
    });
    let ptx = kernel.emit();
    // Float constants are emitted as hex (0F...) in PTX
    assert!(ptx.contains("mov.f32") && ptx.contains("0F"), "Expected const in: {}", ptx);
}

#[test]
fn test_const_u32_wrapper() {
    let kernel = PtxKernel::new("test_const_u32").build(|ctx| {
        let _val = ctx.const_u32(12345);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov") && ptx.contains("12345"), "Expected const in: {}", ptx);
}

#[test]
fn test_shared_ptr_alias() {
    let kernel = PtxKernel::new("test_shared_ptr").shared_memory(256).build(|ctx| {
        let _ptr = ctx.shared_ptr();
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shared"), "Expected shared in: {}", ptx);
}

// =========================================================================
// Warp Shuffle
// =========================================================================

#[test]
fn test_shfl_down_u32() {
    let kernel = PtxKernel::new("test_shfl_down").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _result = ctx.shfl_down_u32(val, 1, 0x1F); // offset=1, mask=0x1F for 32 lanes
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shfl"), "Expected shfl in: {}", ptx);
}
