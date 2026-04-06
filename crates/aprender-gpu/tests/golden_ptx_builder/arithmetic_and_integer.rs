//! Golden tests for arithmetic and integer PTX operations.

use trueno_gpu::ptx::{PtxArithmetic, PtxControl, PtxKernel};

// ============================================================================
// ARITHMETIC OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_add_f32_instruction() {
    let kernel = PtxKernel::new("test_add_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(1.0);
        let b = ctx.mov_f32_imm(2.0);
        let _c = ctx.add_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("add.f32"),
        "GOLDEN FAIL: add.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_sub_f32_instruction() {
    let kernel = PtxKernel::new("test_sub_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(5.0);
        let b = ctx.mov_f32_imm(3.0);
        let _c = ctx.sub_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("sub.f32"),
        "GOLDEN FAIL: sub.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_f32_instruction() {
    let kernel = PtxKernel::new("test_mul_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        let _c = ctx.mul_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.f32"),
        "GOLDEN FAIL: mul.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_div_f32_with_rounding() {
    let kernel = PtxKernel::new("test_div_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(10.0);
        let b = ctx.mov_f32_imm(3.0);
        let _c = ctx.div_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div.rn.f32") || ptx.contains("div.f32"),
        "GOLDEN FAIL: div.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_fma_f32_instruction() {
    let kernel = PtxKernel::new("test_fma_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        let c = ctx.mov_f32_imm(1.0);
        let _d = ctx.fma_f32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("fma.rn.f32"),
        "GOLDEN FAIL: fma.rn.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// INTEGER OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_add_u32_instruction() {
    let kernel = PtxKernel::new("test_add_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let _b = ctx.add_u32(a, 5);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("add.u32"),
        "GOLDEN FAIL: add.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_lo_u32_instruction() {
    let kernel = PtxKernel::new("test_mul_lo_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let _b = ctx.mul_u32(a, 5);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.lo.u32") || ptx.contains("mul.u32"),
        "GOLDEN FAIL: mul.lo.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_wide_u32_instruction() {
    let kernel = PtxKernel::new("test_mul_wide_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(1000000);
        let _b = ctx.mul_wide_u32(a, 1000000);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.wide.u32"),
        "GOLDEN FAIL: mul.wide.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mad_lo_u32_instruction() {
    let kernel = PtxKernel::new("test_mad_lo_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let c = ctx.mov_u32_imm(5);
        let _d = ctx.mad_lo_u32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mad.lo.u32"),
        "GOLDEN FAIL: mad.lo.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}
