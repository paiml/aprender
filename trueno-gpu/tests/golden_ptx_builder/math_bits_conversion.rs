//! Golden tests for type conversion, math, and bit operations.

use trueno_gpu::ptx::{PtxArithmetic, PtxControl, PtxKernel};

// ============================================================================
// TYPE CONVERSION - Golden Tests
// ============================================================================

#[test]
fn golden_cvt_u64_u32_instruction() {
    let kernel = PtxKernel::new("test_cvt_u64_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _wide = ctx.cvt_u64_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt.u64.u32"),
        "GOLDEN FAIL: cvt.u64.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_f32_u32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _float = ctx.cvt_f32_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("f32") && ptx.contains("u32"),
        "GOLDEN FAIL: cvt.f32.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MATH OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_rsqrt_f32_instruction() {
    let kernel = PtxKernel::new("test_rsqrt_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(4.0);
        let _result = ctx.rsqrt_f32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("rsqrt"), "GOLDEN FAIL: rsqrt instruction not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_ex2_f32_instruction() {
    let kernel = PtxKernel::new("test_ex2_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(2.0);
        let _result = ctx.ex2_f32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ex2"), "GOLDEN FAIL: ex2 instruction not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_sin_f32_instruction() {
    let kernel = PtxKernel::new("test_sin_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(1.57);
        let _result = ctx.sin_f32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("sin"), "GOLDEN FAIL: sin instruction not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_max_f32_instruction() {
    let kernel = PtxKernel::new("test_max_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(1.0);
        let b = ctx.mov_f32_imm(2.0);
        let _result = ctx.max_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("max.f32"), "GOLDEN FAIL: max.f32 instruction not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_min_u32_instruction() {
    let kernel = PtxKernel::new("test_min_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(5);
        let b = ctx.mov_u32_imm(10);
        let _result = ctx.min_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("min.u32"), "GOLDEN FAIL: min.u32 instruction not found\nPTX:\n{}", ptx);
}

// ============================================================================
// BIT OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_popc_u32_instruction() {
    let kernel = PtxKernel::new("test_popc_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let _result = ctx.popc_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("popc"), "GOLDEN FAIL: popc instruction not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_clz_u32_instruction() {
    let kernel = PtxKernel::new("test_clz_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0x0F000000);
        let _result = ctx.clz_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("clz"), "GOLDEN FAIL: clz instruction not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_bfind_u32_instruction() {
    let kernel = PtxKernel::new("test_bfind_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0x0F000000);
        let _result = ctx.bfind_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("bfind"), "GOLDEN FAIL: bfind instruction not found\nPTX:\n{}", ptx);
}
