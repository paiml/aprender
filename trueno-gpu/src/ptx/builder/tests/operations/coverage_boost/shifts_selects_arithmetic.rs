//! COVERAGE-BOOST: Shift, Bitwise, Select/Predicate, and Arithmetic Operations

use super::*;

// =========================================================================
// Shift and Bitwise Operations
// =========================================================================

#[test]
fn test_shr_u32_imm() {
    let kernel = PtxKernel::new("test_shr_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(256);
        let _shifted = ctx.shr_u32_imm(val, 4);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shr"), "Expected shr in: {}", ptx);
}

#[test]
fn test_shl_u32_imm() {
    let kernel = PtxKernel::new("test_shl_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(16);
        let _shifted = ctx.shl_u32_imm(val, 4);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shl"), "Expected shl in: {}", ptx);
}

#[test]
fn test_and_u32_imm() {
    let kernel = PtxKernel::new("test_and_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let _masked = ctx.and_u32_imm(val, 0x0F);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("and"), "Expected and in: {}", ptx);
}

#[test]
fn test_or_u32_into() {
    let kernel = PtxKernel::new("test_or_into").build(|ctx| {
        let dest = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0xF0);
        let b = ctx.mov_u32_imm(0x0F);
        ctx.or_u32_into(dest, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("or"), "Expected or in: {}", ptx);
}

// =========================================================================
// Select/Predicate Operations
// =========================================================================

#[test]
fn test_selp_u32() {
    let kernel = PtxKernel::new("test_selp_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let pred = ctx.setp_lt_u32(a, b);
        let _result = ctx.selp_u32(pred, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("selp"), "Expected selp in: {}", ptx);
}

#[test]
fn test_selp_f32() {
    let kernel = PtxKernel::new("test_selp_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(1.0);
        let b = ctx.mov_f32_imm(2.0);
        let pred = ctx.setp_gt_f32(b, a); // b > a is true
        let _result = ctx.selp_f32(pred, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("selp"), "Expected selp in: {}", ptx);
}

#[test]
fn test_setp_gt_f32() {
    let kernel = PtxKernel::new("test_setp_gt").build(|ctx| {
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(1.0);
        let _pred = ctx.setp_gt_f32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("setp.gt"), "Expected setp.gt in: {}", ptx);
}

// =========================================================================
// Arithmetic Operations
// =========================================================================

#[test]
fn test_sub_f32() {
    let kernel = PtxKernel::new("test_sub_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(5.0);
        let b = ctx.mov_f32_imm(3.0);
        let _result = ctx.sub_f32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("sub.f32"), "Expected sub.f32 in: {}", ptx);
}

#[test]
fn test_rcp_f32() {
    let kernel = PtxKernel::new("test_rcp").build(|ctx| {
        let val = ctx.mov_f32_imm(4.0);
        let _recip = ctx.rcp_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    // rcp requires .approx modifier for f32 per PTX ISA
    assert!(ptx.contains("rcp.approx.f32"), "Expected rcp.approx.f32 in: {}", ptx);
}

#[test]
fn test_abs_f32() {
    let kernel = PtxKernel::new("test_abs").build(|ctx| {
        let val = ctx.mov_f32_imm(-3.125);
        let _result = ctx.abs_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("abs"), "Expected abs in: {}", ptx);
}

#[test]
fn test_mul_lo_s32() {
    let kernel = PtxKernel::new("test_mul_s32").build(|ctx| {
        let a = ctx.mov_s32_imm(-10);
        let b = ctx.mov_s32_imm(5);
        let _result = ctx.mul_lo_s32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul"), "Expected mul in: {}", ptx);
}

#[test]
fn test_min_s32() {
    let kernel = PtxKernel::new("test_min_s32").build(|ctx| {
        let a = ctx.mov_s32_imm(-10);
        let b = ctx.mov_s32_imm(5);
        let _result = ctx.min_s32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("min"), "Expected min in: {}", ptx);
}

#[test]
fn test_max_s32() {
    let kernel = PtxKernel::new("test_max_s32").build(|ctx| {
        let a = ctx.mov_s32_imm(-10);
        let b = ctx.mov_s32_imm(5);
        let _result = ctx.max_s32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("max"), "Expected max in: {}", ptx);
}

#[test]
fn test_mov_s32_imm() {
    let kernel = PtxKernel::new("test_mov_s32").build(|ctx| {
        let _val = ctx.mov_s32_imm(-12345);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov") && ptx.contains("12345"), "Expected mov in: {}", ptx);
}
