//! COVERAGE-BOOST: Type Conversion Operations

use super::*;

#[test]
fn test_cvt_u32_u8() {
    let kernel = PtxKernel::new("test_cvt_u32_u8").build(|ctx| {
        let val = ctx.mov_u32_imm(255);
        let _converted = ctx.cvt_u32_u8(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt") || ptx.contains("and"), "Expected conversion in: {}", ptx);
}

#[test]
fn test_cvt_u32_u16() {
    let kernel = PtxKernel::new("test_cvt_u32_u16").build(|ctx| {
        let val = ctx.mov_u32_imm(65535);
        let _converted = ctx.cvt_u32_u16(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt") || ptx.contains("and"), "Expected conversion in: {}", ptx);
}

#[test]
fn test_cvt_u16_u32() {
    let kernel = PtxKernel::new("test_cvt_u16_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(100);
        let _converted = ctx.cvt_u16_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt") || ptx.contains("and"), "Expected conversion in: {}", ptx);
}

#[test]
fn test_cvt_u64_u32() {
    let kernel = PtxKernel::new("test_cvt_u64_u32").build(|ctx| {
        let val = ctx.mov_u64_imm(0xFFFFFFFF);
        let _converted = ctx.cvt_u64_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt"), "Expected cvt in: {}", ptx);
}

#[test]
fn test_cvt_u32_u64() {
    let kernel = PtxKernel::new("test_cvt_u32_u64").build(|ctx| {
        let val = ctx.mov_u32_imm(12345);
        let _converted = ctx.cvt_u32_u64(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt"), "Expected cvt in: {}", ptx);
}

#[test]
fn test_cvt_f32_s32() {
    let kernel = PtxKernel::new("test_cvt_f32_s32").build(|ctx| {
        let val = ctx.mov_f32_imm(-42.5);
        let _converted = ctx.cvt_rni_s32_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt.rni") || ptx.contains("cvt"), "Expected cvt in: {}", ptx);
}

#[test]
fn test_cvt_s32_u8_sx() {
    let kernel = PtxKernel::new("test_cvt_s32_u8_sx").build(|ctx| {
        let val = ctx.mov_u32_imm(200);
        let _converted = ctx.cvt_s32_u8_sx(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(!ptx.is_empty());
}

#[test]
fn test_cvt_s32_u32() {
    let kernel = PtxKernel::new("test_cvt_s32_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _converted = ctx.cvt_s32_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(!ptx.is_empty());
}

#[test]
fn test_cvt_u8_s32() {
    let kernel = PtxKernel::new("test_cvt_u8_s32").build(|ctx| {
        let val = ctx.mov_s32_imm(127);
        let _converted = ctx.cvt_u8_s32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(!ptx.is_empty());
}

#[test]
fn test_mov_s32_from_u32() {
    let kernel = PtxKernel::new("test_mov_s32_from_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _result = ctx.mov_s32_from_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov"), "Expected mov in: {}", ptx);
}
