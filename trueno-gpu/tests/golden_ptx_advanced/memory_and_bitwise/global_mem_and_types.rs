//! Global memory operations (various types) and type conversions (various widths).
//!
//! IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE

use super::*;

// ============================================================================
// GLOBAL MEMORY OPERATIONS - Various Types
// ============================================================================

#[test]
fn golden_st_global_u32_instruction() {
    let kernel = PtxKernel::new("test_st_global_u32").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_u32_imm(42);
        ctx.st_global_u32(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("u32"),
        "GOLDEN FAIL: st.global.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_global_u64_instruction() {
    let kernel = PtxKernel::new("test_ld_global_u64").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u64(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global") && ptx.contains("u64"),
        "GOLDEN FAIL: ld.global.u64 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_u64_instruction() {
    let kernel = PtxKernel::new("test_st_global_u64").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_u64_imm(0xDEADBEEF);
        ctx.st_global_u64(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("u64"),
        "GOLDEN FAIL: st.global.u64 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_global_u8_instruction() {
    let kernel = PtxKernel::new("test_ld_global_u8").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u8(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global") && ptx.contains("u8"),
        "GOLDEN FAIL: ld.global.u8 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_u8_instruction() {
    let kernel = PtxKernel::new("test_st_global_u8").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_u32_imm(255);
        ctx.st_global_u8(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("u8"),
        "GOLDEN FAIL: st.global.u8 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_global_u16_instruction() {
    let kernel = PtxKernel::new("test_ld_global_u16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u16(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global") && ptx.contains("u16"),
        "GOLDEN FAIL: ld.global.u16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_u16_instruction() {
    let kernel = PtxKernel::new("test_st_global_u16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_u32_imm(65535);
        ctx.st_global_u16(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("u16"),
        "GOLDEN FAIL: st.global.u16 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// TYPE CONVERSIONS - Various Width
// ============================================================================

#[test]
fn golden_cvt_u32_u8_instruction() {
    let kernel = PtxKernel::new("test_cvt_u32_u8").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let byte = ctx.ld_global_u8(ptr);
        let _wide = ctx.cvt_u32_u8(byte);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("u32"),
        "GOLDEN FAIL: cvt.u32.u8 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_u32_u16_instruction() {
    let kernel = PtxKernel::new("test_cvt_u32_u16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let half = ctx.ld_global_u16(ptr);
        let _wide = ctx.cvt_u32_u16(half);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("u32"),
        "GOLDEN FAIL: cvt.u32.u16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_u16_u32_instruction() {
    let kernel = PtxKernel::new("test_cvt_u16_u32").build(|ctx| {
        let wide = ctx.mov_u32_imm(65535);
        let _narrow = ctx.cvt_u16_u32(wide);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("u16"),
        "GOLDEN FAIL: cvt.u16.u32 not found\nPTX:\n{}",
        ptx
    );
}
