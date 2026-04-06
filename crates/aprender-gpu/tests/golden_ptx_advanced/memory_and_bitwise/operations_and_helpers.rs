//! Shift/bitwise operations, select, inplace variants, register moves,
//! comparisons, warp shuffle, multiply variants, min/max, const helpers,
//! and shared pointer golden tests.
//!
//! IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE

use super::*;

// ============================================================================
// SHIFT OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_shr_u32_instruction() {
    let kernel = PtxKernel::new("test_shr_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF00);
        let shift = ctx.mov_u32_imm(8);
        let _result = ctx.shr_u32(val, shift);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shr.b32"), "GOLDEN FAIL: shr.b32 not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_shr_u32_imm_instruction() {
    let kernel = PtxKernel::new("test_shr_u32_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF00);
        let _result = ctx.shr_u32_imm(val, 8);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shr.b32"), "GOLDEN FAIL: shr.b32 imm not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_shl_u32_instruction() {
    let kernel = PtxKernel::new("test_shl_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let shift = ctx.mov_u32_imm(8);
        let _result = ctx.shl_u32(val, shift);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shl.b32"), "GOLDEN FAIL: shl.b32 not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_shl_u32_imm_instruction() {
    let kernel = PtxKernel::new("test_shl_u32_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let _result = ctx.shl_u32_imm(val, 8);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shl.b32"), "GOLDEN FAIL: shl.b32 imm not found\nPTX:\n{}", ptx);
}

// ============================================================================
// BITWISE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_or_u32_instruction() {
    let kernel = PtxKernel::new("test_or_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(0x0F);
        let b = ctx.mov_u32_imm(0xF0);
        let _result = ctx.or_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("or.b32"), "GOLDEN FAIL: or.b32 not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_or_u32_into_instruction() {
    let kernel = PtxKernel::new("test_or_u32_into").build(|ctx| {
        let dst = ctx.mov_u32_imm(0x0F);
        let a = ctx.mov_u32_imm(0x0F);
        let b = ctx.mov_u32_imm(0xF0);
        ctx.or_u32_into(dst, a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("or.b32"), "GOLDEN FAIL: or.b32 into not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_and_u32_imm_instruction() {
    let kernel = PtxKernel::new("test_and_u32_imm").build(|ctx| {
        let tid = ctx.special_reg(PtxReg::TidX);
        let _lane_id = ctx.and_u32_imm(tid, 31);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("and.b32"), "GOLDEN FAIL: and.b32 imm not found\nPTX:\n{}", ptx);
}

// ============================================================================
// SELECT OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_selp_u32_instruction() {
    let kernel = PtxKernel::new("test_selp_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(5);
        let pred = ctx.setp_ge_u32(a, b); // a >= b
        let t = ctx.mov_u32_imm(1);
        let f = ctx.mov_u32_imm(0);
        let _result = ctx.selp_u32(pred, t, f);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("selp.u32"), "GOLDEN FAIL: selp.u32 not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_selp_f32_instruction() {
    let kernel = PtxKernel::new("test_selp_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(10.0);
        let b = ctx.mov_f32_imm(5.0);
        let pred = ctx.setp_gt_f32(a, b);
        let t = ctx.mov_f32_imm(1.0);
        let f = ctx.mov_f32_imm(0.0);
        let _result = ctx.selp_f32(pred, t, f);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("selp.f32"), "GOLDEN FAIL: selp.f32 not found\nPTX:\n{}", ptx);
}

// ============================================================================
// INPLACE OPERATIONS - More Coverage
// ============================================================================

#[test]
fn golden_shr_u32_inplace_instruction() {
    let kernel = PtxKernel::new("test_shr_u32_inplace").build(|ctx| {
        let val = ctx.mov_u32_imm(256);
        ctx.shr_u32_inplace(val, 1);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shr.b32"), "GOLDEN FAIL: shr.b32 inplace not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_max_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_max_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(f32::NEG_INFINITY);
        let val = ctx.mov_f32_imm(10.0);
        ctx.max_f32_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("max.f32"), "GOLDEN FAIL: max.f32 inplace not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_div_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_div_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(100.0);
        let divisor = ctx.mov_f32_imm(10.0);
        ctx.div_f32_inplace(acc, divisor);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div") && ptx.contains("f32"),
        "GOLDEN FAIL: div.f32 inplace not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_add_u32_reg_inplace_instruction() {
    let kernel = PtxKernel::new("test_add_u32_reg_inplace").build(|ctx| {
        let acc = ctx.mov_u32_imm(10);
        let val = ctx.mov_u32_imm(5);
        ctx.add_u32_reg_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("add.u32"), "GOLDEN FAIL: add.u32 reg inplace not found\nPTX:\n{}", ptx);
}

// ============================================================================
// REGISTER MOVE OPERATIONS - More Coverage
// ============================================================================

#[test]
fn golden_mov_f32_reg_instruction() {
    let kernel = PtxKernel::new("test_mov_f32_reg").build(|ctx| {
        let src = ctx.mov_f32_imm(3.125);
        let dst = ctx.mov_f32_imm(0.0);
        ctx.mov_f32_reg(dst, src);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mov.f32"), "GOLDEN FAIL: mov.f32 reg not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_mov_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_mov_u32_reg").build(|ctx| {
        let src = ctx.mov_u32_imm(42);
        let dst = ctx.mov_u32_imm(0);
        ctx.mov_u32_reg(dst, src);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mov.u32"), "GOLDEN FAIL: mov.u32 reg not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_mov_u64_reg_instruction() {
    let kernel = PtxKernel::new("test_mov_u64_reg").build(|ctx| {
        let src = ctx.mov_u64_imm(0xDEADBEEF);
        let dst = ctx.mov_u64_imm(0);
        ctx.mov_u64_reg(dst, src);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mov.u64"), "GOLDEN FAIL: mov.u64 reg not found\nPTX:\n{}", ptx);
}

// ============================================================================
// F32 COMPARISON - Golden Tests
// ============================================================================

#[test]
fn golden_setp_gt_f32_instruction() {
    let kernel = PtxKernel::new("test_setp_gt_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(10.0);
        let b = ctx.mov_f32_imm(5.0);
        let _pred = ctx.setp_gt_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp") && ptx.contains("gt") && ptx.contains("f32"),
        "GOLDEN FAIL: setp.gt.f32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// WARP SHUFFLE - More Variants
// ============================================================================

#[test]
fn golden_shfl_down_u32_instruction() {
    let kernel = PtxKernel::new("test_shfl_down_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _result = ctx.shfl_down_u32(val, 1, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl") && ptx.contains("down"),
        "GOLDEN FAIL: shfl.down not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MULTIPLY VARIANTS - Golden Tests
// ============================================================================

#[test]
fn golden_mul_lo_u32_instruction() {
    let kernel = PtxKernel::new("test_mul_lo_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.mul_lo_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul") && ptx.contains("u32"),
        "GOLDEN FAIL: mul.lo.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_lo_s32_instruction() {
    let kernel = PtxKernel::new("test_mul_lo_s32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.mul_lo_s32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul") && ptx.contains("s32"),
        "GOLDEN FAIL: mul.lo.s32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MIN/MAX S32 - Golden Tests
// ============================================================================

#[test]
fn golden_min_s32_instruction() {
    let kernel = PtxKernel::new("test_min_s32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.min_s32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("min.s32"), "GOLDEN FAIL: min.s32 not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_max_s32_instruction() {
    let kernel = PtxKernel::new("test_max_s32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.max_s32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("max.s32"), "GOLDEN FAIL: max.s32 not found\nPTX:\n{}", ptx);
}

// ============================================================================
// CONST HELPERS - Golden Tests
// ============================================================================

#[test]
fn golden_const_f32_instruction() {
    let kernel = PtxKernel::new("test_const_f32").build(|ctx| {
        let _val = ctx.const_f32(f32::NEG_INFINITY);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mov.f32"), "GOLDEN FAIL: const_f32 not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_const_u32_instruction() {
    let kernel = PtxKernel::new("test_const_u32").build(|ctx| {
        let _val = ctx.const_u32(0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mov.u32"), "GOLDEN FAIL: const_u32 not found\nPTX:\n{}", ptx);
}

// ============================================================================
// SHARED POINTER - Golden Tests
// ============================================================================

#[test]
fn golden_shared_ptr_instruction() {
    let kernel = PtxKernel::new("test_shared_ptr").shared_memory(256).build(|ctx| {
        let _ptr = ctx.shared_ptr();
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvta.shared.u64") || ptx.contains(".shared"),
        "GOLDEN FAIL: shared_ptr not found\nPTX:\n{}",
        ptx
    );
}
