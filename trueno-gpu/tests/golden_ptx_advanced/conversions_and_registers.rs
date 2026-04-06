//! Type conversion, shared memory integer, and special register operations.
//!
//! IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE

use trueno_gpu::ptx::{PtxControl, PtxKernel, PtxMemory, PtxReg};

// ============================================================================
// TYPE CONVERSION OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_cvt_u32_u64_instruction() {
    let kernel = PtxKernel::new("test_cvt_u32_u64").build(|ctx| {
        let wide = ctx.mov_u64_imm(42);
        let _narrow = ctx.cvt_u32_u64(wide);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("cvt.u32.u64"), "GOLDEN FAIL: cvt.u32.u64 not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_cvt_f32_s32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_s32").build(|ctx| {
        let int_val = ctx.mov_u32_imm(42);
        let _float_val = ctx.cvt_f32_s32(int_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("f32") && ptx.contains("s32"),
        "GOLDEN FAIL: cvt.f32.s32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SHARED MEMORY INTEGER OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_ld_shared_u32_instruction() {
    let kernel = PtxKernel::new("test_ld_shared_u32").shared_memory(256).build(|ctx| {
        let offset = ctx.mov_u32_imm(0);
        let _val = ctx.ld_shared_u32(offset);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.shared") && ptx.contains("u32"),
        "GOLDEN FAIL: ld.shared.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_shared_u32_instruction() {
    let kernel = PtxKernel::new("test_st_shared_u32").shared_memory(256).build(|ctx| {
        let offset = ctx.mov_u32_imm(0);
        let val = ctx.mov_u32_imm(42);
        ctx.st_shared_u32(offset, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.shared") && ptx.contains("u32"),
        "GOLDEN FAIL: st.shared.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_shared_u16_instruction() {
    let kernel = PtxKernel::new("test_st_shared_u16").shared_memory(256).build(|ctx| {
        let offset = ctx.mov_u32_imm(0);
        let val = ctx.mov_u32_imm(42);
        ctx.st_shared_u16(offset, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.shared") && ptx.contains("u16"),
        "GOLDEN FAIL: st.shared.u16 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SPECIAL REGISTERS - Golden Tests (Y, Z dimensions)
// ============================================================================

#[test]
fn golden_special_reg_tid_y() {
    let kernel = PtxKernel::new("test_tid_y").build(|ctx| {
        let _tid = ctx.special_reg(PtxReg::TidY);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%tid.y"), "GOLDEN FAIL: %tid.y not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_special_reg_tid_z() {
    let kernel = PtxKernel::new("test_tid_z").build(|ctx| {
        let _tid = ctx.special_reg(PtxReg::TidZ);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%tid.z"), "GOLDEN FAIL: %tid.z not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_special_reg_ctaid_y() {
    let kernel = PtxKernel::new("test_ctaid_y").build(|ctx| {
        let _ctaid = ctx.special_reg(PtxReg::CtaIdY);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%ctaid.y"), "GOLDEN FAIL: %ctaid.y not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_special_reg_ctaid_z() {
    let kernel = PtxKernel::new("test_ctaid_z").build(|ctx| {
        let _ctaid = ctx.special_reg(PtxReg::CtaIdZ);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%ctaid.z"), "GOLDEN FAIL: %ctaid.z not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_special_reg_laneid() {
    let kernel = PtxKernel::new("test_laneid").build(|ctx| {
        let _lane = ctx.special_reg(PtxReg::LaneId);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%laneid"), "GOLDEN FAIL: %laneid not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_special_reg_warpid() {
    let kernel = PtxKernel::new("test_warpid").build(|ctx| {
        let _warp = ctx.special_reg(PtxReg::WarpId);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%warpid"), "GOLDEN FAIL: %warpid not found\nPTX:\n{}", ptx);
}
