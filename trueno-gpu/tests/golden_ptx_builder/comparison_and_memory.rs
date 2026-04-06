//! Golden tests for comparison and memory PTX operations.

use trueno_gpu::ptx::{PtxComparison, PtxControl, PtxKernel, PtxMemory, PtxType};

// ============================================================================
// COMPARISON OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_setp_lt_u32_instruction() {
    let kernel = PtxKernel::new("test_setp_lt_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(5);
        let b = ctx.mov_u32_imm(10);
        let _pred = ctx.setp_lt_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp.lt.u32"),
        "GOLDEN FAIL: setp.lt.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_setp_ge_u32_instruction() {
    let kernel = PtxKernel::new("test_setp_ge_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(5);
        let _pred = ctx.setp_ge_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp.ge.u32"),
        "GOLDEN FAIL: setp.ge.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_setp_eq_u32_instruction() {
    let kernel = PtxKernel::new("test_setp_eq_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(10);
        let _pred = ctx.setp_eq_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp.eq.u32"),
        "GOLDEN FAIL: setp.eq.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MEMORY OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_ld_global_f32_instruction() {
    let kernel = PtxKernel::new("test_ld_global_f32").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_f32(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global.f32"),
        "GOLDEN FAIL: ld.global.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_f32_instruction() {
    let kernel = PtxKernel::new("test_st_global_f32").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_f32_imm(42.0);
        ctx.st_global_f32(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global.f32"),
        "GOLDEN FAIL: st.global.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_shared_f32_instruction() {
    let kernel = PtxKernel::new("test_ld_shared_f32").shared_memory(256).build(|ctx| {
        let offset = ctx.mov_u32_imm(0);
        let _val = ctx.ld_shared_f32(offset);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.shared.f32"),
        "GOLDEN FAIL: ld.shared.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_shared_f32_instruction() {
    let kernel = PtxKernel::new("test_st_shared_f32").shared_memory(256).build(|ctx| {
        let offset = ctx.mov_u32_imm(0);
        let val = ctx.mov_f32_imm(1.0);
        ctx.st_shared_f32(offset, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.shared.f32"),
        "GOLDEN FAIL: st.shared.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}
