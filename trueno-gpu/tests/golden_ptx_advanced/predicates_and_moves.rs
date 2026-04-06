//! Vector load/store, predicate, register move, and inplace operations.
//!
//! IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE

use trueno_gpu::ptx::{PtxComparison, PtxControl, PtxKernel, PtxType};

// ============================================================================
// VECTOR LOAD/STORE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_ld_global_f32_v4_instruction() {
    let kernel = PtxKernel::new("test_ld_v4").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _vals = ctx.ld_global_f32_v4(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0]
    assert!(
        ptx.contains("ld.global") && ptx.contains("v4"),
        "GOLDEN FAIL: v4 load not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// PREDICATE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_and_pred_instruction() {
    let kernel = PtxKernel::new("test_and_pred").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(1);
        let c = ctx.mov_u32_imm(0);
        let p1 = ctx.setp_eq_u32(a, b);
        let p2 = ctx.setp_eq_u32(a, c);
        let _p3 = ctx.and_pred(p1, p2);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: and.pred %p{dst}, %p{src1}, %p{src2}
    assert!(ptx.contains("and.pred"), "GOLDEN FAIL: and.pred instruction not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_branch_if_not_instruction() {
    let kernel = PtxKernel::new("test_branch_if_not").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(0);
        let pred = ctx.setp_eq_u32(a, b);
        ctx.branch_if_not(pred, "not_taken");
        ctx.label("not_taken");
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: @!%p{pred} bra label
    assert!(
        ptx.contains("@!%p") && ptx.contains("bra"),
        "GOLDEN FAIL: negated predicate branch not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// REGISTER MOVE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_mov_u64_imm_instruction() {
    let kernel = PtxKernel::new("test_mov_u64_imm").build(|ctx| {
        let _val = ctx.mov_u64_imm(0xDEADBEEFCAFEBABE);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: mov.u64 %rd{dst}, {immediate}
    assert!(ptx.contains("mov.u64"), "GOLDEN FAIL: mov.u64 instruction not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_mov_u64_into_instruction() {
    let kernel = PtxKernel::new("test_mov_u64_into").build(|ctx| {
        let dst = ctx.mov_u64_imm(0);
        ctx.mov_u64_into(dst, 42);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.u64"),
        "GOLDEN FAIL: mov.u64 into instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mov_u32_into_instruction() {
    let kernel = PtxKernel::new("test_mov_u32_into").build(|ctx| {
        let dst = ctx.mov_u32_imm(0);
        ctx.mov_u32_into(dst, 42);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.u32"),
        "GOLDEN FAIL: mov.u32 into instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// INPLACE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_add_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_add_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(1.0);
        let val = ctx.mov_f32_imm(2.0);
        ctx.add_f32_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("add.f32"), "GOLDEN FAIL: add.f32 inplace not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_mul_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_mul_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(2.0);
        let val = ctx.mov_f32_imm(3.0);
        ctx.mul_f32_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mul.f32"), "GOLDEN FAIL: mul.f32 inplace not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_add_u32_inplace_instruction() {
    let kernel = PtxKernel::new("test_add_u32_inplace").build(|ctx| {
        let acc = ctx.mov_u32_imm(10);
        ctx.add_u32_inplace(acc, 5);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("add.u32"), "GOLDEN FAIL: add.u32 inplace not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_fma_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_fma_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(1.0);
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        ctx.fma_f32_inplace(acc, a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("fma"), "GOLDEN FAIL: fma inplace not found\nPTX:\n{}", ptx);
}
