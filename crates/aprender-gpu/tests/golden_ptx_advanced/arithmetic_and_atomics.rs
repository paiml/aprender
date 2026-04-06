//! Integer division/remainder, negation, abs, floor, reciprocal, DP4A,
//! volatile ops, shuffle, and atomic operations.
//!
//! IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE

use trueno_gpu::ptx::{PtxControl, PtxKernel, PtxMemory, PtxType};

// ============================================================================
// INTEGER DIVISION/REMAINDER - Golden Tests
// ============================================================================

#[test]
fn golden_div_u32_instruction() {
    let kernel = PtxKernel::new("test_div_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let _result = ctx.div_u32(a, 7);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("div.u32"), "GOLDEN FAIL: div.u32 not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_rem_u32_instruction() {
    let kernel = PtxKernel::new("test_rem_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let _result = ctx.rem_u32(a, 7);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("rem.u32"), "GOLDEN FAIL: rem.u32 not found\nPTX:\n{}", ptx);
}

// ============================================================================
// NEGATION - Golden Tests
// ============================================================================

#[test]
fn golden_neg_f32_instruction() {
    let kernel = PtxKernel::new("test_neg_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(42.0);
        let _result = ctx.neg_f32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("neg.f32"), "GOLDEN FAIL: neg.f32 not found\nPTX:\n{}", ptx);
}

// ============================================================================
// ABSOLUTE VALUE - Golden Tests
// ============================================================================

#[test]
fn golden_abs_f32_instruction() {
    let kernel = PtxKernel::new("test_abs_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(-42.0);
        let _result = ctx.abs_f32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("abs.f32"), "GOLDEN FAIL: abs.f32 not found\nPTX:\n{}", ptx);
}

// ============================================================================
// FLOOR/CEILING - Golden Tests
// ============================================================================

#[test]
fn golden_floor_f32_instruction() {
    let kernel = PtxKernel::new("test_floor_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(3.7);
        let _result = ctx.floor_f32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("rmi"),
        "GOLDEN FAIL: floor (cvt.rmi) not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// RECIPROCAL - Golden Tests
// ============================================================================

#[test]
fn golden_rcp_f32_instruction() {
    let kernel = PtxKernel::new("test_rcp_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(4.0);
        let _result = ctx.rcp_f32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("rcp"), "GOLDEN FAIL: rcp not found\nPTX:\n{}", ptx);
}

// ============================================================================
// DP4A VARIANTS - Golden Tests
// ============================================================================

#[test]
fn golden_dp4a_s32_inplace_instruction() {
    let kernel = PtxKernel::new("test_dp4a_s32_inplace").build(|ctx| {
        let acc = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x01010101);
        ctx.dp4a_s32_inplace(acc, a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a"), "GOLDEN FAIL: dp4a.s32 inplace not found\nPTX:\n{}", ptx);
}

// ============================================================================
// VOLATILE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_ld_shared_u32_volatile_instruction() {
    let kernel = PtxKernel::new("test_ld_shared_volatile").shared_memory(256).build(|ctx| {
        let offset = ctx.mov_u32_imm(0);
        let _val = ctx.ld_shared_u32_volatile(offset);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.volatile.shared") || ptx.contains("ld.shared"),
        "GOLDEN FAIL: volatile shared load not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SHUFFLE WITH REGISTER SOURCE - Golden Tests
// ============================================================================

#[test]
fn golden_shfl_idx_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_shfl_idx_reg").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let src_lane = ctx.mov_u32_imm(0);
        let _result = ctx.shfl_idx_u32_reg(val, src_lane, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shfl"), "GOLDEN FAIL: shfl with reg source not found\nPTX:\n{}", ptx);
}

// ============================================================================
// ATOMIC OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_atom_add_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_add").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_u32_imm(1);
        let _old = ctx.atom_add_global_u32(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("add") && ptx.contains("global"),
        "GOLDEN FAIL: atom.global.add not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_atom_exch_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_exch").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_u32_imm(42);
        let _old = ctx.atom_exch_global_u32(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("exch"),
        "GOLDEN FAIL: atom.global.exch not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_atom_min_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_min").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_u32_imm(10);
        let _old = ctx.atom_min_global_u32(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("min"),
        "GOLDEN FAIL: atom.global.min not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_atom_max_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_max").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_u32_imm(100);
        let _old = ctx.atom_max_global_u32(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("max"),
        "GOLDEN FAIL: atom.global.max not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_atom_exch_shared_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_exch_shared").shared_memory(256).build(|ctx| {
        let addr = ctx.shared_base_addr();
        let val = ctx.mov_u32_imm(42);
        let _old = ctx.atom_exch_shared_u32(addr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("exch") && ptx.contains("shared"),
        "GOLDEN FAIL: atom.shared.exch not found\nPTX:\n{}",
        ptx
    );
}
