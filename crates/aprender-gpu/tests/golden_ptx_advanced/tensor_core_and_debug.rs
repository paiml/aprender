//! F16 operations, WMMA tensor core operations, debug helpers,
//! and conversion with rounding.
//!
//! IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE

use trueno_gpu::ptx::{PtxControl, PtxKernel, PtxReg, PtxType, WmmaLayout};

// ============================================================================
// F16 OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_cvt_f16_f32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f16_f32").build(|ctx| {
        let f32_val = ctx.mov_f32_imm(3.125);
        let _f16_val = ctx.cvt_f16_f32(f32_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("f16"),
        "GOLDEN FAIL: cvt.f16.f32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_f32_f16_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_f16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let f16_val = ctx.ld_global_f16(ptr);
        let _f32_val = ctx.cvt_f32_f16(f16_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("f32"),
        "GOLDEN FAIL: cvt.f32.f16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_global_f16_instruction() {
    let kernel = PtxKernel::new("test_ld_global_f16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_f16(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // PTX uses .b16 for f16 loads
    assert!(
        ptx.contains("ld.global") && ptx.contains("b16"),
        "GOLDEN FAIL: ld.global.b16 (f16) not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_f16_instruction() {
    let kernel = PtxKernel::new("test_st_global_f16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let f32_val = ctx.mov_f32_imm(3.125);
        let f16_val = ctx.cvt_f16_f32(f32_val);
        ctx.st_global_f16(ptr, f16_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // PTX ISA requires st.global.b16 for half-precision stores (not st.global.f16)
    assert!(
        ptx.contains("st.global.b16"),
        "GOLDEN FAIL: st.global.b16 (f16 store) not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// WMMA TENSOR CORE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_wmma_load_a_f16_instruction() {
    let kernel = PtxKernel::new("test_wmma_load_a").param(PtxType::U64, "a_ptr").build(|ctx| {
        let a_ptr = ctx.load_param_u64("a_ptr");
        let _frag = ctx.wmma_load_a_f16(a_ptr, 16, WmmaLayout::RowMajor);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("load") && ptx.contains("f16"),
        "GOLDEN FAIL: wmma.load.a.f16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_load_b_f16_instruction() {
    let kernel = PtxKernel::new("test_wmma_load_b").param(PtxType::U64, "b_ptr").build(|ctx| {
        let b_ptr = ctx.load_param_u64("b_ptr");
        let _frag = ctx.wmma_load_b_f16(b_ptr, 16, WmmaLayout::ColMajor);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("load") && ptx.contains("f16"),
        "GOLDEN FAIL: wmma.load.b.f16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_load_c_f32_instruction() {
    let kernel = PtxKernel::new("test_wmma_load_c").param(PtxType::U64, "c_ptr").build(|ctx| {
        let c_ptr = ctx.load_param_u64("c_ptr");
        let _frag = ctx.wmma_load_c_f32(c_ptr, 16, WmmaLayout::RowMajor);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("load") && ptx.contains("f32"),
        "GOLDEN FAIL: wmma.load.c.f32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_init_c_zero_instruction() {
    let kernel = PtxKernel::new("test_wmma_init_c_zero").build(|ctx| {
        let _frag = ctx.wmma_init_c_zero();
        ctx.ret();
    });

    let ptx = kernel.emit();
    // wmma_init_c_zero just moves 0.0 into registers
    assert!(
        ptx.contains("mov.f32"),
        "GOLDEN FAIL: wmma_init_c_zero (mov.f32 0.0) not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_mma_f16_f32_instruction() {
    let kernel = PtxKernel::new("test_wmma_mma")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .build(|ctx| {
            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let frag_a = ctx.wmma_load_a_f16(a_ptr, 16, WmmaLayout::RowMajor);
            let frag_b = ctx.wmma_load_b_f16(b_ptr, 16, WmmaLayout::ColMajor);
            let frag_c = ctx.wmma_init_c_zero();
            let _frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("mma"),
        "GOLDEN FAIL: wmma.mma not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_store_d_f32_instruction() {
    let kernel = PtxKernel::new("test_wmma_store_d")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "d_ptr")
        .build(|ctx| {
            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let d_ptr = ctx.load_param_u64("d_ptr");
            let frag_a = ctx.wmma_load_a_f16(a_ptr, 16, WmmaLayout::RowMajor);
            let frag_b = ctx.wmma_load_b_f16(b_ptr, 16, WmmaLayout::ColMajor);
            let frag_c = ctx.wmma_init_c_zero();
            let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);
            ctx.wmma_store_d_f32(d_ptr, &frag_d, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("store"),
        "GOLDEN FAIL: wmma.store.d not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// DEBUG HELPERS - Golden Tests
// ============================================================================

#[test]
fn golden_emit_debug_marker_instruction() {
    let kernel =
        PtxKernel::new("test_debug_marker").param(PtxType::U64, "debug_ptr").build(|ctx| {
            let debug_ptr = ctx.load_param_u64("debug_ptr");
            let _slot = ctx.emit_debug_marker(debug_ptr, 0xDEAD);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // emit_debug_marker uses atom.add, st.global
    assert!(
        ptx.contains("atom") && ptx.contains("add") && ptx.contains("st.global"),
        "GOLDEN FAIL: emit_debug_marker (atom.add + st.global) not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_emit_debug_value_instruction() {
    let kernel = PtxKernel::new("test_debug_value").param(PtxType::U64, "debug_ptr").build(|ctx| {
        let debug_ptr = ctx.load_param_u64("debug_ptr");
        let value = ctx.special_reg(PtxReg::TidX);
        let _slot = ctx.emit_debug_value(debug_ptr, value);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // emit_debug_value uses atom.add, st.global
    assert!(
        ptx.contains("atom") && ptx.contains("st.global"),
        "GOLDEN FAIL: emit_debug_value (atom + st.global) not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// CONVERSION WITH ROUNDING - Golden Tests
// ============================================================================

#[test]
fn golden_cvt_rni_s32_f32_instruction() {
    let kernel = PtxKernel::new("test_cvt_rni_s32_f32").build(|ctx| {
        let f_val = ctx.mov_f32_imm(3.7);
        let _i_val = ctx.cvt_rni_s32_f32(f_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("s32"),
        "GOLDEN FAIL: cvt.rni.s32.f32 not found\nPTX:\n{}",
        ptx
    );
}
