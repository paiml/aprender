use super::super::*;

use crate::ptx::instructions::WmmaLayout;

// ========================================================================
// TENSOR CORE (WMMA) TESTS - IMP-1000a
// ========================================================================

#[test]
fn test_wmma_load_a_f16() {
    let kernel = PtxKernel::new("test_wmma_load_a")
        .param(PtxType::U64, "a_ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("a_ptr");
            let _frag_a = ctx.wmma_load_a_f16(ptr, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 a_ptr"),
        "Expected a_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_load_b_f16() {
    let kernel = PtxKernel::new("test_wmma_load_b")
        .param(PtxType::U64, "b_ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("b_ptr");
            let _frag_b = ctx.wmma_load_b_f16(ptr, 16, WmmaLayout::ColMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 b_ptr"),
        "Expected b_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_mma_f16_f32() {
    let kernel = PtxKernel::new("test_wmma_mma")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .build(|ctx| {
            let a = ctx.load_param_u64("a_ptr");
            let b = ctx.load_param_u64("b_ptr");
            let c = ctx.load_param_u64("c_ptr");

            let frag_a = ctx.wmma_load_a_f16(a, 16, WmmaLayout::RowMajor);
            let frag_b = ctx.wmma_load_b_f16(b, 16, WmmaLayout::ColMajor);
            let frag_c = ctx.wmma_load_c_f32(c, 16, WmmaLayout::RowMajor);

            let _frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Verify kernel structure
    assert!(
        ptx.contains(".visible .entry test_wmma_mma"),
        "Expected kernel entry, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_store_d_f32() {
    let kernel = PtxKernel::new("test_wmma_store")
        .param(PtxType::U64, "d_ptr")
        .build(|ctx| {
            let d = ctx.load_param_u64("d_ptr");
            // Create empty fragment for test
            let frag_d = vec![ctx.mov_f32_imm(0.0)];
            ctx.wmma_store_d_f32(d, &frag_d, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 d_ptr"),
        "Expected d_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_cvt_rn_f16x2_f32_packs_hi_then_lo_into_b32() {
    let kernel = PtxKernel::new("test_cvt_f16x2").build(|ctx| {
        let hi = ctx.mov_f32_imm(1.5);
        let lo = ctx.mov_f32_imm(-2.0);
        let _packed = ctx.cvt_rn_f16x2_f32(hi, lo);
        ctx.ret();
    });

    let ptx = kernel.emit();
    let line = ptx
        .lines()
        .find(|l| l.contains("cvt.rn.f16x2.f32"))
        .unwrap_or_else(|| panic!("no cvt.rn.f16x2.f32 in:\n{ptx}"));
    // dst is a b32 register; operand order is hi, lo.
    let ops: Vec<&str> = line
        .trim()
        .trim_end_matches(';')
        .split_whitespace()
        .skip(1)
        .collect::<String>()
        .leak()
        .split(',')
        .collect();
    assert_eq!(ops.len(), 3, "{line}");
    assert!(ops[0].starts_with("%rb"), "dst must be b32: {line}");
    assert!(ptx.contains(".reg .b32"), "{ptx}");
    assert!(
        !ptx.contains(".reg .f16x2"),
        "f16x2 is never a register type: {ptx}"
    );
}

#[test]
fn test_cvt_f16_f32() {
    let kernel = PtxKernel::new("test_cvt_f16").build(|ctx| {
        let f32_val = ctx.mov_f32_imm(1.5);
        let _f16_val = ctx.cvt_f16_f32(f32_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt"),
        "Expected cvt instruction, got: {}",
        ptx
    );
}

#[test]
fn test_cvt_f32_f16() {
    let kernel = PtxKernel::new("test_cvt_f32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let f16_val = ctx.ld_global_f16(ptr);
            let _f32_val = ctx.cvt_f32_f16(f16_val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 ptr"),
        "Expected ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_ld_st_global_f16() {
    let kernel = PtxKernel::new("test_f16_mem")
        .param(PtxType::U64, "in_ptr")
        .param(PtxType::U64, "out_ptr")
        .build(|ctx| {
            let in_ptr = ctx.load_param_u64("in_ptr");
            let out_ptr = ctx.load_param_u64("out_ptr");
            let val = ctx.ld_global_f16(in_ptr);
            ctx.st_global_f16(out_ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 in_ptr") && ptx.contains(".param .u64 out_ptr"),
        "Expected both params, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_load_c_f32_fragment() {
    // Test WMMA load C (accumulator) fragment
    let kernel = PtxKernel::new("test_wmma_load_c")
        .shared_memory(1024)
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let _frag_c = ctx.wmma_load_c_f32(addr, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma.load.c.sync.aligned"),
        "Expected wmma.load.c in: {}",
        ptx
    );
}

#[test]
fn test_wmma_store_d_empty_fragment() {
    // Test WMMA store with empty fragment (should be no-op)
    let kernel = PtxKernel::new("test_wmma_store_empty").build(|ctx| {
        let addr = ctx.shared_base_addr();
        let empty_frag: Vec<VirtualReg> = Vec::new();
        ctx.wmma_store_d_f32(addr, &empty_frag, 16, WmmaLayout::RowMajor);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // With empty fragment, wmma_store should return early
    assert!(
        !ptx.contains("wmma.store"),
        "Expected no wmma.store with empty fragment"
    );
}

#[test]
fn test_wmma_layout_col_major() {
    // Test column-major WMMA layout
    let kernel = PtxKernel::new("test_wmma_col")
        .shared_memory(1024)
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let _frag_a = ctx.wmma_load_a_f16(addr, 16, WmmaLayout::ColMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains(".col."), "Expected .col. layout in: {}", ptx);
}
