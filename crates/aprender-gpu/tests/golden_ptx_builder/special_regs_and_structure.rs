//! Golden tests for special registers, module structure, dp4a, and memory barriers.

use trueno_gpu::ptx::{PtxControl, PtxKernel, PtxModule, PtxReg, PtxType};

// ============================================================================
// SPECIAL REGISTERS - Golden Tests
// ============================================================================

#[test]
fn golden_special_reg_tid_x() {
    let kernel = PtxKernel::new("test_tid_x").build(|ctx| {
        let _tid = ctx.special_reg(PtxReg::TidX);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%tid.x"), "GOLDEN FAIL: %tid.x special reg not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_special_reg_ctaid_x() {
    let kernel = PtxKernel::new("test_ctaid_x").build(|ctx| {
        let _ctaid = ctx.special_reg(PtxReg::CtaIdX);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%ctaid.x"), "GOLDEN FAIL: %ctaid.x special reg not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_special_reg_ntid_x() {
    let kernel = PtxKernel::new("test_ntid_x").build(|ctx| {
        let _ntid = ctx.special_reg(PtxReg::NtidX);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("%ntid.x"), "GOLDEN FAIL: %ntid.x special reg not found\nPTX:\n{}", ptx);
}

// ============================================================================
// MODULE STRUCTURE - Golden Tests
// ============================================================================

#[test]
fn golden_module_structure() {
    let module = PtxModule::new().version(8, 0).target("sm_80").add_kernel(
        PtxKernel::new("test_kernel").build(|ctx| {
            ctx.ret();
        }),
    );

    let ptx = module.emit();

    assert!(
        ptx.contains(".version 8.0"),
        "GOLDEN FAIL: .version directive not found\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".target sm_80"),
        "GOLDEN FAIL: .target directive not found\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".entry test_kernel"),
        "GOLDEN FAIL: .entry directive not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_kernel_params() {
    let kernel = PtxKernel::new("test_params")
        .param(PtxType::U64, "input_ptr")
        .param(PtxType::U64, "output_ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let _in_ptr = ctx.load_param_u64("input_ptr");
            let _out_ptr = ctx.load_param_u64("output_ptr");
            let _n = ctx.load_param_u32("n");
            ctx.ret();
        });

    let ptx = kernel.emit();

    assert!(
        ptx.contains(".param .u64 input_ptr"),
        "GOLDEN FAIL: input_ptr param not found\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".param .u64 output_ptr"),
        "GOLDEN FAIL: output_ptr param not found\nPTX:\n{}",
        ptx
    );
    assert!(ptx.contains(".param .u32 n"), "GOLDEN FAIL: n param not found\nPTX:\n{}", ptx);
}

#[test]
fn golden_shared_memory_declaration() {
    let kernel = PtxKernel::new("test_shared").shared_memory(4096).build(|ctx| {
        ctx.ret();
    });

    let ptx = kernel.emit();

    assert!(
        ptx.contains(".shared") && ptx.contains("4096"),
        "GOLDEN FAIL: shared memory declaration not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// DP4A (INT8 DOT PRODUCT) - Golden Tests
// ============================================================================

#[test]
fn golden_dp4a_u32_instruction() {
    let kernel = PtxKernel::new("test_dp4a_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x01010101);
        let c = ctx.mov_u32_imm(0);
        let _d = ctx.dp4a_u32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a"), "GOLDEN FAIL: dp4a instruction not found\nPTX:\n{}", ptx);
}

// ============================================================================
// MEMORY BARRIER - Golden Tests
// ============================================================================

#[test]
fn golden_membar_cta_instruction() {
    let kernel = PtxKernel::new("test_membar_cta").build(|ctx| {
        ctx.membar_cta();
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("membar.cta"),
        "GOLDEN FAIL: membar.cta instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_membar_gl_instruction() {
    let kernel = PtxKernel::new("test_membar_gl").build(|ctx| {
        ctx.membar_gl();
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("membar.gl"),
        "GOLDEN FAIL: membar.gl instruction not found\nPTX:\n{}",
        ptx
    );
}
