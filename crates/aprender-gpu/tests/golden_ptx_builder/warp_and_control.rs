//! Golden tests for warp operations and control flow PTX instructions.

use trueno_gpu::ptx::{PtxComparison, PtxControl, PtxKernel};

// ============================================================================
// WARP OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_shfl_down_f32_instruction() {
    let kernel = PtxKernel::new("test_shfl_down_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(1.0);
        let _result = ctx.shfl_down_f32(val, 16, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl.sync.down") || ptx.contains("shfl.down"),
        "GOLDEN FAIL: shfl.down instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_shfl_idx_f32_instruction() {
    let kernel = PtxKernel::new("test_shfl_idx_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(1.0);
        let _result = ctx.shfl_idx_f32(val, 0, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl.sync.idx") || ptx.contains("shfl.idx"),
        "GOLDEN FAIL: shfl.idx instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ballot_sync_instruction() {
    let kernel = PtxKernel::new("test_ballot_sync").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(0);
        let pred = ctx.setp_eq_u32(a, b);
        let _result = ctx.ballot_sync(pred, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("vote.sync.ballot") || ptx.contains("vote.ballot"),
        "GOLDEN FAIL: ballot instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// CONTROL FLOW - Golden Tests
// ============================================================================

#[test]
fn golden_bar_sync_instruction() {
    let kernel = PtxKernel::new("test_bar_sync")
        .shared_memory(256)
        .build(|ctx| {
            ctx.bar_sync(0);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: bar.sync instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_branch_instruction() {
    let kernel = PtxKernel::new("test_branch").build(|ctx| {
        ctx.branch("target");
        ctx.label("target");
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("bra target"),
        "GOLDEN FAIL: bra instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_branch_if_instruction() {
    let kernel = PtxKernel::new("test_branch_if").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(0);
        let pred = ctx.setp_eq_u32(a, b);
        ctx.branch_if(pred, "taken");
        ctx.label("taken");
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("@%p") && ptx.contains("bra taken"),
        "GOLDEN FAIL: conditional branch not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_label_emission() {
    let kernel = PtxKernel::new("test_labels").build(|ctx| {
        ctx.label("loop_start");
        ctx.label("loop_end");
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("loop_start:") && ptx.contains("loop_end:"),
        "GOLDEN FAIL: labels not found\nPTX:\n{}",
        ptx
    );
}
