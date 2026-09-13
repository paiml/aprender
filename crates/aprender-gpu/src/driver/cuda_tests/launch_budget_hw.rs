//! Hardware half of O2 — `register_budget` enforced against a real compiled kernel.
//!
//! `crate::launch_budget`'s case table proves the POLICY on any host. These prove the
//! MECHANISM engages: that the driver actually reports a kernel's registers and shared
//! memory back to us, and that the gate can go RED on real hardware. A policy that is
//! never fed by a real query is theater — which is exactly what
//! `contracts/trueno/ptx-codegen-safety-v1.yaml` `register_budget` has been since
//! 2026-04-06.

use super::*;
use crate::driver::budget_query::{
    device_limits, enforce_register_budget, kernel_attributes, occupancy_max_active_blocks_per_sm,
};

/// A kernel with a real body, so the JIT allocates registers it can report back.
/// A bare `ret;` can legitimately compile to zero registers and would prove nothing.
const PTX: &str = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry budget_probe(
    .param .u64 p_out,
    .param .u32 p_n
) {
    .reg .pred  %p<2>;
    .reg .f32   %f<3>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<4>;
    ld.param.u64 %rd1, [p_out];
    ld.param.u32 %r2, [p_n];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r1, %r3, %r4, %r5;
    setp.ge.s32 %p1, %r1, %r2;
    @%p1 bra DONE;
    mul.wide.s32 %rd3, %r1, 4;
    add.s64 %rd3, %rd2, %rd3;
    cvt.rn.f32.s32 %f1, %r1;
    add.f32 %f2, %f1, %f1;
    st.global.f32 [%rd3], %f2;
DONE:
    ret;
}
"#;

fn probe_function() -> (CudaContext, CudaModule, crate::driver::sys::CUfunction) {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let mut module = CudaModule::from_ptx(&ctx, PTX).expect("Module from_ptx MUST succeed");
    let func = module
        .get_function("budget_probe")
        .expect("entry budget_probe MUST resolve");
    (ctx, module, func)
}

#[test]
fn kernel_attributes_are_read_back_from_the_driver() {
    let (_ctx, _module, func) = probe_function();
    let attrs = unsafe { kernel_attributes(func) }.expect("cuFuncGetAttribute MUST succeed");

    // Prove the MECHANISM engaged: these are properties of the cubin the JIT produced,
    // not of the PTX text. A stub returning zeros would pass a weaker assertion.
    assert!(
        attrs.max_threads_per_block > 0,
        "max_threads_per_block must be positive, got {}",
        attrs.max_threads_per_block
    );
    assert!(
        attrs.num_regs > 0,
        "a kernel with a real body must use registers, got {}",
        attrs.num_regs
    );
    assert!(
        attrs.num_regs <= 255,
        "registers per thread cannot exceed 255, got {}",
        attrs.num_regs
    );
}

#[test]
fn device_limits_are_queried_not_tabulated() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let limits = device_limits(ctx.device()).expect("cuDeviceGetAttribute MUST succeed");

    // The point of querying: this passes on sm_121 (GB10) with no entry in any per-SM
    // table, where the hand-written CU_TARGET_COMPUTE_* list stops at 90.
    assert!(limits.max_threads_per_block >= 1024);
    assert!(limits.max_shared_per_block >= 16 * 1024);
    assert!(limits.max_regs_per_block >= 32 * 1024);
    assert_eq!(limits.warp_size, 32, "warp size is 32 on every CUDA arch");
}

#[test]
fn occupancy_is_positive_for_a_legal_launch() {
    let (_ctx, _module, func) = probe_function();
    let blocks = unsafe { occupancy_max_active_blocks_per_sm(func, 256, 0) }
        .expect("cuOccupancyMaxActiveBlocksPerMultiprocessor MUST succeed");
    // This IS the contract's postcondition, executed for the first time.
    assert!(
        blocks > 0,
        "occupancy must be positive for a 256-thread block"
    );
}

#[test]
fn legal_launch_passes_the_budget() {
    let (ctx, _module, func) = probe_function();
    unsafe { enforce_register_budget(func, ctx.device(), 256, 0) }
        .expect("a 256-thread launch of a tiny kernel MUST satisfy the budget");
}

#[test]
fn budget_goes_red_on_an_oversized_block() {
    // THE FALSIFIER. Without this the gate is unfalsifiable: a check that has never been
    // observed to fail is indistinguishable from one that cannot.
    let (ctx, _module, func) = probe_function();
    let attrs = unsafe { kernel_attributes(func) }.expect("attributes MUST be readable");
    let over = attrs.max_threads_per_block + 1;

    let err = unsafe { enforce_register_budget(func, ctx.device(), over, 0) }
        .expect_err("a block above the kernel's own maximum MUST be refused");
    let msg = err.to_string();
    assert!(
        msg.contains(&over.to_string()),
        "the error must name the requested block size {over}: {msg}"
    );
    // Assert the SPECIFIC violation, not merely that something failed. Checking only
    // `is_err()` here would let this test pass with the kernel-max comparison deleted —
    // the launch would fall through to the device-wide check and still error, so the
    // test would witness nothing. Measured: with that comparison removed this assertion
    // is what turns the hardware test RED.
    assert!(
        msg.contains("kernel's max_threads_per_block"),
        "must be refused by the KERNEL's ceiling, not the device's: {msg}"
    );
}

#[test]
fn budget_goes_red_on_impossible_shared_memory() {
    // Second, independent RED path: shared memory rather than block size, so the gate is
    // not merely one comparison that happens to work.
    let (ctx, _module, func) = probe_function();
    let limits = device_limits(ctx.device()).expect("limits MUST be readable");
    let err = unsafe {
        enforce_register_budget(func, ctx.device(), 128, limits.max_shared_per_block + 1)
    }
    .expect_err("dynamic shared memory above the device limit MUST be refused");
    assert!(
        err.to_string().contains("shared memory"),
        "error must identify the shared-memory budget: {err}"
    );
}
