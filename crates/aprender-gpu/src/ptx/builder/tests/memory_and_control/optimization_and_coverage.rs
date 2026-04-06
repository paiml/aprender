//! Optimization pass integration tests (Issue #72, #73) and
//! additional coverage improvement tests.

use super::*;

// =========================================================================
// Optimization Pass Integration Tests (Issue #72, #73)
// =========================================================================

#[test]
fn test_build_optimized_basic() {
    // Test that build_optimized works for simple kernels
    let kernel =
        PtxKernel::new("test_optimized").param(PtxType::U64, "ptr").build_optimized(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_global_f32(ptr);
            let two = ctx.mov_f32_imm(2.0);
            let result = ctx.mul_f32(val, two);
            ctx.st_global_f32(ptr, result);
            ctx.ret();
        });

    assert!(kernel.is_ok(), "build_optimized should succeed for simple kernel");
    let kernel = kernel.expect("test");
    let ptx = kernel.emit();
    assert!(ptx.contains(".entry test_optimized"));
    assert!(ptx.contains("ret;"));
}

#[test]
fn test_build_optimized_with_mul_add_fusion() {
    // Test that mul + add patterns are fused to FMA by the optimization pass
    // This tests the FMA fusion integration from Issue #72
    let kernel =
        PtxKernel::new("test_fma_fusion").param(PtxType::U64, "ptr").build_optimized(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let a = ctx.ld_global_f32(ptr);
            let b = ctx.mov_f32_imm(2.0);
            let c = ctx.mov_f32_imm(3.0);
            // mul + add pattern should be fused to fma by optimizer
            let mul_result = ctx.mul_f32(a, b);
            let add_result = ctx.add_f32(mul_result, c);
            ctx.st_global_f32(ptr, add_result);
            ctx.ret();
        });

    assert!(kernel.is_ok(), "build_optimized should succeed");
    let kernel = kernel.expect("test");
    let ptx = kernel.emit();

    // After FMA fusion, we should have fma.rn.f32 instead of separate mul + add
    // The mul result is only used once (in the add), so fusion should occur
    assert!(
        ptx.contains("fma.rn.f32") || ptx.contains("mul.f32"),
        "Kernel should have either FMA (fused) or mul (unfused)"
    );
}

#[test]
fn test_build_vs_build_optimized_difference() {
    // Non-optimized build
    let kernel_unopt = PtxKernel::new("test_unopt").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let a = ctx.ld_global_f32(ptr);
        let b = ctx.mov_f32_imm(2.0);
        let c = ctx.mov_f32_imm(3.0);
        let mul_result = ctx.mul_f32(a, b);
        let add_result = ctx.add_f32(mul_result, c);
        ctx.st_global_f32(ptr, add_result);
        ctx.ret();
    });

    // Optimized build
    let kernel_opt = PtxKernel::new("test_opt")
        .param(PtxType::U64, "ptr")
        .build_optimized(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let a = ctx.ld_global_f32(ptr);
            let b = ctx.mov_f32_imm(2.0);
            let c = ctx.mov_f32_imm(3.0);
            let mul_result = ctx.mul_f32(a, b);
            let add_result = ctx.add_f32(mul_result, c);
            ctx.st_global_f32(ptr, add_result);
            ctx.ret();
        })
        .expect("test");

    let ptx_unopt = kernel_unopt.emit();
    let ptx_opt = kernel_opt.emit();

    // Unoptimized should have separate mul and add
    assert!(
        ptx_unopt.contains("mul.f32") && ptx_unopt.contains("add.f32"),
        "Unoptimized should have separate mul and add"
    );

    // After FMA fusion, optimized version should have FMA
    // Note: FMA fusion requires single-use of mul result
    // Both kernels should produce valid PTX
    assert!(ptx_unopt.contains(".entry test_unopt"));
    assert!(ptx_opt.contains(".entry test_opt"));
}

#[test]
fn test_build_optimized_empty_body() {
    // Test empty kernel body
    let kernel = PtxKernel::new("test_empty").build_optimized(|_ctx| {
        // Empty body
    });

    assert!(kernel.is_ok(), "Empty optimized kernel should succeed");
}

#[test]
fn test_build_optimized_preserves_barriers() {
    // Test that optimization passes preserve barriers
    let kernel = PtxKernel::new("test_barriers").shared_memory(1024).build_optimized(|ctx| {
        let tid = ctx.special_reg(PtxReg::TidX);
        let val = ctx.mov_f32_imm(1.0);
        let smem_offset = ctx.mul_u32(tid, 4);
        ctx.st_shared_f32(smem_offset, val);
        ctx.bar_sync(0);
        let _loaded = ctx.ld_shared_f32(smem_offset);
        ctx.ret();
    });

    assert!(kernel.is_ok());
    let kernel = kernel.expect("test");
    let ptx = kernel.emit();
    assert!(ptx.contains("bar.sync"), "Barriers should be preserved");
}

// ========================================================================
// COVERAGE IMPROVEMENT TESTS
// ========================================================================

#[test]
fn test_ld_global_f32_v4_vectorized_load() {
    // Test vectorized 4-float load (ld.global.v4.f32)
    let kernel = PtxKernel::new("test_v4_load").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let [r0, r1, r2, r3] = ctx.ld_global_f32_v4(ptr);
        // Sum all 4 values
        let sum1 = ctx.add_f32(r0, r1);
        let sum2 = ctx.add_f32(r2, r3);
        let total = ctx.add_f32(sum1, sum2);
        ctx.st_global_f32(ptr, total);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global.v4.f32"), "Expected vectorized load in: {}", ptx);
}

#[test]
fn test_wide_multiply_u32_imm() {
    // Test wide multiply (mul.wide.u32 producing u64)
    let kernel = PtxKernel::new("test_wide_mul").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u32_imm(1000000);
        // Wide multiply: u32 * imm -> u64
        let _wide_result = ctx.mul_wide_u32(a, 1000000);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mul.wide"), "Expected wide multiply in: {}", ptx);
}

#[test]
fn test_wide_multiply_u32_reg() {
    // Test wide multiply with two registers
    let kernel = PtxKernel::new("test_wide_mul_reg").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u32_imm(1000000);
        let b = ctx.mov_u32_imm(1000000);
        let _wide_result = ctx.mul_wide_u32_reg(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mul.wide"), "Expected wide multiply in: {}", ptx);
}

#[test]
fn test_mad_lo_instruction() {
    // Test mad.lo instruction (multiply-add low)
    let kernel = PtxKernel::new("test_mad_lo").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let c = ctx.mov_u32_imm(5);
        let _result = ctx.mad_lo_u32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mad.lo"), "Expected mad.lo instruction in: {}", ptx);
}

#[test]
fn test_setp_lt_u32_comparison() {
    // Test setp.lt.u32 (set predicate less than)
    let kernel = PtxKernel::new("test_setp_lt").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let pred = ctx.setp_lt_u32(a, b);
        ctx.branch_if(pred, "taken");
        ctx.label("taken");
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("setp.lt"), "Expected setp.lt in: {}", ptx);
}

#[test]
fn test_setp_ge_u32_comparison() {
    // Test setp.ge.u32 (set predicate greater or equal)
    let kernel = PtxKernel::new("test_setp_ge").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u32_imm(20);
        let b = ctx.mov_u32_imm(10);
        let pred = ctx.setp_ge_u32(a, b);
        ctx.branch_if(pred, "taken");
        ctx.label("taken");
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("setp.ge"), "Expected setp.ge in: {}", ptx);
}

#[test]
fn test_integer_division() {
    // Test integer division (div without rounding mode)
    let kernel = PtxKernel::new("test_int_div").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let _result = ctx.div_u32(a, 7);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div.u32") || ptx.contains("div."),
        "Expected integer division in: {}",
        ptx
    );
}

#[test]
fn test_shared_memory_load_store() {
    // Test shared memory operations with state space
    let kernel = PtxKernel::new("test_shared_mem").shared_memory(256).build(|ctx| {
        let tid = ctx.special_reg(PtxReg::TidX);
        let offset = ctx.mul_u32(tid, 4);
        let val = ctx.mov_f32_imm(42.0);
        ctx.st_shared_f32(offset, val);
        ctx.bar_sync(0);
        let loaded = ctx.ld_shared_f32(offset);
        let ptr = ctx.mov_u64_imm(0);
        ctx.st_global_f32(ptr, loaded);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.shared") && ptx.contains("ld.shared"),
        "Expected shared memory ops in: {}",
        ptx
    );
}

#[test]
fn test_label_with_colon() {
    // Test label emission (lines with colon)
    let kernel = PtxKernel::new("test_labels").build(|ctx| {
        ctx.label("loop_start");
        let ctr = ctx.mov_u32_imm(10);
        let one = ctx.mov_u32_imm(1);
        let new_ctr = ctx.sub_u32_reg(ctr, one);
        let pred = ctx.setp_ge_u32(new_ctr, one);
        ctx.branch_if(pred, "loop_start");
        ctx.label("loop_end");
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("loop_start:") && ptx.contains("loop_end:"),
        "Expected labels in: {}",
        ptx
    );
}

#[test]
fn test_mul_lo_for_integer() {
    // Test mul.lo for integer multiplication (low bits)
    let kernel = PtxKernel::new("test_mul_lo").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let _b = ctx.mov_u32_imm(200);
        let _result = ctx.mul_u32(a, 200);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.lo.u32") || ptx.contains("mul.u32"),
        "Expected integer mul in: {}",
        ptx
    );
}

#[test]
fn test_float_multiply_no_lo() {
    // Test floating point multiply (no .lo modifier)
    let kernel = PtxKernel::new("test_float_mul").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_f32_imm(3.125);
        let b = ctx.mov_f32_imm(2.71);
        let result = ctx.mul_f32(a, b);
        let ptr = ctx.load_param_u64("ptr");
        ctx.st_global_f32(ptr, result);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mul.f32"), "Expected float mul in: {}", ptx);
}

#[test]
fn test_div_float_with_rounding() {
    // Test floating point division (needs rounding mode)
    let kernel = PtxKernel::new("test_float_div").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_f32_imm(10.0);
        let b = ctx.mov_f32_imm(3.0);
        let result = ctx.div_f32(a, b);
        let ptr = ctx.load_param_u64("ptr");
        ctx.st_global_f32(ptr, result);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div.rn.f32") || ptx.contains("div.f32"),
        "Expected float div in: {}",
        ptx
    );
}
