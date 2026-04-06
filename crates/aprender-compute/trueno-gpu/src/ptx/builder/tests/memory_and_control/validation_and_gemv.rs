//! Module validation tests, FMA/global loads, and coalesced GEMV tests
//! (Decoder Throughput Spec 5.3).

use super::*;

// ========================================================================
// COVERAGE IMPROVEMENT TESTS - TRUENO-SPEC-014
// ========================================================================

#[test]
fn test_validate_valid_module() {
    // Test that a valid module passes validation
    let module = PtxModule::new().version(8, 0).target("sm_70").address_size(64);

    assert!(module.validate().is_ok());
}

#[test]
fn test_validate_minimum_version() {
    // PTX version 7.0 should be valid (minimum supported)
    let module = PtxModule::new().version(7, 0).target("sm_70");
    assert!(module.validate().is_ok());
}

#[test]
fn test_validate_invalid_version() {
    // PTX version below 7.0 should fail
    let module = PtxModule::new().version(6, 5).target("sm_70");
    assert!(module.validate().is_err());
}

#[test]
fn test_validate_invalid_target() {
    // Invalid compute capability should fail
    let module = PtxModule::new().version(8, 0).target("sm_invalid");
    assert!(module.validate().is_err());
}

#[test]
fn test_validate_sm_30_too_old() {
    // sm_30 is too old (below sm_50)
    let module = PtxModule::new().version(8, 0).target("sm_30");
    assert!(module.validate().is_err());
}

#[test]
fn test_ptx_module_default() {
    let module = PtxModule::default();
    // Default version is (8, 0)
    assert_eq!(module.version, (8, 0));
}

#[test]
fn test_fma_f32() {
    let kernel = PtxKernel::new("test_fma").build(|ctx| {
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        let c = ctx.mov_f32_imm(4.0);
        let _result = ctx.fma_f32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("fma"), "Expected fma instruction, got: {}", ptx);
}

#[test]
fn test_ld_global_u32() {
    let kernel = PtxKernel::new("test_ld_u32").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u32(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global"), "Expected ld.global instruction, got: {}", ptx);
}

#[test]
fn test_ld_global_u64() {
    // PAR-118: Test u64 load for pointer arrays in batched attention
    let kernel = PtxKernel::new("test_ld_u64").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u64(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global.u64"), "Expected ld.global.u64 instruction, got: {}", ptx);
}

#[test]
fn test_ld_global_u8() {
    let kernel = PtxKernel::new("test_ld_u8").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u8(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global"), "Expected ld.global instruction, got: {}", ptx);
}

#[test]
fn test_ld_global_u16() {
    let kernel = PtxKernel::new("test_ld_u16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u16(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global"), "Expected ld.global instruction, got: {}", ptx);
}

// ========================================================================
// COALESCED GEMV TESTS - DECODER THROUGHPUT SPEC 5.3
// ========================================================================

#[test]
fn test_mul_lo_u32() {
    // mul_lo_u32: u32 * u32 -> u32 (low bits only)
    let kernel = PtxKernel::new("test_mul_lo").build(|ctx| {
        let a = ctx.mov_u32_imm(256);
        let b = ctx.mov_u32_imm(16);
        let _result = ctx.mul_lo_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mul.lo.u32"), "Expected 'mul.lo.u32' instruction, got: {}", ptx);
}

#[test]
fn test_shared_base_addr() {
    // shared_base_addr: Get pointer to 'smem' shared memory array
    let kernel = PtxKernel::new("test_smem_addr").shared_memory(1024).build(|ctx| {
        let smem_ptr = ctx.shared_base_addr();
        let val = ctx.mov_f32_imm(1.0);
        ctx.st_shared_f32(smem_ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Must declare shared memory
    assert!(ptx.contains(".shared"), "Expected shared memory declaration, got: {}", ptx);
    // Must reference 'smem' label
    assert!(ptx.contains("smem"), "Expected 'smem' reference, got: {}", ptx);
}

#[test]
fn test_ld_global_f32_predicated() {
    // ld_global_f32_predicated: Load with predicate guard and default value
    let kernel = PtxKernel::new("test_pred_load")
        .param(PtxType::U64, "ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let n = ctx.load_param_u32("n");
            let idx = ctx.mov_u32_imm(5);
            let valid = ctx.setp_lt_u32(idx, n);
            let _val = ctx.ld_global_f32_predicated(ptr, valid, 0.0);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Must contain predicated load (@p ld.global.f32)
    assert!(
        ptx.contains("@%p") && ptx.contains("ld.global.f32"),
        "Expected predicated ld.global.f32, got: {}",
        ptx
    );
    // Must initialize with default value first
    assert!(
        ptx.contains("mov.f32") && ptx.contains("0F00000000"),
        "Expected mov.f32 with 0.0 default, got: {}",
        ptx
    );
}

#[test]
fn test_coalesced_gemv_kernel_structure() {
    // Integration test: verify CoalescedGemv kernel PTX structure
    let kernel = PtxKernel::new("gemv_coalesced")
        .param(PtxType::U64, "y_ptr")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "x_ptr")
        .param(PtxType::U32, "k_dim")
        .param(PtxType::U32, "n_dim")
        .shared_memory(4096 * 4) // Cache x vector
        .build(|ctx| {
            // Minimal structure test
            let block_id = ctx.special_reg(PtxReg::CtaIdX);
            let thread_id = ctx.special_reg(PtxReg::TidX);
            let block_size = ctx.mov_u32_imm(256);
            let col_base = ctx.mul_lo_u32(block_id, block_size);
            let col = ctx.add_u32_reg(col_base, thread_id);

            let n_dim = ctx.load_param_u32("n_dim");
            let oob = ctx.setp_ge_u32(col, n_dim);
            ctx.branch_if(oob, "exit");

            let sum = ctx.mov_f32_imm(0.0);
            let smem = ctx.shared_base_addr();

            // Load x into shared memory with predicate
            let x_ptr = ctx.load_param_u64("x_ptr");
            let k_dim = ctx.load_param_u32("k_dim");
            let valid = ctx.setp_lt_u32(thread_id, k_dim);
            let x_offset = ctx.mul_wide_u32(thread_id, 4);
            let x_addr = ctx.add_u64(x_ptr, x_offset);
            let x_val = ctx.ld_global_f32_predicated(x_addr, valid, 0.0);

            let smem_offset = ctx.mul_u32(thread_id, 4);
            let smem_offset_64 = ctx.cvt_u64_u32(smem_offset);
            let smem_addr = ctx.add_u64(smem, smem_offset_64);
            ctx.st_shared_f32(smem_addr, x_val);
            ctx.bar_sync(0);

            // Store result
            let y_ptr = ctx.load_param_u64("y_ptr");
            let y_offset = ctx.mul_wide_u32(col, 4);
            let y_addr = ctx.add_u64(y_ptr, y_offset);
            ctx.st_global_f32(y_addr, sum);

            ctx.label("exit");
            ctx.ret();
        });

    let ptx = kernel.emit();

    // Verify all critical components
    assert!(ptx.contains(".entry gemv_coalesced"), "Missing entry point");
    assert!(ptx.contains(".shared"), "Missing shared memory");
    assert!(ptx.contains("bar.sync"), "Missing barrier");
    assert!(ptx.contains("mul.lo.u32"), "Missing mul.lo.u32");
    assert!(ptx.contains("@%p"), "Missing predicated instruction");
}
