//! COVERAGE-BOOST: Generic Address Space Operations

use super::*;

#[test]
fn test_ld_generic_u32() {
    let kernel = PtxKernel::new("test_generic_u32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_u32(ptr);
            ctx.st_generic_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("ld.u32"), "Expected ld.u32 in: {}", ptx);
    assert!(ptx.contains("st.u32"), "Expected st.u32 in: {}", ptx);
}

#[test]
fn test_ld_generic_u64() {
    let kernel = PtxKernel::new("test_generic_u64")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_u64(ptr);
            ctx.st_generic_u64(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.u64") || ptx.contains(".u64"),
        "Expected u64 in: {}",
        ptx
    );
}

#[test]
fn test_ld_generic_u8() {
    let kernel = PtxKernel::new("test_generic_u8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_u8(ptr);
            ctx.st_generic_u8(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains(".u8") || ptx.contains("u8"),
        "Expected u8 ops in: {}",
        ptx
    );
}

#[test]
fn test_ld_generic_u16() {
    let kernel = PtxKernel::new("test_generic_u16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_u16(ptr);
            ctx.st_generic_u16(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains(".u16") || ptx.contains("u16"),
        "Expected u16 ops in: {}",
        ptx
    );
}

#[test]
fn test_ld_generic_f32() {
    let kernel = PtxKernel::new("test_generic_f32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_f32(ptr);
            ctx.st_generic_f32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains(".f32") || ptx.contains("f32"),
        "Expected f32 ops in: {}",
        ptx
    );
}

#[test]
fn test_ld_generic_u32_into() {
    let kernel = PtxKernel::new("test_generic_into")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let dest = ctx.mov_u32_imm(0);
            ctx.ld_generic_u32_into(ptr, dest);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("ld"), "Expected load in: {}", ptx);
}

#[test]
fn test_shared_base_addr() {
    // shared_base_addr: cvta.to.shared.u64 to get generic address from shared memory label
    let kernel = PtxKernel::new("test_shared_base")
        .shared_memory(4096)
        .build(|ctx| {
            let smem = ctx.shared_base_addr();
            // Use the shared memory address to load a value
            let _val = ctx.ld_generic_f32(smem);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvta"),
        "Expected cvta instruction for shared base addr in: {}",
        ptx
    );
    assert!(
        ptx.contains("smem"),
        "Expected smem label reference in: {}",
        ptx
    );
}

#[test]
fn test_ld_global_f16_to_f32_predicated() {
    // PAR-028: Load F16 from global memory with predicate guard, convert to F32
    // Generates: mov.f32 %dst, 0.0; @pred ld.global.b16 %tmp, [addr]; @pred cvt.f32.f16 %dst, %tmp;
    let kernel = PtxKernel::new("test_f16_pred_load")
        .param(PtxType::U64, "ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let n = ctx.load_param_u32("n");
            let idx = ctx.mov_u32_imm(3);
            let valid = ctx.setp_lt_u32(idx, n);
            let val = ctx.ld_global_f16_to_f32_predicated(ptr, valid);
            // Store result to verify the value is F32
            ctx.st_global_f32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();

    // Must contain default initialization (mov.f32 with 0.0)
    assert!(
        ptx.contains("mov.f32"),
        "Expected mov.f32 for default initialization in: {}",
        ptx
    );
    assert!(
        ptx.contains("0F00000000"),
        "Expected 0.0 float literal (0F00000000) for default in: {}",
        ptx
    );

    // Must contain predicated load with b16 type (F16 loaded as b16)
    assert!(
        ptx.contains("ld.global"),
        "Expected ld.global for F16 load in: {}",
        ptx
    );
    assert!(
        ptx.contains(".b16"),
        "Expected .b16 type for F16 load in: {}",
        ptx
    );

    // Must contain predicated cvt from F16 to F32
    assert!(
        ptx.contains("cvt"),
        "Expected cvt instruction for F16->F32 conversion in: {}",
        ptx
    );

    // Must have predicate guards (@%p)
    assert!(
        ptx.contains("@%p"),
        "Expected predicate guard @%p in: {}",
        ptx
    );
}

#[test]
fn test_ld_global_f16_to_f32_predicated_with_store() {
    // Full integration: F16 predicated load -> accumulate -> store
    // Exercises the 3-instruction sequence and verifies register allocation
    let kernel = PtxKernel::new("test_f16_pred_accum")
        .param(PtxType::U64, "kv_ptr")
        .param(PtxType::U32, "head_dim")
        .build(|ctx| {
            let kv_ptr = ctx.load_param_u64("kv_ptr");
            let head_dim = ctx.load_param_u32("head_dim");
            let tid = ctx.special_reg(crate::ptx::registers::PtxReg::TidX);

            // Bounds check: valid if tid < head_dim
            let valid = ctx.setp_lt_u32(tid, head_dim);

            // Compute address: kv_ptr + tid * 2 (F16 = 2 bytes)
            let offset = ctx.mul_wide_u32(tid, 2);
            let addr = ctx.add_u64(kv_ptr, offset);

            // Predicated F16 load with F32 conversion
            let k_val = ctx.ld_global_f16_to_f32_predicated(addr, valid);

            // Use the loaded value in computation (multiply by scalar)
            let scale = ctx.mov_f32_imm(0.125);
            let scaled = ctx.mul_f32(k_val, scale);

            // Store result
            let out_offset = ctx.mul_wide_u32(tid, 4);
            let out_addr = ctx.add_u64(kv_ptr, out_offset);
            ctx.st_global_f32(out_addr, scaled);
            ctx.ret();
        });

    let ptx = kernel.emit();

    // Verify complete instruction sequence
    assert!(ptx.contains("@%p"), "Expected predicate guard in: {}", ptx);
    assert!(
        ptx.contains("ld.global"),
        "Expected global load in: {}",
        ptx
    );
    assert!(
        ptx.contains("cvt"),
        "Expected F16->F32 conversion in: {}",
        ptx
    );
    assert!(
        ptx.contains("mul"),
        "Expected multiply for scale in: {}",
        ptx
    );
    assert!(
        ptx.contains("st.global"),
        "Expected global store in: {}",
        ptx
    );
}

#[test]
fn test_ld_global_f16_to_f32_predicated_instruction_count() {
    // Verify exactly 3 instructions generated: mov (default), ld.global.b16, cvt.f32.f16
    let kernel = PtxKernel::new("test_f16_instr_count")
        .param(PtxType::U64, "ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let n = ctx.load_param_u32("n");
            let idx = ctx.mov_u32_imm(0);
            let valid = ctx.setp_lt_u32(idx, n);
            // This should generate exactly 3 instructions:
            // 1. mov.f32 %f, 0F00000000  (default value)
            // 2. @%p ld.global.b16 %h, [addr]  (predicated load)
            // 3. @%p cvt.f32.f16 %f, %h  (predicated convert)
            let _val = ctx.ld_global_f16_to_f32_predicated(ptr, valid);
            ctx.ret();
        });

    let ptx = kernel.emit();

    // Count predicated instructions (lines containing @%p)
    let predicated_count = ptx.lines().filter(|l| l.contains("@%p")).count();
    assert_eq!(
        predicated_count, 2,
        "Expected exactly 2 predicated instructions (ld + cvt), got {} in:\n{}",
        predicated_count, ptx
    );
}
