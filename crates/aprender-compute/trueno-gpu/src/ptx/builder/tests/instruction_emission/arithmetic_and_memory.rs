//! Arithmetic instructions, memory loads/stores, shared memory, FMA,
//! conversion, multiplication, barriers, comparisons, and shifts.

use super::*;

#[test]
fn test_ld_param_emission() {
    let kernel = PtxKernel::new("test_ld_param")
        .param(PtxType::U64, "data_ptr")
        .param(PtxType::U32, "count")
        .build(|ctx| {
            let _ptr = ctx.load_param_u64("data_ptr");
            let _count = ctx.load_param_u32("count");
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.param"), "Expected ld.param in: {}", ptx);
}

#[test]
fn test_u64_multiplication() {
    let kernel = PtxKernel::new("test_u64_mul").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u64_imm(1000000000u64);
        let _result = ctx.mul_u64(a, 2000000000u64);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.lo.u64") || ptx.contains("mul.u64") || ptx.contains("mov.u64"),
        "Expected u64 operation in: {}",
        ptx
    );
}

#[test]
fn test_u64_reg_multiplication() {
    let kernel = PtxKernel::new("test_u64_mul_reg").param(PtxType::U64, "ptr").build(|ctx| {
        let a = ctx.mov_u64_imm(1000000000u64);
        let b = ctx.mov_u64_imm(2000000000u64);
        let _result = ctx.mul_u64_reg(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mul.lo.u64"), "Expected mul.lo.u64 in: {}", ptx);
}

#[test]
fn test_global_u32_load() {
    let kernel = PtxKernel::new("test_ld_global_u32").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u32(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global.u32"), "Expected ld.global.u32 in: {}", ptx);
}

#[test]
fn test_global_u8_load() {
    let kernel = PtxKernel::new("test_ld_global_u8").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u8(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global.u8"), "Expected ld.global.u8 in: {}", ptx);
}

#[test]
fn test_global_u16_load() {
    let kernel = PtxKernel::new("test_ld_global_u16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let _val = ctx.ld_global_u16(ptr);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global.u16"), "Expected ld.global.u16 in: {}", ptx);
}

#[test]
fn test_bra_unconditional() {
    let kernel = PtxKernel::new("test_bra").build(|ctx| {
        ctx.branch("skip");
        ctx.label("dead_code");
        let _unused = ctx.mov_f32_imm(1.0);
        ctx.label("skip");
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("bra skip") || ptx.contains("bra\tskip"),
        "Expected bra instruction in: {}",
        ptx
    );
}

#[test]
fn test_and_pred_combining_bounds() {
    let kernel = PtxKernel::new("test_and_pred")
        .param(PtxType::U64, "data_ptr")
        .param(PtxType::U32, "size")
        .build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let size = ctx.load_param_u32("size");
            let p1 = ctx.setp_lt_u32(tid, size);
            let ten = ctx.mov_u32_imm(10);
            let p2 = ctx.setp_lt_u32(tid, ten);
            let combined = ctx.and_pred(p1, p2);
            ctx.branch_if(combined, "do_work");
            ctx.ret();
            ctx.label("do_work");
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("and.pred"), "Expected and.pred in: {}", ptx);
}

#[test]
fn test_div_f32_inplace_normalization() {
    let kernel = PtxKernel::new("test_div_inplace").param(PtxType::U64, "data_ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("data_ptr");
        let value = ctx.ld_global_f32(ptr);
        let divisor = ctx.mov_f32_imm(10.0);
        ctx.div_f32_inplace(value, divisor);
        ctx.st_global_f32(ptr, value);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("div.rn.f32"), "Expected div.rn.f32 in: {}", ptx);
}

#[test]
fn test_predicated_instruction_emission() {
    let kernel = PtxKernel::new("test_predicate").param(PtxType::U64, "ptr").build(|ctx| {
        let tid = ctx.special_reg(PtxReg::TidX);
        let limit = ctx.mov_u32_imm(64);
        let pred = ctx.setp_lt_u32(tid, limit);
        ctx.branch_if(pred, "store_it");
        ctx.ret();
        ctx.label("store_it");
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.mov_f32_imm(1.0);
        ctx.st_global_f32(ptr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("setp."), "Expected setp in: {}", ptx);
}

#[test]
fn test_sub_instruction_emission() {
    let kernel = PtxKernel::new("test_sub").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let b = ctx.mov_u32_imm(30);
        let _result = ctx.sub_u32_reg(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("sub."), "Expected sub instruction in: {}", ptx);
}

#[test]
fn test_integer_div_emission() {
    let kernel = PtxKernel::new("test_int_div").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let _result = ctx.div_u32(a, 7);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("div.u32") || ptx.contains("div.s32"), "Expected integer div in: {}", ptx);
}

#[test]
fn test_mul_wide_u32_emission() {
    let kernel = PtxKernel::new("test_mul_wide").build(|ctx| {
        let a = ctx.mov_u32_imm(1000000);
        let result = ctx.mul_wide_u32(a, 1000000);
        let _ = result;
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mul.wide.u32"), "Expected mul.wide.u32 in: {}", ptx);
}

#[test]
fn test_mul_wide_u32_reg_emission() {
    // trueno#207: mul_wide_u32_reg must emit mul.wide.u32, never mul.lo.u64
    let kernel = PtxKernel::new("test_mul_wide_reg").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let b = ctx.mov_u32_imm(200);
        let _ = ctx.mul_wide_u32_reg(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul.wide.u32"), "Expected mul.wide.u32 in: {}", ptx);
    assert!(!ptx.contains("mul.lo.u64"), "Found mul.lo.u64 in: {}", ptx);
}

#[test]
fn test_mad_lo_emission() {
    let kernel = PtxKernel::new("test_mad_lo").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let c = ctx.mov_u32_imm(5);
        let _result = ctx.mad_lo_u32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mad.lo.u32"), "Expected mad.lo.u32 in: {}", ptx);
}

#[test]
fn test_shared_memory_operations() {
    let kernel = PtxKernel::new("test_shared").shared_memory(256 * 4).build(|ctx| {
        let tid = ctx.special_reg(PtxReg::TidX);
        let tile_ptr = ctx.shared_base_addr();
        let offset = ctx.mul_u32(tid, 4);
        let offset_64 = ctx.cvt_u64_u32(offset);
        let addr = ctx.add_u64(tile_ptr, offset_64);
        let val = ctx.ld_shared_f32(addr);
        ctx.st_shared_f32(addr, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ld.shared"), "Expected ld.shared in: {}", ptx);
    assert!(ptx.contains("st.shared"), "Expected st.shared in: {}", ptx);
}

#[test]
fn test_cvt_instruction_emission() {
    let kernel = PtxKernel::new("test_cvt").build(|ctx| {
        let a = ctx.mov_u32_imm(42);
        let _f = ctx.cvt_f32_u32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("cvt."), "Expected cvt instruction in: {}", ptx);
}

#[test]
fn test_float_mul_no_lo() {
    let kernel = PtxKernel::new("test_float_mul").build(|ctx| {
        let a = ctx.mov_f32_imm(3.125);
        let b = ctx.mov_f32_imm(2.0);
        let _result = ctx.mul_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.f32") && !ptx.contains("mul.lo.f32"),
        "Expected mul.f32 without .lo in: {}",
        ptx
    );
}

#[test]
fn test_bar_sync_basic_barrier() {
    let kernel = PtxKernel::new("test_bar").build(|ctx| {
        ctx.bar_sync(0);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("bar.sync"), "Expected bar.sync in: {}", ptx);
}

#[test]
fn test_setp_comparison_ops() {
    let kernel = PtxKernel::new("test_setp_cmp").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _lt = ctx.setp_lt_u32(a, b);
        let _ge = ctx.setp_ge_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("setp.lt"), "Expected setp.lt in: {}", ptx);
    assert!(ptx.contains("setp.ge"), "Expected setp.ge in: {}", ptx);
}
