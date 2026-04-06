//! Warp shuffle, F16 shared store, special instructions (ex2, rsqrt, rem),
//! negated predicates, type conversions, bitwise/shift ops, inplace ops,
//! register moves, f64 literals, emit_operand variants, max/min,
//! shared memory bytes, address size, and signed wide multiply.

use super::*;

#[test]
fn test_st_shared_f16_instruction() {
    let kernel = PtxKernel::new("test_st_shared_f16").shared_memory(256).build(|ctx| {
        let addr = ctx.shared_base_addr();
        let val = ctx.mov_f32_imm(1.0);
        let f16_val = ctx.cvt_f16_f32(val);
        ctx.st_shared_f16(addr, f16_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("st.shared"), "Expected st.shared in: {}", ptx);
    assert!(ptx.contains(".b16"), "Expected .b16 type in: {}", ptx);
}

#[test]
fn test_shfl_down_f32_warp_shuffle() {
    let kernel = PtxKernel::new("test_shfl_down").build(|ctx| {
        let val = ctx.mov_f32_imm(1.0);
        let shuffled = ctx.shfl_down_f32(val, 16, 0xFFFFFFFF);
        let _sum = ctx.add_f32(val, shuffled);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shfl.sync.down.b32"), "Expected shfl.sync.down.b32 in: {}", ptx);
}

#[test]
fn test_shfl_idx_f32_warp_broadcast() {
    let kernel = PtxKernel::new("test_shfl_idx").build(|ctx| {
        let val = ctx.mov_f32_imm(1.0);
        let _broadcast = ctx.shfl_idx_f32(val, 0, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shfl.sync.idx.b32"), "Expected shfl.sync.idx.b32 in: {}", ptx);
}

#[test]
fn test_min_u32_instruction() {
    let kernel = PtxKernel::new("test_min_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let b = ctx.mov_u32_imm(50);
        let _min = ctx.min_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("min.u32"), "Expected min.u32 in: {}", ptx);
}

#[test]
fn test_ex2_f32_exponential() {
    let kernel = PtxKernel::new("test_ex2").build(|ctx| {
        let val = ctx.mov_f32_imm(2.0);
        let _exp = ctx.ex2_f32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("ex2.approx"), "Expected ex2.approx in: {}", ptx);
}

#[test]
fn test_rsqrt_f32_instruction() {
    let kernel = PtxKernel::new("test_rsqrt").build(|ctx| {
        let val = ctx.mov_f32_imm(4.0);
        let _rsqrt = ctx.rsqrt_f32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("rsqrt.approx"), "Expected rsqrt.approx in: {}", ptx);
}

#[test]
fn test_rem_u32_remainder() {
    let kernel = PtxKernel::new("test_rem").build(|ctx| {
        let val = ctx.mov_u32_imm(100);
        let _rem = ctx.rem_u32(val, 32);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("rem.u32"), "Expected rem.u32 in: {}", ptx);
}

#[test]
fn test_branch_if_not_negated_predicate() {
    let kernel = PtxKernel::new("test_branch_if_not").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let pred = ctx.setp_lt_u32(a, b);
        ctx.branch_if_not(pred, "skip");
        ctx.label("skip");
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("@!"), "Expected negated predicate @! in: {}", ptx);
    assert!(ptx.contains("bra skip"), "Expected bra skip in: {}", ptx);
}

#[test]
fn test_cvt_u32_u8_conversion() {
    let kernel = PtxKernel::new("test_cvt_u32_u8").param(PtxType::U64, "src").build(|ctx| {
        let addr = ctx.load_param_u64("src");
        let byte_val = ctx.ld_global_u8(addr);
        let _u32_val = ctx.cvt_u32_u8(byte_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("cvt.u32"), "Expected cvt.u32 conversion in: {}", ptx);
}

#[test]
fn test_shr_u32_shift_right() {
    let kernel = PtxKernel::new("test_shr_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(256);
        let shift = ctx.mov_u32_imm(4);
        let _shifted = ctx.shr_u32(val, shift);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shr.b32"), "Expected shr.b32 in: {}", ptx);
}

#[test]
fn test_and_u32_bitwise() {
    let kernel = PtxKernel::new("test_and_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(0xFF00);
        let b = ctx.mov_u32_imm(0x0FF0);
        let _result = ctx.and_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("and.b32"), "Expected and.b32 in: {}", ptx);
}

#[test]
fn test_or_u32_bitwise() {
    let kernel = PtxKernel::new("test_or_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(0xFF00);
        let b = ctx.mov_u32_imm(0x00FF);
        let _result = ctx.or_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("or.b32"), "Expected or.b32 in: {}", ptx);
}

#[test]
fn test_shl_u32_shift_left() {
    let kernel = PtxKernel::new("test_shl_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(1);
        let shift = ctx.mov_u32_imm(8);
        let _shifted = ctx.shl_u32(val, shift);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shl.b32"), "Expected shl.b32 in: {}", ptx);
}

#[test]
fn test_shr_u32_inplace_shift() {
    let kernel = PtxKernel::new("test_shr_inplace").build(|ctx| {
        let val = ctx.mov_u32_imm(256);
        ctx.shr_u32_inplace(val, 1);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("shr.b32"), "Expected shr.b32 in: {}", ptx);
}

#[test]
fn test_max_f32_inplace_operation() {
    let kernel = PtxKernel::new("test_max_inplace").build(|ctx| {
        let running_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
        let new_val = ctx.mov_f32_imm(5.0);
        ctx.max_f32_inplace(running_max, new_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("max.f32"), "Expected max.f32 in: {}", ptx);
}

#[test]
fn test_mov_f32_reg_copy() {
    let kernel = PtxKernel::new("test_mov_f32_reg").build(|ctx| {
        let src = ctx.mov_f32_imm(1.5);
        let dst = ctx.mov_f32_imm(0.0);
        ctx.mov_f32_reg(dst, src);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mov.f32"), "Expected mov.f32 in: {}", ptx);
}

#[test]
fn test_mul_f32_inplace_scaling() {
    let kernel = PtxKernel::new("test_mul_inplace").build(|ctx| {
        let val = ctx.mov_f32_imm(2.0);
        let scale = ctx.mov_f32_imm(0.5);
        ctx.mul_f32_inplace(val, scale);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("mul.f32"), "Expected mul.f32 in: {}", ptx);
}

#[test]
fn test_f64_literal_format() {
    let kernel = PtxKernel::new("test_f64").build(|ctx| {
        let _f32_val = ctx.mov_f32_imm(std::f64::consts::PI as f32);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("0F"), "Expected hex float literal 0F prefix in: {}", ptx);
}

#[test]
fn test_emit_operand_addr_with_offset() {
    use crate::ptx::instructions::Operand;
    use crate::ptx::registers::VirtualReg;

    let vreg = VirtualReg::new(0, PtxType::U64);
    let addr_op = Operand::Addr { base: vreg, offset: 128 };
    let result = emit_operand(&addr_op);
    assert!(result.contains("+128"), "Expected offset +128 in: {}", result);
}

#[test]
fn test_emit_shared_mem_operand_with_offset() {
    use crate::ptx::instructions::Operand;
    use crate::ptx::registers::VirtualReg;

    let vreg = VirtualReg::new(0, PtxType::U64);
    let addr_op = Operand::Addr { base: vreg, offset: 64 };
    let result = emit_shared_mem_operand(&addr_op);
    assert!(result.contains("+64"), "Expected offset +64 in: {}", result);
}

#[test]
fn test_emit_global_mem_operand_with_offset() {
    use crate::ptx::instructions::Operand;
    use crate::ptx::registers::VirtualReg;

    let vreg = VirtualReg::new(0, PtxType::U64);
    let addr_op = Operand::Addr { base: vreg, offset: 256 };
    let result = emit_global_mem_operand(&addr_op);
    assert!(result.contains("+256"), "Expected offset +256 in: {}", result);
}

#[test]
fn test_max_f32_non_inplace() {
    let kernel = PtxKernel::new("test_max_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(3.0);
        let b = ctx.mov_f32_imm(5.0);
        let _max = ctx.max_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("max.f32"), "Expected max.f32 in: {}", ptx);
}

#[test]
fn test_kernel_get_shared_memory_bytes() {
    let kernel = PtxKernel::new("test_smem").shared_memory(4096);
    assert_eq!(kernel.shared_memory_bytes(), 4096);
}

#[test]
fn test_module_get_address_size() {
    let module = PtxModule::new().address_size(32);
    assert_eq!(module.get_address_size(), 32);
}

#[test]
fn test_signed_wide_multiply() {
    use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};

    let vreg = VirtualReg::new(0, PtxType::S32);
    let instr = PtxInstruction::new(PtxOp::Mul, PtxType::S64)
        .dst(Operand::Reg(vreg))
        .src(Operand::Reg(vreg))
        .src(Operand::ImmI64(100));

    let ptx = emit_instruction(&instr);
    assert!(ptx.contains("mul.wide.s32"), "Expected mul.wide.s32 in: {}", ptx);
}
