use super::*;
use crate::ptx::registers::RegisterAllocator;

/// Mock builder for testing traits in isolation
struct MockBuilder {
    registers: RegisterAllocator,
    instructions: Vec<PtxInstruction>,
    labels: Vec<String>,
}

impl MockBuilder {
    fn new() -> Self {
        Self { registers: RegisterAllocator::new(), instructions: Vec::new(), labels: Vec::new() }
    }
}

impl KernelBuilderCore for MockBuilder {
    fn registers_mut(&mut self) -> &mut RegisterAllocator {
        &mut self.registers
    }
    fn instructions_mut(&mut self) -> &mut Vec<PtxInstruction> {
        &mut self.instructions
    }
    fn labels_mut(&mut self) -> &mut Vec<String> {
        &mut self.labels
    }
}

#[test]
fn test_add_f32() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::F32);
    let b = builder.registers.allocate_virtual(PtxType::F32);

    let result = builder.add_f32(a, b);

    assert_eq!(builder.instructions.len(), 1);
    assert_eq!(builder.instructions[0].op, PtxOp::Add);
    assert!(result.id() > 0);
}

#[test]
fn test_fma_f32() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::F32);
    let b = builder.registers.allocate_virtual(PtxType::F32);
    let c = builder.registers.allocate_virtual(PtxType::F32);

    let result = builder.fma_f32(a, b, c);

    assert_eq!(builder.instructions.len(), 1);
    assert_eq!(builder.instructions[0].op, PtxOp::Fma);
    assert!(result.id() > 0);
}

#[test]
fn test_dp4a_s32() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::U32);
    let b = builder.registers.allocate_virtual(PtxType::U32);
    let c = builder.registers.allocate_virtual(PtxType::S32);

    let result = builder.dp4a_s32(a, b, c);

    assert_eq!(builder.instructions.len(), 1);
    assert_eq!(builder.instructions[0].op, PtxOp::Dp4a);
    assert!(result.id() > 0);
}

#[test]
fn test_transcendentals() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::F32);

    let _sin = builder.sin_f32(a);
    let _cos = builder.cos_f32(a);
    let _sqrt = builder.sqrt_f32(a);
    let _rsqrt = builder.rsqrt_f32(a);
    let _rcp = builder.rcp_f32(a);
    let _ex2 = builder.ex2_f32(a);
    let _lg2 = builder.lg2_f32(a);

    assert_eq!(builder.instructions.len(), 7);
}

#[test]
fn test_sub_mul_div_f32() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::F32);
    let b = builder.registers.allocate_virtual(PtxType::F32);

    let _sub = builder.sub_f32(a, b);
    let _mul = builder.mul_f32(a, b);
    let _div = builder.div_f32(a, b);

    assert_eq!(builder.instructions.len(), 3);
    assert_eq!(builder.instructions[0].op, PtxOp::Sub);
    assert_eq!(builder.instructions[1].op, PtxOp::Mul);
    assert_eq!(builder.instructions[2].op, PtxOp::Div);
}

#[test]
fn test_neg_abs_f32() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::F32);

    let _neg = builder.neg_f32(a);
    let _abs = builder.abs_f32(a);

    assert_eq!(builder.instructions.len(), 2);
    assert_eq!(builder.instructions[0].op, PtxOp::Neg);
    assert_eq!(builder.instructions[1].op, PtxOp::Abs);
}

#[test]
fn test_min_max_f32() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::F32);
    let b = builder.registers.allocate_virtual(PtxType::F32);

    let _min = builder.min_f32(a, b);
    let _max = builder.max_f32(a, b);

    assert_eq!(builder.instructions.len(), 2);
    assert_eq!(builder.instructions[0].op, PtxOp::Min);
    assert_eq!(builder.instructions[1].op, PtxOp::Max);
}

#[test]
fn test_u32_operations() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::U32);
    let b = builder.registers.allocate_virtual(PtxType::U32);

    let _add = builder.add_u32(a, b);
    let _sub = builder.sub_u32(a, b);
    let _mul = builder.mul_u32(a, b);

    assert_eq!(builder.instructions.len(), 3);
    assert_eq!(builder.instructions[0].ty, PtxType::U32);
}

#[test]
fn test_u64_operations() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::U64);
    let b = builder.registers.allocate_virtual(PtxType::U64);

    let _add = builder.add_u64(a, b);

    assert_eq!(builder.instructions.len(), 1);
    assert_eq!(builder.instructions[0].ty, PtxType::U64);
}

#[test]
fn test_mul_wide_u32() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::U32);

    let _wide = builder.mul_wide_u32(a, 4);

    assert_eq!(builder.instructions.len(), 1);
    assert_eq!(builder.instructions[0].op, PtxOp::MulWide);
    assert_eq!(builder.instructions[0].ty, PtxType::U32);
}

#[test]
fn test_mul_wide_u32_reg() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::U32);
    let b = builder.registers.allocate_virtual(PtxType::U32);

    let _wide = builder.mul_wide_u32_reg(a, b);

    assert_eq!(builder.instructions.len(), 1);
    assert_eq!(builder.instructions[0].op, PtxOp::MulWide);
    assert_eq!(builder.instructions[0].ty, PtxType::U32);
}

#[test]
fn test_mad_lo_u32() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::U32);
    let b = builder.registers.allocate_virtual(PtxType::U32);
    let c = builder.registers.allocate_virtual(PtxType::U32);

    let _mad = builder.mad_lo_u32(a, b, c);

    assert_eq!(builder.instructions.len(), 1);
    assert_eq!(builder.instructions[0].op, PtxOp::MadLo);
}

#[test]
fn test_into_operations() {
    let mut builder = MockBuilder::new();
    let dst_u32 = builder.registers.allocate_virtual(PtxType::U32);
    let dst_u64 = builder.registers.allocate_virtual(PtxType::U64);
    let dst_f32 = builder.registers.allocate_virtual(PtxType::F32);
    let a = builder.registers.allocate_virtual(PtxType::U32);
    let b = builder.registers.allocate_virtual(PtxType::U32);
    let a64 = builder.registers.allocate_virtual(PtxType::U64);
    let b64 = builder.registers.allocate_virtual(PtxType::U64);
    let af = builder.registers.allocate_virtual(PtxType::F32);
    let bf = builder.registers.allocate_virtual(PtxType::F32);
    let cf = builder.registers.allocate_virtual(PtxType::F32);

    builder.add_u32_into(dst_u32, a, b);
    builder.add_u64_into(dst_u64, a64, b64);
    builder.fma_f32_into(dst_f32, af, bf, cf);

    assert_eq!(builder.instructions.len(), 3);
}

#[test]
fn test_dp4a_variants() {
    let mut builder = MockBuilder::new();
    let a = builder.registers.allocate_virtual(PtxType::U32);
    let b = builder.registers.allocate_virtual(PtxType::U32);
    let c_s = builder.registers.allocate_virtual(PtxType::S32);
    let c_u = builder.registers.allocate_virtual(PtxType::U32);

    let _dp4a_s = builder.dp4a_s32(a, b, c_s);
    let _dp4a_u = builder.dp4a_u32(a, b, c_u);

    let dst = builder.registers.allocate_virtual(PtxType::S32);
    builder.dp4a_s32_into(dst, a, b, c_s);

    assert_eq!(builder.instructions.len(), 3);
    assert_eq!(builder.instructions[0].op, PtxOp::Dp4a);
    assert_eq!(builder.instructions[1].op, PtxOp::Dp4a);
    assert_eq!(builder.instructions[2].op, PtxOp::Dp4a);
}
