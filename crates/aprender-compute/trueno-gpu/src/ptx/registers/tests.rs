use super::*;

#[test]
fn test_special_register_strings() {
    assert_eq!(PtxReg::TidX.to_ptx_string(), "%tid.x");
    assert_eq!(PtxReg::CtaIdX.to_ptx_string(), "%ctaid.x");
    assert_eq!(PtxReg::NtidX.to_ptx_string(), "%ntid.x");
    assert_eq!(PtxReg::LaneId.to_ptx_string(), "%laneid");
    assert_eq!(PtxReg::WarpId.to_ptx_string(), "%warpid");
}

#[test]
fn test_special_register_types() {
    assert_eq!(PtxReg::TidX.data_type(), PtxType::U32);
    assert_eq!(PtxReg::Clock64.data_type(), PtxType::U64);
}

#[test]
fn test_virtual_register_creation() {
    let vreg = VirtualReg::new(0, PtxType::F32);
    assert_eq!(vreg.id(), 0);
    assert_eq!(vreg.ty(), PtxType::F32);
}

#[test]
fn test_virtual_register_string() {
    let vreg = VirtualReg::new(5, PtxType::F32);
    assert_eq!(vreg.to_ptx_string(), "%f5");

    let vreg_u32 = VirtualReg::new(3, PtxType::U32);
    assert_eq!(vreg_u32.to_ptx_string(), "%r3");

    let vreg_pred = VirtualReg::new(1, PtxType::Pred);
    assert_eq!(vreg_pred.to_ptx_string(), "%p1");
}

#[test]
fn test_live_range_overlap() {
    let r1 = LiveRange::new(0, 5);
    let r2 = LiveRange::new(3, 8);
    let r3 = LiveRange::new(5, 10);
    let r4 = LiveRange::new(10, 15);

    assert!(r1.overlaps(&r2)); // 3-5 overlap
    assert!(!r1.overlaps(&r3)); // r1 ends at 5, r3 starts at 5
    assert!(!r1.overlaps(&r4));
}

#[test]
fn test_register_allocator() {
    let mut alloc = RegisterAllocator::new();

    // Per-type IDs: each type starts from 0
    let r1 = alloc.allocate_virtual(PtxType::F32);
    let r2 = alloc.allocate_virtual(PtxType::F32);
    let r3 = alloc.allocate_virtual(PtxType::U32);

    assert_eq!(r1.id(), 0); // First F32
    assert_eq!(r2.id(), 1); // Second F32
    assert_eq!(r3.id(), 0); // First U32 (different type, starts at 0)

    // Verify types are correct
    assert_eq!(r1.ty(), PtxType::F32);
    assert_eq!(r3.ty(), PtxType::U32);
}

#[test]
fn test_pressure_report() {
    let mut alloc = RegisterAllocator::new();

    let _ = alloc.allocate_virtual(PtxType::F32);
    let _ = alloc.allocate_virtual(PtxType::F32);
    let _ = alloc.allocate_virtual(PtxType::F32);

    let report = alloc.pressure_report();
    assert_eq!(report.max_live, 3);
    assert_eq!(report.spill_count, 0);
}

#[test]
fn test_emit_declarations() {
    let mut alloc = RegisterAllocator::new();

    let _ = alloc.allocate_virtual(PtxType::F32);
    let _ = alloc.allocate_virtual(PtxType::F32);
    let _ = alloc.allocate_virtual(PtxType::U32);

    let decls = alloc.emit_declarations();
    assert!(decls.contains(".reg .f32"));
    assert!(decls.contains(".reg .u32") || decls.contains(".reg .s32"));
}

/// Regression test: U32 and S32 now use different prefixes (%r vs %ri).
/// Before the fix, both used %r, causing CUDA_ERROR_INVALID_PTX.
#[test]
fn test_u32_s32_separate_prefixes() {
    let mut alloc = RegisterAllocator::new();

    // Allocate U32 registers: %r0, %r1, %r2
    let u0 = alloc.allocate_virtual(PtxType::U32);
    let u1 = alloc.allocate_virtual(PtxType::U32);
    let u2 = alloc.allocate_virtual(PtxType::U32);

    // Allocate S32 registers: %ri0, %ri1 (separate prefix)
    let s0 = alloc.allocate_virtual(PtxType::S32);
    let s1 = alloc.allocate_virtual(PtxType::S32);

    // U32 has its own counter starting at 0
    assert_eq!(u0.id(), 0);
    assert_eq!(u1.id(), 1);
    assert_eq!(u2.id(), 2);

    // S32 has its own counter starting at 0 (different prefix → no conflict)
    assert_eq!(s0.id(), 0);
    assert_eq!(s1.id(), 1);

    // Physical names must use different prefixes
    assert_eq!(u0.to_ptx_string(), "%r0");
    assert_eq!(s0.to_ptx_string(), "%ri0");
    assert_ne!(u0.to_ptx_string(), s0.to_ptx_string());

    // Declarations: separate .u32 %r<3> and .s32 %ri<2>
    let decls = alloc.emit_declarations();
    assert!(decls.contains(".reg .u32  %r<3>"), "Missing u32 decl in:\n{decls}");
    assert!(decls.contains(".reg .s32  %ri<2>"), "Missing s32 decl in:\n{decls}");
}

#[test]
fn test_all_special_registers() {
    // Test all PtxReg variants for 100% coverage
    assert_eq!(PtxReg::TidY.to_ptx_string(), "%tid.y");
    assert_eq!(PtxReg::TidZ.to_ptx_string(), "%tid.z");
    assert_eq!(PtxReg::CtaIdY.to_ptx_string(), "%ctaid.y");
    assert_eq!(PtxReg::CtaIdZ.to_ptx_string(), "%ctaid.z");
    assert_eq!(PtxReg::NtidY.to_ptx_string(), "%ntid.y");
    assert_eq!(PtxReg::NtidZ.to_ptx_string(), "%ntid.z");
    assert_eq!(PtxReg::NctaIdX.to_ptx_string(), "%nctaid.x");
    assert_eq!(PtxReg::NctaIdY.to_ptx_string(), "%nctaid.y");
    assert_eq!(PtxReg::NctaIdZ.to_ptx_string(), "%nctaid.z");
    assert_eq!(PtxReg::SmId.to_ptx_string(), "%smid");
    assert_eq!(PtxReg::Clock.to_ptx_string(), "%clock");
}

#[test]
fn test_extend_live_range() {
    let mut alloc = RegisterAllocator::new();
    let vreg = alloc.allocate_virtual(PtxType::F32);

    // Advance and extend
    alloc.next_instruction();
    alloc.next_instruction();
    alloc.extend_live_range(vreg);

    // The live range should now be [0, 3)
    let report = alloc.pressure_report();
    assert_eq!(report.max_live, 1);
}

#[test]
fn test_next_instruction() {
    let mut alloc = RegisterAllocator::new();

    // Initially at instruction 0
    let _ = alloc.allocate_virtual(PtxType::F32);
    alloc.next_instruction();
    let _ = alloc.allocate_virtual(PtxType::F32);
    alloc.next_instruction();
    let _ = alloc.allocate_virtual(PtxType::F32);

    // Three registers allocated
    assert_eq!(alloc.pressure_report().max_live, 3);
}

#[test]
fn test_virtual_register_display() {
    let vreg = VirtualReg::new(7, PtxType::F64);
    let display_str = format!("{}", vreg);
    assert_eq!(display_str, "%fd7");

    let vreg_u64 = VirtualReg::new(2, PtxType::U64);
    let display_str_u64 = format!("{}", vreg_u64);
    assert_eq!(display_str_u64, "%rd2");
}

#[test]
fn test_virtual_register_write_to() {
    let vreg = VirtualReg::new(3, PtxType::F32);
    let mut buffer = String::new();
    vreg.write_to(&mut buffer).expect("test");
    assert_eq!(buffer, "%f3");
}

#[test]
fn test_register_allocator_default() {
    let alloc = RegisterAllocator::default();
    assert_eq!(alloc.pressure_report().max_live, 0);
}

#[test]
fn test_physical_reg() {
    let preg = PhysicalReg(5);
    assert_eq!(preg.0, 5);
}

#[test]
fn test_register_pressure_fields() {
    let pressure = RegisterPressure { max_live: 10, spill_count: 0, utilization: 0.039 };
    assert_eq!(pressure.max_live, 10);
    assert_eq!(pressure.spill_count, 0);
    assert!((pressure.utilization - 0.039).abs() < f64::EPSILON);
}

#[test]
fn test_live_range_fields() {
    let range = LiveRange::new(5, 15);
    assert_eq!(range.start, 5);
    assert_eq!(range.end, 15);
}
