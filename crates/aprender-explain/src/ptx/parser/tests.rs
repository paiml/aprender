use super::*;

const SAMPLE_PTX: &str = r"
.version 8.0
.target sm_70
.address_size 64

.entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .f32 %f<24>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<12>;
    .reg .pred %p<4>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r1, [param_n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mad.lo.s32 %r5, %r4, %r3, %r2;

    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra exit;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

exit:
    ret;
}
";

#[test]
fn test_parse_registers() {
    let analyzer = PtxAnalyzer::new();
    let usage = analyzer.parse_registers(SAMPLE_PTX);

    assert_eq!(usage.f32_regs, 24);
    assert_eq!(usage.b32_regs, 18);
    assert_eq!(usage.b64_regs, 12);
    assert_eq!(usage.pred_regs, 4);
}

#[test]
fn test_parse_memory_ops() {
    let analyzer = PtxAnalyzer::new();
    let memory = analyzer.parse_memory_ops(SAMPLE_PTX);

    assert_eq!(memory.global_loads, 2);
    assert_eq!(memory.global_stores, 1);
    assert_eq!(memory.shared_loads, 0);
    assert_eq!(memory.shared_stores, 0);
}

#[test]
fn test_count_instructions() {
    let analyzer = PtxAnalyzer::new();
    let count = analyzer.count_instructions(SAMPLE_PTX);

    // Should count: ld.param (4) + mov (3) + mad + setp + mul + add (4) + ld.global (2) + st.global + ret
    assert!(count >= 15, "Expected >= 15 instructions, got {}", count);
}

#[test]
fn test_extract_kernel_name() {
    let analyzer = PtxAnalyzer::new();
    let name = analyzer.extract_kernel_name(SAMPLE_PTX);
    assert_eq!(name, "vector_add");
}

#[test]
fn test_extract_kernel_name_missing() {
    let analyzer = PtxAnalyzer::new();
    let name = analyzer.extract_kernel_name("// no kernel here");
    assert_eq!(name, "unknown");
}

#[test]
fn test_analyze_full_report() {
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(SAMPLE_PTX).unwrap();

    assert_eq!(report.name, "vector_add");
    assert_eq!(report.target, "PTX");
    assert_eq!(report.registers.f32_regs, 24);
    assert_eq!(report.memory.global_loads, 2);
    assert!(report.estimated_occupancy > 0.0);
}

#[test]
fn test_detect_spills() {
    let analyzer = PtxAnalyzer::new();

    // No spills in sample PTX
    let warnings = analyzer.detect_muda(SAMPLE_PTX);
    let spill_warnings: Vec<_> =
        warnings.iter().filter(|w| matches!(w.muda_type, MudaType::Transport)).collect();
    assert!(spill_warnings.is_empty());

    // PTX with spills
    let ptx_with_spills = r"
        .local .align 4 .b8 __local_depot[32];
        .reg .f32 %f<4>;
    ";
    let spill_warning = analyzer.detect_spills(ptx_with_spills);
    assert!(spill_warning.is_some());
}

#[test]
fn test_detect_high_register_pressure() {
    let analyzer = PtxAnalyzer::new();

    let high_reg_ptx = r"
        .entry big_kernel()
        {
            .reg .f32 %f<200>;
            ret;
        }
    ";

    let warnings = analyzer.detect_muda(high_reg_ptx);
    let reg_warnings: Vec<_> =
        warnings.iter().filter(|w| matches!(w.muda_type, MudaType::Overprocessing)).collect();
    assert!(!reg_warnings.is_empty());
}

#[test]
fn test_json_output() {
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(SAMPLE_PTX).unwrap();

    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(json.contains("vector_add"));
    assert!(json.contains("PTX"));
    assert!(json.contains("f32_regs"));
}

#[test]
fn test_occupancy_estimation() {
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(SAMPLE_PTX).unwrap();

    // 58 total registers should give good occupancy
    assert!(
        report.estimated_occupancy > 0.5,
        "Expected > 50% occupancy, got {}",
        report.estimated_occupancy
    );
}

#[test]
fn test_roofline_estimation() {
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(SAMPLE_PTX).unwrap();

    // Vector add is memory-bound
    assert!(report.roofline.memory_bound, "Vector add should be memory-bound");
}

/// F030 (Memory): Identifies coalesced pattern (tid*4 detected)
#[test]
fn f030_memory_identifies_coalesced_pattern() {
    let analyzer = PtxAnalyzer::new();

    // PTX with tid-based indexing (coalesced pattern)
    let coalesced_ptx = r"
        .entry coalesced_kernel()
        {
            .reg .f32 %f<4>;
            .reg .b32 %r<4>;
            .reg .b64 %rd<4>;
            // tid.x-based indexing indicates coalesced access
            mov.u32 %r0, %tid.x;
            mul.wide.u32 %rd0, %r0, 4;
            ld.global.f32 %f0, [%rd0];
            st.global.f32 [%rd0], %f0;
            ret;
        }
    ";

    let memory = analyzer.parse_memory_ops(coalesced_ptx);

    // Should detect tid references indicating coalesced access
    assert!(memory.coalesced_ratio > 0.0, "Should detect tid-based coalesced pattern");
    assert!(memory.global_loads > 0, "Should detect global loads");
    assert!(memory.global_stores > 0, "Should detect global stores");
}

/// F034: Warns on <80% coalescing ratio
#[test]
fn f034_warn_low_coalescing() {
    let analyzer = PtxAnalyzer::new();

    // PTX with many global loads but no tid references (uncoalesced pattern)
    let uncoalesced_ptx = r"
        .entry uncoalesced_kernel()
        {
            .reg .f32 %f<4>;
            .reg .b64 %rd<4>;
            // Many loads without tid-based indexing
            ld.global.f32 %f0, [%rd0];
            ld.global.f32 %f1, [%rd1];
            ld.global.f32 %f2, [%rd2];
            ld.global.f32 %f3, [%rd3];
            st.global.f32 [%rd0], %f0;
            st.global.f32 [%rd1], %f1;
            st.global.f32 [%rd2], %f2;
            st.global.f32 [%rd3], %f3;
            ret;
        }
    ";

    let warnings = analyzer.detect_muda(uncoalesced_ptx);
    let coalescing_warnings: Vec<_> = warnings
        .iter()
        .filter(|w| matches!(w.muda_type, MudaType::Waiting))
        .filter(|w| w.description.contains("coalescing"))
        .collect();

    assert!(!coalescing_warnings.is_empty(), "Should warn on <80% coalescing ratio");
}
