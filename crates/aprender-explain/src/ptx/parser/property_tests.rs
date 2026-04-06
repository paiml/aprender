use super::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_register_count_non_negative(
        f32_count in 0u32..100,
        b32_count in 0u32..100,
    ) {
        let ptx = format!(
            ".entry test() {{ .reg .f32 %f<{}>; .reg .b32 %r<{}>; ret; }}",
            f32_count, b32_count
        );
        let analyzer = PtxAnalyzer::new();
        let usage = analyzer.parse_registers(&ptx);

        prop_assert_eq!(usage.f32_regs, f32_count);
        prop_assert_eq!(usage.b32_regs, b32_count);
        prop_assert!(usage.total() >= f32_count + b32_count);
    }

    #[test]
    fn prop_occupancy_bounded(regs in 1u32..256) {
        let usage = RegisterUsage {
            f32_regs: regs,
            ..Default::default()
        };
        let occ = usage.estimated_occupancy();
        prop_assert!((0.0..=1.0).contains(&occ));
    }

    #[test]
    fn prop_memory_counts_non_negative(
        global_ld in 0usize..50,
        global_st in 0usize..50,
    ) {
        let mut ptx = String::from(".entry test() {\n");
        for _ in 0..global_ld {
            ptx.push_str("    ld.global.f32 %f1, [%rd1];\n");
        }
        for _ in 0..global_st {
            ptx.push_str("    st.global.f32 [%rd1], %f1;\n");
        }
        ptx.push_str("    ret;\n}");

        let analyzer = PtxAnalyzer::new();
        let memory = analyzer.parse_memory_ops(&ptx);

        prop_assert_eq!(memory.global_loads, global_ld as u32);
        prop_assert_eq!(memory.global_stores, global_st as u32);
    }
}
