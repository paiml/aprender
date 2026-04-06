use super::*;
use crate::ptx::types::PtxType;
use proptest::prelude::*;

proptest! {
    /// align_split_point always produces result >= lower bound
    #[test]
    fn align_split_point_gte_lower(split in 0usize..1000, lower in 0usize..100, step in 1usize..32) {
        let result = align_split_point(split, lower, step);
        prop_assert!(result >= lower, "result {} < lower {}", result, lower);
    }

    /// align_split_point result is aligned to step boundary from lower
    #[test]
    fn align_split_point_aligned(split in 0usize..1000, lower in 0usize..100, step in 1usize..32) {
        let result = align_split_point(split, lower, step);
        if result > lower {
            prop_assert_eq!((result - lower) % step, 0,
                "result {} not aligned to step {} from lower {}", result, step, lower);
        }
    }

    /// align_split_point with step=1 returns max(split, lower)
    #[test]
    fn align_split_point_unit_step(split in 0usize..1000, lower in 0usize..100) {
        let result = align_split_point(split, lower, 1);
        let expected = split.max(lower);
        prop_assert_eq!(result, expected);
    }

    /// is_split_profitable with heavy ops always returns true
    #[test]
    fn heavy_ops_always_profitable(_dummy in 0u8..6) {
        let heavy_ops = [
            PtxOp::Ld,
            PtxOp::St,
            PtxOp::WmmaMma,
            PtxOp::WmmaLoadA,
            PtxOp::WmmaLoadB,
            PtxOp::WmmaLoadC,
            PtxOp::WmmaStoreD,
        ];

        for op in &heavy_ops {
            let instr = PtxInstruction::new(op.clone(), PtxType::F32);
            prop_assert!(is_split_profitable(&[instr], 100),
                "Heavy op {:?} should trigger profitability", op);
        }
    }

    /// is_split_profitable with light ops respects threshold
    #[test]
    fn light_ops_respect_threshold(count in 1usize..50, threshold in 1usize..100) {
        let light_instrs: Vec<_> = (0..count)
            .map(|_| PtxInstruction::new(PtxOp::Add, PtxType::F32))
            .collect();

        let result = is_split_profitable(&light_instrs, threshold);

        // Light ops trigger when count >= threshold
        prop_assert_eq!(result, count >= threshold,
            "count={}, threshold={}, result={}", count, threshold, result);
    }

    /// LoopPredicate::then_is_second is consistent
    #[test]
    fn loop_predicate_then_is_second_consistent(_dummy in 0u8..4) {
        // LessThan and LessEqual: then branch is first (smaller values)
        prop_assert!(!LoopPredicate::LessThan.then_is_second());
        prop_assert!(!LoopPredicate::LessEqual.then_is_second());

        // GreaterThan and GreaterEqual: then branch is second (larger values)
        prop_assert!(LoopPredicate::GreaterThan.then_is_second());
        prop_assert!(LoopPredicate::GreaterEqual.then_is_second());
    }

    /// analyze is idempotent - calling twice gives same result
    #[test]
    fn analyze_idempotent(instr_count in 0usize..10) {
        let instructions: Vec<_> = (0..instr_count)
            .map(|i| {
                if i % 3 == 0 {
                    PtxInstruction::new(PtxOp::Setp, PtxType::Pred)
                } else if i % 3 == 1 {
                    PtxInstruction::new(PtxOp::Ld, PtxType::F32)
                } else {
                    PtxInstruction::new(PtxOp::Add, PtxType::F32)
                }
            })
            .collect();

        let config = LoopSplitConfig::default();
        let first = analyze(&instructions, &config);
        let second = analyze(&instructions, &config);

        prop_assert!(is_idempotent(&first, &second));
    }

    /// LoopPredicate::from_cmp_op handles all comparison types
    #[test]
    fn from_cmp_op_complete(_dummy in 0u8..6) {
        // Should return Some for these
        prop_assert!(LoopPredicate::from_cmp_op(CmpOp::Lt).is_some());
        prop_assert!(LoopPredicate::from_cmp_op(CmpOp::Le).is_some());
        prop_assert!(LoopPredicate::from_cmp_op(CmpOp::Gt).is_some());
        prop_assert!(LoopPredicate::from_cmp_op(CmpOp::Ge).is_some());

        // Should return None for these
        prop_assert!(LoopPredicate::from_cmp_op(CmpOp::Eq).is_none());
        prop_assert!(LoopPredicate::from_cmp_op(CmpOp::Ne).is_none());
    }
}
