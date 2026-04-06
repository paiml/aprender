//! EXTREME TDD Tests for WASM Coverage Tooling
//!
//! Per spec: `docs/specifications/probar-wasm-coverage-tooling.md`
//!
//! Tests written FIRST following Popperian falsification methodology.
//! Each test represents a falsifiable hypothesis about coverage behavior.

#![allow(
    clippy::redundant_clone,
    clippy::float_cmp,
    clippy::single_char_pattern,
    clippy::needless_range_loop,
    clippy::clone_on_copy,
    unused_results
)]

use super::*;

// ============================================================================
// §5.2 Poka-Yoke Tests: Type-Safe Block IDs
// ============================================================================

mod block_id_tests {
    use super::*;

    /// H₀-BLOCK-01: BlockId is copy and equality comparable
    #[test]
    fn test_block_id_copy_and_eq() {
        let id1 = BlockId::new(42);
        let id2 = id1; // Copy
        assert_eq!(id1, id2);
    }

    /// H₀-BLOCK-02: BlockId preserves inner value
    #[test]
    fn test_block_id_inner_value() {
        let id = BlockId::new(12345);
        assert_eq!(id.as_u32(), 12345);
    }

    /// H₀-BLOCK-03: BlockId is hashable for use in collections
    #[test]
    fn test_block_id_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BlockId::new(1));
        set.insert(BlockId::new(2));
        set.insert(BlockId::new(1)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    /// H₀-BLOCK-04: BlockId ordering is consistent
    #[test]
    fn test_block_id_ordering() {
        let id1 = BlockId::new(1);
        let id2 = BlockId::new(2);
        assert!(id1 < id2);
    }
}

mod function_id_tests {
    use super::*;

    /// H₀-FUNC-01: FunctionId is distinct from BlockId (type safety)
    #[test]
    fn test_function_id_type_safety() {
        let func_id = FunctionId::new(42);
        let block_id = BlockId::new(42);
        // These should NOT be comparable at compile time
        // This test just verifies they're separate types
        assert_eq!(func_id.as_u32(), block_id.as_u32());
    }

    /// H₀-FUNC-02: FunctionId is hashable
    #[test]
    fn test_function_id_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FunctionId::new(1));
        set.insert(FunctionId::new(2));
        assert_eq!(set.len(), 2);
    }
}

mod edge_id_tests {
    use super::*;

    /// H₀-EDGE-01: EdgeId encodes source and target correctly
    #[test]
    fn test_edge_id_encoding() {
        let from = BlockId::new(100);
        let to = BlockId::new(200);
        let edge = EdgeId::new(from, to);

        assert_eq!(edge.source(), from);
        assert_eq!(edge.target(), to);
    }

    /// H₀-EDGE-02: EdgeId is unique for different (from, to) pairs
    #[test]
    fn test_edge_id_uniqueness() {
        let edge1 = EdgeId::new(BlockId::new(1), BlockId::new(2));
        let edge2 = EdgeId::new(BlockId::new(2), BlockId::new(1));
        let edge3 = EdgeId::new(BlockId::new(1), BlockId::new(2));

        assert_ne!(edge1, edge2); // Order matters
        assert_eq!(edge1, edge3); // Same edges are equal
    }

    /// H₀-EDGE-03: EdgeId handles max u32 values
    #[test]
    fn test_edge_id_max_values() {
        let from = BlockId::new(u32::MAX);
        let to = BlockId::new(u32::MAX);
        let edge = EdgeId::new(from, to);

        assert_eq!(edge.source(), from);
        assert_eq!(edge.target(), to);
    }

    /// H₀-EDGE-04: EdgeId is hashable
    #[test]
    fn test_edge_id_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(EdgeId::new(BlockId::new(1), BlockId::new(2)));
        set.insert(EdgeId::new(BlockId::new(3), BlockId::new(4)));
        assert_eq!(set.len(), 2);
    }
}

// ============================================================================
// §5.1.1 Soft Jidoka Tests: Stop vs LogAndContinue
// ============================================================================

mod jidoka_tests {
    use super::*;

    /// H₀-JIDOKA-01: Uninstrumented execution triggers hard stop
    #[test]
    fn test_uninstrumented_execution_is_hard_stop() {
        let violation = CoverageViolation::UninstrumentedExecution {
            block_id: BlockId::new(42),
        };
        assert_eq!(violation.action(), JidokaAction::Stop);
    }

    /// H₀-JIDOKA-02: Impossible edge triggers hard stop
    #[test]
    fn test_impossible_edge_is_hard_stop() {
        let violation = CoverageViolation::ImpossibleEdge {
            from: BlockId::new(1),
            to: BlockId::new(2),
        };
        assert_eq!(violation.action(), JidokaAction::Stop);
    }

    /// H₀-JIDOKA-03: Counter overflow triggers soft stop (log and continue)
    #[test]
    fn test_counter_overflow_is_soft_stop() {
        let violation = CoverageViolation::CounterOverflow {
            block_id: BlockId::new(42),
        };
        assert_eq!(violation.action(), JidokaAction::LogAndContinue);
    }

    /// H₀-JIDOKA-04: Coverage regression triggers soft stop
    #[test]
    fn test_coverage_regression_is_soft_stop() {
        let violation = CoverageViolation::CoverageRegression {
            expected: 95.0,
            actual: 90.0,
        };
        assert_eq!(violation.action(), JidokaAction::LogAndContinue);
    }

    /// H₀-JIDOKA-05: Tainted blocks tracker works correctly
    #[test]
    fn test_tainted_blocks_tracking() {
        let mut tainted = TaintedBlocks::new();
        let block = BlockId::new(42);

        assert!(!tainted.is_tainted(block));

        tainted.taint(
            block,
            CoverageViolation::CounterOverflow { block_id: block },
        );

        assert!(tainted.is_tainted(block));
        assert_eq!(tainted.tainted_count(), 1);
    }

    /// H₀-JIDOKA-06: Multiple taints on same block don't duplicate
    #[test]
    fn test_tainted_blocks_no_duplicates() {
        let mut tainted = TaintedBlocks::new();
        let block = BlockId::new(42);

        tainted.taint(
            block,
            CoverageViolation::CounterOverflow { block_id: block },
        );
        tainted.taint(
            block,
            CoverageViolation::CounterOverflow { block_id: block },
        );

        assert_eq!(tainted.tainted_count(), 1);
        assert_eq!(tainted.violation_count(), 2); // But violations are logged
    }
}

// ============================================================================
// §5.3.1 Thread-Local Buffering Tests: Muda Elimination
// ============================================================================

mod thread_local_tests {
    use super::*;

    /// H₀-TL-01: ThreadLocalCounters initializes to zero
    #[test]
    fn test_thread_local_init_zero() {
        let counters = ThreadLocalCounters::new(100);
        for i in 0..100 {
            assert_eq!(counters.get(BlockId::new(i)), 0);
        }
    }

    /// H₀-TL-02: Increment updates local counter
    #[test]
    fn test_thread_local_increment() {
        let mut counters = ThreadLocalCounters::new(100);
        let block = BlockId::new(42);

        counters.increment(block);
        counters.increment(block);
        counters.increment(block);

        assert_eq!(counters.get(block), 3);
    }

    /// H₀-TL-03: Flush threshold triggers automatic flush
    #[test]
    fn test_thread_local_flush_threshold() {
        let mut counters = ThreadLocalCounters::with_flush_threshold(100, 10);

        // Increment 15 times (should trigger flush after 10)
        for _ in 0..15 {
            counters.increment(BlockId::new(0));
        }

        assert!(counters.flush_count() >= 1);
    }

    /// H₀-TL-04: Manual flush resets local counters
    #[test]
    fn test_thread_local_manual_flush() {
        let mut counters = ThreadLocalCounters::new(100);
        let block = BlockId::new(42);

        counters.increment(block);
        counters.increment(block);

        let flushed = counters.flush();
        assert_eq!(flushed[42], 2);
        assert_eq!(counters.get(block), 0); // Reset after flush
    }

    /// H₀-TL-05: Flush returns correct counts for all blocks
    #[test]
    fn test_thread_local_flush_all_blocks() {
        let mut counters = ThreadLocalCounters::new(5);

        counters.increment(BlockId::new(0));
        counters.increment(BlockId::new(1));
        counters.increment(BlockId::new(1));
        counters.increment(BlockId::new(2));
        counters.increment(BlockId::new(2));
        counters.increment(BlockId::new(2));

        let flushed = counters.flush();
        assert_eq!(flushed[0], 1);
        assert_eq!(flushed[1], 2);
        assert_eq!(flushed[2], 3);
        assert_eq!(flushed[3], 0);
        assert_eq!(flushed[4], 0);
    }
}

// ============================================================================
// §5.4.1 Superblock Tests: Heijunka Tiling
// ============================================================================

mod superblock_tests {
    use super::*;

    /// H₀-SB-01: SuperblockId is distinct type
    #[test]
    fn test_superblock_id_type() {
        let id = SuperblockId::new(42);
        assert_eq!(id.as_u32(), 42);
    }

    /// H₀-SB-02: Superblock contains blocks
    #[test]
    fn test_superblock_contains_blocks() {
        let blocks = vec![BlockId::new(1), BlockId::new(2), BlockId::new(3)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks.clone(), FunctionId::new(0));

        assert_eq!(superblock.block_count(), 3);
        assert!(superblock.contains(BlockId::new(2)));
        assert!(!superblock.contains(BlockId::new(99)));
    }

    /// H₀-SB-03: SuperblockBuilder groups by target size
    #[test]
    fn test_superblock_builder_target_size() {
        let builder = SuperblockBuilder::new().with_target_size(3);

        let blocks: Vec<BlockId> = (0..10).map(BlockId::new).collect();
        let superblocks = builder.build_from_blocks(&blocks, FunctionId::new(0));

        // 10 blocks with target size 3 = 4 superblocks (3+3+3+1)
        assert_eq!(superblocks.len(), 4);
        assert_eq!(superblocks[0].block_count(), 3);
        assert_eq!(superblocks[3].block_count(), 1); // Remainder
    }

    /// H₀-SB-04: SuperblockBuilder respects max size
    #[test]
    fn test_superblock_builder_max_size() {
        let builder = SuperblockBuilder::new()
            .with_target_size(100)
            .with_max_size(5);

        let blocks: Vec<BlockId> = (0..10).map(BlockId::new).collect();
        let superblocks = builder.build_from_blocks(&blocks, FunctionId::new(0));

        // Max size 5 overrides target size 100
        assert_eq!(superblocks.len(), 2);
        assert_eq!(superblocks[0].block_count(), 5);
        assert_eq!(superblocks[1].block_count(), 5);
    }

    /// H₀-SB-05: Empty block list produces no superblocks
    #[test]
    fn test_superblock_builder_empty() {
        let builder = SuperblockBuilder::new();
        let blocks: Vec<BlockId> = vec![];
        let superblocks = builder.build_from_blocks(&blocks, FunctionId::new(0));

        assert!(superblocks.is_empty());
    }

    /// H₀-SB-06: Superblock cost estimate is block count by default
    #[test]
    fn test_superblock_cost_estimate() {
        let blocks = vec![BlockId::new(1), BlockId::new(2), BlockId::new(3)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks, FunctionId::new(0));

        assert_eq!(superblock.cost_estimate(), 3);
    }
}

// ============================================================================
// §5.3 Memory View Tests
// ============================================================================

mod memory_view_tests {
    use super::*;

    /// H₀-MEM-01: CoverageMemoryView reads counter correctly
    #[test]
    fn test_memory_view_read_counter() {
        // Simulate WASM linear memory with counters at offset 0
        let mut memory = vec![0u8; 80]; // 10 counters * 8 bytes

        // Write counter value 42 at block 0
        memory[0..8].copy_from_slice(&42u64.to_le_bytes());

        // Write counter value 100 at block 5
        memory[40..48].copy_from_slice(&100u64.to_le_bytes());

        let view = CoverageMemoryView::new(&memory, 0, 10);

        assert_eq!(view.read_counter(BlockId::new(0)), 42);
        assert_eq!(view.read_counter(BlockId::new(5)), 100);
        assert_eq!(view.read_counter(BlockId::new(9)), 0);
    }

    /// H₀-MEM-02: CoverageMemoryView reads all counters
    #[test]
    fn test_memory_view_read_all() {
        let mut memory = vec![0u8; 40]; // 5 counters

        memory[0..8].copy_from_slice(&1u64.to_le_bytes());
        memory[8..16].copy_from_slice(&2u64.to_le_bytes());
        memory[16..24].copy_from_slice(&3u64.to_le_bytes());
        memory[24..32].copy_from_slice(&4u64.to_le_bytes());
        memory[32..40].copy_from_slice(&5u64.to_le_bytes());

        let view = CoverageMemoryView::new(&memory, 0, 5);
        let counters = view.read_all_counters();

        assert_eq!(counters, vec![1, 2, 3, 4, 5]);
    }

    /// H₀-MEM-03: CoverageMemoryView handles offset correctly
    #[test]
    fn test_memory_view_with_offset() {
        let mut memory = vec![0u8; 100];

        // Counters start at offset 16
        memory[16..24].copy_from_slice(&42u64.to_le_bytes());

        let view = CoverageMemoryView::new(&memory, 16, 5);
        assert_eq!(view.read_counter(BlockId::new(0)), 42);
    }

    /// H₀-MEM-04: CoverageMemoryView block count is correct
    #[test]
    fn test_memory_view_block_count() {
        let memory = vec![0u8; 80];
        let view = CoverageMemoryView::new(&memory, 0, 10);
        assert_eq!(view.block_count(), 10);
    }
}

// ============================================================================
// §9.1 Coverage Collector Tests
// ============================================================================

mod collector_tests {
    use super::*;

    /// H₀-COLL-01: CoverageConfig builder works correctly
    #[test]
    fn test_coverage_config_builder() {
        let config = CoverageConfig::builder()
            .granularity(Granularity::BasicBlock)
            .parallel(true)
            .jidoka_enabled(true)
            .build();

        assert_eq!(config.granularity, Granularity::BasicBlock);
        assert!(config.parallel);
        assert!(config.jidoka_enabled);
    }

    /// H₀-COLL-02: CoverageConfig defaults are sensible
    #[test]
    fn test_coverage_config_defaults() {
        let config = CoverageConfig::default();

        assert_eq!(config.granularity, Granularity::Function);
        assert!(!config.parallel);
        assert!(config.jidoka_enabled);
    }

    /// H₀-COLL-03: CoverageCollector tracks session state
    #[test]
    fn test_collector_session_state() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        assert!(!collector.is_session_active());

        collector.begin_session("test_session");
        assert!(collector.is_session_active());

        let _report = collector.end_session();
        assert!(!collector.is_session_active());
    }

    /// H₀-COLL-04: CoverageCollector tracks test state
    #[test]
    fn test_collector_test_state() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        collector.begin_session("test_session");
        assert!(!collector.is_test_active());

        collector.begin_test("test_1");
        assert!(collector.is_test_active());

        collector.end_test();
        assert!(!collector.is_test_active());

        let _ = collector.end_session();
    }

    /// H₀-COLL-05: CoverageCollector records block hits
    #[test]
    fn test_collector_records_hits() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        collector.begin_session("test");
        collector.begin_test("test_1");

        collector.record_hit(BlockId::new(0));
        collector.record_hit(BlockId::new(1));
        collector.record_hit(BlockId::new(0));

        collector.end_test();
        let report = collector.end_session();

        assert_eq!(report.get_hit_count(BlockId::new(0)), 2);
        assert_eq!(report.get_hit_count(BlockId::new(1)), 1);
    }
}

// ============================================================================
// §6 Coverage Report Tests
// ============================================================================

mod report_tests {
    use super::*;

    /// H₀-RPT-01: CoverageReport calculates percentage correctly
    #[test]
    fn test_report_coverage_percentage() {
        let mut report = CoverageReport::new(10); // 10 blocks

        report.record_hit(BlockId::new(0));
        report.record_hit(BlockId::new(1));
        report.record_hit(BlockId::new(2));
        report.record_hit(BlockId::new(3));
        report.record_hit(BlockId::new(4));

        let summary = report.summary();
        assert_eq!(summary.total_blocks, 10);
        assert_eq!(summary.covered_blocks, 5);
        assert!((summary.coverage_percent - 50.0).abs() < 0.01);
    }

    /// H₀-RPT-02: CoverageReport handles zero blocks
    #[test]
    fn test_report_zero_blocks() {
        let report = CoverageReport::new(0);
        let summary = report.summary();

        assert_eq!(summary.total_blocks, 0);
        assert_eq!(summary.covered_blocks, 0);
        // Coverage of empty set is 100% (vacuously true)
        assert!((summary.coverage_percent - 100.0).abs() < 0.01);
    }

    /// H₀-RPT-03: CoverageReport tracks hit counts
    #[test]
    fn test_report_hit_counts() {
        let mut report = CoverageReport::new(5);

        report.record_hit(BlockId::new(0));
        report.record_hit(BlockId::new(0));
        report.record_hit(BlockId::new(0));
        report.record_hit(BlockId::new(1));

        assert_eq!(report.get_hit_count(BlockId::new(0)), 3);
        assert_eq!(report.get_hit_count(BlockId::new(1)), 1);
        assert_eq!(report.get_hit_count(BlockId::new(2)), 0);
    }

    /// H₀-RPT-04: CoverageReport lists uncovered blocks
    #[test]
    fn test_report_uncovered_blocks() {
        let mut report = CoverageReport::new(5);

        report.record_hit(BlockId::new(1));
        report.record_hit(BlockId::new(3));

        let uncovered = report.uncovered_blocks();
        assert_eq!(uncovered.len(), 3);
        assert!(uncovered.contains(&BlockId::new(0)));
        assert!(uncovered.contains(&BlockId::new(2)));
        assert!(uncovered.contains(&BlockId::new(4)));
    }

    /// H₀-RPT-05: BlockCoverage records source location
    #[test]
    fn test_block_coverage_source_location() {
        let block_cov = BlockCoverage {
            block_id: BlockId::new(42),
            hit_count: 10,
            source_location: Some("src/pong.rs:142".to_string()),
            function_name: Some("update_ball".to_string()),
        };

        assert_eq!(
            block_cov.source_location,
            Some("src/pong.rs:142".to_string())
        );
        assert_eq!(block_cov.function_name, Some("update_ball".to_string()));
    }
}

// ============================================================================
// §6 Falsification Hypothesis Tests
// ============================================================================

mod hypothesis_tests {
    use super::*;

    /// H₀-HYP-01: Determinism hypothesis detects variance
    #[test]
    fn test_hypothesis_determinism() {
        let hypothesis = CoverageHypothesis::determinism();
        assert_eq!(hypothesis.name(), "H0-COV-01");

        // Same coverage = not rejected
        let result = hypothesis.evaluate(&[95.0, 95.0, 95.0, 95.0, 95.0]);
        assert!(!result.rejected);

        // Different coverage = rejected
        let result = hypothesis.evaluate(&[95.0, 90.0, 92.0, 88.0, 95.0]);
        assert!(result.rejected);
    }

    /// H₀-HYP-02: Completeness hypothesis checks threshold
    #[test]
    fn test_hypothesis_completeness() {
        let hypothesis = CoverageHypothesis::completeness(95.0);
        assert_eq!(hypothesis.name(), "H0-COV-02");

        // Above threshold = not rejected
        let result = hypothesis.evaluate(&[96.0, 97.0, 95.5, 96.2, 95.1]);
        assert!(!result.rejected);

        // Below threshold = rejected
        let result = hypothesis.evaluate(&[90.0, 91.0, 89.0, 92.0, 88.0]);
        assert!(result.rejected);
    }

    /// H₀-HYP-03: No regression hypothesis compares to baseline
    #[test]
    fn test_hypothesis_no_regression() {
        let baseline = 90.0;
        let hypothesis = CoverageHypothesis::no_regression(baseline);
        assert_eq!(hypothesis.name(), "H0-COV-03");

        // Same or better = not rejected
        let result = hypothesis.evaluate(&[91.0, 92.0, 90.5, 91.2, 90.0]);
        assert!(!result.rejected);

        // Worse = rejected
        let result = hypothesis.evaluate(&[85.0, 84.0, 86.0, 83.0, 85.0]);
        assert!(result.rejected);
    }

    /// H₀-HYP-04: NullificationConfig sets correct parameters
    #[test]
    fn test_nullification_config() {
        let config = NullificationConfig::princeton();
        assert_eq!(config.runs, 5);
        assert!((config.alpha - 0.05).abs() < 0.001);
    }

    /// H₀-HYP-05: NullificationResult reports p-value
    #[test]
    fn test_nullification_result() {
        let result = NullificationResult {
            hypothesis_name: "H0-COV-01".to_string(),
            rejected: false,
            p_value: 0.42,
            effect_size: 0.08,
            confidence_interval: (94.1, 96.2),
        };

        assert!(!result.rejected);
        assert!(result.p_value > 0.05); // Not significant
    }
}

// ============================================================================
// §5.4 Executor Tests
// ============================================================================

mod executor_tests {
    use super::*;

    /// H₀-EXEC-01: CoverageExecutor creates with superblocks
    #[test]
    fn test_executor_new() {
        let blocks = vec![BlockId::new(0), BlockId::new(1), BlockId::new(2)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks, FunctionId::new(0));
        let executor = CoverageExecutor::new(vec![superblock]);

        assert_eq!(executor.superblock_count(), 1);
        assert_eq!(executor.total_block_count(), 3);
    }

    /// H₀-EXEC-02: CoverageExecutor with_workers builder works
    #[test]
    fn test_executor_with_workers() {
        let blocks = vec![BlockId::new(0)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks, FunctionId::new(0));
        let executor = CoverageExecutor::new(vec![superblock]).with_workers(8);

        assert_eq!(executor.worker_count(), 8);
    }

    /// H₀-EXEC-03: CoverageExecutor with_work_stealing builder works
    #[test]
    fn test_executor_with_work_stealing() {
        let blocks = vec![BlockId::new(0)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks, FunctionId::new(0));
        let executor = CoverageExecutor::new(vec![superblock]).with_work_stealing(false);

        // Just verify it doesn't panic - work_stealing is internal
        assert_eq!(executor.superblock_count(), 1);
    }

    /// H₀-EXEC-04: CoverageExecutor execute runs test function
    #[test]
    fn test_executor_execute() {
        let blocks1 = vec![BlockId::new(0), BlockId::new(1)];
        let blocks2 = vec![BlockId::new(2), BlockId::new(3)];
        let sb1 = Superblock::new(SuperblockId::new(0), blocks1, FunctionId::new(0));
        let sb2 = Superblock::new(SuperblockId::new(1), blocks2, FunctionId::new(0));

        let executor = CoverageExecutor::new(vec![sb1, sb2]);

        let report = executor.execute(|sb| SuperblockResult {
            id: sb.id(),
            success: true,
            error: None,
        });

        let summary = report.summary();
        assert_eq!(summary.covered_blocks, 4);
        assert!((summary.coverage_percent - 100.0).abs() < 0.01);
    }

    /// H₀-EXEC-05: CoverageExecutor handles failed superblocks
    #[test]
    fn test_executor_execute_with_failure() {
        let blocks1 = vec![BlockId::new(0), BlockId::new(1)];
        let blocks2 = vec![BlockId::new(2), BlockId::new(3)];
        let sb1 = Superblock::new(SuperblockId::new(0), blocks1, FunctionId::new(0));
        let sb2 = Superblock::new(SuperblockId::new(1), blocks2, FunctionId::new(0));

        let executor = CoverageExecutor::new(vec![sb1, sb2]);

        let report = executor.execute(|sb| {
            // Only first superblock succeeds
            if sb.id().as_u32() == 0 {
                SuperblockResult {
                    id: sb.id(),
                    success: true,
                    error: None,
                }
            } else {
                SuperblockResult {
                    id: sb.id(),
                    success: false,
                    error: Some("Test failed".to_string()),
                }
            }
        });

        let summary = report.summary();
        assert_eq!(summary.covered_blocks, 2);
        assert!((summary.coverage_percent - 50.0).abs() < 0.01);
    }

    /// H₀-EXEC-06: CoverageExecutor empty superblocks
    #[test]
    fn test_executor_empty() {
        let executor = CoverageExecutor::new(vec![]);

        assert_eq!(executor.superblock_count(), 0);
        assert_eq!(executor.total_block_count(), 0);

        let report = executor.execute(|_| SuperblockResult {
            id: SuperblockId::new(0),
            success: true,
            error: None,
        });

        let summary = report.summary();
        assert_eq!(summary.total_blocks, 0);
    }

    /// H₀-EXEC-07: SuperblockResult creation
    #[test]
    fn test_superblock_result() {
        let result = SuperblockResult {
            id: SuperblockId::new(42),
            success: true,
            error: None,
        };

        assert_eq!(result.id.as_u32(), 42);
        assert!(result.success);
        assert!(result.error.is_none());
    }

    /// H₀-EXEC-08: SuperblockResult with error
    #[test]
    fn test_superblock_result_with_error() {
        let result = SuperblockResult {
            id: SuperblockId::new(42),
            success: false,
            error: Some("Something went wrong".to_string()),
        };

        assert!(!result.success);
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

mod additional_block_tests {
    use super::*;

    /// H₀-BLOCK-05: BlockId debug format
    #[test]
    fn test_block_id_debug() {
        let id = BlockId::new(42);
        let debug = format!("{:?}", id);
        assert!(debug.contains("BlockId"));
        assert!(debug.contains("42"));
    }

    /// H₀-BLOCK-06: FunctionId ordering
    #[test]
    fn test_function_id_ordering() {
        let id1 = FunctionId::new(1);
        let id2 = FunctionId::new(2);
        assert!(id1 < id2);
        assert!(id2 > id1);
    }

    /// H₀-BLOCK-07: FunctionId debug format
    #[test]
    fn test_function_id_debug() {
        let id = FunctionId::new(42);
        let debug = format!("{:?}", id);
        assert!(debug.contains("FunctionId"));
        assert!(debug.contains("42"));
    }

    /// H₀-BLOCK-08: FunctionId copy
    #[test]
    fn test_function_id_copy() {
        let id1 = FunctionId::new(42);
        let id2 = id1; // Copy
        assert_eq!(id1, id2);
    }

    /// H₀-BLOCK-09: EdgeId debug format
    #[test]
    fn test_edge_id_debug() {
        let edge = EdgeId::new(BlockId::new(1), BlockId::new(2));
        let debug = format!("{:?}", edge);
        assert!(debug.contains("EdgeId"));
    }

    /// H₀-BLOCK-10: EdgeId ordering
    #[test]
    fn test_edge_id_ordering() {
        let edge1 = EdgeId::new(BlockId::new(0), BlockId::new(1));
        let edge2 = EdgeId::new(BlockId::new(1), BlockId::new(0));
        assert!(edge1 < edge2 || edge2 < edge1); // Different ordering
    }

    /// H₀-BLOCK-11: EdgeId as_u64
    #[test]
    fn test_edge_id_as_u64() {
        let edge = EdgeId::new(BlockId::new(1), BlockId::new(2));
        let raw = edge.as_u64();
        // from=1 is in upper 32 bits, to=2 is in lower 32 bits
        assert_eq!(raw, (1u64 << 32) | 2u64);
    }
}

mod additional_jidoka_tests {
    use super::*;

    /// H₀-JIDOKA-07: JidokaAction equality
    #[test]
    fn test_jidoka_action_equality() {
        assert_eq!(JidokaAction::Stop, JidokaAction::Stop);
        assert_eq!(JidokaAction::LogAndContinue, JidokaAction::LogAndContinue);
        assert_eq!(JidokaAction::Warn, JidokaAction::Warn);
        assert_ne!(JidokaAction::Stop, JidokaAction::Warn);
    }

    /// H₀-JIDOKA-08: JidokaAction debug format
    #[test]
    fn test_jidoka_action_debug() {
        let action = JidokaAction::Stop;
        let debug = format!("{:?}", action);
        assert!(debug.contains("Stop"));
    }

    /// H₀-JIDOKA-09: TaintedBlocks new is empty
    #[test]
    fn test_tainted_blocks_new_empty() {
        let tainted = TaintedBlocks::new();
        assert_eq!(tainted.tainted_count(), 0);
        assert_eq!(tainted.violation_count(), 0);
    }

    /// H₀-JIDOKA-10: CoverageViolation debug format
    #[test]
    fn test_coverage_violation_debug() {
        let violation = CoverageViolation::UninstrumentedExecution {
            block_id: BlockId::new(42),
        };
        let debug = format!("{:?}", violation);
        assert!(debug.contains("UninstrumentedExecution"));
    }

    /// H₀-JIDOKA-11: CoverageViolation clone
    #[test]
    fn test_coverage_violation_clone() {
        let v1 = CoverageViolation::CounterOverflow {
            block_id: BlockId::new(42),
        };
        let v2 = v1.clone();
        assert_eq!(v1.action(), v2.action());
    }
}

mod additional_superblock_tests {
    use super::*;

    /// H₀-SB-07: Superblock id accessor
    #[test]
    fn test_superblock_id_accessor() {
        let blocks = vec![BlockId::new(1)];
        let superblock = Superblock::new(SuperblockId::new(42), blocks, FunctionId::new(0));
        assert_eq!(superblock.id().as_u32(), 42);
    }

    /// H₀-SB-08: Superblock function accessor
    #[test]
    fn test_superblock_function_accessor() {
        let blocks = vec![BlockId::new(1)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks, FunctionId::new(99));
        assert_eq!(superblock.function().as_u32(), 99);
    }

    /// H₀-SB-09: Superblock iter
    #[test]
    fn test_superblock_iter() {
        let blocks = vec![BlockId::new(1), BlockId::new(2), BlockId::new(3)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks.clone(), FunctionId::new(0));

        let iterated: Vec<BlockId> = superblock.iter().copied().collect();
        assert_eq!(iterated, blocks);
    }

    /// H₀-SB-10: Superblock debug format
    #[test]
    fn test_superblock_debug() {
        let blocks = vec![BlockId::new(1)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks, FunctionId::new(0));
        let debug = format!("{:?}", superblock);
        assert!(debug.contains("Superblock"));
    }

    /// H₀-SB-11: SuperblockId hashable
    #[test]
    fn test_superblock_id_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SuperblockId::new(1));
        set.insert(SuperblockId::new(2));
        set.insert(SuperblockId::new(1));
        assert_eq!(set.len(), 2);
    }

    /// H₀-SB-12: SuperblockId ordering
    #[test]
    fn test_superblock_id_ordering() {
        let id1 = SuperblockId::new(1);
        let id2 = SuperblockId::new(2);
        assert!(id1 < id2);
    }

    /// H₀-SB-13: SuperblockBuilder default
    #[test]
    fn test_superblock_builder_default() {
        let builder = SuperblockBuilder::default();
        let blocks: Vec<BlockId> = (0..100).map(BlockId::new).collect();
        let superblocks = builder.build_from_blocks(&blocks, FunctionId::new(0));
        // Default target size is 64, so 100 blocks = 2 superblocks
        assert_eq!(superblocks.len(), 2);
    }
}

mod additional_memory_tests {
    use super::*;

    /// H₀-MEM-05: CoverageMemoryView empty
    #[test]
    fn test_memory_view_empty() {
        let memory: Vec<u8> = vec![];
        let view = CoverageMemoryView::new(&memory, 0, 0);
        assert_eq!(view.block_count(), 0);
        let counters = view.read_all_counters();
        assert!(counters.is_empty());
    }

    /// H₀-MEM-06: CoverageMemoryView debug
    #[test]
    fn test_memory_view_debug() {
        let memory = vec![0u8; 16];
        let view = CoverageMemoryView::new(&memory, 0, 2);
        let debug = format!("{:?}", view);
        assert!(debug.contains("CoverageMemoryView"));
    }
}

mod additional_hypothesis_tests {
    use super::*;

    /// H₀-HYP-06: Mutation correlation hypothesis
    #[test]
    fn test_hypothesis_mutation_correlation() {
        let hypothesis = CoverageHypothesis::mutation_correlation(0.8);
        assert_eq!(hypothesis.name(), "H0-COV-04");

        // High coverage = good correlation estimate, not rejected
        let result = hypothesis.evaluate(&[95.0, 96.0, 94.0, 95.0, 95.0]);
        assert!(!result.rejected);
    }

    /// H₀-HYP-07: Empty observations rejected
    #[test]
    fn test_hypothesis_empty_observations() {
        let hypothesis = CoverageHypothesis::determinism();
        let result = hypothesis.evaluate(&[]);
        assert!(result.rejected);
        assert_eq!(result.effect_size, f64::INFINITY);
    }

    /// H₀-HYP-08: NullificationConfig custom
    #[test]
    fn test_nullification_config_custom() {
        let config = NullificationConfig::new(10, 0.01);
        assert_eq!(config.runs, 10);
        assert!((config.alpha - 0.01).abs() < 0.0001);
    }

    /// H₀-HYP-09: NullificationConfig default is princeton
    #[test]
    fn test_nullification_config_default() {
        let config = NullificationConfig::default();
        assert_eq!(config.runs, 5);
        assert!((config.alpha - 0.05).abs() < 0.001);
    }

    /// H₀-HYP-10: NullificationResult report format
    #[test]
    fn test_nullification_result_report() {
        let result = NullificationResult {
            hypothesis_name: "H0-COV-01".to_string(),
            rejected: true,
            p_value: 0.01,
            effect_size: 1.5,
            confidence_interval: (88.0, 92.0),
        };

        let report = result.report();
        assert!(report.contains("H0-COV-01"));
        assert!(report.contains("REJECTED"));
        assert!(report.contains("p=0.010"));
    }

    /// H₀-HYP-11: NullificationResult is_significant
    #[test]
    fn test_nullification_result_is_significant() {
        let significant = NullificationResult {
            hypothesis_name: "test".to_string(),
            rejected: true,
            p_value: 0.01,
            effect_size: 1.0,
            confidence_interval: (0.0, 0.0),
        };
        assert!(significant.is_significant());

        let not_significant = NullificationResult {
            hypothesis_name: "test".to_string(),
            rejected: false,
            p_value: 0.10,
            effect_size: 0.1,
            confidence_interval: (0.0, 0.0),
        };
        assert!(!not_significant.is_significant());
    }

    /// H₀-HYP-12: CoverageHypothesis debug
    #[test]
    fn test_coverage_hypothesis_debug() {
        let hypothesis = CoverageHypothesis::Completeness { threshold: 95.0 };
        let debug = format!("{:?}", hypothesis);
        assert!(debug.contains("Completeness"));
        assert!(debug.contains("95"));
    }
}

mod additional_collector_tests {
    use super::*;

    /// H₀-COLL-06: Granularity variants
    #[test]
    fn test_granularity_variants() {
        let g1 = Granularity::Function;
        let g2 = Granularity::BasicBlock;
        let g3 = Granularity::Edge;

        assert_eq!(g1, Granularity::Function);
        assert_ne!(g1, g2);
        assert_ne!(g2, g3);
    }

    /// H₀-COLL-07: Granularity debug
    #[test]
    fn test_granularity_debug() {
        let g = Granularity::BasicBlock;
        let debug = format!("{:?}", g);
        assert!(debug.contains("BasicBlock"));
    }

    /// H₀-COLL-08: CoverageConfig debug
    #[test]
    fn test_coverage_config_debug() {
        let config = CoverageConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("CoverageConfig"));
    }

    /// H₀-COLL-09: CoverageCollector session without active doesn't panic
    #[test]
    fn test_collector_end_without_begin() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        // End session without begin - should not panic
        let report = collector.end_session();
        assert_eq!(report.summary().total_blocks, 0);
    }

    /// H₀-COLL-10: CoverageCollector end test without begin doesn't panic
    #[test]
    fn test_collector_end_test_without_begin() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        collector.begin_session("test");
        // End test without begin - should not panic
        collector.end_test();
        let _ = collector.end_session();
    }
}

mod additional_report_tests {
    use super::*;

    /// H₀-RPT-06: CoverageReport debug
    #[test]
    fn test_coverage_report_debug() {
        let report = CoverageReport::new(5);
        let debug = format!("{:?}", report);
        assert!(debug.contains("CoverageReport"));
    }

    /// H₀-RPT-07: CoverageSummary debug
    #[test]
    fn test_coverage_summary_debug() {
        let report = CoverageReport::new(5);
        let summary = report.summary();
        let debug = format!("{:?}", summary);
        assert!(debug.contains("CoverageSummary"));
    }

    /// H₀-RPT-08: CoverageReport covered_blocks
    #[test]
    fn test_coverage_report_covered_blocks() {
        let mut report = CoverageReport::new(5);
        report.record_hit(BlockId::new(1));
        report.record_hit(BlockId::new(3));

        let covered = report.covered_blocks();
        assert_eq!(covered.len(), 2);
        assert!(covered.contains(&BlockId::new(1)));
        assert!(covered.contains(&BlockId::new(3)));
    }

    /// H₀-RPT-09: CoverageReport merge
    #[test]
    fn test_coverage_report_merge() {
        let mut report1 = CoverageReport::new(5);
        report1.record_hit(BlockId::new(0));
        report1.record_hit(BlockId::new(1));

        let mut report2 = CoverageReport::new(5);
        report2.record_hit(BlockId::new(2));
        report2.record_hit(BlockId::new(3));

        report1.merge(&report2);

        let summary = report1.summary();
        assert_eq!(summary.covered_blocks, 4);
    }

    /// H₀-RPT-10: BlockCoverage default values
    #[test]
    fn test_block_coverage_default() {
        let block_cov = BlockCoverage {
            block_id: BlockId::new(0),
            hit_count: 0,
            source_location: None,
            function_name: None,
        };

        assert!(block_cov.source_location.is_none());
        assert!(block_cov.function_name.is_none());
        assert_eq!(block_cov.hit_count, 0);
    }
}

mod additional_thread_local_tests {
    use super::*;

    /// H₀-TL-06: ThreadLocalCounters debug
    #[test]
    fn test_thread_local_debug() {
        let counters = ThreadLocalCounters::new(5);
        let debug = format!("{:?}", counters);
        assert!(debug.contains("ThreadLocalCounters"));
    }

    /// H₀-TL-07: Out of bounds get returns 0
    #[test]
    fn test_thread_local_out_of_bounds() {
        let counters = ThreadLocalCounters::new(5);
        // Block 99 is out of bounds - should return 0
        assert_eq!(counters.get(BlockId::new(99)), 0);
    }

    /// H₀-TL-08: Out of bounds increment doesn't panic
    #[test]
    fn test_thread_local_increment_out_of_bounds() {
        let mut counters = ThreadLocalCounters::new(5);
        // Should not panic, just silently ignored
        counters.increment(BlockId::new(99));
        assert_eq!(counters.get(BlockId::new(99)), 0);
    }
}

mod comprehensive_jidoka_tests {
    use super::*;

    /// H₀-JIDOKA-12: CoverageViolation affected_block for uninstrumented
    #[test]
    fn test_affected_block_uninstrumented() {
        let violation = CoverageViolation::UninstrumentedExecution {
            block_id: BlockId::new(42),
        };
        assert_eq!(violation.affected_block(), Some(BlockId::new(42)));
    }

    /// H₀-JIDOKA-13: CoverageViolation affected_block for overflow
    #[test]
    fn test_affected_block_overflow() {
        let violation = CoverageViolation::CounterOverflow {
            block_id: BlockId::new(99),
        };
        assert_eq!(violation.affected_block(), Some(BlockId::new(99)));
    }

    /// H₀-JIDOKA-14: CoverageViolation affected_block for impossible edge
    #[test]
    fn test_affected_block_impossible_edge() {
        let violation = CoverageViolation::ImpossibleEdge {
            from: BlockId::new(1),
            to: BlockId::new(2),
        };
        assert_eq!(violation.affected_block(), Some(BlockId::new(1)));
    }

    /// H₀-JIDOKA-15: CoverageViolation affected_block for regression (None)
    #[test]
    fn test_affected_block_regression() {
        let violation = CoverageViolation::CoverageRegression {
            expected: 95.0,
            actual: 90.0,
        };
        assert_eq!(violation.affected_block(), None);
    }

    /// H₀-JIDOKA-16: CoverageViolation description for uninstrumented
    #[test]
    fn test_description_uninstrumented() {
        let violation = CoverageViolation::UninstrumentedExecution {
            block_id: BlockId::new(42),
        };
        let desc = violation.description();
        assert!(desc.contains("42"));
        assert!(desc.contains("not instrumented"));
    }

    /// H₀-JIDOKA-17: CoverageViolation description for overflow
    #[test]
    fn test_description_overflow() {
        let violation = CoverageViolation::CounterOverflow {
            block_id: BlockId::new(99),
        };
        let desc = violation.description();
        assert!(desc.contains("99"));
        assert!(desc.contains("overflow"));
    }

    /// H₀-JIDOKA-18: CoverageViolation description for impossible edge
    #[test]
    fn test_description_impossible_edge() {
        let violation = CoverageViolation::ImpossibleEdge {
            from: BlockId::new(1),
            to: BlockId::new(2),
        };
        let desc = violation.description();
        assert!(desc.contains("1"));
        assert!(desc.contains("2"));
        assert!(desc.contains("Impossible"));
    }

    /// H₀-JIDOKA-19: CoverageViolation description for regression
    #[test]
    fn test_description_regression() {
        let violation = CoverageViolation::CoverageRegression {
            expected: 95.0,
            actual: 90.0,
        };
        let desc = violation.description();
        assert!(desc.contains("95"));
        assert!(desc.contains("90"));
        assert!(desc.contains("regression"));
    }

    /// H₀-JIDOKA-20: TaintedBlocks tainted_blocks accessor
    #[test]
    fn test_tainted_blocks_accessor() {
        let mut tainted = TaintedBlocks::new();
        tainted.taint(
            BlockId::new(1),
            CoverageViolation::CounterOverflow {
                block_id: BlockId::new(1),
            },
        );
        tainted.taint(
            BlockId::new(2),
            CoverageViolation::CounterOverflow {
                block_id: BlockId::new(2),
            },
        );

        let blocks = tainted.tainted_blocks();
        assert_eq!(blocks.len(), 2);
    }

    /// H₀-JIDOKA-21: TaintedBlocks violations_for specific block
    #[test]
    fn test_violations_for_block() {
        let mut tainted = TaintedBlocks::new();
        tainted.taint(
            BlockId::new(1),
            CoverageViolation::CounterOverflow {
                block_id: BlockId::new(1),
            },
        );
        tainted.taint(
            BlockId::new(1),
            CoverageViolation::UninstrumentedExecution {
                block_id: BlockId::new(1),
            },
        );
        tainted.taint(
            BlockId::new(2),
            CoverageViolation::CounterOverflow {
                block_id: BlockId::new(2),
            },
        );

        let violations = tainted.violations_for(BlockId::new(1));
        assert_eq!(violations.len(), 2);
    }

    /// H₀-JIDOKA-22: TaintedBlocks clear
    #[test]
    fn test_tainted_blocks_clear() {
        let mut tainted = TaintedBlocks::new();
        tainted.taint(
            BlockId::new(1),
            CoverageViolation::CounterOverflow {
                block_id: BlockId::new(1),
            },
        );

        assert_eq!(tainted.tainted_count(), 1);

        tainted.clear();
        assert_eq!(tainted.tainted_count(), 0);
        assert_eq!(tainted.violation_count(), 0);
    }

    /// H₀-JIDOKA-23: TaintedBlocks record_violation with block
    #[test]
    fn test_record_violation_with_block() {
        let mut tainted = TaintedBlocks::new();
        tainted.record_violation(CoverageViolation::CounterOverflow {
            block_id: BlockId::new(42),
        });

        assert!(tainted.is_tainted(BlockId::new(42)));
        assert_eq!(tainted.violation_count(), 1);
    }

    /// H₀-JIDOKA-24: TaintedBlocks record_violation without block
    #[test]
    fn test_record_violation_without_block() {
        let mut tainted = TaintedBlocks::new();
        tainted.record_violation(CoverageViolation::CoverageRegression {
            expected: 95.0,
            actual: 90.0,
        });

        assert_eq!(tainted.tainted_count(), 0);
        assert_eq!(tainted.violation_count(), 1);
    }

    /// H₀-JIDOKA-25: TaintedBlocks all_violations accessor
    #[test]
    fn test_all_violations_accessor() {
        let mut tainted = TaintedBlocks::new();
        tainted.taint(
            BlockId::new(1),
            CoverageViolation::CounterOverflow {
                block_id: BlockId::new(1),
            },
        );

        let violations = tainted.all_violations();
        assert_eq!(violations.len(), 1);
    }
}

mod comprehensive_memory_tests {
    use super::*;

    /// H₀-MEM-07: CoverageMemoryView read with large offset
    #[test]
    fn test_memory_view_large_offset() {
        let mut memory = vec![0u8; 200];

        // Write counter at offset 100
        memory[100..108].copy_from_slice(&999u64.to_le_bytes());

        let view = CoverageMemoryView::new(&memory, 100, 5);
        assert_eq!(view.read_counter(BlockId::new(0)), 999);
    }

    /// H₀-MEM-08: CoverageMemoryView multiple counters read
    #[test]
    fn test_memory_view_multiple_counters() {
        let mut memory = vec![0u8; 80]; // 10 counters

        for i in 0..10u64 {
            let offset = (i * 8) as usize;
            memory[offset..offset + 8].copy_from_slice(&(i * 10).to_le_bytes());
        }

        let view = CoverageMemoryView::new(&memory, 0, 10);
        let counters = view.read_all_counters();

        for i in 0..10 {
            assert_eq!(counters[i], (i as u64) * 10);
        }
    }
}

mod comprehensive_report_tests {
    use super::*;

    /// H₀-RPT-11: CoverageReport set_session_name
    #[test]
    fn test_report_set_session_name() {
        let mut report = CoverageReport::new(5);
        report.set_session_name("my_session");

        assert_eq!(report.session_name(), Some("my_session"));
    }

    /// H₀-RPT-12: CoverageReport add_test
    #[test]
    fn test_report_add_test() {
        let mut report = CoverageReport::new(5);
        report.add_test("test_1");
        report.add_test("test_2");

        assert_eq!(report.tests().len(), 2);
        assert_eq!(report.tests()[0], "test_1");
    }

    /// H₀-RPT-13: CoverageReport record_hits
    #[test]
    fn test_report_record_hits() {
        let mut report = CoverageReport::new(5);
        report.record_hits(BlockId::new(0), 10);

        assert_eq!(report.get_hit_count(BlockId::new(0)), 10);
    }

    /// H₀-RPT-14: CoverageReport record_violation
    #[test]
    fn test_report_record_violation() {
        let mut report = CoverageReport::new(5);
        report.record_violation(CoverageViolation::CounterOverflow {
            block_id: BlockId::new(0),
        });

        assert_eq!(report.violation_count(), 1);
    }

    /// H₀-RPT-15: CoverageReport set_source_location
    #[test]
    fn test_report_set_source_location() {
        let mut report = CoverageReport::new(5);
        report.set_source_location(BlockId::new(0), "src/main.rs:42");

        let coverages = report.block_coverages();
        assert_eq!(
            coverages[0].source_location,
            Some("src/main.rs:42".to_string())
        );
    }

    /// H₀-RPT-16: CoverageReport set_function_name
    #[test]
    fn test_report_set_function_name() {
        let mut report = CoverageReport::new(5);
        report.set_function_name(BlockId::new(0), "my_function");

        let coverages = report.block_coverages();
        assert_eq!(coverages[0].function_name, Some("my_function".to_string()));
    }

    /// H₀-RPT-17: CoverageReport is_covered
    #[test]
    fn test_report_is_covered() {
        let mut report = CoverageReport::new(5);
        report.record_hit(BlockId::new(0));

        assert!(report.is_covered(BlockId::new(0)));
        assert!(!report.is_covered(BlockId::new(1)));
    }

    /// H₀-RPT-18: CoverageReport coverage_percent
    #[test]
    fn test_report_coverage_percent_direct() {
        let mut report = CoverageReport::new(10);
        report.record_hit(BlockId::new(0));
        report.record_hit(BlockId::new(1));
        report.record_hit(BlockId::new(2));

        assert!((report.coverage_percent() - 30.0).abs() < 0.01);
    }

    /// H₀-RPT-19: CoverageReport total_blocks accessor
    #[test]
    fn test_report_total_blocks() {
        let report = CoverageReport::new(42);
        assert_eq!(report.total_blocks(), 42);
    }

    /// H₀-RPT-20: CoverageReport block_coverages
    #[test]
    fn test_report_block_coverages() {
        let mut report = CoverageReport::new(3);
        report.record_hit(BlockId::new(0));
        report.record_hits(BlockId::new(1), 5);

        let coverages = report.block_coverages();
        assert_eq!(coverages.len(), 3);
        assert_eq!(coverages[0].hit_count, 1);
        assert_eq!(coverages[1].hit_count, 5);
        assert_eq!(coverages[2].hit_count, 0);
    }

    /// H₀-RPT-21: CoverageReport is_tainted
    #[test]
    fn test_report_is_tainted() {
        let mut report = CoverageReport::new(5);
        report.record_violation(CoverageViolation::CounterOverflow {
            block_id: BlockId::new(0),
        });

        assert!(report.is_tainted(BlockId::new(0)));
        assert!(!report.is_tainted(BlockId::new(1)));
    }

    /// H₀-RPT-22: CoverageReport violations accessor
    #[test]
    fn test_report_violations_accessor() {
        let mut report = CoverageReport::new(5);
        report.record_violation(CoverageViolation::CounterOverflow {
            block_id: BlockId::new(0),
        });

        assert_eq!(report.violations().len(), 1);
    }

    /// H₀-RPT-23: CoverageReport default
    #[test]
    fn test_report_default() {
        let report = CoverageReport::default();
        assert_eq!(report.total_blocks(), 0);
    }

    /// H₀-RPT-24: CoverageReport merge with source locations
    #[test]
    fn test_report_merge_source_locations() {
        let mut report1 = CoverageReport::new(5);
        report1.set_source_location(BlockId::new(0), "a.rs:1");

        let mut report2 = CoverageReport::new(5);
        report2.set_source_location(BlockId::new(1), "b.rs:2");

        report1.merge(&report2);

        let coverages = report1.block_coverages();
        assert_eq!(coverages[0].source_location, Some("a.rs:1".to_string()));
        assert_eq!(coverages[1].source_location, Some("b.rs:2".to_string()));
    }

    /// H₀-RPT-25: CoverageReport merge with tests
    #[test]
    fn test_report_merge_tests() {
        let mut report1 = CoverageReport::new(5);
        report1.add_test("test_1");

        let mut report2 = CoverageReport::new(5);
        report2.add_test("test_2");

        report1.merge(&report2);

        assert_eq!(report1.tests().len(), 2);
    }
}

mod comprehensive_superblock_tests {
    use super::*;

    /// H₀-SB-14: Superblock set_cost_estimate
    #[test]
    fn test_superblock_set_cost_estimate() {
        let blocks = vec![BlockId::new(1), BlockId::new(2)];
        let mut superblock = Superblock::new(SuperblockId::new(0), blocks, FunctionId::new(0));

        superblock.set_cost_estimate(100);
        assert_eq!(superblock.cost_estimate(), 100);
    }

    /// H₀-SB-15: Superblock blocks accessor
    #[test]
    fn test_superblock_blocks_accessor() {
        let blocks = vec![BlockId::new(1), BlockId::new(2), BlockId::new(3)];
        let superblock = Superblock::new(SuperblockId::new(0), blocks.clone(), FunctionId::new(0));

        assert_eq!(superblock.blocks(), &blocks);
    }

    /// H₀-SB-16: SuperblockBuilder next_id
    #[test]
    fn test_superblock_builder_next_id() {
        let builder = SuperblockBuilder::new();
        assert_eq!(builder.next_id(), 0);
    }

    /// H₀-SB-17: SuperblockBuilder build_from_function_blocks
    #[test]
    fn test_superblock_builder_from_function_blocks() {
        let builder = SuperblockBuilder::new().with_target_size(2);

        let function_blocks = vec![
            (FunctionId::new(0), vec![BlockId::new(0), BlockId::new(1)]),
            (FunctionId::new(1), vec![BlockId::new(2), BlockId::new(3)]),
        ];

        let superblocks = builder.build_from_function_blocks(&function_blocks);
        assert_eq!(superblocks.len(), 2);
        assert_eq!(superblocks[0].function().as_u32(), 0);
        assert_eq!(superblocks[1].function().as_u32(), 1);
    }

    /// H₀-SB-18: Superblock clone
    #[test]
    fn test_superblock_clone() {
        let blocks = vec![BlockId::new(1)];
        let sb1 = Superblock::new(SuperblockId::new(0), blocks, FunctionId::new(0));
        let sb2 = sb1.clone();

        assert_eq!(sb1.id(), sb2.id());
        assert_eq!(sb1.block_count(), sb2.block_count());
    }
}

mod comprehensive_collector_tests {
    use super::*;

    /// H₀-COLL-11: CoverageConfig granularity edge
    #[test]
    fn test_coverage_config_edge_granularity() {
        let config = CoverageConfig::builder()
            .granularity(Granularity::Edge)
            .build();

        assert_eq!(config.granularity, Granularity::Edge);
    }

    /// H₀-COLL-12: Granularity clone
    #[test]
    fn test_granularity_clone() {
        let g1 = Granularity::BasicBlock;
        let g2 = g1.clone();
        assert_eq!(g1, g2);
    }

    /// H₀-COLL-13: CoverageConfig clone
    #[test]
    fn test_coverage_config_clone() {
        let config1 = CoverageConfig::builder().parallel(true).build();
        let config2 = config1.clone();
        assert_eq!(config1.parallel, config2.parallel);
    }

    /// H₀-COLL-14: CoverageConfig checkpoint_interval
    #[test]
    fn test_coverage_config_checkpoint_interval() {
        let config = CoverageConfig::builder().checkpoint_interval(30).build();

        assert_eq!(config.checkpoint_interval, Some(30));
    }

    /// H₀-COLL-15: CoverageConfig max_blocks
    #[test]
    fn test_coverage_config_max_blocks() {
        let config = CoverageConfig::builder().max_blocks(5000).build();

        assert_eq!(config.max_blocks, 5000);
    }

    /// H₀-COLL-16: CoverageConfig max_blocks zero becomes default
    #[test]
    fn test_coverage_config_max_blocks_zero() {
        let config = CoverageConfig::builder().max_blocks(0).build();

        assert_eq!(config.max_blocks, 100_000);
    }

    /// H₀-COLL-17: CoverageCollector is_session_active
    #[test]
    fn test_collector_is_session_active() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        assert!(!collector.is_session_active());
        collector.begin_session("test");
        assert!(collector.is_session_active());
        let _ = collector.end_session();
        assert!(!collector.is_session_active());
    }

    /// H₀-COLL-18: CoverageCollector is_test_active
    #[test]
    fn test_collector_is_test_active() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        collector.begin_session("test");
        assert!(!collector.is_test_active());
        collector.begin_test("my_test");
        assert!(collector.is_test_active());
        collector.end_test();
        assert!(!collector.is_test_active());
    }

    /// H₀-COLL-19: CoverageCollector current_test
    #[test]
    fn test_collector_current_test() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        collector.begin_session("session");
        assert!(collector.current_test().is_none());
        collector.begin_test("my_test");
        assert_eq!(collector.current_test(), Some("my_test"));
        collector.end_test();
        assert!(collector.current_test().is_none());
    }

    /// H₀-COLL-20: CoverageCollector config accessor
    #[test]
    fn test_collector_config_accessor() {
        let config = CoverageConfig::builder()
            .granularity(Granularity::Edge)
            .parallel(true)
            .build();

        let collector = CoverageCollector::new(config);

        assert_eq!(collector.config().granularity, Granularity::Edge);
        assert!(collector.config().parallel);
    }

    /// H₀-COLL-21: CoverageCollector record_violation with Stop
    #[test]
    fn test_collector_record_stop_violation() {
        let config = CoverageConfig::default();
        let mut collector = CoverageCollector::new(config);

        collector.begin_session("test");
        collector.record_violation(CoverageViolation::UninstrumentedExecution {
            block_id: BlockId::new(0),
        });

        let report = collector.end_session();
        assert_eq!(report.violation_count(), 1);
    }

    /// H₀-COLL-22: CoverageCollector jidoka disabled
    #[test]
    fn test_collector_jidoka_disabled() {
        let config = CoverageConfig::builder().jidoka_enabled(false).build();
        let mut collector = CoverageCollector::new(config);

        collector.begin_session("test");
        collector.record_violation(CoverageViolation::CounterOverflow {
            block_id: BlockId::new(0),
        });

        let report = collector.end_session();
        // Violation not recorded when jidoka disabled
        assert_eq!(report.violation_count(), 0);
    }

    /// H₀-COLL-23: Granularity Path variant
    #[test]
    fn test_granularity_path() {
        let g = Granularity::Path;
        let debug = format!("{:?}", g);
        assert!(debug.contains("Path"));
    }

    /// H₀-COLL-24: CoverageConfig default values
    #[test]
    fn test_coverage_config_defaults() {
        let config = CoverageConfig::default();

        assert_eq!(config.granularity, Granularity::Function);
        assert!(!config.parallel);
        assert!(config.jidoka_enabled);
        assert!(config.checkpoint_interval.is_none());
        assert_eq!(config.max_blocks, 100_000);
    }
}

mod comprehensive_memory_coverage_tests {
    use super::*;

    /// H₀-MEM-09: CoverageMemoryView counter_base accessor
    #[test]
    fn test_memory_view_counter_base() {
        let memory = vec![0u8; 100];
        let view = CoverageMemoryView::new(&memory, 16, 5);
        assert_eq!(view.counter_base(), 16);
    }

    /// H₀-MEM-10: CoverageMemoryView memory_size accessor
    #[test]
    fn test_memory_view_memory_size() {
        let memory = vec![0u8; 200];
        let view = CoverageMemoryView::new(&memory, 0, 10);
        assert_eq!(view.memory_size(), 200);
    }

    /// H₀-MEM-11: CoverageMemoryView is_covered
    #[test]
    fn test_memory_view_is_covered() {
        let mut memory = vec![0u8; 80];
        // Set block 2 to have count of 5
        memory[16..24].copy_from_slice(&5u64.to_le_bytes());

        let view = CoverageMemoryView::new(&memory, 0, 10);
        assert!(!view.is_covered(BlockId::new(0)));
        assert!(!view.is_covered(BlockId::new(1)));
        assert!(view.is_covered(BlockId::new(2)));
    }

    /// H₀-MEM-12: CoverageMemoryView covered_count
    #[test]
    fn test_memory_view_covered_count() {
        let mut memory = vec![0u8; 80];
        // Set blocks 0, 2, 5 as covered
        memory[0..8].copy_from_slice(&1u64.to_le_bytes());
        memory[16..24].copy_from_slice(&3u64.to_le_bytes());
        memory[40..48].copy_from_slice(&1u64.to_le_bytes());

        let view = CoverageMemoryView::new(&memory, 0, 10);
        assert_eq!(view.covered_count(), 3);
    }

    /// H₀-MEM-13: CoverageMemoryView coverage_percent
    #[test]
    fn test_memory_view_coverage_percent() {
        let mut memory = vec![0u8; 80];
        // Set 5 of 10 blocks as covered
        for i in 0..5 {
            let offset = i * 8;
            memory[offset..offset + 8].copy_from_slice(&1u64.to_le_bytes());
        }

        let view = CoverageMemoryView::new(&memory, 0, 10);
        assert!((view.coverage_percent() - 50.0).abs() < 0.01);
    }

    /// H₀-MEM-14: CoverageMemoryView coverage_percent empty
    #[test]
    fn test_memory_view_coverage_percent_empty() {
        let memory = vec![];
        let view = CoverageMemoryView::new(&memory, 0, 0);
        assert!((view.coverage_percent() - 100.0).abs() < 0.01);
    }

    /// H₀-MEM-15: CoverageMemoryView read_counter out of bounds
    #[test]
    fn test_memory_view_read_counter_out_of_bounds() {
        let memory = vec![0u8; 40];
        let view = CoverageMemoryView::new(&memory, 0, 5);

        // Block 10 is out of range
        assert_eq!(view.read_counter(BlockId::new(10)), 0);
    }

    /// H₀-MEM-16: CoverageMemoryView read_counter memory overflow
    #[test]
    fn test_memory_view_read_counter_memory_overflow() {
        let memory = vec![0u8; 20]; // Only enough for ~2 counters
        let view = CoverageMemoryView::new(&memory, 0, 10); // Claims 10 blocks

        // Block 3 would overflow memory
        assert_eq!(view.read_counter(BlockId::new(3)), 0);
    }
}

mod comprehensive_thread_local_coverage_tests {
    use super::*;

    /// H₀-TL-09: ThreadLocalCounters flush resets counters
    #[test]
    fn test_thread_local_flush_resets() {
        let mut counters = ThreadLocalCounters::new(5);
        counters.increment(BlockId::new(0));
        counters.increment(BlockId::new(1));

        let flushed = counters.flush();

        // After flush, counters should be zero
        assert_eq!(counters.get(BlockId::new(0)), 0);
        assert_eq!(counters.get(BlockId::new(1)), 0);
        // But flushed should contain the values
        assert_eq!(flushed[0], 1);
        assert_eq!(flushed[1], 1);
    }

    /// H₀-TL-10: ThreadLocalCounters with_flush_threshold
    #[test]
    fn test_thread_local_with_flush_threshold() {
        let counters = ThreadLocalCounters::with_flush_threshold(10, 50);

        // Just verify it doesn't panic
        assert_eq!(counters.get(BlockId::new(0)), 0);
        assert_eq!(counters.flush_threshold(), 50);
    }

    /// H₀-TL-11: ThreadLocalCounters block_count
    #[test]
    fn test_thread_local_block_count() {
        let counters = ThreadLocalCounters::new(42);
        assert_eq!(counters.block_count(), 42);
    }

    /// H₀-TL-12: ThreadLocalCounters has_pending
    #[test]
    fn test_thread_local_has_pending() {
        let mut counters = ThreadLocalCounters::new(5);
        assert!(!counters.has_pending());

        counters.increment(BlockId::new(0));
        assert!(counters.has_pending());

        let _ = counters.flush();
        assert!(!counters.has_pending());
    }

    /// H₀-TL-13: ThreadLocalCounters default
    #[test]
    fn test_thread_local_default() {
        let counters = ThreadLocalCounters::default();
        assert_eq!(counters.block_count(), 0);
    }

    /// H₀-TL-14: ThreadLocalCounters auto flush at threshold
    #[test]
    fn test_thread_local_auto_flush_at_threshold() {
        let mut counters = ThreadLocalCounters::with_flush_threshold(5, 3);

        // Increment 3 times to trigger auto-flush
        counters.increment(BlockId::new(0));
        counters.increment(BlockId::new(1));
        counters.increment(BlockId::new(2));

        // Auto-flush should have been triggered
        assert!(counters.flush_count() >= 1);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    /// H₀-INT-01: Full coverage workflow works end-to-end
    #[test]
    fn test_full_coverage_workflow() {
        // Setup
        let config = CoverageConfig::builder()
            .granularity(Granularity::Function)
            .jidoka_enabled(true)
            .build();
        let mut collector = CoverageCollector::new(config);

        // Begin session
        collector.begin_session("pong_tests");

        // Test 1
        collector.begin_test("test_ball_movement");
        collector.record_hit(BlockId::new(0)); // update_ball
        collector.record_hit(BlockId::new(1)); // check_bounds
        collector.record_hit(BlockId::new(0)); // update_ball again
        collector.end_test();

        // Test 2
        collector.begin_test("test_paddle_input");
        collector.record_hit(BlockId::new(2)); // handle_input
        collector.record_hit(BlockId::new(3)); // update_paddle
        collector.end_test();

        // End session and get report
        let report = collector.end_session();
        let summary = report.summary();

        // Verify
        assert_eq!(summary.covered_blocks, 4);
        assert_eq!(report.get_hit_count(BlockId::new(0)), 2);
        assert_eq!(report.get_hit_count(BlockId::new(1)), 1);
    }

    /// H₀-INT-02: Jidoka violations are tracked correctly
    #[test]
    fn test_jidoka_integration() {
        let config = CoverageConfig::builder().jidoka_enabled(true).build();
        let mut collector = CoverageCollector::new(config);

        collector.begin_session("jidoka_test");
        collector.begin_test("test_with_violation");

        // Simulate a soft violation
        collector.record_violation(CoverageViolation::CoverageRegression {
            expected: 95.0,
            actual: 90.0,
        });

        collector.end_test();
        let report = collector.end_session();

        // Should have recorded the violation but continued
        assert_eq!(report.violation_count(), 1);
    }
}
