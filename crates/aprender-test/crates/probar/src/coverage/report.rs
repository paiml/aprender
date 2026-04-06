//! Coverage Report Generation
//!
//! Per spec §9.1 and Appendix B: Coverage Report Schema
//!
//! Generates comprehensive coverage reports including:
//! - Block-level coverage data
//! - Summary statistics
//! - Source location mapping
//! - Nullification test results

use super::{BlockId, CoverageViolation, TaintedBlocks};
use std::collections::HashMap;

/// Coverage summary statistics
#[derive(Debug, Clone)]
pub struct CoverageSummary {
    /// Total number of blocks
    pub total_blocks: usize,
    /// Number of covered blocks (hit_count > 0)
    pub covered_blocks: usize,
    /// Coverage percentage
    pub coverage_percent: f64,
    /// 95% confidence interval (for multiple runs)
    pub confidence_interval: Option<(f64, f64)>,
    /// Effect size (Cohen's d)
    pub effect_size: Option<f64>,
}

/// Per-block coverage information
#[derive(Debug, Clone)]
pub struct BlockCoverage {
    /// Block identifier
    pub block_id: BlockId,
    /// Number of times this block was hit
    pub hit_count: u64,
    /// Source location (e.g., "src/pong.rs:142")
    pub source_location: Option<String>,
    /// Function name containing this block
    pub function_name: Option<String>,
}

/// Coverage report containing all coverage data
#[derive(Debug)]
pub struct CoverageReport {
    /// Total number of blocks
    total_blocks: usize,
    /// Hit counts per block
    hit_counts: HashMap<BlockId, u64>,
    /// Source locations per block
    source_locations: HashMap<BlockId, String>,
    /// Function names per block
    function_names: HashMap<BlockId, String>,
    /// Tainted blocks tracker
    tainted: TaintedBlocks,
    /// Session name
    session_name: Option<String>,
    /// Tests run in this session
    tests: Vec<String>,
}

impl CoverageReport {
    /// Create a new coverage report for the given number of blocks
    #[must_use]
    pub fn new(total_blocks: usize) -> Self {
        Self {
            total_blocks,
            hit_counts: HashMap::new(),
            source_locations: HashMap::new(),
            function_names: HashMap::new(),
            tainted: TaintedBlocks::new(),
            session_name: None,
            tests: Vec::new(),
        }
    }

    /// Set the session name
    pub fn set_session_name(&mut self, name: &str) {
        self.session_name = Some(name.to_string());
    }

    /// Add a test to the report
    pub fn add_test(&mut self, name: &str) {
        self.tests.push(name.to_string());
    }

    /// Record a hit on a block
    pub fn record_hit(&mut self, block: BlockId) {
        *self.hit_counts.entry(block).or_insert(0) += 1;
    }

    /// Record multiple hits on a block
    pub fn record_hits(&mut self, block: BlockId, count: u64) {
        *self.hit_counts.entry(block).or_insert(0) += count;
    }

    /// Record a violation
    pub fn record_violation(&mut self, violation: CoverageViolation) {
        self.tainted.record_violation(violation);
    }

    /// Set source location for a block
    pub fn set_source_location(&mut self, block: BlockId, location: &str) {
        let _ = self.source_locations.insert(block, location.to_string());
    }

    /// Set function name for a block
    pub fn set_function_name(&mut self, block: BlockId, name: &str) {
        let _ = self.function_names.insert(block, name.to_string());
    }

    /// Get the hit count for a block
    #[must_use]
    pub fn get_hit_count(&self, block: BlockId) -> u64 {
        self.hit_counts.get(&block).copied().unwrap_or(0)
    }

    /// Check if a block is covered
    #[must_use]
    pub fn is_covered(&self, block: BlockId) -> bool {
        self.get_hit_count(block) > 0
    }

    /// Get the number of covered blocks (only counts blocks in 0..total_blocks)
    #[must_use]
    pub fn covered_count(&self) -> usize {
        (0..self.total_blocks as u32)
            .map(BlockId::new)
            .filter(|b| self.is_covered(*b))
            .count()
    }

    /// Get the coverage percentage
    #[must_use]
    pub fn coverage_percent(&self) -> f64 {
        if self.total_blocks == 0 {
            return 100.0; // Vacuously true
        }
        (self.covered_count() as f64 / self.total_blocks as f64) * 100.0
    }

    /// Get all uncovered blocks
    #[must_use]
    pub fn uncovered_blocks(&self) -> Vec<BlockId> {
        (0..self.total_blocks as u32)
            .map(BlockId::new)
            .filter(|b| !self.is_covered(*b))
            .collect()
    }

    /// Get all covered blocks
    #[must_use]
    pub fn covered_blocks(&self) -> Vec<BlockId> {
        (0..self.total_blocks as u32)
            .map(BlockId::new)
            .filter(|b| self.is_covered(*b))
            .collect()
    }

    /// Get coverage summary
    #[must_use]
    pub fn summary(&self) -> CoverageSummary {
        CoverageSummary {
            total_blocks: self.total_blocks,
            covered_blocks: self.covered_count(),
            coverage_percent: self.coverage_percent(),
            confidence_interval: None,
            effect_size: None,
        }
    }

    /// Get block coverage details
    #[must_use]
    pub fn block_coverages(&self) -> Vec<BlockCoverage> {
        (0..self.total_blocks as u32)
            .map(|i| {
                let block_id = BlockId::new(i);
                BlockCoverage {
                    block_id,
                    hit_count: self.get_hit_count(block_id),
                    source_location: self.source_locations.get(&block_id).cloned(),
                    function_name: self.function_names.get(&block_id).cloned(),
                }
            })
            .collect()
    }

    /// Get the number of violations recorded
    #[must_use]
    pub fn violation_count(&self) -> usize {
        self.tainted.violation_count()
    }

    /// Get all violations
    #[must_use]
    pub fn violations(&self) -> &[CoverageViolation] {
        self.tainted.all_violations()
    }

    /// Check if a block is tainted
    #[must_use]
    pub fn is_tainted(&self, block: BlockId) -> bool {
        self.tainted.is_tainted(block)
    }

    /// Get the total number of blocks
    #[must_use]
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Get the session name
    #[must_use]
    pub fn session_name(&self) -> Option<&str> {
        self.session_name.as_deref()
    }

    /// Get the list of tests
    #[must_use]
    pub fn tests(&self) -> &[String] {
        &self.tests
    }

    /// Merge another report into this one
    pub fn merge(&mut self, other: &CoverageReport) {
        for (block, count) in &other.hit_counts {
            self.record_hits(*block, *count);
        }
        for (block, location) in &other.source_locations {
            if !self.source_locations.contains_key(block) {
                let _ = self.source_locations.insert(*block, location.clone());
            }
        }
        for (block, name) in &other.function_names {
            if !self.function_names.contains_key(block) {
                let _ = self.function_names.insert(*block, name.clone());
            }
        }
        for test in &other.tests {
            if !self.tests.contains(test) {
                self.tests.push(test.clone());
            }
        }
    }
}

impl Default for CoverageReport {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ============================================================================
    // CoverageReport Tests
    // ============================================================================

    /// Test report creation with various block counts
    #[test]
    fn test_report_new_various_sizes() {
        let report_zero = CoverageReport::new(0);
        assert_eq!(report_zero.total_blocks(), 0);

        let report_one = CoverageReport::new(1);
        assert_eq!(report_one.total_blocks(), 1);

        let report_large = CoverageReport::new(10000);
        assert_eq!(report_large.total_blocks(), 10000);
    }

    /// Test session name getter when None
    #[test]
    fn test_report_session_name_none() {
        let report = CoverageReport::new(5);
        assert!(report.session_name().is_none());
    }

    /// Test session name getter after setting
    #[test]
    fn test_report_session_name_set() {
        let mut report = CoverageReport::new(5);
        report.set_session_name("test_session");
        assert_eq!(report.session_name(), Some("test_session"));
    }

    /// Test tests accessor when empty
    #[test]
    fn test_report_tests_empty() {
        let report = CoverageReport::new(5);
        assert!(report.tests().is_empty());
    }

    /// Test adding multiple tests
    #[test]
    fn test_report_add_multiple_tests() {
        let mut report = CoverageReport::new(5);
        report.add_test("test_1");
        report.add_test("test_2");
        report.add_test("test_3");

        let tests = report.tests();
        assert_eq!(tests.len(), 3);
        assert_eq!(tests[0], "test_1");
        assert_eq!(tests[1], "test_2");
        assert_eq!(tests[2], "test_3");
    }

    /// Test record_hit increments existing count
    #[test]
    fn test_report_record_hit_increments() {
        let mut report = CoverageReport::new(5);
        report.record_hit(BlockId::new(0));
        assert_eq!(report.get_hit_count(BlockId::new(0)), 1);

        report.record_hit(BlockId::new(0));
        assert_eq!(report.get_hit_count(BlockId::new(0)), 2);

        report.record_hit(BlockId::new(0));
        assert_eq!(report.get_hit_count(BlockId::new(0)), 3);
    }

    /// Test record_hits adds multiple at once
    #[test]
    fn test_report_record_hits_bulk() {
        let mut report = CoverageReport::new(5);
        report.record_hits(BlockId::new(0), 100);
        assert_eq!(report.get_hit_count(BlockId::new(0)), 100);

        report.record_hits(BlockId::new(0), 50);
        assert_eq!(report.get_hit_count(BlockId::new(0)), 150);
    }

    /// Test get_hit_count for unhit block
    #[test]
    fn test_report_get_hit_count_unhit() {
        let report = CoverageReport::new(10);
        assert_eq!(report.get_hit_count(BlockId::new(5)), 0);
        assert_eq!(report.get_hit_count(BlockId::new(9)), 0);
    }

    /// Test is_covered for various states
    #[test]
    fn test_report_is_covered() {
        let mut report = CoverageReport::new(5);
        assert!(!report.is_covered(BlockId::new(0)));

        report.record_hit(BlockId::new(0));
        assert!(report.is_covered(BlockId::new(0)));
        assert!(!report.is_covered(BlockId::new(1)));
    }

    /// Test covered_count accuracy
    #[test]
    fn test_report_covered_count() {
        let mut report = CoverageReport::new(10);
        assert_eq!(report.covered_count(), 0);

        report.record_hit(BlockId::new(0));
        assert_eq!(report.covered_count(), 1);

        report.record_hit(BlockId::new(0)); // Same block
        assert_eq!(report.covered_count(), 1);

        report.record_hit(BlockId::new(5));
        assert_eq!(report.covered_count(), 2);
    }

    /// Test coverage_percent with zero blocks (vacuous truth)
    #[test]
    fn test_report_coverage_percent_zero_blocks() {
        let report = CoverageReport::new(0);
        assert!((report.coverage_percent() - 100.0).abs() < 0.001);
    }

    /// Test coverage_percent with no hits
    #[test]
    fn test_report_coverage_percent_no_hits() {
        let report = CoverageReport::new(10);
        assert!((report.coverage_percent() - 0.0).abs() < 0.001);
    }

    /// Test coverage_percent with full coverage
    #[test]
    fn test_report_coverage_percent_full() {
        let mut report = CoverageReport::new(5);
        for i in 0..5 {
            report.record_hit(BlockId::new(i));
        }
        assert!((report.coverage_percent() - 100.0).abs() < 0.001);
    }

    /// Test coverage_percent with partial coverage
    #[test]
    fn test_report_coverage_percent_partial() {
        let mut report = CoverageReport::new(4);
        report.record_hit(BlockId::new(0));
        report.record_hit(BlockId::new(1));
        assert!((report.coverage_percent() - 50.0).abs() < 0.001);
    }

    /// Test uncovered_blocks when all uncovered
    #[test]
    fn test_report_uncovered_blocks_all() {
        let report = CoverageReport::new(3);
        let uncovered = report.uncovered_blocks();
        assert_eq!(uncovered.len(), 3);
        assert!(uncovered.contains(&BlockId::new(0)));
        assert!(uncovered.contains(&BlockId::new(1)));
        assert!(uncovered.contains(&BlockId::new(2)));
    }

    /// Test uncovered_blocks when all covered
    #[test]
    fn test_report_uncovered_blocks_none() {
        let mut report = CoverageReport::new(3);
        for i in 0..3 {
            report.record_hit(BlockId::new(i));
        }
        let uncovered = report.uncovered_blocks();
        assert!(uncovered.is_empty());
    }

    /// Test covered_blocks when none covered
    #[test]
    fn test_report_covered_blocks_none() {
        let report = CoverageReport::new(5);
        let covered = report.covered_blocks();
        assert!(covered.is_empty());
    }

    /// Test covered_blocks when all covered
    #[test]
    fn test_report_covered_blocks_all() {
        let mut report = CoverageReport::new(3);
        for i in 0..3 {
            report.record_hit(BlockId::new(i));
        }
        let covered = report.covered_blocks();
        assert_eq!(covered.len(), 3);
    }

    /// Test summary returns correct values
    #[test]
    fn test_report_summary() {
        let mut report = CoverageReport::new(10);
        report.record_hit(BlockId::new(0));
        report.record_hit(BlockId::new(1));

        let summary = report.summary();
        assert_eq!(summary.total_blocks, 10);
        assert_eq!(summary.covered_blocks, 2);
        assert!((summary.coverage_percent - 20.0).abs() < 0.001);
        assert!(summary.confidence_interval.is_none());
        assert!(summary.effect_size.is_none());
    }

    /// Test block_coverages returns all blocks
    #[test]
    fn test_report_block_coverages_all() {
        let mut report = CoverageReport::new(3);
        report.record_hit(BlockId::new(0));
        report.record_hits(BlockId::new(1), 5);
        report.set_source_location(BlockId::new(0), "test.rs:1");
        report.set_function_name(BlockId::new(0), "test_fn");

        let coverages = report.block_coverages();
        assert_eq!(coverages.len(), 3);

        assert_eq!(coverages[0].block_id, BlockId::new(0));
        assert_eq!(coverages[0].hit_count, 1);
        assert_eq!(coverages[0].source_location, Some("test.rs:1".to_string()));
        assert_eq!(coverages[0].function_name, Some("test_fn".to_string()));

        assert_eq!(coverages[1].hit_count, 5);
        assert!(coverages[1].source_location.is_none());
        assert!(coverages[1].function_name.is_none());

        assert_eq!(coverages[2].hit_count, 0);
    }

    /// Test violation_count after recording
    #[test]
    fn test_report_violation_count() {
        let mut report = CoverageReport::new(5);
        assert_eq!(report.violation_count(), 0);

        report.record_violation(CoverageViolation::CounterOverflow {
            block_id: BlockId::new(0),
        });
        assert_eq!(report.violation_count(), 1);
    }

    /// Test violations accessor
    #[test]
    fn test_report_violations() {
        let mut report = CoverageReport::new(5);
        report.record_violation(CoverageViolation::CounterOverflow {
            block_id: BlockId::new(0),
        });

        let violations = report.violations();
        assert_eq!(violations.len(), 1);
    }

    /// Test is_tainted
    #[test]
    fn test_report_is_tainted() {
        let mut report = CoverageReport::new(5);
        assert!(!report.is_tainted(BlockId::new(0)));

        report.record_violation(CoverageViolation::CounterOverflow {
            block_id: BlockId::new(0),
        });
        assert!(report.is_tainted(BlockId::new(0)));
        assert!(!report.is_tainted(BlockId::new(1)));
    }

    // ============================================================================
    // Merge Tests - Critical for Coverage
    // ============================================================================

    /// Test merge combines hit counts
    #[test]
    fn test_merge_hit_counts() {
        let mut report1 = CoverageReport::new(5);
        report1.record_hit(BlockId::new(0));
        report1.record_hit(BlockId::new(1));

        let mut report2 = CoverageReport::new(5);
        report2.record_hit(BlockId::new(1));
        report2.record_hit(BlockId::new(2));
        report2.record_hits(BlockId::new(0), 10);

        report1.merge(&report2);

        assert_eq!(report1.get_hit_count(BlockId::new(0)), 11); // 1 + 10
        assert_eq!(report1.get_hit_count(BlockId::new(1)), 2); // 1 + 1
        assert_eq!(report1.get_hit_count(BlockId::new(2)), 1);
    }

    /// Test merge does NOT overwrite existing source locations
    #[test]
    fn test_merge_source_locations_no_overwrite() {
        let mut report1 = CoverageReport::new(5);
        report1.set_source_location(BlockId::new(0), "original.rs:1");

        let mut report2 = CoverageReport::new(5);
        report2.set_source_location(BlockId::new(0), "overwrite.rs:1"); // Should NOT overwrite
        report2.set_source_location(BlockId::new(1), "new.rs:1"); // Should be added

        report1.merge(&report2);

        let coverages = report1.block_coverages();
        // Block 0 should keep original location
        assert_eq!(
            coverages[0].source_location,
            Some("original.rs:1".to_string())
        );
        // Block 1 should have new location
        assert_eq!(coverages[1].source_location, Some("new.rs:1".to_string()));
    }

    /// Test merge does NOT overwrite existing function names
    #[test]
    fn test_merge_function_names_no_overwrite() {
        let mut report1 = CoverageReport::new(5);
        report1.set_function_name(BlockId::new(0), "original_fn");

        let mut report2 = CoverageReport::new(5);
        report2.set_function_name(BlockId::new(0), "overwrite_fn"); // Should NOT overwrite
        report2.set_function_name(BlockId::new(1), "new_fn"); // Should be added

        report1.merge(&report2);

        let coverages = report1.block_coverages();
        // Block 0 should keep original name
        assert_eq!(coverages[0].function_name, Some("original_fn".to_string()));
        // Block 1 should have new name
        assert_eq!(coverages[1].function_name, Some("new_fn".to_string()));
    }

    /// Test merge does NOT add duplicate test names
    #[test]
    fn test_merge_tests_no_duplicates() {
        let mut report1 = CoverageReport::new(5);
        report1.add_test("test_1");
        report1.add_test("test_2");

        let mut report2 = CoverageReport::new(5);
        report2.add_test("test_2"); // Duplicate - should NOT be added
        report2.add_test("test_3"); // New - should be added

        report1.merge(&report2);

        let tests = report1.tests();
        assert_eq!(tests.len(), 3);
        assert!(tests.contains(&"test_1".to_string()));
        assert!(tests.contains(&"test_2".to_string()));
        assert!(tests.contains(&"test_3".to_string()));
    }

    /// Test merge adds tests from empty to populated
    #[test]
    fn test_merge_tests_from_empty() {
        let mut report1 = CoverageReport::new(5);

        let mut report2 = CoverageReport::new(5);
        report2.add_test("test_a");
        report2.add_test("test_b");

        report1.merge(&report2);

        assert_eq!(report1.tests().len(), 2);
    }

    /// Test merge with empty other report
    #[test]
    fn test_merge_empty_other() {
        let mut report1 = CoverageReport::new(5);
        report1.record_hit(BlockId::new(0));
        report1.add_test("test_1");

        let report2 = CoverageReport::new(5);

        report1.merge(&report2);

        assert_eq!(report1.get_hit_count(BlockId::new(0)), 1);
        assert_eq!(report1.tests().len(), 1);
    }

    /// Test merge into empty report
    #[test]
    fn test_merge_into_empty() {
        let mut report1 = CoverageReport::new(5);

        let mut report2 = CoverageReport::new(5);
        report2.record_hit(BlockId::new(0));
        report2.set_source_location(BlockId::new(0), "test.rs:1");
        report2.set_function_name(BlockId::new(0), "test_fn");
        report2.add_test("test_1");

        report1.merge(&report2);

        assert_eq!(report1.get_hit_count(BlockId::new(0)), 1);
        let coverages = report1.block_coverages();
        assert_eq!(coverages[0].source_location, Some("test.rs:1".to_string()));
        assert_eq!(coverages[0].function_name, Some("test_fn".to_string()));
        assert_eq!(report1.tests().len(), 1);
    }

    // ============================================================================
    // CoverageSummary Tests
    // ============================================================================

    /// Test CoverageSummary clone
    #[test]
    fn test_coverage_summary_clone() {
        let summary1 = CoverageSummary {
            total_blocks: 100,
            covered_blocks: 80,
            coverage_percent: 80.0,
            confidence_interval: Some((78.0, 82.0)),
            effect_size: Some(0.5),
        };

        let summary2 = summary1;

        assert_eq!(summary2.total_blocks, 100);
        assert_eq!(summary2.covered_blocks, 80);
        assert!((summary2.coverage_percent - 80.0).abs() < 0.001);
        assert_eq!(summary2.confidence_interval, Some((78.0, 82.0)));
        assert_eq!(summary2.effect_size, Some(0.5));
    }

    /// Test CoverageSummary debug format
    #[test]
    fn test_coverage_summary_debug() {
        let summary = CoverageSummary {
            total_blocks: 10,
            covered_blocks: 5,
            coverage_percent: 50.0,
            confidence_interval: None,
            effect_size: None,
        };

        let debug = format!("{:?}", summary);
        assert!(debug.contains("CoverageSummary"));
        assert!(debug.contains("10"));
        assert!(debug.contains("50"));
    }

    // ============================================================================
    // BlockCoverage Tests
    // ============================================================================

    /// Test BlockCoverage clone
    #[test]
    fn test_block_coverage_clone() {
        let bc1 = BlockCoverage {
            block_id: BlockId::new(42),
            hit_count: 100,
            source_location: Some("test.rs:42".to_string()),
            function_name: Some("my_function".to_string()),
        };

        let bc2 = bc1;

        assert_eq!(bc2.block_id, BlockId::new(42));
        assert_eq!(bc2.hit_count, 100);
        assert_eq!(bc2.source_location, Some("test.rs:42".to_string()));
        assert_eq!(bc2.function_name, Some("my_function".to_string()));
    }

    /// Test BlockCoverage debug format
    #[test]
    fn test_block_coverage_debug() {
        let bc = BlockCoverage {
            block_id: BlockId::new(1),
            hit_count: 5,
            source_location: None,
            function_name: None,
        };

        let debug = format!("{:?}", bc);
        assert!(debug.contains("BlockCoverage"));
    }

    /// Test BlockCoverage with all None fields
    #[test]
    fn test_block_coverage_all_none() {
        let bc = BlockCoverage {
            block_id: BlockId::new(0),
            hit_count: 0,
            source_location: None,
            function_name: None,
        };

        assert!(bc.source_location.is_none());
        assert!(bc.function_name.is_none());
    }

    // ============================================================================
    // Default Implementation Test
    // ============================================================================

    /// Test Default creates report with zero blocks
    #[test]
    fn test_report_default() {
        let report = CoverageReport::default();
        assert_eq!(report.total_blocks(), 0);
        assert_eq!(report.covered_count(), 0);
        assert!((report.coverage_percent() - 100.0).abs() < 0.001);
    }

    // ============================================================================
    // Edge Cases
    // ============================================================================

    /// Test recording violation without affected block
    #[test]
    fn test_report_record_violation_no_block() {
        let mut report = CoverageReport::new(5);
        report.record_violation(CoverageViolation::CoverageRegression {
            expected: 95.0,
            actual: 90.0,
        });

        assert_eq!(report.violation_count(), 1);
        // No block should be tainted
        for i in 0..5 {
            assert!(!report.is_tainted(BlockId::new(i)));
        }
    }

    /// Test set_source_location overwrites existing
    #[test]
    fn test_set_source_location_overwrite() {
        let mut report = CoverageReport::new(5);
        report.set_source_location(BlockId::new(0), "first.rs:1");
        report.set_source_location(BlockId::new(0), "second.rs:2");

        let coverages = report.block_coverages();
        assert_eq!(
            coverages[0].source_location,
            Some("second.rs:2".to_string())
        );
    }

    /// Test set_function_name overwrites existing
    #[test]
    fn test_set_function_name_overwrite() {
        let mut report = CoverageReport::new(5);
        report.set_function_name(BlockId::new(0), "first_fn");
        report.set_function_name(BlockId::new(0), "second_fn");

        let coverages = report.block_coverages();
        assert_eq!(coverages[0].function_name, Some("second_fn".to_string()));
    }

    /// Test block_coverages with only some metadata
    #[test]
    fn test_block_coverages_partial_metadata() {
        let mut report = CoverageReport::new(5);
        report.set_source_location(BlockId::new(0), "a.rs:1");
        report.set_function_name(BlockId::new(1), "fn_b");
        report.record_hit(BlockId::new(2));
        // Block 3 and 4 have no metadata

        let coverages = report.block_coverages();

        assert_eq!(coverages[0].source_location, Some("a.rs:1".to_string()));
        assert!(coverages[0].function_name.is_none());

        assert!(coverages[1].source_location.is_none());
        assert_eq!(coverages[1].function_name, Some("fn_b".to_string()));

        assert!(coverages[2].source_location.is_none());
        assert!(coverages[2].function_name.is_none());
        assert_eq!(coverages[2].hit_count, 1);

        assert!(coverages[3].source_location.is_none());
        assert!(coverages[3].function_name.is_none());
        assert_eq!(coverages[3].hit_count, 0);
    }

    /// Test large block count for performance
    #[test]
    fn test_large_block_count() {
        let mut report = CoverageReport::new(10000);

        // Record hits on every 10th block
        for i in (0..10000).step_by(10) {
            report.record_hit(BlockId::new(i));
        }

        assert_eq!(report.covered_count(), 1000);
        assert!((report.coverage_percent() - 10.0).abs() < 0.001);
    }

    /// Test multiple merges
    #[test]
    fn test_multiple_merges() {
        let mut report1 = CoverageReport::new(5);
        report1.record_hit(BlockId::new(0));

        let mut report2 = CoverageReport::new(5);
        report2.record_hit(BlockId::new(1));

        let mut report3 = CoverageReport::new(5);
        report3.record_hit(BlockId::new(2));

        report1.merge(&report2);
        report1.merge(&report3);

        assert_eq!(report1.covered_count(), 3);
    }

    /// Test hit count for block outside total_blocks range
    #[test]
    fn test_hit_count_out_of_range() {
        let mut report = CoverageReport::new(5);
        // Record hit on block that's technically "out of range"
        report.record_hit(BlockId::new(100));

        // Should still be recorded
        assert_eq!(report.get_hit_count(BlockId::new(100)), 1);
        // But won't show in covered_count (which iterates 0..total_blocks)
        assert_eq!(report.covered_count(), 0);
    }

    /// Test uncovered and covered blocks with out-of-range hits
    #[test]
    fn test_blocks_list_range() {
        let mut report = CoverageReport::new(3);
        report.record_hit(BlockId::new(0));
        report.record_hit(BlockId::new(100)); // Out of range

        let covered = report.covered_blocks();
        let uncovered = report.uncovered_blocks();

        // Only considers blocks 0..3
        assert_eq!(covered.len(), 1);
        assert!(covered.contains(&BlockId::new(0)));

        assert_eq!(uncovered.len(), 2);
        assert!(uncovered.contains(&BlockId::new(1)));
        assert!(uncovered.contains(&BlockId::new(2)));
    }
}
