//! Test Sharding for Distributed Execution (Feature G.5)
//!
//! Enables parallel test execution across multiple machines in CI/CD pipelines.
//! Implements deterministic test distribution for reproducible sharding.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Shard configuration for distributed test execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Current shard index (1-based)
    pub current: u32,
    /// Total number of shards
    pub total: u32,
}

impl ShardConfig {
    /// Create a new shard configuration
    ///
    /// # Panics
    ///
    /// Panics if `current` is 0, greater than `total`, or `total` is 0.
    #[must_use]
    pub fn new(current: u32, total: u32) -> Self {
        assert!(total > 0, "Total shards must be greater than 0");
        assert!(
            current > 0,
            "Current shard must be 1-based (greater than 0)"
        );
        assert!(
            current <= total,
            "Current shard ({current}) cannot exceed total ({total})"
        );

        Self { current, total }
    }

    /// Parse shard config from CLI string format "N/M"
    ///
    /// # Errors
    ///
    /// Returns error if format is invalid
    pub fn parse(s: &str) -> Result<Self, ShardParseError> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 2 {
            return Err(ShardParseError::InvalidFormat(s.to_string()));
        }

        let current = parts[0]
            .parse::<u32>()
            .map_err(|_| ShardParseError::InvalidNumber(parts[0].to_string()))?;
        let total = parts[1]
            .parse::<u32>()
            .map_err(|_| ShardParseError::InvalidNumber(parts[1].to_string()))?;

        if total == 0 {
            return Err(ShardParseError::ZeroTotal);
        }
        if current == 0 {
            return Err(ShardParseError::ZeroCurrent);
        }
        if current > total {
            return Err(ShardParseError::CurrentExceedsTotal { current, total });
        }

        Ok(Self { current, total })
    }

    /// Check if a test at given index should run on this shard
    ///
    /// Uses modulo distribution for even test distribution.
    #[must_use]
    pub fn should_run_index(&self, test_index: usize) -> bool {
        (test_index % self.total as usize) + 1 == self.current as usize
    }

    /// Check if a test with given name should run on this shard
    ///
    /// Uses deterministic hash-based distribution for stable assignment.
    #[must_use]
    pub fn should_run_name(&self, test_name: &str) -> bool {
        let hash = Self::hash_test_name(test_name);
        (hash % self.total as u64) + 1 == self.current as u64
    }

    /// Compute deterministic hash for test name
    fn hash_test_name(name: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        hasher.finish()
    }

    /// Filter a list of tests to only those that should run on this shard
    #[must_use]
    pub fn filter_tests<'a>(&self, tests: &'a [&str]) -> Vec<&'a str> {
        tests
            .iter()
            .filter(|name| self.should_run_name(name))
            .copied()
            .collect()
    }

    /// Filter tests by index
    #[must_use]
    pub fn filter_by_index<T: Clone>(&self, items: &[T]) -> Vec<T> {
        items
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.should_run_index(*idx))
            .map(|(_, item)| item.clone())
            .collect()
    }

    /// Get estimated test count for this shard
    #[must_use]
    pub fn estimated_count(&self, total_tests: usize) -> usize {
        let base = total_tests / self.total as usize;
        let remainder = total_tests % self.total as usize;
        if self.current as usize <= remainder {
            base + 1
        } else {
            base
        }
    }

    /// Validate that all shards together cover all tests exactly once
    #[must_use]
    pub fn validate_coverage(total_tests: usize, total_shards: u32) -> bool {
        let mut covered = vec![false; total_tests];

        for shard in 1..=total_shards {
            let config = ShardConfig::new(shard, total_shards);
            for (idx, is_covered) in covered.iter_mut().enumerate() {
                if config.should_run_index(idx) {
                    if *is_covered {
                        return false; // Double coverage
                    }
                    *is_covered = true;
                }
            }
        }

        covered.iter().all(|&c| c)
    }
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            current: 1,
            total: 1,
        }
    }
}

impl std::fmt::Display for ShardConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.current, self.total)
    }
}

/// Errors that can occur when parsing shard configuration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardParseError {
    /// Invalid format (expected "N/M")
    InvalidFormat(String),
    /// Invalid number in shard spec
    InvalidNumber(String),
    /// Total shards cannot be zero
    ZeroTotal,
    /// Current shard cannot be zero (1-based)
    ZeroCurrent,
    /// Current shard exceeds total
    CurrentExceedsTotal {
        /// Current shard number
        current: u32,
        /// Total shard count
        total: u32,
    },
}

impl std::fmt::Display for ShardParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(s) => {
                write!(
                    f,
                    "Invalid shard format '{s}', expected 'N/M' (e.g., '1/4')"
                )
            }
            Self::InvalidNumber(s) => write!(f, "Invalid number in shard spec: '{s}'"),
            Self::ZeroTotal => write!(f, "Total shards cannot be zero"),
            Self::ZeroCurrent => write!(f, "Current shard must be 1-based (cannot be 0)"),
            Self::CurrentExceedsTotal { current, total } => {
                write!(f, "Current shard ({current}) exceeds total ({total})")
            }
        }
    }
}

impl std::error::Error for ShardParseError {}

/// Sharded test runner
#[derive(Debug, Clone)]
pub struct ShardedRunner {
    config: ShardConfig,
    test_names: Vec<String>,
}

impl ShardedRunner {
    /// Create a new sharded runner
    #[must_use]
    pub fn new(config: ShardConfig) -> Self {
        Self {
            config,
            test_names: Vec::new(),
        }
    }

    /// Add tests to the runner
    pub fn add_tests(&mut self, tests: impl IntoIterator<Item = impl Into<String>>) {
        for test in tests {
            self.test_names.push(test.into());
        }
    }

    /// Get tests assigned to this shard
    #[must_use]
    pub fn assigned_tests(&self) -> Vec<&str> {
        self.test_names
            .iter()
            .filter(|name| self.config.should_run_name(name))
            .map(String::as_str)
            .collect()
    }

    /// Get shard configuration
    #[must_use]
    pub fn config(&self) -> ShardConfig {
        self.config
    }

    /// Get total test count
    #[must_use]
    pub fn total_tests(&self) -> usize {
        self.test_names.len()
    }

    /// Get assigned test count
    #[must_use]
    pub fn assigned_count(&self) -> usize {
        self.assigned_tests().len()
    }
}

/// Report for merged shard results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardReport {
    /// Shard configuration used
    pub shard: Option<ShardConfig>,
    /// Number of tests run
    pub tests_run: usize,
    /// Number of tests passed
    pub tests_passed: usize,
    /// Number of tests failed
    pub tests_failed: usize,
    /// Number of tests skipped
    pub tests_skipped: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Failed test names
    pub failed_tests: Vec<String>,
}

impl ShardReport {
    /// Create a new empty report
    #[must_use]
    pub fn new(shard: ShardConfig) -> Self {
        Self {
            shard: Some(shard),
            ..Default::default()
        }
    }

    /// Check if all tests passed
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.tests_failed == 0
    }

    /// Merge multiple shard reports
    #[must_use]
    pub fn merge(reports: &[ShardReport]) -> Self {
        let mut merged = Self::default();

        for report in reports {
            merged.tests_run += report.tests_run;
            merged.tests_passed += report.tests_passed;
            merged.tests_failed += report.tests_failed;
            merged.tests_skipped += report.tests_skipped;
            merged.duration_ms = merged.duration_ms.max(report.duration_ms);
            merged.failed_tests.extend(report.failed_tests.clone());
        }

        merged
    }

    /// Export to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_range_loop,
    clippy::field_reassign_with_default
)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-SHARD-01: ShardConfig creation
    // =========================================================================

    #[test]
    fn h0_shard_01_config_new() {
        let config = ShardConfig::new(1, 4);
        assert_eq!(config.current, 1);
        assert_eq!(config.total, 4);
    }

    #[test]
    fn h0_shard_02_config_display() {
        let config = ShardConfig::new(2, 5);
        assert_eq!(format!("{config}"), "2/5");
    }

    #[test]
    #[should_panic(expected = "Total shards must be greater than 0")]
    fn h0_shard_03_config_zero_total_panics() {
        let _ = ShardConfig::new(1, 0);
    }

    #[test]
    #[should_panic(expected = "Current shard must be 1-based")]
    fn h0_shard_04_config_zero_current_panics() {
        let _ = ShardConfig::new(0, 4);
    }

    #[test]
    #[should_panic(expected = "cannot exceed total")]
    fn h0_shard_05_config_current_exceeds_total_panics() {
        let _ = ShardConfig::new(5, 4);
    }

    // =========================================================================
    // H₀-SHARD-06: Parse from string
    // =========================================================================

    #[test]
    fn h0_shard_06_parse_valid() {
        let config = ShardConfig::parse("2/4").unwrap();
        assert_eq!(config.current, 2);
        assert_eq!(config.total, 4);
    }

    #[test]
    fn h0_shard_07_parse_invalid_format() {
        let err = ShardConfig::parse("2-4").unwrap_err();
        assert!(matches!(err, ShardParseError::InvalidFormat(_)));
    }

    #[test]
    fn h0_shard_08_parse_invalid_number() {
        let err = ShardConfig::parse("abc/4").unwrap_err();
        assert!(matches!(err, ShardParseError::InvalidNumber(_)));
    }

    #[test]
    fn h0_shard_09_parse_zero_total() {
        let err = ShardConfig::parse("1/0").unwrap_err();
        assert!(matches!(err, ShardParseError::ZeroTotal));
    }

    #[test]
    fn h0_shard_10_parse_zero_current() {
        let err = ShardConfig::parse("0/4").unwrap_err();
        assert!(matches!(err, ShardParseError::ZeroCurrent));
    }

    #[test]
    fn h0_shard_11_parse_current_exceeds_total() {
        let err = ShardConfig::parse("5/4").unwrap_err();
        assert!(matches!(err, ShardParseError::CurrentExceedsTotal { .. }));
    }

    // =========================================================================
    // H₀-SHARD-12: Test distribution by index
    // =========================================================================

    #[test]
    fn h0_shard_12_should_run_index_shard1of4() {
        let config = ShardConfig::new(1, 4);
        assert!(config.should_run_index(0)); // 0 % 4 = 0, +1 = 1
        assert!(!config.should_run_index(1)); // 1 % 4 = 1, +1 = 2
        assert!(!config.should_run_index(2)); // 2 % 4 = 2, +1 = 3
        assert!(!config.should_run_index(3)); // 3 % 4 = 3, +1 = 4
        assert!(config.should_run_index(4)); // 4 % 4 = 0, +1 = 1
    }

    #[test]
    fn h0_shard_13_should_run_index_shard2of4() {
        let config = ShardConfig::new(2, 4);
        assert!(!config.should_run_index(0));
        assert!(config.should_run_index(1));
        assert!(!config.should_run_index(2));
        assert!(!config.should_run_index(3));
        assert!(!config.should_run_index(4));
        assert!(config.should_run_index(5));
    }

    #[test]
    fn h0_shard_14_all_shards_cover_all_tests() {
        // 10 tests distributed across 4 shards
        let mut covered = [false; 10];

        for shard in 1..=4 {
            let config = ShardConfig::new(shard, 4);
            for idx in 0..10 {
                if config.should_run_index(idx) {
                    assert!(!covered[idx], "Test {idx} covered twice");
                    covered[idx] = true;
                }
            }
        }

        assert!(covered.iter().all(|&c| c), "All tests must be covered");
    }

    // =========================================================================
    // H₀-SHARD-15: Test distribution by name (hash-based)
    // =========================================================================

    #[test]
    fn h0_shard_15_should_run_name_deterministic() {
        let config = ShardConfig::new(1, 4);
        let result1 = config.should_run_name("test_foo");
        let result2 = config.should_run_name("test_foo");
        assert_eq!(result1, result2, "Same name should give same result");
    }

    #[test]
    fn h0_shard_16_filter_tests_by_name() {
        let config = ShardConfig::new(1, 2);
        let tests = vec!["test_a", "test_b", "test_c", "test_d"];
        let filtered = config.filter_tests(&tests);

        // Should get roughly half the tests
        assert!(!filtered.is_empty());
        assert!(filtered.len() <= tests.len());
    }

    #[test]
    fn h0_shard_17_all_shards_cover_all_names() {
        let tests = vec!["test_1", "test_2", "test_3", "test_4", "test_5"];
        let mut covered = vec![false; tests.len()];

        for shard in 1..=3 {
            let config = ShardConfig::new(shard, 3);
            for (idx, test) in tests.iter().enumerate() {
                if config.should_run_name(test) {
                    covered[idx] = true;
                }
            }
        }

        assert!(covered.iter().all(|&c| c), "All tests must be covered");
    }

    // =========================================================================
    // H₀-SHARD-18: Filter by index
    // =========================================================================

    #[test]
    fn h0_shard_18_filter_by_index() {
        let config = ShardConfig::new(1, 2);
        let items = vec!["a", "b", "c", "d"];
        let filtered = config.filter_by_index(&items);

        // Shard 1 of 2 gets indices 0, 2
        assert_eq!(filtered, vec!["a", "c"]);
    }

    #[test]
    fn h0_shard_19_filter_by_index_shard2() {
        let config = ShardConfig::new(2, 2);
        let items = vec!["a", "b", "c", "d"];
        let filtered = config.filter_by_index(&items);

        // Shard 2 of 2 gets indices 1, 3
        assert_eq!(filtered, vec!["b", "d"]);
    }

    // =========================================================================
    // H₀-SHARD-20: Estimated count
    // =========================================================================

    #[test]
    fn h0_shard_20_estimated_count_even() {
        let config = ShardConfig::new(1, 4);
        assert_eq!(config.estimated_count(100), 25);
    }

    #[test]
    fn h0_shard_21_estimated_count_uneven() {
        // 10 tests / 3 shards = 3 each + 1 remainder
        let config1 = ShardConfig::new(1, 3);
        let config2 = ShardConfig::new(2, 3);
        let config3 = ShardConfig::new(3, 3);

        let total =
            config1.estimated_count(10) + config2.estimated_count(10) + config3.estimated_count(10);
        assert_eq!(total, 10);
    }

    // =========================================================================
    // H₀-SHARD-22: Validate coverage
    // =========================================================================

    #[test]
    fn h0_shard_22_validate_coverage_success() {
        assert!(ShardConfig::validate_coverage(100, 4));
        assert!(ShardConfig::validate_coverage(10, 3));
        assert!(ShardConfig::validate_coverage(7, 7));
    }

    // =========================================================================
    // H₀-SHARD-23: ShardedRunner
    // =========================================================================

    #[test]
    fn h0_shard_23_runner_new() {
        let config = ShardConfig::new(1, 4);
        let runner = ShardedRunner::new(config);
        assert_eq!(runner.config(), config);
        assert_eq!(runner.total_tests(), 0);
    }

    #[test]
    fn h0_shard_24_runner_add_tests() {
        let config = ShardConfig::new(1, 2);
        let mut runner = ShardedRunner::new(config);
        runner.add_tests(vec!["test_a", "test_b", "test_c"]);

        assert_eq!(runner.total_tests(), 3);
        assert!(runner.assigned_count() > 0);
    }

    #[test]
    fn h0_shard_25_runner_assigned_tests() {
        let config = ShardConfig::new(1, 2);
        let mut runner = ShardedRunner::new(config);
        runner.add_tests(vec!["test_a", "test_b", "test_c", "test_d"]);

        let assigned = runner.assigned_tests();
        assert!(!assigned.is_empty());
        assert!(assigned.len() <= 4);
    }

    // =========================================================================
    // H₀-SHARD-26: ShardReport
    // =========================================================================

    #[test]
    fn h0_shard_26_report_new() {
        let config = ShardConfig::new(1, 4);
        let report = ShardReport::new(config);
        assert_eq!(report.shard, Some(config));
        assert_eq!(report.tests_run, 0);
    }

    #[test]
    fn h0_shard_27_report_is_success() {
        let mut report = ShardReport::default();
        report.tests_passed = 10;
        report.tests_failed = 0;
        assert!(report.is_success());

        report.tests_failed = 1;
        assert!(!report.is_success());
    }

    #[test]
    fn h0_shard_28_report_merge() {
        let mut r1 = ShardReport::default();
        r1.tests_run = 10;
        r1.tests_passed = 9;
        r1.tests_failed = 1;
        r1.duration_ms = 1000;

        let mut r2 = ShardReport::default();
        r2.tests_run = 10;
        r2.tests_passed = 10;
        r2.tests_failed = 0;
        r2.duration_ms = 500;

        let merged = ShardReport::merge(&[r1, r2]);
        assert_eq!(merged.tests_run, 20);
        assert_eq!(merged.tests_passed, 19);
        assert_eq!(merged.tests_failed, 1);
        assert_eq!(merged.duration_ms, 1000); // Max duration
    }

    #[test]
    fn h0_shard_29_report_to_json() {
        let report = ShardReport::new(ShardConfig::new(1, 2));
        let json = report.to_json();
        assert!(json.contains("tests_run"));
        assert!(json.contains("shard"));
    }

    // =========================================================================
    // H₀-SHARD-30: Default config
    // =========================================================================

    #[test]
    fn h0_shard_30_default_config() {
        let config = ShardConfig::default();
        assert_eq!(config.current, 1);
        assert_eq!(config.total, 1);
        // Default should run all tests
        assert!(config.should_run_index(0));
        assert!(config.should_run_index(100));
    }

    // =========================================================================
    // Additional coverage tests for edge cases
    // =========================================================================

    #[test]
    fn h0_shard_31_parse_invalid_total_number() {
        // Test parsing with invalid second number (total)
        let err = ShardConfig::parse("1/abc").unwrap_err();
        assert!(matches!(err, ShardParseError::InvalidNumber(_)));
        if let ShardParseError::InvalidNumber(s) = &err {
            assert_eq!(s, "abc");
        }
    }

    #[test]
    fn h0_shard_32_parse_too_many_slashes() {
        let err = ShardConfig::parse("1/2/3").unwrap_err();
        assert!(matches!(err, ShardParseError::InvalidFormat(_)));
    }

    #[test]
    fn h0_shard_33_parse_single_number() {
        let err = ShardConfig::parse("5").unwrap_err();
        assert!(matches!(err, ShardParseError::InvalidFormat(_)));
    }

    #[test]
    fn h0_shard_34_error_display_invalid_format() {
        let err = ShardParseError::InvalidFormat("bad".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Invalid shard format"));
        assert!(msg.contains("bad"));
        assert!(msg.contains("N/M"));
    }

    #[test]
    fn h0_shard_35_error_display_invalid_number() {
        let err = ShardParseError::InvalidNumber("xyz".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Invalid number"));
        assert!(msg.contains("xyz"));
    }

    #[test]
    fn h0_shard_36_error_display_zero_total() {
        let err = ShardParseError::ZeroTotal;
        let msg = format!("{err}");
        assert!(msg.contains("Total shards cannot be zero"));
    }

    #[test]
    fn h0_shard_37_error_display_zero_current() {
        let err = ShardParseError::ZeroCurrent;
        let msg = format!("{err}");
        assert!(msg.contains("must be 1-based"));
    }

    #[test]
    fn h0_shard_38_error_display_current_exceeds_total() {
        let err = ShardParseError::CurrentExceedsTotal {
            current: 10,
            total: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("10"));
        assert!(msg.contains('5'));
        assert!(msg.contains("exceeds"));
    }

    #[test]
    fn h0_shard_39_error_is_std_error() {
        let err: Box<dyn std::error::Error> =
            Box::new(ShardParseError::InvalidFormat("test".to_string()));
        // Verify it implements std::error::Error
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn h0_shard_40_validate_coverage_with_zero_tests() {
        // Edge case: zero tests should be valid (trivially covered)
        assert!(ShardConfig::validate_coverage(0, 4));
    }

    #[test]
    fn h0_shard_41_validate_coverage_single_shard() {
        assert!(ShardConfig::validate_coverage(100, 1));
    }

    #[test]
    fn h0_shard_42_report_merge_with_failed_tests() {
        let mut r1 = ShardReport::default();
        r1.tests_run = 5;
        r1.tests_passed = 4;
        r1.tests_failed = 1;
        r1.failed_tests = vec!["test_a".to_string(), "test_b".to_string()];
        r1.duration_ms = 100;

        let mut r2 = ShardReport::default();
        r2.tests_run = 5;
        r2.tests_passed = 3;
        r2.tests_failed = 2;
        r2.failed_tests = vec!["test_c".to_string()];
        r2.duration_ms = 200;

        let merged = ShardReport::merge(&[r1, r2]);
        assert_eq!(merged.tests_run, 10);
        assert_eq!(merged.tests_passed, 7);
        assert_eq!(merged.tests_failed, 3);
        assert_eq!(merged.failed_tests.len(), 3);
        assert!(merged.failed_tests.contains(&"test_a".to_string()));
        assert!(merged.failed_tests.contains(&"test_b".to_string()));
        assert!(merged.failed_tests.contains(&"test_c".to_string()));
        assert_eq!(merged.duration_ms, 200); // max of 100 and 200
    }

    #[test]
    fn h0_shard_43_report_merge_empty() {
        let merged = ShardReport::merge(&[]);
        assert_eq!(merged.tests_run, 0);
        assert_eq!(merged.tests_passed, 0);
        assert_eq!(merged.tests_failed, 0);
        assert!(merged.is_success()); // No failures = success
    }

    #[test]
    fn h0_shard_44_report_with_skipped_tests() {
        let mut report = ShardReport::default();
        report.tests_run = 10;
        report.tests_passed = 8;
        report.tests_failed = 0;
        report.tests_skipped = 2;

        assert!(report.is_success());
        assert_eq!(report.tests_skipped, 2);
    }

    #[test]
    fn h0_shard_45_report_merge_skipped() {
        let mut r1 = ShardReport::default();
        r1.tests_skipped = 3;

        let mut r2 = ShardReport::default();
        r2.tests_skipped = 2;

        let merged = ShardReport::merge(&[r1, r2]);
        assert_eq!(merged.tests_skipped, 5);
    }

    #[test]
    fn h0_shard_46_estimated_count_more_shards_than_tests() {
        // 3 tests across 10 shards
        let config1 = ShardConfig::new(1, 10);
        let config2 = ShardConfig::new(4, 10);
        let config10 = ShardConfig::new(10, 10);

        // Shards 1-3 get 1 test each, shards 4-10 get 0
        assert_eq!(config1.estimated_count(3), 1);
        assert_eq!(config2.estimated_count(3), 0);
        assert_eq!(config10.estimated_count(3), 0);
    }

    #[test]
    fn h0_shard_47_filter_tests_empty_list() {
        let config = ShardConfig::new(1, 2);
        let tests: Vec<&str> = vec![];
        let filtered = config.filter_tests(&tests);
        assert!(filtered.is_empty());
    }

    #[test]
    fn h0_shard_48_filter_by_index_empty() {
        let config = ShardConfig::new(1, 2);
        let items: Vec<i32> = vec![];
        let filtered = config.filter_by_index(&items);
        assert!(filtered.is_empty());
    }

    #[test]
    fn h0_shard_49_runner_add_tests_from_strings() {
        let config = ShardConfig::new(1, 2);
        let mut runner = ShardedRunner::new(config);

        // Test with owned Strings
        runner.add_tests(vec!["test_1".to_string(), "test_2".to_string()]);
        assert_eq!(runner.total_tests(), 2);
    }

    #[test]
    fn h0_shard_50_runner_multiple_add_calls() {
        let config = ShardConfig::new(1, 2);
        let mut runner = ShardedRunner::new(config);

        runner.add_tests(vec!["test_a", "test_b"]);
        runner.add_tests(vec!["test_c"]);

        assert_eq!(runner.total_tests(), 3);
    }

    #[test]
    fn h0_shard_51_config_clone_and_eq() {
        let config1 = ShardConfig::new(2, 4);
        let config2 = config1;
        assert_eq!(config1, config2);

        let config3 = ShardConfig::new(3, 4);
        assert_ne!(config1, config3);
    }

    #[test]
    fn h0_shard_52_hash_test_name_consistency() {
        // Verify same name always produces same hash distribution
        let config = ShardConfig::new(1, 100);

        let mut results = Vec::new();
        for _ in 0..10 {
            results.push(config.should_run_name("consistent_test_name"));
        }

        // All results should be identical
        assert!(results.iter().all(|&r| r == results[0]));
    }

    #[test]
    fn h0_shard_53_different_names_different_distribution() {
        // Different names should (eventually) go to different shards
        let tests = vec![
            "test_alpha",
            "test_beta",
            "test_gamma",
            "test_delta",
            "test_epsilon",
            "test_zeta",
            "test_eta",
            "test_theta",
        ];

        // With 4 shards, different tests should distribute
        let mut shard_counts = [0usize; 4];
        for test in &tests {
            for shard in 1..=4 {
                let config = ShardConfig::new(shard, 4);
                if config.should_run_name(test) {
                    shard_counts[shard as usize - 1] += 1;
                    break;
                }
            }
        }

        // Each test goes to exactly one shard
        let total: usize = shard_counts.iter().sum();
        assert_eq!(total, tests.len());
    }

    #[test]
    fn h0_shard_54_report_json_contains_all_fields() {
        let config = ShardConfig::new(1, 2);
        let mut report = ShardReport::new(config);
        report.tests_run = 10;
        report.tests_passed = 8;
        report.tests_failed = 2;
        report.tests_skipped = 0;
        report.duration_ms = 1234;
        report.failed_tests = vec!["fail_1".to_string(), "fail_2".to_string()];

        let json = report.to_json();
        assert!(json.contains("\"tests_run\": 10"));
        assert!(json.contains("\"tests_passed\": 8"));
        assert!(json.contains("\"tests_failed\": 2"));
        assert!(json.contains("\"tests_skipped\": 0"));
        assert!(json.contains("\"duration_ms\": 1234"));
        assert!(json.contains("fail_1"));
        assert!(json.contains("fail_2"));
    }

    #[test]
    fn h0_shard_55_validate_coverage_large_test_count() {
        // Verify coverage validation with larger numbers
        assert!(ShardConfig::validate_coverage(1000, 16));
        assert!(ShardConfig::validate_coverage(999, 16));
    }

    #[test]
    fn h0_shard_56_shard_config_debug() {
        let config = ShardConfig::new(3, 7);
        let debug = format!("{config:?}");
        assert!(debug.contains("ShardConfig"));
        assert!(debug.contains("current: 3"));
        assert!(debug.contains("total: 7"));
    }

    #[test]
    fn h0_shard_57_shard_parse_error_clone() {
        let err1 = ShardParseError::InvalidFormat("test".to_string());
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn h0_shard_58_sharded_runner_debug() {
        let config = ShardConfig::new(1, 2);
        let runner = ShardedRunner::new(config);
        let debug = format!("{runner:?}");
        assert!(debug.contains("ShardedRunner"));
    }

    #[test]
    fn h0_shard_59_sharded_runner_clone() {
        let config = ShardConfig::new(1, 2);
        let mut runner = ShardedRunner::new(config);
        runner.add_tests(vec!["test_a", "test_b"]);

        let cloned = runner.clone();
        assert_eq!(runner.total_tests(), cloned.total_tests());
        assert_eq!(runner.config(), cloned.config());
    }

    #[test]
    fn h0_shard_60_shard_report_clone() {
        let config = ShardConfig::new(1, 2);
        let mut report = ShardReport::new(config);
        report.tests_run = 5;
        report.failed_tests = vec!["fail".to_string()];

        let cloned = report.clone();
        assert_eq!(report.tests_run, cloned.tests_run);
        assert_eq!(report.failed_tests, cloned.failed_tests);
    }
}
