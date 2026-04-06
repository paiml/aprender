//! Fuzz suite, target config, and summary types

use super::types::FuzzResult;
use std::collections::HashMap;

/// Fuzz target configuration
#[derive(Debug, Clone)]
pub struct FuzzTargetConfig {
    /// Target name
    pub name: String,
    /// Number of iterations
    pub iterations: u64,
    /// Timeout in seconds
    pub timeout_secs: u64,
    /// Seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Enable coverage tracking
    pub track_coverage: bool,
}

impl FuzzTargetConfig {
    /// Create a new fuzz target config
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            iterations: 10000,
            timeout_secs: 60,
            seed: None,
            track_coverage: true,
        }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: u64) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Set seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Fuzz testing suite for cbtop components
#[derive(Debug, Default)]
pub struct FuzzSuite {
    /// Results by target name
    results: HashMap<String, FuzzResult>,
    /// Total test cases
    total_cases: u64,
    /// Total failures
    total_failures: u64,
}

impl FuzzSuite {
    /// Create a new fuzz suite
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a result
    pub fn add_result(&mut self, result: FuzzResult) {
        self.total_cases += result.test_cases;
        self.total_failures += result.failures;
        self.results.insert(result.target.clone(), result);
    }

    /// Get result for a target
    pub fn get_result(&self, target: &str) -> Option<&FuzzResult> {
        self.results.get(target)
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.total_failures == 0
    }

    /// Get total test cases
    pub fn total_cases(&self) -> u64 {
        self.total_cases
    }

    /// Get total failures
    pub fn total_failures(&self) -> u64 {
        self.total_failures
    }

    /// Get all results
    pub fn results(&self) -> &HashMap<String, FuzzResult> {
        &self.results
    }

    /// Generate summary report
    pub fn summary(&self) -> FuzzSummary {
        let targets_passed = self.results.values().filter(|r| r.passed()).count();
        let avg_coverage = if self.results.is_empty() {
            0.0
        } else {
            self.results
                .values()
                .map(|r| r.coverage_percent)
                .sum::<f64>()
                / self.results.len() as f64
        };

        FuzzSummary {
            total_targets: self.results.len(),
            targets_passed,
            total_cases: self.total_cases,
            total_failures: self.total_failures,
            avg_coverage,
            overall_passed: self.all_passed(),
        }
    }
}

/// Summary of fuzz testing results
#[derive(Debug, Clone)]
pub struct FuzzSummary {
    /// Total number of fuzz targets
    pub total_targets: usize,
    /// Number of targets that passed
    pub targets_passed: usize,
    /// Total test cases across all targets
    pub total_cases: u64,
    /// Total failures across all targets
    pub total_failures: u64,
    /// Average coverage percentage
    pub avg_coverage: f64,
    /// Whether all tests passed
    pub overall_passed: bool,
}

impl FuzzSummary {
    /// Get pass rate as percentage
    pub fn pass_rate(&self) -> f64 {
        if self.total_targets == 0 {
            100.0
        } else {
            (self.targets_passed as f64 / self.total_targets as f64) * 100.0
        }
    }
}
