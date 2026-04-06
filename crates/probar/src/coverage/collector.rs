//! Coverage Collector
//!
//! Per spec §9.1: Coverage Collection API
//!
//! Manages coverage collection sessions and test runs.

use super::{BlockId, CoverageReport, CoverageViolation, JidokaAction, ThreadLocalCounters};

/// Coverage granularity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Granularity {
    /// Function-level coverage (coarsest)
    #[default]
    Function,
    /// Basic block coverage
    BasicBlock,
    /// Edge/branch coverage
    Edge,
    /// Path coverage (finest)
    Path,
}

/// Coverage collection configuration
#[derive(Debug, Clone)]
pub struct CoverageConfig {
    /// Coverage granularity
    pub granularity: Granularity,
    /// Enable parallel collection
    pub parallel: bool,
    /// Enable Jidoka guards
    pub jidoka_enabled: bool,
    /// Checkpoint interval (in seconds)
    pub checkpoint_interval: Option<u64>,
    /// Maximum blocks to track
    pub max_blocks: usize,
}

impl CoverageConfig {
    /// Create a builder for coverage config
    #[must_use]
    pub fn builder() -> CoverageConfigBuilder {
        CoverageConfigBuilder::default()
    }
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            granularity: Granularity::Function,
            parallel: false,
            jidoka_enabled: true,
            checkpoint_interval: None,
            max_blocks: 100_000,
        }
    }
}

/// Builder for coverage configuration
#[derive(Debug, Default)]
pub struct CoverageConfigBuilder {
    granularity: Granularity,
    parallel: bool,
    jidoka_enabled: bool,
    checkpoint_interval: Option<u64>,
    max_blocks: usize,
}

impl CoverageConfigBuilder {
    /// Set the coverage granularity
    #[must_use]
    pub fn granularity(mut self, granularity: Granularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Enable parallel collection
    #[must_use]
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Enable Jidoka guards
    #[must_use]
    pub fn jidoka_enabled(mut self, enabled: bool) -> Self {
        self.jidoka_enabled = enabled;
        self
    }

    /// Set checkpoint interval
    #[must_use]
    pub fn checkpoint_interval(mut self, seconds: u64) -> Self {
        self.checkpoint_interval = Some(seconds);
        self
    }

    /// Set maximum blocks
    #[must_use]
    pub fn max_blocks(mut self, max: usize) -> Self {
        self.max_blocks = max;
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> CoverageConfig {
        CoverageConfig {
            granularity: self.granularity,
            parallel: self.parallel,
            jidoka_enabled: self.jidoka_enabled,
            checkpoint_interval: self.checkpoint_interval,
            max_blocks: if self.max_blocks == 0 {
                100_000
            } else {
                self.max_blocks
            },
        }
    }
}

/// Coverage collector for tracking coverage during test execution
#[derive(Debug)]
pub struct CoverageCollector {
    /// Configuration
    config: CoverageConfig,
    /// Current session report
    report: Option<CoverageReport>,
    /// Current test name
    current_test: Option<String>,
    /// Thread-local counters
    counters: ThreadLocalCounters,
    /// Session active flag
    session_active: bool,
    /// Test active flag
    test_active: bool,
}

impl CoverageCollector {
    /// Create a new collector with the given configuration
    #[must_use]
    pub fn new(config: CoverageConfig) -> Self {
        let max_blocks = config.max_blocks;
        Self {
            config,
            report: None,
            current_test: None,
            counters: ThreadLocalCounters::new(max_blocks),
            session_active: false,
            test_active: false,
        }
    }

    /// Begin a coverage collection session
    pub fn begin_session(&mut self, name: &str) {
        let mut report = CoverageReport::new(self.config.max_blocks);
        report.set_session_name(name);
        self.report = Some(report);
        self.session_active = true;
    }

    /// End the current session and return the report
    #[must_use]
    pub fn end_session(&mut self) -> CoverageReport {
        self.flush_counters();
        self.session_active = false;
        self.report.take().unwrap_or_default()
    }

    /// Begin a test within the session
    pub fn begin_test(&mut self, name: &str) {
        self.current_test = Some(name.to_string());
        self.test_active = true;
        if let Some(report) = &mut self.report {
            report.add_test(name);
        }
    }

    /// End the current test
    pub fn end_test(&mut self) {
        self.flush_counters();
        self.current_test = None;
        self.test_active = false;
    }

    /// Record a hit on a block
    pub fn record_hit(&mut self, block: BlockId) {
        self.counters.increment(block);
    }

    /// Record a violation
    pub fn record_violation(&mut self, violation: CoverageViolation) {
        if self.config.jidoka_enabled {
            let action = violation.action();
            if let Some(report) = &mut self.report {
                report.record_violation(violation);
            }

            // Hard stop would panic here in production
            if action == JidokaAction::Stop {
                // In a real implementation, this would trigger the Andon cord
                // For now, we just record it
            }
        }
    }

    /// Flush thread-local counters to the report
    fn flush_counters(&mut self) {
        let counts = self.counters.flush();
        if let Some(report) = &mut self.report {
            for (idx, count) in counts.iter().enumerate() {
                if *count > 0 {
                    report.record_hits(BlockId::new(idx as u32), *count);
                }
            }
        }
    }

    /// Check if a session is active
    #[must_use]
    pub fn is_session_active(&self) -> bool {
        self.session_active
    }

    /// Check if a test is active
    #[must_use]
    pub fn is_test_active(&self) -> bool {
        self.test_active
    }

    /// Get the current test name
    #[must_use]
    pub fn current_test(&self) -> Option<&str> {
        self.current_test.as_deref()
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &CoverageConfig {
        &self.config
    }
}
