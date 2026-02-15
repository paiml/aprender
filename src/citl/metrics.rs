//! Metrics tracking for CITL performance analysis.
//!
//! Provides comprehensive tracking of fix attempts, pattern usage,
//! compilation times, and convergence rates.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// Comprehensive metrics tracker for CITL operations.
///
/// Tracks fix attempt success rates, pattern usage, compilation times,
/// error frequencies, and convergence statistics.
#[derive(Debug)]
pub struct MetricsTracker {
    /// Fix attempt metrics
    fix_attempts: FixAttemptMetrics,
    /// Pattern usage metrics
    pattern_usage: PatternUsageMetrics,
    /// Compilation time metrics
    compilation_times: CompilationTimeMetrics,
    /// Error frequency metrics
    error_frequencies: ErrorFrequencyMetrics,
    /// Convergence metrics
    convergence: ConvergenceMetrics,
    /// Session start time
    session_start: Instant,
}

impl MetricsTracker {
    /// Create a new metrics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fix_attempts: FixAttemptMetrics::new(),
            pattern_usage: PatternUsageMetrics::new(),
            compilation_times: CompilationTimeMetrics::new(),
            error_frequencies: ErrorFrequencyMetrics::new(),
            convergence: ConvergenceMetrics::new(),
            session_start: Instant::now(),
        }
    }

    /// Record a fix attempt result.
    pub fn record_fix_attempt(&mut self, success: bool, error_code: &str) {
        self.fix_attempts.record(success);
        self.error_frequencies.record(error_code);
    }

    /// Record pattern usage.
    pub fn record_pattern_use(&mut self, pattern_id: usize, success: bool) {
        self.pattern_usage.record(pattern_id, success);
    }

    /// Record compilation time.
    pub fn record_compilation_time(&mut self, duration: Duration) {
        self.compilation_times.record(duration);
    }

    /// Record fix convergence (number of iterations to fix).
    pub fn record_convergence(&mut self, iterations: usize, success: bool) {
        self.convergence.record(iterations, success);
    }

    /// Get fix attempt metrics.
    #[must_use]
    pub fn fix_attempts(&self) -> &FixAttemptMetrics {
        &self.fix_attempts
    }

    /// Get pattern usage metrics.
    #[must_use]
    pub fn pattern_usage(&self) -> &PatternUsageMetrics {
        &self.pattern_usage
    }

    /// Get compilation time metrics.
    #[must_use]
    pub fn compilation_times(&self) -> &CompilationTimeMetrics {
        &self.compilation_times
    }

    /// Get error frequency metrics.
    #[must_use]
    pub fn error_frequencies(&self) -> &ErrorFrequencyMetrics {
        &self.error_frequencies
    }

    /// Get convergence metrics.
    #[must_use]
    pub fn convergence(&self) -> &ConvergenceMetrics {
        &self.convergence
    }

    /// Get session duration.
    #[must_use]
    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }

    /// Get a summary of all metrics.
    #[must_use]
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_fix_attempts: self.fix_attempts.total(),
            fix_success_rate: self.fix_attempts.success_rate(),
            total_compilations: self.compilation_times.count(),
            avg_compilation_time_ms: self.compilation_times.average_ms(),
            most_common_errors: self.error_frequencies.top_n(5),
            avg_iterations_to_fix: self.convergence.average_iterations(),
            convergence_rate: self.convergence.success_rate(),
            session_duration: self.session_duration(),
        }
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.fix_attempts = FixAttemptMetrics::new();
        self.pattern_usage = PatternUsageMetrics::new();
        self.compilation_times = CompilationTimeMetrics::new();
        self.error_frequencies = ErrorFrequencyMetrics::new();
        self.convergence = ConvergenceMetrics::new();
        self.session_start = Instant::now();
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for fix attempts.
#[derive(Debug, Clone)]
pub struct FixAttemptMetrics {
    /// Number of successful fixes
    successes: u64,
    /// Number of failed fixes
    failures: u64,
}

impl FixAttemptMetrics {
    /// Create new fix attempt metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            successes: 0,
            failures: 0,
        }
    }

    /// Record a fix attempt.
    pub fn record(&mut self, success: bool) {
        if success {
            self.successes += 1;
        } else {
            self.failures += 1;
        }
    }

    /// Get total attempts.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.successes + self.failures
    }

    /// Get success rate (0.0 to 1.0).
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            self.successes as f64 / total as f64
        }
    }

    /// Get success count.
    #[must_use]
    pub fn successes(&self) -> u64 {
        self.successes
    }

    /// Get failure count.
    #[must_use]
    pub fn failures(&self) -> u64 {
        self.failures
    }
}

impl Default for FixAttemptMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for pattern usage.
#[derive(Debug, Clone)]
pub struct PatternUsageMetrics {
    /// Usage counts by pattern ID
    usage_counts: HashMap<usize, u64>,
    /// Success counts by pattern ID
    success_counts: HashMap<usize, u64>,
}

impl PatternUsageMetrics {
    /// Create new pattern usage metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            usage_counts: HashMap::new(),
            success_counts: HashMap::new(),
        }
    }

    /// Record pattern usage.
    pub fn record(&mut self, pattern_id: usize, success: bool) {
        *self.usage_counts.entry(pattern_id).or_insert(0) += 1;
        if success {
            *self.success_counts.entry(pattern_id).or_insert(0) += 1;
        }
    }

    /// Get usage count for a pattern.
    #[must_use]
    pub fn usage_count(&self, pattern_id: usize) -> u64 {
        *self.usage_counts.get(&pattern_id).unwrap_or(&0)
    }

    /// Get success rate for a pattern.
    #[must_use]
    pub fn pattern_success_rate(&self, pattern_id: usize) -> f64 {
        let usage = self.usage_count(pattern_id);
        if usage == 0 {
            0.0
        } else {
            let successes = *self.success_counts.get(&pattern_id).unwrap_or(&0);
            successes as f64 / usage as f64
        }
    }

    /// Get total patterns used.
    #[must_use]
    pub fn total_patterns_used(&self) -> usize {
        self.usage_counts.len()
    }

    /// Get most used patterns.
    #[must_use]
    pub fn most_used(&self, n: usize) -> Vec<(usize, u64)> {
        let mut counts: Vec<_> = self.usage_counts.iter().map(|(&k, &v)| (k, v)).collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts.truncate(n);
        counts
    }
}

impl Default for PatternUsageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for compilation times.
#[derive(Debug, Clone)]
pub struct CompilationTimeMetrics {
    /// Total compilation time
    total_time: Duration,
    /// Number of compilations
    count: u64,
    /// Minimum compilation time
    min_time: Option<Duration>,
    /// Maximum compilation time
    max_time: Option<Duration>,
}

impl CompilationTimeMetrics {
    /// Create new compilation time metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_time: Duration::ZERO,
            count: 0,
            min_time: None,
            max_time: None,
        }
    }

    /// Record a compilation time.
    pub fn record(&mut self, duration: Duration) {
        self.total_time += duration;
        self.count += 1;

        match self.min_time {
            None => self.min_time = Some(duration),
            Some(min) if duration < min => self.min_time = Some(duration),
            _ => {}
        }

        match self.max_time {
            None => self.max_time = Some(duration),
            Some(max) if duration > max => self.max_time = Some(duration),
            _ => {}
        }
    }

    /// Get compilation count.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get average compilation time in milliseconds.
    #[must_use]
    pub fn average_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_time.as_millis() as f64 / self.count as f64
        }
    }

    /// Get total compilation time.
    #[must_use]
    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    /// Get minimum compilation time.
    #[must_use]
    pub fn min_time(&self) -> Option<Duration> {
        self.min_time
    }

    /// Get maximum compilation time.
    #[must_use]
    pub fn max_time(&self) -> Option<Duration> {
        self.max_time
    }
}

impl Default for CompilationTimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for error frequencies.
#[derive(Debug, Clone)]
pub struct ErrorFrequencyMetrics {
    /// Error counts by error code
    error_counts: HashMap<String, u64>,
}

impl ErrorFrequencyMetrics {
    /// Create new error frequency metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            error_counts: HashMap::new(),
        }
    }

    /// Record an error.
    pub fn record(&mut self, error_code: &str) {
        *self.error_counts.entry(error_code.to_string()).or_insert(0) += 1;
    }

    /// Get count for an error code.
    #[must_use]
    pub fn count(&self, error_code: &str) -> u64 {
        *self.error_counts.get(error_code).unwrap_or(&0)
    }

    /// Get total errors.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.error_counts.values().sum()
    }

    /// Get number of unique error types.
    #[must_use]
    pub fn unique_errors(&self) -> usize {
        self.error_counts.len()
    }

    /// Get top N most common errors.
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<(String, u64)> {
        let mut counts: Vec<_> = self
            .error_counts
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts.truncate(n);
        counts
    }
}

impl Default for ErrorFrequencyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for convergence (iterations to fix).
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Total iterations across all fix attempts
    total_iterations: u64,
    /// Number of successful fixes
    successful_fixes: u64,
    /// Number of failed fixes
    failed_fixes: u64,
    /// Histogram of iterations to fix
    iteration_histogram: HashMap<usize, u64>,
}

include!("metrics_part_02.rs");
include!("metrics_part_03.rs");
