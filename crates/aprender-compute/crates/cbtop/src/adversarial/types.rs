//! Types, enums, and small helper structs for adversarial testing.

use std::time::{Duration, Instant};

/// Error types for adversarial testing
#[derive(Debug, Clone, PartialEq)]
pub enum AdversarialError {
    /// Input tensor contains corrupted data
    CorruptedInput { byte_index: usize, expected_checksum: u32, actual_checksum: u32 },
    /// Memory allocation failed under pressure
    AllocationFailed { requested_bytes: usize },
    /// Zero-size input detected
    ZeroSizeInput,
    /// Input exceeds maximum allowed size
    MaxSizeExceeded { size: usize, max: usize },
    /// Clock skew detected (non-monotonic timestamp)
    ClockSkew { prev: Instant, curr: Instant },
    /// Concurrent access violation detected
    RaceCondition { description: String },
    /// Configuration parsing failed
    ConfigParseError { field: String, reason: String },
    /// Configuration value out of bounds
    ConfigOutOfBounds { field: String, value: String, min: String, max: String },
    /// Integer overflow detected
    IntegerOverflow { operation: String },
    /// Division by zero attempted
    DivisionByZero { numerator: f64 },
    /// NaN value detected in input
    NaNDetected { index: usize },
    /// Infinity value detected in input
    InfinityDetected { index: usize, positive: bool },
    /// Stack depth exceeded
    StackOverflow { depth: usize, max_depth: usize },
    /// Resource exhaustion (memory, handles, etc.)
    ResourceExhausted { resource: String },
    /// Operation timed out
    Timeout { operation: String, elapsed: Duration, limit: Duration },
    /// Operation was cancelled
    Cancelled { operation: String },
    /// Recovery failed after error
    RecoveryFailed { original_error: String, recovery_error: String },
}

/// Result type for adversarial operations
pub type AdversarialResult<T> = Result<T, AdversarialError>;

/// Adversarial tactic category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdversarialTactic {
    /// Random bit flips in input data
    BitFlipInjection,
    /// Simulate memory/CPU pressure
    ResourceStarvation,
    /// Test with manipulated timestamps
    ClockSkew,
    /// Simulate network failures
    NetworkPartition,
    /// Generate pathological configurations
    ConfigFuzzing,
}

impl AdversarialTactic {
    /// Get all tactics
    pub fn all() -> &'static [AdversarialTactic] {
        &[
            AdversarialTactic::BitFlipInjection,
            AdversarialTactic::ResourceStarvation,
            AdversarialTactic::ClockSkew,
            AdversarialTactic::NetworkPartition,
            AdversarialTactic::ConfigFuzzing,
        ]
    }

    /// Get the tactic name
    pub fn name(&self) -> &'static str {
        match self {
            AdversarialTactic::BitFlipInjection => "Bit-Flip Injection",
            AdversarialTactic::ResourceStarvation => "Resource Starvation",
            AdversarialTactic::ClockSkew => "Clock Skew",
            AdversarialTactic::NetworkPartition => "Network Partition",
            AdversarialTactic::ConfigFuzzing => "Config Fuzzing",
        }
    }

    /// Get the tool used for this tactic
    pub fn tool(&self) -> &'static str {
        match self {
            AdversarialTactic::BitFlipInjection => "proptest",
            AdversarialTactic::ResourceStarvation => "stress simulation",
            AdversarialTactic::ClockSkew => "libfaketime simulation",
            AdversarialTactic::NetworkPartition => "timeout simulation",
            AdversarialTactic::ConfigFuzzing => "proptest",
        }
    }
}

/// Numeric operations with overflow/underflow detection
#[derive(Debug, Clone, Copy, Default)]
pub struct CheckedArithmetic;

impl CheckedArithmetic {
    /// Create a new checked arithmetic helper
    pub fn new() -> Self {
        Self
    }

    /// Add with overflow check (F1012)
    pub fn checked_add_i64(a: i64, b: i64) -> AdversarialResult<i64> {
        a.checked_add(b)
            .ok_or_else(|| AdversarialError::IntegerOverflow { operation: format!("{a} + {b}") })
    }

    /// Multiply with overflow check (F1012)
    pub fn checked_mul_i64(a: i64, b: i64) -> AdversarialResult<i64> {
        a.checked_mul(b)
            .ok_or_else(|| AdversarialError::IntegerOverflow { operation: format!("{a} * {b}") })
    }

    /// Add with overflow check for usize
    pub fn checked_add_usize(a: usize, b: usize) -> AdversarialResult<usize> {
        a.checked_add(b)
            .ok_or_else(|| AdversarialError::IntegerOverflow { operation: format!("{a} + {b}") })
    }

    /// Multiply with overflow check for usize
    pub fn checked_mul_usize(a: usize, b: usize) -> AdversarialResult<usize> {
        a.checked_mul(b)
            .ok_or_else(|| AdversarialError::IntegerOverflow { operation: format!("{a} * {b}") })
    }

    /// Division with zero check (F1013)
    pub fn checked_div_f64(a: f64, b: f64) -> AdversarialResult<f64> {
        if b == 0.0 {
            return Err(AdversarialError::DivisionByZero { numerator: a });
        }
        Ok(a / b)
    }

    /// Division with zero check for integers
    pub fn checked_div_i64(a: i64, b: i64) -> AdversarialResult<i64> {
        if b == 0 {
            return Err(AdversarialError::DivisionByZero { numerator: a as f64 });
        }
        Ok(a / b)
    }
}

/// Monotonic timestamp tracker (F1006)
#[derive(Debug, Clone)]
pub struct MonotonicClock {
    last_timestamp: Option<Instant>,
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock {
    /// Create a new monotonic clock tracker
    pub fn new() -> Self {
        Self { last_timestamp: None }
    }

    /// Record a timestamp and verify monotonicity
    pub fn tick(&mut self) -> AdversarialResult<Instant> {
        let now = Instant::now();

        if let Some(prev) = self.last_timestamp {
            // In Rust, Instant is guaranteed monotonic, but we check anyway
            // for systems with unreliable clock sources
            if now < prev {
                return Err(AdversarialError::ClockSkew { prev, curr: now });
            }
        }

        self.last_timestamp = Some(now);
        Ok(now)
    }

    /// Get elapsed time since last tick
    pub fn elapsed(&self) -> Option<Duration> {
        self.last_timestamp.map(|t| t.elapsed())
    }

    /// Reset the clock
    pub fn reset(&mut self) {
        self.last_timestamp = None;
    }
}

/// Current resource usage statistics
#[derive(Debug, Clone, Copy)]
pub struct ResourceUsage {
    /// Current recursion depth
    pub stack_depth: usize,
    /// Current memory allocated
    pub memory_bytes: usize,
    /// Elapsed time since operation start
    pub elapsed: Option<Duration>,
}

/// Summary of adversarial test results
#[derive(Debug, Clone)]
pub struct AdversarialTestSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests that passed (handled adversarial input correctly)
    pub passed: usize,
    /// Tests that failed (panicked or incorrect behavior)
    pub failed: usize,
    /// Tactics tested
    pub tactics_tested: Vec<AdversarialTactic>,
    /// Errors encountered (expected - means system handled correctly)
    pub errors_handled: Vec<String>,
}

impl AdversarialTestSummary {
    /// Create a new empty summary
    pub fn new() -> Self {
        Self {
            total_tests: 0,
            passed: 0,
            failed: 0,
            tactics_tested: Vec::new(),
            errors_handled: Vec::new(),
        }
    }

    /// Record a passing test
    pub fn record_pass(&mut self, tactic: AdversarialTactic) {
        self.total_tests += 1;
        self.passed += 1;
        if !self.tactics_tested.contains(&tactic) {
            self.tactics_tested.push(tactic);
        }
    }

    /// Record a failing test
    pub fn record_fail(&mut self, tactic: AdversarialTactic, reason: &str) {
        self.total_tests += 1;
        self.failed += 1;
        if !self.tactics_tested.contains(&tactic) {
            self.tactics_tested.push(tactic);
        }
        self.errors_handled.push(reason.to_string());
    }

    /// Record a handled error (this is good - system detected the adversarial input)
    pub fn record_error_handled(&mut self, error: &AdversarialError) {
        self.errors_handled.push(format!("{error:?}"));
    }

    /// Get pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_tests == 0 {
            return 0.0;
        }
        (self.passed as f64) / (self.total_tests as f64) * 100.0
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.total_tests > 0
    }
}

impl Default for AdversarialTestSummary {
    fn default() -> Self {
        Self::new()
    }
}
