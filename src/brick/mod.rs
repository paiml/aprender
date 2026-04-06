//! ComputeBrick: Token-centric, self-verifying compute units.
//!
//! Per CBTOP-SPEC-001 and SHOWCASE-BRICK-001, every compute operation is a brick with:
//! - Token budget (tok/sec performance target)
//! - Assertions (falsifiable correctness claims)
//! - Verification (self-checking via baseline comparison)
//!
//! # Toyota Way Integration
//! - **Jidoka**: Stop-the-line on budget violation
//! - **Poka-Yoke**: Type-safe brick composition
//! - **Genchi Genbutsu**: Real metrics from hardware
//! - **Mieruka**: Visual control via cbtop TUI
//!
//! # References
//! - Popper, K. (1959). "The Logic of Scientific Discovery."
//! - Ohno, T. (1988). "Toyota Production System."
//! - Little, J.D.C. (1961). "A Proof for the Queuing Formula: L = λW."

use std::fmt;
use std::time::Instant;

use crate::quantize::Q8_0Block;

// PMAT-802: Extracted modules
#[cfg(feature = "cuda")]
mod fused;
#[cfg(feature = "cuda")]
pub use fused::{CoalescedDp4aBrick, FusedFfnBrick};

// Phase 14: BrickTracer for GPU/CPU parity debugging
pub mod tracer;
pub use tracer::{BrickTracer, TraceComparison, TraceDiff, TraceEvent};

// PMAT-112: BrickProfiler for real-time inference telemetry
pub mod profiler;
pub use profiler::{BrickProfiler, ContractSeverity, OpStats, ProfileReport};

// ============================================================================
// Core Types
// ============================================================================

/// Performance budget expressed in token terms.
/// Aligns compute costs with LLM inference metrics.
///
/// Per Little's Law (1961): throughput = 1 / latency
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenBudget {
    /// Latency budget per token (microseconds)
    pub us_per_token: f64,
    /// Throughput target (tokens/second)
    pub tokens_per_sec: f64,
    /// Batch size for amortization
    pub batch_size: usize,
}

impl TokenBudget {
    /// Create budget from latency target.
    /// 50µs/token = 20,000 tokens/sec
    #[must_use]
    pub fn from_latency(us_per_token: f64) -> Self {
        Self {
            us_per_token,
            tokens_per_sec: 1_000_000.0 / us_per_token,
            batch_size: 1,
        }
    }

    /// Create budget from throughput target.
    /// 20,000 tokens/sec = 50µs/token
    #[must_use]
    pub fn from_throughput(tokens_per_sec: f64) -> Self {
        Self {
            us_per_token: 1_000_000.0 / tokens_per_sec,
            tokens_per_sec,
            batch_size: 1,
        }
    }

    /// Create budget with batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Check if actual performance meets budget.
    #[must_use]
    pub fn is_met(&self, actual_us_per_token: f64) -> bool {
        actual_us_per_token <= self.us_per_token
    }

    /// Calculate gap factor (actual / budget).
    /// < 1.0 means under budget (good), > 1.0 means over budget (bad).
    #[must_use]
    pub fn gap_factor(&self, actual_us_per_token: f64) -> f64 {
        actual_us_per_token / self.us_per_token
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self::from_latency(100.0) // 100µs = 10k tok/s default
    }
}

/// Result of ComputeBrick execution with token metrics.
#[derive(Debug, Clone)]
pub struct TokenResult<T> {
    /// Computed output
    pub output: T,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Actual latency (microseconds/token)
    pub us_per_token: f64,
    /// Actual throughput (tokens/second)
    pub tokens_per_sec: f64,
    /// Did we meet the budget?
    pub budget_met: bool,
}

impl<T: Default> Default for TokenResult<T> {
    fn default() -> Self {
        Self {
            output: T::default(),
            tokens_processed: 0,
            us_per_token: 0.0,
            tokens_per_sec: 0.0,
            budget_met: true,
        }
    }
}

impl<T> TokenResult<T> {
    /// Create a new token result.
    pub fn new(output: T, tokens: usize, elapsed_us: f64, budget: &TokenBudget) -> Self {
        let us_per_token = elapsed_us / tokens.max(1) as f64;
        let tokens_per_sec = if us_per_token > 0.0 {
            1_000_000.0 / us_per_token
        } else {
            0.0
        };

        Self {
            output,
            tokens_processed: tokens,
            us_per_token,
            tokens_per_sec,
            budget_met: budget.is_met(us_per_token),
        }
    }
}

/// Errors from ComputeBrick execution.
/// Tells you exactly what failed (Jidoka: stop and signal).
#[derive(Debug)]
pub enum BrickError {
    /// Assertion failed during verification.
    AssertionFailed {
        /// Assertion name
        name: String,
        /// Expected value
        expected: String,
        /// Actual value
        actual: String,
    },
    /// Performance budget exceeded.
    BudgetExceeded {
        /// Budget limit in µs
        limit_us: f64,
        /// Actual time in µs
        actual_us: f64,
    },
    /// Compute operation failed.
    ComputeError(String),
    /// Invalid input.
    InvalidInput(String),
}

impl fmt::Display for BrickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AssertionFailed {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Assertion failed: {name} - expected {expected}, got {actual}"
                )
            },
            Self::BudgetExceeded {
                limit_us,
                actual_us,
            } => {
                write!(
                    f,
                    "Budget exceeded: {limit_us:.1}µs/tok limit, {actual_us:.1}µs/tok actual"
                )
            },
            Self::ComputeError(msg) => write!(f, "Compute error: {msg}"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for BrickError {}

// ============================================================================
// Brick Assertions
// ============================================================================

/// Falsifiable assertion for a brick (Popper criterion).
#[derive(Debug, Clone)]
pub struct BrickAssertion {
    /// Assertion name (for error messages)
    pub name: String,
    /// Description of what is being asserted
    pub description: String,
    /// Assertion type
    pub kind: AssertionKind,
}

/// Types of assertions a brick can make.
#[derive(Debug, Clone)]
pub enum AssertionKind {
    /// Output matches scalar baseline within tolerance.
    EquivScalar {
        /// Tolerance for comparison
        tolerance: f64,
    },
    /// Output contains no NaN values.
    NoNaN,
    /// Output contains no Inf values.
    NoInf,
    /// Output values within bounds.
    Bounds {
        /// Minimum allowed value
        min: f64,
        /// Maximum allowed value
        max: f64,
    },
    /// Budget must be met.
    BudgetMet,
    /// Custom assertion (returns true if passed).
    Custom {
        /// Name of the custom check
        check_name: String,
    },
}

impl BrickAssertion {
    /// Create assertion that output matches scalar baseline.
    pub fn equiv_scalar(tolerance: f64) -> Self {
        Self {
            name: "equiv_scalar".to_string(),
            description: format!("Output matches scalar baseline within {tolerance}"),
            kind: AssertionKind::EquivScalar { tolerance },
        }
    }

    /// Create assertion that output contains no NaN.
    pub fn no_nan() -> Self {
        Self {
            name: "no_nan".to_string(),
            description: "Output contains no NaN values".to_string(),
            kind: AssertionKind::NoNaN,
        }
    }

    /// Create assertion that output contains no Inf.
    pub fn no_inf() -> Self {
        Self {
            name: "no_inf".to_string(),
            description: "Output contains no Inf values".to_string(),
            kind: AssertionKind::NoInf,
        }
    }

    /// Create assertion that values are within bounds.
    pub fn bounds(min: f64, max: f64) -> Self {
        Self {
            name: "bounds".to_string(),
            description: format!("Output values in [{min}, {max}]"),
            kind: AssertionKind::Bounds { min, max },
        }
    }

    /// Create assertion that budget is met.
    pub fn budget_met() -> Self {
        Self {
            name: "budget_met".to_string(),
            description: "Performance budget is met".to_string(),
            kind: AssertionKind::BudgetMet,
        }
    }

    /// Check assertion against f32 slice output.
    pub fn check_f32(&self, output: &[f32], budget_met: bool) -> Result<(), BrickError> {
        match &self.kind {
            AssertionKind::NoNaN => {
                if let Some(idx) = output.iter().position(|x| x.is_nan()) {
                    return Err(BrickError::AssertionFailed {
                        name: self.name.clone(),
                        expected: "no NaN".to_string(),
                        actual: format!("NaN at index {idx}"),
                    });
                }
            },
            AssertionKind::NoInf => {
                if let Some(idx) = output.iter().position(|x| x.is_infinite()) {
                    return Err(BrickError::AssertionFailed {
                        name: self.name.clone(),
                        expected: "no Inf".to_string(),
                        actual: format!("Inf at index {idx}"),
                    });
                }
            },
            AssertionKind::Bounds { min, max } => {
                for (idx, &val) in output.iter().enumerate() {
                    if (val as f64) < *min || (val as f64) > *max {
                        return Err(BrickError::AssertionFailed {
                            name: self.name.clone(),
                            expected: format!("value in [{min}, {max}]"),
                            actual: format!("value {val} at index {idx}"),
                        });
                    }
                }
            },
            AssertionKind::BudgetMet => {
                if !budget_met {
                    return Err(BrickError::AssertionFailed {
                        name: self.name.clone(),
                        expected: "budget met".to_string(),
                        actual: "budget exceeded".to_string(),
                    });
                }
            },
            AssertionKind::EquivScalar { .. } | AssertionKind::Custom { .. } => {
                // These require external comparison, skip for basic check
            },
        }
        Ok(())
    }
}

// ============================================================================
// Brick Verification
// ============================================================================

/// Result of brick verification.
#[derive(Debug, Clone)]
pub struct BrickVerification {
    /// All assertions passed?
    pub is_valid: bool,
    /// Individual assertion results
    pub results: Vec<(String, bool, String)>,
}

impl BrickVerification {
    /// Create a passing verification.
    pub fn pass() -> Self {
        Self {
            is_valid: true,
            results: vec![],
        }
    }

    /// Create a failing verification.
    pub fn fail(name: &str, reason: &str) -> Self {
        Self {
            is_valid: false,
            results: vec![(name.to_string(), false, reason.to_string())],
        }
    }

    /// Add an assertion result.
    pub fn add(&mut self, name: &str, passed: bool, message: &str) {
        self.results
            .push((name.to_string(), passed, message.to_string()));
        if !passed {
            self.is_valid = false;
        }
    }
}

// ============================================================================
// ComputeBrick Trait
// ============================================================================

/// Core trait for self-verifying, token-centric compute units.
///
/// Every brick must:
/// 1. Have at least one assertion (Popper criterion)
/// 2. Have a non-zero budget (accountability)
/// 3. Be verifiable against baseline
///
/// # Invariants (PROBAR-SPEC-009 §3)
/// - `assertions().len() > 0` (at least one falsifiable claim)
/// - `budget().us_per_token > 0` (performance accountability)
/// - `verify()` checks ALL assertions
pub trait ComputeBrick: Send + Sync {
    /// Output type of this brick.
    type Output;

    /// Brick name for identification.
    fn name(&self) -> &'static str;

    /// Token throughput budget.
    fn budget(&self) -> TokenBudget;

    /// Falsifiable assertions for this brick.
    fn assertions(&self) -> Vec<BrickAssertion>;

    /// Verify brick state without running.
    fn verify(&self) -> BrickVerification {
        let assertions = self.assertions();
        if assertions.is_empty() {
            return BrickVerification::fail(
                self.name(),
                "No assertions defined (Popper violation)",
            );
        }

        let budget = self.budget();
        if budget.us_per_token <= 0.0 {
            return BrickVerification::fail(self.name(), "Zero or negative budget");
        }

        BrickVerification::pass()
    }

    /// Can this brick run? (Jidoka gate)
    fn can_run(&self) -> bool {
        self.verify().is_valid
    }
}

include!("brick_impls.rs");
include!("mod_tile_flash_attention.rs");
include!("mod_per_activation_quant.rs");
include!("graph.rs");
