//! Native Brick types for presentar-core.
//!
//! These types define the Brick Architecture interface natively, eliminating
//! the heavyweight jugar-probar dependency from the presentar-core production
//! build. The types are API-compatible with `jugar_probar::brick::*`.
//!
//! # Design Rationale (CB-081)
//!
//! jugar-probar is a test framework with heavy transitive deps (image, rav1e,
//! mp4, etc.). Pulling it as a production dependency inflates all downstream
//! consumers. By defining the Brick trait natively in presentar-core, we keep
//! the architecture while eliminating ~100 transitive dependencies.

use std::time::Duration;

/// Brick assertion that must be verified at runtime.
///
/// Assertions are falsifiable hypotheses about the UI state.
/// If any assertion fails, the brick is falsified.
#[derive(Debug, Clone, PartialEq)]
pub enum BrickAssertion {
    /// Text content must be visible (not hidden, not zero-opacity)
    TextVisible,

    /// WCAG 2.1 AA contrast ratio requirement (4.5:1 for normal text)
    ContrastRatio(f32),

    /// Maximum render latency in milliseconds
    MaxLatencyMs(u32),

    /// Element must be present in DOM
    ElementPresent(String),

    /// Element must be focusable for accessibility
    Focusable,

    /// Custom assertion with name and validation function ID
    Custom {
        /// Assertion name for error reporting
        name: String,
        /// Validation function identifier
        validator_id: u64,
    },
}

impl BrickAssertion {
    /// Create a text visibility assertion
    #[must_use]
    pub const fn text_visible() -> Self {
        Self::TextVisible
    }

    /// Create a contrast ratio assertion (WCAG 2.1 AA)
    #[must_use]
    pub const fn contrast_ratio(ratio: f32) -> Self {
        Self::ContrastRatio(ratio)
    }

    /// Create a max latency assertion
    #[must_use]
    pub const fn max_latency_ms(ms: u32) -> Self {
        Self::MaxLatencyMs(ms)
    }

    /// Create an element presence assertion
    #[must_use]
    pub fn element_present(selector: impl Into<String>) -> Self {
        Self::ElementPresent(selector.into())
    }
}

/// Performance budget for a brick.
///
/// Budgets are enforced at runtime. Exceeding the budget triggers
/// a Jidoka (stop-the-line) alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickBudget {
    /// Maximum time for measure phase
    pub measure_ms: u32,
    /// Maximum time for layout phase
    pub layout_ms: u32,
    /// Maximum time for paint phase
    pub paint_ms: u32,
    /// Total budget (may be less than sum of phases)
    pub total_ms: u32,
}

impl BrickBudget {
    /// Create a budget with equal distribution across phases
    #[must_use]
    pub const fn uniform(total_ms: u32) -> Self {
        let phase_ms = total_ms / 3;
        Self {
            measure_ms: phase_ms,
            layout_ms: phase_ms,
            paint_ms: phase_ms,
            total_ms,
        }
    }

    /// Create a custom budget with specified phase limits
    #[must_use]
    pub const fn new(measure_ms: u32, layout_ms: u32, paint_ms: u32) -> Self {
        Self {
            measure_ms,
            layout_ms,
            paint_ms,
            total_ms: measure_ms + layout_ms + paint_ms,
        }
    }

    /// Convert to Duration
    #[must_use]
    pub const fn as_duration(&self) -> Duration {
        Duration::from_millis(self.total_ms as u64)
    }
}

impl Default for BrickBudget {
    fn default() -> Self {
        // Default: 16ms total for 60fps
        Self::uniform(16)
    }
}

/// Result of verifying brick assertions
#[derive(Debug, Clone)]
pub struct BrickVerification {
    /// All assertions that passed
    pub passed: Vec<BrickAssertion>,
    /// All assertions that failed with reasons
    pub failed: Vec<(BrickAssertion, String)>,
    /// Time taken to verify
    pub verification_time: Duration,
}

impl BrickVerification {
    /// Check if all assertions passed
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.failed.is_empty()
    }

    /// Get the falsification score (passed / total)
    #[must_use]
    pub fn score(&self) -> f32 {
        let total = self.passed.len() + self.failed.len();
        if total == 0 {
            1.0
        } else {
            self.passed.len() as f32 / total as f32
        }
    }
}

/// Budget violation report
#[derive(Debug, Clone)]
pub struct BudgetViolation {
    /// Name of the brick that violated
    pub brick_name: String,
    /// Budget that was exceeded
    pub budget: BrickBudget,
    /// Actual time taken
    pub actual: Duration,
    /// Phase that exceeded (if known)
    pub phase: Option<BrickPhase>,
}

/// Rendering phase for budget tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickPhase {
    /// Measure phase (compute intrinsic size)
    Measure,
    /// Layout phase (position children)
    Layout,
    /// Paint phase (generate draw commands)
    Paint,
}

/// Core Brick trait - the foundation of the Brick Architecture.
///
/// All UI components implement this trait. The trait defines:
/// 1. Assertions that must pass for the brick to be valid
/// 2. Performance budget that must not be exceeded
/// 3. HTML/CSS generation for rendering targets
///
/// # Trait Bound
///
/// Presentar's `Widget` trait requires `Brick`:
/// ```rust,ignore
/// pub trait Widget: Brick + Send + Sync { ... }
/// ```
///
/// This ensures every widget has verifiable assertions and budgets.
pub trait Brick: Send + Sync {
    /// Get the brick's unique type name
    fn brick_name(&self) -> &'static str;

    /// Get all assertions for this brick
    fn assertions(&self) -> &[BrickAssertion];

    /// Get the performance budget
    fn budget(&self) -> BrickBudget;

    /// Verify all assertions against current state
    ///
    /// Returns a verification result with passed/failed assertions.
    fn verify(&self) -> BrickVerification;

    /// Generate HTML for this brick (WASM target)
    ///
    /// Returns the HTML string that represents this brick.
    /// Must be deterministic (same state -> same output).
    fn to_html(&self) -> String;

    /// Generate CSS for this brick (WASM target)
    ///
    /// Returns the CSS rules for styling this brick.
    /// Must be deterministic and scoped to avoid conflicts.
    fn to_css(&self) -> String;

    /// Get the test ID for DOM queries
    fn test_id(&self) -> Option<&str> {
        None
    }

    /// Check if this brick can be rendered (all assertions pass)
    fn can_render(&self) -> bool {
        self.verify().is_valid()
    }
}

/// Yuan Gate: Zero-swallow error handling for bricks
///
/// Named after the Yuan dynasty's strict quality standards.
/// Every error must be explicitly handled - no silent drops.
#[derive(Debug, Clone)]
pub enum BrickError {
    /// Assertion failed during verification
    AssertionFailed {
        /// The assertion that failed
        assertion: BrickAssertion,
        /// Reason for failure
        reason: String,
    },

    /// Budget exceeded during rendering
    BudgetExceeded(BudgetViolation),

    /// Invalid state transition
    InvalidTransition {
        /// Current state
        from: String,
        /// Attempted target state
        to: String,
        /// Reason transition is invalid
        reason: String,
    },

    /// Missing required child brick
    MissingChild {
        /// Expected child brick name
        expected: String,
    },

    /// HTML generation failed
    HtmlGenerationFailed {
        /// Reason for failure
        reason: String,
    },
}

impl std::fmt::Display for BrickError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AssertionFailed { assertion, reason } => {
                write!(f, "Assertion {assertion:?} failed: {reason}")
            }
            Self::BudgetExceeded(violation) => {
                write!(
                    f,
                    "Budget exceeded for {}: {:?} > {:?}",
                    violation.brick_name, violation.actual, violation.budget.total_ms
                )
            }
            Self::InvalidTransition { from, to, reason } => {
                write!(f, "Invalid transition {from} -> {to}: {reason}")
            }
            Self::MissingChild { expected } => {
                write!(f, "Missing required child brick: {expected}")
            }
            Self::HtmlGenerationFailed { reason } => {
                write!(f, "HTML generation failed: {reason}")
            }
        }
    }
}

impl std::error::Error for BrickError {}

/// Result type for brick operations
pub type BrickResult<T> = Result<T, BrickError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brick_assertion_text_visible() {
        let a = BrickAssertion::text_visible();
        assert_eq!(a, BrickAssertion::TextVisible);
    }

    #[test]
    fn test_brick_assertion_contrast_ratio() {
        let a = BrickAssertion::contrast_ratio(4.5);
        assert_eq!(a, BrickAssertion::ContrastRatio(4.5));
    }

    #[test]
    fn test_brick_assertion_max_latency() {
        let a = BrickAssertion::max_latency_ms(16);
        assert_eq!(a, BrickAssertion::MaxLatencyMs(16));
    }

    #[test]
    fn test_brick_assertion_element_present() {
        let a = BrickAssertion::element_present(".button");
        assert_eq!(a, BrickAssertion::ElementPresent(".button".into()));
    }

    #[test]
    fn test_brick_budget_uniform() {
        let b = BrickBudget::uniform(16);
        assert_eq!(b.total_ms, 16);
        assert_eq!(b.measure_ms, 5);
        assert_eq!(b.layout_ms, 5);
        assert_eq!(b.paint_ms, 5);
    }

    #[test]
    fn test_brick_budget_custom() {
        let b = BrickBudget::new(4, 4, 8);
        assert_eq!(b.total_ms, 16);
    }

    #[test]
    fn test_brick_budget_default() {
        let b = BrickBudget::default();
        assert_eq!(b.total_ms, 16);
    }

    #[test]
    fn test_brick_budget_as_duration() {
        let b = BrickBudget::uniform(16);
        assert_eq!(b.as_duration(), Duration::from_millis(16));
    }

    #[test]
    fn test_brick_verification_valid() {
        let v = BrickVerification {
            passed: vec![BrickAssertion::TextVisible],
            failed: vec![],
            verification_time: Duration::from_micros(1),
        };
        assert!(v.is_valid());
        assert!((v.score() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_brick_verification_invalid() {
        let v = BrickVerification {
            passed: vec![],
            failed: vec![(BrickAssertion::TextVisible, "not visible".into())],
            verification_time: Duration::from_micros(1),
        };
        assert!(!v.is_valid());
        assert!((v.score() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_brick_verification_empty() {
        let v = BrickVerification {
            passed: vec![],
            failed: vec![],
            verification_time: Duration::from_micros(1),
        };
        assert!(v.is_valid());
        assert!((v.score() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_brick_phase_variants() {
        let phases = [BrickPhase::Measure, BrickPhase::Layout, BrickPhase::Paint];
        assert_eq!(phases.len(), 3);
        assert_ne!(BrickPhase::Measure, BrickPhase::Paint);
    }

    #[test]
    fn test_brick_error_display() {
        let err = BrickError::AssertionFailed {
            assertion: BrickAssertion::TextVisible,
            reason: "hidden".into(),
        };
        let s = format!("{err}");
        assert!(s.contains("TextVisible"));
        assert!(s.contains("hidden"));
    }

    #[test]
    fn test_brick_error_budget_exceeded() {
        let err = BrickError::BudgetExceeded(BudgetViolation {
            brick_name: "test".into(),
            budget: BrickBudget::uniform(16),
            actual: Duration::from_millis(32),
            phase: Some(BrickPhase::Paint),
        });
        let s = format!("{err}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_brick_error_is_error() {
        let err: Box<dyn std::error::Error> = Box::new(BrickError::MissingChild {
            expected: "child".into(),
        });
        assert!(err.to_string().contains("child"));
    }
}
