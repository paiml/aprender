//! Brick Testing Utilities (PROBAR-SPEC-009)
//!
//! Test utilities for validating Brick implementations from presentar.
//! Provides Playwright-style assertions for Brick verification gates.
//!
//! ## Toyota Way Application
//!
//! - **Jidoka**: Fail-fast on Brick verification failures
//! - **Poka-Yoke**: Type-safe assertion methods prevent invalid tests
//! - **Genchi Genbutsu**: Test actual Brick output, not mocked data
//!
//! ## Example
//!
//! ```ignore
//! use jugar_probar::tui::{BrickAssertion, assert_brick_valid};
//! use presentar_terminal::SparklineBlock;
//!
//! let block = SparklineBlock::new(60);
//! assert_brick_valid(&block).unwrap();
//! ```

use std::fmt;
use std::time::{Duration, Instant};

#[cfg(feature = "compute-blocks")]
use presentar_core::Brick;

/// Error returned when Brick verification fails.
#[derive(Debug, Clone)]
pub struct BrickVerificationError {
    /// Name of the Brick that failed
    pub brick_name: String,
    /// Failed assertions with reasons
    pub failures: Vec<(String, String)>,
    /// Verification duration
    pub duration: Duration,
}

impl fmt::Display for BrickVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Brick '{}' verification failed:", self.brick_name)?;
        for (assertion, reason) in &self.failures {
            writeln!(f, "  - {}: {}", assertion, reason)?;
        }
        writeln!(f, "  (verified in {:?})", self.duration)?;
        Ok(())
    }
}

impl std::error::Error for BrickVerificationError {}

/// Error returned when Brick exceeds its performance budget.
#[derive(Debug, Clone)]
pub struct BudgetExceededError {
    /// Name of the Brick
    pub brick_name: String,
    /// Budget phase that exceeded (collect/layout/render)
    pub phase: String,
    /// Actual duration
    pub actual_ms: f64,
    /// Budget limit
    pub budget_ms: f64,
}

impl fmt::Display for BudgetExceededError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Brick '{}' exceeded {} budget: {:.2}ms > {:.2}ms",
            self.brick_name, self.phase, self.actual_ms, self.budget_ms
        )
    }
}

impl std::error::Error for BudgetExceededError {}

/// Result of a single assertion check.
#[derive(Debug, Clone)]
pub struct BrickAssertionResult {
    /// Assertion name/description
    pub name: String,
    /// Whether the assertion passed
    pub passed: bool,
    /// Failure reason if applicable
    pub reason: Option<String>,
}

/// Brick test assertion builder (Playwright-style API).
///
/// Provides fluent assertions for testing Brick implementations.
///
/// ## Example
///
/// ```ignore
/// use jugar_probar::tui::BrickTestAssertion;
///
/// BrickTestAssertion::new(&my_brick)
///     .to_be_valid()
///     .to_have_budget_under(16)
///     .to_pass_all_assertions();
/// ```
#[derive(Debug)]
pub struct BrickTestAssertion<'a, B> {
    brick: &'a B,
    soft: bool,
    errors: Vec<String>,
}

#[cfg(feature = "compute-blocks")]
#[allow(clippy::panic)] // Intentional panics for test assertions
impl<'a, B: Brick> BrickTestAssertion<'a, B> {
    /// Create a new Brick assertion.
    pub fn new(brick: &'a B) -> Self {
        Self {
            brick,
            soft: false,
            errors: Vec::new(),
        }
    }

    /// Enable soft assertions (collect errors instead of failing immediately).
    #[must_use]
    pub fn soft(mut self) -> Self {
        self.soft = true;
        self
    }

    /// Assert the Brick passes verification.
    pub fn to_be_valid(&mut self) -> &mut Self {
        let verification = self.brick.verify();
        if !verification.is_valid() {
            let msg = format!(
                "Brick '{}' failed verification: {:?}",
                self.brick.brick_name(),
                &verification.failed
            );
            if self.soft {
                self.errors.push(msg);
            } else {
                panic!("{}", msg);
            }
        }
        self
    }

    /// Assert the Brick's total budget is under the given milliseconds.
    pub fn to_have_budget_under(&mut self, max_ms: u32) -> &mut Self {
        let budget = self.brick.budget();
        let total = budget.total_ms;
        if total > max_ms {
            let msg = format!(
                "Brick '{}' budget {}ms exceeds limit {}ms",
                self.brick.brick_name(),
                total,
                max_ms
            );
            if self.soft {
                self.errors.push(msg);
            } else {
                panic!("{}", msg);
            }
        }
        self
    }

    /// Assert all Brick assertions pass.
    pub fn to_pass_all_assertions(&mut self) -> &mut Self {
        let verification = self.brick.verify();
        let failed = &verification.failed;
        if !failed.is_empty() {
            let msg = format!(
                "Brick '{}' has {} failed assertions: {:?}",
                self.brick.brick_name(),
                failed.len(),
                failed
            );
            if self.soft {
                self.errors.push(msg);
            } else {
                panic!("{}", msg);
            }
        }
        self
    }

    /// Assert the Brick can render (Jidoka gate passes).
    pub fn to_be_renderable(&mut self) -> &mut Self {
        if !self.brick.can_render() {
            let msg = format!(
                "Brick '{}' cannot render (Jidoka gate failed)",
                self.brick.brick_name()
            );
            if self.soft {
                self.errors.push(msg);
            } else {
                panic!("{}", msg);
            }
        }
        self
    }

    /// Get collected errors (for soft assertions).
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Assert no errors were collected (for soft assertions).
    pub fn assert_no_errors(&self) {
        if !self.errors.is_empty() {
            panic!(
                "Brick '{}' had {} soft assertion failures:\n{}",
                self.brick.brick_name(),
                self.errors.len(),
                self.errors.join("\n")
            );
        }
    }
}

/// Assert a Brick passes verification.
///
/// ## Example
///
/// ```ignore
/// use jugar_probar::tui::assert_brick_valid;
///
/// assert_brick_valid(&my_brick).unwrap();
/// ```
#[cfg(feature = "compute-blocks")]
pub fn assert_brick_valid<B: Brick>(brick: &B) -> Result<(), BrickVerificationError> {
    let start = Instant::now();
    let verification = brick.verify();
    let duration = start.elapsed();

    if verification.is_valid() {
        Ok(())
    } else {
        Err(BrickVerificationError {
            brick_name: brick.brick_name().to_string(),
            failures: verification
                .failed
                .iter()
                .map(|(a, r)| (format!("{:?}", a), r.clone()))
                .collect(),
            duration,
        })
    }
}

/// Assert a Brick's execution time is within budget.
///
/// ## Example
///
/// ```ignore
/// use jugar_probar::tui::assert_brick_budget;
///
/// assert_brick_budget(&my_brick, || {
///     my_brick.render();
/// }, "render").unwrap();
/// ```
#[cfg(feature = "compute-blocks")]
pub fn assert_brick_budget<B: Brick, F: FnOnce()>(
    brick: &B,
    operation: F,
    phase: &str,
) -> Result<Duration, BudgetExceededError> {
    let budget = brick.budget();
    let limit_ms = match phase {
        "measure" => budget.measure_ms as f64,
        "layout" => budget.layout_ms as f64,
        "paint" => budget.paint_ms as f64,
        "total" => budget.total_ms as f64,
        _ => budget.total_ms as f64,
    };

    let start = Instant::now();
    operation();
    let duration = start.elapsed();
    let actual_ms = duration.as_secs_f64() * 1000.0;

    if actual_ms <= limit_ms {
        Ok(duration)
    } else {
        Err(BudgetExceededError {
            brick_name: brick.brick_name().to_string(),
            phase: phase.to_string(),
            actual_ms,
            budget_ms: limit_ms,
        })
    }
}

/// Measure Brick verification score (0.0 - 1.0).
///
/// Returns the ratio of passed assertions to total assertions.
#[cfg(feature = "compute-blocks")]
pub fn brick_verification_score<B: Brick>(brick: &B) -> f64 {
    let verification = brick.verify();
    f64::from(verification.score())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brick_verification_error_display() {
        let err = BrickVerificationError {
            brick_name: "TestBrick".to_string(),
            failures: vec![("MinWidth".to_string(), "too narrow".to_string())],
            duration: Duration::from_millis(5),
        };
        let display = format!("{}", err);
        assert!(display.contains("TestBrick"));
        assert!(display.contains("MinWidth"));
        assert!(display.contains("too narrow"));
    }

    #[test]
    fn test_brick_verification_error_multiple_failures() {
        let err = BrickVerificationError {
            brick_name: "ComplexBrick".to_string(),
            failures: vec![
                ("MinWidth".to_string(), "too narrow".to_string()),
                ("MinHeight".to_string(), "too short".to_string()),
                ("Contrast".to_string(), "insufficient".to_string()),
            ],
            duration: Duration::from_micros(500),
        };
        let display = format!("{}", err);
        assert!(display.contains("ComplexBrick"));
        assert!(display.contains("MinWidth"));
        assert!(display.contains("MinHeight"));
        assert!(display.contains("Contrast"));
    }

    #[test]
    fn test_brick_verification_error_is_error() {
        let err = BrickVerificationError {
            brick_name: "Test".to_string(),
            failures: vec![],
            duration: Duration::ZERO,
        };
        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_budget_exceeded_error_display() {
        let err = BudgetExceededError {
            brick_name: "TestBrick".to_string(),
            phase: "render".to_string(),
            actual_ms: 20.5,
            budget_ms: 16.0,
        };
        let display = format!("{}", err);
        assert!(display.contains("TestBrick"));
        assert!(display.contains("render"));
        assert!(display.contains("20.50"));
        assert!(display.contains("16.00"));
    }

    #[test]
    fn test_budget_exceeded_error_phases() {
        for phase in ["measure", "layout", "paint", "total"] {
            let err = BudgetExceededError {
                brick_name: "PhaseBrick".to_string(),
                phase: phase.to_string(),
                actual_ms: 25.0,
                budget_ms: 10.0,
            };
            let display = format!("{}", err);
            assert!(display.contains(phase));
            assert!(display.contains("25.00"));
            assert!(display.contains("10.00"));
        }
    }

    #[test]
    fn test_budget_exceeded_error_is_error() {
        let err = BudgetExceededError {
            brick_name: "Test".to_string(),
            phase: "render".to_string(),
            actual_ms: 1.0,
            budget_ms: 0.5,
        };
        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_assertion_result() {
        let result = BrickAssertionResult {
            name: "MinWidth".to_string(),
            passed: true,
            reason: None,
        };
        assert!(result.passed);
        assert!(result.reason.is_none());
    }

    #[test]
    fn test_assertion_result_failed() {
        let result = BrickAssertionResult {
            name: "ContrastRatio".to_string(),
            passed: false,
            reason: Some("Ratio 3.2 below minimum 4.5".to_string()),
        };
        assert!(!result.passed);
        assert_eq!(
            result.reason.as_deref(),
            Some("Ratio 3.2 below minimum 4.5")
        );
    }

    #[test]
    fn test_assertion_result_debug() {
        let result = BrickAssertionResult {
            name: "Test".to_string(),
            passed: true,
            reason: None,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("BrickAssertionResult"));
        assert!(debug.contains("Test"));
    }

    #[test]
    fn test_assertion_result_clone() {
        let result = BrickAssertionResult {
            name: "Original".to_string(),
            passed: true,
            reason: Some("reason".to_string()),
        };
        let cloned = result.clone();
        assert_eq!(result.name, cloned.name);
        assert_eq!(result.passed, cloned.passed);
        assert_eq!(result.reason, cloned.reason);
    }
}

#[cfg(all(test, feature = "compute-blocks"))]
mod compute_block_tests {
    use super::*;
    use presentar_core::{BrickAssertion, BrickBudget, BrickVerification};

    // Mock Brick for testing
    struct MockBrick {
        name: &'static str,
        valid: bool,
        budget: BrickBudget,
        can_render: bool,
    }

    impl MockBrick {
        fn new_valid() -> Self {
            Self {
                name: "MockBrick",
                valid: true,
                budget: BrickBudget {
                    measure_ms: 4,
                    layout_ms: 4,
                    paint_ms: 4,
                    total_ms: 16,
                },
                can_render: true,
            }
        }

        fn new_invalid() -> Self {
            Self {
                name: "InvalidBrick",
                valid: false,
                budget: BrickBudget {
                    measure_ms: 4,
                    layout_ms: 4,
                    paint_ms: 4,
                    total_ms: 16,
                },
                can_render: false,
            }
        }

        fn with_budget(mut self, total_ms: u32) -> Self {
            self.budget.total_ms = total_ms;
            self
        }
    }

    impl Brick for MockBrick {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            self.budget
        }

        fn verify(&self) -> BrickVerification {
            if self.valid {
                BrickVerification {
                    passed: vec![BrickAssertion::TextVisible],
                    failed: vec![],
                    verification_time: Duration::from_micros(100),
                }
            } else {
                BrickVerification {
                    passed: vec![],
                    failed: vec![(BrickAssertion::TextVisible, "Text not visible".to_string())],
                    verification_time: Duration::from_micros(100),
                }
            }
        }

        fn to_html(&self) -> String {
            "<div>Mock</div>".to_string()
        }

        fn to_css(&self) -> String {
            ".mock {}".to_string()
        }

        fn can_render(&self) -> bool {
            self.can_render
        }
    }

    #[test]
    fn test_brick_test_assertion_new() {
        let brick = MockBrick::new_valid();
        let assertion = BrickTestAssertion::new(&brick);
        assert!(!assertion.soft);
        assert!(assertion.errors.is_empty());
    }

    #[test]
    fn test_brick_test_assertion_soft() {
        let brick = MockBrick::new_valid();
        let assertion = BrickTestAssertion::new(&brick).soft();
        assert!(assertion.soft);
    }

    #[test]
    fn test_brick_test_assertion_to_be_valid_passes() {
        let brick = MockBrick::new_valid();
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion.to_be_valid();
        assert!(assertion.errors.is_empty());
    }

    #[test]
    fn test_brick_test_assertion_to_be_valid_soft_collects_error() {
        let brick = MockBrick::new_invalid();
        let mut assertion = BrickTestAssertion::new(&brick).soft();
        assertion.to_be_valid();
        assert_eq!(assertion.errors.len(), 1);
        assert!(assertion.errors[0].contains("InvalidBrick"));
    }

    #[test]
    #[should_panic(expected = "InvalidBrick")]
    fn test_brick_test_assertion_to_be_valid_panics() {
        let brick = MockBrick::new_invalid();
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion.to_be_valid();
    }

    #[test]
    fn test_brick_test_assertion_to_have_budget_under_passes() {
        let brick = MockBrick::new_valid().with_budget(10);
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion.to_have_budget_under(20);
        assert!(assertion.errors.is_empty());
    }

    #[test]
    fn test_brick_test_assertion_to_have_budget_under_soft_collects_error() {
        let brick = MockBrick::new_valid().with_budget(30);
        let mut assertion = BrickTestAssertion::new(&brick).soft();
        assertion.to_have_budget_under(20);
        assert_eq!(assertion.errors.len(), 1);
        assert!(assertion.errors[0].contains("30ms"));
        assert!(assertion.errors[0].contains("20ms"));
    }

    #[test]
    #[should_panic(expected = "budget")]
    fn test_brick_test_assertion_to_have_budget_under_panics() {
        let brick = MockBrick::new_valid().with_budget(30);
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion.to_have_budget_under(20);
    }

    #[test]
    fn test_brick_test_assertion_to_pass_all_assertions_passes() {
        let brick = MockBrick::new_valid();
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion.to_pass_all_assertions();
        assert!(assertion.errors.is_empty());
    }

    #[test]
    fn test_brick_test_assertion_to_pass_all_assertions_soft_collects_error() {
        let brick = MockBrick::new_invalid();
        let mut assertion = BrickTestAssertion::new(&brick).soft();
        assertion.to_pass_all_assertions();
        assert_eq!(assertion.errors.len(), 1);
        assert!(assertion.errors[0].contains("1 failed"));
    }

    #[test]
    #[should_panic(expected = "failed assertions")]
    fn test_brick_test_assertion_to_pass_all_assertions_panics() {
        let brick = MockBrick::new_invalid();
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion.to_pass_all_assertions();
    }

    #[test]
    fn test_brick_test_assertion_to_be_renderable_passes() {
        let brick = MockBrick::new_valid();
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion.to_be_renderable();
        assert!(assertion.errors.is_empty());
    }

    #[test]
    fn test_brick_test_assertion_to_be_renderable_soft_collects_error() {
        let brick = MockBrick::new_invalid();
        let mut assertion = BrickTestAssertion::new(&brick).soft();
        assertion.to_be_renderable();
        assert_eq!(assertion.errors.len(), 1);
        assert!(assertion.errors[0].contains("cannot render"));
    }

    #[test]
    #[should_panic(expected = "cannot render")]
    fn test_brick_test_assertion_to_be_renderable_panics() {
        let brick = MockBrick::new_invalid();
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion.to_be_renderable();
    }

    #[test]
    fn test_brick_test_assertion_errors() {
        let brick = MockBrick::new_invalid();
        let mut assertion = BrickTestAssertion::new(&brick).soft();
        assertion.to_be_valid();
        assertion.to_be_renderable();
        assert_eq!(assertion.errors().len(), 2);
    }

    #[test]
    fn test_brick_test_assertion_assert_no_errors_passes() {
        let brick = MockBrick::new_valid();
        let assertion = BrickTestAssertion::new(&brick).soft();
        assertion.assert_no_errors(); // Should not panic
    }

    #[test]
    #[should_panic(expected = "soft assertion failures")]
    fn test_brick_test_assertion_assert_no_errors_panics() {
        let brick = MockBrick::new_invalid();
        let mut assertion = BrickTestAssertion::new(&brick).soft();
        assertion.to_be_valid();
        assertion.assert_no_errors();
    }

    #[test]
    fn test_brick_test_assertion_chaining() {
        let brick = MockBrick::new_valid().with_budget(10);
        let mut assertion = BrickTestAssertion::new(&brick);
        assertion
            .to_be_valid()
            .to_have_budget_under(20)
            .to_pass_all_assertions()
            .to_be_renderable();
        assert!(assertion.errors.is_empty());
    }

    #[test]
    fn test_assert_brick_valid_passes() {
        let brick = MockBrick::new_valid();
        let result = assert_brick_valid(&brick);
        assert!(result.is_ok());
    }

    #[test]
    fn test_assert_brick_valid_fails() {
        let brick = MockBrick::new_invalid();
        let result = assert_brick_valid(&brick);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.brick_name, "InvalidBrick");
        assert!(!err.failures.is_empty());
    }

    #[test]
    fn test_assert_brick_budget_passes() {
        let brick = MockBrick::new_valid().with_budget(1000);
        let result = assert_brick_budget(
            &brick,
            || {
                // Fast operation
                std::hint::black_box(1 + 1);
            },
            "total",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_assert_brick_budget_measure_phase() {
        let brick = MockBrick::new_valid();
        let result = assert_brick_budget(&brick, || {}, "measure");
        assert!(result.is_ok());
    }

    #[test]
    fn test_assert_brick_budget_layout_phase() {
        let brick = MockBrick::new_valid();
        let result = assert_brick_budget(&brick, || {}, "layout");
        assert!(result.is_ok());
    }

    #[test]
    fn test_assert_brick_budget_paint_phase() {
        let brick = MockBrick::new_valid();
        let result = assert_brick_budget(&brick, || {}, "paint");
        assert!(result.is_ok());
    }

    #[test]
    fn test_assert_brick_budget_unknown_phase() {
        let brick = MockBrick::new_valid();
        let result = assert_brick_budget(&brick, || {}, "unknown");
        assert!(result.is_ok()); // Falls back to total_ms
    }

    #[test]
    fn test_brick_verification_score_valid() {
        let brick = MockBrick::new_valid();
        let score = brick_verification_score(&brick);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_brick_verification_score_invalid() {
        let brick = MockBrick::new_invalid();
        let score = brick_verification_score(&brick);
        assert!(score >= 0.0);
        assert!(score < 1.0);
    }
}
