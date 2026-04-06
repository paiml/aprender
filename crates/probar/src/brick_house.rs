//! BrickHouse: Budgeted Composition of Bricks (PROBAR-SPEC-009)
//!
//! A `BrickHouse` composes multiple bricks with a total performance budget.
//! Individual bricks contribute to the house budget, and the house validates
//! that the sum of brick budgets does not exceed the house budget.
//!
//! # Design Philosophy
//!
//! ```text
//! BrickHouse(1000ms)
//! ├── StatusBrick(50ms)
//! ├── WaveformBrick(100ms)
//! ├── TranscriptionBrick(600ms)
//! └── ControlsBrick(50ms)
//! Total: 800ms < 1000ms budget ✓
//! ```
//!
//! # Jidoka: Stop-the-Line
//!
//! If any brick exceeds its budget, the BrickHouse triggers a Jidoka alert
//! and halts rendering. This prevents cascading performance failures.
//!
//! # Example
//!
//! ```rust,ignore
//! use probar::brick_house::{BrickHouse, BrickHouseBuilder};
//!
//! let house = BrickHouseBuilder::new("whisper-app")
//!     .budget_ms(1000)
//!     .brick(status_brick, 50)
//!     .brick(waveform_brick, 100)
//!     .brick(transcription_brick, 600)
//!     .build()?;
//! ```
//!
//! # References
//!
//! - Toyota Production System: Jidoka (autonomation)
//! - PROBAR-SPEC-009: Bug Hunting Probador

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::brick::{
    Brick, BrickBudget, BrickError, BrickPhase, BrickResult, BrickVerification, BudgetViolation,
};

/// A composed house of bricks with a total performance budget.
///
/// The BrickHouse ensures:
/// 1. Sum of brick budgets ≤ house budget
/// 2. All brick assertions pass before rendering
/// 3. Runtime budget violations trigger Jidoka alerts
#[derive(Debug)]
pub struct BrickHouse {
    /// House name for identification
    name: String,
    /// Total budget for the house
    budget: BrickBudget,
    /// Bricks with their allocated budgets
    bricks: Vec<BrickEntry>,
    /// Budget report from last render
    last_report: Option<BudgetReport>,
}

/// Entry for a brick in the house
struct BrickEntry {
    /// The brick instance
    brick: Arc<dyn Brick>,
    /// Allocated budget (may differ from brick's intrinsic budget)
    allocated_ms: u32,
    /// Last measured render time
    last_render_time: Option<Duration>,
}

impl std::fmt::Debug for BrickEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrickEntry")
            .field("brick_name", &self.brick.brick_name())
            .field("allocated_ms", &self.allocated_ms)
            .field("last_render_time", &self.last_render_time)
            .finish()
    }
}

/// Report on budget usage after a render cycle
#[derive(Debug, Clone)]
pub struct BudgetReport {
    /// House name
    pub house_name: String,
    /// Total budget allocated
    pub total_budget_ms: u32,
    /// Total time used
    pub total_used_ms: u32,
    /// Individual brick timings
    pub brick_timings: HashMap<String, BrickTiming>,
    /// Any budget violations
    pub violations: Vec<BudgetViolation>,
    /// Timestamp of report
    pub timestamp: std::time::SystemTime,
}

/// Timing information for a single brick
#[derive(Debug, Clone)]
pub struct BrickTiming {
    /// Brick name
    pub name: String,
    /// Allocated budget
    pub budget_ms: u32,
    /// Actual time used
    pub used_ms: u32,
    /// Whether budget was exceeded
    pub exceeded: bool,
}

impl BudgetReport {
    /// Check if the house stayed within budget
    #[must_use]
    pub fn within_budget(&self) -> bool {
        self.violations.is_empty() && self.total_used_ms <= self.total_budget_ms
    }

    /// Get budget utilization as percentage
    #[must_use]
    pub fn utilization(&self) -> f32 {
        if self.total_budget_ms == 0 {
            0.0
        } else {
            (self.total_used_ms as f32 / self.total_budget_ms as f32) * 100.0
        }
    }

    /// Get all violations
    #[must_use]
    pub fn violations(&self) -> &[BudgetViolation] {
        &self.violations
    }
}

impl BrickHouse {
    /// Create a new brick house with the given name and budget
    #[must_use]
    pub fn new(name: impl Into<String>, budget_ms: u32) -> Self {
        Self {
            name: name.into(),
            budget: BrickBudget::uniform(budget_ms),
            bricks: Vec::new(),
            last_report: None,
        }
    }

    /// Add a brick with a specific budget allocation
    ///
    /// # Errors
    ///
    /// Returns an error if adding this brick would exceed the house budget.
    pub fn add_brick(&mut self, brick: Arc<dyn Brick>, budget_ms: u32) -> BrickResult<()> {
        let current_total: u32 = self.bricks.iter().map(|b| b.allocated_ms).sum();
        let new_total = current_total + budget_ms;

        if new_total > self.budget.total_ms {
            return Err(BrickError::BudgetExceeded(BudgetViolation {
                brick_name: brick.brick_name().to_string(),
                budget: self.budget,
                actual: Duration::from_millis(new_total as u64),
                phase: None,
            }));
        }

        self.bricks.push(BrickEntry {
            brick,
            allocated_ms: budget_ms,
            last_render_time: None,
        });

        Ok(())
    }

    /// Get the house name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the total budget
    #[must_use]
    pub fn budget(&self) -> BrickBudget {
        self.budget
    }

    /// Get the number of bricks
    #[must_use]
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Get remaining budget after allocations
    #[must_use]
    pub fn remaining_budget_ms(&self) -> u32 {
        let allocated: u32 = self.bricks.iter().map(|b| b.allocated_ms).sum();
        self.budget.total_ms.saturating_sub(allocated)
    }

    /// Verify all bricks in the house
    ///
    /// Returns verification results for all bricks.
    pub fn verify_all(&self) -> Vec<(&str, BrickVerification)> {
        self.bricks
            .iter()
            .map(|entry| {
                let name = entry.brick.brick_name();
                let verification = entry.brick.verify();
                (name, verification)
            })
            .collect()
    }

    /// Check if the house can render (all bricks valid)
    #[must_use]
    pub fn can_render(&self) -> bool {
        self.bricks.iter().all(|entry| entry.brick.can_render())
    }

    /// Render all bricks and track timing
    ///
    /// Returns the generated HTML for all bricks.
    ///
    /// # Errors
    ///
    /// Returns an error if any brick exceeds its budget (Jidoka).
    pub fn render(&mut self) -> BrickResult<String> {
        let mut html_parts = Vec::new();
        let mut timings = HashMap::new();
        let mut violations = Vec::new();
        let mut total_used_ms = 0u32;

        for entry in &mut self.bricks {
            let start = Instant::now();

            // Verify before render
            let verification = entry.brick.verify();
            if !verification.is_valid() {
                let (assertion, reason) = verification
                    .failed
                    .first()
                    .map(|(a, r)| (a.clone(), r.clone()))
                    .unwrap_or_else(|| {
                        (crate::brick::BrickAssertion::TextVisible, "Unknown".into())
                    });
                return Err(BrickError::AssertionFailed { assertion, reason });
            }

            // Generate HTML
            let html = entry.brick.to_html();
            html_parts.push(html);

            let elapsed = start.elapsed();
            let elapsed_ms = elapsed.as_millis() as u32;
            entry.last_render_time = Some(elapsed);
            total_used_ms += elapsed_ms;

            let exceeded = elapsed_ms > entry.allocated_ms;
            let brick_name = entry.brick.brick_name().to_string();

            timings.insert(
                brick_name.clone(),
                BrickTiming {
                    name: brick_name.clone(),
                    budget_ms: entry.allocated_ms,
                    used_ms: elapsed_ms,
                    exceeded,
                },
            );

            if exceeded {
                violations.push(BudgetViolation {
                    brick_name,
                    budget: BrickBudget::uniform(entry.allocated_ms),
                    actual: elapsed,
                    phase: Some(BrickPhase::Paint),
                });
            }
        }

        // Store report
        self.last_report = Some(BudgetReport {
            house_name: self.name.clone(),
            total_budget_ms: self.budget.total_ms,
            total_used_ms,
            brick_timings: timings,
            violations: violations.clone(),
            timestamp: std::time::SystemTime::now(),
        });

        // Jidoka: stop-the-line on violations
        if !violations.is_empty() {
            return Err(BrickError::BudgetExceeded(
                violations.into_iter().next().expect("violations not empty"),
            ));
        }

        Ok(html_parts.join("\n"))
    }

    /// Get the last budget report
    #[must_use]
    pub fn last_report(&self) -> Option<&BudgetReport> {
        self.last_report.as_ref()
    }

    /// Generate combined CSS for all bricks
    #[must_use]
    pub fn to_css(&self) -> String {
        self.bricks
            .iter()
            .map(|entry| entry.brick.to_css())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Builder for constructing a BrickHouse
pub struct BrickHouseBuilder {
    name: String,
    budget_ms: u32,
    bricks: Vec<(Arc<dyn Brick>, u32)>,
}

impl std::fmt::Debug for BrickHouseBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrickHouseBuilder")
            .field("name", &self.name)
            .field("budget_ms", &self.budget_ms)
            .field("brick_count", &self.bricks.len())
            .finish()
    }
}

impl BrickHouseBuilder {
    /// Create a new builder with the given house name
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            budget_ms: 1000, // Default 1 second
            bricks: Vec::new(),
        }
    }

    /// Set the total budget in milliseconds
    #[must_use]
    pub fn budget_ms(mut self, ms: u32) -> Self {
        self.budget_ms = ms;
        self
    }

    /// Add a brick with a specific budget allocation
    #[must_use]
    pub fn brick(mut self, brick: Arc<dyn Brick>, budget_ms: u32) -> Self {
        self.bricks.push((brick, budget_ms));
        self
    }

    /// Build the BrickHouse
    ///
    /// # Errors
    ///
    /// Returns an error if the total brick budgets exceed the house budget.
    pub fn build(self) -> BrickResult<BrickHouse> {
        let total_brick_budget: u32 = self.bricks.iter().map(|(_, ms)| *ms).sum();

        if total_brick_budget > self.budget_ms {
            return Err(BrickError::BudgetExceeded(BudgetViolation {
                brick_name: self.name.clone(),
                budget: BrickBudget::uniform(self.budget_ms),
                actual: Duration::from_millis(total_brick_budget as u64),
                phase: None,
            }));
        }

        let mut house = BrickHouse::new(self.name, self.budget_ms);
        for (brick, budget) in self.bricks {
            house.add_brick(brick, budget)?;
        }

        Ok(house)
    }
}

/// Jidoka alert for budget violations
///
/// This struct captures the context when a brick exceeds its budget,
/// enabling root cause analysis.
#[derive(Debug, Clone)]
pub struct JidokaAlert {
    /// The brick house that triggered the alert
    pub house_name: String,
    /// The specific brick that exceeded budget
    pub brick_name: String,
    /// Budget that was exceeded
    pub budget_ms: u32,
    /// Actual time taken
    pub actual_ms: u32,
    /// Phase where violation occurred
    pub phase: Option<BrickPhase>,
    /// Timestamp of alert
    pub timestamp: std::time::SystemTime,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
}

impl JidokaAlert {
    /// Create a new alert from a budget violation
    #[must_use]
    pub fn from_violation(house_name: &str, violation: &BudgetViolation) -> Self {
        Self {
            house_name: house_name.to_string(),
            brick_name: violation.brick_name.clone(),
            budget_ms: violation.budget.total_ms,
            actual_ms: violation.actual.as_millis() as u32,
            phase: violation.phase,
            timestamp: std::time::SystemTime::now(),
            stack_trace: None,
        }
    }

    /// Get the overage percentage
    #[must_use]
    pub fn overage_percent(&self) -> f32 {
        if self.budget_ms == 0 {
            0.0
        } else {
            ((self.actual_ms as f32 / self.budget_ms as f32) - 1.0) * 100.0
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::brick::BrickAssertion;

    struct SimpleBrick {
        name: &'static str,
    }

    impl Brick for SimpleBrick {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(1),
            }
        }

        fn to_html(&self) -> String {
            format!("<div class=\"{}\">{}</div>", self.name, self.name)
        }

        fn to_css(&self) -> String {
            format!(".{} {{ display: block; }}", self.name)
        }
    }

    #[test]
    fn test_brick_house_creation() {
        let house = BrickHouse::new("test-house", 1000);
        assert_eq!(house.name(), "test-house");
        assert_eq!(house.budget().total_ms, 1000);
        assert_eq!(house.brick_count(), 0);
    }

    #[test]
    fn test_brick_house_add_brick() {
        let mut house = BrickHouse::new("test-house", 1000);
        let brick = Arc::new(SimpleBrick { name: "test" });

        house.add_brick(brick, 100).expect("should add brick");
        assert_eq!(house.brick_count(), 1);
        assert_eq!(house.remaining_budget_ms(), 900);
    }

    #[test]
    fn test_brick_house_budget_exceeded() {
        let mut house = BrickHouse::new("test-house", 100);
        let brick1 = Arc::new(SimpleBrick { name: "brick1" });
        let brick2 = Arc::new(SimpleBrick { name: "brick2" });

        house.add_brick(brick1, 60).expect("should add first brick");
        let result = house.add_brick(brick2, 60);

        assert!(result.is_err());
    }

    #[test]
    fn test_brick_house_builder() {
        let brick1 = Arc::new(SimpleBrick { name: "status" });
        let brick2 = Arc::new(SimpleBrick { name: "content" });

        let house = BrickHouseBuilder::new("app")
            .budget_ms(1000)
            .brick(brick1, 100)
            .brick(brick2, 200)
            .build()
            .expect("should build house");

        assert_eq!(house.brick_count(), 2);
        assert_eq!(house.remaining_budget_ms(), 700);
    }

    #[test]
    fn test_brick_house_builder_exceeds_budget() {
        let brick1 = Arc::new(SimpleBrick { name: "big" });
        let brick2 = Arc::new(SimpleBrick { name: "bigger" });

        let result = BrickHouseBuilder::new("app")
            .budget_ms(100)
            .brick(brick1, 60)
            .brick(brick2, 60)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_brick_house_render() {
        let brick = Arc::new(SimpleBrick { name: "test" });
        let mut house = BrickHouse::new("test-house", 1000);
        house.add_brick(brick, 100).expect("should add brick");

        let html = house.render().expect("should render");
        assert!(html.contains("test"));
    }

    #[test]
    fn test_jidoka_alert() {
        let violation = BudgetViolation {
            brick_name: "slow-brick".into(),
            budget: BrickBudget::uniform(100),
            actual: Duration::from_millis(150),
            phase: Some(BrickPhase::Paint),
        };

        let alert = JidokaAlert::from_violation("app", &violation);
        assert_eq!(alert.overage_percent(), 50.0);
    }

    #[test]
    fn test_budget_report_within_budget() {
        let report = BudgetReport {
            house_name: "test".into(),
            total_budget_ms: 1000,
            total_used_ms: 500,
            brick_timings: HashMap::new(),
            violations: vec![],
            timestamp: std::time::SystemTime::now(),
        };
        assert!(report.within_budget());
        assert_eq!(report.utilization(), 50.0);
    }

    #[test]
    fn test_budget_report_over_budget() {
        let violation = BudgetViolation {
            brick_name: "test".into(),
            budget: BrickBudget::uniform(100),
            actual: Duration::from_millis(150),
            phase: None,
        };
        let report = BudgetReport {
            house_name: "test".into(),
            total_budget_ms: 1000,
            total_used_ms: 1500,
            brick_timings: HashMap::new(),
            violations: vec![violation],
            timestamp: std::time::SystemTime::now(),
        };
        assert!(!report.within_budget());
        assert!(!report.violations().is_empty());
    }

    #[test]
    fn test_budget_report_zero_budget() {
        let report = BudgetReport {
            house_name: "test".into(),
            total_budget_ms: 0,
            total_used_ms: 0,
            brick_timings: HashMap::new(),
            violations: vec![],
            timestamp: std::time::SystemTime::now(),
        };
        assert_eq!(report.utilization(), 0.0);
    }

    #[test]
    fn test_brick_timing() {
        let timing = BrickTiming {
            name: "test".into(),
            budget_ms: 100,
            used_ms: 50,
            exceeded: false,
        };
        assert_eq!(timing.name, "test");
        assert!(!timing.exceeded);
    }

    #[test]
    fn test_brick_timing_exceeded() {
        let timing = BrickTiming {
            name: "slow".into(),
            budget_ms: 100,
            used_ms: 150,
            exceeded: true,
        };
        assert!(timing.exceeded);
    }

    #[test]
    fn test_brick_house_verify_all() {
        let brick1 = Arc::new(SimpleBrick { name: "brick1" });
        let brick2 = Arc::new(SimpleBrick { name: "brick2" });
        let mut house = BrickHouse::new("test", 1000);
        house.add_brick(brick1, 100).unwrap();
        house.add_brick(brick2, 100).unwrap();

        let verifications = house.verify_all();
        assert_eq!(verifications.len(), 2);
    }

    #[test]
    fn test_brick_house_can_render() {
        let brick = Arc::new(SimpleBrick { name: "test" });
        let mut house = BrickHouse::new("test", 1000);
        house.add_brick(brick, 100).unwrap();

        assert!(house.can_render());
    }

    #[test]
    fn test_brick_entry_debug() {
        let brick = Arc::new(SimpleBrick { name: "test" });
        let entry = BrickEntry {
            brick,
            allocated_ms: 100,
            last_render_time: Some(Duration::from_millis(50)),
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_brick_house_to_css() {
        let brick1 = Arc::new(SimpleBrick { name: "brick1" });
        let brick2 = Arc::new(SimpleBrick { name: "brick2" });
        let mut house = BrickHouse::new("test", 1000);
        house.add_brick(brick1, 100).unwrap();
        house.add_brick(brick2, 100).unwrap();

        let css = house.to_css();
        assert!(css.contains("brick1"));
        assert!(css.contains("brick2"));
    }

    #[test]
    fn test_brick_house_last_report_none() {
        let house = BrickHouse::new("test", 1000);
        assert!(house.last_report().is_none());
    }

    #[test]
    fn test_brick_house_render_populates_report() {
        let brick = Arc::new(SimpleBrick { name: "test" });
        let mut house = BrickHouse::new("test-house", 1000);
        house.add_brick(brick, 100).unwrap();

        let _ = house.render().unwrap();

        let report = house.last_report();
        assert!(report.is_some());
        let report = report.unwrap();
        assert_eq!(report.house_name, "test-house");
        assert!(report.brick_timings.contains_key("test"));
    }

    #[test]
    fn test_budget_report_violations() {
        let violation = BudgetViolation {
            brick_name: "slow".into(),
            budget: BrickBudget::uniform(100),
            actual: Duration::from_millis(150),
            phase: Some(BrickPhase::Paint),
        };
        let report = BudgetReport {
            house_name: "test".into(),
            total_budget_ms: 1000,
            total_used_ms: 150,
            brick_timings: HashMap::new(),
            violations: vec![violation],
            timestamp: std::time::SystemTime::now(),
        };

        assert!(!report.within_budget());
        assert_eq!(report.violations().len(), 1);
    }

    #[test]
    fn test_brick_house_builder_debug() {
        let builder = BrickHouseBuilder::new("test-app").budget_ms(500);
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("test-app"));
        assert!(debug_str.contains("500"));
    }

    #[test]
    fn test_brick_house_debug() {
        let mut house = BrickHouse::new("test-house", 1000);
        let brick = Arc::new(SimpleBrick { name: "test" });
        house.add_brick(brick, 100).unwrap();

        let debug_str = format!("{:?}", house);
        assert!(debug_str.contains("test-house"));
        assert!(debug_str.contains("1000"));
    }

    #[test]
    fn test_jidoka_alert_zero_budget() {
        let violation = BudgetViolation {
            brick_name: "test".into(),
            budget: BrickBudget::uniform(0),
            actual: Duration::from_millis(10),
            phase: None,
        };

        let alert = JidokaAlert::from_violation("house", &violation);
        assert_eq!(alert.overage_percent(), 0.0);
    }

    #[test]
    fn test_jidoka_alert_fields() {
        let violation = BudgetViolation {
            brick_name: "slow-brick".into(),
            budget: BrickBudget::uniform(100),
            actual: Duration::from_millis(200),
            phase: Some(BrickPhase::Layout),
        };

        let alert = JidokaAlert::from_violation("my-house", &violation);
        assert_eq!(alert.house_name, "my-house");
        assert_eq!(alert.brick_name, "slow-brick");
        assert_eq!(alert.budget_ms, 100);
        assert_eq!(alert.actual_ms, 200);
        assert!(alert.phase.is_some());
        assert!(alert.stack_trace.is_none());
        assert_eq!(alert.overage_percent(), 100.0);
    }

    #[test]
    fn test_brick_entry_debug_no_render_time() {
        let brick = Arc::new(SimpleBrick { name: "test" });
        let entry = BrickEntry {
            brick,
            allocated_ms: 100,
            last_render_time: None,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("None"));
    }

    // Test for a brick that fails verification
    struct FailingBrick {
        name: &'static str,
    }

    impl Brick for FailingBrick {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[BrickAssertion::TextVisible]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![(BrickAssertion::TextVisible, "Text not visible".to_string())],
                verification_time: Duration::from_micros(1),
            }
        }

        fn to_html(&self) -> String {
            format!("<div class=\"{}\">{}</div>", self.name, self.name)
        }

        fn to_css(&self) -> String {
            format!(".{} {{ display: block; }}", self.name)
        }

        fn can_render(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_brick_house_render_failing_brick() {
        let brick = Arc::new(FailingBrick { name: "failing" });
        let mut house = BrickHouse::new("test-house", 1000);
        house.add_brick(brick, 100).unwrap();

        let result = house.render();
        assert!(result.is_err());
    }

    #[test]
    fn test_brick_house_can_render_with_failing_brick() {
        let brick = Arc::new(FailingBrick { name: "failing" });
        let mut house = BrickHouse::new("test-house", 1000);
        house.add_brick(brick, 100).unwrap();

        assert!(!house.can_render());
    }

    #[test]
    fn test_brick_house_multiple_bricks_render() {
        let brick1 = Arc::new(SimpleBrick { name: "header" });
        let brick2 = Arc::new(SimpleBrick { name: "content" });
        let brick3 = Arc::new(SimpleBrick { name: "footer" });

        let mut house = BrickHouse::new("page", 1000);
        house.add_brick(brick1, 100).unwrap();
        house.add_brick(brick2, 200).unwrap();
        house.add_brick(brick3, 100).unwrap();

        let html = house.render().unwrap();
        assert!(html.contains("header"));
        assert!(html.contains("content"));
        assert!(html.contains("footer"));
    }

    #[test]
    fn test_brick_house_builder_with_many_bricks() {
        let brick1 = Arc::new(SimpleBrick { name: "a" });
        let brick2 = Arc::new(SimpleBrick { name: "b" });
        let brick3 = Arc::new(SimpleBrick { name: "c" });

        let house = BrickHouseBuilder::new("multi")
            .budget_ms(1000)
            .brick(brick1, 100)
            .brick(brick2, 200)
            .brick(brick3, 300)
            .build()
            .unwrap();

        assert_eq!(house.brick_count(), 3);
        assert_eq!(house.remaining_budget_ms(), 400);
    }

    #[test]
    fn test_budget_report_utilization_100_percent() {
        let report = BudgetReport {
            house_name: "test".into(),
            total_budget_ms: 100,
            total_used_ms: 100,
            brick_timings: HashMap::new(),
            violations: vec![],
            timestamp: std::time::SystemTime::now(),
        };
        assert_eq!(report.utilization(), 100.0);
    }

    #[test]
    fn test_budget_report_utilization_over_budget() {
        let report = BudgetReport {
            house_name: "test".into(),
            total_budget_ms: 100,
            total_used_ms: 200,
            brick_timings: HashMap::new(),
            violations: vec![],
            timestamp: std::time::SystemTime::now(),
        };
        assert_eq!(report.utilization(), 200.0);
        // Over budget but no violations means it's not within budget
        assert!(!report.within_budget());
    }

    #[test]
    fn test_brick_house_empty_render() {
        let mut house = BrickHouse::new("empty", 1000);
        let html = house.render().unwrap();
        assert!(html.is_empty());
    }

    #[test]
    fn test_brick_house_empty_css() {
        let house = BrickHouse::new("empty", 1000);
        let css = house.to_css();
        assert!(css.is_empty());
    }

    #[test]
    fn test_brick_house_verify_all_empty() {
        let house = BrickHouse::new("empty", 1000);
        let verifications = house.verify_all();
        assert!(verifications.is_empty());
    }
}
