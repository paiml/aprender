//! Popperian Falsification Framework (PIXEL-001 v2.1 Phase 5)
//!
//! Implements the falsifiability gateway and hypothesis testing per Popper's
//! scientific methodology. Claims that cannot be falsified are not science.

use super::terminal::ConfidenceInterval;

/// Popperian Falsifiability Gate (Jidoka - stop-the-line)
/// If hypothesis cannot be falsified, the entire analysis is invalid.
#[derive(Debug, Clone)]
pub struct FalsifiabilityGate {
    /// Minimum threshold for falsifiability score (0-25)
    pub gateway_threshold: f32,
}

impl Default for FalsifiabilityGate {
    fn default() -> Self {
        Self {
            gateway_threshold: 15.0,
        }
    }
}

impl FalsifiabilityGate {
    /// Create a new gate with custom threshold
    #[must_use]
    pub fn new(gateway_threshold: f32) -> Self {
        Self { gateway_threshold }
    }

    /// Evaluate a hypothesis against the falsifiability gateway
    #[must_use]
    pub fn evaluate(&self, hypothesis: &FalsifiableHypothesis) -> GateResult {
        if hypothesis.falsifiability_score < self.gateway_threshold {
            GateResult::Failed {
                score: 0.0,
                reason: "INSUFFICIENT FALSIFIABILITY - NOT EVALUABLE AS SCIENCE".to_string(),
            }
        } else {
            GateResult::Passed {
                score: hypothesis.falsifiability_score,
            }
        }
    }

    /// Evaluate multiple hypotheses, returning overall result
    #[must_use]
    pub fn evaluate_all(&self, hypotheses: &[FalsifiableHypothesis]) -> GateResult {
        for h in hypotheses {
            if let result @ GateResult::Failed { .. } = self.evaluate(h) {
                return result;
            }
        }

        let total_score: f32 = hypotheses.iter().map(|h| h.falsifiability_score).sum();
        let avg_score = if hypotheses.is_empty() {
            0.0
        } else {
            total_score / hypotheses.len() as f32
        };

        GateResult::Passed { score: avg_score }
    }
}

/// Result of falsifiability gate evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum GateResult {
    /// Hypothesis passed the gateway
    Passed {
        /// Falsifiability score achieved
        score: f32,
    },
    /// Hypothesis failed the gateway
    Failed {
        /// Score assigned (usually 0)
        score: f32,
        /// Reason for failure
        reason: String,
    },
}

impl GateResult {
    /// Check if result is passed
    #[must_use]
    pub fn is_passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }

    /// Get the score
    #[must_use]
    pub fn score(&self) -> f32 {
        match self {
            Self::Passed { score } | Self::Failed { score, .. } => *score,
        }
    }
}

/// A condition that would falsify the hypothesis
#[derive(Debug, Clone)]
pub struct FalsificationCondition {
    /// Description of the condition
    pub description: String,
    /// Operator for comparison
    pub operator: ComparisonOperator,
    /// Target value
    pub target: f32,
}

impl FalsificationCondition {
    /// Create a new condition
    #[must_use]
    pub fn new(description: &str, operator: ComparisonOperator, target: f32) -> Self {
        Self {
            description: description.to_string(),
            operator,
            target,
        }
    }

    /// Check if condition is met (hypothesis is falsified)
    #[must_use]
    pub fn is_falsified(&self, actual: f32) -> bool {
        match self.operator {
            ComparisonOperator::LessThan => actual < self.target,
            ComparisonOperator::LessOrEqual => actual <= self.target,
            ComparisonOperator::GreaterThan => actual > self.target,
            ComparisonOperator::GreaterOrEqual => actual >= self.target,
            ComparisonOperator::Equal => (actual - self.target).abs() < f32::EPSILON,
            ComparisonOperator::NotEqual => (actual - self.target).abs() >= f32::EPSILON,
        }
    }
}

/// Comparison operators for falsification conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
    /// <
    LessThan,
    /// <=
    LessOrEqual,
    /// >
    GreaterThan,
    /// >=
    GreaterOrEqual,
    /// ==
    Equal,
    /// !=
    NotEqual,
}

/// A falsifiable hypothesis about pixel coverage (full Popperian specification)
#[derive(Debug, Clone)]
pub struct FalsifiableHypothesis {
    /// Hypothesis ID (e.g., "H0-COV-01")
    pub id: String,
    /// H₀: The null hypothesis to falsify
    pub null_hypothesis: String,
    /// Measurable threshold (falsification criterion)
    pub threshold: f32,
    /// Actual measured value
    pub actual: Option<f32>,
    /// Confidence interval for statistical rigor
    pub confidence_interval: Option<ConfidenceInterval>,
    /// What would falsify this claim
    pub falsification_conditions: Vec<FalsificationCondition>,
    /// Falsifiability score (0-25, gate requires >= 15)
    pub falsifiability_score: f32,
    /// Whether the hypothesis has been falsified
    pub falsified: bool,
}

impl FalsifiableHypothesis {
    /// Create a new hypothesis builder
    #[must_use]
    pub fn builder(id: &str) -> FalsifiableHypothesisBuilder {
        FalsifiableHypothesisBuilder::new(id)
    }

    /// Create a standard coverage threshold hypothesis
    #[must_use]
    pub fn coverage_threshold(id: &str, threshold: f32) -> Self {
        Self::builder(id)
            .null_hypothesis(&format!(
                "Coverage exceeds {:.0}% of screen pixels",
                threshold * 100.0
            ))
            .threshold(threshold)
            .falsification_condition(FalsificationCondition::new(
                &format!("Coverage < {:.0}%", threshold * 100.0),
                ComparisonOperator::LessThan,
                threshold,
            ))
            .falsifiability_score(20.0) // High falsifiability: clear measurement
            .build()
    }

    /// Create a gap size hypothesis
    #[must_use]
    pub fn max_gap_size(id: &str, max_gap_percent: f32) -> Self {
        Self::builder(id)
            .null_hypothesis(&format!(
                "No gap region exceeds {:.0}% of total area",
                max_gap_percent * 100.0
            ))
            .threshold(max_gap_percent)
            .falsification_condition(FalsificationCondition::new(
                &format!("Gap > {:.0}% detected", max_gap_percent * 100.0),
                ComparisonOperator::GreaterThan,
                max_gap_percent,
            ))
            .falsifiability_score(22.0) // Very high falsifiability: specific region
            .build()
    }

    /// Create an SSIM threshold hypothesis
    #[must_use]
    pub fn ssim_threshold(id: &str, min_ssim: f32) -> Self {
        Self::builder(id)
            .null_hypothesis(&format!(
                "Rendered heatmap matches reference within SSIM >= {:.2}",
                min_ssim
            ))
            .threshold(min_ssim)
            .falsification_condition(FalsificationCondition::new(
                &format!("SSIM < {:.2}", min_ssim),
                ComparisonOperator::LessThan,
                min_ssim,
            ))
            .falsifiability_score(25.0) // Maximum falsifiability: pixel-perfect
            .build()
    }

    /// Evaluate the hypothesis with actual measurement
    #[must_use]
    pub fn evaluate(&self, actual: f32) -> FalsifiableHypothesis {
        let mut result = self.clone();
        result.actual = Some(actual);

        // Check all falsification conditions
        result.falsified = self
            .falsification_conditions
            .iter()
            .any(|c| c.is_falsified(actual));

        result
    }
}

/// Builder for FalsifiableHypothesis
#[derive(Debug, Default)]
pub struct FalsifiableHypothesisBuilder {
    id: String,
    null_hypothesis: String,
    threshold: f32,
    confidence_interval: Option<ConfidenceInterval>,
    falsification_conditions: Vec<FalsificationCondition>,
    falsifiability_score: f32,
}

impl FalsifiableHypothesisBuilder {
    /// Create new builder
    #[must_use]
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            falsifiability_score: 15.0, // Default to gateway threshold
            ..Default::default()
        }
    }

    /// Set null hypothesis
    #[must_use]
    pub fn null_hypothesis(mut self, hypothesis: &str) -> Self {
        self.null_hypothesis = hypothesis.to_string();
        self
    }

    /// Set threshold
    #[must_use]
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set confidence interval
    #[must_use]
    pub fn confidence_interval(mut self, ci: ConfidenceInterval) -> Self {
        self.confidence_interval = Some(ci);
        self
    }

    /// Add falsification condition
    #[must_use]
    pub fn falsification_condition(mut self, condition: FalsificationCondition) -> Self {
        self.falsification_conditions.push(condition);
        self
    }

    /// Set falsifiability score
    #[must_use]
    pub fn falsifiability_score(mut self, score: f32) -> Self {
        self.falsifiability_score = score.clamp(0.0, 25.0);
        self
    }

    /// Build the hypothesis
    #[must_use]
    pub fn build(self) -> FalsifiableHypothesis {
        FalsifiableHypothesis {
            id: self.id,
            null_hypothesis: self.null_hypothesis,
            threshold: self.threshold,
            actual: None,
            confidence_interval: self.confidence_interval,
            falsification_conditions: self.falsification_conditions,
            falsifiability_score: self.falsifiability_score,
            falsified: false,
        }
    }
}

/// Layer of falsification testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalsificationLayer {
    /// L1: Unit tests - direct falsification via assertions
    Unit,
    /// L2: Property tests - statistical falsification via proptest
    Property,
    /// L3: Mutation tests - meta-falsification via mutation score
    Mutation,
}

impl FalsificationLayer {
    /// Get the layer number
    #[must_use]
    pub fn number(&self) -> u8 {
        match self {
            Self::Unit => 1,
            Self::Property => 2,
            Self::Mutation => 3,
        }
    }

    /// Get layer description
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Unit => "Direct falsification via assertions",
            Self::Property => "Statistical falsification via proptest",
            Self::Mutation => "Meta-falsification via mutation score",
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // =========================================================================
    // FalsifiabilityGate Tests (H0-GATE-XX)
    // =========================================================================

    #[test]
    fn h0_gate_01_default_threshold() {
        let gate = FalsifiabilityGate::default();
        assert!((gate.gateway_threshold - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_gate_02_custom_threshold() {
        let gate = FalsifiabilityGate::new(20.0);
        assert!((gate.gateway_threshold - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_gate_03_evaluate_pass() {
        let gate = FalsifiabilityGate::default();
        let hypothesis = FalsifiableHypothesis::coverage_threshold("H0-COV-01", 0.85);
        let result = gate.evaluate(&hypothesis);
        assert!(result.is_passed());
        assert!((result.score() - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_gate_04_evaluate_fail() {
        let gate = FalsifiabilityGate::new(23.0);
        let hypothesis = FalsifiableHypothesis::coverage_threshold("H0-COV-01", 0.85);
        let result = gate.evaluate(&hypothesis);
        assert!(!result.is_passed());
        assert!((result.score() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_gate_05_evaluate_all_pass() {
        let gate = FalsifiabilityGate::default();
        let hypotheses = vec![
            FalsifiableHypothesis::coverage_threshold("H0-COV-01", 0.85),
            FalsifiableHypothesis::max_gap_size("H0-COV-02", 0.15),
        ];
        let result = gate.evaluate_all(&hypotheses);
        assert!(result.is_passed());
    }

    #[test]
    fn h0_gate_06_evaluate_all_fail() {
        let gate = FalsifiabilityGate::new(24.0);
        let hypotheses = vec![
            FalsifiableHypothesis::coverage_threshold("H0-COV-01", 0.85), // score: 20
            FalsifiableHypothesis::max_gap_size("H0-COV-02", 0.15),       // score: 22
        ];
        let result = gate.evaluate_all(&hypotheses);
        assert!(!result.is_passed()); // First fails immediately
    }

    // =========================================================================
    // FalsificationCondition Tests (H0-COND-XX)
    // =========================================================================

    #[test]
    fn h0_cond_01_less_than() {
        let cond =
            FalsificationCondition::new("Coverage < 85%", ComparisonOperator::LessThan, 0.85);
        assert!(cond.is_falsified(0.80)); // 80% < 85%, falsified
        assert!(!cond.is_falsified(0.90)); // 90% >= 85%, not falsified
    }

    #[test]
    fn h0_cond_02_greater_than() {
        let cond = FalsificationCondition::new("Gap > 15%", ComparisonOperator::GreaterThan, 0.15);
        assert!(cond.is_falsified(0.20)); // 20% > 15%, falsified
        assert!(!cond.is_falsified(0.10)); // 10% <= 15%, not falsified
    }

    #[test]
    fn h0_cond_03_equal() {
        let cond = FalsificationCondition::new("Score == 1.0", ComparisonOperator::Equal, 1.0);
        assert!(cond.is_falsified(1.0));
        assert!(!cond.is_falsified(0.99));
    }

    // =========================================================================
    // FalsifiableHypothesis Tests (H0-HYP-XX)
    // =========================================================================

    #[test]
    fn h0_hyp_01_coverage_threshold_pass() {
        let hypothesis = FalsifiableHypothesis::coverage_threshold("H0-COV-01", 0.85);
        let result = hypothesis.evaluate(0.90);
        assert!(!result.falsified);
        assert!((result.actual.unwrap() - 0.90).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_hyp_02_coverage_threshold_fail() {
        let hypothesis = FalsifiableHypothesis::coverage_threshold("H0-COV-01", 0.85);
        let result = hypothesis.evaluate(0.80);
        assert!(result.falsified);
    }

    #[test]
    fn h0_hyp_03_gap_size_pass() {
        let hypothesis = FalsifiableHypothesis::max_gap_size("H0-COV-02", 0.15);
        let result = hypothesis.evaluate(0.10);
        assert!(!result.falsified);
    }

    #[test]
    fn h0_hyp_04_gap_size_fail() {
        let hypothesis = FalsifiableHypothesis::max_gap_size("H0-COV-02", 0.15);
        let result = hypothesis.evaluate(0.20);
        assert!(result.falsified);
    }

    #[test]
    fn h0_hyp_05_ssim_threshold() {
        let hypothesis = FalsifiableHypothesis::ssim_threshold("H0-VIS-01", 0.99);
        assert!((hypothesis.falsifiability_score - 25.0).abs() < f32::EPSILON);
        let result = hypothesis.evaluate(0.985);
        assert!(result.falsified);
    }

    #[test]
    fn h0_hyp_06_builder() {
        let hypothesis = FalsifiableHypothesis::builder("H0-CUSTOM")
            .null_hypothesis("Custom hypothesis")
            .threshold(0.75)
            .falsifiability_score(18.0)
            .falsification_condition(FalsificationCondition::new(
                "Value < 75%",
                ComparisonOperator::LessThan,
                0.75,
            ))
            .build();

        assert_eq!(hypothesis.id, "H0-CUSTOM");
        assert!((hypothesis.falsifiability_score - 18.0).abs() < f32::EPSILON);
        assert_eq!(hypothesis.falsification_conditions.len(), 1);
    }

    // =========================================================================
    // FalsificationLayer Tests (H0-LAYER-XX)
    // =========================================================================

    #[test]
    fn h0_layer_01_numbers() {
        assert_eq!(FalsificationLayer::Unit.number(), 1);
        assert_eq!(FalsificationLayer::Property.number(), 2);
        assert_eq!(FalsificationLayer::Mutation.number(), 3);
    }

    #[test]
    fn h0_layer_02_descriptions() {
        assert!(FalsificationLayer::Unit.description().contains("assertion"));
        assert!(FalsificationLayer::Property
            .description()
            .contains("proptest"));
        assert!(FalsificationLayer::Mutation
            .description()
            .contains("mutation"));
    }

    // =========================================================================
    // GateResult Tests (H0-RESULT-XX)
    // =========================================================================

    #[test]
    fn h0_result_01_passed() {
        let result = GateResult::Passed { score: 20.0 };
        assert!(result.is_passed());
        assert!((result.score() - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_result_02_failed() {
        let result = GateResult::Failed {
            score: 0.0,
            reason: "Test failure".to_string(),
        };
        assert!(!result.is_passed());
        assert!((result.score() - 0.0).abs() < f32::EPSILON);
    }
}

// =============================================================================
// Property-Based Tests (Extreme TDD - L2 Falsification Layer)
// =============================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    // =========================================================================
    // FalsifiabilityGate Property Tests (PROP-GATE-XX)
    // =========================================================================

    proptest! {
        /// PROP-GATE-01: Gate with 0 threshold always passes
        #[test]
        fn prop_gate_01_zero_threshold_passes(score in 0.0f32..=25.0) {
            let gate = FalsifiabilityGate::new(0.0);
            let hypothesis = FalsifiableHypothesis::builder("H0-TEST")
                .null_hypothesis("Test")
                .falsifiability_score(score)
                .build();
            let result = gate.evaluate(&hypothesis);
            prop_assert!(result.is_passed());
        }

        /// PROP-GATE-02: Gate with 25 threshold only passes scores >= 25
        #[test]
        fn prop_gate_02_max_threshold(score in 0.0f32..25.0) {
            let gate = FalsifiabilityGate::new(25.0);
            let hypothesis = FalsifiableHypothesis::builder("H0-TEST")
                .null_hypothesis("Test")
                .falsifiability_score(score)
                .build();
            let result = gate.evaluate(&hypothesis);
            prop_assert!(!result.is_passed(), "Score {} should fail threshold 25", score);
        }

        /// PROP-GATE-03: Score exactly at threshold passes
        #[test]
        fn prop_gate_03_exact_threshold(threshold in 0.0f32..=25.0) {
            let gate = FalsifiabilityGate::new(threshold);
            let hypothesis = FalsifiableHypothesis::builder("H0-TEST")
                .null_hypothesis("Test")
                .falsifiability_score(threshold)
                .build();
            let result = gate.evaluate(&hypothesis);
            prop_assert!(result.is_passed());
        }
    }

    // =========================================================================
    // Hypothesis Property Tests (PROP-HYP-XX)
    // =========================================================================

    proptest! {
        /// PROP-HYP-01: Coverage hypothesis falsified when actual < threshold
        #[test]
        fn prop_hyp_01_coverage_falsified(
            threshold in 0.01f32..=1.0,
            delta in 0.01f32..=0.5
        ) {
            let actual = (threshold - delta).max(0.0);
            let hypothesis = FalsifiableHypothesis::coverage_threshold("H0-COV", threshold);
            let result = hypothesis.evaluate(actual);
            prop_assert!(result.falsified, "Should be falsified: {} < {}", actual, threshold);
        }

        /// PROP-HYP-02: Coverage hypothesis not falsified when actual >= threshold
        #[test]
        fn prop_hyp_02_coverage_not_falsified(
            threshold in 0.0f32..=0.99,
            delta in 0.0f32..=0.5
        ) {
            let actual = (threshold + delta).min(1.0);
            let hypothesis = FalsifiableHypothesis::coverage_threshold("H0-COV", threshold);
            let result = hypothesis.evaluate(actual);
            prop_assert!(!result.falsified, "Should not be falsified: {} >= {}", actual, threshold);
        }

        /// PROP-HYP-03: Gap hypothesis falsified when actual > max
        #[test]
        fn prop_hyp_03_gap_falsified(
            max_gap in 0.0f32..=0.99,
            delta in 0.01f32..=0.5
        ) {
            let actual = (max_gap + delta).min(1.0);
            let hypothesis = FalsifiableHypothesis::max_gap_size("H0-GAP", max_gap);
            let result = hypothesis.evaluate(actual);
            prop_assert!(result.falsified, "Should be falsified: {} > {}", actual, max_gap);
        }

        /// PROP-HYP-04: SSIM hypothesis score is maximum (25)
        #[test]
        fn prop_hyp_04_ssim_max_score(threshold in 0.0f32..=1.0) {
            let hypothesis = FalsifiableHypothesis::ssim_threshold("H0-SSIM", threshold);
            prop_assert!((hypothesis.falsifiability_score - 25.0).abs() < f32::EPSILON);
        }
    }

    // =========================================================================
    // Condition Property Tests (PROP-COND-XX)
    // =========================================================================

    proptest! {
        /// PROP-COND-01: LessThan falsifies when actual < target
        #[test]
        fn prop_cond_01_less_than(target in -100.0f32..=100.0, delta in 0.01f32..=50.0) {
            let cond = FalsificationCondition::new("Test", ComparisonOperator::LessThan, target);
            let actual = target - delta;
            prop_assert!(cond.is_falsified(actual));
        }

        /// PROP-COND-02: GreaterThan falsifies when actual > target
        #[test]
        fn prop_cond_02_greater_than(target in -100.0f32..=100.0, delta in 0.01f32..=50.0) {
            let cond = FalsificationCondition::new("Test", ComparisonOperator::GreaterThan, target);
            let actual = target + delta;
            prop_assert!(cond.is_falsified(actual));
        }

        /// PROP-COND-03: Equal falsifies when actual == target
        #[test]
        fn prop_cond_03_equal(target in -100.0f32..=100.0) {
            let cond = FalsificationCondition::new("Test", ComparisonOperator::Equal, target);
            prop_assert!(cond.is_falsified(target));
        }

        /// PROP-COND-04: NotEqual falsifies when actual != target
        #[test]
        fn prop_cond_04_not_equal(target in -100.0f32..=100.0, delta in 0.01f32..=50.0) {
            let cond = FalsificationCondition::new("Test", ComparisonOperator::NotEqual, target);
            let actual = target + delta;
            prop_assert!(cond.is_falsified(actual));
        }
    }

    // =========================================================================
    // Builder Property Tests (PROP-BUILD-XX)
    // =========================================================================

    proptest! {
        /// PROP-BUILD-01: Builder preserves ID
        #[test]
        fn prop_build_01_preserves_id(id in "[A-Z0-9-]{1,20}") {
            let hypothesis = FalsifiableHypothesis::builder(&id)
                .null_hypothesis("Test")
                .build();
            prop_assert_eq!(hypothesis.id, id);
        }

        /// PROP-BUILD-02: Builder clamps score to [0, 25]
        #[test]
        fn prop_build_02_clamps_score(score in -100.0f32..=100.0) {
            let hypothesis = FalsifiableHypothesis::builder("H0-TEST")
                .null_hypothesis("Test")
                .falsifiability_score(score)
                .build();
            prop_assert!(hypothesis.falsifiability_score >= 0.0);
            prop_assert!(hypothesis.falsifiability_score <= 25.0);
        }
    }
}
