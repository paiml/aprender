//! Behavioral integrity metrics
//!
//! Core metrics tracking for model promotion gates including
//! equivalence scoring, syscall matching, timing variance, and semantic equivalence.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::assessment::IntegrityAssessment;
use super::counts::ViolationCounts;
use super::violation::{MetamorphicRelationType, MetamorphicViolation};

/// Behavioral integrity metrics for model promotion gates
///
/// Tracks multiple dimensions of behavioral consistency to determine
/// if a model is ready for promotion to production.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BehavioralIntegrity {
    /// Overall equivalence score (0.0 - 1.0)
    /// Measures how well the model's behavior matches expected patterns
    pub equivalence_score: f64,

    /// Syscall pattern match score (0.0 - 1.0)
    /// Measures consistency of system call patterns during inference
    pub syscall_match: f64,

    /// Timing variance score (0.0 - 1.0, lower = more consistent)
    /// Measures consistency of inference timing
    pub timing_variance: f64,

    /// Semantic equivalence score (0.0 - 1.0)
    /// Measures semantic consistency of model outputs
    pub semantic_equiv: f64,

    /// List of metamorphic violations detected
    pub violations: Vec<MetamorphicViolation>,

    /// Timestamp when metrics were collected
    pub timestamp: DateTime<Utc>,

    /// Number of test cases evaluated
    pub test_count: u32,

    /// Model version or identifier being evaluated
    pub model_id: String,
}

impl BehavioralIntegrity {
    /// Create new behavioral integrity metrics
    pub fn new(
        equivalence_score: f64,
        syscall_match: f64,
        timing_variance: f64,
        semantic_equiv: f64,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            equivalence_score: equivalence_score.clamp(0.0, 1.0),
            syscall_match: syscall_match.clamp(0.0, 1.0),
            timing_variance: timing_variance.clamp(0.0, 1.0),
            semantic_equiv: semantic_equiv.clamp(0.0, 1.0),
            violations: Vec::new(),
            timestamp: Utc::now(),
            test_count: 0,
            model_id: model_id.into(),
        }
    }

    /// Create perfect behavioral integrity (all scores = 1.0, variance = 0.0)
    pub fn perfect(model_id: impl Into<String>) -> Self {
        Self::new(1.0, 1.0, 0.0, 1.0, model_id)
    }

    /// Add a metamorphic violation
    pub fn add_violation(&mut self, violation: MetamorphicViolation) {
        self.violations.push(violation);
    }

    /// Set the test count
    pub fn with_test_count(mut self, count: u32) -> Self {
        self.test_count = count;
        self
    }

    /// Calculate the composite integrity score
    ///
    /// Weighted average of all metrics (timing variance is inverted)
    pub fn composite_score(&self) -> f64 {
        // Weights for each metric
        const W_EQUIV: f64 = 0.3;
        const W_SYSCALL: f64 = 0.2;
        const W_TIMING: f64 = 0.2;
        const W_SEMANTIC: f64 = 0.3;

        let timing_score = 1.0 - self.timing_variance; // Invert: lower variance = better

        W_EQUIV * self.equivalence_score
            + W_SYSCALL * self.syscall_match
            + W_TIMING * timing_score
            + W_SEMANTIC * self.semantic_equiv
    }

    /// Check if the model passes promotion gate
    ///
    /// Requires:
    /// - Composite score >= threshold (default 0.9)
    /// - No critical violations
    /// - Timing variance < 0.2
    pub fn passes_gate(&self, threshold: f64) -> bool {
        self.composite_score() >= threshold
            && !self.has_critical_violations()
            && self.timing_variance < 0.2
    }

    /// Check if there are any critical violations
    pub fn has_critical_violations(&self) -> bool {
        self.violations.iter().any(MetamorphicViolation::is_critical)
    }

    /// Get count of violations by severity level
    pub fn violation_counts(&self) -> ViolationCounts {
        let critical = self.violations.iter().filter(|v| v.is_critical()).count() as u32;
        let warnings =
            self.violations.iter().filter(|v| v.is_warning() && !v.is_critical()).count() as u32;
        let minor = self.violations.iter().filter(|v| !v.is_warning()).count() as u32;

        ViolationCounts { critical, warnings, minor, total: self.violations.len() as u32 }
    }

    /// Get violations grouped by relation type
    pub fn violations_by_type(
        &self,
    ) -> std::collections::HashMap<MetamorphicRelationType, Vec<&MetamorphicViolation>> {
        let mut map = std::collections::HashMap::new();
        for v in &self.violations {
            map.entry(v.relation_type).or_insert_with(Vec::new).push(v);
        }
        map
    }

    /// Get the most severe violation, if any
    pub fn most_severe_violation(&self) -> Option<&MetamorphicViolation> {
        self.violations
            .iter()
            .max_by(|a, b| a.severity.partial_cmp(&b.severity).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get a human-readable assessment
    pub fn assessment(&self) -> IntegrityAssessment {
        let score = self.composite_score();
        let counts = self.violation_counts();

        if counts.critical > 0 {
            IntegrityAssessment::Critical
        } else if score < 0.5 {
            IntegrityAssessment::Poor
        } else if score < 0.7 {
            IntegrityAssessment::Fair
        } else if score < 0.9 {
            IntegrityAssessment::Good
        } else {
            IntegrityAssessment::Excellent
        }
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let counts = self.violation_counts();
        format!(
            "Model: {}\n\
             Composite Score: {:.1}%\n\
             Assessment: {}\n\
             Violations: {} critical, {} warnings, {} minor\n\
             Tests Run: {}\n\
             Gate Status: {}",
            self.model_id,
            self.composite_score() * 100.0,
            self.assessment(),
            counts.critical,
            counts.warnings,
            counts.minor,
            self.test_count,
            if self.passes_gate(0.9) { "PASS" } else { "FAIL" }
        )
    }
}
