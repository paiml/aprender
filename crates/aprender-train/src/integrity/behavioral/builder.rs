//! Builder pattern for BehavioralIntegrity
//!
//! Provides a fluent API for constructing BehavioralIntegrity instances.

use super::metrics::BehavioralIntegrity;
use super::violation::MetamorphicViolation;

/// Builder for BehavioralIntegrity
#[derive(Debug, Clone)]
pub struct BehavioralIntegrityBuilder {
    equivalence_score: f64,
    syscall_match: f64,
    timing_variance: f64,
    semantic_equiv: f64,
    violations: Vec<MetamorphicViolation>,
    test_count: u32,
    model_id: String,
}

impl BehavioralIntegrityBuilder {
    /// Create a new builder
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            equivalence_score: 0.0,
            syscall_match: 0.0,
            timing_variance: 1.0,
            semantic_equiv: 0.0,
            violations: Vec::new(),
            test_count: 0,
            model_id: model_id.into(),
        }
    }

    /// Set equivalence score
    pub fn equivalence_score(mut self, score: f64) -> Self {
        self.equivalence_score = score;
        self
    }

    /// Set syscall match score
    pub fn syscall_match(mut self, score: f64) -> Self {
        self.syscall_match = score;
        self
    }

    /// Set timing variance
    pub fn timing_variance(mut self, variance: f64) -> Self {
        self.timing_variance = variance;
        self
    }

    /// Set semantic equivalence score
    pub fn semantic_equiv(mut self, score: f64) -> Self {
        self.semantic_equiv = score;
        self
    }

    /// Add a violation
    pub fn violation(mut self, violation: MetamorphicViolation) -> Self {
        self.violations.push(violation);
        self
    }

    /// Set test count
    pub fn test_count(mut self, count: u32) -> Self {
        self.test_count = count;
        self
    }

    /// Build the BehavioralIntegrity instance
    pub fn build(self) -> BehavioralIntegrity {
        let mut integrity = BehavioralIntegrity::new(
            self.equivalence_score,
            self.syscall_match,
            self.timing_variance,
            self.semantic_equiv,
            self.model_id,
        );
        integrity.violations = self.violations;
        integrity.test_count = self.test_count;
        integrity
    }
}
