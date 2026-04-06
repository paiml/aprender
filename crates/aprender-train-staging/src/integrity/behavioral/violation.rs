//! Metamorphic violation types for behavioral testing
//!
//! Defines violation types and metamorphic relation categories
//! used in behavioral integrity verification.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A metamorphic violation detected during behavioral testing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetamorphicViolation {
    /// Unique identifier for this violation
    pub id: String,

    /// Type of metamorphic relation violated
    pub relation_type: MetamorphicRelationType,

    /// Description of the violation
    pub description: String,

    /// Input that caused the violation
    pub input_description: String,

    /// Expected behavior
    pub expected: String,

    /// Actual behavior observed
    pub actual: String,

    /// Severity of the violation (0.0 - 1.0, higher = more severe)
    pub severity: f64,

    /// When the violation was detected
    pub detected_at: DateTime<Utc>,
}

impl MetamorphicViolation {
    /// Create a new metamorphic violation
    pub fn new(
        id: impl Into<String>,
        relation_type: MetamorphicRelationType,
        description: impl Into<String>,
        input_description: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
        severity: f64,
    ) -> Self {
        Self {
            id: id.into(),
            relation_type,
            description: description.into(),
            input_description: input_description.into(),
            expected: expected.into(),
            actual: actual.into(),
            severity: severity.clamp(0.0, 1.0),
            detected_at: Utc::now(),
        }
    }

    /// Check if this is a critical violation (severity >= 0.8)
    pub fn is_critical(&self) -> bool {
        self.severity >= 0.8
    }

    /// Check if this is a warning-level violation (severity >= 0.5)
    pub fn is_warning(&self) -> bool {
        self.severity >= 0.5
    }
}

/// Types of metamorphic relations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetamorphicRelationType {
    /// Additive: f(x + c) should relate to f(x) in predictable way
    Additive,
    /// Multiplicative: f(k * x) should relate to f(x)
    Multiplicative,
    /// Permutation: f(permute(x)) should relate to f(x)
    Permutation,
    /// Composition: f(g(x)) should relate to g(f(x)) or similar
    Composition,
    /// Negation: f(-x) should relate to f(x)
    Negation,
    /// Inclusion: f(x ⊂ y) implies relation between f(x) and f(y)
    Inclusion,
    /// Identity: f(x) should equal f(x) across invocations
    Identity,
    /// Custom relation type
    Custom,
}

impl std::fmt::Display for MetamorphicRelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Additive => write!(f, "additive"),
            Self::Multiplicative => write!(f, "multiplicative"),
            Self::Permutation => write!(f, "permutation"),
            Self::Composition => write!(f, "composition"),
            Self::Negation => write!(f, "negation"),
            Self::Inclusion => write!(f, "inclusion"),
            Self::Identity => write!(f, "identity"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metamorphic_violation_new() {
        let violation = MetamorphicViolation::new(
            "v1",
            MetamorphicRelationType::Additive,
            "Test violation",
            "input x=5",
            "f(x+1) = f(x) + 1",
            "f(x+1) = f(x) + 2",
            0.7,
        );

        assert_eq!(violation.id, "v1");
        assert_eq!(violation.relation_type, MetamorphicRelationType::Additive);
        assert_eq!(violation.description, "Test violation");
        assert!((violation.severity - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_metamorphic_violation_severity_clamped() {
        let high = MetamorphicViolation::new(
            "v1",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            1.5, // should be clamped to 1.0
        );
        assert!((high.severity - 1.0).abs() < 1e-6);

        let low = MetamorphicViolation::new(
            "v2",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            -0.5, // should be clamped to 0.0
        );
        assert!((low.severity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_critical() {
        let critical = MetamorphicViolation::new(
            "v1",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.8,
        );
        assert!(critical.is_critical());

        let non_critical = MetamorphicViolation::new(
            "v2",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.79,
        );
        assert!(!non_critical.is_critical());
    }

    #[test]
    fn test_is_warning() {
        let warning = MetamorphicViolation::new(
            "v1",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.5,
        );
        assert!(warning.is_warning());

        let non_warning = MetamorphicViolation::new(
            "v2",
            MetamorphicRelationType::Identity,
            "desc",
            "input",
            "exp",
            "act",
            0.49,
        );
        assert!(!non_warning.is_warning());
    }

    #[test]
    fn test_relation_type_display() {
        assert_eq!(MetamorphicRelationType::Additive.to_string(), "additive");
        assert_eq!(MetamorphicRelationType::Multiplicative.to_string(), "multiplicative");
        assert_eq!(MetamorphicRelationType::Permutation.to_string(), "permutation");
        assert_eq!(MetamorphicRelationType::Composition.to_string(), "composition");
        assert_eq!(MetamorphicRelationType::Negation.to_string(), "negation");
        assert_eq!(MetamorphicRelationType::Inclusion.to_string(), "inclusion");
        assert_eq!(MetamorphicRelationType::Identity.to_string(), "identity");
        assert_eq!(MetamorphicRelationType::Custom.to_string(), "custom");
    }

    #[test]
    fn test_relation_type_eq() {
        assert_eq!(MetamorphicRelationType::Additive, MetamorphicRelationType::Additive);
        assert_ne!(MetamorphicRelationType::Additive, MetamorphicRelationType::Multiplicative);
    }

    #[test]
    fn test_relation_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MetamorphicRelationType::Additive);
        set.insert(MetamorphicRelationType::Additive);
        assert_eq!(set.len(), 1);
        set.insert(MetamorphicRelationType::Identity);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_violation_serde() {
        let violation = MetamorphicViolation::new(
            "v1",
            MetamorphicRelationType::Additive,
            "Test violation",
            "input",
            "expected",
            "actual",
            0.7,
        );

        let json = serde_json::to_string(&violation).expect("JSON serialization should succeed");
        let deserialized: MetamorphicViolation =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(violation.id, deserialized.id);
        assert_eq!(violation.relation_type, deserialized.relation_type);
    }

    #[test]
    fn test_violation_clone() {
        let violation = MetamorphicViolation::new(
            "v1",
            MetamorphicRelationType::Additive,
            "Test",
            "input",
            "expected",
            "actual",
            0.5,
        );
        let cloned = violation.clone();
        assert_eq!(violation, cloned);
    }
}
