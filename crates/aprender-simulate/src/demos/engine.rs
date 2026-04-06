//! Unified Demo Engine Trait
//!
//! Per specification SIMULAR-DEMO-001: All demos MUST implement `DemoEngine`.
//! This trait ensures:
//! - YAML-first configuration (single source of truth)
//! - Deterministic replay (same seed → same output)
//! - Renderer independence (TUI/WASM produce identical state sequences)
//! - Metamorphic testing support
//! - Falsification criteria evaluation
//!
//! # Peer-Reviewed Foundation
//!
//! - Chen et al. (2018): Metamorphic testing
//! - Lavoie & Hendren (2015): Deterministic replay
//! - Faulk et al. (2020): Property-based testing for scientific code

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

/// Errors that can occur in demo operations.
#[derive(Debug, Error)]
pub enum DemoError {
    /// YAML parsing failed.
    #[error("YAML parse error: {0}")]
    YamlParse(#[from] serde_yaml::Error),

    /// Configuration validation failed.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Schema validation failed.
    #[error("Schema validation error: {0}")]
    Schema(String),

    /// Metamorphic relation verification failed.
    #[error("Metamorphic relation {id} failed: {message}")]
    MetamorphicFailure { id: String, message: String },

    /// Invariant violation detected.
    #[error("Invariant violation: {0}")]
    InvariantViolation(String),

    /// State serialization failed.
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Severity levels for falsification criteria.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Must pass - simulation is invalid if this fails.
    Critical,
    /// Should pass - indicates a problem but simulation continues.
    #[default]
    Major,
    /// Nice to have - informational only.
    Minor,
}

/// A single falsification criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationCriterion {
    /// Unique criterion ID (e.g., "TSP-GAP-001").
    pub id: String,

    /// Human-readable name.
    pub name: String,

    /// Metric being evaluated.
    pub metric: String,

    /// Threshold value.
    pub threshold: f64,

    /// Condition expression (e.g., "gap <= threshold").
    pub condition: String,

    /// Tolerance for floating-point comparisons.
    #[serde(default)]
    pub tolerance: f64,

    /// Severity level.
    #[serde(default)]
    pub severity: Severity,
}

/// Result of evaluating a falsification criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionResult {
    /// Criterion ID.
    pub id: String,

    /// Whether the criterion passed.
    pub passed: bool,

    /// Actual value observed.
    pub actual: f64,

    /// Expected threshold.
    pub expected: f64,

    /// Human-readable message.
    pub message: String,

    /// Severity of this criterion.
    pub severity: Severity,
}

/// A metamorphic relation for testing without oracles.
///
/// Per Chen et al. (2018): Metamorphic relations verify invariants
/// without requiring knowledge of the correct output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetamorphicRelation {
    /// Unique relation ID (e.g., "MR-PermutationInvariance").
    pub id: String,

    /// Human-readable description.
    pub description: String,

    /// Transform to apply to source input.
    pub source_transform: String,

    /// Expected relation between source and follow-up outputs.
    pub expected_relation: String,

    /// Tolerance for numerical comparisons.
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
}

fn default_tolerance() -> f64 {
    1e-10
}

/// Result of verifying a metamorphic relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MrResult {
    /// Relation ID.
    pub id: String,

    /// Whether the relation held.
    pub passed: bool,

    /// Detailed message.
    pub message: String,

    /// Source output value (if applicable).
    pub source_value: Option<f64>,

    /// Follow-up output value (if applicable).
    pub followup_value: Option<f64>,
}

/// Demo metadata from YAML configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoMeta {
    /// Unique identifier (e.g., "TSP-BAY-020").
    pub id: String,

    /// Semantic version.
    pub version: String,

    /// Demo type (e.g., "tsp", "orbit", "monte\_carlo").
    pub demo_type: String,

    /// Human-readable description.
    #[serde(default)]
    pub description: String,

    /// Author.
    #[serde(default)]
    pub author: String,

    /// Creation date.
    #[serde(default)]
    pub created: String,
}

/// MANDATORY trait for ALL demos (EDD-compliant).
///
/// Per specification SIMULAR-DEMO-001, all demos must implement this trait
/// to ensure:
/// - YAML-first configuration
/// - Deterministic replay
/// - Renderer independence
/// - Falsification support
/// - Metamorphic testing
///
/// # Example
///
/// ```ignore
/// impl DemoEngine for TspEngine {
///     type Config = TspConfig;
///     type State = TspState;
///     type StepResult = TspStepResult;
///
///     fn from_yaml(yaml: &str) -> Result<Self, DemoError> {
///         let config: TspConfig = serde_yaml::from_str(yaml)?;
///         Ok(Self::from_config(config))
///     }
///     // ... other methods
/// }
/// ```
pub trait DemoEngine: Sized + Clone {
    /// Configuration type loaded from YAML.
    type Config: DeserializeOwned + Debug;

    /// State snapshot for replay/audit.
    type State: Clone + Serialize + DeserializeOwned + PartialEq + Debug;

    /// Result of a single step.
    type StepResult: Debug;

    // === Lifecycle ===

    /// Create engine from YAML configuration string.
    ///
    /// # Errors
    ///
    /// Returns `DemoError::YamlParse` if YAML is invalid.
    /// Returns `DemoError::Validation` if config fails validation.
    fn from_yaml(yaml: &str) -> Result<Self, DemoError>;

    /// Create engine from config struct.
    fn from_config(config: Self::Config) -> Self;

    /// Get the current configuration.
    fn config(&self) -> &Self::Config;

    /// Reset to initial state (same seed = same result).
    fn reset(&mut self);

    /// Reset with a new seed.
    fn reset_with_seed(&mut self, seed: u64);

    // === Execution ===

    /// Execute one step (deterministic given state + seed).
    fn step(&mut self) -> Self::StepResult;

    /// Execute N steps.
    fn run(&mut self, n: usize) -> Vec<Self::StepResult> {
        (0..n).map(|_| self.step()).collect()
    }

    /// Check if simulation is complete/converged.
    fn is_complete(&self) -> bool;

    // === State Access ===

    /// Get current state snapshot (for replay verification).
    fn state(&self) -> Self::State;

    /// Restore from a state snapshot.
    fn restore(&mut self, state: &Self::State);

    /// Get current step number.
    fn step_count(&self) -> u64;

    /// Get seed for reproducibility.
    fn seed(&self) -> u64;

    /// Get demo metadata.
    fn meta(&self) -> &DemoMeta;

    // === EDD Compliance ===

    /// Get falsification criteria from config.
    fn falsification_criteria(&self) -> Vec<FalsificationCriterion>;

    /// Evaluate all criteria against current state.
    fn evaluate_criteria(&self) -> Vec<CriterionResult>;

    /// Check if all criteria pass.
    fn is_verified(&self) -> bool {
        self.evaluate_criteria()
            .iter()
            .filter(|r| r.severity == Severity::Critical)
            .all(|r| r.passed)
    }

    // === Metamorphic Relations ===

    /// Get metamorphic relations for this demo.
    fn metamorphic_relations(&self) -> Vec<MetamorphicRelation>;

    /// Verify a specific metamorphic relation.
    fn verify_mr(&self, mr: &MetamorphicRelation) -> MrResult;

    /// Verify all metamorphic relations.
    fn verify_all_mrs(&self) -> Vec<MrResult> {
        self.metamorphic_relations()
            .iter()
            .map(|mr| self.verify_mr(mr))
            .collect()
    }
}

/// Helper trait for demos that support deterministic replay.
///
/// Per Lavoie & Hendren (2015): Given identical configuration and seed,
/// two independent runs must produce bit-identical state sequences.
pub trait DeterministicReplay: DemoEngine {
    /// Verify that two runs with same config produce identical results.
    fn verify_determinism(&self, other: &Self) -> bool {
        self.state() == other.state()
    }

    /// Get a checksum of the current state for quick comparison.
    fn state_checksum(&self) -> u64;
}

/// Helper trait for demos with renderer-independent core logic.
///
/// The core engine MUST be renderer-agnostic. Both TUI and WASM:
/// - Load from the SAME YAML
/// - Use the SAME engine
/// - Produce the SAME state sequence
pub trait RendererIndependent: DemoEngine {
    /// Render-independent data for current state.
    type RenderData: Clone + Serialize;

    /// Get data needed for rendering (without actually rendering).
    fn render_data(&self) -> Self::RenderData;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_default() {
        assert_eq!(Severity::default(), Severity::Major);
    }

    #[test]
    fn test_severity_serialization() {
        let critical = Severity::Critical;
        let json = serde_json::to_string(&critical).expect("serialize");
        assert_eq!(json, "\"critical\"");

        let deserialized: Severity = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, Severity::Critical);
    }

    #[test]
    fn test_demo_meta_serialization() {
        let meta = DemoMeta {
            id: "TEST-001".to_string(),
            version: "1.0.0".to_string(),
            demo_type: "test".to_string(),
            description: "Test demo".to_string(),
            author: "PAIML".to_string(),
            created: "2025-12-12".to_string(),
        };

        let json = serde_json::to_string(&meta).expect("serialize");
        assert!(json.contains("TEST-001"));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn test_demo_meta_deserialization() {
        let yaml = r#"
id: "DEMO-001"
version: "2.0.0"
demo_type: "orbit"
description: "Orbit demo"
author: "Test"
created: "2025-01-01"
"#;
        let meta: DemoMeta = serde_yaml::from_str(yaml).expect("deserialize");
        assert_eq!(meta.id, "DEMO-001");
        assert_eq!(meta.demo_type, "orbit");
    }

    #[test]
    fn test_falsification_criterion_serialization() {
        let criterion = FalsificationCriterion {
            id: "GAP-001".to_string(),
            name: "Optimality gap".to_string(),
            metric: "gap".to_string(),
            threshold: 0.20,
            condition: "gap <= threshold".to_string(),
            tolerance: 1e-6,
            severity: Severity::Major,
        };

        let json = serde_json::to_string(&criterion).expect("serialize");
        assert!(json.contains("GAP-001"));
        assert!(json.contains("0.2"));
    }

    #[test]
    fn test_criterion_result() {
        let result = CriterionResult {
            id: "TEST".to_string(),
            passed: true,
            actual: 0.15,
            expected: 0.20,
            message: "Passed".to_string(),
            severity: Severity::Critical,
        };

        assert!(result.passed);
        assert!(result.actual < result.expected);
    }

    #[test]
    fn test_metamorphic_relation_default_tolerance() {
        let yaml = r#"
id: "MR-Test"
description: "Test relation"
source_transform: "identity"
expected_relation: "unchanged"
"#;
        let mr: MetamorphicRelation = serde_yaml::from_str(yaml).expect("deserialize");
        assert!((mr.tolerance - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_mr_result() {
        let result = MrResult {
            id: "MR-Energy".to_string(),
            passed: true,
            message: "Energy conserved".to_string(),
            source_value: Some(1000.0),
            followup_value: Some(1000.0),
        };

        assert!(result.passed);
        assert_eq!(result.source_value, result.followup_value);
    }

    #[test]
    fn test_demo_error_display() {
        let err = DemoError::Validation("invalid config".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Validation error"));
        assert!(msg.contains("invalid config"));
    }

    #[test]
    fn test_demo_error_metamorphic() {
        let err = DemoError::MetamorphicFailure {
            id: "MR-001".to_string(),
            message: "Invariant broken".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("MR-001"));
        assert!(msg.contains("Invariant broken"));
    }

    #[test]
    fn test_demo_error_from_yaml() {
        let bad_yaml = "{{{{not valid yaml";
        let result: Result<DemoMeta, _> = serde_yaml::from_str(bad_yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_demo_error_validation() {
        let err = DemoError::Validation("config invalid".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Validation error"));
        assert!(msg.contains("config invalid"));
    }

    #[test]
    fn test_demo_error_schema() {
        let err = DemoError::Schema("schema mismatch".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Schema validation error"));
    }

    #[test]
    fn test_demo_error_invariant_violation() {
        let err = DemoError::InvariantViolation("state corrupted".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Invariant violation"));
    }

    #[test]
    fn test_demo_error_serialization() {
        let err = DemoError::Serialization("failed to serialize".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Serialization error"));
    }

    #[test]
    fn test_severity_minor_serialization() {
        let minor = Severity::Minor;
        let json = serde_json::to_string(&minor).expect("serialize");
        assert_eq!(json, "\"minor\"");

        let deserialized: Severity = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, Severity::Minor);
    }

    #[test]
    fn test_severity_all_variants() {
        let severities = [Severity::Critical, Severity::Major, Severity::Minor];
        for sev in severities {
            let json = serde_json::to_string(&sev).expect("serialize");
            let deserialized: Severity = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(sev, deserialized);
        }
    }

    #[test]
    fn test_falsification_criterion_deserialization() {
        let yaml = r#"
id: "CRIT-001"
name: "Test Criterion"
metric: "accuracy"
threshold: 0.95
condition: "accuracy >= threshold"
tolerance: 0.001
severity: "critical"
"#;
        let criterion: FalsificationCriterion = serde_yaml::from_str(yaml).expect("deserialize");
        assert_eq!(criterion.id, "CRIT-001");
        assert_eq!(criterion.severity, Severity::Critical);
        assert!((criterion.tolerance - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_falsification_criterion_defaults() {
        let yaml = r#"
id: "CRIT-002"
name: "Minimal"
metric: "val"
threshold: 1.0
condition: "val > 0"
"#;
        let criterion: FalsificationCriterion = serde_yaml::from_str(yaml).expect("deserialize");
        assert_eq!(criterion.severity, Severity::Major); // default
        assert!((criterion.tolerance - 0.0).abs() < 1e-15); // default
    }

    #[test]
    fn test_criterion_result_serialization() {
        let result = CriterionResult {
            id: "RES-001".to_string(),
            passed: false,
            actual: 0.85,
            expected: 0.95,
            message: "Below threshold".to_string(),
            severity: Severity::Critical,
        };

        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("RES-001"));
        assert!(json.contains("false"));
        assert!(json.contains("0.85"));
    }

    #[test]
    fn test_metamorphic_relation_with_tolerance() {
        let yaml = r#"
id: "MR-Energy"
description: "Energy conservation"
source_transform: "time_reverse"
expected_relation: "energy_unchanged"
tolerance: 1e-6
"#;
        let mr: MetamorphicRelation = serde_yaml::from_str(yaml).expect("deserialize");
        assert_eq!(mr.id, "MR-Energy");
        assert!((mr.tolerance - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn test_mr_result_with_none_values() {
        let result = MrResult {
            id: "MR-None".to_string(),
            passed: true,
            message: "No comparison needed".to_string(),
            source_value: None,
            followup_value: None,
        };

        assert!(result.passed);
        assert!(result.source_value.is_none());
        assert!(result.followup_value.is_none());
    }

    #[test]
    fn test_mr_result_serialization() {
        let result = MrResult {
            id: "MR-Serialize".to_string(),
            passed: false,
            message: "Values diverged".to_string(),
            source_value: Some(100.0),
            followup_value: Some(101.0),
        };

        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("MR-Serialize"));
        assert!(json.contains("100"));
        assert!(json.contains("101"));
    }

    #[test]
    fn test_demo_meta_with_defaults() {
        let yaml = r#"
id: "MIN-001"
version: "0.1.0"
demo_type: "test"
"#;
        let meta: DemoMeta = serde_yaml::from_str(yaml).expect("deserialize");
        assert_eq!(meta.id, "MIN-001");
        assert!(meta.description.is_empty());
        assert!(meta.author.is_empty());
        assert!(meta.created.is_empty());
    }

    #[test]
    fn test_demo_meta_roundtrip() {
        let meta = DemoMeta {
            id: "ROUND-001".to_string(),
            version: "1.2.3".to_string(),
            demo_type: "orbit".to_string(),
            description: "Round trip test".to_string(),
            author: "Test Author".to_string(),
            created: "2025-12-12".to_string(),
        };

        let yaml = serde_yaml::to_string(&meta).expect("serialize");
        let restored: DemoMeta = serde_yaml::from_str(&yaml).expect("deserialize");

        assert_eq!(meta.id, restored.id);
        assert_eq!(meta.version, restored.version);
        assert_eq!(meta.demo_type, restored.demo_type);
        assert_eq!(meta.description, restored.description);
    }

    #[test]
    fn test_criterion_result_deserialization() {
        let json = r#"{
            "id": "DES-001",
            "passed": true,
            "actual": 0.99,
            "expected": 0.95,
            "message": "Exceeded threshold",
            "severity": "minor"
        }"#;

        let result: CriterionResult = serde_json::from_str(json).expect("deserialize");
        assert_eq!(result.id, "DES-001");
        assert!(result.passed);
        assert_eq!(result.severity, Severity::Minor);
    }

    #[test]
    fn test_metamorphic_relation_serialization() {
        let mr = MetamorphicRelation {
            id: "MR-Serial".to_string(),
            description: "Test MR".to_string(),
            source_transform: "identity".to_string(),
            expected_relation: "equal".to_string(),
            tolerance: 1e-8,
        };

        let yaml = serde_yaml::to_string(&mr).expect("serialize");
        assert!(yaml.contains("MR-Serial"));
        assert!(yaml.contains("identity"));
    }
}
