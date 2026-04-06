//! Experiment specification for EDD - YAML-driven declarative experiments.
//!
//! Every experiment in the EDD framework must be declaratively specified with:
//! - Explicit seed for reproducibility
//! - Falsification criteria for scientific validity
//! - Governing equation reference
//! - Hypotheses to test
//!
//! # Example YAML Specification
//!
//! ```yaml
//! experiment:
//!   name: "Little's Law Validation"
//!   seed: 42
//!   emc: "littles_law_v1.0"
//!   hypothesis:
//!     null: "L ≠ λW"
//!     alternative: "L = λW holds under stochastic conditions"
//!   falsification:
//!     - criterion: "relative_error > 0.05"
//!       action: reject_model
//! ```

use super::model_card::EquationModelCard;
use serde::{Deserialize, Serialize};

/// A hypothesis to test in the experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentHypothesis {
    /// The null hypothesis (what we try to disprove)
    pub null: String,
    /// The alternative hypothesis
    pub alternative: String,
    /// Significance level (α)
    #[serde(default = "default_alpha")]
    pub alpha: f64,
}

fn default_alpha() -> f64 {
    0.05
}

impl ExperimentHypothesis {
    /// Create a new hypothesis.
    #[must_use]
    pub fn new(null: &str, alternative: &str) -> Self {
        Self {
            null: null.to_string(),
            alternative: alternative.to_string(),
            alpha: 0.05,
        }
    }

    /// Set the significance level.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

/// Action to take when falsification criterion is met.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FalsificationAction {
    /// Log warning but continue
    Warn,
    /// Stop the experiment
    Stop,
    /// Reject the model
    RejectModel,
    /// Flag for manual review
    FlagReview,
}

/// A criterion that would falsify the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationCriterion {
    /// Name/description of the criterion
    pub name: String,
    /// Mathematical expression defining the criterion
    pub criterion: String,
    /// Action to take if criterion is met
    pub action: FalsificationAction,
    /// Additional context or explanation
    #[serde(default)]
    pub context: String,
}

impl FalsificationCriterion {
    /// Create a new falsification criterion.
    #[must_use]
    pub fn new(name: &str, criterion: &str, action: FalsificationAction) -> Self {
        Self {
            name: name.to_string(),
            criterion: criterion.to_string(),
            action,
            context: String::new(),
        }
    }

    /// Add context.
    #[must_use]
    pub fn with_context(mut self, context: &str) -> Self {
        self.context = context.to_string();
        self
    }
}

/// Experiment specification following EDD principles.
///
/// Every experiment must have:
/// - Explicit seed (Pillar 3: Seed It)
/// - Falsification criteria (Pillar 4: Falsify It)
/// - Reference to EMC (Pillar 1: Prove It)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSpec {
    /// Unique name for this experiment
    name: String,
    /// Random seed for reproducibility
    seed: u64,
    /// Reference to the Equation Model Card
    #[serde(default)]
    emc_reference: Option<String>,
    /// Hypothesis to test
    #[serde(default)]
    hypothesis: Option<ExperimentHypothesis>,
    /// Falsification criteria
    #[serde(default)]
    falsification_criteria: Vec<FalsificationCriterion>,
    /// Number of replications
    #[serde(default = "default_replications")]
    replications: u32,
    /// Warmup period (time units)
    #[serde(default)]
    warmup: f64,
    /// Run length (time units)
    #[serde(default = "default_run_length")]
    run_length: f64,
    /// Description
    #[serde(default)]
    description: String,
}

fn default_replications() -> u32 {
    30
}

fn default_run_length() -> f64 {
    1000.0
}

impl ExperimentSpec {
    /// Create a new experiment spec builder.
    #[must_use]
    pub fn builder() -> ExperimentSpecBuilder {
        ExperimentSpecBuilder::new()
    }

    /// Get the experiment name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get the EMC reference.
    #[must_use]
    pub fn emc_reference(&self) -> Option<&str> {
        self.emc_reference.as_deref()
    }

    /// Get the hypothesis.
    #[must_use]
    pub fn hypothesis(&self) -> Option<&ExperimentHypothesis> {
        self.hypothesis.as_ref()
    }

    /// Get falsification criteria.
    #[must_use]
    pub fn falsification_criteria(&self) -> &[FalsificationCriterion] {
        &self.falsification_criteria
    }

    /// Get number of replications.
    #[must_use]
    pub fn replications(&self) -> u32 {
        self.replications
    }

    /// Get warmup period.
    #[must_use]
    pub fn warmup(&self) -> f64 {
        self.warmup
    }

    /// Get run length.
    #[must_use]
    pub fn run_length(&self) -> f64 {
        self.run_length
    }

    /// Get description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Parse from YAML string.
    ///
    /// # Errors
    /// Returns error if YAML is invalid or missing required fields.
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        serde_yaml::from_str(yaml).map_err(|e| format!("Failed to parse experiment YAML: {e}"))
    }

    /// Serialize to YAML string.
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn to_yaml(&self) -> Result<String, String> {
        serde_yaml::to_string(self).map_err(|e| format!("Failed to serialize experiment: {e}"))
    }

    /// Validate the experiment specification.
    ///
    /// # Errors
    /// Returns error if validation fails.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.name.is_empty() {
            errors.push("Experiment must have a name".to_string());
        }

        if self.replications == 0 {
            errors.push("Experiment must have at least 1 replication".to_string());
        }

        if self.run_length <= 0.0 {
            errors.push("Run length must be positive".to_string());
        }

        if self.warmup < 0.0 {
            errors.push("Warmup cannot be negative".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Builder for `ExperimentSpec`.
#[derive(Debug, Default)]
pub struct ExperimentSpecBuilder {
    name: Option<String>,
    seed: Option<u64>,
    emc_reference: Option<String>,
    hypothesis: Option<ExperimentHypothesis>,
    falsification_criteria: Vec<FalsificationCriterion>,
    replications: u32,
    warmup: f64,
    run_length: f64,
    description: String,
}

impl ExperimentSpecBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            replications: 30,
            run_length: 1000.0,
            ..Default::default()
        }
    }

    /// Set the experiment name.
    #[must_use]
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set the seed (required).
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the EMC reference.
    #[must_use]
    pub fn emc_reference(mut self, reference: &str) -> Self {
        self.emc_reference = Some(reference.to_string());
        self
    }

    /// Set the EMC directly (extracts reference).
    #[must_use]
    pub fn emc(mut self, emc: &EquationModelCard) -> Self {
        self.emc_reference = Some(format!("{}@{}", emc.name, emc.version));
        self
    }

    /// Set the hypothesis.
    #[must_use]
    pub fn hypothesis(mut self, hypothesis: ExperimentHypothesis) -> Self {
        self.hypothesis = Some(hypothesis);
        self
    }

    /// Add a falsification criterion.
    #[must_use]
    pub fn add_falsification_criterion(mut self, criterion: FalsificationCriterion) -> Self {
        self.falsification_criteria.push(criterion);
        self
    }

    /// Set number of replications.
    #[must_use]
    pub fn replications(mut self, n: u32) -> Self {
        self.replications = n;
        self
    }

    /// Set warmup period.
    #[must_use]
    pub fn warmup(mut self, warmup: f64) -> Self {
        self.warmup = warmup;
        self
    }

    /// Set run length.
    #[must_use]
    pub fn run_length(mut self, length: f64) -> Self {
        self.run_length = length;
        self
    }

    /// Set description.
    #[must_use]
    pub fn description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Build the experiment spec.
    ///
    /// # Errors
    /// Returns error if required fields are missing.
    pub fn build(self) -> Result<ExperimentSpec, String> {
        let name = self.name.ok_or("Experiment must have a name")?;
        let seed = self
            .seed
            .ok_or("Experiment must have an explicit seed (Pillar 3: Seed It)")?;

        Ok(ExperimentSpec {
            name,
            seed,
            emc_reference: self.emc_reference,
            hypothesis: self.hypothesis,
            falsification_criteria: self.falsification_criteria,
            replications: self.replications,
            warmup: self.warmup,
            run_length: self.run_length,
            description: self.description,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_requires_seed() {
        let result = ExperimentSpec::builder().name("Test Experiment").build();

        assert!(result.is_err());
        assert!(result.err().map(|e| e.contains("seed")).unwrap_or(false));
    }

    #[test]
    fn test_experiment_requires_name() {
        let result = ExperimentSpec::builder().seed(42).build();

        assert!(result.is_err());
        assert!(result.err().map(|e| e.contains("name")).unwrap_or(false));
    }

    #[test]
    fn test_experiment_builds_with_required_fields() {
        let result = ExperimentSpec::builder()
            .name("Test Experiment")
            .seed(42)
            .build();

        assert!(result.is_ok());
        let spec = result.ok();
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert_eq!(spec.seed(), 42);
        assert_eq!(spec.name(), "Test Experiment");
    }

    #[test]
    fn test_experiment_with_hypothesis() {
        let spec = ExperimentSpec::builder()
            .name("Little's Law Test")
            .seed(42)
            .hypothesis(ExperimentHypothesis::new(
                "L ≠ λW",
                "L = λW holds under stochastic conditions",
            ))
            .build()
            .ok();

        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert!(spec.hypothesis().is_some());
        assert_eq!(spec.hypothesis().unwrap().null, "L ≠ λW");
    }

    #[test]
    fn test_experiment_with_falsification() {
        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .add_falsification_criterion(FalsificationCriterion::new(
                "Error too high",
                "relative_error > 0.05",
                FalsificationAction::RejectModel,
            ))
            .build()
            .ok();

        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert_eq!(spec.falsification_criteria().len(), 1);
    }

    #[test]
    fn test_experiment_yaml_roundtrip() {
        let spec = ExperimentSpec::builder()
            .name("YAML Test")
            .seed(12345)
            .replications(50)
            .warmup(100.0)
            .run_length(5000.0)
            .description("Test experiment for YAML serialization")
            .build()
            .ok();

        assert!(spec.is_some());
        let spec = spec.unwrap();
        let yaml = spec.to_yaml();
        assert!(yaml.is_ok());
        let yaml = yaml.ok().unwrap();
        assert!(yaml.contains("name: YAML Test"));
        assert!(yaml.contains("seed: 12345"));
    }

    #[test]
    fn test_experiment_validation() {
        let spec = ExperimentSpec::builder()
            .name("Valid Experiment")
            .seed(42)
            .replications(30)
            .run_length(1000.0)
            .build()
            .ok();

        assert!(spec.is_some());
        let result = spec.unwrap().validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_hypothesis_alpha() {
        let hypothesis = ExperimentHypothesis::new("H0", "H1").with_alpha(0.01);

        assert!((hypothesis.alpha - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_falsification_action_serialization() {
        let criterion =
            FalsificationCriterion::new("test", "x > 0", FalsificationAction::RejectModel);

        let yaml = serde_yaml::to_string(&criterion).ok();
        assert!(yaml.is_some());
        let yaml = yaml.unwrap();
        assert!(yaml.contains("reject_model"));
    }

    #[test]
    fn test_experiment_spec_from_yaml() {
        let yaml = r#"
name: "Test"
seed: 42
replications: 10
warmup: 50.0
run_length: 500.0
description: "A test experiment"
"#;
        let spec = ExperimentSpec::from_yaml(yaml);
        assert!(spec.is_ok());
        let spec = spec.ok().unwrap();
        assert_eq!(spec.name(), "Test");
        assert_eq!(spec.seed(), 42);
        assert_eq!(spec.replications(), 10);
        assert!((spec.warmup() - 50.0).abs() < f64::EPSILON);
        assert!((spec.run_length() - 500.0).abs() < f64::EPSILON);
        assert_eq!(spec.description(), "A test experiment");
    }

    #[test]
    fn test_experiment_spec_from_yaml_invalid() {
        let yaml = "invalid: [yaml";
        let result = ExperimentSpec::from_yaml(yaml);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("Failed to parse"));
    }

    #[test]
    fn test_experiment_validation_fails_empty_name() {
        let spec = ExperimentSpec {
            name: String::new(),
            seed: 42,
            emc_reference: None,
            hypothesis: None,
            falsification_criteria: Vec::new(),
            replications: 30,
            warmup: 0.0,
            run_length: 1000.0,
            description: String::new(),
        };
        let result = spec.validate();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("name")));
    }

    #[test]
    fn test_experiment_validation_fails_zero_replications() {
        let spec = ExperimentSpec {
            name: "Test".to_string(),
            seed: 42,
            emc_reference: None,
            hypothesis: None,
            falsification_criteria: Vec::new(),
            replications: 0,
            warmup: 0.0,
            run_length: 1000.0,
            description: String::new(),
        };
        let result = spec.validate();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("replication")));
    }

    #[test]
    fn test_experiment_validation_fails_negative_run_length() {
        let spec = ExperimentSpec {
            name: "Test".to_string(),
            seed: 42,
            emc_reference: None,
            hypothesis: None,
            falsification_criteria: Vec::new(),
            replications: 30,
            warmup: 0.0,
            run_length: -100.0,
            description: String::new(),
        };
        let result = spec.validate();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("Run length")));
    }

    #[test]
    fn test_experiment_validation_fails_negative_warmup() {
        let spec = ExperimentSpec {
            name: "Test".to_string(),
            seed: 42,
            emc_reference: None,
            hypothesis: None,
            falsification_criteria: Vec::new(),
            replications: 30,
            warmup: -10.0,
            run_length: 1000.0,
            description: String::new(),
        };
        let result = spec.validate();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("Warmup")));
    }

    #[test]
    fn test_experiment_spec_emc_reference_getter() {
        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .emc_reference("test_emc@1.0")
            .build()
            .ok()
            .unwrap();

        assert_eq!(spec.emc_reference(), Some("test_emc@1.0"));
    }

    #[test]
    fn test_experiment_spec_builder_defaults() {
        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .build()
            .ok()
            .unwrap();

        // Check defaults
        assert_eq!(spec.replications(), 30);
        assert!((spec.run_length() - 1000.0).abs() < f64::EPSILON);
        assert!((spec.warmup() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_experiment_spec_builder_emc() {
        use crate::edd::equation::Citation;
        use crate::edd::model_card::{EmcBuilder, VerificationTest};

        let emc = EmcBuilder::new()
            .name("TestEMC")
            .version("2.0.0")
            .equation("y = x")
            .citation(Citation::new(&["Author"], "Journal", 2024))
            .add_verification_test_full(
                VerificationTest::new("test", 1.0, 0.1).with_input("x", 1.0),
            )
            .build()
            .ok()
            .unwrap();

        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .emc(&emc)
            .build()
            .ok()
            .unwrap();

        assert!(spec.emc_reference().is_some());
        let emc_ref = spec.emc_reference().unwrap();
        assert!(emc_ref.contains("TestEMC"));
        assert!(emc_ref.contains("2.0.0"));
    }

    #[test]
    fn test_experiment_spec_builder_description() {
        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .description("A detailed description")
            .build()
            .ok()
            .unwrap();

        assert_eq!(spec.description(), "A detailed description");
    }

    #[test]
    fn test_experiment_spec_builder_warmup() {
        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .warmup(200.0)
            .build()
            .ok()
            .unwrap();

        assert!((spec.warmup() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_experiment_spec_builder_run_length() {
        let spec = ExperimentSpec::builder()
            .name("Test")
            .seed(42)
            .run_length(5000.0)
            .build()
            .ok()
            .unwrap();

        assert!((spec.run_length() - 5000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_falsification_criterion_with_context() {
        let criterion = FalsificationCriterion::new("Test", "x > 0", FalsificationAction::Warn)
            .with_context("Additional context here");

        assert_eq!(criterion.context, "Additional context here");
    }

    #[test]
    fn test_falsification_action_variants() {
        assert_ne!(FalsificationAction::Warn, FalsificationAction::Stop);
        assert_ne!(FalsificationAction::Stop, FalsificationAction::RejectModel);
        assert_ne!(
            FalsificationAction::RejectModel,
            FalsificationAction::FlagReview
        );
        assert_ne!(FalsificationAction::FlagReview, FalsificationAction::Warn);
    }

    #[test]
    fn test_default_alpha() {
        assert!((default_alpha() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_replications() {
        assert_eq!(default_replications(), 30);
    }

    #[test]
    fn test_default_run_length() {
        assert!((default_run_length() - 1000.0).abs() < f64::EPSILON);
    }
}
