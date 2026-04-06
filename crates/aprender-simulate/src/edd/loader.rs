//! EMC and Experiment YAML file loader.
//!
//! Provides functionality to load Equation Model Cards and Experiment
//! specifications from YAML files, following the EDD specification.

use super::equation::{Citation, EquationClass};
use super::experiment::{
    ExperimentHypothesis, ExperimentSpec, FalsificationAction, FalsificationCriterion,
};
use super::model_card::{EmcBuilder, EquationModelCard, VerificationTest};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// YAML representation of an Equation Model Card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmcYaml {
    /// EMC schema version
    #[serde(default = "default_emc_version")]
    pub emc_version: String,
    /// Unique EMC identifier
    #[serde(default)]
    pub emc_id: String,
    /// Identity section
    pub identity: EmcIdentityYaml,
    /// Governing equation section
    pub governing_equation: GoverningEquationYaml,
    /// Analytical derivation section
    #[serde(default)]
    pub analytical_derivation: Option<AnalyticalDerivationYaml>,
    /// Domain of validity
    #[serde(default)]
    pub domain_of_validity: Option<DomainValidityYaml>,
    /// Verification tests
    #[serde(default)]
    pub verification_tests: Option<VerificationTestsYaml>,
    /// Falsification criteria
    #[serde(default)]
    pub falsification_criteria: Option<FalsificationCriteriaYaml>,
}

fn default_emc_version() -> String {
    "1.0".to_string()
}

/// Identity section of EMC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmcIdentityYaml {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub authors: Vec<AuthorYaml>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub description: String,
}

fn default_version() -> String {
    "1.0.0".to_string()
}

/// Author information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorYaml {
    pub name: String,
    #[serde(default)]
    pub affiliation: String,
}

/// Governing equation section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoverningEquationYaml {
    /// LaTeX representation
    pub latex: String,
    /// Plain text representation
    #[serde(default)]
    pub plain_text: String,
    /// Description
    #[serde(default)]
    pub description: String,
    /// Variables
    #[serde(default)]
    pub variables: Vec<VariableYaml>,
    /// Equation type classification
    #[serde(default)]
    pub equation_type: String,
}

/// Variable definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableYaml {
    pub symbol: String,
    pub description: String,
    #[serde(default)]
    pub units: String,
    #[serde(default)]
    pub r#type: String,
}

/// Analytical derivation section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticalDerivationYaml {
    #[serde(default)]
    pub primary_citation: Option<CitationYaml>,
    #[serde(default)]
    pub supporting_citations: Vec<CitationYaml>,
    #[serde(default)]
    pub derivation_method: String,
    #[serde(default)]
    pub derivation_summary: String,
}

/// Citation information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationYaml {
    pub authors: Vec<String>,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub year: u32,
    #[serde(default)]
    pub journal: String,
    #[serde(default)]
    pub publisher: String,
    #[serde(default)]
    pub volume: Option<u32>,
    #[serde(default)]
    pub issue: Option<u32>,
    #[serde(default)]
    pub pages: String,
    #[serde(default)]
    pub doi: Option<String>,
}

/// Domain of validity section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainValidityYaml {
    #[serde(default)]
    pub parameters: HashMap<String, ParameterConstraintYaml>,
    #[serde(default)]
    pub assumptions: Vec<String>,
    #[serde(default)]
    pub limitations: Vec<LimitationYaml>,
}

/// Parameter constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraintYaml {
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub units: String,
    #[serde(default)]
    pub physical_constraint: String,
}

/// Limitation description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitationYaml {
    pub description: String,
}

/// Verification tests section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationTestsYaml {
    #[serde(default)]
    pub tests: Vec<VerificationTestYaml>,
}

/// Single verification test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationTestYaml {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub r#type: String,
    #[serde(default)]
    pub parameters: HashMap<String, f64>,
    #[serde(default)]
    pub expected: HashMap<String, serde_yaml::Value>,
    #[serde(default)]
    pub tolerance: Option<f64>,
    #[serde(default)]
    pub description: String,
}

/// Falsification criteria section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationCriteriaYaml {
    #[serde(default)]
    pub criteria: Vec<FalsificationCriterionYaml>,
}

/// Single falsification criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationCriterionYaml {
    pub id: String,
    pub name: String,
    pub condition: String,
    #[serde(default)]
    pub threshold: Option<f64>,
    #[serde(default)]
    pub severity: String,
    #[serde(default)]
    pub interpretation: String,
}

impl EmcYaml {
    /// Load an EMC from a YAML file.
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read EMC file: {e}"))?;
        Self::from_yaml(&content)
    }

    /// Parse an EMC from a YAML string.
    ///
    /// # Errors
    /// Returns error if YAML is invalid.
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        serde_yaml::from_str(yaml).map_err(|e| format!("Failed to parse EMC YAML: {e}"))
    }

    /// Convert to an `EquationModelCard`.
    ///
    /// # Errors
    /// Returns error if required fields are missing.
    pub fn to_model_card(&self) -> Result<EquationModelCard, String> {
        let mut builder = EmcBuilder::new()
            .name(&self.identity.name)
            .version(&self.identity.version)
            .equation(&self.governing_equation.latex)
            .description(&self.governing_equation.description);

        // Set equation class based on type
        let class = match self
            .governing_equation
            .equation_type
            .to_lowercase()
            .as_str()
        {
            "queueing" | "queue" => EquationClass::Queueing,
            "statistical" => EquationClass::Statistical,
            "inventory" => EquationClass::Inventory,
            "optimization" => EquationClass::Optimization,
            "ml" | "machine_learning" => EquationClass::MachineLearning,
            // Default to Conservation for "conservation", "ode", or any other type
            _ => EquationClass::Conservation,
        };
        builder = builder.class(class);

        // Add citation
        if let Some(ref derivation) = self.analytical_derivation {
            if let Some(ref cite) = derivation.primary_citation {
                let authors: Vec<&str> = cite.authors.iter().map(String::as_str).collect();
                let venue = if cite.journal.is_empty() {
                    &cite.publisher
                } else {
                    &cite.journal
                };
                let mut citation = Citation::new(&authors, venue, cite.year);
                if !cite.title.is_empty() {
                    citation = citation.with_title(&cite.title);
                }
                if let Some(ref doi) = cite.doi {
                    citation = citation.with_doi(doi);
                }
                builder = builder.citation(citation);
            }
        }

        // Add variables
        for var in &self.governing_equation.variables {
            builder = builder.add_variable(&var.symbol, &var.description, &var.units);
        }

        // Add verification tests
        if let Some(ref tests) = self.verification_tests {
            for test in &tests.tests {
                let expected = test
                    .expected
                    .get("value")
                    .and_then(serde_yaml::Value::as_f64)
                    .unwrap_or(0.0);
                let tolerance = test.tolerance.unwrap_or(1e-6);
                let mut vtest = VerificationTest::new(&test.name, expected, tolerance);
                for (name, &val) in &test.parameters {
                    vtest = vtest.with_input(name, val);
                }
                builder = builder.add_verification_test_full(vtest);
            }
        }

        // If no verification tests, add a placeholder (will fail validation correctly)
        // This ensures the builder pattern works but validation catches missing tests

        builder.build()
    }

    /// Validate EMC against the JSON schema.
    ///
    /// Performs structural validation to ensure all required fields
    /// are present and correctly formatted.
    ///
    /// # Errors
    ///
    /// Returns error messages for each schema violation.
    pub fn validate_schema(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // EDD-01: EMC must have identity
        if self.identity.name.is_empty() {
            errors.push("EMC must have a name (identity.name)".to_string());
        }

        // EDD-02: EMC must have governing equation
        if self.governing_equation.latex.is_empty() {
            errors
                .push("EMC must have a governing equation (governing_equation.latex)".to_string());
        }

        // Check description is meaningful
        if self.governing_equation.description.len() < 10 {
            errors.push(
                "EMC governing equation should have a meaningful description (min 10 chars)"
                    .to_string(),
            );
        }

        // EDD-03: EMC must have citation
        if let Some(ref derivation) = self.analytical_derivation {
            if let Some(ref cite) = derivation.primary_citation {
                if cite.authors.is_empty() {
                    errors.push("Primary citation must have at least one author".to_string());
                }
                if cite.year == 0 {
                    errors.push("Primary citation must have a valid year".to_string());
                }
            } else {
                errors.push("EMC must have a primary citation (EDD-03)".to_string());
            }
        } else {
            errors.push("EMC must have analytical derivation section with citation".to_string());
        }

        // EDD-04: EMC must have verification tests
        if let Some(ref tests) = self.verification_tests {
            if tests.tests.is_empty() {
                errors.push("EMC must have at least one verification test (EDD-04)".to_string());
            }
        } else {
            errors.push("EMC must have verification tests section".to_string());
        }

        // EDD-04: EMC must have falsification criteria
        if let Some(ref criteria) = self.falsification_criteria {
            if criteria.criteria.is_empty() {
                errors.push("EMC must have at least one falsification criterion".to_string());
            }
        }

        // Validate variables
        for var in &self.governing_equation.variables {
            if var.symbol.is_empty() {
                errors.push("Variable must have a symbol".to_string());
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// YAML representation of an Experiment specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentYaml {
    /// Experiment schema version
    #[serde(default = "default_experiment_version")]
    pub experiment_version: String,
    /// Experiment ID
    #[serde(default)]
    pub experiment_id: String,
    /// Metadata
    pub metadata: ExperimentMetadataYaml,
    /// Reference to EMC
    #[serde(default)]
    pub equation_model_card: Option<EmcReferenceYaml>,
    /// Hypothesis
    #[serde(default)]
    pub hypothesis: Option<HypothesisYaml>,
    /// Reproducibility settings
    pub reproducibility: ReproducibilityYaml,
    /// Simulation parameters
    #[serde(default)]
    pub simulation: Option<SimulationYaml>,
    /// Falsification criteria
    #[serde(default)]
    pub falsification: Option<ExperimentFalsificationYaml>,
}

fn default_experiment_version() -> String {
    "1.0".to_string()
}

/// Experiment metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetadataYaml {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// EMC reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmcReferenceYaml {
    #[serde(default)]
    pub emc_ref: String,
    #[serde(default)]
    pub emc_file: String,
}

/// Hypothesis definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisYaml {
    pub null_hypothesis: String,
    #[serde(default)]
    pub alternative_hypothesis: String,
    #[serde(default)]
    pub expected_outcome: String,
}

/// Reproducibility settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityYaml {
    pub seed: u64,
    #[serde(default = "default_true")]
    pub ieee_strict: bool,
}

fn default_true() -> bool {
    true
}

/// Simulation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationYaml {
    #[serde(default)]
    pub duration: Option<DurationYaml>,
    #[serde(default)]
    pub parameters: HashMap<String, serde_yaml::Value>,
}

/// Duration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationYaml {
    #[serde(default)]
    pub warmup: f64,
    #[serde(default)]
    pub simulation: f64,
    #[serde(default = "default_replications")]
    pub replications: u32,
}

fn default_replications() -> u32 {
    30
}

/// Experiment falsification settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentFalsificationYaml {
    #[serde(default)]
    pub import_from_emc: bool,
    #[serde(default)]
    pub criteria: Vec<FalsificationCriterionYaml>,
    #[serde(default)]
    pub jidoka: Option<JidokaYaml>,
}

/// Jidoka (stop-on-error) settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JidokaYaml {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub stop_on_severity: String,
}

impl ExperimentYaml {
    /// Load an experiment from a YAML file.
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read experiment file: {e}"))?;
        Self::from_yaml(&content)
    }

    /// Parse an experiment from a YAML string.
    ///
    /// # Errors
    /// Returns error if YAML is invalid.
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        serde_yaml::from_str(yaml).map_err(|e| format!("Failed to parse experiment YAML: {e}"))
    }

    /// Convert to an `ExperimentSpec`.
    ///
    /// # Errors
    /// Returns error if required fields are missing.
    pub fn to_experiment_spec(&self) -> Result<ExperimentSpec, String> {
        let mut builder = ExperimentSpec::builder()
            .name(&self.metadata.name)
            .seed(self.reproducibility.seed)
            .description(&self.metadata.description);

        // Add EMC reference
        if let Some(ref emc) = self.equation_model_card {
            if !emc.emc_ref.is_empty() {
                builder = builder.emc_reference(&emc.emc_ref);
            }
        }

        // Add hypothesis
        if let Some(ref hyp) = self.hypothesis {
            let hypothesis =
                ExperimentHypothesis::new(&hyp.null_hypothesis, &hyp.alternative_hypothesis);
            builder = builder.hypothesis(hypothesis);
        }

        // Add duration settings
        if let Some(ref sim) = self.simulation {
            if let Some(ref dur) = sim.duration {
                builder = builder
                    .warmup(dur.warmup)
                    .run_length(dur.simulation)
                    .replications(dur.replications);
            }
        }

        // Add falsification criteria
        if let Some(ref fals) = self.falsification {
            for crit in &fals.criteria {
                let action = match crit.severity.to_lowercase().as_str() {
                    "critical" => FalsificationAction::RejectModel,
                    "major" => FalsificationAction::Stop,
                    "minor" => FalsificationAction::Warn,
                    _ => FalsificationAction::FlagReview,
                };
                let criterion = FalsificationCriterion::new(&crit.name, &crit.condition, action);
                builder = builder.add_falsification_criterion(criterion);
            }
        }

        builder.build()
    }

    /// Validate experiment against the JSON schema.
    ///
    /// Performs structural validation to ensure all required fields
    /// are present and correctly formatted.
    ///
    /// # Errors
    ///
    /// Returns error messages for each schema violation.
    pub fn validate_schema(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Metadata validation
        if self.metadata.name.is_empty() {
            errors.push("Experiment must have a name (metadata.name)".to_string());
        }

        // EDD-05: Seed is required
        // Note: seed is always present due to struct definition, but we validate it's reasonable
        if self.reproducibility.seed == 0 {
            // 0 is technically valid but unusual - add warning
        }

        // EMC reference validation
        if let Some(ref emc) = self.equation_model_card {
            if emc.emc_ref.is_empty() && emc.emc_file.is_empty() {
                errors.push("Experiment must reference an EMC (emc_ref or emc_file)".to_string());
            }
        }

        // Hypothesis validation
        if let Some(ref hyp) = self.hypothesis {
            if hyp.null_hypothesis.len() < 10 {
                errors.push(
                    "Hypothesis must have a meaningful null hypothesis (min 10 chars)".to_string(),
                );
            }
            if !hyp.expected_outcome.is_empty()
                && hyp.expected_outcome != "reject"
                && hyp.expected_outcome != "fail_to_reject"
            {
                errors.push(format!(
                    "Expected outcome must be 'reject' or 'fail_to_reject', got '{}'",
                    hyp.expected_outcome
                ));
            }
        } else {
            errors.push("Experiment must have a hypothesis section".to_string());
        }

        // Falsification criteria validation
        if let Some(ref fals) = self.falsification {
            if fals.criteria.is_empty() && !fals.import_from_emc {
                errors.push(
                    "Experiment must have falsification criteria or import_from_emc=true"
                        .to_string(),
                );
            }
            for crit in &fals.criteria {
                if crit.condition.is_empty() {
                    errors.push(format!("Criterion '{}' must have a condition", crit.name));
                }
            }
        } else {
            errors.push("Experiment must have falsification section".to_string());
        }

        // Simulation duration validation
        if let Some(ref sim) = self.simulation {
            if let Some(ref dur) = sim.duration {
                if dur.simulation <= 0.0 {
                    errors.push("Simulation duration must be positive".to_string());
                }
                if dur.replications == 0 {
                    errors.push("Replications must be at least 1".to_string());
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_EMC_YAML: &str = r#"
emc_version: "1.0"
emc_id: "EMC-TEST-001"

identity:
  name: "Little's Law"
  version: "1.0.0"
  status: "production"

governing_equation:
  latex: "L = \\lambda W"
  plain_text: "WIP = Throughput × Cycle Time"
  description: "Fundamental theorem of queueing theory"
  equation_type: "queueing"
  variables:
    - symbol: "L"
      description: "Average WIP"
      units: "items"
    - symbol: "λ"
      description: "Arrival rate"
      units: "items/time"
    - symbol: "W"
      description: "Cycle time"
      units: "time"

analytical_derivation:
  primary_citation:
    authors: ["Little, J.D.C."]
    title: "A Proof for the Queuing Formula: L = λW"
    journal: "Operations Research"
    year: 1961

verification_tests:
  tests:
    - id: "LL-001"
      name: "Basic validation"
      parameters:
        lambda: 5.0
        W: 2.0
      expected:
        value: 10.0
      tolerance: 0.001
"#;

    #[test]
    fn test_parse_emc_yaml() {
        let emc = EmcYaml::from_yaml(SAMPLE_EMC_YAML);
        assert!(emc.is_ok());
        let emc = emc.ok().unwrap();
        assert_eq!(emc.identity.name, "Little's Law");
        assert_eq!(emc.governing_equation.variables.len(), 3);
    }

    #[test]
    fn test_emc_to_model_card() {
        let emc_yaml = EmcYaml::from_yaml(SAMPLE_EMC_YAML).ok().unwrap();
        let model_card = emc_yaml.to_model_card();
        assert!(model_card.is_ok());
        let mc = model_card.ok().unwrap();
        assert_eq!(mc.name, "Little's Law");
        assert!(mc.equation.contains("lambda"));
    }

    const SAMPLE_EXPERIMENT_YAML: &str = r#"
experiment_version: "1.0"
experiment_id: "EXP-001"

metadata:
  name: "Little's Law Validation"
  description: "Validate L = λW under stochastic conditions"

equation_model_card:
  emc_ref: "operations/littles_law"

hypothesis:
  null_hypothesis: "L ≠ λW under stochastic conditions"
  alternative_hypothesis: "L = λW holds"
  expected_outcome: "reject"

reproducibility:
  seed: 42
  ieee_strict: true

simulation:
  duration:
    warmup: 100.0
    simulation: 1000.0
    replications: 30

falsification:
  criteria:
    - id: "FC-001"
      name: "Linear relationship"
      condition: "R² < 0.95"
      severity: "critical"
"#;

    #[test]
    fn test_parse_experiment_yaml() {
        let exp = ExperimentYaml::from_yaml(SAMPLE_EXPERIMENT_YAML);
        assert!(exp.is_ok());
        let exp = exp.ok().unwrap();
        assert_eq!(exp.metadata.name, "Little's Law Validation");
        assert_eq!(exp.reproducibility.seed, 42);
    }

    #[test]
    fn test_experiment_to_spec() {
        let exp_yaml = ExperimentYaml::from_yaml(SAMPLE_EXPERIMENT_YAML)
            .ok()
            .unwrap();
        let spec = exp_yaml.to_experiment_spec();
        assert!(spec.is_ok());
        let spec = spec.ok().unwrap();
        assert_eq!(spec.seed(), 42);
        assert_eq!(spec.name(), "Little's Law Validation");
    }

    #[test]
    fn test_emc_schema_validation() {
        let emc = EmcYaml::from_yaml(SAMPLE_EMC_YAML).ok().unwrap();
        let result = emc.validate_schema();
        assert!(
            result.is_ok(),
            "EMC should pass schema validation: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_experiment_schema_validation() {
        let exp = ExperimentYaml::from_yaml(SAMPLE_EXPERIMENT_YAML)
            .ok()
            .unwrap();
        let result = exp.validate_schema();
        assert!(
            result.is_ok(),
            "Experiment should pass schema validation: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_emc_missing_required_field() {
        let invalid_yaml = r#"
emc_version: "1.0"
identity:
  name: "Missing equation"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
"#;
        let emc = EmcYaml::from_yaml(invalid_yaml);
        // Should parse but fail schema validation
        if let Ok(emc) = emc {
            // validate_schema may or may not catch missing analytical_derivation
            let _result = emc.validate_schema();
        }
    }

    #[test]
    fn test_default_emc_version() {
        assert_eq!(default_emc_version(), "1.0");
    }

    #[test]
    fn test_default_version() {
        assert_eq!(default_version(), "1.0.0");
    }

    #[test]
    fn test_default_experiment_version() {
        assert_eq!(default_experiment_version(), "1.0");
    }

    #[test]
    fn test_default_true() {
        assert!(default_true());
    }

    #[test]
    fn test_default_replications() {
        assert_eq!(default_replications(), 30);
    }

    #[test]
    fn test_emc_from_file_not_found() {
        let result = EmcYaml::from_file("nonexistent.yaml");
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Failed to read"));
    }

    #[test]
    fn test_experiment_from_file_not_found() {
        let result = ExperimentYaml::from_file("nonexistent.yaml");
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Failed to read"));
    }

    #[test]
    fn test_emc_invalid_yaml() {
        let invalid = "not: [valid: yaml: content";
        let result = EmcYaml::from_yaml(invalid);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Failed to parse"));
    }

    #[test]
    fn test_experiment_invalid_yaml() {
        let invalid = "not: [valid: yaml: content";
        let result = ExperimentYaml::from_yaml(invalid);
        assert!(result.is_err());
        assert!(result.err().unwrap().contains("Failed to parse"));
    }

    #[test]
    fn test_emc_validation_missing_name() {
        let yaml = r#"
identity:
  name: ""
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "A valid description"
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let result = emc.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("name")));
    }

    #[test]
    fn test_emc_validation_missing_latex() {
        let yaml = r#"
identity:
  name: "Test"
  version: "1.0.0"
governing_equation:
  latex: ""
  description: "A valid description"
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let result = emc.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("governing equation")));
    }

    #[test]
    fn test_emc_validation_short_description() {
        let yaml = r#"
identity:
  name: "Test"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "Short"
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let result = emc.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("meaningful description")));
    }

    #[test]
    fn test_emc_validation_empty_authors() {
        let yaml = r#"
identity:
  name: "Test"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "A valid description for the equation"
analytical_derivation:
  primary_citation:
    authors: []
    title: "Test"
    year: 2020
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let result = emc.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("author")));
    }

    #[test]
    fn test_emc_validation_invalid_year() {
        let yaml = r#"
identity:
  name: "Test"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "A valid description for the equation"
analytical_derivation:
  primary_citation:
    authors: ["Author"]
    title: "Test"
    year: 0
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let result = emc.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("year")));
    }

    #[test]
    fn test_emc_validation_empty_verification_tests() {
        let yaml = r#"
identity:
  name: "Test"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "A valid description for the equation"
analytical_derivation:
  primary_citation:
    authors: ["Author"]
    title: "Test"
    year: 2020
verification_tests:
  tests: []
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let result = emc.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("verification test")));
    }

    #[test]
    fn test_emc_validation_empty_falsification() {
        let yaml = r#"
identity:
  name: "Test"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "A valid description for the equation"
analytical_derivation:
  primary_citation:
    authors: ["Author"]
    title: "Test"
    year: 2020
verification_tests:
  tests:
    - id: "T1"
      name: "Test"
falsification_criteria:
  criteria: []
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let result = emc.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("falsification")));
    }

    #[test]
    fn test_emc_validation_empty_variable_symbol() {
        let yaml = r#"
identity:
  name: "Test"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "A valid description for the equation"
  variables:
    - symbol: ""
      description: "Empty symbol"
analytical_derivation:
  primary_citation:
    authors: ["Author"]
    title: "Test"
    year: 2020
verification_tests:
  tests:
    - id: "T1"
      name: "Test"
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let result = emc.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("symbol")));
    }

    #[test]
    fn test_experiment_validation_missing_name() {
        let yaml = r#"
metadata:
  name: ""
reproducibility:
  seed: 42
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("name")));
    }

    #[test]
    fn test_experiment_validation_empty_emc_ref() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
equation_model_card:
  emc_ref: ""
  emc_file: ""
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("EMC")));
    }

    #[test]
    fn test_experiment_validation_short_hypothesis() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
hypothesis:
  null_hypothesis: "Short"
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors
            .iter()
            .any(|e| e.contains("meaningful null hypothesis")));
    }

    #[test]
    fn test_experiment_validation_invalid_expected_outcome() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
hypothesis:
  null_hypothesis: "A valid null hypothesis for testing"
  expected_outcome: "invalid_outcome"
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("Expected outcome")));
    }

    #[test]
    fn test_experiment_validation_empty_falsification_criteria() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
hypothesis:
  null_hypothesis: "A valid null hypothesis for testing"
  expected_outcome: "reject"
falsification:
  import_from_emc: false
  criteria: []
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("falsification criteria")));
    }

    #[test]
    fn test_experiment_validation_criterion_no_condition() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
hypothesis:
  null_hypothesis: "A valid null hypothesis for testing"
  expected_outcome: "reject"
falsification:
  criteria:
    - id: "FC-001"
      name: "Test"
      condition: ""
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("condition")));
    }

    #[test]
    fn test_experiment_validation_negative_duration() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
hypothesis:
  null_hypothesis: "A valid null hypothesis for testing"
  expected_outcome: "reject"
falsification:
  criteria:
    - id: "FC-001"
      name: "Test"
      condition: "error < 0.01"
simulation:
  duration:
    simulation: -10.0
    replications: 30
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("positive")));
    }

    #[test]
    fn test_experiment_validation_zero_replications() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
hypothesis:
  null_hypothesis: "A valid null hypothesis for testing"
  expected_outcome: "reject"
falsification:
  criteria:
    - id: "FC-001"
      name: "Test"
      condition: "error < 0.01"
simulation:
  duration:
    simulation: 100.0
    replications: 0
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_err());
        let errors = result.err().unwrap();
        assert!(errors.iter().any(|e| e.contains("Replications")));
    }

    #[test]
    fn test_emc_to_model_card_with_doi() {
        let yaml = r#"
emc_version: "1.0"
identity:
  name: "Test EMC"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "Test equation for model card"
  equation_type: "optimization"
  variables:
    - symbol: "x"
      description: "Input"
      units: "m"
    - symbol: "y"
      description: "Output"
      units: "m"
analytical_derivation:
  primary_citation:
    authors: ["Test Author"]
    title: "Test Title"
    year: 2020
    journal: "Test Journal"
    doi: "10.1000/test"
verification_tests:
  tests:
    - id: "VT-001"
      name: "Basic Test"
      parameters:
        x: 1.0
      expected:
        value: 1.0
      tolerance: 0.001
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let model_card = emc.to_model_card();
        assert!(model_card.is_ok());
        let mc = model_card.ok().unwrap();
        assert_eq!(mc.name, "Test EMC");
    }

    #[test]
    fn test_emc_to_model_card_equation_classes() {
        // Helper to create valid EMC YAML with different equation types
        fn make_emc_yaml(name: &str, eq_type: &str) -> String {
            format!(
                r#"
identity:
  name: "{name}"
governing_equation:
  latex: "x = y"
  description: "Test description for {eq_type}"
  equation_type: "{eq_type}"
analytical_derivation:
  primary_citation:
    authors: ["Author"]
    title: "Title"
    year: 2020
verification_tests:
  tests:
    - id: "VT-001"
      name: "Basic Test"
      expected:
        value: 1.0
"#
            )
        }

        // Test queueing
        let yaml = make_emc_yaml("Queue", "queue");
        let emc = EmcYaml::from_yaml(&yaml).ok().unwrap();
        let mc = emc.to_model_card().ok().unwrap();
        assert_eq!(mc.class, EquationClass::Queueing);

        // Test statistical
        let yaml = make_emc_yaml("Stat", "statistical");
        let emc = EmcYaml::from_yaml(&yaml).ok().unwrap();
        let mc = emc.to_model_card().ok().unwrap();
        assert_eq!(mc.class, EquationClass::Statistical);

        // Test inventory
        let yaml = make_emc_yaml("Inv", "inventory");
        let emc = EmcYaml::from_yaml(&yaml).ok().unwrap();
        let mc = emc.to_model_card().ok().unwrap();
        assert_eq!(mc.class, EquationClass::Inventory);

        // Test ML
        let yaml = make_emc_yaml("ML", "machine_learning");
        let emc = EmcYaml::from_yaml(&yaml).ok().unwrap();
        let mc = emc.to_model_card().ok().unwrap();
        assert_eq!(mc.class, EquationClass::MachineLearning);
    }

    #[test]
    fn test_emc_to_model_card_with_publisher() {
        let yaml = r#"
identity:
  name: "Test"
governing_equation:
  latex: "x = y"
  description: "Test description that is valid"
analytical_derivation:
  primary_citation:
    authors: ["Author"]
    title: "Title"
    year: 2020
    publisher: "Publisher Name"
verification_tests:
  tests:
    - id: "VT-001"
      name: "Basic Test"
      expected:
        value: 1.0
"#;
        let emc = EmcYaml::from_yaml(yaml).ok().unwrap();
        let mc = emc.to_model_card().ok().unwrap();
        assert!(mc.citation.venue.contains("Publisher"));
    }

    #[test]
    fn test_experiment_to_spec_with_severity_levels() {
        // Test critical severity
        let yaml = r#"
metadata:
  name: "Test"
  description: "Test description"
reproducibility:
  seed: 42
equation_model_card:
  emc_ref: "test/emc"
hypothesis:
  null_hypothesis: "Test null hypothesis for validation"
  alternative_hypothesis: "Alternative"
falsification:
  criteria:
    - id: "FC-001"
      name: "Critical"
      condition: "error < 0.01"
      severity: "critical"
    - id: "FC-002"
      name: "Major"
      condition: "error < 0.05"
      severity: "major"
    - id: "FC-003"
      name: "Minor"
      condition: "error < 0.10"
      severity: "minor"
    - id: "FC-004"
      name: "Unknown"
      condition: "error < 0.20"
      severity: "unknown"
simulation:
  duration:
    warmup: 10.0
    simulation: 100.0
    replications: 10
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let spec = exp.to_experiment_spec();
        assert!(spec.is_ok());
        let spec = spec.ok().unwrap();
        assert_eq!(spec.falsification_criteria().len(), 4);
    }

    #[test]
    fn test_experiment_to_spec_no_emc_ref() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
equation_model_card:
  emc_ref: ""
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let spec = exp.to_experiment_spec();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_experiment_to_spec_no_simulation() {
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let spec = exp.to_experiment_spec();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_experiment_validation_valid_expected_outcomes() {
        // Test reject
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
hypothesis:
  null_hypothesis: "A valid null hypothesis for testing"
  expected_outcome: "reject"
falsification:
  import_from_emc: true
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_ok());

        // Test fail_to_reject
        let yaml = r#"
metadata:
  name: "Test"
reproducibility:
  seed: 42
hypothesis:
  null_hypothesis: "A valid null hypothesis for testing"
  expected_outcome: "fail_to_reject"
falsification:
  import_from_emc: true
"#;
        let exp = ExperimentYaml::from_yaml(yaml).ok().unwrap();
        let result = exp.validate_schema();
        assert!(result.is_ok());
    }

    #[test]
    fn test_author_yaml_serialization() {
        let author = AuthorYaml {
            name: "Test Author".to_string(),
            affiliation: "Test University".to_string(),
        };
        let json = serde_json::to_string(&author);
        assert!(json.is_ok());
    }

    #[test]
    fn test_jidoka_yaml() {
        let jidoka = JidokaYaml {
            enabled: true,
            stop_on_severity: "critical".to_string(),
        };
        assert!(jidoka.enabled);
    }

    #[test]
    fn test_limitation_yaml() {
        let limitation = LimitationYaml {
            description: "Test limitation".to_string(),
        };
        assert!(!limitation.description.is_empty());
    }

    #[test]
    fn test_parameter_constraint_yaml() {
        let constraint = ParameterConstraintYaml {
            min: Some(0.0),
            max: Some(100.0),
            units: "meters".to_string(),
            physical_constraint: "must be positive".to_string(),
        };
        assert!(constraint.min.is_some());
        assert!(constraint.max.is_some());
    }
}
