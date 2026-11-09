//! EMC and Experiment YAML file loader.
//!
//! Provides functionality to load Equation Model Cards and Experiment
//! specifications from YAML files, following the EDD specification.

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
}

mod emc_convert;
mod emc_validate;
mod experiment_convert;
mod experiment_validate;

#[cfg(test)]
mod tests;
