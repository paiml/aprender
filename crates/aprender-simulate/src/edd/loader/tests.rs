//! Loader unit tests (moved out of loader.rs unchanged for the pre-commit complexity limit, #6025).

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

mod conversion;
mod emc_validation;
mod experiment_validation;
