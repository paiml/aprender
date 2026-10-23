//! Conversion tests (moved out of loader.rs unchanged for the pre-commit complexity limit, #6025).

use super::*;
use crate::edd::equation::EquationClass;

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
