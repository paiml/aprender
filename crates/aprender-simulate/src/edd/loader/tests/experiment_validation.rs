//! Experiment validation tests (moved out of loader.rs unchanged for the pre-commit complexity limit, #6025).

use super::*;

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
