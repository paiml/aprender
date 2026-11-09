//! EMC validation tests (moved out of loader.rs unchanged for the pre-commit complexity limit, #6025).

use super::*;

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
