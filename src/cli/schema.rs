//! EMC schema validation.
//!
//! This module contains the EMC YAML schema validation logic.
//! Extracted to enable comprehensive testing of validation rules.

/// Validate an EMC YAML document against the schema.
///
/// Returns a tuple of (errors, warnings).
/// Errors indicate schema violations that must be fixed.
/// Warnings indicate missing recommended fields.
///
/// # Arguments
///
/// * `yaml` - The parsed YAML document to validate
///
/// # Returns
///
/// A tuple of (errors, warnings) vectors.
#[must_use]
pub fn validate_emc_schema(yaml: &serde_yaml::Value) -> (Vec<String>, Vec<String>) {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Required top-level fields
    let required = [
        "emc_version",
        "emc_id",
        "identity",
        "governing_equation",
        "analytical_derivation",
        "domain_of_validity",
    ];

    for field in required {
        if yaml.get(field).is_none() {
            errors.push(format!("Missing required field: {field}"));
        }
    }

    // Validate identity section
    if let Some(identity) = yaml.get("identity") {
        if identity.get("name").is_none() {
            errors.push("Missing required field: identity.name".to_string());
        }
        if identity.get("version").is_none() {
            warnings.push("Missing recommended field: identity.version".to_string());
        }
    }

    // Validate governing_equation section
    if let Some(eq) = yaml.get("governing_equation") {
        if eq.get("latex").is_none() && eq.get("plain_text").is_none() {
            errors.push("governing_equation must have 'latex' or 'plain_text'".to_string());
        }
    }

    // Validate analytical_derivation section
    if let Some(deriv) = yaml.get("analytical_derivation") {
        if deriv.get("primary_citation").is_none() {
            errors.push("Missing: analytical_derivation.primary_citation".to_string());
        }
    }

    // EDD-required sections (warnings only, not hard errors)
    if yaml.get("verification_tests").is_none() {
        warnings.push("Missing EDD-required section: verification_tests".to_string());
    }
    if yaml.get("falsification_criteria").is_none() {
        warnings.push("Missing EDD-required section: falsification_criteria".to_string());
    }

    (errors, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::Value;

    fn minimal_valid_emc() -> Value {
        serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test EMC"
              version: "1.0.0"
            governing_equation:
              latex: "E = mc^2"
            analytical_derivation:
              primary_citation: "Einstein, A. (1905)"
            domain_of_validity:
              description: "All domains"
        "#,
        )
        .ok()
        .unwrap_or(Value::Null)
    }

    #[test]
    fn test_valid_emc_no_errors() {
        let yaml = minimal_valid_emc();
        let (errors, warnings) = validate_emc_schema(&yaml);
        assert!(errors.is_empty(), "Unexpected errors: {errors:?}");
        // Warnings for missing EDD sections are expected
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn test_missing_emc_version() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_id: "test/example"
            identity:
              name: "Test"
            governing_equation:
              latex: "x"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors.iter().any(|e| e.contains("emc_version")));
    }

    #[test]
    fn test_missing_emc_id() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            identity:
              name: "Test"
            governing_equation:
              latex: "x"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors.iter().any(|e| e.contains("emc_id")));
    }

    #[test]
    fn test_missing_identity() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            governing_equation:
              latex: "x"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors.iter().any(|e| e.contains("identity")));
    }

    #[test]
    fn test_missing_identity_name() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              version: "1.0.0"
            governing_equation:
              latex: "x"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors.iter().any(|e| e.contains("identity.name")));
    }

    #[test]
    fn test_missing_identity_version_warning() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
            governing_equation:
              latex: "x"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, warnings) = validate_emc_schema(&yaml);
        assert!(errors.is_empty());
        assert!(warnings.iter().any(|w| w.contains("identity.version")));
    }

    #[test]
    fn test_missing_governing_equation() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors.iter().any(|e| e.contains("governing_equation")));
    }

    #[test]
    fn test_governing_equation_no_latex_or_plaintext() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
            governing_equation:
              description: "some equation"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors
            .iter()
            .any(|e| e.contains("latex") || e.contains("plain_text")));
    }

    #[test]
    fn test_governing_equation_with_plain_text_only() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
              version: "1.0"
            governing_equation:
              plain_text: "E equals mc squared"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        // Should not have the latex/plain_text error
        assert!(!errors
            .iter()
            .any(|e| e.contains("latex") || e.contains("plain_text")));
    }

    #[test]
    fn test_missing_analytical_derivation() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
            governing_equation:
              latex: "x"
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors.iter().any(|e| e.contains("analytical_derivation")));
    }

    #[test]
    fn test_missing_primary_citation() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
            governing_equation:
              latex: "x"
            analytical_derivation:
              secondary_sources: []
            domain_of_validity: {}
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors.iter().any(|e| e.contains("primary_citation")));
    }

    #[test]
    fn test_missing_domain_of_validity() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
            governing_equation:
              latex: "x"
            analytical_derivation:
              primary_citation: "cite"
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        assert!(errors.iter().any(|e| e.contains("domain_of_validity")));
    }

    #[test]
    fn test_edd_sections_warnings() {
        let yaml = minimal_valid_emc();
        let (_, warnings) = validate_emc_schema(&yaml);
        assert!(warnings.iter().any(|w| w.contains("verification_tests")));
        assert!(warnings
            .iter()
            .any(|w| w.contains("falsification_criteria")));
    }

    #[test]
    fn test_with_verification_tests_no_warning() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
              version: "1.0"
            governing_equation:
              latex: "x"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
            verification_tests:
              - id: test1
                description: "Test"
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (_, warnings) = validate_emc_schema(&yaml);
        assert!(!warnings.iter().any(|w| w.contains("verification_tests")));
    }

    #[test]
    fn test_with_falsification_criteria_no_warning() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            emc_version: "1.0"
            emc_id: "test/example"
            identity:
              name: "Test"
              version: "1.0"
            governing_equation:
              latex: "x"
            analytical_derivation:
              primary_citation: "cite"
            domain_of_validity: {}
            falsification_criteria:
              - id: crit1
                description: "Criteria"
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (_, warnings) = validate_emc_schema(&yaml);
        assert!(!warnings
            .iter()
            .any(|w| w.contains("falsification_criteria")));
    }

    #[test]
    fn test_empty_yaml() {
        let yaml = Value::Null;
        let (errors, _) = validate_emc_schema(&yaml);
        // Should have errors for all required fields
        assert!(errors.len() >= 6);
    }

    #[test]
    fn test_multiple_errors() {
        let yaml: Value = serde_yaml::from_str(
            r#"
            identity:
              version: "1.0"
        "#,
        )
        .ok()
        .unwrap_or(Value::Null);

        let (errors, _) = validate_emc_schema(&yaml);
        // Should have multiple errors
        assert!(errors.len() >= 5);
        assert!(errors.iter().any(|e| e.contains("emc_version")));
        assert!(errors.iter().any(|e| e.contains("emc_id")));
        assert!(errors.iter().any(|e| e.contains("governing_equation")));
        assert!(errors.iter().any(|e| e.contains("analytical_derivation")));
        assert!(errors.iter().any(|e| e.contains("domain_of_validity")));
    }
}
