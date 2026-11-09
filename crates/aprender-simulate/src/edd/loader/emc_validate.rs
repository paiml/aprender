//! EMC schema validation (split out of loader.rs for the pre-commit complexity limit, #6025).

use super::EmcYaml;

impl EmcYaml {
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

        self.check_identity(&mut errors);
        self.check_citation(&mut errors);
        self.check_tests_and_criteria(&mut errors);

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

    fn check_identity(&self, errors: &mut Vec<String>) {
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
    }

    fn check_citation(&self, errors: &mut Vec<String>) {
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
    }

    fn check_tests_and_criteria(&self, errors: &mut Vec<String>) {
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
    }
}
