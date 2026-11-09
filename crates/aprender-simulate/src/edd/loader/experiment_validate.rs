//! Experiment schema validation (split out of loader.rs for the pre-commit complexity
//! limit, #6025).

use super::ExperimentYaml;

impl ExperimentYaml {
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

        self.check_hypothesis(&mut errors);
        self.check_falsification(&mut errors);
        self.check_duration(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn check_hypothesis(&self, errors: &mut Vec<String>) {
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
    }

    fn check_falsification(&self, errors: &mut Vec<String>) {
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
    }

    fn check_duration(&self, errors: &mut Vec<String>) {
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
    }
}
