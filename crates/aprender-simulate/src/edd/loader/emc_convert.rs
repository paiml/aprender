//! EMC to `EquationModelCard` conversion (split out of loader.rs for the pre-commit
//! complexity limit, #6025).

use super::EmcYaml;
use crate::edd::equation::{Citation, EquationClass};
use crate::edd::model_card::{EmcBuilder, EquationModelCard, VerificationTest};

impl EmcYaml {
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

        builder = builder.class(self.equation_class());
        builder = self.add_citation(builder);

        // Add variables
        for var in &self.governing_equation.variables {
            builder = builder.add_variable(&var.symbol, &var.description, &var.units);
        }

        builder = self.add_verification_tests(builder);

        // If no verification tests, add a placeholder (will fail validation correctly)
        // This ensures the builder pattern works but validation catches missing tests

        builder.build()
    }

    /// Equation class based on the governing equation's type.
    fn equation_class(&self) -> EquationClass {
        match self
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
        }
    }

    fn add_citation(&self, builder: EmcBuilder) -> EmcBuilder {
        let Some(ref derivation) = self.analytical_derivation else {
            return builder;
        };
        let Some(ref cite) = derivation.primary_citation else {
            return builder;
        };
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
        builder.citation(citation)
    }

    fn add_verification_tests(&self, mut builder: EmcBuilder) -> EmcBuilder {
        let Some(ref tests) = self.verification_tests else {
            return builder;
        };
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
        builder
    }
}
