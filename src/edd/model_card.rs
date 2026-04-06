//! Equation Model Card (EMC) - Mandatory documentation for EDD simulations.
//!
//! The EMC bridges mathematics and code, ensuring every simulation is grounded
//! in peer-reviewed theory. No simulation can run without a complete EMC.
//!
//! # EMC Schema (9 Required Sections)
//!
//! 1. **Identity**: Name, UUID, version
//! 2. **Governing Equation**: LaTeX, analytical derivation
//! 3. **Variables**: All parameters with units and constraints
//! 4. **Verification Tests**: Analytical solutions for TDD
//! 5. **Domain Constraints**: Valid operating ranges
//! 6. **Falsification Criteria**: How to disprove the model
//! 7. **References**: Peer-reviewed citations
//! 8. **Implementation Notes**: Numerical considerations
//! 9. **Lineage**: Parent equations, derivations

use super::equation::{Citation, EquationClass, EquationVariable, GoverningEquation};
use std::collections::HashMap;

// Re-export Citation for convenience
pub use super::equation::Citation as EmcCitation;

/// Domain constraint specifying valid operating ranges.
#[derive(Debug, Clone)]
pub struct DomainConstraint {
    /// Name of the constraint
    pub name: String,
    /// Mathematical expression (e.g., "0 < ρ < 1")
    pub expression: String,
    /// Rationale for this constraint
    pub rationale: String,
    /// What happens when violated
    pub violation_behavior: ViolationBehavior,
}

/// Behavior when a domain constraint is violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationBehavior {
    /// Log warning but continue
    Warn,
    /// Halt simulation (Jidoka)
    Halt,
    /// Clamp value to valid range
    Clamp,
    /// Return special value (NaN, Infinity)
    Special,
}

/// A verification test case with known analytical solution.
#[derive(Debug, Clone)]
pub struct VerificationTest {
    /// Description of the test case
    pub description: String,
    /// Input values
    pub inputs: HashMap<String, f64>,
    /// Expected output value
    pub expected: f64,
    /// Tolerance for comparison
    pub tolerance: f64,
    /// Source of the analytical solution
    pub source: Option<String>,
}

impl VerificationTest {
    /// Create a new verification test.
    #[must_use]
    pub fn new(description: &str, expected: f64, tolerance: f64) -> Self {
        Self {
            description: description.to_string(),
            inputs: HashMap::new(),
            expected,
            tolerance,
            source: None,
        }
    }

    /// Add an input value.
    #[must_use]
    pub fn with_input(mut self, name: &str, value: f64) -> Self {
        self.inputs.insert(name.to_string(), value);
        self
    }

    /// Add the source reference.
    #[must_use]
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = Some(source.to_string());
        self
    }
}

/// Falsification criterion defining how to disprove the model.
#[derive(Debug, Clone)]
pub struct FalsificationCriterion {
    /// Name of the criterion
    pub name: String,
    /// Description of what would disprove the model
    pub description: String,
    /// Statistical test to use
    pub test_method: String,
    /// Significance level (e.g., 0.05)
    pub alpha: f64,
}

/// Implementation notes for numerical considerations.
#[derive(Debug, Clone)]
pub struct ImplementationNote {
    /// Topic (e.g., "Numerical Stability")
    pub topic: String,
    /// The note content
    pub content: String,
    /// Severity/importance
    pub importance: NoteImportance,
}

/// Importance level for implementation notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteImportance {
    /// Informational only
    Info,
    /// Should be considered
    Important,
    /// Must be addressed
    Critical,
}

/// Equation Model Card - Complete documentation for a governing equation.
///
/// An EMC must be attached to any simulation using the equation. It provides:
/// - Traceability to peer-reviewed literature
/// - Verification test cases from analytical solutions
/// - Domain constraints for valid operation
/// - Falsification criteria for scientific validity
#[derive(Debug, Clone)]
pub struct EquationModelCard {
    /// Unique name for this EMC
    pub name: String,
    /// Version string
    pub version: String,
    /// The governing equation in LaTeX
    pub equation: String,
    /// Equation classification
    pub class: EquationClass,
    /// Primary citation
    pub citation: Citation,
    /// Additional references
    pub references: Vec<Citation>,
    /// Variables in the equation
    pub variables: Vec<EquationVariable>,
    /// Verification tests
    pub verification_tests: Vec<VerificationTest>,
    /// Domain constraints
    pub domain_constraints: Vec<DomainConstraint>,
    /// Falsification criteria
    pub falsification_criteria: Vec<FalsificationCriterion>,
    /// Implementation notes
    pub implementation_notes: Vec<ImplementationNote>,
    /// Description/abstract
    pub description: String,
    /// Parent EMCs this derives from
    pub lineage: Vec<String>,
}

/// Builder for `EquationModelCard`.
///
/// Enforces that all required fields are provided before building.
#[derive(Debug, Default)]
pub struct EmcBuilder {
    name: Option<String>,
    version: String,
    equation: Option<String>,
    class: Option<EquationClass>,
    citation: Option<Citation>,
    references: Vec<Citation>,
    variables: Vec<EquationVariable>,
    verification_tests: Vec<VerificationTest>,
    domain_constraints: Vec<DomainConstraint>,
    falsification_criteria: Vec<FalsificationCriterion>,
    implementation_notes: Vec<ImplementationNote>,
    description: String,
    lineage: Vec<String>,
}

impl EmcBuilder {
    /// Create a new EMC builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: "1.0.0".to_string(),
            ..Default::default()
        }
    }

    /// Set the EMC name.
    #[must_use]
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set the version.
    #[must_use]
    pub fn version(mut self, version: &str) -> Self {
        self.version = version.to_string();
        self
    }

    /// Set the governing equation (LaTeX).
    #[must_use]
    pub fn equation(mut self, equation: &str) -> Self {
        self.equation = Some(equation.to_string());
        self
    }

    /// Set the equation class.
    #[must_use]
    pub fn class(mut self, class: EquationClass) -> Self {
        self.class = Some(class);
        self
    }

    /// Set the primary citation.
    #[must_use]
    pub fn citation(mut self, citation: Citation) -> Self {
        self.citation = Some(citation);
        self
    }

    /// Add an additional reference.
    #[must_use]
    pub fn add_reference(mut self, reference: Citation) -> Self {
        self.references.push(reference);
        self
    }

    /// Add a variable.
    #[must_use]
    pub fn add_variable(mut self, symbol: &str, name: &str, units: &str) -> Self {
        self.variables
            .push(EquationVariable::new(symbol, name, units));
        self
    }

    /// Add a variable with full specification.
    #[must_use]
    pub fn add_variable_full(mut self, variable: EquationVariable) -> Self {
        self.variables.push(variable);
        self
    }

    /// Add a verification test.
    #[must_use]
    pub fn add_verification_test(
        mut self,
        description: &str,
        expected: f64,
        tolerance: f64,
    ) -> Self {
        self.verification_tests
            .push(VerificationTest::new(description, expected, tolerance));
        self
    }

    /// Add a verification test with full specification.
    #[must_use]
    pub fn add_verification_test_full(mut self, test: VerificationTest) -> Self {
        self.verification_tests.push(test);
        self
    }

    /// Add a domain constraint.
    #[must_use]
    pub fn add_domain_constraint(mut self, constraint: DomainConstraint) -> Self {
        self.domain_constraints.push(constraint);
        self
    }

    /// Add a falsification criterion.
    #[must_use]
    pub fn add_falsification_criterion(mut self, criterion: FalsificationCriterion) -> Self {
        self.falsification_criteria.push(criterion);
        self
    }

    /// Add an implementation note.
    #[must_use]
    pub fn add_implementation_note(mut self, note: ImplementationNote) -> Self {
        self.implementation_notes.push(note);
        self
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Add a parent EMC to the lineage.
    #[must_use]
    pub fn add_lineage(mut self, parent: &str) -> Self {
        self.lineage.push(parent.to_string());
        self
    }

    /// Build the EMC, returning an error if required fields are missing.
    ///
    /// # Required Fields
    /// - name
    /// - equation
    /// - citation
    /// - At least one verification test
    ///
    /// # Errors
    /// Returns `Err` with a description of missing fields.
    pub fn build(self) -> Result<EquationModelCard, String> {
        let name = self.name.ok_or("EMC requires a name")?;
        let equation = self.equation.ok_or("EMC requires a governing equation")?;
        let citation = self
            .citation
            .ok_or("EMC requires a citation (analytical derivation)")?;

        if self.verification_tests.is_empty() {
            return Err("EMC requires at least one verification test".to_string());
        }

        Ok(EquationModelCard {
            name,
            version: self.version,
            equation,
            class: self.class.unwrap_or(EquationClass::Conservation),
            citation,
            references: self.references,
            variables: self.variables,
            verification_tests: self.verification_tests,
            domain_constraints: self.domain_constraints,
            falsification_criteria: self.falsification_criteria,
            implementation_notes: self.implementation_notes,
            description: self.description,
            lineage: self.lineage,
        })
    }
}

impl EquationModelCard {
    /// Create a new EMC builder.
    #[must_use]
    pub fn builder() -> EmcBuilder {
        EmcBuilder::new()
    }

    /// Create an EMC from a `GoverningEquation` implementation.
    #[must_use]
    pub fn from_equation<E: GoverningEquation>(equation: &E) -> EmcBuilder {
        EmcBuilder::new()
            .name(equation.name())
            .equation(equation.latex())
            .class(equation.class())
            .citation(equation.citation())
            .description(equation.description())
    }

    /// Run all verification tests.
    ///
    /// Returns a list of (`test_name`, passed, message) tuples.
    pub fn run_verification_tests<F>(&self, evaluator: F) -> Vec<(String, bool, String)>
    where
        F: Fn(&HashMap<String, f64>) -> f64,
    {
        self.verification_tests
            .iter()
            .map(|test| {
                let actual = evaluator(&test.inputs);
                let passed = (actual - test.expected).abs() <= test.tolerance;
                let message = if passed {
                    format!(
                        "PASS: {} (expected={}, actual={})",
                        test.description, test.expected, actual
                    )
                } else {
                    format!(
                        "FAIL: {} (expected={}, actual={}, tolerance={})",
                        test.description, test.expected, actual, test.tolerance
                    )
                };
                (test.description.clone(), passed, message)
            })
            .collect()
    }

    /// Check if all verification tests pass.
    ///
    /// # Errors
    /// Returns a list of failure messages if any verification tests fail.
    pub fn verify<F>(&self, evaluator: F) -> Result<(), Vec<String>>
    where
        F: Fn(&HashMap<String, f64>) -> f64,
    {
        let results = self.run_verification_tests(evaluator);
        let failures: Vec<String> = results
            .into_iter()
            .filter(|(_, passed, _)| !passed)
            .map(|(_, _, msg)| msg)
            .collect();

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emc_builder_requires_name() {
        let result = EmcBuilder::new()
            .equation("L = \\lambda W")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test("test", 1.0, 0.1)
            .build();

        assert!(result.is_err());
        assert!(result.err().map(|e| e.contains("name")).unwrap_or(false));
    }

    #[test]
    fn test_emc_builder_requires_equation() {
        let result = EmcBuilder::new()
            .name("Test EMC")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test("test", 1.0, 0.1)
            .build();

        assert!(result.is_err());
        assert!(result
            .err()
            .map(|e| e.contains("equation"))
            .unwrap_or(false));
    }

    #[test]
    fn test_emc_builder_requires_citation() {
        let result = EmcBuilder::new()
            .name("Test EMC")
            .equation("L = \\lambda W")
            .add_verification_test("test", 1.0, 0.1)
            .build();

        assert!(result.is_err());
        assert!(result
            .err()
            .map(|e| e.contains("citation"))
            .unwrap_or(false));
    }

    #[test]
    fn test_emc_builder_requires_verification_tests() {
        let result = EmcBuilder::new()
            .name("Test EMC")
            .equation("L = \\lambda W")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .build();

        assert!(result.is_err());
        assert!(result
            .err()
            .map(|e| e.contains("verification"))
            .unwrap_or(false));
    }

    #[test]
    fn test_emc_builder_complete_builds() {
        let result = EmcBuilder::new()
            .name("Little's Law")
            .equation("L = \\lambda W")
            .citation(Citation::new(
                &["Little, J.D.C."],
                "Operations Research",
                1961,
            ))
            .add_variable("L", "wip", "items")
            .add_variable("lambda", "arrival_rate", "items/time")
            .add_variable("W", "cycle_time", "time")
            .add_verification_test("L = λW for λ=5, W=2 => L=10", 10.0, 1e-10)
            .build();

        assert!(result.is_ok());
        let emc = result.ok();
        assert!(emc.is_some());
        let emc = emc.unwrap();
        assert_eq!(emc.name, "Little's Law");
        assert_eq!(emc.variables.len(), 3);
        assert_eq!(emc.verification_tests.len(), 1);
    }

    #[test]
    fn test_emc_run_verification_tests() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test_full(
                VerificationTest::new("identity", 5.0, 0.1).with_input("x", 5.0),
            )
            .build()
            .ok();

        assert!(emc.is_some());
        let emc = emc.unwrap();
        let results = emc.run_verification_tests(|inputs| inputs.get("x").copied().unwrap_or(0.0));

        assert_eq!(results.len(), 1);
        assert!(results[0].1); // passed
    }

    #[test]
    fn test_domain_constraint() {
        let constraint = DomainConstraint {
            name: "Utilization bound".to_string(),
            expression: "0 < ρ < 1".to_string(),
            rationale: "Utilization must be less than 100% for stable queue".to_string(),
            violation_behavior: ViolationBehavior::Halt,
        };

        assert_eq!(constraint.violation_behavior, ViolationBehavior::Halt);
    }

    #[test]
    fn test_falsification_criterion() {
        let criterion = FalsificationCriterion {
            name: "Little's Law".to_string(),
            description: "L should equal λW within statistical tolerance".to_string(),
            test_method: "Two-sample t-test".to_string(),
            alpha: 0.05,
        };

        assert!((criterion.alpha - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_emc_verify_pass() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test_full(
                VerificationTest::new("identity", 5.0, 0.1).with_input("x", 5.0),
            )
            .build()
            .ok()
            .unwrap();

        let result = emc.verify(|inputs| inputs.get("x").copied().unwrap_or(0.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_emc_verify_fail() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = 2*x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test_full(
                VerificationTest::new("double", 10.0, 0.01).with_input("x", 5.0),
            )
            .build()
            .ok()
            .unwrap();

        // Return wrong value to trigger failure
        let result = emc.verify(|_inputs| 100.0);
        assert!(result.is_err());
        let failures = result.err().unwrap();
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("FAIL"));
    }

    #[test]
    fn test_emc_run_verification_tests_fail() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test_full(
                VerificationTest::new("identity", 5.0, 0.001).with_input("x", 5.0),
            )
            .build()
            .ok()
            .unwrap();

        let results = emc.run_verification_tests(|_| 999.0); // Return wrong value
        assert_eq!(results.len(), 1);
        assert!(!results[0].1); // failed
        assert!(results[0].2.contains("FAIL"));
    }

    #[test]
    fn test_emc_builder_version() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .version("2.0.0")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.version, "2.0.0");
    }

    #[test]
    fn test_emc_builder_class() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .class(EquationClass::Optimization)
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.class, EquationClass::Optimization);
    }

    #[test]
    fn test_emc_builder_add_reference() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_reference(Citation::new(&["Other"], "Other Journal", 2020))
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.references.len(), 1);
    }

    #[test]
    fn test_emc_builder_add_variable_full() {
        let var =
            EquationVariable::new("x", "input", "units").with_description("Test input variable");

        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_variable_full(var)
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.variables.len(), 1);
        assert_eq!(emc.variables[0].description, "Test input variable");
    }

    #[test]
    fn test_emc_builder_add_domain_constraint() {
        let constraint = DomainConstraint {
            name: "positivity".to_string(),
            expression: "x > 0".to_string(),
            rationale: "Must be positive".to_string(),
            violation_behavior: ViolationBehavior::Warn,
        };
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_domain_constraint(constraint)
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.domain_constraints.len(), 1);
        assert_eq!(
            emc.domain_constraints[0].violation_behavior,
            ViolationBehavior::Warn
        );
    }

    #[test]
    fn test_emc_builder_add_falsification_criterion() {
        let criterion = FalsificationCriterion {
            name: "test".to_string(),
            description: "description".to_string(),
            test_method: "method".to_string(),
            alpha: 0.01,
        };
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_falsification_criterion(criterion)
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.falsification_criteria.len(), 1);
        assert!((emc.falsification_criteria[0].alpha - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_emc_builder_add_implementation_note() {
        let note = ImplementationNote {
            topic: "Test Topic".to_string(),
            content: "Important Note".to_string(),
            importance: NoteImportance::Info,
        };
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_implementation_note(note)
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.implementation_notes.len(), 1);
        assert_eq!(emc.implementation_notes[0].content, "Important Note");
    }

    #[test]
    fn test_emc_builder_description() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .description("A test description")
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.description, "A test description");
    }

    #[test]
    fn test_emc_builder_add_lineage() {
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("y = x")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_lineage("parent-emc")
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        assert_eq!(emc.lineage.len(), 1);
        assert_eq!(emc.lineage[0], "parent-emc");
    }

    #[test]
    fn test_verification_test_builder() {
        let test = VerificationTest::new("test desc", 10.0, 0.01)
            .with_input("x", 5.0)
            .with_input("y", 2.0);

        assert_eq!(test.description, "test desc");
        assert!((test.expected - 10.0).abs() < f64::EPSILON);
        assert!((test.tolerance - 0.01).abs() < f64::EPSILON);
        assert_eq!(test.inputs.len(), 2);
    }

    #[test]
    fn test_implementation_note() {
        let note = ImplementationNote {
            topic: "Numerical".to_string(),
            content: "Use Welford's algorithm for numerical stability".to_string(),
            importance: NoteImportance::Important,
        };
        assert!(note.content.contains("Welford"));
        assert_eq!(note.importance, NoteImportance::Important);
    }

    #[test]
    fn test_note_importance_variants() {
        assert_ne!(NoteImportance::Info, NoteImportance::Important);
        assert_ne!(NoteImportance::Important, NoteImportance::Critical);
        assert_ne!(NoteImportance::Critical, NoteImportance::Info);
    }

    #[test]
    fn test_violation_behavior_variants() {
        assert_ne!(ViolationBehavior::Halt, ViolationBehavior::Warn);
        assert_ne!(ViolationBehavior::Warn, ViolationBehavior::Clamp);
        assert_ne!(ViolationBehavior::Clamp, ViolationBehavior::Halt);
    }

    #[test]
    fn test_equation_model_card_builder() {
        let builder = EquationModelCard::builder();
        assert!(builder.build().is_err());
    }
}
