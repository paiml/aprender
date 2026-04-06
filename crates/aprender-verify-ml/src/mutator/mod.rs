//! AST mutation operators
//!
//! This module provides mutation operators for systematic AST mutations.
//! Based on Jia & Harman (2011) mutation operator catalog.
//!
//! # Mutation Operators
//!
//! | Operator | Status | Description | Example |
//! |----------|--------|-------------|---------|
//! | AOR | ✓ Implemented | Arithmetic operator replacement | `a + b` → `a - b` |
//! | ROR | ✓ Implemented | Relational operator replacement | `a < b` → `a <= b` |
//! | BSR | ✓ Implemented | Boundary substitution | `0` → `-1`, `""` → `" "` |
//! | LOR | Planned | Logical operator replacement | `a and b` → `a or b` |
//! | UOI | Planned | Unary operator insertion | `x` → `-x` |
//! | ABS | Planned | Absolute value insertion | `x` → `abs(x)` |
//! | SDL | Planned | Statement deletion | Delete random statement |
//! | SVR | Planned | Scalar variable replacement | `x` → `y` (same type) |
//!
//! ## Implementation Status
//!
//! Current implementation provides 3/8 operators (AOR, ROR, BSR) using string-based
//! mutations. Full AST-based mutation support is planned for future releases.
//! See GitHub issues for operator implementation tracking.
//!
//! # AST-based Mutations
//!
//! The `AstMutator` provides proper AST-level mutations using the `PythonNode`
//! representation. Unlike string-based mutations, these guarantee syntactic validity.
//!
//! ```rust,ignore
//! use verificar::mutator::AstMutator;
//! use verificar::generator::PythonNode;
//!
//! let ast = PythonNode::BinOp { ... };
//! let mutator = AstMutator::new();
//! let mutations = mutator.mutate(&ast);
//! ```

mod ast_mutator;
pub mod cwe_bash;
mod operators;

pub use ast_mutator::{AstMutation, AstMutator};
pub use cwe_bash::{generate_cwe_mutations, CweMutation};
pub use operators::MutationOperator;

use crate::{Error, Result};

/// Mutated code with metadata
#[derive(Debug, Clone)]
pub struct MutatedCode {
    /// Original code before mutation
    pub original: String,
    /// Mutated code
    pub mutated: String,
    /// Operator applied
    pub operator: MutationOperator,
    /// Location of mutation (line, column)
    pub location: (usize, usize),
    /// Description of the mutation
    pub description: String,
}

/// AST mutator for systematic code mutation
#[derive(Debug, Default)]
pub struct Mutator {
    /// Enabled mutation operators
    enabled_operators: Vec<MutationOperator>,
}

impl Mutator {
    /// Create a new mutator with all operators enabled
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled_operators: MutationOperator::all(),
        }
    }

    /// Create a mutator with specific operators enabled
    #[must_use]
    pub fn with_operators(operators: Vec<MutationOperator>) -> Self {
        Self {
            enabled_operators: operators,
        }
    }

    /// Generate all possible mutations for the given code
    ///
    /// # Errors
    ///
    /// Returns an error if mutation fails
    pub fn mutate(&self, code: &str) -> Result<Vec<MutatedCode>> {
        if code.is_empty() {
            return Err(Error::Mutation("cannot mutate empty code".to_string()));
        }

        contract_pre_mutation_soundness!(code);
        let mut mutations = Vec::new();

        for operator in &self.enabled_operators {
            let operator_mutations = self.apply_operator(code, operator)?;
            mutations.extend(operator_mutations);
        }

        Ok(mutations)
    }

    /// Apply a single mutation operator to the code
    ///
    /// Current implementation uses string-based mutations for AOR, ROR, BSR.
    /// Other operators return empty vectors (no mutations).
    fn apply_operator(&self, code: &str, operator: &MutationOperator) -> Result<Vec<MutatedCode>> {
        let mutations = match operator {
            MutationOperator::Aor => self.apply_aor(code),
            MutationOperator::Ror => self.apply_ror(code),
            MutationOperator::Lor => self.apply_lor(code),
            MutationOperator::Uoi => self.apply_uoi(code),
            MutationOperator::Abs => self.apply_abs(code),
            MutationOperator::Sdl => self.apply_sdl(code),
            MutationOperator::Svr => self.apply_svr(code),
            MutationOperator::Bsr => self.apply_bsr(code),
        };

        Ok(mutations)
    }

    fn apply_aor(&self, code: &str) -> Vec<MutatedCode> {
        let mut mutations = Vec::new();

        // Simple string-based replacement for now
        if code.contains('+') {
            mutations.push(MutatedCode {
                original: code.to_string(),
                mutated: code.replace('+', "-"),
                operator: MutationOperator::Aor,
                location: (1, code.find('+').unwrap_or(0)),
                description: "Replace + with -".to_string(),
            });
        }

        mutations
    }

    fn apply_ror(&self, code: &str) -> Vec<MutatedCode> {
        let mut mutations = Vec::new();

        if code.contains('<') && !code.contains("<=") {
            mutations.push(MutatedCode {
                original: code.to_string(),
                mutated: code.replace('<', "<="),
                operator: MutationOperator::Ror,
                location: (1, code.find('<').unwrap_or(0)),
                description: "Replace < with <=".to_string(),
            });
        }

        mutations
    }

    fn apply_lor(&self, _code: &str) -> Vec<MutatedCode> {
        // LOR (Logical operator replacement) not yet implemented
        Vec::new()
    }

    fn apply_uoi(&self, _code: &str) -> Vec<MutatedCode> {
        // UOI (Unary operator insertion) not yet implemented
        Vec::new()
    }

    fn apply_abs(&self, _code: &str) -> Vec<MutatedCode> {
        // ABS (Absolute value insertion) not yet implemented
        Vec::new()
    }

    fn apply_sdl(&self, _code: &str) -> Vec<MutatedCode> {
        // SDL (Statement deletion) not yet implemented
        Vec::new()
    }

    fn apply_svr(&self, _code: &str) -> Vec<MutatedCode> {
        // SVR (Scalar variable replacement) not yet implemented
        Vec::new()
    }

    fn apply_bsr(&self, code: &str) -> Vec<MutatedCode> {
        let mut mutations = Vec::new();

        // Replace 0 with -1 (boundary substitution)
        if code.contains(" 0") || code.contains("=0") {
            mutations.push(MutatedCode {
                original: code.to_string(),
                mutated: code.replace(" 0", " -1").replace("=0", "=-1"),
                operator: MutationOperator::Bsr,
                location: (1, 0),
                description: "Replace 0 with -1".to_string(),
            });
        }

        mutations
    }

    /// Get the enabled operators
    #[must_use]
    pub fn enabled_operators(&self) -> &[MutationOperator] {
        &self.enabled_operators
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutator_new() {
        let mutator = Mutator::new();
        assert_eq!(mutator.enabled_operators().len(), 8);
    }

    #[test]
    fn test_mutator_default() {
        let mutator = Mutator::default();
        assert!(mutator.enabled_operators().is_empty());
    }

    #[test]
    fn test_mutator_with_operators() {
        let ops = vec![MutationOperator::Aor, MutationOperator::Ror];
        let mutator = Mutator::with_operators(ops);
        assert_eq!(mutator.enabled_operators().len(), 2);
    }

    #[test]
    fn test_mutator_aor() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Aor]);
        let mutations = mutator
            .mutate("x = a + b")
            .expect("mutation should succeed");
        assert!(!mutations.is_empty());
        assert!(mutations[0].mutated.contains('-'));
    }

    #[test]
    fn test_mutator_aor_no_operator() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Aor]);
        let mutations = mutator.mutate("x = 5").expect("mutation should succeed");
        assert!(mutations.is_empty());
    }

    #[test]
    fn test_mutator_ror() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Ror]);
        let mutations = mutator
            .mutate("if x < 5:")
            .expect("mutation should succeed");
        assert!(!mutations.is_empty());
        assert!(mutations[0].mutated.contains("<="));
    }

    #[test]
    fn test_mutator_ror_already_lte() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Ror]);
        let mutations = mutator
            .mutate("if x <= 5:")
            .expect("mutation should succeed");
        assert!(mutations.is_empty()); // Should not mutate <= to <==
    }

    #[test]
    fn test_mutator_ror_no_operator() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Ror]);
        let mutations = mutator.mutate("x = 5").expect("mutation should succeed");
        assert!(mutations.is_empty());
    }

    #[test]
    fn test_mutator_lor() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Lor]);
        let mutations = mutator.mutate("x and y").expect("mutation should succeed");
        assert!(mutations.is_empty()); // LOR not yet implemented
    }

    #[test]
    fn test_mutator_uoi() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Uoi]);
        let mutations = mutator.mutate("x = 5").expect("mutation should succeed");
        assert!(mutations.is_empty()); // UOI not yet implemented
    }

    #[test]
    fn test_mutator_abs() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Abs]);
        let mutations = mutator.mutate("x = -5").expect("mutation should succeed");
        assert!(mutations.is_empty()); // ABS not yet implemented
    }

    #[test]
    fn test_mutator_sdl() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Sdl]);
        let mutations = mutator.mutate("x = 5").expect("mutation should succeed");
        assert!(mutations.is_empty()); // SDL not yet implemented
    }

    #[test]
    fn test_mutator_svr() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Svr]);
        let mutations = mutator.mutate("x = y").expect("mutation should succeed");
        assert!(mutations.is_empty()); // SVR not yet implemented
    }

    #[test]
    fn test_mutator_bsr_space_zero() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate("x = 0").expect("mutation should succeed");
        assert!(!mutations.is_empty());
        assert!(mutations[0].mutated.contains("-1"));
    }

    #[test]
    fn test_mutator_bsr_equals_zero() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate("x=0").expect("mutation should succeed");
        assert!(!mutations.is_empty());
        assert!(mutations[0].mutated.contains("=-1"));
    }

    #[test]
    fn test_mutator_bsr_no_zero() {
        let mutator = Mutator::with_operators(vec![MutationOperator::Bsr]);
        let mutations = mutator.mutate("x = 5").expect("mutation should succeed");
        assert!(mutations.is_empty());
    }

    #[test]
    fn test_mutator_empty_code() {
        let mutator = Mutator::new();
        let result = mutator.mutate("");
        assert!(result.is_err());
    }

    #[test]
    fn test_mutator_all_operators() {
        let mutator = Mutator::new();
        // Code that triggers AOR, ROR, and BSR
        let mutations = mutator
            .mutate("x = a + b if y < 0")
            .expect("mutation should succeed");
        assert!(!mutations.is_empty());
        // Should have AOR (+ to -), ROR (< to <=), and BSR (0 to -1)
        assert!(mutations
            .iter()
            .any(|m| m.operator == MutationOperator::Aor));
        assert!(mutations
            .iter()
            .any(|m| m.operator == MutationOperator::Ror));
        // BSR checks for " 0" pattern
        let mutations2 = mutator.mutate("x = 0").expect("mutation should succeed");
        assert!(mutations2
            .iter()
            .any(|m| m.operator == MutationOperator::Bsr));
    }

    #[test]
    fn test_mutated_code_debug() {
        let mc = MutatedCode {
            original: "x + 1".to_string(),
            mutated: "x - 1".to_string(),
            operator: MutationOperator::Aor,
            location: (1, 2),
            description: "Replace + with -".to_string(),
        };
        let debug = format!("{:?}", mc);
        assert!(debug.contains("MutatedCode"));
        assert!(debug.contains("Aor"));
    }

    #[test]
    fn test_mutated_code_clone() {
        let mc = MutatedCode {
            original: "x + 1".to_string(),
            mutated: "x - 1".to_string(),
            operator: MutationOperator::Aor,
            location: (1, 2),
            description: "Replace + with -".to_string(),
        };
        let cloned = mc.clone();
        assert_eq!(cloned.original, mc.original);
        assert_eq!(cloned.mutated, mc.mutated);
    }
}
