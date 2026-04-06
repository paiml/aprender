//! Coq theorem stub generation from YAML contracts.
//!
//! Generates `.v` files with `Require Import` statements, definitions
//! from equations, and theorem stubs from proof obligations.

use std::fmt::Write;

use crate::schema::Contract;

/// Generate a Coq `.v` file from a contract.
///
/// If the contract has a `coq_spec` section, uses its imports and
/// definitions. Otherwise generates stubs from equations and obligations.
pub fn generate_coq_spec(contract: &Contract, stem: &str) -> String {
    let mut out = String::with_capacity(2048);

    // Header
    let _ = writeln!(
        out,
        "(* Generated from {stem} v{} *)",
        contract.metadata.version
    );
    let _ = writeln!(out, "(* {} *)", contract.metadata.description);
    let _ = writeln!(out, "(* Regenerate with: pv coq {stem}.yaml *)\n");

    // Imports
    if let Some(ref spec) = contract.coq_spec {
        for import in &spec.imports {
            let _ = writeln!(out, "{import}");
        }
        let _ = writeln!(out);

        // Explicit definitions from coq_spec
        for def in &spec.definitions {
            let _ = writeln!(out, "(** Definition: {} *)", def.name);
            let _ = writeln!(out, "{}\n", def.statement);
        }

        // Obligation links
        for ob in &spec.obligations {
            let _ = writeln!(
                out,
                "(** Obligation: {} → Coq lemma: {} [{}] *)",
                ob.links_to, ob.coq_lemma, ob.status
            );
        }
        let _ = writeln!(out);
    } else {
        // Default imports
        let _ = writeln!(out, "Require Import Reals.");
        let _ = writeln!(out, "Require Import List.");
        let _ = writeln!(out, "Open Scope R_scope.\n");
    }

    // Definitions from equations
    for (name, eq) in &contract.equations {
        let coq_name = name.replace('-', "_");
        let _ = writeln!(out, "(** Equation: {name} *)");
        let _ = writeln!(
            out,
            "(** Formula: {} *)",
            eq.formula.lines().next().unwrap_or("")
        );
        let _ = writeln!(out, "Definition {coq_name} := (* TODO: formalize *).");
        let _ = writeln!(out);
    }

    // Theorem stubs from proof obligations
    for (i, ob) in contract.proof_obligations.iter().enumerate() {
        let theorem_name = ob
            .property
            .to_lowercase()
            .replace([' ', '-'], "_")
            .replace(['(', ')', ',', '.', '|', '<', '>', '=', '+', '*', '/'], "");

        let _ = writeln!(
            out,
            "(** Obligation {}: {} [{}] *)",
            i + 1,
            ob.property,
            ob.obligation_type
        );

        if let Some(ref formal) = ob.formal {
            let _ = writeln!(out, "(** Formal: {formal} *)");
        }

        // Check if this has a Lean proof we can reference
        if let Some(ref lean) = ob.lean {
            let _ = writeln!(
                out,
                "(** Lean equivalent: {}.{} ({}) *)",
                lean.module.as_deref().unwrap_or("?"),
                lean.theorem,
                lean.status
            );
        }

        // Check coq_spec for existing link
        let coq_status = contract
            .coq_spec
            .as_ref()
            .and_then(|spec| {
                spec.obligations
                    .iter()
                    .find(|o| o.links_to == ob.property)
                    .map(|o| o.status.as_str())
            })
            .unwrap_or("stub");

        if coq_status == "proved" {
            let _ = writeln!(out, "Theorem {theorem_name} : (* proved *).");
        } else {
            let _ = writeln!(out, "Theorem {theorem_name} :");
            let _ = writeln!(out, "  (* TODO: formalize from obligation *)");
            let _ = writeln!(out, "  True.");
            let _ = writeln!(out, "Proof.");
            let _ = writeln!(out, "  admit.");
            let _ = writeln!(out, "Admitted.\n");
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn generates_coq_for_minimal_contract() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
    formal: "|f(x)| < inf"
falsification_tests: []
"#,
        )
        .unwrap();

        let output = generate_coq_spec(&contract, "test-v1");
        assert!(output.contains("Generated from test-v1"));
        assert!(output.contains("Require Import Reals."));
        assert!(output.contains("Definition f"));
        assert!(output.contains("Theorem output_finite"));
        assert!(output.contains("admit."));
    }

    #[test]
    fn uses_coq_spec_imports() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Coq spec"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
coq_spec:
  module: "TestSpec"
  imports:
    - "Require Import ZArith."
    - "Require Import Lia."
  definitions:
    - name: "my_def"
      statement: "Definition my_def := 42."
  obligations:
    - links_to: "output finite"
      coq_lemma: "test_lemma"
      status: "admitted"
falsification_tests: []
"#,
        )
        .unwrap();

        let output = generate_coq_spec(&contract, "coq-test-v1");
        assert!(output.contains("Require Import ZArith."));
        assert!(output.contains("Require Import Lia."));
        assert!(output.contains("Definition my_def := 42."));
        assert!(output.contains("test_lemma"));
    }

    #[test]
    fn references_lean_proof() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Lean ref"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "test prop"
    lean:
      theorem: test_thm
      module: Test.Module
      status: proved
falsification_tests: []
"#,
        )
        .unwrap();

        let output = generate_coq_spec(&contract, "lean-ref-v1");
        assert!(output.contains("Lean equivalent: Test.Module.test_thm (proved)"));
    }
}
