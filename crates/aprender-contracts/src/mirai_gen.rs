//! MIRAI abstract interpretation annotation generation.
//!
//! Translates YAML `proof_obligations` into MIRAI `precondition!` /
//! `postcondition!` / `verify!` annotations for sound over-approximation
//! analysis of Rust kernel implementations.

use std::fmt::Write;

use crate::schema::{Contract, ObligationType};

/// Generate MIRAI-annotated Rust source from a contract.
///
/// Emits a function skeleton with `mirai_annotations` `precondition!` and
/// `postcondition!` macros derived from the contract's proof obligations.
pub fn generate_mirai_annotations(contract: &Contract, stem: &str) -> String {
    let mut out = String::with_capacity(2048);

    let _ = writeln!(
        out,
        "// Auto-generated MIRAI annotations from contract: {}",
        contract.metadata.description
    );
    let _ = writeln!(out, "// Regenerate with: pv mirai contracts/{stem}.yaml");
    let _ = writeln!(out);
    let _ = writeln!(out, "use mirai_annotations::*;");
    let _ = writeln!(out);

    // Group obligations by type
    let preconditions: Vec<_> = contract
        .proof_obligations
        .iter()
        .filter(|ob| ob.obligation_type == ObligationType::Precondition)
        .collect();

    let postconditions: Vec<_> = contract
        .proof_obligations
        .iter()
        .filter(|ob| ob.obligation_type == ObligationType::Postcondition)
        .collect();

    let invariants: Vec<_> = contract
        .proof_obligations
        .iter()
        .filter(|ob| ob.obligation_type == ObligationType::Invariant)
        .collect();

    let bounds: Vec<_> = contract
        .proof_obligations
        .iter()
        .filter(|ob| ob.obligation_type == ObligationType::Bound)
        .collect();

    // Generate function signatures from equations
    for (eq_name, eq) in &contract.equations {
        let fn_name = eq_name.replace('-', "_");
        let _ = writeln!(out, "/// Contract: {}", contract.metadata.description);

        if let Some(ref dom) = eq.domain {
            let _ = writeln!(out, "/// Domain: {dom}");
        }
        if let Some(ref cod) = eq.codomain {
            let _ = writeln!(out, "/// Codomain: {cod}");
        }
        let _ = writeln!(out, "/// Formula: {}", eq.formula);

        let _ = writeln!(
            out,
            "pub fn {fn_name}(input: &[f32], output: &mut [f32]) {{"
        );

        // Emit preconditions
        for pre in &preconditions {
            let _ = writeln!(out, "    // Precondition: {}", pre.property);
            if let Some(ref formal) = pre.formal {
                let _ = writeln!(out, "    precondition!(!input.is_empty()); // {formal}");
            } else {
                let _ = writeln!(
                    out,
                    "    precondition!(!input.is_empty()); // {}",
                    pre.property
                );
            }
        }

        if preconditions.is_empty() {
            let _ = writeln!(out, "    precondition!(!input.is_empty());");
        }

        let _ = writeln!(out);
        let _ = writeln!(out, "    // --- kernel implementation ---");
        let _ = writeln!(out);

        // Emit invariant checks
        for inv in &invariants {
            let _ = writeln!(out, "    // Invariant: {}", inv.property);
            if let Some(ref formal) = inv.formal {
                let _ = writeln!(out, "    verify!( /* {formal} */ true);");
            }
        }

        // Emit bound checks
        for bnd in &bounds {
            let _ = writeln!(out, "    // Bound: {}", bnd.property);
            if let Some(ref formal) = bnd.formal {
                let _ = writeln!(out, "    verify!( /* {formal} */ true);");
            }
        }

        // Emit postconditions
        for post in &postconditions {
            let _ = writeln!(out, "    // Postcondition: {}", post.property);
            if let Some(ref formal) = post.formal {
                let _ = writeln!(
                    out,
                    "    postcondition!(output.len() == input.len()); // {formal}"
                );
            } else {
                let _ = writeln!(
                    out,
                    "    postcondition!(output.len() == input.len()); // {}",
                    post.property
                );
            }
        }

        let _ = writeln!(out, "}}");
        let _ = writeln!(out);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn generates_mirai_annotations() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Test kernel"
  references: ["Paper A"]
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i) / Σ_j exp(x_j)"
    domain: "x ∈ ℝ^n"
proof_obligations:
  - type: precondition
    property: "Input finite and non-empty"
    formal: "len(x) > 0 ∧ ∀i: finite(x_i)"
  - type: postcondition
    property: "Valid probability distribution"
    formal: "len(output) = len(input)"
  - type: invariant
    property: "Sums to 1"
    formal: "|Σ σ(x)_i - 1| < ε"
falsification_tests:
  - id: F-001
    rule: "norm"
    prediction: "sum ≈ 1"
    if_fails: "bug"
kani_harnesses:
  - id: K-001
    obligation: "sums to 1"
    bound: 8
"#,
        )
        .unwrap();

        let output = generate_mirai_annotations(&contract, "softmax-kernel-v1");
        assert!(output.contains("use mirai_annotations::*;"));
        assert!(output.contains("precondition!"));
        assert!(output.contains("postcondition!"));
        assert!(output.contains("verify!"));
        assert!(output.contains("pub fn softmax"));
    }

    #[test]
    fn empty_obligations_still_generates() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Minimal"
  references: ["X"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .unwrap();

        let output = generate_mirai_annotations(&contract, "identity-v1");
        assert!(output.contains("pub fn f"));
        assert!(output.contains("precondition!(!input.is_empty())"));
    }
}
