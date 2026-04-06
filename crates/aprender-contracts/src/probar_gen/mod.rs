//! Probar property-test generator — Phase 5 of the pipeline.
//!
//! Generates property-based test source code from both the
//! `falsification_tests` and `proof_obligations` sections of
//! YAML contracts. Maps obligation types to test patterns:
//!
//! | Obligation | Pattern |
//! |---|---|
//! | invariant | `#[probar::property]` with random inputs |
//! | equivalence | comparison of two implementations |
//! | bound | range checking on all outputs |
//! | monotonicity | ordered pair metamorphic test |
//! | idempotency | `f(f(x)) == f(x)` |
//! | linearity | `f(αx) == α·f(x)` metamorphic |
//! | symmetry | permutation invariance |
//! | associativity | `(a⊕b)⊕c == a⊕(b⊕c)` |
//! | conservation | `Q(before) == Q(after)` |

mod patterns;
mod wired;

use crate::schema::{Contract, ObligationType, ProofObligation};

use patterns::{
    generate_associativity_body, generate_bound_body, generate_conservation_body,
    generate_equivalence_body, generate_idempotency_body, generate_invariant_body,
    generate_linearity_body, generate_monotonicity_body, generate_ordering_body,
    generate_symmetry_body,
};
pub use wired::generate_wired_probar_tests;

/// Generate probar property-based tests from a contract.
///
/// Produces two sections:
/// 1. Obligation-derived property tests (from `proof_obligations`)
/// 2. Falsification test stubs (from `falsification_tests`)
pub fn generate_probar_tests(contract: &Contract) -> String {
    if contract.proof_obligations.is_empty() && contract.falsification_tests.is_empty() {
        return String::from("// No probar tests defined in this contract.\n");
    }

    let mut out = String::new();

    out.push_str("#[cfg(test)]\nmod probar_tests {\n");
    out.push_str("    use super::*;\n\n");

    // Section 1: Property tests from proof obligations
    if !contract.proof_obligations.is_empty() {
        out.push_str(
            "    // === Property tests derived from proof \
             obligations ===\n\n",
        );
        for (i, ob) in contract.proof_obligations.iter().enumerate() {
            generate_obligation_test(&mut out, ob, i);
        }
    }

    // Section 2: Falsification test stubs
    if !contract.falsification_tests.is_empty() {
        out.push_str("    // === Falsification test stubs ===\n\n");
        for test in &contract.falsification_tests {
            generate_falsification_stub(&mut out, test);
        }
    }

    out.push_str("}\n");
    out
}

/// Emit a single property test derived from a proof obligation
fn generate_obligation_test(out: &mut String, ob: &ProofObligation, index: usize) {
    let fn_name = obligation_fn_name(ob, index);
    let pattern = obligation_pattern(ob.obligation_type);

    // Doc comment
    out.push_str(&format!(
        "    /// Obligation: {} ({})\n",
        ob.property, ob.obligation_type
    ));
    if let Some(ref formal) = ob.formal {
        out.push_str(&format!("    /// Formal: {formal}\n"));
    }
    out.push_str(&format!("    /// Pattern: {pattern}\n"));
    if let Some(tol) = ob.tolerance {
        out.push_str(&format!("    /// Tolerance: {tol}\n"));
    }

    out.push_str("    #[test]\n");
    out.push_str(&format!("    fn {fn_name}() {{\n"));

    // Body based on obligation type
    match ob.obligation_type {
        ObligationType::Completeness
        | ObligationType::Soundness
        | ObligationType::Invariant
        | ObligationType::Involution
        | ObligationType::Determinism
        | ObligationType::Roundtrip
        | ObligationType::StateMachine
        | ObligationType::Classification
        | ObligationType::Independence
        | ObligationType::Termination
        | ObligationType::Precondition
        | ObligationType::Postcondition
        | ObligationType::Frame
        | ObligationType::LoopInvariant
        | ObligationType::LoopVariant
        | ObligationType::OldState
        | ObligationType::Subcontract => {
            generate_invariant_body(out, ob);
        }
        ObligationType::Equivalence => {
            generate_equivalence_body(out, ob);
        }
        ObligationType::Bound => {
            generate_bound_body(out, ob);
        }
        ObligationType::Monotonicity => {
            generate_monotonicity_body(out, ob);
        }
        ObligationType::Idempotency => {
            generate_idempotency_body(out, ob);
        }
        ObligationType::Linearity => {
            generate_linearity_body(out, ob);
        }
        ObligationType::Symmetry => {
            generate_symmetry_body(out, ob);
        }
        ObligationType::Associativity => {
            generate_associativity_body(out, ob);
        }
        ObligationType::Conservation => {
            generate_conservation_body(out, ob);
        }
        ObligationType::Ordering => {
            generate_ordering_body(out, ob);
        }
    }

    out.push_str("    }\n\n");
}

/// Derive a test function name from the obligation property text
fn obligation_fn_name(ob: &ProofObligation, index: usize) -> String {
    let base = ob
        .property
        .to_lowercase()
        .replace(|c: char| !c.is_alphanumeric(), "_")
        .trim_matches('_')
        .to_string();
    if base.is_empty() {
        format!("prop_obligation_{index}")
    } else {
        format!("prop_{base}")
    }
}

/// Return a mathematical pattern description for the given obligation type
fn obligation_pattern(ot: ObligationType) -> &'static str {
    match ot {
        ObligationType::Invariant => "∀x ∈ Domain: P(f(x)) — property holds for all inputs",
        ObligationType::Equivalence => "∀x: |f(x) - g(x)| < ε — two implementations agree",
        ObligationType::Bound => "∀x: a ≤ f(x)_i ≤ b — output range bounded",
        ObligationType::Monotonicity => "x_i > x_j → f(x)_i > f(x)_j — order preserved",
        ObligationType::Idempotency => "f(f(x)) = f(x) — applying twice gives same result",
        ObligationType::Linearity => "f(αx) = α·f(x) — homogeneous scaling",
        ObligationType::Symmetry => "f(permute(x)) related to f(x) — permutation property",
        ObligationType::Associativity => "(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) — grouping invariant",
        ObligationType::Conservation => "Q(before) = Q(after) — conserved quantity",
        ObligationType::Ordering => "a ≤ b → f(a) ≤ f(b) — order relation maintained",
        ObligationType::Completeness => "∀ required elements present — completeness verified",
        ObligationType::Soundness => "∀x: P(x) → Q(f(x)) — soundness of transformation",
        ObligationType::Involution => "f(f(x)) = x — involution (self-inverse)",
        ObligationType::Determinism => "f(x) = f(x) — deterministic output for same input",
        ObligationType::Roundtrip => "decode(encode(x)) = x — roundtrip fidelity",
        ObligationType::StateMachine => "S × A → S — valid state transitions",
        ObligationType::Classification => "f(x) ∈ C — output belongs to valid class set",
        ObligationType::Independence => "P(A∩B) = P(A)·P(B) — statistical independence",
        ObligationType::Termination => "algorithm terminates in finite steps",
        // Eiffel DbC types
        ObligationType::Precondition => "P(input) — caller must guarantee before call",
        ObligationType::Postcondition => "P(in) → Q(out) — kernel guarantees if pre holds",
        ObligationType::Frame => "modifies(S) ∧ preserves(T\\S) — only S may change",
        ObligationType::LoopInvariant => "∀ iter i: P(state_i) — maintained across iterations",
        ObligationType::LoopVariant => "V(state) ∈ ℕ, strictly decreasing — termination witness",
        ObligationType::OldState => "Q(old(state), new(state)) — relates pre to post state",
        ObligationType::Subcontract => "weaken(pre) ∧ strengthen(post) — behavioral subtyping",
    }
}

/// Emit an `unimplemented!()` test stub for a falsification test entry
fn generate_falsification_stub(out: &mut String, test: &crate::schema::FalsificationTest) {
    // Doc comment
    out.push_str(&format!("    /// {}: {}\n", test.id, test.rule));
    out.push_str(&format!("    /// Prediction: {}\n", test.prediction));
    out.push_str(&format!("    /// If fails: {}\n", test.if_fails));

    let fn_name = test.id.to_lowercase().replace('-', "_");

    out.push_str(&format!("    #[test]\n    fn prop_{fn_name}() {{\n"));

    if let Some(ref method) = test.test {
        out.push_str(&format!("        // Method: {method}\n"));
    }

    out.push_str(&format!(
        "        unimplemented!(\"Implement falsification \
         test for {}\")\n",
        test.id
    ));
    out.push_str("    }\n\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn generate_empty_probar_tests() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "No tests"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("No probar tests"));
    }

    #[test]
    fn generate_falsification_stubs() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Probar test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests:
  - id: FALSIFY-001
    rule: "normalization"
    prediction: "sum(output) ≈ 1.0"
    if_fails: "missing max subtraction"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("mod probar_tests"));
        assert!(code.contains("fn prop_falsify_001()"));
        assert!(code.contains("Falsification test stubs"));
    }

    #[test]
    fn generate_invariant_property_test() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Invariant test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|sum(f(x)) - 1.0| < ε"
    tolerance: 1.0e-6
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("fn prop_output_sums_to_1()"));
        assert!(code.contains("invariant"));
        assert!(code.contains("postcondition"));
        assert!(code.contains("1e-6"));
    }

    #[test]
    fn generate_equivalence_property_test() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Equivalence"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: equivalence
    property: "SIMD matches scalar"
    tolerance: 8.0
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("fn prop_simd_matches_scalar()"));
        assert!(code.contains("equivalence"));
        assert!(code.contains("reference_impl"));
        assert!(code.contains("optimized_impl"));
    }

    #[test]
    fn generate_monotonicity_test() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Monotonicity"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: monotonicity
    property: "Order preservation"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("fn prop_order_preservation()"));
        assert!(code.contains("monotonicity"));
        assert!(code.contains("input[i] > input[j]"));
    }

    #[test]
    fn generate_all_obligation_types() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "All types"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "invariant test"
  - type: equivalence
    property: "equivalence test"
  - type: bound
    property: "bound test"
  - type: monotonicity
    property: "monotonicity test"
  - type: idempotency
    property: "idempotency test"
  - type: linearity
    property: "linearity test"
  - type: symmetry
    property: "symmetry test"
  - type: associativity
    property: "associativity test"
  - type: conservation
    property: "conservation test"
  - type: ordering
    property: "ordering test"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        assert!(code.contains("invariant"));
        assert!(code.contains("equivalence"));
        assert!(code.contains("bound"));
        assert!(code.contains("monotonicity"));
        assert!(code.contains("idempotency"));
        assert!(code.contains("linearity"));
        assert!(code.contains("symmetry"));
        assert!(code.contains("associativity"));
        assert!(code.contains("conservation"));
        assert!(code.contains("ordering"));
        // 10 test functions
        assert_eq!(code.matches("#[test]").count(), 10);
    }

    #[test]
    fn generate_mixed_obligations_and_falsification() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Mixed"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is finite"
    if_fails: "overflow"
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let code = generate_probar_tests(&contract);
        // Both sections present
        assert!(code.contains("Property tests derived from proof obligations"));
        assert!(code.contains("Falsification test stubs"));
        assert_eq!(code.matches("#[test]").count(), 2);
    }
}
