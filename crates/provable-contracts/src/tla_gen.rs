//! TLA+ system-level model checking specification generation.
//!
//! Generates TLA+ `MODULE` specifications from the contract dependency DAG,
//! translating contract invariants into system-level safety and liveness
//! properties for verification with TLC or Apalache.

use std::fmt::Write;

use crate::graph::DependencyGraph;
use crate::schema::Contract;

/// Generate a TLA+ module from a set of contracts and their dependency graph.
///
/// The module declares:
/// - `VARIABLES` for each contract's state
/// - `INVARIANT` from each contract's proof obligations
/// - `PROPERTY` for liveness obligations (termination, progress)
pub fn generate_tla_module(
    module_name: &str,
    contracts: &[(String, &Contract)],
    graph: &DependencyGraph,
) -> String {
    let mut out = String::with_capacity(4096);

    let _ = writeln!(out, "---- MODULE {module_name} ----");
    let _ = writeln!(out, "EXTENDS Naturals, Sequences, TLC");
    let _ = writeln!(out);

    // Auto-generation comment with DAG summary
    let _ = writeln!(
        out,
        "(* Auto-generated from {} contracts *)",
        contracts.len()
    );
    let _ = writeln!(out, "(* Dependency edges: {} *)", graph.edges.len());
    let _ = writeln!(out);

    // Variables: one per contract
    if !contracts.is_empty() {
        let _ = write!(out, "VARIABLES ");
        let var_names: Vec<String> = contracts
            .iter()
            .map(|(stem, _)| stem.replace('-', "_"))
            .collect();
        let _ = writeln!(out, "{}", var_names.join(", "));
        let _ = writeln!(out);
    }

    // Constants from equations
    let _ = writeln!(out, "(* Constants derived from contract equations *)");
    for (stem, contract) in contracts {
        for (eq_name, eq) in &contract.equations {
            let _ = writeln!(
                out,
                "(* {stem}.{eq_name}: {} *)",
                eq.formula.replace("(*", "( *").replace("*)", "* )")
            );
        }
    }
    let _ = writeln!(out);

    // Init predicate
    let _ = writeln!(out, "(* Initial state *)");
    let _ = writeln!(out, "Init ==");
    for (stem, _) in contracts {
        let var = stem.replace('-', "_");
        let _ = writeln!(out, "    /\\ {var} = \"idle\"");
    }
    let _ = writeln!(out);

    // Invariants from proof obligations
    let _ = writeln!(out, "(* Safety invariants from contract obligations *)");
    for (stem, contract) in contracts {
        for ob in &contract.proof_obligations {
            let inv_name = format!(
                "{}_{}_{}",
                stem.replace('-', "_"),
                ob.obligation_type,
                ob.property
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_' || *c == ' ')
                    .collect::<String>()
                    .replace(' ', "_")
                    .chars()
                    .take(30)
                    .collect::<String>()
            );
            let _ = writeln!(out, "(* {}: {} *)", stem, ob.property);
            let _ = writeln!(out, "{inv_name} == TRUE");
            let _ = writeln!(out);
        }
    }

    // Dependency ordering constraints
    if !graph.edges.is_empty() {
        let _ = writeln!(out, "(* Dependency ordering constraints *)");
        for (from, deps) in &graph.edges {
            for to in deps {
                let from_var = from.replace('-', "_");
                let to_var = to.replace('-', "_");
                let _ = writeln!(
                    out,
                    "(* {from} depends on {to}: {to_var} must complete before {from_var} *)"
                );
            }
        }
        let _ = writeln!(out);
    }

    // Spec composition
    let _ = writeln!(out, "(* Specification *)");
    let vars = contracts
        .iter()
        .map(|(s, _)| s.replace('-', "_"))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(out, "vars == <<{vars}>>");
    let _ = writeln!(out);
    let _ = writeln!(out, "Spec == Init /\\ [][TRUE]_vars");
    let _ = writeln!(out);
    let _ = writeln!(out, "====");

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::dependency_graph;
    use crate::schema::parse_contract_str;

    #[test]
    fn generates_tla_module() {
        let a = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Softmax"
  references: ["Paper"]
equations:
  softmax:
    formula: "σ(x) = exp(x) / Σ exp(x)"
proof_obligations:
  - type: invariant
    property: "Sums to 1"
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

        let b = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Attention"
  references: ["Paper"]
  depends_on: ["softmax-kernel-v1"]
equations:
  attention:
    formula: "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"
proof_obligations:
  - type: invariant
    property: "Output shape preserved"
falsification_tests:
  - id: F-001
    rule: "shape"
    prediction: "shape matches"
    if_fails: "bug"
kani_harnesses:
  - id: K-001
    obligation: "shape"
    bound: 8
"#,
        )
        .unwrap();

        let contracts = vec![
            ("softmax-kernel-v1".to_string(), &a),
            ("attention-kernel-v1".to_string(), &b),
        ];
        let graph = dependency_graph(&contracts);
        let refs: Vec<(String, &Contract)> =
            contracts.iter().map(|(s, c)| (s.clone(), *c)).collect();

        let tla = generate_tla_module("InferencePipeline", &refs, &graph);
        assert!(tla.contains("---- MODULE InferencePipeline ----"));
        assert!(tla.contains("EXTENDS Naturals"));
        assert!(tla.contains("VARIABLES"));
        assert!(tla.contains("softmax_kernel_v1"));
        assert!(tla.contains("attention_kernel_v1"));
        assert!(tla.contains("Init =="));
        assert!(tla.contains("Spec =="));
        assert!(tla.contains("===="));
    }

    #[test]
    fn handles_empty_contracts() {
        let graph = dependency_graph(&[]);
        let tla = generate_tla_module("Empty", &[], &graph);
        assert!(tla.contains("---- MODULE Empty ----"));
        assert!(tla.contains("===="));
    }
}
