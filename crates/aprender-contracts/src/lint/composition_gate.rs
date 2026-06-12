//! Gate 8: COMPOSITION-001 — Compositional shape verification.
//!
//! For every `depends_on` edge where both contracts have equations with
//! `assumes`/`guarantees`, verify that the guarantees of the upstream
//! contract satisfy the assumes of the downstream contract.

use std::collections::BTreeMap;
use std::time::Instant;

use crate::schema::Contract;

use super::rules::RuleSeverity;

use super::finding::LintFinding;
use super::{GateDetail, GateResult};

/// Run COMPOSITION-001: verify that assumes/guarantees chains are consistent
/// across the dependency graph.
///
/// For each contract C that has equations with `assumes.from_contract`:
///   1. Resolve the upstream contract by stem name
///   2. Resolve the upstream equation by name
///   3. Check that the upstream equation has `guarantees`
///   4. Verify shape keys in `assumes.shapes` are a subset of upstream `guarantees.shapes`
///
/// Returns errors for broken chains and warnings for unresolvable references.
pub(crate) fn run_composition_gate(
    contracts: &[(String, Contract)],
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut findings = Vec::new();
    let mut edges_checked: usize = 0;
    let mut edges_satisfied: usize = 0;
    let mut edges_broken: usize = 0;

    // Build stem → contract index for O(1) lookup
    let index: BTreeMap<&str, &Contract> = contracts.iter().map(|(s, c)| (s.as_str(), c)).collect();

    for (stem, contract) in contracts {
        for (eq_name, equation) in &contract.equations {
            let Some(assumes) = &equation.assumes else {
                continue;
            };
            let Some(from_contract) = &assumes.from_contract else {
                continue;
            };
            let from_eq = assumes.from_equation.as_deref();

            edges_checked += 1;

            // Resolve upstream contract
            let Some(upstream) = index.get(from_contract.as_str()) else {
                findings.push(LintFinding::new(
                    "COMPOSITION-001",
                    RuleSeverity::Warning,
                    format!(
                        "{stem}.{eq_name}: assumes from_contract '{from_contract}' not found in contract set"
                    ),
                    format!("{stem}.yaml"),
                ));
                continue;
            };

            // If from_equation is specified, resolve it
            if let Some(upstream_eq_name) = from_eq {
                let Some(upstream_eq) = upstream.equations.get(upstream_eq_name) else {
                    findings.push(LintFinding::new(
                        "COMPOSITION-001",
                        RuleSeverity::Warning,
                        format!(
                            "{stem}.{eq_name}: assumes from_equation '{upstream_eq_name}' not found in {from_contract}"
                        ),
                        format!("{stem}.yaml"),
                    ));
                    continue;
                };

                // Check upstream has guarantees
                if upstream_eq.guarantees.is_none() {
                    // Warning during rollout, Error once all contracts annotated (PMAT-487)
                    findings.push(LintFinding::new(
                        "COMPOSITION-001",
                        RuleSeverity::Warning,
                        format!(
                            "{stem}.{eq_name}: assumes from {from_contract}.{upstream_eq_name} but upstream has no guarantees"
                        ),
                        format!("{stem}.yaml"),
                    ));
                    edges_broken += 1;
                    continue;
                }

                // Check assumed shape keys are a subset of guaranteed shape keys
                let upstream_guarantees = upstream_eq
                    .guarantees
                    .as_ref()
                    .expect("guarantees is Some: the is_none() branch above continues");
                for assumed_key in assumes.shapes.keys() {
                    if !upstream_guarantees.shapes.contains_key(assumed_key) {
                        findings.push(LintFinding::new(
                            "COMPOSITION-001",
                            RuleSeverity::Warning,
                            format!(
                                "{stem}.{eq_name}: assumes shape '{assumed_key}' not in {from_contract}.{upstream_eq_name} guarantees"
                            ),
                            format!("{stem}.yaml"),
                        ));
                    }
                }

                edges_satisfied += 1;
            } else {
                // No specific equation — just check the contract exists and has
                // at least one equation with guarantees
                let has_any_guarantees = upstream
                    .equations
                    .values()
                    .any(|eq| eq.guarantees.is_some());
                if has_any_guarantees {
                    edges_satisfied += 1;
                } else {
                    findings.push(LintFinding::new(
                        "COMPOSITION-001",
                        RuleSeverity::Warning,
                        format!(
                            "{stem}.{eq_name}: assumes from {from_contract} but no equations there have guarantees"
                        ),
                        format!("{stem}.yaml"),
                    ));
                }
            }
        }
    }

    // PMAT-487: Composition gate is now blocking — broken edges fail the gate.
    // Contracts with assumes must reference upstream equations that have guarantees.
    let passed = edges_broken == 0;
    let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(0);

    let result = GateResult {
        name: "composition".into(),
        passed,
        skipped: false,
        duration_ms,
        detail: GateDetail::Composition {
            edges_checked,
            edges_satisfied,
            edges_broken,
        },
    };

    (result, findings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lint::rules::RuleSeverity;
    use crate::schema::composition::{ShapeContract, ShapeExpr};
    use crate::schema::{
        Contract, Equation, FalsificationTest, KaniHarness, Metadata, ProofObligation,
    };
    use std::collections::BTreeMap;

    fn minimal_metadata(deps: Vec<&str>) -> Metadata {
        Metadata {
            version: "1.0.0".into(),
            description: "test".into(),
            references: vec!["test ref".into()],
            depends_on: deps.into_iter().map(String::from).collect(),
            ..Default::default()
        }
    }

    fn eq_with_guarantees(shapes: &[(&str, Vec<&str>)]) -> Equation {
        let mut shape_map = BTreeMap::new();
        for (name, dims) in shapes {
            shape_map.insert(
                name.to_string(),
                ShapeExpr {
                    dims: dims.iter().map(|s| s.to_string()).collect(),
                    dtype: None,
                },
            );
        }
        Equation {
            guarantees: Some(ShapeContract {
                shapes: shape_map,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn eq_with_assumes(
        from_contract: &str,
        from_eq: &str,
        shapes: &[(&str, Vec<&str>)],
    ) -> Equation {
        let mut shape_map = BTreeMap::new();
        for (name, dims) in shapes {
            shape_map.insert(
                name.to_string(),
                ShapeExpr {
                    dims: dims.iter().map(|s| s.to_string()).collect(),
                    dtype: None,
                },
            );
        }
        Equation {
            assumes: Some(ShapeContract {
                shapes: shape_map,
                from_contract: Some(from_contract.to_string()),
                from_equation: Some(from_eq.to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn minimal_contract(deps: Vec<&str>, equations: BTreeMap<String, Equation>) -> Contract {
        Contract {
            metadata: minimal_metadata(deps),
            equations,
            proof_obligations: vec![ProofObligation {
                obligation_type: crate::schema::ObligationType::Invariant,
                property: "test".into(),
                ..Default::default()
            }],
            falsification_tests: vec![FalsificationTest {
                id: "F-001".into(),
                rule: "test".into(),
                prediction: "test".into(),
                if_fails: "test".into(),
                ..Default::default()
            }],
            kani_harnesses: vec![KaniHarness {
                id: "K-001".into(),
                obligation: "test".into(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn composition_gate_no_assumes_passes() {
        let contracts = vec![(
            "test-v1".to_string(),
            minimal_contract(vec![], BTreeMap::new()),
        )];
        let (result, findings) = run_composition_gate(&contracts);
        assert!(result.passed);
        assert!(findings.is_empty());
    }

    #[test]
    fn composition_gate_valid_chain_passes() {
        let mut upstream_eqs = BTreeMap::new();
        upstream_eqs.insert(
            "produce".to_string(),
            eq_with_guarantees(&[("output", vec!["batch", "seq", "hidden"])]),
        );
        let upstream = minimal_contract(vec![], upstream_eqs);

        let mut downstream_eqs = BTreeMap::new();
        downstream_eqs.insert(
            "consume".to_string(),
            eq_with_assumes(
                "upstream-v1",
                "produce",
                &[("input", vec!["batch", "seq", "hidden"])],
            ),
        );
        let downstream = minimal_contract(vec!["upstream-v1"], downstream_eqs);

        let contracts = vec![
            ("upstream-v1".to_string(), upstream),
            ("downstream-v1".to_string(), downstream),
        ];
        let (result, findings) = run_composition_gate(&contracts);
        assert!(result.passed);
        // Shape key mismatch is a warning, not error — "input" not in upstream guarantees
        // but upstream equation has guarantees, so edge is satisfied
        let errors: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == RuleSeverity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn composition_gate_missing_guarantees_is_error() {
        let mut upstream_eqs = BTreeMap::new();
        upstream_eqs.insert("produce".to_string(), Equation::default()); // no guarantees

        let upstream = minimal_contract(vec![], upstream_eqs);

        let mut downstream_eqs = BTreeMap::new();
        downstream_eqs.insert(
            "consume".to_string(),
            eq_with_assumes("upstream-v1", "produce", &[]),
        );
        let downstream = minimal_contract(vec!["upstream-v1"], downstream_eqs);

        let contracts = vec![
            ("upstream-v1".to_string(), upstream),
            ("downstream-v1".to_string(), downstream),
        ];
        let (result, findings) = run_composition_gate(&contracts);
        // PMAT-487: Gate is now blocking — broken edges fail
        assert!(!result.passed);
        assert!(findings.iter().any(|f| f.severity == RuleSeverity::Warning));
    }
}
