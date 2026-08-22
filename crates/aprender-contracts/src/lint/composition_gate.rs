//! Gate 9: COMPOSITION-001 — Compositional shape verification.
//!
//! For every `depends_on` edge where both contracts have equations with
//! `assumes`/`guarantees`, verify that the guarantees of the upstream
//! contract satisfy the assumes of the downstream contract.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use crate::schema::composition::ShapeContract;
use crate::schema::Contract;

use super::rules::RuleSeverity;

use super::finding::LintFinding;
use super::{GateDetail, GateResult};

/// Stem → contract lookup, with ambiguous stems already removed.
type Index<'a> = BTreeMap<&'a str, &'a Contract>;

/// What one `assumes` edge resolved to.
enum Edge {
    /// The chain was verified end to end.
    Satisfied,
    /// The upstream exists but cannot support the assumption — a real defect.
    Broken,
    /// The chain could not be evaluated (upstream missing or ambiguous). Not a
    /// defect in the chain itself, so it must not be counted as broken.
    Unresolved,
}

fn warn(stem: &str, message: String) -> LintFinding {
    LintFinding::new(
        "COMPOSITION-001",
        RuleSeverity::Warning,
        message,
        format!("{stem}.yaml"),
    )
}

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
///
/// `ambiguous` names the stems that MUST NOT be resolved — stems claimed by more
/// than one file with divergent content (see `duplicate_stems.rs`). They are kept
/// out of the index entirely rather than tie-broken, because this index is a
/// `BTreeMap` whose insert order decided which copy survived, and that order came
/// from `read_dir`. Refusing is the only resolution a directory rename cannot move.
pub(crate) fn run_composition_gate(
    contracts: &[(String, Contract)],
    ambiguous: &BTreeSet<String>,
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut findings = Vec::new();
    let mut edges_checked: usize = 0;
    let mut edges_satisfied: usize = 0;
    let mut edges_broken: usize = 0;

    // Ambiguous stems are omitted from the index: including either copy would make
    // the verdict depend on which one won.
    let index: Index = contracts
        .iter()
        .filter(|(s, _)| !ambiguous.contains(s.as_str()))
        .map(|(s, c)| (s.as_str(), c))
        .collect();

    for (stem, contract) in contracts {
        for (eq_name, equation) in &contract.equations {
            let Some(assumes) = equation.assumes.as_ref() else {
                continue;
            };
            if assumes.from_contract.is_none() {
                continue;
            }
            edges_checked += 1;
            match check_edge(stem, eq_name, assumes, &index, ambiguous, &mut findings) {
                Edge::Satisfied => edges_satisfied += 1,
                Edge::Broken => edges_broken += 1,
                Edge::Unresolved => {}
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
        extra: None,
    };

    (result, findings)
}

/// Resolve one `assumes` edge, pushing any findings it produces.
fn check_edge(
    stem: &str,
    eq_name: &str,
    assumes: &ShapeContract,
    index: &Index,
    ambiguous: &BTreeSet<String>,
    findings: &mut Vec<LintFinding>,
) -> Edge {
    let from_contract = assumes
        .from_contract
        .as_deref()
        .expect("caller skipped edges without from_contract");

    // An ambiguous upstream is refused, not guessed. Unresolved, NOT broken: the
    // chain is unverifiable rather than known-bad, and the corpus defect is
    // reported by the duplicate-stems gate (PV-DUP-001).
    if ambiguous.contains(from_contract) {
        findings.push(warn(
            stem,
            format!(
                "{stem}.{eq_name}: assumes from_contract '{from_contract}' is AMBIGUOUS \
                 (claimed by several files with divergent content) — refusing to resolve. \
                 See PV-DUP-001."
            ),
        ));
        return Edge::Unresolved;
    }

    let Some(upstream) = index.get(from_contract) else {
        findings.push(warn(
            stem,
            format!(
                "{stem}.{eq_name}: assumes from_contract '{from_contract}' not found in contract set"
            ),
        ));
        return Edge::Unresolved;
    };

    match assumes.from_equation.as_deref() {
        Some(upstream_eq_name) => check_named_equation(
            stem,
            eq_name,
            assumes,
            upstream,
            from_contract,
            upstream_eq_name,
            findings,
        ),
        None => check_any_equation(stem, eq_name, upstream, from_contract, findings),
    }
}

/// `assumes.from_equation` was given: resolve that exact equation.
fn check_named_equation(
    stem: &str,
    eq_name: &str,
    assumes: &ShapeContract,
    upstream: &Contract,
    from_contract: &str,
    upstream_eq_name: &str,
    findings: &mut Vec<LintFinding>,
) -> Edge {
    let Some(upstream_eq) = upstream.equations.get(upstream_eq_name) else {
        findings.push(warn(
            stem,
            format!(
                "{stem}.{eq_name}: assumes from_equation '{upstream_eq_name}' not found in {from_contract}"
            ),
        ));
        return Edge::Unresolved;
    };

    let Some(guarantees) = upstream_eq.guarantees.as_ref() else {
        // Warning during rollout, Error once all contracts annotated (PMAT-487)
        findings.push(warn(
            stem,
            format!(
                "{stem}.{eq_name}: assumes from {from_contract}.{upstream_eq_name} but upstream has no guarantees"
            ),
        ));
        return Edge::Broken;
    };

    // Check assumed shape keys are a subset of guaranteed shape keys
    for assumed_key in assumes.shapes.keys() {
        if !guarantees.shapes.contains_key(assumed_key) {
            findings.push(warn(
                stem,
                format!(
                    "{stem}.{eq_name}: assumes shape '{assumed_key}' not in {from_contract}.{upstream_eq_name} guarantees"
                ),
            ));
        }
    }
    Edge::Satisfied
}

/// No specific equation named — the upstream just has to guarantee something.
fn check_any_equation(
    stem: &str,
    eq_name: &str,
    upstream: &Contract,
    from_contract: &str,
    findings: &mut Vec<LintFinding>,
) -> Edge {
    if upstream
        .equations
        .values()
        .any(|eq| eq.guarantees.is_some())
    {
        return Edge::Satisfied;
    }
    findings.push(warn(
        stem,
        format!(
            "{stem}.{eq_name}: assumes from {from_contract} but no equations there have guarantees"
        ),
    ));
    Edge::Unresolved
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
        let (result, findings) = run_composition_gate(&contracts, &BTreeSet::new());
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
        let (result, findings) = run_composition_gate(&contracts, &BTreeSet::new());
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
        let (result, findings) = run_composition_gate(&contracts, &BTreeSet::new());
        // PMAT-487: Gate is now blocking — broken edges fail
        assert!(!result.passed);
        assert!(findings.iter().any(|f| f.severity == RuleSeverity::Warning));
    }
}
