//! Contract and codebase scoring.
//!
//! Provides quantitative quality assessment for individual contracts
//! and codebases that consume them. Five dimensions per contract,
//! ten dimensions per codebase (`PVScore`), grades A-F.
//!
//! Spec: `docs/specifications/sub/scoring.md`, `docs/specifications/sub/pvscore.md`

mod codebase;
pub mod drift;
mod pvscore;
mod types;

pub use codebase::{score_codebase, score_codebase_full, score_codebase_with_pagerank};
pub use pvscore::pvscore_10dim;
pub use types::{CodebaseScore, ContractScore, Grade, ScoreProbe, ScoringGap, ScoringWeights};

use crate::binding::{BindingRegistry, ImplStatus};
use crate::schema::{Contract, KaniHarness, KaniStrategy, LeanStatus};

/// Score a single contract against its completeness and verification depth.
///
/// Five dimensions (weights from spec):
/// - D1: Specification depth (20%)
/// - D2: Falsification coverage (25%)
/// - D3: Kani proof coverage (25%)
/// - D4: Lean proof coverage (10%)
/// - D5: Binding coverage (20%)
pub fn score_contract(
    contract: &Contract,
    binding: Option<&BindingRegistry>,
    stem: &str,
) -> ContractScore {
    score_contract_weighted(contract, binding, stem, &ScoringWeights::default())
}

/// Score a contract with custom weights for each dimension.
pub fn score_contract_weighted(
    contract: &Contract,
    binding: Option<&BindingRegistry>,
    stem: &str,
    weights: &ScoringWeights,
) -> ContractScore {
    let w = weights.normalized();
    let mut probes = Vec::new();

    let spec_depth = compute_spec_depth(contract, &mut probes);
    let falsification = compute_falsification_coverage(contract, &mut probes);
    let kani = compute_kani_coverage(contract, &mut probes);
    let lean = compute_lean_coverage(contract, &mut probes);
    let binding_cov = compute_binding_coverage(contract, binding, stem, &mut probes);

    // Non-kernel contracts (registries, model-family schemas, patterns,
    // reference documents) are data-only: they define lookup tables or
    // metadata, not executable functions. Give them full credit for
    // binding/lean so the composite reflects their declarative nature.
    let (effective_binding, effective_lean) = if contract.requires_proofs() {
        (binding_cov, lean)
    } else {
        (1.0, lean.max(0.5))
    };

    let composite = spec_depth * w.spec_depth
        + falsification * w.falsification
        + kani * w.kani
        + effective_lean * w.lean
        + effective_binding * w.binding;

    ContractScore {
        stem: stem.to_string(),
        spec_depth,
        falsification_coverage: falsification,
        kani_coverage: kani,
        lean_coverage: effective_lean,
        binding_coverage: effective_binding,
        composite,
        grade: Grade::from_score(composite),
        probes,
    }
}

#[allow(clippy::too_many_lines)]
fn compute_spec_depth(contract: &Contract, probes: &mut Vec<ScoreProbe>) -> f64 {
    let mut score = 0.0;

    // Has equations (0.30)
    let has_equations = !contract.equations.is_empty();
    probes.push(ScoreProbe {
        dimension: "spec_depth".into(),
        probe: "has_equations".into(),
        outcome: has_equations,
        detail: if has_equations {
            format!("{} equation(s)", contract.equations.len())
        } else {
            "(no equations)".into()
        },
    });
    if has_equations {
        score += 0.30;
    }

    // Per-equation probes: domain and invariants
    for (name, eq) in &contract.equations {
        let has_domain = eq.domain.is_some();
        probes.push(ScoreProbe {
            dimension: "spec_depth".into(),
            probe: format!("{name}: domain"),
            outcome: has_domain,
            detail: if has_domain {
                eq.domain.as_deref().unwrap_or("").to_string()
            } else {
                "(no domain)".into()
            },
        });
        let has_inv = !eq.invariants.is_empty();
        probes.push(ScoreProbe {
            dimension: "spec_depth".into(),
            probe: format!("{name}: invariants"),
            outcome: has_inv,
            detail: if has_inv {
                format!("{} invariant(s)", eq.invariants.len())
            } else {
                "(no invariants)".into()
            },
        });
    }

    // Has domains on equations (0.15)
    let has_domains = contract.equations.values().any(|eq| eq.domain.is_some());
    if has_domains {
        score += 0.15;
    }

    // Has invariants on equations (0.15)
    let has_invariants = contract
        .equations
        .values()
        .any(|eq| !eq.invariants.is_empty());
    if has_invariants {
        score += 0.15;
    }

    // Has kernel structure (0.15)
    let has_ks = contract.kernel_structure.is_some();
    probes.push(ScoreProbe {
        dimension: "spec_depth".into(),
        probe: "kernel_structure".into(),
        outcome: has_ks,
        detail: if has_ks {
            "present".into()
        } else {
            "(no kernel_structure)".into()
        },
    });
    if has_ks {
        score += 0.15;
    }

    // Has tolerances on obligations (0.10)
    let has_tolerances = contract
        .proof_obligations
        .iter()
        .any(|ob| ob.tolerance.is_some());
    probes.push(ScoreProbe {
        dimension: "spec_depth".into(),
        probe: "has_tolerances".into(),
        outcome: has_tolerances,
        detail: if has_tolerances {
            "at least one obligation has tolerance".into()
        } else {
            "(no tolerances)".into()
        },
    });
    if has_tolerances {
        score += 0.10;
    }

    // Has references (0.10)
    let has_refs = !contract.metadata.references.is_empty();
    probes.push(ScoreProbe {
        dimension: "spec_depth".into(),
        probe: "references".into(),
        outcome: has_refs,
        detail: if has_refs {
            format!("{} reference(s)", contract.metadata.references.len())
        } else {
            "(no references)".into()
        },
    });
    if has_refs {
        score += 0.10;
    }

    // Has depends_on (0.05)
    let has_deps = !contract.metadata.depends_on.is_empty();
    probes.push(ScoreProbe {
        dimension: "spec_depth".into(),
        probe: "depends_on".into(),
        outcome: has_deps,
        detail: if has_deps {
            format!("{} dep(s)", contract.metadata.depends_on.len())
        } else {
            "(no depends_on)".into()
        },
    });
    if has_deps {
        score += 0.05;
    }

    score
}

#[allow(clippy::cast_precision_loss)]
fn compute_falsification_coverage(contract: &Contract, probes: &mut Vec<ScoreProbe>) -> f64 {
    let total = contract.proof_obligations.len();

    // Emit a probe per obligation
    for ob in &contract.proof_obligations {
        let matching_test = contract
            .falsification_tests
            .iter()
            .find(|t| t.rule == ob.property);
        let has_test = matching_test.is_some();
        probes.push(ScoreProbe {
            dimension: "falsification".into(),
            probe: ob.property.clone(),
            outcome: has_test,
            detail: if let Some(t) = matching_test {
                t.id.clone()
            } else {
                "(no test)".into()
            },
        });
    }

    // Also probe any tests that don't match an obligation
    for t in &contract.falsification_tests {
        let matches_obligation = contract
            .proof_obligations
            .iter()
            .any(|ob| ob.property == t.rule);
        if !matches_obligation {
            probes.push(ScoreProbe {
                dimension: "falsification".into(),
                probe: t.rule.clone(),
                outcome: true,
                detail: format!("{} (unmatched)", t.id),
            });
        }
    }

    if total == 0 {
        return if contract.falsification_tests.is_empty() {
            0.0
        } else {
            1.0
        };
    }

    // Preserve original behaviour: min(test_count, obligation_count) / obligation_count
    let covered = contract.falsification_tests.len().min(total);
    covered as f64 / total as f64
}

#[allow(clippy::cast_precision_loss)]
fn compute_kani_coverage(contract: &Contract, probes: &mut Vec<ScoreProbe>) -> f64 {
    let total = contract.proof_obligations.len();

    // Emit a probe per obligation showing which harness covers it
    for ob in &contract.proof_obligations {
        let matching_harness = contract
            .kani_harnesses
            .iter()
            .find(|h| h.obligation == ob.property);
        let has_harness = matching_harness.is_some();
        probes.push(ScoreProbe {
            dimension: "kani".into(),
            probe: ob.property.clone(),
            outcome: has_harness,
            detail: if let Some(h) = matching_harness {
                let strategy_note = h.strategy.map(|s| format!(" [{s}]")).unwrap_or_default();
                format!("{}{strategy_note}", h.id)
            } else {
                "(no harness)".into()
            },
        });
    }

    // Also probe harnesses that don't match any obligation
    for h in &contract.kani_harnesses {
        let matches_obligation = contract
            .proof_obligations
            .iter()
            .any(|ob| ob.property == h.obligation);
        if !matches_obligation {
            probes.push(ScoreProbe {
                dimension: "kani".into(),
                probe: h.obligation.clone(),
                outcome: true,
                detail: format!("{} (unmatched)", h.id),
            });
        }
    }

    if total == 0 {
        return if contract.kani_harnesses.is_empty() {
            0.0
        } else {
            1.0
        };
    }

    let strategy_weight = |h: &KaniHarness| -> f64 {
        match h.strategy.as_ref() {
            Some(KaniStrategy::Exhaustive) => 1.0,
            Some(KaniStrategy::BoundedInt) => 0.9,
            Some(KaniStrategy::StubFloat) => 0.8,
            Some(KaniStrategy::Compositional) => 0.7,
            None => 0.5,
        }
    };

    let weighted_sum: f64 = contract.kani_harnesses.iter().map(strategy_weight).sum();
    (weighted_sum / total as f64).min(1.0)
}

#[allow(clippy::cast_precision_loss)]
fn compute_lean_coverage(contract: &Contract, probes: &mut Vec<ScoreProbe>) -> f64 {
    // Emit a probe per obligation showing lean status
    for ob in &contract.proof_obligations {
        match &ob.lean {
            Some(lean) if lean.status != LeanStatus::NotApplicable => {
                let is_proved = lean.status == LeanStatus::Proved;
                probes.push(ScoreProbe {
                    dimension: "lean".into(),
                    probe: ob.property.clone(),
                    outcome: is_proved,
                    detail: if is_proved {
                        format!("{} (proved)", lean.theorem)
                    } else {
                        format!("{} ({})", lean.theorem, lean.status)
                    },
                });
            }
            Some(lean) => {
                // NotApplicable — still informative as a probe
                probes.push(ScoreProbe {
                    dimension: "lean".into(),
                    probe: ob.property.clone(),
                    outcome: false,
                    detail: format!("{} (not-applicable)", lean.theorem),
                });
            }
            None => {
                probes.push(ScoreProbe {
                    dimension: "lean".into(),
                    probe: ob.property.clone(),
                    outcome: false,
                    detail: "(no lean_theorem)".into(),
                });
            }
        }
    }

    let applicable: Vec<_> = contract
        .proof_obligations
        .iter()
        .filter(|ob| {
            ob.lean
                .as_ref()
                .is_some_and(|l| l.status != LeanStatus::NotApplicable)
        })
        .collect();

    if applicable.is_empty() {
        return 0.0;
    }

    let proved = applicable
        .iter()
        .filter(|ob| {
            ob.lean
                .as_ref()
                .is_some_and(|l| l.status == LeanStatus::Proved)
        })
        .count();

    proved as f64 / applicable.len() as f64
}

#[allow(clippy::cast_precision_loss)]
fn compute_binding_coverage(
    _contract: &Contract,
    binding: Option<&BindingRegistry>,
    stem: &str,
    probes: &mut Vec<ScoreProbe>,
) -> f64 {
    let Some(binding) = binding else {
        probes.push(ScoreProbe {
            dimension: "binding".into(),
            probe: "(all)".into(),
            outcome: false,
            detail: "(no binding registry)".into(),
        });
        return 0.0;
    };

    let relevant = binding.bindings_for(stem);

    if relevant.is_empty() {
        probes.push(ScoreProbe {
            dimension: "binding".into(),
            probe: stem.into(),
            outcome: false,
            detail: "(no bindings for this contract)".into(),
        });
        return 0.0;
    }

    // Emit a probe per binding entry
    for b in &relevant {
        let status_str = match b.status {
            ImplStatus::Implemented => "implemented",
            ImplStatus::Partial => "partial",
            ImplStatus::NotImplemented => "not_implemented",
            ImplStatus::Pending => "pending",
        };
        let is_implemented = b.status == ImplStatus::Implemented;
        let fn_name = b
            .function
            .as_deref()
            .or(b.module_path.as_deref())
            .unwrap_or("?");
        probes.push(ScoreProbe {
            dimension: "binding".into(),
            probe: b.equation.clone(),
            outcome: is_implemented,
            detail: format!("{fn_name} ({status_str})"),
        });
    }

    let implemented: f64 = relevant
        .iter()
        .map(|b| match b.status {
            ImplStatus::Implemented => 1.0,
            ImplStatus::Partial => 0.5,
            ImplStatus::NotImplemented | ImplStatus::Pending => 0.0,
        })
        .sum();

    implemented / relevant.len() as f64
}

#[cfg(test)]
mod tests {
    include!("scoring_tests.rs");
}
