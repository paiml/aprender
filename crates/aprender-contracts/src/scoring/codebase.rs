//! Codebase scoring — how well a consumer project uses contracts.

use std::collections::{BTreeSet, HashMap};

use crate::binding::{normalize_contract_id, BindingRegistry, ImplStatus};
use crate::schema::{Contract, LeanStatus};

use super::score_contract;
use super::types::{CodebaseScore, Grade, ScoringGap};

/// Score a codebase that consumes contracts via a binding registry.
///
/// Five dimensions (weights from spec):
/// - CD1: Contract coverage (30%) — fraction of available contracts that are bound
/// - CD2: Binding completeness (20%) — implemented / total bindings
/// - CD3: Mean contract score (20%) — avg composite of bound contracts
/// - CD4: Proof depth distribution (15%) — weighted L1-L5 distribution
/// - CD5: Drift detection (15%) — via git timestamp comparison
///
/// Optional `pagerank` scores weight gap analysis by dependency importance.
#[allow(clippy::cast_precision_loss)]
pub fn score_codebase(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
) -> CodebaseScore {
    score_codebase_with_pagerank(contracts, binding, None)
}

/// Score a codebase with pagerank-weighted gap analysis.
///
/// `drift_override` provides a pre-computed CD5 drift score (0.0-1.0).
/// Use [`super::drift::compute_drift`] + [`super::drift::detect_stale_contracts`]
/// to compute it from git timestamps. Pass `None` to default to 1.0 (no drift).
#[allow(clippy::cast_precision_loss, clippy::implicit_hasher)]
pub fn score_codebase_with_pagerank(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
    pagerank: Option<&HashMap<String, f64>>,
) -> CodebaseScore {
    score_codebase_full(contracts, binding, pagerank, None)
}

/// Score a codebase with all optional enrichment: pagerank + drift.
#[allow(clippy::cast_precision_loss, clippy::implicit_hasher)]
pub fn score_codebase_full(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
    pagerank: Option<&HashMap<String, f64>>,
    drift_override: Option<f64>,
) -> CodebaseScore {
    let bound_stems: BTreeSet<_> = binding
        .bindings
        .iter()
        .map(|b| b.contract.as_str())
        .collect();

    // CD1: Contract coverage (Option C — fraction of DECLARED contracts covered)
    // Only counts contracts the binding.yaml references, not all 427 equations.
    // A repo that declares 49 bindings and implements all 49 gets 100%.
    let unique_declared: BTreeSet<_> = binding
        .bindings
        .iter()
        .map(|b| normalize_contract_id(&b.contract))
        .collect();
    let declared_count = unique_declared.len();

    let contract_coverage = if declared_count == 0 {
        0.0
    } else {
        // How many declared contracts actually exist in the contract directory?
        let resolved = unique_declared
            .iter()
            .filter(|stem| {
                contracts
                    .iter()
                    .any(|(s, _)| normalize_contract_id(s) == **stem)
            })
            .count();
        resolved as f64 / declared_count as f64
    };

    // CD2: Binding completeness (implementation status of declared bindings)
    let total_bindings = binding.bindings.len();
    let implemented_bindings: f64 = binding
        .bindings
        .iter()
        .map(|b| match b.status {
            ImplStatus::Implemented => 1.0,
            ImplStatus::Partial => 0.5,
            ImplStatus::NotImplemented | ImplStatus::Pending => 0.0,
        })
        .sum();
    let binding_completeness = if total_bindings == 0 {
        0.0
    } else {
        implemented_bindings / total_bindings as f64
    };

    // CD2b: Critical path completeness (Section 28 v4)
    // Developer declares critical functions in binding.yaml critical_path.
    // Score = entries with matching bindings / total declared.
    let critical_path_coverage = if binding.critical_path.is_empty() {
        binding_completeness // fallback: no declaration = use binding completeness
    } else {
        let covered = binding
            .critical_path
            .iter()
            .filter(|cp| {
                binding.bindings.iter().any(|b| {
                    b.function
                        .as_deref()
                        .is_some_and(|f| f.contains(cp.as_str()))
                })
            })
            .count();
        #[allow(clippy::cast_precision_loss)]
        let ratio = covered as f64 / binding.critical_path.len() as f64;
        ratio
    };

    // CD3: Mean contract score of bound contracts
    let bound_scores: Vec<f64> = contracts
        .iter()
        .filter(|(stem, _)| bound_stems.contains(stem.as_str()))
        .map(|(stem, c)| score_contract(c, Some(binding), stem).composite)
        .collect();
    let mean_contract_score = if bound_scores.is_empty() {
        0.0
    } else {
        bound_scores.iter().sum::<f64>() / bound_scores.len() as f64
    };

    // CD4: Proof depth distribution (weighted L1-L5)
    let proof_depth_dist = compute_proof_depth(contracts, &bound_stems);

    // CD5: Drift detection
    let drift = drift_override.unwrap_or(1.0);

    let composite = contract_coverage * 0.25
        + critical_path_coverage * 0.20
        + mean_contract_score * 0.20
        + proof_depth_dist * 0.15
        + drift * 0.20;

    let top_gaps = compute_gaps(contracts, binding, &bound_stems, pagerank);

    CodebaseScore {
        path: "codebase".to_string(),
        contract_coverage,
        binding_completeness: critical_path_coverage,
        mean_contract_score,
        proof_depth_dist,
        drift,
        // Deferred (PMAT-940): reverse_coverage is hardcoded to 0.0 — needs wiring
        // reverse_coverage_report into the scoring pipeline to compute actual value.
        reverse_coverage: 0.0,
        mutation_testing: 1.0,
        ci_pipeline_depth: 1.0,
        proof_freshness: 1.0,
        defect_patterns: 1.0,
        composite,
        grade: Grade::from_score(composite),
        top_gaps,
    }
}

#[allow(clippy::cast_precision_loss)]
fn compute_proof_depth(contracts: &[(String, &Contract)], bound_stems: &BTreeSet<&str>) -> f64 {
    let mut total_obligations = 0usize;
    let mut weighted_sum = 0.0;

    for (stem, contract) in contracts {
        if !bound_stems.contains(stem.as_str()) {
            continue;
        }
        for ob in &contract.proof_obligations {
            total_obligations += 1;
            weighted_sum += 0.1; // L1 (type system)
            if !contract.falsification_tests.is_empty() {
                weighted_sum += 0.3; // L3 (probar)
            }
            if !contract.kani_harnesses.is_empty() {
                weighted_sum += 0.4; // L4 (Kani)
            }
            if ob
                .lean
                .as_ref()
                .is_some_and(|l| l.status == LeanStatus::Proved)
            {
                weighted_sum += 0.2; // L5 (Lean)
            }
        }
    }

    if total_obligations == 0 {
        return 0.0;
    }
    (weighted_sum / total_obligations as f64).min(1.0)
}

/// Compute gaps with impact-weighted scoring per spec Section 4:
/// `impact = (1.0 - obligation_coverage) * dependency_fanout * tier_weight`
///
/// `dependency_fanout` uses pagerank when available, otherwise falls back
/// to reverse dependency count + 1.
#[allow(clippy::cast_precision_loss)]
fn compute_gaps(
    contracts: &[(String, &Contract)],
    binding: &BindingRegistry,
    bound_stems: &BTreeSet<&str>,
    pagerank: Option<&HashMap<String, f64>>,
) -> Vec<ScoringGap> {
    let mut gaps = Vec::new();

    // Pre-compute reverse dependency counts for fallback
    let rev_dep_counts = compute_reverse_dep_counts(contracts);

    for (stem, contract) in contracts {
        if !bound_stems.contains(stem.as_str()) {
            continue;
        }
        let ob_count = contract.proof_obligations.len();
        let kani_count = contract.kani_harnesses.len();
        let ft_count = contract.falsification_tests.len();
        let fanout = dependency_fanout(stem, pagerank, &rev_dep_counts);

        if ob_count > 0 && kani_count < ob_count {
            let coverage = kani_count as f64 / ob_count as f64;
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "kani_coverage".into(),
                current: coverage,
                target: 1.0,
                impact: (1.0 - coverage) * fanout,
                action: "Write #[kani::proof] harnesses".into(),
            });
        }

        if ob_count > 0 && ft_count < ob_count {
            let coverage = ft_count as f64 / ob_count as f64;
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "falsification_coverage".into(),
                current: coverage,
                target: 1.0,
                impact: (1.0 - coverage) * fanout,
                action: "Write probar property tests".into(),
            });
        }

        let partial_count = binding
            .bindings_for(stem)
            .iter()
            .filter(|b| b.status == ImplStatus::Partial)
            .count();
        if partial_count > 0 {
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "binding_partial".into(),
                current: 0.5,
                target: 1.0,
                impact: 0.5 * fanout,
                action: "Complete partial implementations".into(),
            });
        }

        let unimpl_count = binding
            .bindings_for(stem)
            .iter()
            .filter(|b| b.status == ImplStatus::NotImplemented)
            .count();
        if unimpl_count > 0 {
            gaps.push(ScoringGap {
                contract: stem.clone(),
                dimension: "binding_coverage".into(),
                current: 0.0,
                target: 1.0,
                impact: 1.0 * fanout,
                action: "Implement bound equations".into(),
            });
        }
    }

    gaps.sort_by(|a, b| {
        b.impact
            .partial_cmp(&a.impact)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    gaps.truncate(10);
    gaps
}

/// Compute dependency fanout for a contract stem.
///
/// Uses pagerank score when available (normalized to 1.0-10.0 range),
/// otherwise falls back to reverse dependency count + 1.
#[allow(clippy::cast_precision_loss)]
fn dependency_fanout(
    stem: &str,
    pagerank: Option<&HashMap<String, f64>>,
    rev_dep_counts: &HashMap<&str, usize>,
) -> f64 {
    if let Some(pr) = pagerank {
        if let Some(&score) = pr.get(stem) {
            let max_pr = pr.values().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_pr = pr.values().copied().fold(f64::INFINITY, f64::min);
            let range = max_pr - min_pr;
            if range > 1e-12 {
                // Normalize to 1.0-10.0 range for readable impact scores
                return 1.0 + 9.0 * (score - min_pr) / range;
            }
        }
    }
    // Fallback: reverse dep count + 1 (so isolated contracts still have impact 1.0)
    (rev_dep_counts.get(stem).copied().unwrap_or(0) + 1) as f64
}

/// Count how many contracts depend on each stem.
fn compute_reverse_dep_counts<'a>(contracts: &'a [(String, &Contract)]) -> HashMap<&'a str, usize> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, contract) in contracts {
        for dep in &contract.metadata.depends_on {
            // Find the matching stem in our contracts
            for (stem, _) in contracts {
                if stem == dep || stem.strip_suffix(".yaml").is_some_and(|s| s == dep) {
                    *counts.entry(stem.as_str()).or_default() += 1;
                }
            }
        }
    }
    counts
}

#[cfg(test)]
#[path = "codebase_tests.rs"]
mod tests;
