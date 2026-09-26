//! ONT-001 ONT-3b (#4073) — the `refinement` gate: every `models:` entry in `formalization.yaml` names a Rust item
//! the workspace resolves (through ONT-3a's `syn` walk), and the count of theorem-bearing Lean modules with no L4
//! model is held by a shrink-only `unrefined_baseline`. See [`crate::discharge::refinement`] for what a model earns.
//!
//! Rules (`reject:`, exit 1):
//!
//! - PV-ONT-031 — a `model_of` does not resolve: a ghost formalization, named;
//! - PV-ONT-032 — a `models:` entry is malformed (no module/model_of/evidence, unknown `relation.kind`, a module
//!   modelled twice) or names a module the discharge summary does not list;
//! - PV-ONT-033 — `unrefined_baseline` is absent, below the measured count (a new unrefined module), or above it
//!   (stale: lower it, or the next regression hides under the slack).
//!
//! Declines (exit 2): no Lean tree or discharge summary beside the corpus (the fixture corpora), or the resolver's
//! positive control did not fire.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::discharge::{refinement, summary};
use crate::ontology::extract::code::positive_control;
use crate::ontology::verdict::Verdict;

use super::finding::LintFinding;
use super::ratchet_gates::RatchetOutcome;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult};

/// The gate's name, under `--gate` and in a full run.
pub const GATE: &str = "refinement";

/// The Lean tree, relative to the corpus's parent.
pub const LEAN_DIR: &str = "crates/aprender-contracts-staging/lean";

/// The counters `--gate refinement` reports, flattened into [`GateExtra::Refinement`]'s JSON.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RefinementCounters {
    /// Well-formed `models:` entries.
    pub models: usize,
    pub resolved: usize,
    /// Models granting L4 (`extraction`/`simulation`, resolving, in the tree).
    pub l4_models: usize,
    /// Models granting L3 (`test_witnessed`, resolving, in the tree).
    pub l3_models: usize,
    /// `model_of` that does not resolve (PV-ONT-031).
    pub ghosts: usize,
    /// Modules the discharge summary lists with at least one theorem.
    pub theorem_modules: usize,
    /// Of those, the ones no L4 model covers.
    pub unrefined: usize,
    /// `formalization.yaml`'s shrink-only ceiling on `unrefined`; `null` when it declares none.
    pub unrefined_baseline: Option<usize>,
    /// `fired` — the resolver found a present item and refused an absent one this run.
    pub pc_resolver: String,
    pub violations: usize,
}

fn finding(rule: &str, msg: String) -> LintFinding {
    LintFinding::new(
        rule,
        RuleSeverity::Error,
        msg,
        format!("{LEAN_DIR}/formalization.yaml"),
    )
}

/// Run the gate over `contract_dir`.
#[must_use]
pub fn run_refinement_gate(contract_dir: &Path) -> RatchetOutcome {
    let root = crate::ontology::extract::repo_root(contract_dir);
    run_with(&root.join(LEAN_DIR), refinement::workspace_resolver(&root))
}

/// The gate over the Lean tree at `lean_dir`, resolving `model_of` with `resolve`.
pub fn run_with(
    lean_dir: &Path,
    resolve: impl FnMut(&str) -> Result<(), String>,
) -> RatchetOutcome {
    let start = Instant::now();
    if !positive_control() {
        return RatchetOutcome::Declined(
            "positive control pc_resolver did not fire: the resolver cannot tell a present item from an absent one"
                .into(),
        );
    }
    let s = match summary::load(&summary::summary_path(lean_dir)) {
        Ok(s) => s,
        Err(e) => return RatchetOutcome::Declined(format!("no discharge summary to refine: {e}")),
    };
    let rec = match refinement::load(lean_dir) {
        Ok(r) => r,
        Err(e) => return RatchetOutcome::Declined(format!("no formalization.yaml: {e}")),
    };
    let theorem_modules: BTreeMap<String, Vec<String>> = s
        .modules
        .iter()
        .filter(|m| !m.theorems.is_empty())
        .map(|m| (m.path.clone(), m.theorems.clone()))
        .collect();
    let tree: BTreeSet<String> = s.modules.iter().map(|m| m.path.clone()).collect();
    let judged = refinement::judge(&rec.models, &tree, resolve);

    let mut findings: Vec<LintFinding> = rec
        .malformed
        .iter()
        .map(|m| finding("PV-ONT-032", m.clone()))
        .collect();
    let mut c = RefinementCounters {
        models: judged.len(),
        theorem_modules: theorem_modules.len(),
        unrefined_baseline: rec.unrefined_baseline,
        pc_resolver: crate::ontology::witness::FIRED.to_string(),
        ..RefinementCounters::default()
    };
    for j in &judged {
        let m = &j.model;
        match &j.resolved {
            Ok(()) => c.resolved += 1,
            Err(why) => {
                c.ghosts += 1;
                findings.push(finding(
                    "PV-ONT-031",
                    format!(
                        "{}: model_of `{}` does not resolve (ghost formalization): {why}",
                        m.module, m.model_of
                    ),
                ));
            }
        }
        if !j.in_tree {
            findings.push(finding(
                "PV-ONT-032",
                format!("{}: not a module the discharge summary lists", m.module),
            ));
        }
        match j.level() {
            Some(4) => c.l4_models += 1,
            Some(3) => c.l3_models += 1,
            _ => {}
        }
    }
    let l4 = refinement::l4_modules(&judged);
    c.unrefined = refinement::unrefined(&theorem_modules, &l4).len();
    match rec.unrefined_baseline {
        None => findings.push(finding(
            "PV-ONT-033",
            format!(
                "no unrefined_baseline: {} theorem-bearing modules have no L4 model; declare it",
                c.unrefined
            ),
        )),
        Some(b) if c.unrefined > b => findings.push(finding(
            "PV-ONT-033",
            format!(
                "{} modules are unrefined, above unrefined_baseline {b}: a module lost its model or a new one has none",
                c.unrefined
            ),
        )),
        Some(b) if c.unrefined < b => findings.push(finding(
            "PV-ONT-033",
            format!(
                "unrefined_baseline {b} is stale: {} modules are unrefined; lower it (shrink-only)",
                c.unrefined
            ),
        )),
        Some(_) => {}
    }

    c.violations = findings.len();
    let verdict = if findings.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Fail
    };
    let result = GateResult {
        name: GATE.into(),
        passed: verdict == Verdict::Pass,
        skipped: false,
        verdict,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Validate {
            contracts: c.models,
            errors: findings.len(),
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Refinement(Box::new(c))),
    };
    RatchetOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

#[cfg(test)]
#[path = "refinement_gate_tests.rs"]
mod tests;
