//! ONT-001 §5 ONT-4e — the `refines` gate: every `A refines B` is Liskov (R-20), by a witness pv-sat wrote and this
//! run re-checked with [`crate::ontology::liskov::check`].
//!
//! The gate does not reason. It reads the pairs, classifies them (legacy / prose / checkable), finds the witness
//! named by the checkable pairs' digest, and re-verifies it. `pc_checker` feeds the checker
//! `fixtures/liskov-corrupt.json` (a chain from a premise with a different atom) on every run; a checker that
//! accepts it accepts any implication, and the gate then has no verdict to give.
//!
//! Rules (`reject:`, exit 1):
//!
//! - PV-ONT-024 — a checked Liskov violation, `A refines B: precondition strengthened (PRE-1)` and its siblings;
//! - PV-ONT-025 — the Liskov witness does not check against the current pairs;
//! - PV-ONT-026 — a `requires`/`ensures`/`invariants` clause that is not `{id, statement, formal, formal_status}`;
//! - PV-ONT-027 — `liskov_prose` rose above `ont.liskov_prose` in `lint-baseline.json` (shrink-only).
//!
//! Otherwise: a `prose` clause in any pair is `Unknown{Prose}` naming it; a legacy pair (no `formal_status` on
//! either side) has nothing to check and is counted as such — with no checkable pair the verdict is `Pass` and
//! `liskov_pairs_checked: 0` says why. Declines: no Σ (`NoCheckable`), a missing or foreign witness
//! (`WitnessStale`, run `make contracts`), a control that did not fire (`PositiveControlFailed`). A malformed Σ is
//! exit 3.

use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::ontology::liskov::{
    check, liskov_corpus, liskov_sha256, liskov_witness_path, pc_checker, LiskovWitness, Pair,
    PairClass, LISKOV_WITNESS_DIR,
};
use crate::ontology::sigma::SigmaError;
use crate::ontology::verdict::{Reason, Verdict};
use crate::ontology::witness::FIRED;

use super::finding::LintFinding;
use super::relations_gate::{corpus_documents, typed_graph, TypedGraph};
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult};

/// The gate's name, under `--gate` and in a full run.
pub const GATE: &str = "refines";

/// What one run answers. Only [`RefinesOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum RefinesOutcome {
    /// No `ontology.yaml` under the corpus.
    NoSigma,
    /// Σ does not parse, or does not satisfy its own integrity rules.
    Malformed(SigmaError),
    /// A positive control did not fire: `which` is `pc_checker` or `pc_reasoner`.
    PositiveControlFailed { which: &'static str, why: String },
    /// The Liskov witness is missing, unreadable, or names different pairs.
    WitnessStale { path: String, why: String },
    /// Measured.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// What the report says about the witness it checked.
#[derive(Debug, Clone, Serialize)]
pub struct LiskovWitnessReport {
    /// Relative to the corpus: `witness/liskov/<liskov_sha256>.json`.
    pub path: String,
    pub pc_reasoner: String,
    pub liskov_sha256: String,
    pub reasoner_git_sha: Option<String>,
}

/// The counters `--gate refines` reports, flattened into [`GateExtra::Refines`]'s JSON.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RefinesCounters {
    /// `refines` edges in the typed graph.
    pub refines_pairs: usize,
    /// Pairs whose three obligations were checked through the witness.
    pub liskov_pairs_checked: usize,
    /// Pairs with no `formal_status` clause on either side: nothing to check.
    pub liskov_pairs_legacy: usize,
    /// Pairs left `Unknown{Prose}` by a `prose` clause.
    pub liskov_prose: usize,
    /// `ont.liskov_prose` in `lint-baseline.json`, when recorded.
    pub liskov_prose_baseline: Option<usize>,
    /// The prose clauses, `<contract>.<block> <id>`.
    pub prose_clauses: Vec<String>,
    /// `requires` / `ensures` / new-shape `invariants` clauses across the corpus.
    pub requires_n: usize,
    pub ensures_n: usize,
    pub invariants_n: usize,
    /// `fired` — the checker refused the corrupt Liskov fixture this run.
    pub pc_checker: String,
    /// Findings.
    pub violations: usize,
    /// The witness this run checked; absent when no pair was checkable.
    pub witness: Option<LiskovWitnessReport>,
}

/// Run the gate over `contract_dir`.
#[must_use]
pub fn run_refines_gate(contract_dir: &Path) -> RefinesOutcome {
    let start = Instant::now();
    let pc = match pc_checker() {
        Ok(fired) => fired,
        Err(why) => {
            return RefinesOutcome::PositiveControlFailed {
                which: "pc_checker",
                why,
            }
        }
    };
    let (ids, edges) = match typed_graph(contract_dir) {
        TypedGraph::NoSigma => return RefinesOutcome::NoSigma,
        TypedGraph::Malformed(e) => return RefinesOutcome::Malformed(e),
        TypedGraph::Read { ids, edges } => (ids, edges),
    };
    let corpus = liskov_corpus(&corpus_documents(contract_dir), &edges);

    let mut findings: Vec<LintFinding> = corpus
        .malformed
        .iter()
        .map(|(file, why)| {
            LintFinding::new(
                "PV-ONT-026",
                RuleSeverity::Error,
                format!("clause is not {{id, statement, formal, formal_status}}: {why}"),
                file.clone(),
            )
        })
        .collect();
    let mut c = RefinesCounters {
        refines_pairs: corpus.refines_edges,
        requires_n: corpus.counts.0,
        ensures_n: corpus.counts.1,
        invariants_n: corpus.counts.2,
        pc_checker: pc.to_string(),
        liskov_prose_baseline: baseline_liskov_prose(contract_dir),
        ..RefinesCounters::default()
    };
    let mut checkable: Vec<Pair> = Vec::new();
    for p in corpus.pairs {
        match p.class() {
            PairClass::Legacy => c.liskov_pairs_legacy += 1,
            PairClass::Prose(names) => {
                c.liskov_prose += 1;
                c.prose_clauses.extend(names);
            }
            PairClass::Checkable => checkable.push(p),
        }
    }
    if let Some(baseline) = c.liskov_prose_baseline {
        if c.liskov_prose > baseline {
            findings.push(LintFinding::new(
                "PV-ONT-027",
                RuleSeverity::Error,
                format!(
                    "liskov_prose rose {baseline} -> {}: a refines pair with a prose clause was added ({}). The baseline in contracts/lint-baseline.json is shrink-only",
                    c.liskov_prose,
                    c.prose_clauses.join(", ")
                ),
                "contracts/lint-baseline.json",
            ));
        }
    }

    if !checkable.is_empty() {
        match checked_witness(contract_dir, &checkable) {
            Err(outcome) => return outcome,
            Ok((report, verdicts)) => {
                c.liskov_pairs_checked = checkable.len();
                c.witness = Some(report);
                findings.extend(verdicts);
            }
        }
    }

    c.violations = findings.len();
    let verdict = if !findings.is_empty() {
        Verdict::Fail
    } else if c.liskov_prose > 0 {
        Verdict::Unknown(Reason::Prose)
    } else {
        Verdict::Pass
    };
    let result = GateResult {
        name: GATE.into(),
        passed: verdict == Verdict::Pass,
        skipped: false,
        verdict,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Validate {
            contracts: ids.len(),
            errors: findings.len(),
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Refines(Box::new(c))),
    };
    RefinesOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

/// Find, read and check the witness for `checkable`; the report and the findings it proves, or the decline.
fn checked_witness(
    contract_dir: &Path,
    checkable: &[Pair],
) -> Result<(LiskovWitnessReport, Vec<LintFinding>), RefinesOutcome> {
    let sha = liskov_sha256(checkable);
    let shown = format!("{LISKOV_WITNESS_DIR}/{sha}.json");
    let stale = |why: String| RefinesOutcome::WitnessStale {
        path: shown.clone(),
        why,
    };
    let text = std::fs::read_to_string(liskov_witness_path(contract_dir, &sha))
        .map_err(|e| stale(format!("{shown}: {e}")))?;
    let witness: LiskovWitness = serde_json::from_str(&text)
        .map_err(|e| stale(format!("{shown} does not parse as a Liskov witness: {e}")))?;
    if witness.liskov_sha256 != sha {
        return Err(stale(format!(
            "{shown} records liskov_sha256 {}, and the pairs are {sha}",
            witness.liskov_sha256
        )));
    }
    if witness.pc_reasoner != FIRED {
        return Err(RefinesOutcome::PositiveControlFailed {
            which: "pc_reasoner",
            why: format!("{shown} records pc_reasoner `{}`", witness.pc_reasoner),
        });
    }
    let findings = match check(checkable, &witness) {
        Ok(violations) => violations
            .iter()
            .map(|v| {
                LintFinding::new(
                    "PV-ONT-024",
                    RuleSeverity::Error,
                    v.to_string(),
                    format!("contracts/{}.yaml", v.a),
                )
            })
            .collect(),
        Err(e) => vec![LintFinding::new(
            "PV-ONT-025",
            RuleSeverity::Error,
            format!("{shown} does not check against the current refines pairs: {e}. Re-run `make contracts`; never edit a witness by hand"),
            format!("contracts/{shown}"),
        )],
    };
    let report = LiskovWitnessReport {
        path: shown,
        pc_reasoner: witness.pc_reasoner,
        liskov_sha256: sha,
        reasoner_git_sha: witness.reasoner_git_sha,
    };
    Ok((report, findings))
}

/// `ont.liskov_prose` from `<contract_dir>/lint-baseline.json`, when it is recorded.
fn baseline_liskov_prose(contract_dir: &Path) -> Option<usize> {
    let raw = std::fs::read_to_string(contract_dir.join("lint-baseline.json")).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&raw).ok()?;
    doc.get("ont")?
        .get("liskov_prose")?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
}

/// The reason a non-verdict outcome declines with.
#[must_use]
pub fn decline_reason(outcome: &RefinesOutcome) -> Option<Reason> {
    match outcome {
        RefinesOutcome::NoSigma => Some(Reason::NoCheckable),
        RefinesOutcome::PositiveControlFailed { .. } => Some(Reason::PositiveControlFailed),
        RefinesOutcome::WitnessStale { .. } => Some(Reason::WitnessStale),
        RefinesOutcome::Malformed(_) | RefinesOutcome::Ran { .. } => None,
    }
}

/// What a non-verdict outcome could not check, in one line.
#[must_use]
pub fn why(outcome: &RefinesOutcome) -> String {
    match outcome {
        RefinesOutcome::NoSigma => "no contracts/ontology.yaml".into(),
        RefinesOutcome::Malformed(e) => format!("Σ is malformed: {e}"),
        RefinesOutcome::PositiveControlFailed { which, why } => {
            format!("positive control {which} did not fire: {why}")
        }
        RefinesOutcome::WitnessStale { why, .. } => format!(
            "the Liskov witness is stale: {why}. Run `make contracts` to regenerate it with pv-sat"
        ),
        RefinesOutcome::Ran { .. } => String::new(),
    }
}

/// The lines a `--gate refines` run prints to stderr before its exit: each rejection, or the prose clauses that
/// left the verdict `Unknown{Prose}`.
#[must_use]
pub fn explain(result: &GateResult, findings: &[LintFinding]) -> Vec<String> {
    match result.verdict {
        Verdict::Fail => findings.iter().map(|f| f.message.clone()).collect(),
        Verdict::Unknown(Reason::Prose) => match &result.extra {
            Some(GateExtra::Refines(c)) => vec![format!(
                "refines: Unknown{{Prose}} — prose clause(s), not checked: {}",
                c.prose_clauses.join(", ")
            )],
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

#[cfg(test)]
#[path = "refines_gate_tests.rs"]
mod tests;
