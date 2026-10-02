//! ONT-001 §5 ONT-5 — the `ont-consistency` gate: the corpus's typed relations are jointly satisfiable, and the
//! witness that says so checks.
//!
//! The gate does not reason. `pv-sat` reasons, off the lint path, and writes `contracts/witness/<sha>.json`; this
//! gate recomputes the graph, finds the witness named by it, and re-verifies the witness with
//! [`crate::ontology::witness::check`]. Two positive controls stand between a reasoner and a verdict:
//!
//! - **`pc_checker`** — on every run, the checker is fed a corrupt core (`fixtures/unsat-core-corrupt.json`, a
//!   derivation that uses `B ⇒ C` before anything derived `B`) and must refuse it. A checker that accepts it would
//!   accept any core; the gate then has no verdict to give (`Unknown{PositiveControlFailed}`).
//! - **`pc_reasoner`** — pv-sat plants `A⇒B, B⇒C, contradicts(A,C)` before it writes, and records `fired` only
//!   when it drew the core `[A, B, C]`. A witness that does not say `fired` is `Unknown{PositiveControlFailed}`.
//!
//! Rules (`reject:`, exit 1):
//!
//! - PV-ONT-022 — the relations are inconsistent: a checked `unsat_core`, named contract by contract;
//! - PV-ONT-023 — the witness does not check against the current graph (a reasoner that lied, or a hand edit).
//!
//! Declines (exit 2), never a Pass: no Σ; zero typed relation clauses (`Unknown{NoCheckable}`, R-2); no witness, or
//! one whose `census_id_set_sha256` / `relations_sha256` is not the corpus's (`Unknown{WitnessStale}` — run
//! `make contracts`). A Σ that does not parse is an error (exit 3).

use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::ontology::sigma::SigmaError;
use crate::ontology::verdict::Reason;
use crate::ontology::witness::{
    census_id_set_sha256, check, pc_checker, relations_sha256, unencoded_edges, witness_path,
    Checked, ClauseSet, Witness, WitnessResult, FIRED,
};

use super::finding::LintFinding;
use super::relations_gate::{typed_graph, TypedGraph};
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult, Verdict};

/// The gate's name, under `--gate` and in a full run.
pub const GATE: &str = "ont-consistency";

/// What one run answers. Only [`ConsistencyOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum ConsistencyOutcome {
    /// No `ontology.yaml` under the corpus.
    NoSigma,
    /// Σ does not parse, or does not satisfy its own integrity rules.
    Malformed(SigmaError),
    /// Not one typed relation clause to reason over (R-2).
    NoCheckable { contracts_checked: usize },
    /// A positive control did not fire: `which` is `pc_checker` or `pc_reasoner`, `why` says what it did instead.
    PositiveControlFailed { which: &'static str, why: String },
    /// The witness is missing, unreadable, or names a different graph.
    WitnessStale {
        path: String,
        why: String,
        census_id_set_sha256: String,
        relations_sha256: String,
    },
    /// A fresh witness, checked.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// What the report says about the witness it checked (`.witness.pc_reasoner`, `.witness.stale`).
#[derive(Debug, Clone, Serialize)]
pub struct WitnessReport {
    /// Relative to the corpus: `witness/<relations_sha256>.json`.
    pub path: String,
    /// `unsat_core` or `model`.
    pub kind: String,
    /// As pv-sat recorded it.
    pub pc_reasoner: String,
    /// Always `false` in a report: a stale witness never reaches `Ran`.
    pub stale: bool,
    pub census_id_set_sha256: String,
    pub relations_sha256: String,
    pub reasoner_git_sha: Option<String>,
    pub cpu_ms: u64,
}

/// Run the gate over `contract_dir`.
#[must_use]
pub fn run_consistency_gate(contract_dir: &Path) -> ConsistencyOutcome {
    let start = Instant::now();
    let pc = match pc_checker() {
        Ok(fired) => fired,
        Err(why) => {
            return ConsistencyOutcome::PositiveControlFailed {
                which: "pc_checker",
                why,
            }
        }
    };
    let (ids, edges) = match typed_graph(contract_dir) {
        TypedGraph::NoSigma => return ConsistencyOutcome::NoSigma,
        TypedGraph::Malformed(e) => return ConsistencyOutcome::Malformed(e),
        TypedGraph::Read { ids, edges } => (ids, edges),
    };
    let cs = ClauseSet::from_graph(&ids, &edges);
    let checkable_n = cs.checkable_n();
    if checkable_n == 0 {
        return ConsistencyOutcome::NoCheckable {
            contracts_checked: ids.len(),
        };
    }

    let census_sha = census_id_set_sha256(&ids);
    let relations_sha = relations_sha256(&edges);
    let path = witness_path(contract_dir, &relations_sha);
    let shown = format!("witness/{relations_sha}.json");
    let stale = |why: String| ConsistencyOutcome::WitnessStale {
        path: shown.clone(),
        why,
        census_id_set_sha256: census_sha.clone(),
        relations_sha256: relations_sha.clone(),
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return stale(format!("{shown}: {e}")),
    };
    let witness: Witness = match serde_json::from_str(&text) {
        Ok(w) => w,
        Err(e) => return stale(format!("{shown} does not parse as a witness: {e}")),
    };
    if witness.relations_sha256 != relations_sha || witness.census_id_set_sha256 != census_sha {
        return stale(format!(
            "{shown} records census {} / relations {}, and the corpus is census {census_sha} / relations {relations_sha}",
            witness.census_id_set_sha256, witness.relations_sha256
        ));
    }
    if witness.pc_reasoner != FIRED {
        return ConsistencyOutcome::PositiveControlFailed {
            which: "pc_reasoner",
            why: format!("{shown} records pc_reasoner `{}`", witness.pc_reasoner),
        };
    }

    let (findings, core) = judge(&cs, &witness, &shown);
    let passed = findings.is_empty();
    let report = WitnessReport {
        path: shown,
        kind: match witness.result {
            WitnessResult::UnsatCore(_) => "unsat_core".into(),
            WitnessResult::Model(_) => "model".into(),
        },
        pc_reasoner: witness.pc_reasoner.clone(),
        stale: false,
        census_id_set_sha256: census_sha,
        relations_sha256: relations_sha,
        reasoner_git_sha: witness.reasoner_git_sha.clone(),
        cpu_ms: witness.cpu_ms,
    };
    let result = GateResult {
        name: GATE.into(),
        passed,
        skipped: false,
        verdict: Verdict::from_gate(passed, false),
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Validate {
            contracts: ids.len(),
            errors: findings.len(),
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(GateExtra::Consistency {
            checkable_n,
            units: cs.units.len(),
            unencoded_edges: unencoded_edges(&edges),
            pc_checker: pc.to_string(),
            core,
            violations: findings.len(),
            witness: report,
        }),
    };
    ConsistencyOutcome::Ran {
        result: Box::new(result),
        findings,
    }
}

/// The findings a fresh witness earns, and the core when it proves one.
fn judge(cs: &ClauseSet, witness: &Witness, shown: &str) -> (Vec<LintFinding>, Vec<String>) {
    let lie = |why: String| {
        LintFinding::new(
            "PV-ONT-023",
            RuleSeverity::Error,
            format!("{shown} does not check against the current graph: {why}. Re-run `make contracts`; never edit a witness by hand"),
            format!("contracts/{shown}"),
        )
    };
    if witness.checkable_n != cs.checkable_n() {
        return (
            vec![lie(format!(
                "it records checkable_n {} and the graph has {}",
                witness.checkable_n,
                cs.checkable_n()
            ))],
            Vec::new(),
        );
    }
    match check(cs, &witness.result) {
        Ok(Checked::Sat) => (Vec::new(), Vec::new()),
        Ok(Checked::Unsat { core }) => {
            let core: Vec<String> = core.into_iter().collect();
            let (a, b) = match &witness.result {
                WitnessResult::UnsatCore(c) => c.conflict.clone(),
                // `check` answers Unsat only for a core; a model that "proved" one is a lie like any other.
                WitnessResult::Model(_) => {
                    return (vec![lie("a model checked as a core".into())], Vec::new())
                }
            };
            let finding = LintFinding::new(
                "PV-ONT-022",
                RuleSeverity::Error,
                format!(
                    "the typed relations are inconsistent: `{a}` contradicts `{b}`, and the core [{}] puts both in force",
                    core.join(", ")
                ),
                format!("contracts/{a}.yaml"),
            );
            (vec![finding], core)
        }
        Err(e) => (vec![lie(e.to_string())], Vec::new()),
    }
}

/// The reason a non-verdict outcome declines with, for `run_lint`'s report and the CLI's exit.
#[must_use]
pub fn decline_reason(outcome: &ConsistencyOutcome) -> Option<Reason> {
    match outcome {
        ConsistencyOutcome::NoSigma | ConsistencyOutcome::NoCheckable { .. } => {
            Some(Reason::NoCheckable)
        }
        ConsistencyOutcome::PositiveControlFailed { .. } => Some(Reason::PositiveControlFailed),
        ConsistencyOutcome::WitnessStale { .. } => Some(Reason::WitnessStale),
        ConsistencyOutcome::Malformed(_) | ConsistencyOutcome::Ran { .. } => None,
    }
}

/// What a non-verdict outcome could not check, in one line.
#[must_use]
pub fn why(outcome: &ConsistencyOutcome) -> String {
    match outcome {
        ConsistencyOutcome::NoSigma => "no contracts/ontology.yaml".into(),
        ConsistencyOutcome::Malformed(e) => format!("Σ is malformed: {e}"),
        ConsistencyOutcome::NoCheckable { contracts_checked } => format!(
            "no typed relation clause in {contracts_checked} contracts — R-2: zero is a decline"
        ),
        ConsistencyOutcome::PositiveControlFailed { which, why } => {
            format!("positive control {which} did not fire: {why}")
        }
        ConsistencyOutcome::WitnessStale { why, .. } => {
            format!(
                "the witness is stale: {why}. Run `make contracts` to regenerate it with pv-sat"
            )
        }
        ConsistencyOutcome::Ran { .. } => String::new(),
    }
}

#[cfg(test)]
#[path = "consistency_gate_tests.rs"]
mod tests;
