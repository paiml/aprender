//! ONT-001 §5 ONT-4c5 — capability cells: a required cell of the model-capability ladder is oracle-proven or
//! RED. `Unknown{NotRun}` on a required cell never arms.
//!
//! **Domain D** — the cells ONT-4c1's `ladder-green` quantifies over, restricted to `required: true` rungs:
//! `{ "<rung>@<host>" : rung.required, host ∈ receipts::expected_hosts(rung, receipts::receipt_hosts(all)) }`.
//! One definition: the host axis is ladder-green's own function, not a copy.
//!
//! **Cells are measured at the current release, V\*, only.** V\* is the greatest `version` over every receipt
//! `receipts::read_all` read; a version that does not parse as dotted numerals is refused (the declaration's
//! fault, never guessed). A cell is measured ONLY by a witness row (its `sha256` equals the rung's) in a V\*
//! receipt for that host. Older versions are never consulted: every required rung carries the same sha across
//! 0.68.1 / 0.68.2 / 0.69.1, so ladder-green's cross-version "∃ witness row" would let a 0.68.2 green row hide a
//! 0.69.1 DEFER or a missing 0.69.1 row (plan grill round 1, measured on this tree).
//!
//! **Per row, a label beats the measurement.** A row label (`verdict`, `label`) that `Verdict::from_label` maps to
//! `Unknown(_)` — DEFER, MANUAL, NO-VERDICT (`Unknown(NotRun)`), SKIP, REPORT, WARN (their own reason) — puts the
//! cell in `not_run` whatever its `green` says; a label
//! that maps to `Fail` makes it Fail; `Pass` labels never upgrade a row (the measurement decides); a label
//! `from_label` does not know is REFUSED by name and the row is NotRun. Otherwise Pass = `receipts::row_is_green`
//! and Fail = its negation. **Several V\* rows for one cell** combine: any NotRun → NotRun; else any Fail → Fail;
//! else Pass — a green duplicate never hides a DEFER or a Fail.
//!
//! **The set difference is the validator's.** `not_run = D \ { cells measured Pass or Fail }`. An absent cell is
//! NotRun, never folded into Fail: this row ADMITS Fail (ladder-green is what refuses it), so a folded absence
//! would pass. [`apply`] writes, on the rung's `model:Model` node, `model:capabilityCell "<rung>@<host>=<v>"` per
//! found cell (the extractor's part: only what it finds), and `model:notRunCell "<rung>@<host>"` per member of
//! `not_run` and `model:refusedLabel "<file>: <rung>: <label>"` per refused label (the validator's part). The
//! shape `capability-cells` says `maxCount 0` on both. Corpus and positive control call the SAME [`apply`].

use std::collections::{BTreeMap, BTreeSet};

use crate::ontology::extract::gguf::{model, Rung};
use crate::ontology::rdf::{iri, Graph, Term};
use crate::ontology::receipts::{self, Receipt, Row};
use crate::ontology::verdict::Verdict;

/// The cell id: `<rung id>@<host>` — the spelling the ONT-4c5 probe's baseline set B uses.
#[must_use]
pub fn cell_id(rung: &str, host: &str) -> String {
    format!("{rung}@{host}")
}

/// What one computation found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityCells {
    /// The release the cells were measured at; `None` when no receipt was read.
    pub v_star: Option<String>,
    /// D: every required cell, sorted.
    pub domain: BTreeSet<String>,
    /// The cells a V\* witness row measured, with their verdict (Pass / Fail / `Unknown(NotRun)`).
    pub cells: BTreeMap<String, Verdict>,
    /// `D \ { Pass | Fail cells }`, sorted. Every member is a violation of the armed shape.
    pub not_run: BTreeSet<String>,
    /// `<file>: <rung>: <label>` for every row label `Verdict::from_label` does not know.
    pub refused_labels: Vec<String>,
}

/// What the domain cannot be built from: the declaration's fault (exit 3), never guessed and never a verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellsError {
    /// A receipt `version` that is not dotted numerals, so V* cannot be chosen.
    Version { file: String, version: String },
    /// Two required rungs share an `id`. A cell is `<rung>@<host>`, so their cells would merge into one and one
    /// rung's gap would be written onto the other's node (review lane B). ladder-green keys by sha256 and never
    /// merges; this row refuses the declaration instead of guessing which rung a cell belongs to.
    DuplicateRung { id: String, contracts: Vec<String> },
}

impl CellsError {
    /// The file to name in the refusal.
    #[must_use]
    pub fn file(&self) -> String {
        match self {
            Self::Version { file, .. } => file.clone(),
            Self::DuplicateRung { contracts, .. } => contracts.join(", "),
        }
    }
}

impl std::fmt::Display for CellsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Version { file, version } => write!(
                f,
                "{file}: receipt version {version:?} is not dotted numerals, so the current release V* cannot be chosen — refused by name"
            ),
            Self::DuplicateRung { id, contracts } => write!(
                f,
                "required rung id {id:?} is declared more than once ({}) — a capability cell is <rung>@<host>, so the two would merge; refused by name",
                contracts.join(", ")
            ),
        }
    }
}

impl std::error::Error for CellsError {}

/// `"0.69.1"` → `[0, 69, 1]`; `None` for anything that is not one or more dot-separated decimal numerals.
fn version_key(v: &str) -> Option<Vec<u64>> {
    if v.is_empty() {
        return None;
    }
    v.split('.')
        .map(|p| {
            if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) {
                None
            } else {
                p.parse::<u64>().ok()
            }
        })
        .collect()
}

/// V\*: the greatest version over `receipts`, or the first receipt whose version does not parse.
fn v_star(receipts: &[Receipt]) -> Result<Option<String>, CellsError> {
    let mut best: Option<(Vec<u64>, String)> = None;
    for r in receipts {
        let key = version_key(&r.version).ok_or_else(|| CellsError::Version {
            file: r.file.clone(),
            version: r.version.clone(),
        })?;
        if best.as_ref().is_none_or(|(k, _)| key > *k) {
            best = Some((key, r.version.clone()));
        }
    }
    Ok(best.map(|(_, v)| v))
}

/// One row's verdict, label first. Pushes `<file>: <rung>: <label>` for a label nobody knows.
fn row_verdict(rung: &Rung, row: &Row, file: &str, refused: &mut Vec<String>) -> Verdict {
    let mut labelled: Option<Verdict> = None;
    for label in &row.labels {
        let v = match Verdict::from_label(label) {
            Some(Verdict::Pass) => continue,
            Some(v) => v,
            None => {
                refused.push(format!("{file}: {}: {label}", rung.id));
                Verdict::Unknown(crate::ontology::verdict::Reason::NotRun)
            }
        };
        labelled = Some(combine(labelled, v));
    }
    let measured = if receipts::row_is_green(rung, row) {
        Verdict::Pass
    } else {
        Verdict::Fail
    };
    match labelled {
        Some(l) => combine(Some(l), measured),
        None => measured,
    }
}

/// NotRun dominates, then Fail, then Pass. NOT the lattice meet: `Fail ∧ Unknown = Fail` would let a DEFER row
/// beside a failing row be ADMITTED as Fail, and a NotRun cell must never be admitted.
fn combine(acc: Option<Verdict>, v: Verdict) -> Verdict {
    match (acc, v) {
        (Some(Verdict::Unknown(r)), _) | (_, Verdict::Unknown(r)) => Verdict::Unknown(r),
        (Some(Verdict::Fail), _) | (_, Verdict::Fail) => Verdict::Fail,
        _ => Verdict::Pass,
    }
}

/// Two required rungs with one `id`: refused before anything is computed.
fn unique_required_ids(rungs: &[Rung]) -> Result<(), CellsError> {
    let mut seen: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for r in rungs.iter().filter(|r| r.required) {
        seen.entry(r.id.as_str())
            .or_default()
            .push(r.contract.clone());
    }
    match seen.into_iter().find(|(_, c)| c.len() > 1) {
        Some((id, contracts)) => Err(CellsError::DuplicateRung {
            id: id.to_string(),
            contracts,
        }),
        None => Ok(()),
    }
}

/// D, the V\* cells, and their difference. `Err` when the declaration cannot be read as a domain: a receipt
/// version that does not parse, or two required rungs sharing an id.
pub fn compute(rungs: &[Rung], all: &[Receipt]) -> Result<CapabilityCells, CellsError> {
    unique_required_ids(rungs)?;
    let v_star = v_star(all)?;
    let hosts = receipts::receipt_hosts(all);
    let mut out = CapabilityCells {
        v_star: v_star.clone(),
        ..CapabilityCells::default()
    };
    for rung in rungs.iter().filter(|r| r.required) {
        for host in receipts::expected_hosts(rung, &hosts) {
            let id = cell_id(&rung.id, &host);
            out.domain.insert(id.clone());
            let mut cell: Option<Verdict> = None;
            for rec in all
                .iter()
                .filter(|r| Some(&r.version) == v_star.as_ref() && r.host == host)
            {
                for row in rec.rows.iter().filter(|w| {
                    w.id == rung.id && w.sha256.as_deref() == Some(rung.sha256.as_str())
                }) {
                    let v = row_verdict(rung, row, &rec.file, &mut out.refused_labels);
                    cell = Some(combine(cell, v));
                }
            }
            match cell {
                Some(v @ (Verdict::Pass | Verdict::Fail)) => {
                    out.cells.insert(id, v);
                }
                Some(v) => {
                    out.cells.insert(id.clone(), v);
                    out.not_run.insert(id);
                }
                None => {
                    out.not_run.insert(id);
                }
            }
        }
    }
    out.refused_labels.sort();
    out.refused_labels.dedup();
    Ok(out)
}

/// [`compute`], then write its edges onto each rung's `model:Model` node. The ONE entry point: the shapes gate
/// calls it for the corpus and for the positive control alike.
pub fn apply(
    g: &mut Graph,
    rungs: &[Rung],
    all: &[Receipt],
) -> Result<CapabilityCells, CellsError> {
    let cc = compute(rungs, all)?;
    for rung in rungs.iter().filter(|r| r.required) {
        let s = iri("model", &rung.sha256);
        let prefix = format!("{}@", rung.id);
        for (id, v) in cc
            .cells
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
        {
            let word = match v {
                Verdict::Pass => "Pass",
                Verdict::Fail => "Fail",
                Verdict::Unknown(_) => "NotRun",
            };
            g.insert(
                s.clone(),
                model("capabilityCell"),
                Term::string(format!("{id}={word}")),
            );
        }
        for id in cc.not_run.iter().filter(|k| k.starts_with(&prefix)) {
            g.insert(s.clone(), model("notRunCell"), Term::string(id));
        }
        for r in cc
            .refused_labels
            .iter()
            .filter(|r| r.split(": ").nth(1) == Some(rung.id.as_str()))
        {
            g.insert(s.clone(), model("refusedLabel"), Term::string(r));
        }
    }
    Ok(cc)
}

/// How many `model:notRunCell` edges `g` holds — the gate's runtime check that the corpus went through
/// [`apply`] (a count that differs from `|not_run|` is a broken wiring, declined, never graded).
#[must_use]
pub fn not_run_edges(g: &Graph) -> usize {
    let p = model("notRunCell");
    g.iter().filter(|t| t.predicate == p).count()
}

#[cfg(test)]
#[path = "capability_cells_tests.rs"]
mod tests;
