//! ONT-001 §5 ONT-5 — the consistency witness: the graph pv-sat reasons over, and the checker that re-verifies
//! what it wrote.
//!
//! **The reasoner is not here.** `pv-sat` (`crates/aprender-contracts-cli/src/bin/pv-sat/`) is private to its bin
//! target and depends on this library, never the reverse (F-7): the library holds the graph, the two digests that
//! name a witness, the witness's shape, and a checker small enough to read in one sitting. A reasoner that lies
//! is caught here in O(n), whatever it did to reach its answer.
//!
//! **The encoding is propositional Horn.** One variable per contract id, true when the contract is in force:
//!
//! | relation | clause |
//! |---|---|
//! | every id no contract `supersedes` | `A` (a unit: a live contract is asserted) |
//! | `A depends_on B`, `A refines B` | `¬A ∨ B` (in force only if what it rests on is) |
//! | `A contradicts B` | `¬A ∨ ¬B` (never both in force) |
//!
//! A superseded contract is not asserted, and it is not forbidden either: something live that depends on it drags
//! it back in, and a contradiction it carries then counts. Horn-SAT is decided by unit propagation, so both
//! answers have a linear certificate:
//!
//! - **`unsat_core`** — an ordered derivation: units, then implications whose premise an earlier step derived,
//!   closed by a `contradicts` clause whose two sides both were. Every clause it names must exist in the CURRENT
//!   graph, so a stale or invented core is refused even when it is internally consistent.
//! - **`model`** — the variables assigned false. Every clause is evaluated against it.
//!
//! **A witness is named by what it witnesses** — `witness/<relations_sha256>.json`, and it records
//! `census_id_set_sha256` too, because the units come from the id set. The gate recomputes both; a mismatch is
//! `Unknown{WitnessStale}`, never a verdict about a graph nobody reasoned over.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Roles encoded as `¬A ∨ B`.
pub const IMPLYING_ROLES: [&str; 2] = ["depends_on", "refines"];
/// The role encoded as `¬A ∨ ¬B`.
pub const CONFLICT_ROLE: &str = "contradicts";
/// The role that withdraws its target's unit.
pub const RETIRING_ROLE: &str = "supersedes";

/// `pc_reasoner` / `pc_checker` value when the control drew the answer it must.
pub const FIRED: &str = "fired";

/// One typed edge, as the `relations` gate reads it (symmetric roles already closed).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypedEdge {
    pub from: String,
    pub role: String,
    pub to: String,
}

/// The Horn clause set a corpus denotes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClauseSet {
    /// Asserted variables.
    pub units: BTreeSet<String>,
    /// `(A, B)` is `¬A ∨ B`.
    pub implies: BTreeSet<(String, String)>,
    /// `(A, B)` with `A < B` is `¬A ∨ ¬B`.
    pub conflicts: BTreeSet<(String, String)>,
}

impl ClauseSet {
    /// The encoding in the module docs. Edges through a role it does not encode are left out, and counted by
    /// [`unencoded_edges`].
    #[must_use]
    pub fn from_graph(ids: &BTreeSet<String>, edges: &BTreeSet<TypedEdge>) -> Self {
        let retired: BTreeSet<&str> = edges
            .iter()
            .filter(|e| e.role == RETIRING_ROLE)
            .map(|e| e.to.as_str())
            .collect();
        let mut cs = Self {
            units: ids
                .iter()
                .filter(|id| !retired.contains(id.as_str()))
                .cloned()
                .collect(),
            ..Self::default()
        };
        for e in edges {
            if IMPLYING_ROLES.contains(&e.role.as_str()) {
                cs.implies.insert((e.from.clone(), e.to.clone()));
            } else if e.role == CONFLICT_ROLE {
                cs.conflicts.insert(ordered(&e.from, &e.to));
            }
        }
        cs
    }

    /// The clauses a typed relation produced — the gate's denominator. Units are not counted: every corpus has
    /// them, so they cannot make a graph worth reasoning over.
    #[must_use]
    pub fn checkable_n(&self) -> usize {
        self.implies.len() + self.conflicts.len()
    }
}

/// Edges through a role the encoding does not name (none, while Σ declares exactly the four above).
#[must_use]
pub fn unencoded_edges(edges: &BTreeSet<TypedEdge>) -> usize {
    edges
        .iter()
        .filter(|e| {
            !IMPLYING_ROLES.contains(&e.role.as_str())
                && e.role != CONFLICT_ROLE
                && e.role != RETIRING_ROLE
        })
        .count()
}

fn ordered(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

fn sha256_lines<I: IntoIterator<Item = String>>(lines: I) -> String {
    let mut h = Sha256::new();
    for l in lines {
        h.update(l.as_bytes());
        h.update(b"\n");
    }
    format!("{:x}", h.finalize())
}

/// sha256 over the sorted contract ids, one per line.
#[must_use]
pub fn census_id_set_sha256(ids: &BTreeSet<String>) -> String {
    sha256_lines(ids.iter().cloned())
}

/// sha256 over the sorted typed edges, `from\trole\tto` one per line.
#[must_use]
pub fn relations_sha256(edges: &BTreeSet<TypedEdge>) -> String {
    sha256_lines(
        edges
            .iter()
            .map(|e| format!("{}\t{}\t{}", e.from, e.role, e.to)),
    )
}

/// Where the witness for a relation set lives.
#[must_use]
pub fn witness_path(contract_dir: &Path, relations_sha: &str) -> PathBuf {
    contract_dir
        .join("witness")
        .join(format!("{relations_sha}.json"))
}

/// `contracts/witness/<sha>.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Witness {
    pub census_id_set_sha256: String,
    pub relations_sha256: String,
    /// `git rev-parse HEAD` where pv-sat ran, when it could ask.
    pub reasoner_git_sha: Option<String>,
    pub checkable_n: usize,
    pub result: WitnessResult,
    /// `fired` — pv-sat's plant `A⇒B, B⇒C, contradicts(A,C)` yielded the core `[A, B, C]` before it wrote this.
    pub pc_reasoner: String,
    pub cpu_ms: u64,
}

/// The reasoner's answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum WitnessResult {
    UnsatCore(Core),
    Model(Model),
}

/// An ordered derivation of a conflict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Core {
    pub steps: Vec<Step>,
    /// The `contradicts` clause both of whose sides the steps derived.
    pub conflict: (String, String),
}

/// One derivation step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    /// A unit clause.
    Unit(String),
    /// `¬A ∨ B`, with `A` derived by an earlier step.
    Implies(String, String),
}

/// A satisfying assignment, as the variables it makes false (the rest are true).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    #[serde(rename = "false")]
    pub false_vars: Vec<String>,
}

/// What a certificate that checks proves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Checked {
    /// The clause set has a model.
    Sat,
    /// The clause set is unsatisfiable; these contracts are the core.
    Unsat { core: BTreeSet<String> },
}

/// Why a certificate does not check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckError {
    EmptyCore,
    NotAUnit(String),
    NoSuchImplication(String, String),
    PremiseNotDerived { premise: String, step: usize },
    NoSuchConflict(String, String),
    ConflictSideNotDerived(String),
    ModelFalsifiesUnit(String),
    ModelFalsifiesImplication(String, String),
    ModelFalsifiesConflict(String, String),
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCore => write!(f, "the core derives nothing"),
            Self::NotAUnit(v) => write!(
                f,
                "step asserts `{v}`, which is not a unit of the current graph"
            ),
            Self::NoSuchImplication(a, b) => {
                write!(
                    f,
                    "step uses `{a} ⇒ {b}`, which is not a clause of the current graph"
                )
            }
            Self::PremiseNotDerived { premise, step } => {
                write!(
                    f,
                    "step {step} uses `{premise}` before any earlier step derived it"
                )
            }
            Self::NoSuchConflict(a, b) => {
                write!(
                    f,
                    "the conflict `{a} ⊥ {b}` is not a clause of the current graph"
                )
            }
            Self::ConflictSideNotDerived(v) => {
                write!(f, "the conflict names `{v}`, which the core never derived")
            }
            Self::ModelFalsifiesUnit(v) => write!(f, "the model makes the unit `{v}` false"),
            Self::ModelFalsifiesImplication(a, b) => {
                write!(
                    f,
                    "the model makes `{a}` true and `{b}` false, against `{a} ⇒ {b}`"
                )
            }
            Self::ModelFalsifiesConflict(a, b) => {
                write!(
                    f,
                    "the model makes both `{a}` and `{b}` true, against `{a} ⊥ {b}`"
                )
            }
        }
    }
}

impl std::error::Error for CheckError {}

/// Check a certificate against the clause set it claims to answer.
///
/// # Errors
///
/// The first clause or step the certificate gets wrong.
pub fn check(cs: &ClauseSet, result: &WitnessResult) -> Result<Checked, CheckError> {
    match result {
        WitnessResult::UnsatCore(core) => check_core(cs, core),
        WitnessResult::Model(model) => check_model(cs, model).map(|()| Checked::Sat),
    }
}

fn check_core(cs: &ClauseSet, core: &Core) -> Result<Checked, CheckError> {
    if core.steps.is_empty() {
        return Err(CheckError::EmptyCore);
    }
    let mut derived: BTreeSet<&str> = BTreeSet::new();
    for (i, step) in core.steps.iter().enumerate() {
        derived.insert(check_step(cs, &derived, i, step)?);
    }
    let (a, b) = &core.conflict;
    if !cs.conflicts.contains(&ordered(a, b)) {
        return Err(CheckError::NoSuchConflict(a.clone(), b.clone()));
    }
    if let Some(side) = [a, b].into_iter().find(|s| !derived.contains(s.as_str())) {
        return Err(CheckError::ConflictSideNotDerived(side.clone()));
    }
    Ok(Checked::Unsat {
        core: derived.into_iter().map(str::to_string).collect(),
    })
}

/// One derivation step against the graph and what is derived so far; the variable it derives.
fn check_step<'a>(
    cs: &ClauseSet,
    derived: &BTreeSet<&str>,
    i: usize,
    step: &'a Step,
) -> Result<&'a str, CheckError> {
    match step {
        Step::Unit(v) if cs.units.contains(v) => Ok(v),
        Step::Unit(v) => Err(CheckError::NotAUnit(v.clone())),
        Step::Implies(a, b) if !cs.implies.contains(&(a.clone(), b.clone())) => {
            Err(CheckError::NoSuchImplication(a.clone(), b.clone()))
        }
        Step::Implies(a, _) if !derived.contains(a.as_str()) => {
            Err(CheckError::PremiseNotDerived {
                premise: a.clone(),
                step: i,
            })
        }
        Step::Implies(_, b) => Ok(b),
    }
}

fn check_model(cs: &ClauseSet, model: &Model) -> Result<(), CheckError> {
    let f: BTreeSet<&str> = model.false_vars.iter().map(String::as_str).collect();
    if let Some(u) = cs.units.iter().find(|u| f.contains(u.as_str())) {
        return Err(CheckError::ModelFalsifiesUnit(u.clone()));
    }
    if let Some((a, b)) = cs
        .implies
        .iter()
        .find(|(a, b)| !f.contains(a.as_str()) && f.contains(b.as_str()))
    {
        return Err(CheckError::ModelFalsifiesImplication(a.clone(), b.clone()));
    }
    if let Some((a, b)) = cs
        .conflicts
        .iter()
        .find(|(a, b)| !f.contains(a.as_str()) && !f.contains(b.as_str()))
    {
        return Err(CheckError::ModelFalsifiesConflict(a.clone(), b.clone()));
    }
    Ok(())
}

/// A clause set and a certificate for it, as a fixture file holds them (the checker's positive control).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificateFixture {
    pub clauses: ClauseSet,
    pub result: WitnessResult,
}

/// The corrupt core the gate feeds its checker on every run: a derivation that uses `B ⇒ C` before anything
/// derived `B`. A checker that accepts it would accept any core.
pub const CORRUPT_CORE_FIXTURE: &str = include_str!("../../fixtures/unsat-core-corrupt.json");

/// `pc_checker`: `Ok(FIRED)` when the checker refuses [`CORRUPT_CORE_FIXTURE`], and `Err` naming what went wrong
/// otherwise.
///
/// # Errors
///
/// The fixture does not parse, or the checker accepted it.
pub fn pc_checker() -> Result<&'static str, String> {
    let fx: CertificateFixture = serde_json::from_str(CORRUPT_CORE_FIXTURE)
        .map_err(|e| format!("the corrupt-core fixture does not parse: {e}"))?;
    match check(&fx.clauses, &fx.result) {
        Err(_) => Ok(FIRED),
        Ok(c) => Err(format!(
            "the checker accepted the corrupt-core fixture as {c:?}"
        )),
    }
}

/// Every variable any clause names.
#[must_use]
pub fn variables(cs: &ClauseSet) -> BTreeSet<String> {
    let mut v: BTreeSet<String> = cs.units.clone();
    for (a, b) in cs.implies.iter().chain(cs.conflicts.iter()) {
        v.insert(a.clone());
        v.insert(b.clone());
    }
    v
}

/// `implies` as an adjacency list, for a reasoner's propagation.
#[must_use]
pub fn implication_adjacency(cs: &ClauseSet) -> BTreeMap<&str, Vec<&str>> {
    let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (a, b) in &cs.implies {
        adj.entry(a.as_str()).or_default().push(b.as_str());
    }
    adj
}

#[cfg(test)]
#[path = "witness_tests.rs"]
mod tests;
