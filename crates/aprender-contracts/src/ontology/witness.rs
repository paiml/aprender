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

/// A clause set whose answer is known by CONSTRUCTION, with the certificate the construction yields (ONT-9, F-12).
///
/// The generator never asks a reasoner or the checker: a satisfiable instance is built around a planted assignment
/// and admits only clauses that assignment satisfies, so the planted assignment is its model; an unsatisfiable one
/// is built around a planted derivation — unit, implication chain, `contradicts` — and every clause added after it
/// only adds constraints, so the derivation stays a core. The checker's verdict on the certificate must equal the
/// construction, and a checker weakened to accept less or more is caught by the instances either side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Planted {
    pub clauses: ClauseSet,
    pub certificate: WitnessResult,
    pub satisfiable: bool,
}

/// splitmix64: a generator the planted corpus can be reproduced from by its seed alone.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n > 0; the modulo bias is irrelevant at these sizes).
    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap_or(0)
    }

    fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

/// The planted instance for `(seed, n, satisfiable)` over the variables `v00..v{n-1}`. `n` is raised to 1 for a
/// satisfiable instance and to 2 for an unsatisfiable one (a conflict needs two sides).
#[must_use]
pub fn ont_planted(seed: u64, n: usize, satisfiable: bool) -> Planted {
    let mut rng = Mix(seed);
    let n = n.max(if satisfiable { 1 } else { 2 });
    let vars: Vec<String> = (0..n).map(|i| format!("v{i:02}")).collect();
    if satisfiable {
        plant_sat(&mut rng, &vars)
    } else {
        plant_unsat(&mut rng, &vars)
    }
}

/// Two distinct variables.
fn pair<'a>(rng: &mut Mix, vars: &'a [String]) -> (&'a String, &'a String) {
    let a = rng.below(vars.len());
    let b = (a + 1 + rng.below(vars.len() - 1)) % vars.len();
    (&vars[a], &vars[b])
}

fn plant_sat(rng: &mut Mix, vars: &[String]) -> Planted {
    let truth: BTreeMap<&str, bool> = vars
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), i == 0 || rng.coin()))
        .collect();
    let mut cs = ClauseSet::default();
    for (v, &t) in &truth {
        if t && (*v == vars[0] || rng.below(3) == 0) {
            cs.units.insert((*v).to_string());
        }
    }
    if vars.len() > 1 {
        for _ in 0..2 * vars.len() {
            let (a, b) = pair(rng, vars);
            if !(truth[a.as_str()] && !truth[b.as_str()]) {
                cs.implies.insert((a.clone(), b.clone()));
            }
        }
        for _ in 0..vars.len() {
            let (a, b) = pair(rng, vars);
            if !(truth[a.as_str()] && truth[b.as_str()]) {
                cs.conflicts.insert(ordered(a, b));
            }
        }
    }
    let false_vars = truth
        .iter()
        .filter(|(_, &t)| !t)
        .map(|(v, _)| (*v).to_string())
        .collect();
    Planted {
        clauses: cs,
        certificate: WitnessResult::Model(Model { false_vars }),
        satisfiable: true,
    }
}

/// A unit and the implication chain it drives through `chain`, as clauses and as derivation steps.
fn plant_chain(cs: &mut ClauseSet, steps: &mut Vec<Step>, chain: &[&String]) {
    cs.units.insert(chain[0].clone());
    steps.push(Step::Unit(chain[0].clone()));
    for w in chain.windows(2) {
        cs.implies.insert((w[0].clone(), w[1].clone()));
        steps.push(Step::Implies(w[0].clone(), w[1].clone()));
    }
}

fn plant_unsat(rng: &mut Mix, vars: &[String]) -> Planted {
    let mut order: Vec<&String> = vars.iter().collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let mut cs = ClauseSet::default();
    let mut steps = Vec::new();
    let conflict = if rng.coin() {
        // One chain; the conflict joins an earlier link to its end (the pc_reasoner plant is the 3-link case).
        let k = 2 + rng.below(order.len() - 1);
        plant_chain(&mut cs, &mut steps, &order[..k]);
        (order[rng.below(k - 1)].clone(), order[k - 1].clone())
    } else {
        // Two disjoint chains, each from its own unit; the conflict joins their ends.
        let k = 1 + rng.below(order.len() - 1);
        let m = 1 + rng.below(order.len() - k);
        plant_chain(&mut cs, &mut steps, &order[..k]);
        plant_chain(&mut cs, &mut steps, &order[k..k + m]);
        (order[k - 1].clone(), order[k + m - 1].clone())
    };
    cs.conflicts.insert(ordered(&conflict.0, &conflict.1));
    for _ in 0..vars.len() {
        let (a, b) = pair(rng, vars);
        match rng.below(3) {
            0 => cs.units.insert(a.clone()),
            1 => cs.implies.insert((a.clone(), b.clone())),
            _ => cs.conflicts.insert(ordered(a, b)),
        };
    }
    Planted {
        clauses: cs,
        certificate: WitnessResult::UnsatCore(Core { steps, conflict }),
        satisfiable: false,
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    const V: [&str; 3] = ["a", "b", "c"];
    /// The six implications and three conflicts over `V`.
    const IMP: [(usize, usize); 6] = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
    const CON: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

    fn s(i: usize) -> String {
        V[i].to_string()
    }

    /// Does the assignment `mask` (bit i = V[i] true) satisfy every clause? Evaluated on the bits, not by `check`.
    fn satisfies(units: u8, imp: u8, con: u8, mask: u8) -> bool {
        let t = |i: usize| mask >> i & 1 == 1;
        (0..3).all(|i| units >> i & 1 == 0 || t(i))
            && IMP
                .iter()
                .enumerate()
                .all(|(k, &(a, b))| imp >> k & 1 == 0 || !t(a) || t(b))
            && CON
                .iter()
                .enumerate()
                .all(|(k, &(a, b))| con >> k & 1 == 0 || !(t(a) && t(b)))
    }

    /// KANI-ONT-9-1: checker soundness at three variables. Any clause set over `V`, any core of at most three steps:
    /// if `check` accepts the core, no assignment satisfies the clause set; if it accepts a model, that model does.
    #[kani::proof]
    #[kani::unwind(10)]
    fn kani_ont_9_1() {
        let (units, imp, con): (u8, u8, u8) = (kani::any(), kani::any(), kani::any());
        kani::assume(units < 8 && imp < 64 && con < 8);
        let cs = ClauseSet {
            units: (0..3).filter(|i| units >> i & 1 == 1).map(s).collect(),
            implies: (0..6)
                .filter(|k| imp >> k & 1 == 1)
                .map(|k| (s(IMP[k].0), s(IMP[k].1)))
                .collect(),
            conflicts: (0..3)
                .filter(|k| con >> k & 1 == 1)
                .map(|k| (s(CON[k].0), s(CON[k].1)))
                .collect(),
        };
        let len: usize = kani::any();
        kani::assume((1..=3).contains(&len));
        let steps: Vec<Step> = (0..len)
            .map(|_| {
                let k: usize = kani::any();
                kani::assume(k < 9);
                if k < 3 {
                    Step::Unit(s(k))
                } else {
                    Step::Implies(s(IMP[k - 3].0), s(IMP[k - 3].1))
                }
            })
            .collect();
        let c: usize = kani::any();
        kani::assume(c < 3);
        let core = WitnessResult::UnsatCore(Core {
            steps,
            conflict: (s(CON[c].0), s(CON[c].1)),
        });
        if check(&cs, &core).is_ok() {
            assert!((0u8..8).all(|m| !satisfies(units, imp, con, m)));
        }
        let mask: u8 = kani::any();
        kani::assume(mask < 8);
        let model = WitnessResult::Model(Model {
            false_vars: (0..3).filter(|i| mask >> i & 1 == 0).map(s).collect(),
        });
        assert_eq!(check(&cs, &model).is_ok(), satisfies(units, imp, con, mask));
    }
}

#[cfg(test)]
#[path = "witness_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "witness_ont9_tests.rs"]
mod ont9_tests;
