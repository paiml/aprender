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

/// The reasoner's answer. Generic over the variable id so `KANI-ONT-9-1` can run the SAME checker over `u8` ids
/// in bounded `Vec`s; production and the witness file are `String` (the default).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum WitnessResult<I = String> {
    UnsatCore(Core<I>),
    Model(Model<I>),
}

/// An ordered derivation of a conflict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Core<I = String> {
    pub steps: Vec<Step<I>>,
    /// The `contradicts` clause both of whose sides the steps derived.
    pub conflict: (I, I),
}

/// One derivation step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step<I = String> {
    /// A unit clause.
    Unit(I),
    /// `¬A ∨ B`, with `A` derived by an earlier step.
    Implies(I, I),
}

/// A satisfying assignment, as the variables it makes false (the rest are true).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model<I = String> {
    #[serde(rename = "false")]
    pub false_vars: Vec<I>,
}

/// What a certificate that checks proves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Checked<S = BTreeSet<String>> {
    /// The clause set has a model.
    Sat,
    /// The clause set is unsatisfiable; these contracts are the core.
    Unsat { core: S },
}

/// Why a certificate does not check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckError<I = String> {
    EmptyCore,
    NotAUnit(I),
    NoSuchImplication(I, I),
    PremiseNotDerived { premise: I, step: usize },
    NoSuchConflict(I, I),
    ConflictSideNotDerived(I),
    ModelFalsifiesUnit(I),
    ModelFalsifiesImplication(I, I),
    ModelFalsifiesConflict(I, I),
}

impl<I: fmt::Display> fmt::Display for CheckError<I> {
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

impl<I: fmt::Debug + fmt::Display> std::error::Error for CheckError<I> {}

/// The set the checker accumulates variables in. Production uses `BTreeSet`. `KANI-ONT-9-1` uses a fixed-array
/// `witness_small::Bounded` with no heap, because CBMC cannot handle B-tree code or `Vec` pointer checks over
/// symbolic keys in useful time (both measured 2026-09-24).
pub trait IdSet<I>: Default {
    /// Is `x` in the set?
    fn has(&self, x: &I) -> bool;
    /// Add `x` (no-op when present).
    fn put(&mut self, x: I);
}

impl<I: Ord> IdSet<I> for BTreeSet<I> {
    fn has(&self, x: &I) -> bool {
        self.contains(x)
    }
    fn put(&mut self, x: I) {
        self.insert(x);
    }
}

/// What the checker asks of a clause set: membership, and the clauses to test a model against.
pub trait ClauseView<I> {
    /// Is `v` a unit?
    fn has_unit(&self, v: &I) -> bool;
    /// Is `¬a ∨ b` a clause?
    fn has_implies(&self, a: &I, b: &I) -> bool;
    /// Is `¬a ∨ ¬b` a clause (unordered)?
    fn has_conflict(&self, a: &I, b: &I) -> bool;
    /// Every unit.
    fn units<'s>(&'s self) -> impl Iterator<Item = &'s I>
    where
        I: 's;
    /// Every `(A, B)` read as `¬A ∨ B`.
    fn implications<'s>(&'s self) -> impl Iterator<Item = (&'s I, &'s I)>
    where
        I: 's;
    /// Every `(A, B)` read as `¬A ∨ ¬B`.
    fn conflict_pairs<'s>(&'s self) -> impl Iterator<Item = (&'s I, &'s I)>
    where
        I: 's;
}

impl ClauseView<String> for ClauseSet {
    fn has_unit(&self, v: &String) -> bool {
        self.units.contains(v)
    }
    fn has_implies(&self, a: &String, b: &String) -> bool {
        self.implies.contains(&(a.clone(), b.clone()))
    }
    fn has_conflict(&self, a: &String, b: &String) -> bool {
        self.conflicts.contains(&ordered(a, b))
    }
    fn units<'s>(&'s self) -> impl Iterator<Item = &'s String>
    where
        String: 's,
    {
        self.units.iter()
    }
    fn implications<'s>(&'s self) -> impl Iterator<Item = (&'s String, &'s String)>
    where
        String: 's,
    {
        self.implies.iter().map(|(a, b)| (a, b))
    }
    fn conflict_pairs<'s>(&'s self) -> impl Iterator<Item = (&'s String, &'s String)>
    where
        String: 's,
    {
        self.conflicts.iter().map(|(a, b)| (a, b))
    }
}

/// Check a certificate against the clause set it claims to answer.
///
/// # Errors
///
/// The first clause or step the certificate gets wrong.
pub fn check(cs: &ClauseSet, result: &WitnessResult) -> Result<Checked, CheckError> {
    check_with(cs, result)
}

/// [`check`] over any id type, clause view and set: the one implementation both production (`String`,
/// `BTreeSet`) and `KANI-ONT-9-1` (`u8`, `Vec`) run.
///
/// # Errors
///
/// The first clause or step the certificate gets wrong.
pub fn check_with<I, S, C>(cs: &C, result: &WitnessResult<I>) -> Result<Checked<S>, CheckError<I>>
where
    I: Ord + Clone,
    S: IdSet<I>,
    C: ClauseView<I>,
{
    match result {
        WitnessResult::UnsatCore(core) => check_core(cs, core),
        WitnessResult::Model(model) => check_model::<I, S, C>(cs, model).map(|()| Checked::Sat),
    }
}

fn check_core<I, S, C>(cs: &C, core: &Core<I>) -> Result<Checked<S>, CheckError<I>>
where
    I: Ord + Clone,
    S: IdSet<I>,
    C: ClauseView<I>,
{
    if core.steps.is_empty() {
        return Err(CheckError::EmptyCore);
    }
    let mut derived = S::default();
    for (i, step) in core.steps.iter().enumerate() {
        let v = check_step(cs, &derived, i, step)?.clone();
        derived.put(v);
    }
    let (a, b) = &core.conflict;
    if !cs.has_conflict(a, b) {
        return Err(CheckError::NoSuchConflict(a.clone(), b.clone()));
    }
    if let Some(side) = [a, b].into_iter().find(|s| !derived.has(s)) {
        return Err(CheckError::ConflictSideNotDerived(side.clone()));
    }
    Ok(Checked::Unsat { core: derived })
}

/// One derivation step against the graph and what is derived so far; the variable it derives.
fn check_step<'a, I, S, C>(
    cs: &C,
    derived: &S,
    i: usize,
    step: &'a Step<I>,
) -> Result<&'a I, CheckError<I>>
where
    I: Ord + Clone,
    S: IdSet<I>,
    C: ClauseView<I>,
{
    match step {
        Step::Unit(v) if cs.has_unit(v) => Ok(v),
        Step::Unit(v) => Err(CheckError::NotAUnit(v.clone())),
        Step::Implies(a, b) if !cs.has_implies(a, b) => {
            Err(CheckError::NoSuchImplication(a.clone(), b.clone()))
        }
        Step::Implies(a, _) if !derived.has(a) => Err(CheckError::PremiseNotDerived {
            premise: a.clone(),
            step: i,
        }),
        Step::Implies(_, b) => Ok(b),
    }
}

fn check_model<I, S, C>(cs: &C, model: &Model<I>) -> Result<(), CheckError<I>>
where
    I: Ord + Clone,
    S: IdSet<I>,
    C: ClauseView<I>,
{
    let mut f = S::default();
    for v in &model.false_vars {
        f.put(v.clone());
    }
    if let Some(u) = cs.units().find(|u| f.has(u)) {
        return Err(CheckError::ModelFalsifiesUnit(u.clone()));
    }
    if let Some((a, b)) = cs.implications().find(|(a, b)| !f.has(a) && f.has(b)) {
        return Err(CheckError::ModelFalsifiesImplication(a.clone(), b.clone()));
    }
    if let Some((a, b)) = cs.conflict_pairs().find(|(a, b)| !f.has(a) && !f.has(b)) {
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

/// ONT-001 §5 ONT-9, F-12 — planted-solution generation: every graph's answer is fixed by how it was BUILT, and
/// the checker's verdict must equal the construction. A consistent-by-construction graph's planted model checks
/// and no core does; a planted contradiction's derivation checks and no model does. A brute-force oracle over all
/// 2ⁿ assignments checks the construction itself first, so a generator bug cannot pass as a checker property.
#[cfg(test)]
mod planted {
    use super::super::witness_planted::{
        adversarial_cores, adversarial_models, consistent, contradicted, satisfiable, Rng,
    };
    use super::{check, Checked};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn ont_planted(seed in any::<u64>()) {
            let mut rng = Rng::new(seed);

            let sat = consistent(&mut rng);
            prop_assert!(satisfiable(&sat.clauses, sat.n), "the generator built an unsatisfiable 'consistent' graph");
            prop_assert_eq!(check(&sat.clauses, &sat.planted), Ok(Checked::Sat));
            for c in adversarial_cores(&mut rng, &sat.clauses, sat.n) {
                let v = check(&sat.clauses, &c);
                prop_assert!(v.is_err(), "a consistent-by-construction graph got a core: {:?} for {:?} over {:?}", v, c, sat.clauses);
            }

            let unsat = contradicted(&mut rng);
            prop_assert!(!satisfiable(&unsat.clauses, unsat.n), "the generator built a satisfiable 'contradiction'");
            prop_assert!(
                matches!(check(&unsat.clauses, &unsat.planted), Ok(Checked::Unsat { .. })),
                "the planted derivation did not check: {:?}", check(&unsat.clauses, &unsat.planted)
            );
            for m in adversarial_models(&mut rng, unsat.n) {
                let v = check(&unsat.clauses, &m);
                prop_assert!(v.is_err(), "a planted contradiction got a model: {:?} for {:?} over {:?}", v, m, unsat.clauses);
            }
        }
    }
}

/// ONT-001 §5 ONT-9 — **KANI-ONT-9-1**: a certificate the checker accepts is true under relation semantics. Over
/// every graph on three variables (every unit, implication and conflict clause present or absent), every core of
/// one to [`CORE_BOUND`](super::witness_small::CORE_BOUND) steps and every model: `check_with` returning
/// `Unsat` ⇒ no assignment satisfies the graph; `Sat` ⇒ one does. The oracle reads the clause flags, not the
/// checker's collections (`witness_small.rs`). The harness runs [`check_with`] — the one implementation production
/// runs — instantiated at `u8` ids in fixed-capacity arrays. Two runs on 2026-09-24 hit their time caps without a verdict:
/// the `String`/`BTreeSet` instantiation at 2400 s, and a `u8`/`Vec` one at 1800 s (18 046 checks). The L2 twin
/// `witness_small::tests::kani_ont_9_1_twin` asserts both instantiations give identical verdicts.
#[cfg(kani)]
mod kani_proofs {
    use super::super::witness_small::{
        accepted_verdict_is_semantic_u8, core_as, id_u8, model_as, SmallGraph, CORE_BOUND,
        STEP_CODES,
    };

    /// KANI-ONT-9-1 (cores of exactly `LEN` steps): a core the checker accepts is contradictory under relation
    /// semantics and names only known variables, for every graph, every `LEN`-step core and every conflict pair.
    /// `LEN` is concrete per harness so the certificate's `Vec` has a concrete size. A symbolic length over all three
    /// was cut off at 1800 s (25 298 checks, measured 2026-09-24).
    fn cores_of_len<const LEN: usize>() {
        let g = any_graph();
        let mut codes = [0u8; LEN];
        for c in &mut codes {
            let k: u8 = kani::any();
            kani::assume(k < STEP_CODES);
            *c = k;
        }
        let conflict: usize = kani::any();
        kani::assume(conflict < 3);
        assert!(accepted_verdict_is_semantic_u8(
            &g,
            &core_as(&codes, conflict, id_u8)
        ));
    }

    /// KANI-ONT-9-1, cores of CORE_BOUND = 3 steps (the full bound). Unwind 9: the longest loop is the oracle's 8
    /// assignments (+1 exit); clause lists are ≤ 6, cores ≤ 3.
    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::solver(cadical)]
    fn kani_ont_9_1() {
        cores_of_len::<CORE_BOUND>();
    }

    /// KANI-ONT-9-1, cores of 1 step.
    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::solver(cadical)]
    fn kani_ont_9_1_len1() {
        cores_of_len::<1>();
    }

    /// KANI-ONT-9-1, cores of 2 steps.
    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::solver(cadical)]
    fn kani_ont_9_1_len2() {
        cores_of_len::<2>();
    }

    /// KANI-ONT-9-1 (models): a model the checker accepts satisfies the graph, for every graph and every
    /// assignment. A separate harness from the cores so each symex stays small.
    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::solver(cadical)]
    fn kani_ont_9_1_model() {
        let g = any_graph();
        let false_bits: u8 = kani::any();
        kani::assume(false_bits < 8);
        assert!(accepted_verdict_is_semantic_u8(
            &g,
            &model_as(false_bits, id_u8)
        ));
    }

    fn any_graph() -> SmallGraph {
        SmallGraph {
            units: kani::any(),
            implies: kani::any(),
            conflicts: kani::any(),
        }
    }
}
