//! ONT-001 §5 ONT-9 — the bounded universe `KANI-ONT-9-1` quantifies over, and the relation-semantics oracle it
//! is judged by.
//!
//! **The oracle does not share the checker's code.** A [`SmallGraph`] is three variables and one `bool` per
//! possible clause; satisfiability is decided by evaluating every one of the 2³ assignments against those bools.
//! The checker sees the same graph only through [`SmallGraph::clause_set`] — strings in `BTreeSet`s — so a checker
//! bug and an oracle bug would have to be the same bug written twice in two representations.
//!
//! **Why three variables and cores of at most three steps is the whole claim at this bound.** A step derives one
//! variable; a core whose steps all check derives at most three distinct ones, and its first derivation of each is
//! itself a core that checks. So every accepted core over three variables has an accepted core of ≤ 3 steps with
//! the same derived set — the bound loses nothing *over three variables*. It says nothing about four; that is the
//! stated residual, and `ont_planted` (proptest, up to eight variables) is the L2 evidence beyond it.
//!
//! Compiled under `cfg(test)` (the L2 twin runs it) and `cfg(kani)` (the harness in `witness.rs` runs it); never
//! in a release build.

use std::collections::BTreeSet;

use super::witness::{
    check, check_with, CheckError, Checked, ClauseSet, ClauseView, Core, IdSet, Model, Step,
    WitnessResult,
};

/// The three variables.
pub const VARS: [&str; 3] = ["a", "b", "c"];
/// Every ordered pair of distinct variables — the possible `¬A ∨ B` clauses.
pub const IMPLY_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
/// Every unordered pair — the possible `¬A ∨ ¬B` clauses.
pub const CONFLICT_PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
/// The longest core the harness builds. Measured, not chosen: the pv-sat plant's core is `[A, B, C]`, three
/// steps, and the committed corpus witness is a model (no core at all); three is also |VARS|, which the module
/// docs show is complete at this bound.
pub const CORE_BOUND: usize = 3;
/// Step codes: `0..3` is `Unit(VARS[k])`, `3..9` is `Implies(IMPLY_PAIRS[k - 3])`.
pub const STEP_CODES: u8 = 9;

/// A graph over [`VARS`], one flag per possible clause.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SmallGraph {
    pub units: [bool; 3],
    pub implies: [bool; 6],
    pub conflicts: [bool; 3],
}

impl SmallGraph {
    /// The same graph as the checker reads it.
    #[must_use]
    pub fn clause_set(&self) -> ClauseSet {
        let mut cs = ClauseSet::default();
        for (i, v) in VARS.iter().enumerate() {
            if self.units[i] {
                cs.units.insert((*v).to_string());
            }
        }
        for (k, (a, b)) in IMPLY_PAIRS.iter().enumerate() {
            if self.implies[k] {
                cs.implies
                    .insert((VARS[*a].to_string(), VARS[*b].to_string()));
            }
        }
        for (k, (a, b)) in CONFLICT_PAIRS.iter().enumerate() {
            if self.conflicts[k] {
                cs.conflicts
                    .insert((VARS[*a].to_string(), VARS[*b].to_string()));
            }
        }
        cs
    }

    /// Does `assign` (bit `i` set ⇔ `VARS[i]` true) satisfy every clause? Relation semantics, read off the flags.
    #[must_use]
    pub fn satisfied_by(&self, assign: u8) -> bool {
        let t = |i: usize| assign & (1 << i) != 0;
        let units = (0..3).all(|i| !self.units[i] || t(i));
        let implies = (0..6).all(|k| {
            let (a, b) = IMPLY_PAIRS[k];
            !self.implies[k] || !t(a) || t(b)
        });
        let conflicts = (0..3).all(|k| {
            let (a, b) = CONFLICT_PAIRS[k];
            !self.conflicts[k] || !t(a) || !t(b)
        });
        units && implies && conflicts
    }

    /// Some assignment satisfies the graph.
    #[must_use]
    pub fn satisfiable(&self) -> bool {
        (0u8..8).any(|m| self.satisfied_by(m))
    }
}

/// A duplicate-free set of at most `N` `u8` ids in a fixed array. No heap: with `Vec`s the model half alone
/// generated 18 046 verification conditions, mostly Vec pointer checks, and CaDiCaL did not finish within 30
/// min (measured 2026-09-24). Overflowing `N` panics, so a harness that verifies has also shown that it never does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounded<const N: usize> {
    ids: [u8; N],
    len: usize,
}

impl<const N: usize> Default for Bounded<N> {
    fn default() -> Self {
        Self {
            ids: [0; N],
            len: 0,
        }
    }
}

impl<const N: usize> Bounded<N> {
    /// The members, in insertion order.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.ids[..self.len]
    }

    fn push(&mut self, x: u8) {
        assert!(self.len < N, "Bounded<{N}> overflow");
        self.ids[self.len] = x;
        self.len += 1;
    }
}

impl<const N: usize> IdSet<u8> for Bounded<N> {
    fn has(&self, x: &u8) -> bool {
        self.as_slice().iter().any(|y| y == x)
    }
    fn put(&mut self, x: u8) {
        if !self.has(&x) {
            self.push(x);
        }
    }
}

/// A list of at most `N` id pairs in a fixed array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pairs<const N: usize> {
    pairs: [(u8, u8); N],
    len: usize,
}

impl<const N: usize> Default for Pairs<N> {
    fn default() -> Self {
        Self {
            pairs: [(0, 0); N],
            len: 0,
        }
    }
}

impl<const N: usize> Pairs<N> {
    fn as_slice(&self) -> &[(u8, u8)] {
        &self.pairs[..self.len]
    }

    fn push(&mut self, p: (u8, u8)) {
        assert!(self.len < N, "Pairs<{N}> overflow");
        self.pairs[self.len] = p;
        self.len += 1;
    }
}

/// The same graph as `u8` ids in bounded fixed-capacity lists. This is what `KANI-ONT-9-1` hands the checker.
/// Id `i` is `VARS[i]`; conflicts are stored `(lo, hi)`, as [`ClauseSet::from_graph`] stores them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArrayClauses {
    pub units: Bounded<3>,
    pub implies: Pairs<6>,
    pub conflicts: Pairs<3>,
}

impl ClauseView<u8> for ArrayClauses {
    fn has_unit(&self, v: &u8) -> bool {
        self.units.has(v)
    }
    fn has_implies(&self, a: &u8, b: &u8) -> bool {
        self.implies.as_slice().contains(&(*a, *b))
    }
    fn has_conflict(&self, a: &u8, b: &u8) -> bool {
        let k = (*a.min(b), *a.max(b));
        self.conflicts.as_slice().contains(&k)
    }
    fn units<'s>(&'s self) -> impl Iterator<Item = &'s u8>
    where
        u8: 's,
    {
        self.units.as_slice().iter()
    }
    fn implications<'s>(&'s self) -> impl Iterator<Item = (&'s u8, &'s u8)>
    where
        u8: 's,
    {
        self.implies.as_slice().iter().map(|(a, b)| (a, b))
    }
    fn conflict_pairs<'s>(&'s self) -> impl Iterator<Item = (&'s u8, &'s u8)>
    where
        u8: 's,
    {
        self.conflicts.as_slice().iter().map(|(a, b)| (a, b))
    }
}

impl SmallGraph {
    /// The same graph as `u8` ids in fixed-capacity lists, clauses in the order `clause_set` iterates them.
    #[must_use]
    pub fn array_clauses(&self) -> ArrayClauses {
        let mut ac = ArrayClauses::default();
        for i in 0..3 {
            if self.units[i] {
                ac.units.push(id_u8(i));
            }
        }
        for (k, (a, b)) in IMPLY_PAIRS.iter().enumerate() {
            if self.implies[k] {
                ac.implies.push((id_u8(*a), id_u8(*b)));
            }
        }
        for (k, (a, b)) in CONFLICT_PAIRS.iter().enumerate() {
            if self.conflicts[k] {
                ac.conflicts.push((id_u8(*a), id_u8(*b)));
            }
        }
        ac
    }
}

/// `VARS[i]` as a `String` id.
#[must_use]
pub fn id_string(i: usize) -> String {
    VARS[i].to_string()
}

/// `VARS[i]` as a `u8` id.
#[must_use]
pub fn id_u8(i: usize) -> u8 {
    // i < 3 everywhere this is called.
    u8::try_from(i).unwrap_or(u8::MAX)
}

/// Decode a step code (see [`STEP_CODES`]) with the given id map.
#[must_use]
pub fn step_as<I>(code: u8, id: fn(usize) -> I) -> Step<I> {
    // `checked_sub`, not `k - 3`: with `% 6` after it, `k + 3` indexes the same pair, so a `-` here is an
    // operator no test can pin (an equivalent mutant, #4587).
    match usize::from(code).checked_sub(3) {
        None => Step::Unit(id(usize::from(code))),
        Some(j) => {
            let (a, b) = IMPLY_PAIRS[j % 6];
            Step::Implies(id(a), id(b))
        }
    }
}

/// Decode a step code (see [`STEP_CODES`]).
#[must_use]
pub fn step(code: u8) -> Step {
    step_as(code, id_string)
}

/// A core from step codes and a conflict-pair index, with the given id map.
#[must_use]
pub fn core_as<I>(codes: &[u8], conflict: usize, id: fn(usize) -> I) -> WitnessResult<I> {
    let (a, b) = CONFLICT_PAIRS[conflict % 3];
    let mut steps = Vec::with_capacity(codes.len());
    for c in codes {
        steps.push(step_as(*c, id));
    }
    WitnessResult::UnsatCore(Core {
        steps,
        conflict: (id(a), id(b)),
    })
}

/// A core from step codes and a conflict-pair index.
#[must_use]
pub fn core(codes: &[u8], conflict: usize) -> WitnessResult {
    core_as(codes, conflict, id_string)
}

/// A model from a false-set bitmask, with the given id map.
#[must_use]
pub fn model_as<I>(false_bits: u8, id: fn(usize) -> I) -> WitnessResult<I> {
    let mut false_vars = Vec::with_capacity(3);
    for i in 0..3 {
        if false_bits & (1 << i) != 0 {
            false_vars.push(id(i));
        }
    }
    WitnessResult::Model(Model { false_vars })
}

/// A model from a false-set bitmask.
#[must_use]
pub fn model(false_bits: u8) -> WitnessResult {
    model_as(false_bits, id_string)
}

/// The production instantiation: `String` ids, `BTreeSet`s.
pub fn check_string(g: &SmallGraph, r: &WitnessResult) -> Result<Checked, CheckError> {
    check(&g.clause_set(), r)
}

/// The Kani instantiation: `u8` ids, fixed-capacity lists — the same `check_with`.
pub fn check_u8(
    g: &SmallGraph,
    r: &WitnessResult<u8>,
) -> Result<Checked<Bounded<3>>, CheckError<u8>> {
    check_with(&g.array_clauses(), r)
}

/// The property, given what the checker answered and whether an accepted core names only known variables.
fn semantic<S, E>(
    g: &SmallGraph,
    verdict: &Result<Checked<S>, E>,
    core_ok: fn(&S) -> bool,
) -> bool {
    match verdict {
        Ok(Checked::Unsat { core }) => !g.satisfiable() && core_ok(core),
        Ok(Checked::Sat) => g.satisfiable(),
        Err(_) => true,
    }
}

/// `KANI-ONT-9-1`'s property: whatever the checker ACCEPTS is true under relation semantics — an accepted core
/// means no assignment satisfies the graph, an accepted model means one does, and an accepted core names only
/// variables the graph has. A refusal is always sound (it claims nothing). `String` instantiation; pass it
/// `check_string(g, r)`. It takes the verdict, not the certificate, so a test can hand it an UNSOUND verdict and
/// watch it say no — over a sound checker alone it is always true and could not be told from `true` (#4587).
#[must_use]
pub fn accepted_verdict_is_semantic(g: &SmallGraph, verdict: &Result<Checked, CheckError>) -> bool {
    semantic(g, verdict, core_names_known)
}

/// The same property over the `u8` instantiation — the body `KANI-ONT-9-1` proves, over `check_u8(g, r)`.
#[must_use]
pub fn accepted_verdict_is_semantic_u8(
    g: &SmallGraph,
    verdict: &Result<Checked<Bounded<3>>, CheckError<u8>>,
) -> bool {
    semantic(g, verdict, core_names_known_u8)
}

/// An accepted core is non-empty and names only variables the graph has (`String` ids).
fn core_names_known(core: &BTreeSet<String>) -> bool {
    !core.is_empty() && core.iter().all(|v| VARS.contains(&v.as_str()))
}

/// An accepted core is non-empty and names only variables the graph has (`u8` ids).
fn core_names_known_u8(core: &Bounded<3>) -> bool {
    !core.as_slice().is_empty() && core.as_slice().iter().all(|v| usize::from(*v) < VARS.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The oracle itself, on the plant: `a` asserted, `a⇒b`, `b⇒c`, `a ⊥ c` has no model; drop the conflict and
    /// it has one. An oracle that answers one constant fails one of the two.
    #[test]
    fn the_oracle_decides_the_plant() {
        let mut g = SmallGraph {
            units: [true, false, false],
            ..SmallGraph::default()
        };
        g.implies[0] = true; // a ⇒ b
        g.implies[3] = true; // b ⇒ c
        g.conflicts[1] = true; // a ⊥ c
        assert!(!g.satisfiable());
        assert!(accepted_verdict_is_semantic(
            &g,
            &check_string(&g, &core(&[0, 3, 6], 1))
        ));
        assert!(matches!(
            check(&g.clause_set(), &core(&[0, 3, 6], 1)),
            Ok(Checked::Unsat { .. })
        ));
        g.conflicts[1] = false;
        assert!(g.satisfiable());
        assert!(matches!(
            check(&g.clause_set(), &model(0)),
            Ok(Checked::Sat)
        ));
    }

    /// `semantic` on hand-built verdicts: each arm is false when only one of its conjuncts holds.
    #[test]
    fn semantic_needs_both_halves_of_an_accepted_core() {
        let unsat = SmallGraph {
            units: [true, false, false],
            conflicts: [false; 3],
            implies: [false; 6],
        };
        let mut unsat = unsat;
        unsat.implies[0] = true;
        unsat.implies[3] = true;
        unsat.conflicts[1] = true;
        assert!(!unsat.satisfiable());
        let sat = SmallGraph::default();
        assert!(sat.satisfiable());
        let core = |c: u8| -> Result<Checked<u8>, ()> { Ok(Checked::Unsat { core: c }) };
        assert!(semantic(&unsat, &core(0), |_| true));
        assert!(!semantic(&unsat, &core(0), |_| false));
        assert!(!semantic(&sat, &core(0), |_| true));
        assert!(!semantic(&sat, &core(0), |_| false));
        let ok_sat: Result<Checked<u8>, ()> = Ok(Checked::Sat);
        assert!(semantic(&sat, &ok_sat, |_| false));
        assert!(!semantic(&unsat, &ok_sat, |_| true));
        let refused: Result<Checked<u8>, ()> = Err(());
        assert!(semantic(&unsat, &refused, |_| false));
    }

    /// The property is a check, not a constant: handed an UNSOUND verdict — a core for a satisfiable graph, a
    /// model for an unsatisfiable one, a core that is empty or names a variable the graph lacks — it says no.
    #[test]
    fn accepted_verdict_is_semantic_refuses_an_unsound_verdict() {
        let mut unsat = SmallGraph {
            units: [true, false, false],
            ..SmallGraph::default()
        };
        unsat.implies[0] = true;
        unsat.implies[3] = true;
        unsat.conflicts[1] = true;
        assert!(!unsat.satisfiable());
        let sat = SmallGraph::default();
        let set = |v: &[&str]| {
            v.iter()
                .map(|x| (*x).to_string())
                .collect::<BTreeSet<String>>()
        };
        let unsat_core = |c| -> Result<Checked, CheckError> { Ok(Checked::Unsat { core: c }) };
        assert!(accepted_verdict_is_semantic(
            &unsat,
            &unsat_core(set(&["a"]))
        ));
        assert!(!accepted_verdict_is_semantic(
            &sat,
            &unsat_core(set(&["a"]))
        ));
        assert!(!accepted_verdict_is_semantic(&unsat, &Ok(Checked::Sat)));
        assert!(!accepted_verdict_is_semantic(&unsat, &unsat_core(set(&[]))));
        assert!(!accepted_verdict_is_semantic(
            &unsat,
            &unsat_core(set(&["a", "z"]))
        ));

        let ids = |v: &[u8]| {
            let mut b = Bounded::<3>::default();
            for x in v {
                b.put(*x);
            }
            b
        };
        let unsat_u8 =
            |c| -> Result<Checked<Bounded<3>>, CheckError<u8>> { Ok(Checked::Unsat { core: c }) };
        assert!(accepted_verdict_is_semantic_u8(
            &unsat,
            &unsat_u8(ids(&[0, 2]))
        ));
        assert!(!accepted_verdict_is_semantic_u8(&sat, &unsat_u8(ids(&[0]))));
        assert!(!accepted_verdict_is_semantic_u8(&unsat, &Ok(Checked::Sat)));
        assert!(!accepted_verdict_is_semantic_u8(
            &unsat,
            &unsat_u8(ids(&[]))
        ));
        // id 3 == VARS.len(): one past the last variable
        assert!(!accepted_verdict_is_semantic_u8(
            &unsat,
            &unsat_u8(ids(&[0, 3]))
        ));
    }

    /// Codes `3..9` are the six implications in order; a code past 9 wraps (`% 6`).
    #[test]
    fn step_codes_decode_to_their_implication_pair() {
        for (j, (a, b)) in IMPLY_PAIRS.iter().enumerate() {
            let code = u8::try_from(j + 3).unwrap_or(u8::MAX);
            assert_eq!(step_as(code, id_u8), Step::Implies(id_u8(*a), id_u8(*b)));
        }
        assert_eq!(step_as(2, id_u8), Step::Unit(2));
        assert_eq!(step_as(9, id_u8), step_as(3, id_u8));
    }

    /// Bit `i` of the mask names variable `i`; the decoded steps land on the exact ids.
    #[test]
    fn model_and_step_decode_to_exact_ids() {
        let WitnessResult::Model(m) = model(0b110) else {
            panic!("model");
        };
        assert_eq!(m.false_vars, ["b", "c"]);
        let WitnessResult::Model(m) = model(0b001) else {
            panic!("model");
        };
        assert_eq!(m.false_vars, ["a"]);
        let WitnessResult::Model(m) = model_as(0b101, id_u8) else {
            panic!("model");
        };
        assert_eq!(m.false_vars, [0u8, 2]);
        assert_eq!(step(0), Step::Unit("a".to_string()));
        assert_eq!(step(2), Step::Unit("c".to_string()));
        assert_eq!(step(3), Step::Implies("a".to_string(), "b".to_string()));
        assert_eq!(step(8), Step::Implies("c".to_string(), "b".to_string()));
    }

    /// A `u8` verdict renamed into `String` ids, so the two instantiations can be compared exactly.
    fn relabel(v: Result<Checked<Bounded<3>>, CheckError<u8>>) -> Result<Checked, CheckError> {
        let s = |i: u8| id_string(usize::from(i));
        match v {
            Ok(Checked::Sat) => Ok(Checked::Sat),
            Ok(Checked::Unsat { core }) => Ok(Checked::Unsat {
                core: core.as_slice().iter().map(|v| s(*v)).collect(),
            }),
            Err(e) => Err(match e {
                CheckError::EmptyCore => CheckError::EmptyCore,
                CheckError::NotAUnit(a) => CheckError::NotAUnit(s(a)),
                CheckError::NoSuchImplication(a, b) => CheckError::NoSuchImplication(s(a), s(b)),
                CheckError::PremiseNotDerived { premise, step } => CheckError::PremiseNotDerived {
                    premise: s(premise),
                    step,
                },
                CheckError::NoSuchConflict(a, b) => CheckError::NoSuchConflict(s(a), s(b)),
                CheckError::ConflictSideNotDerived(a) => CheckError::ConflictSideNotDerived(s(a)),
                CheckError::ModelFalsifiesUnit(a) => CheckError::ModelFalsifiesUnit(s(a)),
                CheckError::ModelFalsifiesImplication(a, b) => {
                    CheckError::ModelFalsifiesImplication(s(a), s(b))
                }
                CheckError::ModelFalsifiesConflict(a, b) => {
                    CheckError::ModelFalsifiesConflict(s(a), s(b))
                }
            }),
        }
    }

    /// The two instantiations agree on a hand-picked accept and refusal of each kind, so a relabel that
    /// collapses everything would not pass `kani_ont_9_1_twin` vacuously.
    #[test]
    fn both_instantiations_agree_on_the_plant() {
        let mut g = SmallGraph {
            units: [true, false, false],
            ..SmallGraph::default()
        };
        g.implies[0] = true;
        g.implies[3] = true;
        g.conflicts[1] = true;
        for (codes, conflict) in [(&[0u8, 3, 6][..], 1usize), (&[0, 6][..], 1), (&[1][..], 0)] {
            let s = check_string(&g, &core(codes, conflict));
            assert_eq!(relabel(check_u8(&g, &core_as(codes, conflict, id_u8))), s);
        }
        assert!(matches!(
            check_string(&g, &core(&[0, 3, 6], 1)),
            Ok(Checked::Unsat { .. })
        ));
        assert!(matches!(
            check_string(&g, &core(&[0, 6], 1)),
            Err(CheckError::PremiseNotDerived { .. })
        ));
        assert!(matches!(
            check_string(&g, &core(&[1], 0)),
            Err(CheckError::NotAUnit(_))
        ));
    }

    fn graph() -> impl Strategy<Value = SmallGraph> {
        (any::<[bool; 3]>(), any::<[bool; 6]>(), any::<[bool; 3]>()).prop_map(
            |(units, implies, conflicts)| SmallGraph {
                units,
                implies,
                conflicts,
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2048))]
        /// The L2 twin of `KANI-ONT-9-1`: the same body, sampled rather than exhausted, over BOTH
        /// instantiations — the `u8`/`Vec` one Kani proves and the `String`/`BTreeSet` one production runs — and
        /// the two give the identical verdict (relabelled) on every input.
        #[test]
        fn kani_ont_9_1_twin(
            g in graph(),
            codes in proptest::collection::vec(0u8..STEP_CODES, 1..=CORE_BOUND),
            conflict in 0usize..3,
            false_bits in 0u8..8,
        ) {
            prop_assert!(accepted_verdict_is_semantic(&g, &check_string(&g, &core(&codes, conflict))));
            prop_assert!(accepted_verdict_is_semantic(&g, &check_string(&g, &model(false_bits))));
            prop_assert!(accepted_verdict_is_semantic_u8(&g, &check_u8(&g, &core_as(&codes, conflict, id_u8))));
            prop_assert!(accepted_verdict_is_semantic_u8(&g, &check_u8(&g, &model_as(false_bits, id_u8))));
            prop_assert_eq!(relabel(check_u8(&g, &core_as(&codes, conflict, id_u8))), check_string(&g, &core(&codes, conflict)));
            prop_assert_eq!(relabel(check_u8(&g, &model_as(false_bits, id_u8))), check_string(&g, &model(false_bits)));
        }
    }

    /// Every step code decodes as [`STEP_CODES`] documents: `0..3` a unit, `3..9` the implication
    /// `IMPLY_PAIRS[k - 3]`, and a code past the table wraps mod 6 rather than panicking.
    #[test]
    fn every_step_code_decodes_as_documented() {
        let pair = |k: usize| {
            let (a, b) = IMPLY_PAIRS[k];
            Step::Implies(id_string(a), id_string(b))
        };
        for code in 0..=u8::MAX {
            let k = usize::from(code);
            let want = if k < 3 {
                Step::Unit(id_string(k))
            } else {
                pair((k - 3) % 6)
            };
            assert_eq!(step(code), want, "code {code}");
        }
        assert_eq!(step(3), pair(0));
        assert_eq!(step(8), pair(5));
        assert_eq!(step(9), pair(0));
    }
}
