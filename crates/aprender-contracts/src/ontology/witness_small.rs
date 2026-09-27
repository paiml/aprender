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

use super::witness::{check, Checked, ClauseSet, Core, Model, Step, WitnessResult};

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

/// Decode a step code (see [`STEP_CODES`]).
#[must_use]
pub fn step(code: u8) -> Step {
    let k = usize::from(code);
    if k < 3 {
        Step::Unit(VARS[k].to_string())
    } else {
        let (a, b) = IMPLY_PAIRS[(k - 3) % 6];
        Step::Implies(VARS[a].to_string(), VARS[b].to_string())
    }
}

/// A core from step codes and a conflict-pair index.
#[must_use]
pub fn core(codes: &[u8], conflict: usize) -> WitnessResult {
    let (a, b) = CONFLICT_PAIRS[conflict % 3];
    WitnessResult::UnsatCore(Core {
        steps: codes.iter().map(|c| step(*c)).collect(),
        conflict: (VARS[a].to_string(), VARS[b].to_string()),
    })
}

/// A model from a false-set bitmask.
#[must_use]
pub fn model(false_bits: u8) -> WitnessResult {
    WitnessResult::Model(Model {
        false_vars: (0..3)
            .filter(|i| false_bits & (1 << i) != 0)
            .map(|i| VARS[i].to_string())
            .collect(),
    })
}

/// `KANI-ONT-9-1`'s property: whatever the checker ACCEPTS is true under relation semantics — an accepted core
/// means no assignment satisfies the graph, an accepted model means one does, and an accepted core names only
/// variables the graph has. A refusal is always sound (it claims nothing).
#[must_use]
pub fn accepted_verdict_is_semantic(g: &SmallGraph, r: &WitnessResult) -> bool {
    match check(&g.clause_set(), r) {
        Ok(Checked::Unsat { core }) => {
            let known: BTreeSet<String> = VARS.iter().map(|v| (*v).to_string()).collect();
            !g.satisfiable() && !core.is_empty() && core.is_subset(&known)
        }
        Ok(Checked::Sat) => g.satisfiable(),
        Err(_) => true,
    }
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
        assert!(accepted_verdict_is_semantic(&g, &core(&[0, 3, 6], 1)));
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
        /// The L2 twin of `KANI-ONT-9-1`: the same body, sampled rather than exhausted.
        #[test]
        fn kani_ont_9_1_twin(
            g in graph(),
            codes in proptest::collection::vec(0u8..STEP_CODES, 1..=CORE_BOUND),
            conflict in 0usize..3,
            false_bits in 0u8..8,
        ) {
            prop_assert!(accepted_verdict_is_semantic(&g, &core(&codes, conflict)));
            prop_assert!(accepted_verdict_is_semantic(&g, &model(false_bits)));
        }
    }
}
