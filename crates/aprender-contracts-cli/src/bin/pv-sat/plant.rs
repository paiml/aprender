//! `pc_reasoner`: the plant pv-sat must see through before it may write a witness.
//!
//! `A⇒B, B⇒C, contradicts(A,C)` with only `A` asserted is unsatisfiable, and its only core is `[A, B, C]` —
//! a reasoner that stops one implication early, or that "finds" the conflict without deriving `C`, draws
//! something else. The core must also pass the library's checker: a reasoner and a checker that disagree about
//! the plant cannot both be trusted with the corpus.

use std::collections::BTreeSet;

use provable_contracts::ontology::witness::{check, Checked, ClauseSet, WitnessResult, FIRED};

use crate::sat::solve;

/// The planted clause set.
pub fn plant() -> ClauseSet {
    let s = |x: &str| x.to_string();
    ClauseSet {
        units: [s("A")].into(),
        implies: [(s("A"), s("B")), (s("B"), s("C"))].into(),
        conflicts: [(s("A"), s("C"))].into(),
    }
}

/// `Ok(FIRED)` when the reasoner draws the core `[A, B, C]` and the checker accepts it.
pub fn pc_reasoner() -> Result<&'static str, String> {
    let cs = plant();
    let result = solve(&cs);
    let WitnessResult::UnsatCore(_) = &result else {
        return Err(format!(
            "the plant is unsatisfiable and the reasoner answered {result:?}"
        ));
    };
    match check(&cs, &result) {
        Ok(Checked::Unsat { core }) => {
            let want: BTreeSet<String> = ["A", "B", "C"].map(String::from).into();
            if core == want {
                Ok(FIRED)
            } else {
                Err(format!(
                    "the plant's core is [A, B, C] and the reasoner drew {core:?}"
                ))
            }
        }
        Ok(Checked::Sat) => Err("the checker read the plant's core as a model".into()),
        Err(e) => Err(format!(
            "the checker refused the reasoner's core for the plant: {e}"
        )),
    }
}

/// The satisfiable twin of the plant — the conflict removed — must yield a model the checker accepts, so a
/// reasoner that answers "unsat" to everything cannot pass `--self-test`.
pub fn pc_model() -> Result<(), String> {
    let mut cs = plant();
    cs.conflicts.clear();
    let result = solve(&cs);
    match check(&cs, &result) {
        Ok(Checked::Sat) => Ok(()),
        other => Err(format!(
            "the plant without its conflict is satisfiable, and got {other:?}"
        )),
    }
}
