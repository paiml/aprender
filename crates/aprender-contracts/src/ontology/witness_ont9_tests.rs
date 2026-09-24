//! ONT-9 (F-12): the checker's verdict on a planted instance equals its construction, and a certificate weakened by
//! one step or one flipped variable is refused. The oracle here is brute force over assignments — it shares no code
//! with `check`, so a checker weakened to accept more, or less, disagrees with it.
//!
//! `checker_is_sound_and_complete_at_three_variables` is the L2 twin of `KANI-ONT-9-1`: the same bounded universe,
//! enumerated instead of solved.

use std::collections::BTreeSet;

use super::*;

/// Every variable any clause names, in order.
fn vars_of(cs: &ClauseSet) -> Vec<String> {
    let mut v: BTreeSet<&String> = cs.units.iter().collect();
    for (a, b) in cs.implies.iter().chain(&cs.conflicts) {
        v.insert(a);
        v.insert(b);
    }
    v.into_iter().cloned().collect()
}

/// Does the set of true variables satisfy every clause? Evaluated directly, not through `check`.
fn satisfies(cs: &ClauseSet, t: &BTreeSet<&str>) -> bool {
    cs.units.iter().all(|u| t.contains(u.as_str()))
        && cs
            .implies
            .iter()
            .all(|(a, b)| !t.contains(a.as_str()) || t.contains(b.as_str()))
        && cs
            .conflicts
            .iter()
            .all(|(a, b)| !(t.contains(a.as_str()) && t.contains(b.as_str())))
}

/// Every assignment to the clause set's variables, as the set each makes true.
fn assignments(cs: &ClauseSet) -> Vec<BTreeSet<String>> {
    let vars = vars_of(cs);
    assert!(
        vars.len() <= 12,
        "brute force over {} variables",
        vars.len()
    );
    (0u32..1 << vars.len())
        .map(|m| {
            vars.iter()
                .enumerate()
                .filter(|(i, _)| m >> i & 1 == 1)
                .map(|(_, v)| v.clone())
                .collect()
        })
        .collect()
}

fn brute_sat(cs: &ClauseSet) -> bool {
    assignments(cs)
        .iter()
        .any(|t| satisfies(cs, &t.iter().map(String::as_str).collect()))
}

fn model_of(cs: &ClauseSet, t: &BTreeSet<String>) -> WitnessResult {
    WitnessResult::Model(Model {
        false_vars: vars_of(cs).into_iter().filter(|v| !t.contains(v)).collect(),
    })
}

/// (seed, n, satisfiable) over the corpus every test below draws from: 1..=10 variables, both answers.
fn corpus() -> impl Iterator<Item = (u64, usize, bool)> {
    (0u64..200).flat_map(|seed| {
        let n = 1 + usize::try_from(seed % 10).unwrap_or(0);
        [(seed, n, true), (seed, n, false)]
    })
}

#[test]
fn planted_verdict_equals_construction() {
    for (seed, n, sat) in corpus() {
        let p = ont_planted(seed, n, sat);
        assert_eq!(p.satisfiable, sat);
        assert_eq!(
            brute_sat(&p.clauses),
            sat,
            "the generator's construction is wrong at {seed}/{n}/{sat}: {:?}",
            p.clauses
        );
        match (check(&p.clauses, &p.certificate), sat) {
            (Ok(Checked::Sat), true) | (Ok(Checked::Unsat { .. }), false) => {}
            (got, _) => panic!(
                "{seed}/{n}/{sat}: construction {sat}, checker {got:?}, clauses {:?}",
                p.clauses
            ),
        }
    }
}

#[test]
fn planting_is_reproducible_by_seed_and_varies_with_it() {
    assert_eq!(ont_planted(7, 6, false), ont_planted(7, 6, false));
    let distinct: BTreeSet<String> = (0..20)
        .map(|s| format!("{:?}", ont_planted(s, 6, false).clauses))
        .collect();
    assert!(distinct.len() > 15, "{} distinct of 20", distinct.len());
}

#[test]
fn every_step_of_a_planted_core_is_load_bearing() {
    for (seed, n, _) in corpus() {
        let p = ont_planted(seed, n, false);
        let WitnessResult::UnsatCore(core) = &p.certificate else {
            panic!("an unsatisfiable plant carries a core")
        };
        for i in 0..core.steps.len() {
            let mut weak = core.clone();
            weak.steps.remove(i);
            assert!(
                check(&p.clauses, &WitnessResult::UnsatCore(weak)).is_err(),
                "{seed}/{n}: the core without step {i} ({:?}) still checked",
                core.steps[i]
            );
        }
    }
}

#[test]
fn no_assignment_passes_as_a_model_of_a_planted_unsat_instance() {
    for (seed, n, _) in corpus().filter(|&(_, n, _)| n <= 8) {
        let p = ont_planted(seed, n, false);
        for t in assignments(&p.clauses) {
            assert!(
                check(&p.clauses, &model_of(&p.clauses, &t)).is_err(),
                "{seed}/{n}: {t:?} accepted as a model of an unsatisfiable set"
            );
        }
    }
}

#[test]
fn a_planted_model_with_a_unit_flipped_false_is_refused() {
    for (seed, n, _) in corpus() {
        let p = ont_planted(seed, n, true);
        let WitnessResult::Model(m) = &p.certificate else {
            panic!("a satisfiable plant carries a model")
        };
        let unit = p.clauses.units.iter().next().expect("v00 is always a unit");
        let mut weak = m.clone();
        weak.false_vars.push(unit.clone());
        assert_eq!(
            check(&p.clauses, &WitnessResult::Model(weak)),
            Err(CheckError::ModelFalsifiesUnit(unit.clone()))
        );
    }
}

/// The bounded universe of `KANI-ONT-9-1`: three variables, 3 units, 6 implications, 3 conflicts — all 4096 clause
/// sets. For each, every core of one to three steps the universe can spell, against every conflict of the universe, and
/// every one of the 8 models. `check` accepts a core only when the set is unsatisfiable (sound), accepts SOME core
/// whenever it is (complete at the bound: a Horn derivation over 3 variables takes at most 3 steps), and accepts a
/// model exactly when brute force says the model satisfies the set.
#[test]
fn checker_is_sound_and_complete_at_three_variables() {
    let v = ["a", "b", "c"].map(String::from);
    let imp: Vec<(String, String)> = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        .iter()
        .map(|&(a, b)| (v[a].clone(), v[b].clone()))
        .collect();
    let con: Vec<(String, String)> = [(0, 1), (0, 2), (1, 2)]
        .iter()
        .map(|&(a, b)| (v[a].clone(), v[b].clone()))
        .collect();
    // Every step the universe can spell — present in the set or not, so a step naming a clause the set lacks is
    // tried too — and every sequence of one to three of them.
    let alphabet: Vec<Step> = v
        .iter()
        .map(|u| Step::Unit(u.clone()))
        .chain(imp.iter().map(|(a, b)| Step::Implies(a.clone(), b.clone())))
        .collect();
    let mut cores: Vec<Vec<Step>> = Vec::new();
    let mut frontier: Vec<Vec<Step>> = vec![vec![]];
    for _ in 0..3 {
        frontier = frontier
            .iter()
            .flat_map(|s| {
                alphabet
                    .iter()
                    .map(move |x| [s.clone(), vec![x.clone()]].concat())
            })
            .collect();
        cores.extend(frontier.iter().cloned());
    }
    let (mut sets, mut cores_checked) = (0, 0usize);
    for bits in 0u32..1 << 12 {
        let cs = ClauseSet {
            units: (0..3)
                .filter(|i| bits >> i & 1 == 1)
                .map(|i| v[i].clone())
                .collect(),
            implies: (0..6)
                .filter(|k| bits >> (3 + k) & 1 == 1)
                .map(|k| imp[k].clone())
                .collect(),
            conflicts: (0..3)
                .filter(|k| bits >> (9 + k) & 1 == 1)
                .map(|k| con[k].clone())
                .collect(),
        };
        let sat = (0u8..8).any(|m| {
            satisfies(
                &cs,
                &(0..3)
                    .filter(|i| m >> i & 1 == 1)
                    .map(|i| v[i].as_str())
                    .collect(),
            )
        });
        let mut accepted_any = false;
        for steps in &cores {
            for conflict in &con {
                let core = WitnessResult::UnsatCore(Core {
                    steps: steps.clone(),
                    conflict: conflict.clone(),
                });
                if check(&cs, &core).is_ok() {
                    assert!(
                        !sat,
                        "sound: a core was accepted for a satisfiable set {cs:?}: {core:?}"
                    );
                    accepted_any = true;
                }
            }
        }
        cores_checked += cores.len() * con.len();
        assert_eq!(accepted_any, !sat, "complete at the bound: {cs:?}");
        for m in 0u8..8 {
            let t: BTreeSet<&str> = (0..3)
                .filter(|i| m >> i & 1 == 1)
                .map(|i| v[i].as_str())
                .collect();
            let model = WitnessResult::Model(Model {
                false_vars: (0..3)
                    .filter(|i| m >> i & 1 == 0)
                    .map(|i| v[i].clone())
                    .collect(),
            });
            assert_eq!(
                check(&cs, &model).is_ok(),
                satisfies(&cs, &t),
                "{cs:?} under {t:?}"
            );
        }
        sets += 1;
    }
    assert_eq!(sets, 4096);
    // 4096 sets × 3 conflicts × (9 + 9² + 9³) step sequences — the whole universe, exactly.
    assert_eq!(
        cores_checked,
        4096 * 3 * 819,
        "the universe was not enumerated"
    );
}
