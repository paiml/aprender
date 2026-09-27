//! The reasoner: Horn-SAT by unit propagation, with a certificate for either answer.
//!
//! Propagation is breadth-first from the units in sorted order, so the derivation — and the witness — is the same
//! bytes on every run for the same clause set. On a conflict the core is the derivation of its two sides and
//! nothing else, in the order propagation reached each variable, which is an order the checker accepts: every
//! premise is derived before the implication that uses it.

use std::collections::{BTreeMap, VecDeque};

use provable_contracts::ontology::witness::{
    implication_adjacency, variables, ClauseSet, Core, Model, Step, WitnessResult,
};

/// Each derived variable: (derivation index, the premise that derived it; `None` for a unit).
type Derived<'a> = BTreeMap<&'a str, (usize, Option<&'a str>)>;

/// Decide `cs`: an `unsat_core` for the first violated conflict (in sorted order), else the least model.
pub fn solve(cs: &ClauseSet) -> WitnessResult {
    let derived = propagate(cs);
    let violated = cs
        .conflicts
        .iter()
        .find(|(a, b)| derived.contains_key(a.as_str()) && derived.contains_key(b.as_str()));
    if let Some((a, b)) = violated {
        let mut needed: BTreeMap<usize, Step> = BTreeMap::new();
        trace(&derived, a, &mut needed);
        trace(&derived, b, &mut needed);
        return WitnessResult::UnsatCore(Core {
            steps: needed.into_values().collect(),
            conflict: (a.clone(), b.clone()),
        });
    }
    WitnessResult::Model(Model {
        false_vars: variables(cs)
            .into_iter()
            .filter(|v| !derived.contains_key(v.as_str()))
            .collect(),
    })
}

/// Breadth-first unit propagation from the sorted units.
fn propagate(cs: &ClauseSet) -> Derived<'_> {
    let adj = implication_adjacency(cs);
    let mut derived: Derived<'_> = BTreeMap::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    for u in &cs.units {
        derived.insert(u.as_str(), (derived.len(), None));
        queue.push_back(u.as_str());
    }
    while let Some(v) = queue.pop_front() {
        for &w in adj.get(v).map_or(&[][..], Vec::as_slice) {
            if !derived.contains_key(w) {
                derived.insert(w, (derived.len(), Some(v)));
                queue.push_back(w);
            }
        }
    }
    derived
}

/// Add the derivation of `side` to `needed`, back to its unit or to a step already there.
fn trace(derived: &Derived<'_>, side: &str, needed: &mut BTreeMap<usize, Step>) {
    let mut v = side;
    while let Some(&(idx, premise)) = derived.get(v) {
        if needed.contains_key(&idx) {
            return;
        }
        let Some(p) = premise else {
            needed.insert(idx, Step::Unit(v.to_string()));
            return;
        };
        needed.insert(idx, Step::Implies(p.to_string(), v.to_string()));
        v = p;
    }
}
