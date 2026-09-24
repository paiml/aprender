//! ONT-001 §5 ONT-9 — planted-solution generation for `ont_planted` (F-12: verdict == construction).
//!
//! Two families of graph, each built so its answer is known before the checker sees it:
//!
//! - **consistent by construction** — draw an assignment σ first, then keep only clauses σ satisfies. σ is a
//!   model, so the graph is satisfiable, the planted model must check, and NO core may check.
//! - **planted contradiction** — assert one unit `u`, lay two implication paths out of it to distinct ends `x`,
//!   `y`, and add `x ⊥ y`; then add noise clauses of every kind. Unit propagation derives both ends from `u`, so
//!   the graph is unsatisfiable whatever the noise, the planted derivation must check, and NO model may check.
//!
//! The adversarial certificates are aimed at the checks a weakened checker drops: derivations that are valid step
//! by step but close on a pair that is not a conflict clause (the "accept any two ids" mutation), on a conflict
//! clause with a side never derived, and step sequences drawn at random from the graph's own clauses and from
//! outside it. A brute-force satisfiability oracle over every assignment cross-checks the construction itself, so a
//! generator bug cannot pass as a checker property.
//!
//! No reasoner lives here (F-7): the planted certificates are written down from the construction, never searched.

use std::collections::BTreeSet;

use super::witness::{ClauseSet, Core, Model, Step, WitnessResult};

/// A splitmix64 stream — deterministic from the proptest seed, so a failing case replays exactly.
pub struct Rng(u64);

impl Rng {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % (n as u64)).unwrap_or(0)
    }

    pub fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

/// Variable name `i`.
#[must_use]
pub fn var(i: usize) -> String {
    format!("c{i}")
}

fn ordered(a: usize, b: usize) -> (String, String) {
    let (x, y) = (var(a), var(b));
    if x <= y {
        (x, y)
    } else {
        (y, x)
    }
}

/// A graph and the certificate its construction guarantees.
pub struct Planted {
    pub n: usize,
    pub clauses: ClauseSet,
    pub planted: WitnessResult,
}

/// Consistent by construction: an assignment first, then only clauses it satisfies.
pub fn consistent(rng: &mut Rng) -> Planted {
    let n = 2 + rng.below(7);
    let mut sigma: Vec<bool> = (0..n).map(|_| rng.coin()).collect();
    sigma[rng.below(n)] = true; // at least one live contract, so the graph has a unit
    let mut cs = ClauseSet::default();
    for (i, t) in sigma.iter().enumerate() {
        if *t && (cs.units.is_empty() || rng.coin()) {
            cs.units.insert(var(i));
        }
    }
    for _ in 0..(3 * n) {
        let (a, b) = (rng.below(n), rng.below(n));
        if a == b {
            continue;
        }
        if !sigma[a] || sigma[b] {
            cs.implies.insert((var(a), var(b)));
        }
        if !(sigma[a] && sigma[b]) {
            cs.conflicts.insert(ordered(a, b));
        }
    }
    let planted = WitnessResult::Model(Model {
        false_vars: (0..n).filter(|i| !sigma[*i]).map(var).collect(),
    });
    Planted {
        n,
        clauses: cs,
        planted,
    }
}

/// A path of distinct variables out of `from`, `len` steps long, avoiding `avoid`.
fn path(rng: &mut Rng, n: usize, from: usize, len: usize, avoid: &BTreeSet<usize>) -> Vec<usize> {
    let mut p = vec![from];
    for _ in 0..len {
        let free: Vec<usize> = (0..n)
            .filter(|v| !p.contains(v) && !avoid.contains(v))
            .collect();
        if free.is_empty() {
            break;
        }
        p.push(free[rng.below(free.len())]);
    }
    p
}

/// A planted contradiction under noise.
pub fn contradicted(rng: &mut Rng) -> Planted {
    let n = 3 + rng.below(6);
    let u = rng.below(n);
    let ll = 1 + rng.below(3);
    let left = path(rng, n, u, ll, &BTreeSet::new());
    let used: BTreeSet<usize> = left[1..].iter().copied().collect();
    let rl = 1 + rng.below(3);
    let right = path(rng, n, u, rl, &used);
    let (x, y) = (*left.last().unwrap_or(&u), *right.last().unwrap_or(&u));
    // Two paths out of u with no shared variable past u end at distinct variables unless both are empty.
    let y = if x == y { (x + 1) % n } else { y };
    let mut cs = ClauseSet::default();
    cs.units.insert(var(u));
    let mut steps = vec![Step::Unit(var(u))];
    let mut derived: BTreeSet<usize> = [u].into();
    for p in [&left, &right] {
        for w in p.windows(2) {
            cs.implies.insert((var(w[0]), var(w[1])));
            if derived.insert(w[1]) {
                steps.push(Step::Implies(var(w[0]), var(w[1])));
            }
        }
    }
    if !derived.contains(&y) {
        // right collapsed to [u] and y was bumped: derive it from u directly.
        cs.implies.insert((var(u), var(y)));
        steps.push(Step::Implies(var(u), var(y)));
    }
    cs.conflicts.insert(ordered(x, y));
    for _ in 0..(2 * n) {
        let (a, b) = (rng.below(n), rng.below(n));
        match rng.below(3) {
            0 => {
                cs.units.insert(var(a));
            }
            1 if a != b => {
                cs.implies.insert((var(a), var(b)));
            }
            _ if a != b => {
                cs.conflicts.insert(ordered(a, b));
            }
            _ => {}
        }
    }
    let conflict = (var(x), var(y));
    Planted {
        n,
        clauses: cs,
        planted: WitnessResult::UnsatCore(Core { steps, conflict }),
    }
}

/// Every variable a clause set names, as indices.
fn index(v: &str) -> Option<usize> {
    v.strip_prefix('c').and_then(|d| d.parse().ok())
}

/// Brute force: does any of the 2ⁿ assignments satisfy every clause?
#[must_use]
pub fn satisfiable(cs: &ClauseSet, n: usize) -> bool {
    let t = |m: u32, v: &str| index(v).is_some_and(|i| m & (1 << i) != 0);
    (0u32..(1 << n)).any(|m| {
        cs.units.iter().all(|u| t(m, u))
            && cs.implies.iter().all(|(a, b)| !t(m, a) || t(m, b))
            && cs.conflicts.iter().all(|(a, b)| !t(m, a) || !t(m, b))
    })
}

/// Cores aimed at a consistent graph. None of them may check.
pub fn adversarial_cores(rng: &mut Rng, cs: &ClauseSet, n: usize) -> Vec<WitnessResult> {
    let mut out = Vec::new();
    for _ in 0..12 {
        // A derivation that is valid step by step: units, then implications whose premise is derived.
        let mut steps = Vec::new();
        let mut derived: Vec<String> = Vec::new();
        for u in &cs.units {
            if rng.coin() || derived.is_empty() {
                steps.push(Step::Unit(u.clone()));
                derived.push(u.clone());
            }
        }
        for _ in 0..n {
            let from: Vec<&(String, String)> = cs
                .implies
                .iter()
                .filter(|(a, b)| derived.contains(a) && !derived.contains(b))
                .collect();
            if from.is_empty() {
                break;
            }
            let (a, b) = from[rng.below(from.len())].clone();
            steps.push(Step::Implies(a, b.clone()));
            derived.push(b);
        }
        // Close it on: two derived ids (conflict or not), a real conflict clause, or any two ids at all.
        let pick = |rng: &mut Rng, d: &[String]| d[rng.below(d.len())].clone();
        let conflict = match rng.below(3) {
            0 => {
                let x = pick(rng, &derived);
                (x, pick(rng, &derived))
            }
            1 if !cs.conflicts.is_empty() => {
                let all: Vec<&(String, String)> = cs.conflicts.iter().collect();
                all[rng.below(all.len())].clone()
            }
            _ => {
                let x = var(rng.below(n));
                (x, var(rng.below(n)))
            }
        };
        out.push(WitnessResult::UnsatCore(Core { steps, conflict }));
        // And a sequence drawn at random, in and out of the graph.
        let random: Vec<Step> = (0..=rng.below(n))
            .map(|_| {
                if rng.coin() {
                    Step::Unit(var(rng.below(n)))
                } else {
                    Step::Implies(var(rng.below(n)), var(rng.below(n)))
                }
            })
            .collect();
        out.push(WitnessResult::UnsatCore(Core {
            steps: random,
            conflict: {
                let x = var(rng.below(n));
                (x, var(rng.below(n)))
            },
        }));
    }
    out
}

/// Models aimed at a contradicted graph: every one of the 2ⁿ when n is small, and random ones otherwise.
pub fn adversarial_models(rng: &mut Rng, n: usize) -> Vec<WitnessResult> {
    let masks: Vec<u32> = if n <= 6 {
        (0u32..(1 << n)).collect()
    } else {
        (0..64)
            .map(|_| u32::try_from(rng.next() & ((1 << n) - 1)).unwrap_or(0))
            .collect()
    };
    masks
        .into_iter()
        .map(|m| {
            WitnessResult::Model(Model {
                false_vars: (0..n).filter(|i| m & (1 << i) != 0).map(var).collect(),
            })
        })
        .collect()
}
