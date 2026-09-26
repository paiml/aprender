//! PRM-C8 `trace-dedup-v1`: near-duplicate clusters over trace diffs (gate
//! G-DUP; contract `trace-dedup-v1`, FALSIFY-TDD-001..004).
//!
//! Reuses the [`crate::cluster`] fingerprint (normalised token shingles,
//! bottom-k sketch, containment), so a re-indented or renamed copy of a diff
//! lands in its original's cluster, exactly as G-CON sees it.
//!
//! - **Candidates (LSH)**: an inverted index over sketch hashes; two items are
//!   compared only if their sketches share a hash. Items whose sketches share
//!   none are never compared — the recall bound, stated, not hidden.
//! - **Match**: either item's sketch is at least [`CONTAINMENT`] contained in
//!   the other's shingle set.
//! - **Closure**: union-find, so near-dup chains close transitively.
//! - **Tiny diffs** (below `MIN_SHINGLES`, no sketch) merge only on identical
//!   normalised tokens.
//! - **Cluster id**: the id of the earliest member by `(at, id)`, so the id
//!   does not depend on input order. The split guard (PRM-C9) assigns a whole
//!   cluster to that member's split.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::cluster::{containment, shingles, tokens, CONTAINMENT, MIN_SHINGLES, SKETCH};

pub const SCHEME: &str = "trace-dedup-v1";

/// One diff to cluster: its id (a `diff_sha256`), first-seen time (RFC 3339
/// UTC, so it orders as text) and the diff text.
#[derive(Debug, Clone, Copy)]
pub struct Item<'a> {
    pub id: &'a str,
    pub at: &'a str,
    pub diff: &'a str,
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (root(parent, a), root(parent, b));
    parent[ra.max(rb)] = ra.min(rb);
}

/// The cluster id of each item, in input order.
#[must_use]
pub fn clusters(items: &[Item]) -> Vec<String> {
    let sets: Vec<HashSet<u64>> = items.iter().map(|i| shingles(i.diff)).collect();
    let sketches: Vec<Option<Vec<u64>>> = sets
        .iter()
        .map(|s| {
            (s.len() >= MIN_SHINGLES).then(|| {
                let mut v: Vec<u64> = s.iter().copied().collect();
                v.sort_unstable();
                v.truncate(SKETCH);
                v
            })
        })
        .collect();
    let mut parent: Vec<usize> = (0..items.len()).collect();
    let mut postings: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut tiny: HashMap<Vec<String>, usize> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        let Some(sk) = &sketches[i] else {
            let first = *tiny.entry(tokens(it.diff)).or_insert(i);
            union(&mut parent, first, i);
            continue;
        };
        let mut seen = HashSet::new();
        for h in sk {
            let js = postings.entry(*h).or_default();
            for &j in js.iter().filter(|&&j| seen.insert(j)) {
                let other = sketches[j].as_deref().unwrap_or_default();
                if containment(sk, &sets[j]) >= CONTAINMENT
                    || containment(other, &sets[i]) >= CONTAINMENT
                {
                    union(&mut parent, i, j);
                }
            }
            js.push(i);
        }
    }
    let mut earliest: BTreeMap<usize, usize> = BTreeMap::new();
    for i in 0..items.len() {
        let r = root(&mut parent, i);
        let e = earliest.entry(r).or_insert(i);
        if (items[i].at, items[i].id) < (items[*e].at, items[*e].id) {
            *e = i;
        }
    }
    (0..items.len())
        .map(|i| items[earliest[&root(&mut parent, i)]].id.to_owned())
        .collect()
}

#[cfg(test)]
#[path = "dedup_tests.rs"]
mod tests;
