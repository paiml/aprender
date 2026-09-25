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

/// The cluster id of each item, in input order.
#[must_use]
pub fn clusters(items: &[Item]) -> Vec<String> {
    items.iter().map(|i| i.id.to_owned()).collect()
}

#[cfg(test)]
#[path = "dedup_tests.rs"]
mod tests;
