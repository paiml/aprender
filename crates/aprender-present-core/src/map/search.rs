//! Case-insensitive prefix search over feature names (APR-MAP-001 MAP-6 slice 2).
//!
//! Ranking, best first: an exact match (after case folding), then the shorter name, then the
//! lower cell id. A cell that matches under several names is reported once, at its best rank.
//! Ties therefore never depend on insertion order.

use super::MapError;

/// Immutable index of `(cell id, feature name)` pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchIndex {
    entries: Vec<Entry>,
}

#[derive(Debug, Clone, PartialEq)]
struct Entry {
    cell: u32,
    lower: String,
    chars: usize,
}

impl SearchIndex {
    /// Builds the index. Rejects an empty feature name with `Malformed("feature name")`.
    pub fn new(features: &[(u32, &str)]) -> Result<Self, MapError> {
        unimplemented!(
            "MAP-6 slice 2 RED: SearchIndex::new {} entries",
            features.len()
        )
    }

    /// Up to `limit` cell ids whose name starts with `query`, ignoring case, best rank first.
    /// An empty query or a zero limit returns nothing.
    pub fn prefix(&self, query: &str, limit: usize) -> Vec<u32> {
        unimplemented!("MAP-6 slice 2 RED: prefix {query} {limit}")
    }
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
