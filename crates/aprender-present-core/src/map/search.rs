//! Case-insensitive prefix search over feature names (APR-MAP-001 MAP-6 slice 2).
//!
//! Ranking, best first: an exact match (after case folding), then the shorter name, then the
//! lower cell id. A cell that matches under several names is reported once, at its best rank.
//! Ties therefore never depend on insertion order.

use super::MapError;
use std::collections::BTreeMap;

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
        let entries = features
            .iter()
            .map(|&(cell, name)| {
                if name.is_empty() {
                    return Err(MapError::Malformed("feature name"));
                }
                let lower = name.to_lowercase();
                let chars = lower.chars().count();
                Ok(Entry { cell, lower, chars })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { entries })
    }

    /// Up to `limit` cell ids whose name starts with `query`, ignoring case, best rank first.
    /// An empty query or a zero limit returns nothing.
    pub fn prefix(&self, query: &str, limit: usize) -> Vec<u32> {
        let q = query.to_lowercase();
        if q.is_empty() || limit == 0 {
            return Vec::new();
        }
        // Best rank per cell: (0 when the name equals the query, else 1; name length; cell id).
        let mut best: BTreeMap<u32, (u8, usize)> = BTreeMap::new();
        for e in self.entries.iter().filter(|e| e.lower.starts_with(&q)) {
            let key = (u8::from(e.lower != q), e.chars);
            best.entry(e.cell)
                .and_modify(|k| *k = (*k).min(key))
                .or_insert(key);
        }
        let mut ranked: Vec<(u8, usize, u32)> = best
            .into_iter()
            .map(|(cell, (exact, chars))| (exact, chars, cell))
            .collect();
        ranked.sort_unstable();
        ranked
            .into_iter()
            .take(limit)
            .map(|(_, _, cell)| cell)
            .collect()
    }
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
