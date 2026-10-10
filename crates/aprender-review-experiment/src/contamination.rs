//! REX-02 contamination check (contract `review-corpus-contamination-v1`,
//! spec §2.2 and rule R-2).
//!
//! A sealed test item leaks into a training file when that file carries the
//! item's diff sha as a literal, or carries any of the item's hunks (header
//! stripped, so a re-based copy still matches). JSON string values are scanned
//! after unescaping, so a diff stored in a JSONL `"diff"` field is seen as the
//! diff it is, not as one escaped line.
//!
//! With [`Index::with_sketches`], a re-spaced, renamed or lightly edited copy
//! is a `cluster` hit too (PRA-001 T7, [`crate::cluster`]).

use crate::cluster::{containment, shingles, CONTAINMENT};
use crate::corpus::{hunk_fingerprints, Sealed};
use std::collections::BTreeMap;

/// Directories that hold training data for this spec (B2 teacher sets,
/// challenger few-shot pools). Every file under them is scanned.
pub const TRAIN_ROOTS: &[&str] = &["docs/audits/review-corpus/train"];

/// One leak: which test item, which file, and how it was seen.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Hit {
    pub item: String,
    pub file: String,
    pub how: &'static str,
}

/// Index of the sealed items: diff sha → id, hunk fingerprint → id, and
/// (optionally) each item's near-dup sketch from [`crate::cluster`].
#[derive(Debug, Default)]
pub struct Index {
    shas: BTreeMap<String, String>,
    hunks: BTreeMap<String, String>,
    sketches: Vec<(String, Vec<u64>)>,
}

impl Index {
    #[must_use]
    pub fn new(sealed: &[Sealed]) -> Self {
        let mut ix = Self::default();
        for s in sealed {
            ix.shas.insert(s.diff_sha256.clone(), s.id.clone());
            for h in &s.hunks {
                ix.hunks.insert(h.clone(), s.id.clone());
            }
        }
        ix
    }

    /// Add sealed near-dup sketches (`id`, bottom-k hashes), so a re-spaced or
    /// renamed copy of a sealed diff is a `cluster` hit.
    #[must_use]
    pub fn with_sketches(mut self, sketches: Vec<(String, Vec<u64>)>) -> Self {
        self.sketches = sketches;
        self
    }

    /// Number of sealed diff shas indexed (0 means the scan proves nothing).
    #[must_use]
    pub fn len(&self) -> usize {
        self.shas.len()
    }

    /// True when nothing is sealed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shas.is_empty()
    }

    fn scan_str(&self, text: &str, file: &str, hits: &mut Vec<Hit>) {
        for (sha, id) in &self.shas {
            if text.contains(sha.as_str()) {
                hits.push(hit(id, file, "diff-sha"));
            }
        }
        for fp in hunk_fingerprints(text) {
            if let Some(id) = self.hunks.get(&fp) {
                hits.push(hit(id, file, "hunk"));
            }
        }
        if self.sketches.is_empty() {
            return;
        }
        let set = shingles(text);
        for (id, s) in &self.sketches {
            if containment(s, &set) >= CONTAINMENT {
                hits.push(hit(id, file, "cluster"));
            }
        }
    }

    /// Every leak of a sealed item into `text` (the contents of `file`).
    #[must_use]
    pub fn scan(&self, text: &str, file: &str) -> Vec<Hit> {
        let mut hits = Vec::new();
        self.scan_str(text, file, &mut hits);
        for line in text.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                let mut strings = Vec::new();
                collect_strings(&v, &mut strings);
                for s in strings {
                    self.scan_str(s, file, &mut hits);
                }
            }
        }
        hits.sort();
        hits.dedup();
        hits
    }
}

fn hit(id: &str, file: &str, how: &'static str) -> Hit {
    Hit {
        item: id.to_string(),
        file: file.to_string(),
        how,
    }
}

fn collect_strings<'a>(v: &'a serde_json::Value, out: &mut Vec<&'a str>) {
    match v {
        serde_json::Value::String(s) => out.push(s),
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_strings(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| collect_strings(x, out)),
        _ => {}
    }
}

#[cfg(test)]
#[path = "contamination_tests.rs"]
mod tests;
