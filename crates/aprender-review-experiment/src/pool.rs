//! Training-pool admission gate (ruling agent-trace item 6; PRA-001 T6,
//! contract `trace-admission-secret-v1`).
//!
//! A captured row enters the fine-tune/distill pool only when it is secret-scan
//! clean under BOTH scanners of [`crate::secret`] AND carries no sealed-test
//! hash (`review-corpus-contamination-v1`).
//! A secret hit QUARANTINES the row as-is: it is never redacted in place, since a
//! redacted row is a row the scanner already failed on once. A sealed-test hit
//! REFUSES the row.

use crate::contamination::Index;
use crate::secret::{scan, Hit};

/// Where one input row went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Clean: the row is in the pool.
    Admitted,
    /// A secret was found; the row is held byte-for-byte outside the pool.
    Quarantined { secrets: Vec<Hit> },
    /// The row carries a sealed test item; it never enters the pool.
    Refused { items: Vec<String> },
    /// The sealed index is empty, so contamination is unproven either way.
    Unchecked,
}

/// The outcome for a JSONL batch: one verdict per non-empty line, and the pool.
#[derive(Debug, Default)]
pub struct Admission {
    pub verdicts: Vec<(String, Verdict)>,
}

impl Admission {
    /// The rows admitted to the training pool, in input order.
    #[must_use]
    pub fn pool(&self) -> Vec<&str> {
        self.with(|v| matches!(v, Verdict::Admitted))
    }

    /// The quarantined rows, byte-identical to their input.
    #[must_use]
    pub fn quarantine(&self) -> Vec<&str> {
        self.with(|v| matches!(v, Verdict::Quarantined { .. }))
    }

    fn with(&self, keep: impl Fn(&Verdict) -> bool) -> Vec<&str> {
        self.verdicts
            .iter()
            .filter(|(_, v)| keep(v))
            .map(|(r, _)| r.as_str())
            .collect()
    }
}

/// Admit a JSONL batch of captured rows against the sealed-test index.
///
/// Sealed-test hits are checked first: such a row is refused outright, even if
/// it also carries a secret, so no test item is ever stored beside training data.
#[must_use]
pub fn admit(rows: &str, sealed: &Index) -> Admission {
    let verdicts = rows
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| (l.to_string(), verdict(l, sealed)))
        .collect();
    Admission { verdicts }
}

fn verdict(row: &str, sealed: &Index) -> Verdict {
    if sealed.is_empty() {
        return Verdict::Unchecked;
    }
    let mut items: Vec<String> = sealed
        .scan(row, "row")
        .into_iter()
        .map(|h| h.item)
        .collect();
    items.dedup();
    if !items.is_empty() {
        return Verdict::Refused { items };
    }
    let found = scan(row);
    if found.is_empty() {
        Verdict::Admitted
    } else {
        Verdict::Quarantined { secrets: found }
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
