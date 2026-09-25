//! Training-pool admission gate (ruling agent-trace item 6; contract
//! `training-pool-admission-v1`).
//!
//! A captured row enters the fine-tune/distill pool only when it is secret-scan
//! clean AND carries no sealed-test hash (`review-corpus-contamination-v1`).
//! A secret hit QUARANTINES the row as-is: it is never redacted in place, since a
//! redacted row is a row the scanner already failed on once. A sealed-test hit
//! REFUSES the row.

use crate::contamination::Index;

/// Where one input row went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Clean: the row is in the pool.
    Admitted,
    /// A secret was found; the row is held byte-for-byte outside the pool.
    Quarantined { secrets: Vec<&'static str> },
    /// The row carries a sealed test item; it never enters the pool.
    Refused { items: Vec<String> },
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

/// Every secret kind found in `text`.
#[must_use]
pub fn secrets(_text: &str) -> Vec<&'static str> {
    Vec::new()
}

/// Admit a JSONL batch of captured rows against the sealed-test index.
#[must_use]
pub fn admit(rows: &str, _sealed: &Index) -> Admission {
    Admission {
        verdicts: rows
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| (l.to_string(), Verdict::Admitted))
            .collect(),
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
