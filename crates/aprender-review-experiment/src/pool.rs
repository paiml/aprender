//! Training-pool admission gate (ruling agent-trace item 6; PRA-001 T6,
//! contract `trace-admission-secret-v1`).
//!
//! A captured row enters the fine-tune/distill pool only when it is secret-scan
//! clean under BOTH scanners of [`crate::secret`] AND carries no sealed-test
//! hash (`review-corpus-contamination-v1`).
//! A secret hit QUARANTINES the row as-is: it is never redacted in place, since a
//! redacted row is a row the scanner already failed on once. A sealed-test hit
//! REFUSES the row.
//!
//! G-PROV (PRM-001 v3 R-18, contract `trace-admission-prov-v1`): a row is
//! train-eligible only when it is a local row (qwen, the 27B teacher: provider
//! `local`, fully terms-tagged per [`crate::terms::tags`]) or a gold label
//! (`label_source` one of [`GOLD_SOURCES`]). Hosted outputs are never
//! eligible, whatever else the row claims, and an untagged row cannot be shown
//! not to be hosted, so it is ineligible too.

use serde_json::Value;

use crate::contamination::Index;
use crate::secret::{scan, Hit};
use crate::terms::{tags, Gap, Provider};

/// The `label_source` values that make a row a gold label (v3 G-PROV): a
/// matured outcome, an HRQ ruling, a sealed-corpus label.
pub const GOLD_SOURCES: [&str; 3] = ["outcome", "hrq_ruling", "sealed_corpus"];

/// Why a row fails G-PROV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prov {
    /// A hosted provider's output: never a training target (R-18).
    Hosted(Provider),
    /// Terms tags missing or inconsistent: provenance unproven.
    Untagged(Vec<Gap>),
    /// A `label_source` that is not a gold source.
    NotGold(String),
}

/// Where one input row went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Clean: the row is in the pool.
    Admitted,
    /// A secret was found; the row is held byte-for-byte outside the pool.
    Quarantined { secrets: Vec<Hit> },
    /// The row carries a sealed test item; it never enters the pool.
    Refused { items: Vec<String> },
    /// Not a local row nor a gold label (G-PROV); it never enters the pool.
    Ineligible(Prov),
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
    // Before the secret scan, so a hosted row never reaches the quarantine,
    // whose rows may be released after review.
    if let Err(p) = provenance(row) {
        return Verdict::Ineligible(p);
    }
    let found = scan(row);
    if found.is_empty() {
        Verdict::Admitted
    } else {
        Verdict::Quarantined { secrets: found }
    }
}

/// G-PROV for one row.
///
/// # Errors
/// The row is hosted, untagged, or labelled by a non-gold source.
pub fn provenance(row: &str) -> Result<(), Prov> {
    let Ok(v) = serde_json::from_str::<Value>(row) else {
        return Err(Prov::Untagged(vec![Gap::NotJson]));
    };
    // A hosted provider refuses first: a gold label never carries a hosted output.
    if let Some(p) = v
        .get("provider")
        .and_then(Value::as_str)
        .and_then(Provider::parse)
    {
        if p != Provider::Local {
            return Err(Prov::Hosted(p));
        }
    }
    if let Some(s) = v.get("label_source") {
        return match s.as_str() {
            Some(g) if GOLD_SOURCES.contains(&g) => Ok(()),
            _ => Err(Prov::NotGold(s.to_string())),
        };
    }
    // Hosted providers returned above, so tagged here means local.
    tags(&v).map(|_| ()).map_err(Prov::Untagged)
}

/// Test fixture: `v` with the terms tags of a local (apr-serve) row added.
#[cfg(test)]
pub(crate) fn local_tagged(mut v: Value) -> String {
    let terms = r#"{"url":"https://www.anthropic.com/legal/commercial-terms","effective_date":"2025-06-17","fetched_at":"2026-09-25T17:00:00Z"}"#;
    if let Value::Object(m) = &mut v {
        m.insert("provider".into(), "local".into());
        m.insert(
            "model_id".into(),
            "3f1a9c0b5e2d4f6a8b7c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c".into(),
        );
        m.insert("access_channel".into(), "apr-serve".into());
        m.insert(
            "terms_ref".into(),
            serde_json::from_str(terms).expect("fixture terms_ref is JSON"),
        );
    }
    v.to_string()
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
