//! REX-12 B2 loop (§5.3 B2a..B2d), `review-b2-loop-v1`.
//!
//! - **Row status.** Each B2 tier waits on one `apr` verb accepting qwen35.
//!   A tier is `Ready` only on a positive acceptance receipt for its verb. A
//!   verb with a `refusal-receipt-v1` row in the refusal ledger reads
//!   `VerbRefused{removed_by}`. A verb with neither is `Unledgered`: nobody
//!   measured it, so it is NotRun too. Absence of a refusal is not
//!   acceptance; the ledger is a bucket, not a census.
//! - **Teacher dataset.** The 27B teacher writes top-k logits at m = 1 over
//!   the train pool (every non-test item). [`teacher_receipt`] refuses the
//!   dataset unless every row is a known non-test item, appears once, carries
//!   exactly k finite logits per position, and the file text contains no
//!   sealed diff sha or hunk. The receipt carries sha, count and 0 test hits.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::contamination::Index;
use crate::corpus::{sha256_hex, Item, Split};

pub const SCHEME: &str = "review-b2-loop-v1";
pub const LEDGER_SCHEMA: &str = "apr-refusal-ledger/v1";

/// The §5.3 B2 tiers and the verb each waits on.
pub const TIERS: [(&str, &str); 4] = [
    ("B2a", "distill"),
    ("B2b", "finetune"),
    ("B2c", "merge"),
    ("B2d", "quantize"),
];

/// One `refusal-receipt-v1` row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Refusal {
    pub verb: String,
    pub reason: String,
    pub exit_code: i32,
    pub removed_by: String,
}

/// `evidence/verbs/refusals.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ledger {
    pub schema: String,
    #[serde(default)]
    pub note: String,
    pub refusals: Vec<Refusal>,
}

/// `refusal-receipt-v1`: `v<major>.<minor>`, `never` or `unscheduled`.
#[must_use]
pub fn valid_removed_by(s: &str) -> bool {
    if matches!(s, "never" | "unscheduled") {
        return true;
    }
    let Some(v) = s.strip_prefix('v') else {
        return false;
    };
    let parts: Vec<&str> = v.split('.').collect();
    parts.len() == 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

/// Parse and validate a refusal ledger; one bad row rejects the file.
pub fn parse_ledger(text: &str) -> Result<Ledger, String> {
    let l: Ledger = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if l.schema != LEDGER_SCHEMA {
        return Err(format!("schema {:?} is not {LEDGER_SCHEMA}", l.schema));
    }
    let mut seen = BTreeSet::new();
    for r in &l.refusals {
        if !valid_removed_by(&r.removed_by) {
            return Err(format!(
                "{}: removed_by {:?} is not v<major>.<minor>, never or unscheduled",
                r.verb, r.removed_by
            ));
        }
        if r.reason.trim().is_empty() || r.exit_code == 0 {
            return Err(format!(
                "{}: a refusal needs a reason and a non-zero exit",
                r.verb
            ));
        }
        if !seen.insert(r.verb.as_str()) {
            return Err(format!("{}: two refusal rows", r.verb));
        }
    }
    Ok(l)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum State {
    /// The verb accepted qwen35: the tier's challengers go to §5.4.
    Ready {
        acceptance: String,
    },
    VerbRefused {
        removed_by: String,
        exit_code: i32,
    },
    /// No refusal row and no acceptance: unmeasured, so NotRun.
    Unledgered,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub schema: String,
    pub tier: String,
    pub verb: String,
    #[serde(flatten)]
    pub state: State,
}

/// The B2 rows. `accepted` maps a verb to its acceptance receipt (a sha or a
/// path); a verb that is both accepted and refused is an error, since one of
/// the two receipts is stale.
pub fn status(ledger: &Ledger, accepted: &BTreeMap<String, String>) -> Result<Vec<Row>, String> {
    let refused: BTreeMap<&str, &Refusal> = ledger
        .refusals
        .iter()
        .map(|r| (r.verb.as_str(), r))
        .collect();
    TIERS
        .iter()
        .map(|&(tier, verb)| {
            let state = match (accepted.get(verb), refused.get(verb)) {
                (Some(_), Some(_)) => {
                    return Err(format!(
                        "{verb}: both accepted and refused; one receipt is stale"
                    ))
                }
                (Some(a), None) => State::Ready {
                    acceptance: a.clone(),
                },
                (None, Some(r)) => State::VerbRefused {
                    removed_by: r.removed_by.clone(),
                    exit_code: r.exit_code,
                },
                (None, None) => State::Unledgered,
            };
            Ok(Row {
                schema: SCHEME.into(),
                tier: tier.into(),
                verb: verb.into(),
                state,
            })
        })
        .collect()
}

/// One teacher row: an item's top-k `(token, logit)` pairs per position.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherRow {
    pub item_id: String,
    pub top_k: Vec<Vec<(u32, f32)>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherReceipt {
    pub schema: String,
    pub teacher_weights_sha256: String,
    pub k: usize,
    /// Samples per item (§5.3 B2a: m = 1).
    pub m: usize,
    pub count: usize,
    pub sha256: String,
    pub sealed_items_indexed: usize,
    pub test_hits: usize,
}

/// Receipt a teacher-logit JSONL, or list every reason it is refused.
pub fn teacher_receipt(
    jsonl: &str,
    teacher_weights_sha256: &str,
    k: usize,
    items: &[Item],
    sealed: &Index,
) -> Result<TeacherReceipt, Vec<String>> {
    let mut e = Vec::new();
    if sealed.is_empty() {
        e.push("no sealed manifest indexed: 0 test hits would prove nothing".into());
    }
    if k == 0 {
        e.push("k = 0".into());
    }
    let split: BTreeMap<&str, Split> = items.iter().map(|i| (i.id.as_str(), i.split)).collect();
    let mut seen = BTreeSet::new();
    for (n, line) in jsonl
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let row: TeacherRow = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(x) => {
                e.push(format!("line {}: {x}", n + 1));
                continue;
            }
        };
        let id = row.item_id.as_str();
        match split.get(id) {
            None => e.push(format!("{id}: not a corpus item")),
            Some(Split::Test) => e.push(format!("{id}: a sealed test item (R-2)")),
            Some(Split::Dev) => {}
        }
        if !seen.insert(row.item_id.clone()) {
            e.push(format!("{id}: more than one row (m = 1)"));
        }
        if row.top_k.is_empty() {
            e.push(format!("{id}: no positions"));
        }
        if let Some(p) = row
            .top_k
            .iter()
            .position(|pos| pos.len() != k || pos.iter().any(|(_, x)| !x.is_finite()))
        {
            e.push(format!("{id}: position {p} is not {k} finite logits"));
        }
    }
    if seen.is_empty() {
        e.push("no teacher rows".into());
    }
    let hits = sealed.scan(jsonl, "teacher");
    if !hits.is_empty() {
        e.push(format!("{} sealed test leak(s): {hits:?}", hits.len()));
    }
    if !e.is_empty() {
        return Err(e);
    }
    Ok(TeacherReceipt {
        schema: SCHEME.into(),
        teacher_weights_sha256: teacher_weights_sha256.into(),
        k,
        m: 1,
        count: seen.len(),
        sha256: sha256_hex(jsonl.as_bytes()),
        sealed_items_indexed: sealed.len(),
        test_hits: 0,
    })
}

#[cfg(test)]
#[path = "b2_tests.rs"]
mod tests;
