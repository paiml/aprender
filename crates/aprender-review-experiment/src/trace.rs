//! `agent-trace-v1`: one row per lane per quorum round, holding the full I/O
//! identity a fine-tune or distill run needs (operator ruling 2026-09-25,
//! items 1, 7, 10, 11). The row is the JSONL index entry; the bytes it names
//! (`input_sha256`, `output_sha256`, `logits_sha256`) are content-addressed
//! blobs on almacen, never in git. This module does not store anything: it
//! lints rows, assigns splits and computes the weekly receipt.
//!
//! - Any unknown identity field makes the row RED ([`lint`]).
//! - `label_tier` keeps silver (hosted-lane output) and gold (corpus or an
//!   outcome join) separable; gold without an outcome is RED.
//! - Splits are keyed by `repo#PR` and assigned by the PR's first-seen time,
//!   never by row, so two rows of one PR can never straddle train and test.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::ledger::{NotRun, Verdict};

pub const SCHEMA: &str = "agent-trace-v1";

/// Top-k the local lane records its logits at.
pub const LOCAL_TOP_K: u32 = 20;

/// Who served the model. Closed: a new provider is a schema change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Provider {
    Anthropic,
    Google,
    /// apr / qwen on a fleet host: the only provider that records logits.
    Local,
}

/// Silver = a hosted lane's output; gold = corpus or an outcome join (S-6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LabelTier {
    Silver,
    Gold,
}

/// Whether the lane ran. A lane with no executor still writes a row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Run {
    Ran {
        output_sha256: String,
        verdict: Option<Verdict>,
        findings: Vec<String>,
        tokens_in: u64,
        tokens_out: u64,
        latency_ms: u64,
    },
    NotRun(NotRun),
}

/// Blob sizes, filled by the store; `None` until it reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bytes {
    pub raw: u64,
    pub zstd: u64,
}

/// One `agent-trace-v1` row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceRow {
    pub schema: String,
    /// RFC 3339 UTC, `YYYY-MM-DDTHH:MM:SSZ`.
    pub at: String,
    pub quorum_id: String,
    pub repo: String,
    pub pr: u64,
    pub head: String,
    pub diff_sha256: String,
    pub lane: String,
    pub model_id: String,
    pub provider: Provider,
    pub served_by: String,
    /// sha of the exact bytes sent.
    pub input_sha256: String,
    /// Decoding parameters as sent; an empty object is a gap.
    pub params: BTreeMap<String, serde_json::Value>,
    pub run: Run,
    /// Local lane only: top-[`LOCAL_TOP_K`] logits blob.
    pub logits_sha256: Option<String>,
    pub label_tier: LabelTier,
    /// Must be `repo#PR`: the split unit is structural, not a convention.
    pub split_guard: String,
    // --- index columns (item 10) ---
    pub agreed: Option<bool>,
    /// This lane's verdict differs from another lane's in the same round.
    pub lane_disagreement: bool,
    /// `pending` until the outcome join (merged / reverted / escape).
    pub outcome: String,
    /// First row in the index carrying this `diff_sha256`.
    pub unique_diff: bool,
    pub bytes: Option<Bytes>,
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_utc(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 20
        && b[4] == b'-'
        && b[7] == b'-'
        && b[10] == b'T'
        && b[13] == b':'
        && b[16] == b':'
        && b[19] == b'Z'
}

fn known(s: &str) -> bool {
    let t = s.trim();
    !t.is_empty() && !t.eq_ignore_ascii_case("unknown") && !t.eq_ignore_ascii_case("null")
}

/// `repo#PR`, the split unit.
#[must_use]
pub fn split_key(repo: &str, pr: u64) -> String {
    format!("{repo}#{pr}")
}

/// RED reasons for one row; empty means the row is admissible to the index.
#[must_use]
pub fn lint(r: &TraceRow) -> Vec<String> {
    let mut e = Vec::new();
    if r.schema != SCHEMA {
        e.push(format!("schema {:?}", r.schema));
    }
    if !is_utc(&r.at) {
        e.push(format!("at {:?} is not RFC 3339 UTC", r.at));
    }
    for (k, v) in [
        ("quorum_id", &r.quorum_id),
        ("repo", &r.repo),
        ("head", &r.head),
        ("lane", &r.lane),
        ("model_id", &r.model_id),
        ("served_by", &r.served_by),
        ("outcome", &r.outcome),
    ] {
        if !known(v) {
            e.push(format!("{k} is unknown"));
        }
    }
    if r.pr == 0 {
        e.push("pr is unknown".into());
    }
    for (k, v) in [
        ("diff_sha256", &r.diff_sha256),
        ("input_sha256", &r.input_sha256),
    ] {
        if !is_sha256(v) {
            e.push(format!("{k} is not a sha256"));
        }
    }
    if r.params.is_empty() {
        e.push("params are unknown".into());
    }
    if r.split_guard != split_key(&r.repo, r.pr) {
        e.push(format!("split_guard {:?} is not repo#PR", r.split_guard));
    }
    if let Run::Ran { output_sha256, .. } = &r.run {
        if !is_sha256(output_sha256) {
            e.push("output_sha256 is not a sha256".into());
        }
        let local = r.provider == Provider::Local;
        match &r.logits_sha256 {
            Some(l) if !local => e.push(format!("logits_sha256 {l:?} on a hosted lane")),
            Some(l) if !is_sha256(l) => e.push("logits_sha256 is not a sha256".into()),
            None if local => e.push(format!("local lane ran without top-{LOCAL_TOP_K} logits")),
            _ => {}
        }
    }
    if r.label_tier == LabelTier::Gold && r.outcome == "pending" {
        e.push("gold without an outcome join".into());
    }
    e
}

/// Parse a JSONL index; a line that does not parse or lint is RED with its
/// line number, never dropped.
pub fn parse_index(text: &str) -> Result<Vec<TraceRow>, Vec<String>> {
    let mut rows = Vec::new();
    let mut red = Vec::new();
    for (i, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        match serde_json::from_str::<TraceRow>(line) {
            Ok(r) => {
                red.extend(lint(&r).into_iter().map(|m| format!("line {}: {m}", i + 1)));
                rows.push(r);
            }
            Err(err) => red.push(format!("line {}: {err}", i + 1)),
        }
    }
    if red.is_empty() {
        Ok(rows)
    } else {
        Err(red)
    }
}

/// Fill the derived index columns in arrival order: `unique_diff` (first
/// row for its diff) and `lane_disagreement` (a PASS/FAIL split in the round).
pub fn derive_columns(rows: &mut [TraceRow]) {
    let mut seen = BTreeSet::new();
    let mut round: BTreeMap<String, BTreeSet<Verdict>> = BTreeMap::new();
    for r in rows.iter() {
        if let Run::Ran {
            verdict: Some(v), ..
        } = r.run
        {
            round.entry(r.quorum_id.clone()).or_default().insert(v);
        }
    }
    for r in rows.iter_mut() {
        r.unique_diff = seen.insert(r.diff_sha256.clone());
        r.lane_disagreement = matches!(
            r.run,
            Run::Ran {
                verdict: Some(_),
                ..
            }
        ) && round.get(&r.quorum_id).is_some_and(|s| s.len() > 1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Split {
    Train,
    Dev,
    Test,
}

/// Assign each `repo#PR` to one split by the PR's first-seen `at`:
/// before `dev_from` → train, before `test_from` → dev, else test.
/// Every row of a PR inherits its PR's split.
#[must_use]
pub fn splits(rows: &[TraceRow], dev_from: &str, test_from: &str) -> BTreeMap<String, Split> {
    let mut first: BTreeMap<String, &str> = BTreeMap::new();
    for r in rows {
        let at = first.entry(r.split_guard.clone()).or_insert(&r.at);
        if r.at.as_str() < *at {
            *at = &r.at;
        }
    }
    first
        .into_iter()
        .map(|(k, at)| {
            let s = if at < dev_from {
                Split::Train
            } else if at < test_from {
                Split::Dev
            } else {
                Split::Test
            };
            (k, s)
        })
        .collect()
}

/// The weekly receipt (item 11). `bytes_*` is `None` when any row in the
/// week lacks sizes: an unmeasured total is not a small one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Weekly {
    pub schema: &'static str,
    /// Inclusive start, exclusive end, both `YYYY-MM-DD`.
    pub from: String,
    pub to: String,
    pub rows: usize,
    pub unique_diffs: usize,
    pub bytes_raw: Option<u64>,
    pub bytes_zstd: Option<u64>,
    /// `lane|tier|outcome` → rows.
    pub by_lane_tier_outcome: BTreeMap<String, usize>,
}

/// Build the weekly receipt over rows with `from <= at < to` (date prefixes).
#[must_use]
pub fn weekly(rows: &[TraceRow], from: &str, to: &str) -> Weekly {
    let week: Vec<&TraceRow> = rows
        .iter()
        .filter(|r| r.at.as_str() >= from && r.at.as_str() < to)
        .collect();
    let diffs: BTreeSet<&str> = week.iter().map(|r| r.diff_sha256.as_str()).collect();
    let sizes: Option<Vec<Bytes>> = week.iter().map(|r| r.bytes).collect();
    let mut by = BTreeMap::new();
    for r in &week {
        let tier = match r.label_tier {
            LabelTier::Silver => "silver",
            LabelTier::Gold => "gold",
        };
        *by.entry(format!("{}|{tier}|{}", r.lane, r.outcome))
            .or_insert(0) += 1;
    }
    Weekly {
        schema: "agent-trace-weekly-v1",
        from: from.into(),
        to: to.into(),
        rows: week.len(),
        unique_diffs: diffs.len(),
        bytes_raw: sizes.as_ref().map(|s| s.iter().map(|b| b.raw).sum()),
        bytes_zstd: sizes.as_ref().map(|s| s.iter().map(|b| b.zstd).sum()),
        by_lane_tier_outcome: by,
    }
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
