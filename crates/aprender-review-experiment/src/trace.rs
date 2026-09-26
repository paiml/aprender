//! PRM-C1 `agent-trace-v1`: one index row per lane per quorum round (spec
//! `research-design-pr-agent-qwen-3.5.md` §2.2, PRM-001 §2.2; contract
//! `agent-trace-v1`, FALSIFY-ATR-001..005).
//!
//! - **Row, not blob.** The row is the `index/agent-trace-v1/YYYY/MM/DD.jsonl`
//!   entry; the wire request and the full reply are CAS blobs on almacen,
//!   named here by `input_sha` / `output_sha`. No blob, transcript or finding
//!   text reaches git (PRIVATE).
//! - **Strict.** An unknown key, a missing key or an unknown enum value fails
//!   to parse. [`lint`] then makes a row RED when its identity is unknown
//!   (the [`crate::terms`] tags, `served_by`, the shas) or its fields
//!   disagree: a `Verdict` without an output, a `NotRun` with one, a local
//!   lane counted or without logits, a hosted output tiered gold, gold with
//!   a pending outcome, a `split_guard` not keyed by `repo#PR`.
//! - **Reused, not restated.** `outcome` is [`Outcome`], `split_guard` is
//!   [`Guard`], a `NotRun` reason is the ledger's [`NotRun`], and provider /
//!   channel / model id / terms are checked by [`terms::tags`].
//! - [`parse_index`] reads one index file; [`tally`] is the weekly receipt's
//!   composition (rows, unique diffs, rows by lane × tier × outcome).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ledger::NotRun;
use crate::outcome_join::Outcome;
use crate::split_guard::Guard;
use crate::terms::{self, Provider};

pub const SCHEMA: &str = "agent-trace-v1";

/// Whether the lane produced a reply (`lane_state`). A `NotRun` row has no
/// output blob and is still a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaneState {
    Verdict,
    Refused,
    NotRun(NotRun),
}

/// The parsed review verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewVerdict {
    Approve,
    RequestChanges,
    Comment,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParseStatus {
    Ok,
    Repaired,
    Failed,
}

/// Set by admission, never by the producer. Silver is a hosted lane's
/// output; gold is a corpus item or an outcome join.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelTier {
    Silver,
    Gold,
    Pending,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    Ok,
    CaptureFailed,
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TermsRef {
    pub url: String,
    pub effective_date: String,
    pub fetched_at: String,
}

/// One part of the wire request, so shared system prompts and diffs dedup
/// across lanes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputPart {
    pub role: String,
    pub blob_sha: String,
    pub bytes: u64,
}

/// Provider-reported where available, otherwise as sent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decoding {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema_sha: Option<String>,
}

/// Provider-reported; `null` is unreported, never 0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tokens {
    pub input: Option<u64>,
    pub output: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatencyMs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<u64>,
    pub total: u64,
}

/// One finding, as text: a count cannot train anything.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Finding {
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
    pub severity: String,
    pub category: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecretScan {
    pub scanner_versions: BTreeMap<String, String>,
    pub hits: u32,
    pub status: String,
}

/// Who wrote the row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    pub binary: String,
    pub version: String,
    pub git_sha: String,
}

/// One `agent-trace-v1` index row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceRow {
    pub schema: String,
    /// ULID: time-sortable, so a PR's first-seen time is its first row's.
    pub trace_id: String,
    pub quorum_id: String,
    pub round: u16,
    pub lane: String,
    pub counted: bool,
    pub provider: String,
    /// Exact: a dated snapshot id, or a local lane's weights sha256.
    pub model_id: String,
    pub access_channel: String,
    pub terms_ref: TermsRef,
    /// The apr release that served a local lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apr_tag: Option<String>,
    /// The cell that actually ran (records a parity-guard fallback).
    pub served_by: String,
    pub repo: String,
    pub pr: u64,
    pub head_sha: String,
    pub base_sha: String,
    pub diff_sha256: String,
    /// The full wire request.
    pub input_sha: String,
    pub input_parts: Vec<InputPart>,
    pub prompt_version: String,
    /// The full raw reply, never truncated; `None` only when nothing replied.
    pub output_sha: Option<String>,
    pub decoding: Decoding,
    pub tokens: Tokens,
    pub latency_ms: LatencyMs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_tps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tps: Option<f64>,
    pub lane_state: LaneState,
    pub verdict: Option<ReviewVerdict>,
    pub findings: Vec<Finding>,
    pub parse_status: ParseStatus,
    /// The `sparse-logits-v1` blob; local lanes only.
    pub logits_sha: Option<String>,
    pub label_tier: LabelTier,
    pub outcome: Outcome,
    pub split_guard: Guard,
    pub secret_scan: SecretScan,
    pub trace_status: TraceStatus,
    /// Whether this lane's verdict matched the round's decision; `None` when
    /// it gave none.
    pub agreed: Option<bool>,
    /// Counted lanes of the round whose verdict differs from this one.
    pub lane_disagreement: u32,
    pub producer: Producer,
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn is_sha256(s: &str) -> bool {
    is_hex(s, 64)
}

/// Crockford base32, 26 characters, first digit ≤ 7 (48-bit time).
fn is_ulid(s: &str) -> bool {
    const ALPHABET: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    s.len() == 26 && s.as_bytes()[0] <= b'7' && s.bytes().all(|b| ALPHABET.contains(&b))
}

/// The millisecond timestamp a ULID carries, or `None` if it is not one.
#[must_use]
pub fn ulid_ms(s: &str) -> Option<u64> {
    const ALPHABET: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    if !is_ulid(s) {
        return None;
    }
    s.bytes().take(10).try_fold(0u64, |acc, b| {
        let d = ALPHABET.iter().position(|&a| a == b)?;
        Some(acc * 32 + d as u64)
    })
}

/// Every reason `row` is RED; empty is GREEN. `raw` is the line's JSON, for
/// the [`terms::tags`] identity check.
#[must_use]
pub fn lint(row: &TraceRow, raw: &Value) -> Vec<String> {
    let mut e = Vec::new();
    let mut need = |ok: bool, why: &str| {
        if !ok {
            e.push(why.to_string());
        }
    };
    need(row.schema == SCHEMA, "schema is not agent-trace-v1");
    need(is_ulid(&row.trace_id), "trace_id is not a ULID");
    for (k, v) in [
        ("quorum_id", &row.quorum_id),
        ("lane", &row.lane),
        ("served_by", &row.served_by),
        ("repo", &row.repo),
        ("prompt_version", &row.prompt_version),
    ] {
        need(!v.trim().is_empty(), &format!("{k} is unknown"));
    }
    need(is_hex(&row.head_sha, 40), "head_sha is not a commit sha");
    need(is_hex(&row.base_sha, 40), "base_sha is not a commit sha");
    need(is_sha256(&row.diff_sha256), "diff_sha256 is not a sha256");
    need(is_sha256(&row.input_sha), "input_sha is not a sha256");
    need(!row.input_parts.is_empty(), "input_parts is empty");
    need(
        row.input_parts.iter().all(|p| is_sha256(&p.blob_sha)),
        "an input part's blob_sha is not a sha256",
    );
    need(
        row.output_sha.as_deref().is_none_or(is_sha256),
        "output_sha is not a sha256",
    );
    let replied = matches!(row.lane_state, LaneState::Verdict);
    need(
        replied == row.output_sha.is_some(),
        "output_sha must be present exactly when lane_state is Verdict",
    );
    need(
        replied || (row.verdict.is_none() && row.findings.is_empty()),
        "a lane that did not reply carries a verdict or findings",
    );
    need(
        !replied || row.parse_status == ParseStatus::Failed || row.verdict.is_some(),
        "a parsed reply has no verdict",
    );
    need(
        row.findings
            .iter()
            .all(|f| f.line_start <= f.line_end && !f.text.trim().is_empty()),
        "a finding has an empty text or an inverted line range",
    );
    lint_provider(row, raw, &mut e);
    lint_lifecycle(row, &mut e);
    e
}

fn lint_provider(row: &TraceRow, raw: &Value, e: &mut Vec<String>) {
    if let Err(gaps) = terms::tags(raw) {
        e.extend(gaps.iter().map(|g| format!("identity: {g:?}")));
    }
    let local = Provider::parse(&row.provider) == Some(Provider::Local);
    if local {
        if row.counted {
            e.push("a local lane is counted".into());
        }
        if row.apr_tag.as_deref().is_none_or(|t| t.trim().is_empty()) {
            e.push("a local lane has no apr_tag".into());
        }
        let replied = matches!(row.lane_state, LaneState::Verdict);
        if replied && !row.logits_sha.as_deref().is_some_and(is_sha256) {
            e.push("a local reply has no logits_sha".into());
        }
    } else if row.logits_sha.is_some() {
        e.push("a hosted lane carries logits_sha".into());
    }
    if !local && row.label_tier == LabelTier::Gold {
        e.push("a hosted output is tiered gold".into());
    }
}

fn lint_lifecycle(row: &TraceRow, e: &mut Vec<String>) {
    if row.label_tier == LabelTier::Gold && row.outcome == Outcome::Pending {
        e.push("gold with a pending outcome".into());
    }
    let key = format!("{}#{}", row.repo, row.pr);
    if row.split_guard.group_key != key {
        e.push(format!(
            "split_guard.group_key {} is not {key}",
            row.split_guard.group_key
        ));
    }
    if row.agreed.is_some() != matches!(row.lane_state, LaneState::Verdict) {
        e.push("agreed must be present exactly when lane_state is Verdict".into());
    }
}

/// Every row of one index file, or every RED line (`line N: why`). Blank
/// lines are skipped; a repeated `trace_id` is RED.
///
/// # Errors
/// Returns every reason any line is RED.
pub fn parse_index(text: &str) -> Result<Vec<TraceRow>, Vec<String>> {
    let (mut rows, mut errs, mut ids) = (Vec::new(), Vec::new(), BTreeSet::new());
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let n = i + 1;
        let parsed = serde_json::from_str::<Value>(line).and_then(|v| {
            let r = TraceRow::deserialize(&v)?;
            Ok((r, v))
        });
        match parsed {
            Err(err) => errs.push(format!("line {n}: {err}")),
            Ok((r, v)) => {
                errs.extend(lint(&r, &v).into_iter().map(|w| format!("line {n}: {w}")));
                if !ids.insert(r.trace_id.clone()) {
                    errs.push(format!("line {n}: trace_id {} repeated", r.trace_id));
                }
                rows.push(r);
            }
        }
    }
    if errs.is_empty() {
        Ok(rows)
    } else {
        Err(errs)
    }
}

/// The weekly receipt's composition (ruling item 11): bytes come from
/// [`crate::datacard::weekly`], the rest from here.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Tally {
    pub rows: u64,
    pub unique_diffs: u64,
    /// `lane/label_tier/outcome` → rows.
    pub by_lane_tier_outcome: BTreeMap<String, u64>,
}

fn wire<T: Serialize>(t: &T) -> String {
    serde_json::to_value(t)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default()
}

#[must_use]
pub fn tally(rows: &[TraceRow]) -> Tally {
    let mut t = Tally::default();
    let mut diffs = BTreeSet::new();
    for r in rows {
        t.rows += 1;
        diffs.insert(r.diff_sha256.as_str());
        let cell = format!("{}/{}/{}", r.lane, wire(&r.label_tier), wire(&r.outcome));
        *t.by_lane_tier_outcome.entry(cell).or_default() += 1;
    }
    t.unique_diffs = diffs.len() as u64;
    t
}

#[cfg(test)]
#[path = "trace_tests.rs"]
mod tests;
