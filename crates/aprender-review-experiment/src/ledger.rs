//! REX-07 day-0 shadow lane: `review-ledger-v2` rows built from quorum
//! receipts (`docs/audits/quorum-*.json`), and the shadow-coverage check.
//! v2 (PRA-001 T2) keeps each counted lane's findings text and its
//! agent-trace-v1 shas; v1 rows stay readable through [`read_row`] and
//! migrate with [`v1::Row::migrate`]. Outcomes are appended as
//! [`OutcomeJoin`]s; a written row is never rewritten.
//!
//! The 4B lane runs inside the quorum rail as a non-voting advisory lane
//! (paiml-implement `advisory-lane.sh`, which records `.advisory_lane` in the
//! receipt). This module does not run the lane; it reads what the rail wrote
//! and holds it to §7 REX-07:
//! - every quorum after activation carries a shadow row (a receipt with no
//!   `advisory_lane` is **missing**, never silently skipped);
//! - the apr row is one typed value, `Verdict | NotRun | Refused` (the
//!   paiml-implement#436 wire form, one spelling with receipt-lint); an
//!   unknown key or value fails to parse, so free text is never a reason;
//! - the shadow is never counted, and the quorum width is the counted lanes
//!   alone, so a gx10-down round keeps its width.

use serde::{Deserialize, Serialize};

pub const SCHEME: &str = "review-ledger-v2";

/// Rows written before v2 stay readable: see [`v1`] and [`read_row`].
pub const SCHEME_V1: &str = "review-ledger-v1";

#[path = "ledger_v1.rs"]
pub mod v1;

#[derive(Debug, Deserialize)]
struct Lane {
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    verdict: Option<String>,
    #[serde(default)]
    findings: Option<Vec<serde_json::Value>>,
    /// agent-trace-v1 blob shas (exact bytes sent / full raw reply).
    #[serde(default)]
    input_sha256: Option<String>,
    #[serde(default)]
    output_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Advisory {
    #[serde(default)]
    state: Option<String>,
    /// The typed apr row. Absent on an `off` lane; absent otherwise is a violation.
    #[serde(default)]
    row: Option<serde_json::Value>,
    #[serde(default)]
    apr: Option<String>,
    #[serde(default)]
    weights_sha256: Option<String>,
    /// The rail's uncounted flag (`counted` there is the counted round's verdict).
    #[serde(default)]
    counts: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct QuorumReceipt {
    ticket: String,
    head: String,
    #[serde(default)]
    pr: Option<u64>,
    #[serde(default)]
    diff_sha256: Option<String>,
    #[serde(default)]
    agreed: Option<bool>,
    #[serde(default)]
    width: Option<u64>,
    #[serde(default)]
    lanes: Vec<Lane>,
    #[serde(default)]
    advisory_lane: Option<Advisory>,
}

/// One counted lane, as the ledger keeps it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneRow {
    pub family: Option<String>,
    pub model: Option<String>,
    pub verdict: Option<String>,
    /// Carried by both versions; equals the text's length when text exists.
    pub findings_count: usize,
    /// The findings text, verbatim: a count cannot train anything. `None` is
    /// text never captured (a v1 row, or a receipt without the key): a trace
    /// gap, never an empty list.
    pub findings: Option<Vec<String>>,
    /// agent-trace-v1 blob shas; a lane without both is a trace gap.
    pub input_sha256: Option<String>,
    pub output_sha256: Option<String>,
}

/// The apr lane's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Verdict {
    #[serde(rename = "PASS")]
    Pass,
    #[serde(rename = "FAIL")]
    Fail,
}

/// Why the apr lane did not run. Closed: a new reason is a schema change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotRun {
    NoExecutor,
    Busy,
    Timeout,
    ContextOverflow,
    TrainActive,
}

/// What removed a cell from the ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RemovedBy {
    Ladder,
    GpuProof,
    Parse,
}

/// The apr lane's result on one quorum — the paiml-implement#436 wire form:
/// `{"Verdict":{"verdict":"PASS","cell":"gx10-cuda","backend":"cuda"}}`,
/// `{"NotRun":"Busy"}`, `{"Refused":{"cell":"gx10-cuda","removed_by":"gpu-proof"}}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Shadow {
    Verdict {
        verdict: Verdict,
        cell: String,
        backend: String,
    },
    NotRun(NotRun),
    Refused {
        cell: String,
        removed_by: RemovedBy,
    },
}

/// One `review-ledger-v2` row (§5.2; PRA-001 T2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub schema: String,
    pub repo: String,
    pub ticket: String,
    pub head: String,
    pub pr: Option<u64>,
    pub diff_sha256: Option<String>,
    pub agreed: Option<bool>,
    pub width: usize,
    pub lanes: Vec<LaneRow>,
    pub shadow: Shadow,
    pub apr_tag: Option<String>,
    /// From the receipt; a row without it is an identity gap.
    pub weights_sha256: Option<String>,
    /// Joined later (merged / reverted ≤ 14 d / escape); `pending` until then.
    pub outcome: String,
}

/// What a set of receipts resolves to.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Coverage {
    pub quorums: usize,
    pub carried: usize,
    /// Receipts with no shadow row: REX-07 requires 0.
    pub missing: Vec<String>,
    /// Receipts where the shadow was counted or changed the width.
    pub violations: Vec<String>,
    /// Rows without an apr tag or weights sha (the §5.2 identity fields).
    pub identity_gaps: usize,
    /// Counted lanes without findings text or both agent-trace-v1 shas: the
    /// Jidoka goal is 0.
    pub trace_gaps: usize,
}

impl Coverage {
    /// §7 REX-07 acceptance: every quorum carries a shadow row, none counted,
    /// and every row names its apr tag and weights sha (PRM v2 item 5).
    #[must_use]
    pub fn holds(&self) -> bool {
        self.quorums > 0
            && self.missing.is_empty()
            && self.violations.is_empty()
            && self.identity_gaps == 0
            && self.trace_gaps == 0
    }
}

/// A finding as text: a string stays verbatim, anything else is its JSON.
fn finding_text(v: &serde_json::Value) -> String {
    v.as_str().map_or_else(|| v.to_string(), str::to_owned)
}

/// An outcome join, appended to its own JSONL. Rows are never rewritten.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeJoin {
    pub repo: String,
    pub head: String,
    pub outcome: String,
    /// RFC 3339 UTC; the latest join for a (repo, head) wins.
    pub at: String,
}

/// A row's outcome read through the append-only joins, latest first;
/// the row's own value (`pending`) when nothing joined.
#[must_use]
pub fn outcome<'a>(row: &'a Row, joins: &'a [OutcomeJoin]) -> &'a str {
    joins
        .iter()
        .filter(|j| j.repo == row.repo && j.head == row.head)
        .max_by(|a, b| a.at.cmp(&b.at))
        .map_or(row.outcome.as_str(), |j| j.outcome.as_str())
}

/// A counted lane whose trace a fine-tune could not use: no findings text,
/// or not both agent-trace-v1 shas.
fn trace_gap(l: &LaneRow) -> bool {
    l.findings.is_none() || l.input_sha256.is_none() || l.output_sha256.is_none()
}

/// A ledger line of either version, dispatched on its `schema`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnyRow {
    V1(v1::Row),
    V2(Row),
}

/// Read one ledger line. An unknown `schema` is an error, never a guess.
pub fn read_row(line: &str) -> Result<AnyRow, String> {
    #[derive(Deserialize)]
    struct Head {
        schema: String,
    }
    let h: Head = serde_json::from_str(line).map_err(|e| e.to_string())?;
    match h.schema.as_str() {
        SCHEME_V1 => serde_json::from_str(line)
            .map(AnyRow::V1)
            .map_err(|e| format!("{SCHEME_V1}: {e}")),
        SCHEME => serde_json::from_str(line)
            .map(AnyRow::V2)
            .map_err(|e| format!("{SCHEME}: {e}")),
        s => Err(format!("unknown ledger schema {s:?}")),
    }
}

/// Build ledger rows from `(name, json)` receipts and measure shadow coverage.
/// A receipt that does not parse is a violation, never dropped.
#[must_use]
pub fn build(receipts: &[(String, String)], repo: &str) -> (Vec<Row>, Coverage) {
    let mut rows = Vec::new();
    let mut c = Coverage {
        quorums: receipts.len(),
        ..Coverage::default()
    };
    for (name, text) in receipts {
        let q: QuorumReceipt = match serde_json::from_str(text) {
            Ok(q) => q,
            Err(e) => {
                c.violations.push(format!("{name}: {e}"));
                continue;
            }
        };
        // `off` is the rail saying the lane was not run: no shadow row.
        let Some(a) = q
            .advisory_lane
            .as_ref()
            .filter(|a| a.state.as_deref() != Some("off"))
        else {
            c.missing.push(name.clone());
            continue;
        };
        let shadow = match a.row.clone().map(serde_json::from_value::<Shadow>) {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                c.violations.push(format!("{name}: apr row: {e}"));
                continue;
            }
            None => {
                c.violations.push(format!("{name}: no typed apr row"));
                continue;
            }
        };
        c.carried += 1;
        if a.counts != Some(false) {
            c.violations
                .push(format!("{name}: shadow lane not recorded as uncounted"));
        }
        let width = q.lanes.len();
        if q.width.is_some_and(|w| w != width as u64) {
            c.violations.push(format!(
                "{name}: width {:?} is not the {width} counted lanes",
                q.width
            ));
        }
        let row = Row {
            schema: SCHEME.into(),
            repo: repo.into(),
            ticket: q.ticket,
            head: q.head,
            pr: q.pr,
            diff_sha256: q.diff_sha256,
            agreed: q.agreed,
            width,
            lanes: q
                .lanes
                .iter()
                .map(|l| LaneRow {
                    family: l.family.clone(),
                    model: l.model.clone(),
                    verdict: l.verdict.clone(),
                    findings_count: l.findings.as_ref().map_or(0, Vec::len),
                    findings: l
                        .findings
                        .as_ref()
                        .map(|f| f.iter().map(finding_text).collect()),
                    input_sha256: l.input_sha256.clone(),
                    output_sha256: l.output_sha256.clone(),
                })
                .collect(),
            shadow,
            apr_tag: a.apr.clone(),
            weights_sha256: a.weights_sha256.clone(),
            outcome: "pending".into(),
        };
        if row.apr_tag.is_none() || row.weights_sha256.is_none() {
            c.identity_gaps += 1;
        }
        c.trace_gaps += row.lanes.iter().filter(|l| trace_gap(l)).count();
        rows.push(row);
    }
    (rows, c)
}

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod tests;
