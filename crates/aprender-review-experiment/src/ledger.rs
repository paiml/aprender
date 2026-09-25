//! REX-07 day-0 shadow lane: `review-ledger-v1` rows built from quorum
//! receipts (`docs/audits/quorum-*.json`), and the shadow-coverage check.
//!
//! The 4B lane runs inside the quorum rail as a non-voting advisory lane
//! (paiml-implement `advisory-lane.sh`, which records `.advisory_lane` in the
//! receipt). This module does not run the lane; it reads what the rail wrote
//! and holds it to §7 REX-07:
//! - every quorum after activation carries a shadow row (a receipt with no
//!   `advisory_lane` is **missing**, never silently skipped);
//! - a lane that did not answer is `Unknown{LaneUnavailable}`, not a verdict;
//! - the shadow is never counted, and the quorum width is the counted lanes
//!   alone, so a gx10-down round keeps its width.

use serde::{Deserialize, Serialize};

pub const SCHEME: &str = "review-ledger-v1";

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
}

#[derive(Debug, Deserialize)]
struct Advisory {
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    verdict: Option<String>,
    #[serde(default)]
    served_by: Option<String>,
    #[serde(default)]
    apr: Option<String>,
    /// The rail's uncounted flag (`counted` there is the counted round's verdict).
    #[serde(default)]
    counts: Option<bool>,
    #[serde(default)]
    why: Option<String>,
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
pub struct LaneRow {
    pub family: Option<String>,
    pub model: Option<String>,
    pub verdict: Option<String>,
    pub findings: usize,
}

/// Why the shadow lane has no verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Unknown {
    LaneUnavailable { why: Option<String> },
}

/// The shadow lane's result on one quorum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state")]
pub enum Shadow {
    Answered { verdict: String, served_by: String },
    Unknown { reason: Unknown },
}

/// One `review-ledger-v1` row (§5.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// The rail does not record it yet; a row without it is an identity gap.
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
}

impl Coverage {
    /// §7 REX-07 acceptance: every quorum carries a shadow row, none counted.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.quorums > 0 && self.missing.is_empty() && self.violations.is_empty()
    }
}

fn shadow(a: &Advisory) -> Shadow {
    match (a.state.as_deref(), &a.verdict, &a.served_by) {
        (Some("answered"), Some(v), Some(s)) => Shadow::Answered {
            verdict: v.clone(),
            served_by: s.clone(),
        },
        _ => Shadow::Unknown {
            reason: Unknown::LaneUnavailable {
                why: a.why.clone().or_else(|| a.state.clone()),
            },
        },
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
                    findings: l.findings.as_ref().map_or(0, Vec::len),
                })
                .collect(),
            shadow: shadow(a),
            apr_tag: a.apr.clone(),
            weights_sha256: None,
            outcome: "pending".into(),
        };
        if row.apr_tag.is_none() || row.weights_sha256.is_none() {
            c.identity_gaps += 1;
        }
        rows.push(row);
    }
    (rows, c)
}

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod tests;
