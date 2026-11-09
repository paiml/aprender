//! The frozen `review-ledger-v1` row, kept so rows written before v2 stay
//! readable. Two shapes were written under that one schema string: the
//! REX-07 original (402aff870, a `state`-tagged shadow) and PRM v2 (a)
//! (89d994bc1, the typed paiml-implement#436 shadow). Both parse here.
//! Nothing writes v1 any more.

use serde::Deserialize;

use super::Shadow;

/// A v1 counted lane: findings were stored as a count only.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneRow {
    pub family: Option<String>,
    pub model: Option<String>,
    pub verdict: Option<String>,
    pub findings: usize,
}

/// 402aff870's reason for a missing shadow verdict.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum Unknown {
    LaneUnavailable { why: Option<String> },
}

/// 402aff870's shadow, before the #436 wire form existed.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "state")]
pub enum LegacyShadow {
    Answered { verdict: String, served_by: String },
    Unknown { reason: Unknown },
}

/// Either v1 shadow; the typed form is tried first.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum V1Shadow {
    Typed(Shadow),
    Legacy(LegacyShadow),
}

/// One `review-ledger-v1` row.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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
    pub shadow: V1Shadow,
    pub apr_tag: Option<String>,
    pub weights_sha256: Option<String>,
    pub outcome: String,
}

impl Row {
    /// Migrate to v2. The finding count carries over. Findings text and
    /// trace shas were never captured, so they stay `None` (a trace gap),
    /// never an empty list. A legacy shadow has no cell or backend, so it
    /// has no typed #436 form: it is refused and stays readable as v1.
    pub fn migrate(self) -> Result<super::Row, String> {
        let shadow = match self.shadow {
            V1Shadow::Typed(s) => s,
            V1Shadow::Legacy(l) => {
                return Err(format!(
                    "{}@{}: legacy shadow {l:?} has no typed #436 form",
                    self.ticket, self.head
                ))
            }
        };
        Ok(super::Row {
            schema: super::SCHEME.into(),
            repo: self.repo,
            ticket: self.ticket,
            head: self.head,
            pr: self.pr,
            diff_sha256: self.diff_sha256,
            agreed: self.agreed,
            width: self.width,
            lanes: self
                .lanes
                .into_iter()
                .map(|l| super::LaneRow {
                    family: l.family,
                    model: l.model,
                    verdict: l.verdict,
                    findings_count: l.findings,
                    findings: None,
                    input_sha256: None,
                    output_sha256: None,
                })
                .collect(),
            shadow,
            apr_tag: self.apr_tag,
            weights_sha256: self.weights_sha256,
            outcome: self.outcome,
        })
    }
}
