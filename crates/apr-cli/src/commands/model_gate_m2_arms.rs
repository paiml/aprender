//! C7 stock-baseline gate: M2 against extra arms (EXT-001 §10 C7, row EXT-30, aprender#4412).
//!
//! C7, verbatim: "**blocks promotion:** a candidate must be non-inferior to upstream
//! stock on every sealed suite (Holm) in addition to the M2 incumbent rule". The
//! statistics are "the same §3.6 statistics as M2", so non-inferior here means *not
//! significantly worse* under M2's worse family: no non-inferiority margin is
//! pre-registered, and this module does not invent one (R-13).
//!
//! Arms: upstream stock Qwen3.5-4B via llama.cpp blocks; the Qwen3.5-9B control and the
//! API arms (Haiku, agy) are report-only. An API arm has a model id and version but no
//! artifact sha, so it is non-hermetic and can never block.

use super::{apply_holm, m2, suite_report, validate, M2Prereg, M2Report, ReleaseClass, Suite};
use super::{SuiteData, SuiteReport, Verdict};
use serde::{Deserialize, Serialize};

/// What an arm is, as the receipt records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ArmIdentity {
    /// A hermetic artifact run locally (stock via llama.cpp, the 9B control).
    Artifact { name: String, sha256: String },
    /// A hosted model: non-hermetic, report-only.
    Api { model_id: String, version: String },
}

/// Whether an arm can refuse promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ArmRole {
    Blocking,
    ReportOnly,
}

/// One comparison arm: the candidate paired against this arm on every sealed suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Arm {
    pub identity: ArmIdentity,
    pub role: ArmRole,
    /// Same suite names and the same candidate items as the incumbent comparison;
    /// the baseline side is this arm.
    pub suites: Vec<Suite>,
}

/// One suite against one arm: M2's paired report plus the arm's own level, so a first
/// release records the stock baseline without claiming improvement over it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ArmSuite {
    /// Pass rate (binary) or mean score (scored) of the arm itself.
    pub arm_level: f64,
    #[serde(flatten)]
    pub report: SuiteReport,
}

/// What M2 found against one arm.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ArmReport {
    pub identity: ArmIdentity,
    pub role: ArmRole,
    pub suites: Vec<ArmSuite>,
}

/// M2 with the C7 arms: the M2 section plus one report per arm, and the combined verdict.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct GateReport {
    pub m2: M2Report,
    pub arms: Vec<ArmReport>,
    pub verdict: Verdict,
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn candidate_of(s: &Suite) -> (Option<&[bool]>, Option<&[f64]>) {
    match &s.data {
        SuiteData::Binary { candidate, .. } => (Some(candidate), None),
        SuiteData::Scored { candidate, .. } => (None, Some(candidate)),
    }
}

fn validate_arm(arm: &Arm, incumbent: &[Suite]) -> Result<(), String> {
    let label = match &arm.identity {
        ArmIdentity::Artifact { name, sha256 } => {
            if !is_sha256_hex(sha256) {
                return Err(format!("C7: arm {name}: sha256 is not 64 lowercase hex"));
            }
            name.clone()
        }
        ArmIdentity::Api { model_id, version } => {
            if arm.role == ArmRole::Blocking {
                return Err(format!(
                    "C7: API arm {model_id} is non-hermetic and cannot block"
                ));
            }
            if model_id.trim().is_empty() || version.trim().is_empty() {
                return Err("C7: an API arm records its model id and version".into());
            }
            model_id.clone()
        }
    };
    // Every arm is paired, whatever the release class: it always has a baseline side.
    validate(ReleaseClass::Improvement, &arm.suites)
        .map_err(|e| format!("C7: arm {label}: {e}"))?;
    let mut want: Vec<&str> = incumbent.iter().map(|s| s.name.as_str()).collect();
    let mut got: Vec<&str> = arm.suites.iter().map(|s| s.name.as_str()).collect();
    want.sort_unstable();
    got.sort_unstable();
    if want != got {
        return Err(format!(
            "C7: arm {label} covers {got:?}, the sealed suites are {want:?}"
        ));
    }
    for s in &arm.suites {
        let inc = incumbent
            .iter()
            .find(|i| i.name == s.name)
            .expect("names checked equal");
        if candidate_of(s) != candidate_of(inc) {
            return Err(format!(
                "C7: arm {label}, suite {}: not the same candidate items",
                s.name
            ));
        }
    }
    Ok(())
}

fn arm_level(s: &Suite) -> f64 {
    match &s.data {
        SuiteData::Binary { baseline, .. } => {
            let b = baseline.as_deref().unwrap_or_default();
            b.iter().filter(|x| **x).count() as f64 / b.len() as f64
        }
        SuiteData::Scored { baseline, .. } => {
            let b = baseline.as_deref().unwrap_or_default();
            b.iter().sum::<f64>() / b.len() as f64
        }
    }
}

fn arm_report(pre: &M2Prereg, arm: &Arm) -> ArmReport {
    let mut reports: Vec<SuiteReport> = arm.suites.iter().map(|s| suite_report(pre, s)).collect();
    apply_holm(pre, &mut reports);
    let suites = arm
        .suites
        .iter()
        .zip(reports)
        .map(|(s, report)| ArmSuite {
            arm_level: arm_level(s),
            report,
        })
        .collect();
    ArmReport {
        identity: arm.identity.clone(),
        role: arm.role,
        suites,
    }
}

fn arm_name(id: &ArmIdentity) -> &str {
    match id {
        ArmIdentity::Artifact { name, .. } => name,
        ArmIdentity::Api { model_id, .. } => model_id,
    }
}

fn combine(m2: &Verdict, arms: &[ArmReport]) -> Verdict {
    if *m2 != Verdict::Promote {
        return m2.clone();
    }
    for a in arms.iter().filter(|a| a.role == ArmRole::Blocking) {
        let reports = a.suites.iter().map(|s| &s.report);
        if let Some(r) = reports
            .clone()
            .find(|r| r.holm_worse.is_some_and(|h| h.rejected))
        {
            return Verdict::Reject {
                reason: format!(
                    "suite {} is significantly worse than {}",
                    r.name,
                    arm_name(&a.identity)
                ),
            };
        }
        if let Some(r) = reports
            .filter(|r| r.n_req.is_some())
            .max_by_key(|r| r.n_req)
        {
            return Verdict::Underpowered {
                n_req: r.n_req.unwrap_or_default(),
                suite: r.name.clone(),
            };
        }
    }
    Verdict::Promote
}

/// M2 against the incumbent, then C7 against every arm.
///
/// # Errors
///
/// Refuses (no verdict) when M2 refuses, when no arm blocks (the stock baseline is
/// mandatory: its absence is not a pass), or when an arm is malformed: an API arm set
/// to block, a bad artifact sha256, suites that differ from the sealed set, or candidate
/// items that differ from the incumbent comparison's.
pub(crate) fn gate(
    pre: &M2Prereg,
    class: ReleaseClass,
    incumbent: &[Suite],
    arms: &[Arm],
) -> Result<GateReport, String> {
    let m2 = m2(pre, class, incumbent)?;
    if !arms.iter().any(|a| a.role == ArmRole::Blocking) {
        return Err("C7: no blocking stock-baseline arm; its absence is not a pass".into());
    }
    for a in arms {
        validate_arm(a, incumbent)?;
    }
    let arms: Vec<ArmReport> = arms.iter().map(|a| arm_report(pre, a)).collect();
    let verdict = combine(&m2.verdict, &arms);
    Ok(GateReport { m2, arms, verdict })
}

#[cfg(test)]
#[path = "model_gate_m2_arms_tests.rs"]
mod tests;
