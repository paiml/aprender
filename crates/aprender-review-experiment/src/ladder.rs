//! PRM-09 promotion ladder (PRM-001 v3 §5.4; contract `rex-promotion-ladder-v2`,
//! which supersedes the REX-09 shadow → tripwire → vote ladder of v1).
//!
//! The ladder reads only a `prm-001-report-v1` report (the §9 template:
//! `spec: PRM-001, version: 3`) and fails closed:
//! - **shadow** is the default, and anything the ladder cannot read keeps it;
//! - **tripwire** needs H4 and the replay p95 within the queue budget;
//! - **tie-breaker** (decides only a 1–1 split) also needs H5, H7, H1, H2 and
//!   7-day availability ≥ 95%;
//! - **vote** also needs ≥ 20 decided tie-breaks whose 14-day outcome agreement
//!   is at least the voters'.
//!
//! H4/H5 are decided by the Holm-adjusted p; the CI is report-only. H1/H2 (count
//! tests) and H7 (δ-free, outside the Holm family) are decided by their verdict.
//! `hardware_ruling.rung_eligible` caps the rung. Lambda is rung-2 shadow only
//! (R-9): a lambda or unnamed primary cell caps the lane at tripwire (S-4).
//!
//! [`evidence`] is the rung one report grants; [`decide`] steps the lane from
//! its current rung by at most one rung either way, except that S-4 caps at once.

use serde::{Deserialize, Serialize};

pub const SPEC: &str = "PRM-001";
pub const SPEC_VERSION: u64 = 3;
/// §5.4: 7-day availability for tie-breaker.
pub const MIN_AVAILABILITY: f64 = 0.95;
/// §5.4: decided tie-breaks for vote.
pub const MIN_TIE_BREAKS: u64 = 20;

/// A rung of the ladder, lowest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Shadow,
    Tripwire,
    TieBreaker,
    Vote,
}

impl Mode {
    const ALL: [Mode; 4] = [Mode::Shadow, Mode::Tripwire, Mode::TieBreaker, Mode::Vote];

    fn up(self) -> Mode {
        Self::ALL[(self as usize + 1).min(3)]
    }

    fn down(self) -> Mode {
        Self::ALL[(self as usize).saturating_sub(1)]
    }
}

#[derive(Debug, Deserialize)]
struct Hypothesis {
    id: String,
    /// Holm-adjusted one-sided p. `ci` may be present and is ignored.
    #[serde(default)]
    p_holm: Option<f64>,
    #[serde(default)]
    verdict: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Results {
    #[serde(default)]
    hypotheses: Vec<Hypothesis>,
}

#[derive(Debug, Default, Deserialize)]
struct Lane {
    #[serde(default)]
    availability_7d: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct Speed {
    #[serde(default)]
    within_budget: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct Ruling {
    #[serde(default)]
    primary: Option<String>,
    #[serde(default)]
    rung_eligible: Option<Mode>,
}

#[derive(Debug, Default, Deserialize)]
struct Promotion {
    #[serde(default)]
    tie_breaks_decided: Option<u64>,
    #[serde(default)]
    tie_break_outcome_agreement: Option<f64>,
    #[serde(default)]
    voter_outcome_agreement: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Report {
    #[serde(default)]
    spec: Option<String>,
    #[serde(default)]
    version: Option<u64>,
    prereg_sha: String,
    #[serde(default)]
    exploratory: bool,
    #[serde(default)]
    lane: Lane,
    #[serde(default)]
    speed: Speed,
    #[serde(default)]
    results: Results,
    #[serde(default)]
    hardware_ruling: Option<Ruling>,
    #[serde(default)]
    promotion: Promotion,
}

/// The ladder's decision and every reason it stopped below `vote`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub mode: Mode,
    pub reasons: Vec<String>,
}

fn verdict_holds<'a>(r: &'a Report, id: &str) -> Result<&'a Hypothesis, String> {
    let h = r
        .results
        .hypotheses
        .iter()
        .find(|h| h.id == id)
        .ok_or_else(|| format!("{id}: not in the report"))?;
    if h.verdict.as_deref() == Some("holds") {
        Ok(h)
    } else {
        Err(format!("{id}: verdict {:?}", h.verdict))
    }
}

/// H4/H5: an explicit `holds` verdict whose Holm-adjusted p is finite and ≤ α.
fn holm_holds(r: &Report, id: &str) -> Result<(), String> {
    match verdict_holds(r, id)?.p_holm {
        Some(p) if p.is_finite() && (0.0..=crate::stats::ALPHA).contains(&p) => Ok(()),
        p => Err(format!(
            "{id}: Holm p {p:?} is not ≤ {}",
            crate::stats::ALPHA
        )),
    }
}

fn shadow(reason: String) -> Decision {
    Decision {
        mode: Mode::Shadow,
        reasons: vec![reason],
    }
}

/// The rung the evidence alone grants, and every unmet gate.
fn gates(r: &Report, reasons: &mut Vec<String>) -> Mode {
    let mut trip = Vec::new();
    trip.extend(holm_holds(r, "H4").err());
    if r.speed.within_budget != Some(true) {
        trip.push(format!(
            "speed: replay p95 within the queue budget is {:?}",
            r.speed.within_budget
        ));
    }
    if !trip.is_empty() {
        reasons.extend(trip);
        return Mode::Shadow;
    }
    let mut tb: Vec<String> = [
        holm_holds(r, "H5"),
        verdict_holds(r, "H7").map(|_| ()),
        verdict_holds(r, "H1").map(|_| ()),
        verdict_holds(r, "H2").map(|_| ()),
    ]
    .into_iter()
    .filter_map(Result::err)
    .collect();
    match r.lane.availability_7d {
        Some(a) if a >= MIN_AVAILABILITY => {}
        a => tb.push(format!("availability_7d {a:?} is not ≥ {MIN_AVAILABILITY}")),
    }
    if !tb.is_empty() {
        reasons.extend(tb);
        return Mode::Tripwire;
    }
    let p = &r.promotion;
    let decided = p.tie_breaks_decided.unwrap_or(0);
    if decided < MIN_TIE_BREAKS {
        reasons.push(format!(
            "tie_breaks_decided {decided} is not ≥ {MIN_TIE_BREAKS}"
        ));
        return Mode::TieBreaker;
    }
    match (p.tie_break_outcome_agreement, p.voter_outcome_agreement) {
        (Some(t), Some(v)) if t >= v => Mode::Vote,
        (t, v) => {
            reasons.push(format!(
                "tie-break outcome agreement {t:?} is not ≥ the voters' {v:?}"
            ));
            Mode::TieBreaker
        }
    }
}

/// `true` when the primary cell is lambda or unnamed (R-9: lambda may not sit
/// at tie-breaker or above).
fn s4(r: &Report) -> bool {
    r.hardware_ruling
        .as_ref()
        .and_then(|h| h.primary.as_deref())
        .is_none_or(|p| p.starts_with("lambda"))
}

fn read(report_json: &str, prereg_sha: &str) -> Result<Report, Decision> {
    let r: Report = serde_json::from_str(report_json)
        .map_err(|e| shadow(format!("report does not parse: {e}")))?;
    if r.spec.as_deref() != Some(SPEC) || r.version != Some(SPEC_VERSION) {
        return Err(shadow(format!(
            "spec {:?} version {:?} is not {SPEC} v{SPEC_VERSION} (prm-001-report-v1)",
            r.spec, r.version
        )));
    }
    if r.prereg_sha != prereg_sha {
        return Err(shadow("prereg_sha differs from the lock".into()));
    }
    if r.exploratory {
        return Err(shadow("exploratory report (R-1)".into()));
    }
    Ok(r)
}

/// The rung one report grants, capped by its hardware ruling. Never errors:
/// an unreadable, foreign, exploratory or stale report is shadow.
#[must_use]
pub fn evidence(report_json: &str, prereg_sha: &str) -> Decision {
    match read(report_json, prereg_sha) {
        Ok(r) => capped(&r),
        Err(d) => d,
    }
}

fn capped(r: &Report) -> Decision {
    let mut reasons = Vec::new();
    let mut mode = gates(r, &mut reasons);
    let Some(h) = &r.hardware_ruling else {
        reasons.push("no hardware ruling".into());
        return Decision {
            mode: Mode::Shadow,
            reasons,
        };
    };
    let cap = h.rung_eligible.unwrap_or(Mode::Shadow);
    if cap < mode {
        reasons.push(format!("hardware ruling caps the lane at {cap:?}"));
        mode = cap;
    }
    if s4(r) && mode > Mode::Tripwire {
        reasons.push(format!(
            "S-4: primary cell {:?} is lambda or unnamed; lambda is shadow-only (R-9)",
            h.primary
        ));
        mode = Mode::Tripwire;
    }
    Decision { mode, reasons }
}

/// Step the lane from `current` on one report: at most one rung up, and a
/// demotion drops exactly one rung. A lambda or unnamed primary cell above
/// tripwire is capped at once (S-4), the one exception to the one-rung rule.
#[must_use]
pub fn decide(report_json: &str, prereg_sha: &str, current: Mode) -> Decision {
    let (mut d, lambda) = match read(report_json, prereg_sha) {
        Ok(r) => (capped(&r), s4(&r)),
        Err(d) => (d, false),
    };
    if lambda && current > Mode::Tripwire {
        d.mode = Mode::Tripwire;
        return d;
    }
    let stepped = d.mode.clamp(current.down(), current.up());
    if stepped != d.mode {
        d.reasons.push(format!(
            "one rung per report: {current:?} → {stepped:?}, not {:?}",
            d.mode
        ));
        d.mode = stepped;
    }
    d
}

#[cfg(test)]
#[path = "ladder_tests.rs"]
mod tests;
