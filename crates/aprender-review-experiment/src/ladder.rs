//! REX-09 promotion ladder: H4/H5 as the shadow → tripwire → vote gates.
//!
//! The ladder reads only a `rex-001-report-v1` report (§9; JSON) and later
//! §5.4 reports of the same schema. It is sequential and fails closed:
//! - **shadow** is the default, and anything the ladder cannot read keeps it;
//! - **tripwire** needs H4 (4B precision ≥ the lowest voting lane's; the
//!   paired-bootstrap CI lower bound ≥ 0);
//! - **vote** needs H4 and H5 (4B recall ≥ the Claude lane's, same test).
//!
//! H5 without H4 is still shadow (R-11: apr casts no deciding vote before
//! both hold). The §4.5 queue budget caps the rung: a `hardware_ruling.mode`
//! of `shadow` or `tripwire` is the most the ladder grants.

use serde::{Deserialize, Serialize};

pub const SCHEME: &str = "rex-001-report-v1";

/// A rung of the ladder, lowest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Shadow,
    Tripwire,
    Vote,
}

#[derive(Debug, Deserialize)]
struct Hypothesis {
    id: String,
    #[serde(default)]
    ci: Option<[f64; 2]>,
    #[serde(default)]
    verdict: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Results {
    #[serde(default)]
    hypotheses: Vec<Hypothesis>,
}

#[derive(Debug, Deserialize)]
struct Ruling {
    mode: Mode,
}

#[derive(Debug, Deserialize)]
struct Report {
    schema: String,
    prereg_sha: String,
    #[serde(default)]
    exploratory: bool,
    #[serde(default)]
    results: Option<Results>,
    #[serde(default)]
    hardware_ruling: Option<Ruling>,
}

/// The ladder's decision and every reason it stopped below `vote`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub mode: Mode,
    pub reasons: Vec<String>,
}

/// H4/H5 hold only on an explicit `holds` verdict with a finite CI whose
/// lower bound is ≥ 0. Any other shape is "does not hold", with the reason.
fn holds(r: &Report, id: &str) -> Result<(), String> {
    let h = r
        .results
        .as_ref()
        .and_then(|x| x.hypotheses.iter().find(|h| h.id == id))
        .ok_or_else(|| format!("{id}: not in the report"))?;
    if h.verdict.as_deref() != Some("holds") {
        return Err(format!("{id}: verdict {:?}", h.verdict));
    }
    match h.ci {
        Some([lo, hi]) if lo.is_finite() && hi.is_finite() && lo >= 0.0 => Ok(()),
        ci => Err(format!("{id}: CI {ci:?} lower bound is not ≥ 0")),
    }
}

fn shadow(reason: String) -> Decision {
    Decision {
        mode: Mode::Shadow,
        reasons: vec![reason],
    }
}

/// Decide the lane's rung from one report. Never errors: an unreadable,
/// foreign, exploratory or stale report is a reason to stay in shadow.
#[must_use]
pub fn decide(report_json: &str, prereg_sha: &str) -> Decision {
    let r: Report = match serde_json::from_str(report_json) {
        Ok(r) => r,
        Err(e) => return shadow(format!("report does not parse: {e}")),
    };
    if r.schema != SCHEME {
        return shadow(format!("schema {} is not {SCHEME}", r.schema));
    }
    if r.prereg_sha != prereg_sha {
        return shadow("prereg_sha differs from the lock".into());
    }
    if r.exploratory {
        return shadow("exploratory report (R-1)".into());
    }
    let mut reasons = Vec::new();
    let mut mode = match (holds(&r, "H4"), holds(&r, "H5")) {
        (Ok(()), Ok(())) => Mode::Vote,
        (Ok(()), Err(e)) => {
            reasons.push(e);
            Mode::Tripwire
        }
        (Err(e4), h5) => {
            reasons.push(e4);
            reasons.extend(h5.err());
            Mode::Shadow
        }
    };
    match &r.hardware_ruling {
        Some(h) if h.mode < mode => {
            reasons.push(format!("§4.5 queue budget caps the lane at {:?}", h.mode));
            mode = h.mode;
        }
        Some(_) => {}
        None => {
            reasons.push("no hardware ruling".into());
            mode = Mode::Shadow;
        }
    }
    Decision { mode, reasons }
}

#[cfg(test)]
#[path = "ladder_tests.rs"]
mod tests;
