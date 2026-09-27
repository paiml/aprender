//! OBS-04 self-report agreement (APR-OBS-001 §0.4, §7; contract `apr-selfreport-agreement-v1`).
//!
//! Server timings (`apr-serve-timing-v1`, OBS-03) are the system describing itself. They are
//! admissible only where they agree with the independent client measurement
//! (`apr-lane-row-v1`, OBS-01). This module joins the two ledgers by `request_id` and checks
//! `|total_ms - wall_ms| <= max(5% of wall_ms, 50 ms)` on every joined pair; the ledger pair
//! is GREEN only when at least 99% of the joined pairs agree. Both thresholds are `[A]`
//! (asserted, spec §7) until OBS-05 measures the real spread.
//!
//! Rules taken from the contracts, not re-decided here:
//! - an inadmissible row (identity block incomplete, a field `null` or `"unknown"`, the
//!   `gpu_proof` key absent, or the wrong `schema`) counts as absent (identity_is_complete);
//! - a join across mismatched `crate_tarball_sha256` / `model_sha256` / `backend` is refused,
//!   not flagged (identity_matches_before_any_ratio);
//! - an empty join is RED (R-2);
//! - a `request_id` that appears twice in one ledger makes the join ambiguous and is RED.

use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// `schema` value of an OBS-01 lane row.
pub const LANE_SCHEMA: &str = "apr-lane-row-v1";
/// `schema` value of an OBS-03 serve timing row.
pub const SERVE_SCHEMA: &str = "apr-serve-timing-v1";

/// The identity block every APR-OBS-001 row carries (`apr-obs-row-identity-v1`).
pub const IDENTITY: [&str; 12] = [
    "schema",
    "ts",
    "host",
    "apr_version",
    "apr_tag",
    "crate_tarball_sha256",
    "binary_sha256",
    "build_identity",
    "model_id",
    "model_sha256",
    "backend",
    "request_id",
];

/// The identity fields that must match before two rows are compared at all.
pub const RATIO_IDENTITY: [&str; 3] = ["crate_tarball_sha256", "model_sha256", "backend"];

/// Relative tolerance: 5% of the client wall time `[A]`.
pub const REL_TOLERANCE: f64 = 0.05;
/// Absolute floor of the tolerance: 50 ms `[A]`.
pub const ABS_TOLERANCE_MS: f64 = 50.0;
/// Minimum share of joined pairs that must agree, in percent `[A]`.
pub const MIN_AGREEING_PERCENT: u64 = 99;

/// One joined pair outside tolerance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Disagreement {
    /// The join key.
    pub request_id: String,
    /// Client-measured wall time (lane row).
    pub wall_ms: f64,
    /// Server-reported total (serve row).
    pub total_ms: f64,
    /// The tolerance this pair was held to.
    pub tolerance_ms: f64,
}

/// Result of checking one lane ledger against one serve ledger.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AgreementReport {
    /// Admissible lane rows.
    pub lane_rows: usize,
    /// Admissible serve rows.
    pub serve_rows: usize,
    /// Lane lines counted as absent, with the reason for each.
    pub inadmissible_lane: Vec<String>,
    /// Serve lines counted as absent, with the reason for each.
    pub inadmissible_serve: Vec<String>,
    /// Pairs joined on `request_id` with matching identity.
    pub joined: usize,
    /// Joined pairs within tolerance.
    pub agreeing: usize,
    /// Joined pairs outside tolerance.
    pub disagreements: Vec<Disagreement>,
    /// Lane rows with no serve row.
    pub unjoined_lane: usize,
    /// Serve rows with no lane row.
    pub unjoined_serve: usize,
    /// Reasons the check refused to compare (each is fatal).
    pub refusals: Vec<String>,
}

impl AgreementReport {
    /// GREEN iff nothing was refused, the join is non-empty, and >= 99% of pairs agree.
    #[must_use]
    pub fn is_green(&self) -> bool {
        self.red_reasons().is_empty()
    }

    /// Every reason the report is RED; empty means GREEN.
    #[must_use]
    pub fn red_reasons(&self) -> Vec<String> {
        let mut reasons = self.refusals.clone();
        if self.joined == 0 {
            reasons.push("empty join: no lane row and serve row share a request_id (R-2)".into());
        } else if (self.agreeing as u64) * 100 < MIN_AGREEING_PERCENT * self.joined as u64 {
            reasons.push(format!(
                "{} of {} joined rows agree, below {}%",
                self.agreeing, self.joined, MIN_AGREEING_PERCENT
            ));
        }
        reasons
    }
}

/// The tolerance a pair with this client wall time is held to.
#[must_use]
pub fn tolerance_ms(wall_ms: f64) -> f64 {
    (wall_ms * REL_TOLERANCE).max(ABS_TOLERANCE_MS)
}

/// Why a parsed row is not admissible, or `None` if it is.
fn inadmissible(row: &Map<String, Value>, schema: &str) -> Option<String> {
    for f in IDENTITY {
        match row.get(f) {
            None | Some(Value::Null) => return Some(format!("identity field {f} missing")),
            Some(Value::String(s)) if s == "unknown" => {
                return Some(format!("identity field {f} is \"unknown\""))
            }
            Some(_) => {}
        }
    }
    if !row.contains_key("gpu_proof") {
        return Some("gpu_proof key missing".into());
    }
    if row.get("schema").and_then(Value::as_str) != Some(schema) {
        return Some(format!("schema is not {schema}"));
    }
    if !row.get("request_id").is_some_and(Value::is_string) {
        return Some("request_id is not a string".into());
    }
    None
}

/// Parse a JSONL ledger into admissible rows keyed by `request_id`.
///
/// `time_field` must be a positive number on every admissible row. Duplicate ids are
/// recorded in `refusals`.
fn parse_ledger(
    text: &str,
    schema: &str,
    time_field: &str,
    rejects: &mut Vec<String>,
    refusals: &mut Vec<String>,
) -> BTreeMap<String, (Map<String, Value>, f64)> {
    let mut rows = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let n = i + 1;
        let row = match serde_json::from_str::<Value>(line) {
            Ok(Value::Object(m)) => m,
            Ok(_) => {
                rejects.push(format!("{schema} line {n}: not a JSON object"));
                continue;
            }
            Err(e) => {
                rejects.push(format!("{schema} line {n}: {e}"));
                continue;
            }
        };
        if let Some(why) = inadmissible(&row, schema) {
            rejects.push(format!("{schema} line {n}: {why}"));
            continue;
        }
        let t = match row.get(time_field).and_then(Value::as_f64) {
            Some(t) if t > 0.0 && t.is_finite() => t,
            _ => {
                rejects.push(format!(
                    "{schema} line {n}: {time_field} is not a positive number"
                ));
                continue;
            }
        };
        let id = row
            .get("request_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if rows.contains_key(&id) {
            refusals.push(format!(
                "{schema}: request_id {id} appears twice; the join is ambiguous"
            ));
            continue;
        }
        rows.insert(id, (row, t));
    }
    rows
}

/// Join a lane ledger and a serve ledger (JSONL text) and check agreement.
#[must_use]
pub fn check_agreement(lane_jsonl: &str, serve_jsonl: &str) -> AgreementReport {
    let mut r = AgreementReport::default();
    let lane = parse_ledger(
        lane_jsonl,
        LANE_SCHEMA,
        "wall_ms",
        &mut r.inadmissible_lane,
        &mut r.refusals,
    );
    let serve = parse_ledger(
        serve_jsonl,
        SERVE_SCHEMA,
        "total_ms",
        &mut r.inadmissible_serve,
        &mut r.refusals,
    );
    r.lane_rows = lane.len();
    r.serve_rows = serve.len();
    for (id, (lrow, wall_ms)) in &lane {
        let Some((srow, total_ms)) = serve.get(id) else {
            r.unjoined_lane += 1;
            continue;
        };
        if let Some(f) = RATIO_IDENTITY
            .iter()
            .find(|f| lrow.get(**f) != srow.get(**f))
        {
            r.refusals.push(format!(
                "request_id {id}: {f} differs between lane and serve rows; comparison refused"
            ));
            continue;
        }
        r.joined += 1;
        let tol = tolerance_ms(*wall_ms);
        if (total_ms - wall_ms).abs() <= tol {
            r.agreeing += 1;
        } else {
            r.disagreements.push(Disagreement {
                request_id: id.clone(),
                wall_ms: *wall_ms,
                total_ms: *total_ms,
                tolerance_ms: tol,
            });
        }
    }
    r.unjoined_serve = serve.keys().filter(|id| !lane.contains_key(*id)).count();
    r
}

#[cfg(test)]
#[path = "selfreport_agreement_tests.rs"]
mod tests;
