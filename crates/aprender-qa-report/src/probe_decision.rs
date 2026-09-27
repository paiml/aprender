//! OBS-12 hourly-probe decision report (APR-OBS-001 §7 row OBS-12, aprender#4499).
//!
//! The question OBS-12 puts to the operator: after 14 nights of the nightly perf ledger
//! (`apr-perf-ledger-v1`, OBS-05), did any RED go undetected for more than 12 h between
//! nightlies? If so, an hourly probe would have shortened it; if not, the nightly suffices.
//! This module only builds the evidence. The probe is built only if the operator approves.
//!
//! Inputs, none of which this module writes (R-10 writer != subject):
//! - the nightly ledger (`apr-perf-ledger-v1`), for the timestamp and binary of each night;
//! - the per-night verdicts of the OBS-06 engine (`apr-perf-red-rules-v1`, one object per
//!   night), which decide what is RED. The RED rules are not re-implemented here;
//! - optionally the lane ledger (`apr-lane-row-v1`), used ONLY for the time a binary was
//!   first seen serving on a host. No ratio is ever computed from lane rows (§2.3).
//!
//! An hourly probe shortens one gap only: from the regression's onset to the first
//! measurement that sees it. So for each regression RED the report bounds that gap:
//! - `max_h` = first bad nightly row ts - the last row before it. The onset lies inside.
//! - `proven_h` = first bad nightly row ts - the earliest time the regression is known to
//!   exist. Only the lane ledger can place it earlier than the measurement: when the
//!   binary changed, the first lane row of the new binary (basis `binary_first_seen`, an
//!   inference that the binary is the cause). Otherwise the basis is `first_measured`
//!   and `proven_h` is 0.
//! - `rule_delay_h` = detection - first bad row. `rolling_2consecutive` fires one night
//!   after the first bad night; a probe running the same rule keeps that delay, so it is
//!   reported and never counted toward the 12 h question.
//!
//! Recommendation: `probe_indicated` when some `proven_h` > 12 h; `probe_not_indicated`
//! when every `max_h` <= 12 h (including no RED at all); `inconclusive` otherwise. Nothing
//! is decided before every declared series has 14 admissible nights (§4 rule 1), and an
//! empty ledger is refused, never read as "no RED" (R-2).

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

/// `schema` of a nightly ledger row (OBS-05).
pub const LEDGER_SCHEMA: &str = "apr-perf-ledger-v1";
/// `schema` of one night's OBS-06 verdict.
pub const RED_SCHEMA: &str = "apr-perf-red-rules-v1";
/// `schema` of a lane row (OBS-01).
pub const LANE_SCHEMA: &str = "apr-lane-row-v1";
/// `schema` of the report this module writes.
pub const REPORT_SCHEMA: &str = "apr-obs-probe-decision-v1";
/// Nights every declared series needs before a decision (spec §4 rule 1, §7 OBS-12).
pub const MIN_NIGHTS: usize = 14;
/// The undetected window OBS-12 asks about, in hours (spec §7 OBS-12).
pub const UNDETECTED_LIMIT_H: f64 = 12.0;

/// The identity block every APR-OBS-001 row carries (`apr-obs-row-identity-v1`, §2.1).
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

/// Backends whose rows must carry `gpu_proof` (§2.5).
pub const GPU_BACKENDS: [&str; 3] = ["cuda", "wgpu", "metal"];

/// OBS-06 rules that are liveness, not regression: they have no onset to bound.
const LIVENESS_RULES: [&str; 2] = ["liveness", "empty_ledger"];

type Series = (String, String);

/// Whether the report may decide at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Every declared series has >= 14 admissible nights and nothing was refused.
    Ready,
    /// Some declared series has fewer than 14 admissible nights.
    InsufficientNights,
    /// An input made the comparison impossible (see `refusals`).
    Refused,
}

/// The evidence-backed answer for the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Recommendation {
    /// Status is not `Ready`.
    NoDecision,
    /// Some RED is proven to have gone undetected for more than 12 h.
    ProbeIndicated,
    /// No RED could have gone undetected for more than 12 h.
    ProbeNotIndicated,
    /// Some RED may have gone undetected for more than 12 h, but none is proven to.
    Inconclusive,
}

/// What the proven onset of a regression rests on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OnsetBasis {
    /// Nothing places the onset before the first nightly row that measured it.
    FirstMeasured,
    /// The regressed binary was first seen serving at `onset_ts` (lane ledger).
    BinaryFirstSeen,
}

/// One regression RED with its undetected-window bounds.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RedEvent {
    /// Host of the series.
    pub host: String,
    /// Backend of the series.
    pub backend: String,
    /// Metric the engine flagged.
    pub metric: String,
    /// OBS-06 rule that fired.
    pub rule: String,
    /// Night the engine flagged it.
    pub night: String,
    /// Timestamp of the ledger row that was judged RED.
    pub detected_ts: String,
    /// Timestamp of the first nightly row that measured the regression.
    pub first_bad_ts: String,
    /// Earliest time the regression is known to exist.
    pub onset_ts: String,
    /// What `onset_ts` rests on.
    pub onset_basis: OnsetBasis,
    /// Hours from `onset_ts` to the first bad row: the gap is at least this long.
    pub proven_h: f64,
    /// Hours from the last row before the first bad row to it; `None` without one.
    pub max_h: Option<f64>,
    /// Hours from the first bad row to detection (the rule's own delay; not counted).
    pub rule_delay_h: f64,
    /// The binary changed between the last clear row and the first bad row.
    pub binary_changed: bool,
}

/// A liveness RED: a declared series with no admissible row that night.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LivenessEvent {
    /// Host of the series.
    pub host: String,
    /// Backend of the series.
    pub backend: String,
    /// Night the row was missing.
    pub night: String,
    /// Timestamp of the last admissible row before that night, if any.
    pub last_row_ts: Option<String>,
}

/// Admissible nights for one declared series.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SeriesNights {
    /// Host of the series.
    pub host: String,
    /// Backend of the series.
    pub backend: String,
    /// Distinct nights with an admissible ledger row.
    pub nights: usize,
}

/// Counts behind the recommendation.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Summary {
    /// Regression REDs considered.
    pub red_events: usize,
    /// Events with `proven_h` > 12 h.
    pub proven_over_limit: usize,
    /// Events whose `max_h` exceeds 12 h (or is unknown).
    pub possibly_over_limit: usize,
    /// Largest `proven_h`.
    pub max_proven_h: f64,
}

/// The OBS-12 report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProbeDecisionReport {
    /// Always [`REPORT_SCHEMA`].
    pub schema: &'static str,
    /// Whether a decision may be made.
    pub status: Status,
    /// The answer, or `no_decision`.
    pub recommendation: Recommendation,
    /// The limit the question uses.
    pub undetected_limit_h: f64,
    /// Nights needed per declared series.
    pub min_nights: usize,
    /// First and last night judged by OBS-06.
    pub window: Option<(String, String)>,
    /// Admissible nights per declared series.
    pub series: Vec<SeriesNights>,
    /// Regression REDs with window bounds.
    pub events: Vec<RedEvent>,
    /// Liveness REDs (listed; they have no onset to bound).
    pub liveness: Vec<LivenessEvent>,
    /// Counts behind the recommendation.
    pub summary: Summary,
    /// Input lines counted as absent, with reasons.
    pub inadmissible: Vec<String>,
    /// Reasons the report refused to decide (each is fatal).
    pub refusals: Vec<String>,
}

#[derive(Debug, Clone)]
struct NightRow {
    ts: DateTime<Utc>,
    binary: String,
}

#[derive(Debug, Clone)]
struct Red {
    series: Series,
    metric: String,
    rule: String,
    night: String,
}

fn str_field<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

fn hours(later: DateTime<Utc>, earlier: DateTime<Utc>) -> f64 {
    (later - earlier).num_seconds() as f64 / 3600.0
}

/// Why an identity-bearing row is not admissible, or `None` if it is.
#[must_use]
pub fn inadmissible_reason(v: &Value, schema: &str) -> Option<String> {
    if str_field(v, "schema") != Some(schema) {
        return Some(format!("schema is not {schema}"));
    }
    for k in IDENTITY {
        match v.get(k) {
            None | Some(Value::Null) => return Some(format!("identity field {k} missing")),
            Some(Value::String(s)) if s.is_empty() || s == "unknown" => {
                return Some(format!("identity field {k} is {s:?}"));
            }
            _ => {}
        }
    }
    let Some(proof) = v.get("gpu_proof") else {
        return Some("gpu_proof key absent".into());
    };
    let gpu = GPU_BACKENDS.contains(&str_field(v, "backend").unwrap_or_default());
    if gpu && proof.is_null() {
        return Some("backend_unproven: GPU row without gpu_proof".into());
    }
    if parse_ts(str_field(v, "ts").unwrap_or_default()).is_none() {
        return Some("ts is not RFC 3339".into());
    }
    None
}

fn parse_lines<'a>(
    text: &'a str,
    what: &'a str,
    inadmissible: &'a mut Vec<String>,
) -> impl Iterator<Item = (usize, Value)> + 'a {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .filter_map(move |(i, l)| match serde_json::from_str::<Value>(l) {
            Ok(v) => Some((i + 1, v)),
            Err(e) => {
                inadmissible.push(format!("{what}:{}: not JSON ({e})", i + 1));
                None
            }
        })
}

fn series_of(v: &Value) -> Series {
    (
        str_field(v, "host").unwrap_or_default().to_string(),
        str_field(v, "backend").unwrap_or_default().to_string(),
    )
}

/// Admissible ledger rows by series and night; the last row of a night wins, as in OBS-06.
fn index_ledger(
    text: &str,
    inadmissible: &mut Vec<String>,
) -> BTreeMap<Series, BTreeMap<String, NightRow>> {
    let mut out: BTreeMap<Series, BTreeMap<String, NightRow>> = BTreeMap::new();
    let rows: Vec<(usize, Value)> = parse_lines(text, "ledger", inadmissible).collect();
    for (line, v) in rows {
        if let Some(why) = inadmissible_reason(&v, LEDGER_SCHEMA) {
            inadmissible.push(format!("ledger:{line}: {why}"));
            continue;
        }
        let ts_s = str_field(&v, "ts").unwrap_or_default();
        let Some(ts) = parse_ts(ts_s) else { continue };
        let row = NightRow {
            ts,
            binary: str_field(&v, "binary_sha256")
                .unwrap_or_default()
                .to_string(),
        };
        let nights = out.entry(series_of(&v)).or_default();
        let night = ts_s.get(..10).unwrap_or_default().to_string();
        if nights.get(&night).is_none_or(|old| old.ts <= ts) {
            nights.insert(night, row);
        }
    }
    out
}

/// First time each (host, backend, binary) was seen in the lane ledger.
fn index_lane(
    text: &str,
    inadmissible: &mut Vec<String>,
) -> BTreeMap<(String, String, String), DateTime<Utc>> {
    let mut out = BTreeMap::new();
    let rows: Vec<(usize, Value)> = parse_lines(text, "lane", inadmissible).collect();
    for (line, v) in rows {
        if let Some(why) = inadmissible_reason(&v, LANE_SCHEMA) {
            inadmissible.push(format!("lane:{line}: {why}"));
            continue;
        }
        let Some(ts) = parse_ts(str_field(&v, "ts").unwrap_or_default()) else {
            continue;
        };
        let (host, backend) = series_of(&v);
        let binary = str_field(&v, "binary_sha256")
            .unwrap_or_default()
            .to_string();
        let first = out.entry((host, backend, binary)).or_insert(ts);
        if ts < *first {
            *first = ts;
        }
    }
    out
}

#[derive(Default)]
struct Verdicts {
    declared: BTreeSet<Series>,
    nights: BTreeSet<String>,
    reds: Vec<Red>,
}

fn read_verdict(v: &Value, out: &mut Verdicts, refusals: &mut Vec<String>) {
    let night = str_field(v, "night").unwrap_or_default().to_string();
    out.nights.insert(night.clone());
    for s in v
        .get("series")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        out.declared.insert(series_of(s));
    }
    for r in v.get("red").and_then(Value::as_array).into_iter().flatten() {
        let rule = str_field(r, "rule").unwrap_or_default().to_string();
        if rule == "empty_ledger" {
            refusals.push(format!("OBS-06 judged the ledger empty on {night} (R-2)"));
        }
        out.reds.push(Red {
            series: series_of(r),
            metric: str_field(r, "metric").unwrap_or_default().to_string(),
            rule,
            night: night.clone(),
        });
    }
}

fn index_verdicts(
    text: &str,
    inadmissible: &mut Vec<String>,
    refusals: &mut Vec<String>,
) -> Verdicts {
    let mut out = Verdicts::default();
    let rows: Vec<(usize, Value)> = parse_lines(text, "red", inadmissible).collect();
    for (line, v) in rows {
        if str_field(&v, "schema") != Some(RED_SCHEMA) {
            inadmissible.push(format!("red:{line}: schema is not {RED_SCHEMA}"));
            continue;
        }
        read_verdict(&v, &mut out, refusals);
    }
    out
}

/// The row before `night` in a series, if any.
fn row_before<'a>(
    rows: &'a BTreeMap<String, NightRow>,
    night: &str,
) -> Option<(&'a String, &'a NightRow)> {
    rows.range(..night.to_string()).next_back()
}

fn bound_event(
    red: &Red,
    rows: &BTreeMap<String, NightRow>,
    lane: &BTreeMap<(String, String, String), DateTime<Utc>>,
) -> Result<RedEvent, String> {
    let (host, backend) = red.series.clone();
    let detected = rows.get(&red.night).ok_or_else(|| {
        format!(
            "RED {host}/{backend} {} on {} has no admissible ledger row",
            red.metric, red.night
        )
    })?;
    // rolling_2consecutive fires on the second bad night: the regression was already
    // measured on the row before.
    let first_bad = if red.rule == "rolling_2consecutive" {
        row_before(rows, &red.night).map_or(detected, |(_, r)| r)
    } else {
        detected
    };
    let first_bad_night = rows
        .iter()
        .find(|(_, r)| r.ts == first_bad.ts)
        .map_or(red.night.clone(), |(n, _)| n.clone());
    let clear = row_before(rows, &first_bad_night).map(|(_, r)| r);
    let binary_changed = clear.is_some_and(|c| c.binary != first_bad.binary);
    let seen = lane
        .get(&(host.clone(), backend.clone(), first_bad.binary.clone()))
        .copied()
        .filter(|t| binary_changed && clear.is_some_and(|c| *t > c.ts) && *t < first_bad.ts);
    let (onset, basis) = seen.map_or((first_bad.ts, OnsetBasis::FirstMeasured), |t| {
        (t, OnsetBasis::BinaryFirstSeen)
    });
    Ok(RedEvent {
        host,
        backend,
        metric: red.metric.clone(),
        rule: red.rule.clone(),
        night: red.night.clone(),
        detected_ts: detected.ts.to_rfc3339(),
        first_bad_ts: first_bad.ts.to_rfc3339(),
        onset_ts: onset.to_rfc3339(),
        onset_basis: basis,
        proven_h: hours(first_bad.ts, onset),
        max_h: clear.map(|c| hours(first_bad.ts, c.ts)),
        rule_delay_h: hours(detected.ts, first_bad.ts),
        binary_changed,
    })
}

fn summarize(events: &[RedEvent]) -> Summary {
    Summary {
        red_events: events.len(),
        proven_over_limit: events
            .iter()
            .filter(|e| e.proven_h > UNDETECTED_LIMIT_H)
            .count(),
        possibly_over_limit: events
            .iter()
            .filter(|e| e.max_h.is_none_or(|h| h > UNDETECTED_LIMIT_H))
            .count(),
        max_proven_h: events.iter().map(|e| e.proven_h).fold(0.0, f64::max),
    }
}

fn recommend(status: Status, s: &Summary) -> Recommendation {
    if status != Status::Ready {
        Recommendation::NoDecision
    } else if s.proven_over_limit > 0 {
        Recommendation::ProbeIndicated
    } else if s.possibly_over_limit == 0 {
        Recommendation::ProbeNotIndicated
    } else {
        Recommendation::Inconclusive
    }
}

fn liveness_event(
    red: &Red,
    ledger: &BTreeMap<Series, BTreeMap<String, NightRow>>,
) -> LivenessEvent {
    let last = ledger
        .get(&red.series)
        .and_then(|rows| row_before(rows, &red.night))
        .map(|(_, r)| r.ts.to_rfc3339());
    LivenessEvent {
        host: red.series.0.clone(),
        backend: red.series.1.clone(),
        night: red.night.clone(),
        last_row_ts: last,
    }
}

/// Build the OBS-12 report from the three ledgers' text (JSONL; `lane` may be empty).
#[must_use]
pub fn build_report(ledger: &str, red_reports: &str, lane: &str) -> ProbeDecisionReport {
    let mut inadmissible = Vec::new();
    let mut refusals = Vec::new();
    let rows = index_ledger(ledger, &mut inadmissible);
    let lane = index_lane(lane, &mut inadmissible);
    let verdicts = index_verdicts(red_reports, &mut inadmissible, &mut refusals);
    if rows.is_empty() {
        refusals.push("the nightly ledger has 0 admissible rows (R-2)".into());
    }
    if verdicts.declared.is_empty() {
        refusals.push("no OBS-06 verdict declares any series".into());
    }
    let empty = BTreeMap::new();
    let mut events = Vec::new();
    let mut liveness = Vec::new();
    for red in &verdicts.reds {
        if LIVENESS_RULES.contains(&red.rule.as_str()) {
            if red.rule == "liveness" {
                liveness.push(liveness_event(red, &rows));
            }
            continue;
        }
        match bound_event(red, rows.get(&red.series).unwrap_or(&empty), &lane) {
            Ok(e) => events.push(e),
            Err(why) => refusals.push(why),
        }
    }
    let series: Vec<SeriesNights> = verdicts
        .declared
        .iter()
        .map(|(h, b)| SeriesNights {
            host: h.clone(),
            backend: b.clone(),
            nights: rows.get(&(h.clone(), b.clone())).map_or(0, BTreeMap::len),
        })
        .collect();
    let status = if !refusals.is_empty() {
        Status::Refused
    } else if series.iter().any(|s| s.nights < MIN_NIGHTS) {
        Status::InsufficientNights
    } else {
        Status::Ready
    };
    let summary = summarize(&events);
    ProbeDecisionReport {
        schema: REPORT_SCHEMA,
        status,
        recommendation: recommend(status, &summary),
        undetected_limit_h: UNDETECTED_LIMIT_H,
        min_nights: MIN_NIGHTS,
        window: verdicts
            .nights
            .first()
            .zip(verdicts.nights.last())
            .map(|(a, b)| (a.clone(), b.clone())),
        series,
        events,
        liveness,
        summary,
        inadmissible,
        refusals,
    }
}

impl ProbeDecisionReport {
    /// Render the report for the operator. Every value comes from the report itself.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut m = String::new();
        let _ = writeln!(m, "# OBS-12 hourly-probe decision (APR-OBS-001 §7)\n");
        let _ = writeln!(m, "- status: `{:?}`", self.status);
        let _ = writeln!(m, "- recommendation: **`{:?}`**", self.recommendation);
        if let Some((a, b)) = &self.window {
            let _ = writeln!(m, "- nights judged: {a} .. {b}");
        }
        let _ = writeln!(
            m,
            "- question: did any RED go undetected > {} h between nightlies? (needs >= {} nights per series)\n",
            self.undetected_limit_h, self.min_nights
        );
        let s = &self.summary;
        let _ = writeln!(
            m,
            "REDs: {} · proven > {} h: {} · possibly > {} h: {} · max proven {:.1} h\n",
            s.red_events,
            self.undetected_limit_h,
            s.proven_over_limit,
            self.undetected_limit_h,
            s.possibly_over_limit,
            s.max_proven_h
        );
        let _ = writeln!(m, "| host | backend | admissible nights |\n|---|---|---|");
        for x in &self.series {
            let _ = writeln!(m, "| {} | {} | {} |", x.host, x.backend, x.nights);
        }
        if !self.events.is_empty() {
            let _ = writeln!(
                m,
                "\n| night | series | metric | rule | onset basis | proven h | max h | rule delay h |\n|---|---|---|---|---|---|---|---|"
            );
            for e in &self.events {
                let max = e
                    .max_h
                    .map_or_else(|| "unknown".into(), |h| format!("{h:.1}"));
                let _ = writeln!(
                    m,
                    "| {} | {}/{} | {} | {} | {:?} | {:.1} | {} | {:.1} |",
                    e.night,
                    e.host,
                    e.backend,
                    e.metric,
                    e.rule,
                    e.onset_basis,
                    e.proven_h,
                    max,
                    e.rule_delay_h
                );
            }
        }
        for l in &self.liveness {
            let last = l.last_row_ts.as_deref().unwrap_or("none");
            let _ = writeln!(
                m,
                "\n- liveness RED {}/{} on {} (last row {last})",
                l.host, l.backend, l.night
            );
        }
        for r in &self.refusals {
            let _ = writeln!(m, "\n- REFUSED: {r}");
        }
        m
    }
}

#[cfg(test)]
#[path = "probe_decision_tests.rs"]
mod tests;
