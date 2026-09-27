//! OBS-08 one view (APR-OBS-001 §7): a page rendered from the ledgers and nothing else.
//!
//! Input is the raw JSONL text of three ledgers — nightly (`apr-perf-ledger-v1`), lane
//! (`apr-lane-row-v1`) and trace (`apr-trace-v1`) — plus two render parameters that are not
//! ledger data and are printed on the page as such: the declared series (from the forjar
//! declaration, never from the rows present) and the night the page is rendered for.
//!
//! The page shows the apr/llama ratio over time per host × backend, lane p50/p95 per token,
//! nightly status per declared series, and the latest trace link. Every value on it is read
//! from a row or computed from rows; there is no constant to hand-edit.
//!
//! Rules taken from the contracts, not re-decided here:
//! - a row with an incomplete identity block (a field absent, `null` or `"unknown"`, the
//!   `gpu_proof` key absent, the wrong `schema`, an unknown backend) or a line over
//!   4096 bytes is inadmissible and counts as absent (`identity_is_complete`,
//!   `append_only_atomic_lines`);
//! - a non-cpu row whose `gpu_proof` is `null` is `backend_unproven` and is kept out of
//!   every series (§2.5, `gpu_claim_is_proven`);
//! - an empty ledger, an empty declaration, and a declared series with no row for the
//!   night are RED (R-2, `every_host_backend_every_night`);
//! - ratios are only the ones each nightly row carries (apr and llama in one run); the
//!   view never divides across rows or hosts (S-4), and marks a point where the series'
//!   `crate_tarball_sha256` or `model_sha256` changed (`ratio_only_within_identity`).

use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// `schema` of a nightly row.
pub const PERF_SCHEMA: &str = "apr-perf-ledger-v1";
/// `schema` of a lane row.
pub const LANE_SCHEMA: &str = "apr-lane-row-v1";
/// `schema` of a trace row.
pub const TRACE_SCHEMA: &str = "apr-trace-v1";

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

/// Backends a row may name (§2.1).
pub const BACKENDS: [&str; 4] = ["cpu", "cuda", "wgpu", "metal"];

/// Longest admissible ledger line in bytes (`append_only_atomic_lines`).
pub const MAX_LINE_BYTES: usize = 4096;

/// The four ratio fields of a nightly row, in page order.
pub const RATIO_FIELDS: [&str; 4] = ["load", "ttft", "prefill", "decode"];

/// One host × backend series.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Series {
    /// forjar host name.
    pub host: String,
    /// cpu / cuda / wgpu / metal.
    pub backend: String,
}

impl Series {
    /// Build a series key.
    #[must_use]
    pub fn new(host: &str, backend: &str) -> Self {
        Self {
            host: host.to_string(),
            backend: backend.to_string(),
        }
    }
}

/// Raw JSONL text of the three ledgers the view reads.
#[derive(Debug, Clone, Default)]
pub struct Ledgers {
    /// Nightly ledger (`apr-perf-ledger-v1`).
    pub perf: String,
    /// Lane ledger (`apr-lane-row-v1`).
    pub lane: String,
    /// Trace ledger (`apr-trace-v1`).
    pub trace: String,
}

impl Ledgers {
    /// Read the three ledgers. A missing file reads as an empty ledger, which the view
    /// renders RED; any other I/O error is returned.
    ///
    /// # Errors
    /// Returns the I/O error of a file that exists but cannot be read.
    pub fn read(
        perf: &std::path::Path,
        lane: &std::path::Path,
        trace: &std::path::Path,
    ) -> std::io::Result<Self> {
        Ok(Self {
            perf: read_or_empty(perf)?,
            lane: read_or_empty(lane)?,
            trace: read_or_empty(trace)?,
        })
    }
}

fn read_or_empty(p: &std::path::Path) -> std::io::Result<String> {
    match std::fs::read_to_string(p) {
        Ok(s) => Ok(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(e),
    }
}

/// What admission made of one line.
#[derive(Debug, Clone, PartialEq)]
pub enum Admission {
    /// Complete identity; usable.
    Admitted(Map<String, Value>),
    /// A non-cpu row without `gpu_proof`: kept out of every series.
    BackendUnproven(Series),
    /// Counts as absent, with the reason.
    Inadmissible(String),
}

/// Admit one ledger line against `schema`.
#[must_use]
pub fn admit(line: &str, schema: &str) -> Admission {
    if line.len() > MAX_LINE_BYTES {
        return Admission::Inadmissible(format!("line exceeds {MAX_LINE_BYTES} bytes"));
    }
    let Ok(Value::Object(row)) = serde_json::from_str::<Value>(line) else {
        return Admission::Inadmissible("not a JSON object".to_string());
    };
    if let Some(f) = IDENTITY.iter().find(|f| !present(&row, f)) {
        return Admission::Inadmissible(format!("identity field `{f}` absent"));
    }
    if row.get("schema").and_then(Value::as_str) != Some(schema) {
        return Admission::Inadmissible(format!("schema is not `{schema}`"));
    }
    if !row.contains_key("gpu_proof") {
        return Admission::Inadmissible("`gpu_proof` key absent".to_string());
    }
    let backend = text(&row, "backend");
    if !BACKENDS.contains(&backend.as_str()) {
        return Admission::Inadmissible(format!("unknown backend `{backend}`"));
    }
    if night(&row).is_none() {
        return Admission::Inadmissible("`ts` is not RFC 3339".to_string());
    }
    if backend != "cpu" && row.get("gpu_proof").is_some_and(Value::is_null) {
        return Admission::BackendUnproven(series_of(&row));
    }
    Admission::Admitted(row)
}

fn present(row: &Map<String, Value>, f: &str) -> bool {
    match row.get(f) {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) => !s.is_empty() && s != "unknown",
        Some(_) => true,
    }
}

fn text(row: &Map<String, Value>, f: &str) -> String {
    match row.get(f) {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

fn series_of(row: &Map<String, Value>) -> Series {
    Series::new(&text(row, "host"), &text(row, "backend"))
}

/// The `YYYY-MM-DD` night of a row's `ts`, when `ts` is RFC 3339.
fn night(row: &Map<String, Value>) -> Option<String> {
    let ts = row.get("ts")?.as_str()?;
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|t| t.with_timezone(&chrono::Utc).format("%Y-%m-%d").to_string())
}

/// Admission counts for one ledger.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LedgerCount {
    /// Non-blank lines read.
    pub read: usize,
    /// Admitted rows.
    pub admitted: usize,
    /// `backend_unproven` rows.
    pub unproven: usize,
    /// Inadmissible rows, by reason.
    pub inadmissible: BTreeMap<String, usize>,
}

fn admit_all(text: &str, schema: &str) -> (Vec<Map<String, Value>>, Vec<Series>, LedgerCount) {
    let mut rows = Vec::new();
    let mut unproven = Vec::new();
    let mut count = LedgerCount::default();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        count.read += 1;
        match admit(line, schema) {
            Admission::Admitted(r) => rows.push(r),
            Admission::BackendUnproven(s) => unproven.push(s),
            Admission::Inadmissible(why) => *count.inadmissible.entry(why).or_default() += 1,
        }
    }
    count.admitted = rows.len();
    count.unproven = unproven.len();
    rows.sort_by_cached_key(text_ts);
    (rows, unproven, count)
}

fn text_ts(row: &Map<String, Value>) -> String {
    row.get("ts")
        .and_then(Value::as_str)
        .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
        .map(|t| t.with_timezone(&chrono::Utc).to_rfc3339())
        .unwrap_or_default()
}

/// One point of a ratio series.
#[derive(Debug, Clone, PartialEq)]
pub struct RatioPoint {
    /// Row `ts`.
    pub ts: String,
    /// Row `apr_version`.
    pub apr_version: String,
    /// `ratio.{load,ttft,prefill,decode}`; `None` when the row does not carry it.
    pub ratio: [Option<f64>; 4],
    /// The series' `crate_tarball_sha256` or `model_sha256` differs from the previous point.
    pub identity_changed: bool,
}

/// Lane per-token latency for one series.
#[derive(Debug, Clone, PartialEq)]
pub struct LanePercentiles {
    /// Rows with `ok = true` and a numeric per-token value.
    pub measured: usize,
    /// Rows with `ok = false`.
    pub failed: usize,
    /// p50 / p95 of `prefill_ms_per_tok`.
    pub prefill: Option<(f64, f64)>,
    /// p50 / p95 of `decode_ms_per_tok`.
    pub decode: Option<(f64, f64)>,
}

/// Nightly status of one declared series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NightStatus {
    /// An admitted row exists for the night.
    Green,
    /// No admitted row for the night; last admitted night if any, and unproven rows seen.
    Missing {
        /// Last night with an admitted row.
        last_seen: Option<String>,
        /// `backend_unproven` rows for this series in the ledger.
        unproven: usize,
    },
}

/// Latest trace of one series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceLink {
    /// Row `ts`.
    pub ts: String,
    /// `level`.
    pub level: String,
    /// `provenance`.
    pub provenance: String,
    /// `payload.path`.
    pub path: String,
    /// `payload.sha256`.
    pub sha256: String,
}

/// The rendered view: structured values plus the RED reasons.
#[derive(Debug, Clone, PartialEq)]
pub struct View {
    /// The night the page is for (render parameter).
    pub night: String,
    /// Declared series (render parameter).
    pub declared: Vec<Series>,
    /// Admission counts: perf, lane, trace.
    pub counts: [LedgerCount; 3],
    /// Ratio points per series, oldest first.
    pub ratios: BTreeMap<Series, Vec<RatioPoint>>,
    /// Lane percentiles per series.
    pub lane: BTreeMap<Series, LanePercentiles>,
    /// Nightly status per declared series.
    pub status: BTreeMap<Series, NightStatus>,
    /// Series with nightly rows that the declaration does not name.
    pub undeclared: Vec<Series>,
    /// Latest trace per series.
    pub traces: BTreeMap<Series, TraceLink>,
    /// Why the page is RED; empty means GREEN.
    pub red: Vec<String>,
}

impl View {
    /// GREEN only when nothing is RED.
    #[must_use]
    pub fn is_green(&self) -> bool {
        self.red.is_empty()
    }
}

/// Build the view from the ledgers, the declared series and the night.
#[must_use]
pub fn build(ledgers: &Ledgers, declared: &[Series], night_of: &str) -> View {
    let (perf, perf_unproven, perf_n) = admit_all(&ledgers.perf, PERF_SCHEMA);
    let (lane, _, lane_n) = admit_all(&ledgers.lane, LANE_SCHEMA);
    let (trace, _, trace_n) = admit_all(&ledgers.trace, TRACE_SCHEMA);
    let mut declared = declared.to_vec();
    declared.sort();
    declared.dedup();

    let mut red = Vec::new();
    for (name, n) in [("nightly", &perf_n), ("lane", &lane_n), ("trace", &trace_n)] {
        if n.admitted == 0 {
            red.push(format!("{name} ledger has 0 admissible rows"));
        }
    }
    if declared.is_empty() {
        red.push("no declared series".to_string());
    }

    let status = night_status(&perf, &perf_unproven, &declared, night_of);
    for (s, st) in &status {
        if st != &NightStatus::Green {
            red.push(format!(
                "{}/{}: no admissible nightly row for {night_of}",
                s.host, s.backend
            ));
        }
    }
    let ratios = ratio_series(&perf);
    let undeclared = ratios
        .keys()
        .filter(|s| !declared.contains(s))
        .cloned()
        .collect();

    View {
        night: night_of.to_string(),
        declared,
        counts: [perf_n, lane_n, trace_n],
        ratios,
        lane: lane_percentiles(&lane),
        status,
        undeclared,
        traces: latest_traces(&trace),
        red,
    }
}

fn night_status(
    perf: &[Map<String, Value>],
    unproven: &[Series],
    declared: &[Series],
    night_of: &str,
) -> BTreeMap<Series, NightStatus> {
    let mut out = BTreeMap::new();
    for s in declared {
        let nights: Vec<String> = perf
            .iter()
            .filter(|r| &series_of(r) == s)
            .filter_map(night)
            .collect();
        let st = if nights.iter().any(|n| n == night_of) {
            NightStatus::Green
        } else {
            NightStatus::Missing {
                last_seen: nights.into_iter().max(),
                unproven: unproven.iter().filter(|u| *u == s).count(),
            }
        };
        out.insert(s.clone(), st);
    }
    out
}

fn ratio_series(perf: &[Map<String, Value>]) -> BTreeMap<Series, Vec<RatioPoint>> {
    let mut out: BTreeMap<Series, Vec<RatioPoint>> = BTreeMap::new();
    let mut last_identity: BTreeMap<Series, (String, String)> = BTreeMap::new();
    for row in perf {
        let s = series_of(row);
        let id = (text(row, "crate_tarball_sha256"), text(row, "model_sha256"));
        let changed = last_identity.get(&s).is_some_and(|prev| prev != &id);
        last_identity.insert(s.clone(), id);
        let ratio = RATIO_FIELDS.map(|k| {
            row.get("ratio")
                .and_then(|r| r.get(k))
                .and_then(Value::as_f64)
        });
        out.entry(s).or_default().push(RatioPoint {
            ts: text(row, "ts"),
            apr_version: text(row, "apr_version"),
            ratio,
            identity_changed: changed,
        });
    }
    out
}

/// Nearest-rank percentile of an ascending slice; `None` when empty.
#[must_use]
pub fn percentile(sorted: &[f64], p: u32) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (u64::from(p) * sorted.len() as u64).div_ceil(100).max(1);
    sorted.get(rank as usize - 1).copied()
}

fn p50_p95(mut v: Vec<f64>) -> Option<(f64, f64)> {
    v.sort_by(f64::total_cmp);
    Some((percentile(&v, 50)?, percentile(&v, 95)?))
}

fn lane_percentiles(lane: &[Map<String, Value>]) -> BTreeMap<Series, LanePercentiles> {
    let mut groups: BTreeMap<Series, (Vec<f64>, Vec<f64>, usize)> = BTreeMap::new();
    for row in lane {
        let g = groups.entry(series_of(row)).or_default();
        if row.get("ok").and_then(Value::as_bool) != Some(true) {
            g.2 += 1;
            continue;
        }
        if let Some(x) = row.get("prefill_ms_per_tok").and_then(Value::as_f64) {
            g.0.push(x);
        }
        if let Some(x) = row.get("decode_ms_per_tok").and_then(Value::as_f64) {
            g.1.push(x);
        }
    }
    groups
        .into_iter()
        .map(|(s, (pre, dec, failed))| {
            let lp = LanePercentiles {
                measured: pre.len().max(dec.len()),
                failed,
                prefill: p50_p95(pre),
                decode: p50_p95(dec),
            };
            (s, lp)
        })
        .collect()
}

fn latest_traces(trace: &[Map<String, Value>]) -> BTreeMap<Series, TraceLink> {
    let mut out = BTreeMap::new();
    for row in trace {
        let payload = row.get("payload").and_then(Value::as_object);
        let field = |k: &str| payload.and_then(|p| p.get(k)).and_then(Value::as_str);
        let (Some(path), Some(sha256)) = (field("path"), field("sha256")) else {
            continue;
        };
        // rows are ts-ascending, so the last insert per series is the latest
        out.insert(
            series_of(row),
            TraceLink {
                ts: text(row, "ts"),
                level: text(row, "level"),
                provenance: text(row, "provenance"),
                path: path.to_string(),
                sha256: sha256.to_string(),
            },
        );
    }
    out
}

fn cell(x: Option<f64>) -> String {
    x.map_or_else(|| "absent".to_string(), |v| format!("{v:.3}"))
}

fn pair(x: Option<(f64, f64)>) -> String {
    x.map_or_else(
        || "no rows".to_string(),
        |(a, b)| format!("{a:.3} / {b:.3}"),
    )
}

/// Render the view as Markdown.
#[must_use]
pub fn to_markdown(v: &View) -> String {
    let mut o = String::new();
    let verdict = if v.is_green() { "GREEN" } else { "RED" };
    let _ = writeln!(
        o,
        "# apr observability — night {}\n\n**{verdict}**\n",
        v.night
    );
    for r in &v.red {
        let _ = writeln!(o, "- RED: {r}");
    }
    o.push_str("\n## Ledgers\n\n| ledger | read | admitted | backend_unproven | inadmissible |\n|---|---|---|---|---|\n");
    for (name, c) in ["nightly", "lane", "trace"].iter().zip(&v.counts) {
        let bad: Vec<String> = c
            .inadmissible
            .iter()
            .map(|(k, n)| format!("{n} × {k}"))
            .collect();
        let _ = writeln!(
            o,
            "| {name} | {} | {} | {} | {} |",
            c.read,
            c.admitted,
            c.unproven,
            bad.join("; ")
        );
    }
    o.push_str("\n## Nightly status\n\n| host | backend | status |\n|---|---|---|\n");
    for (s, st) in &v.status {
        let t = match st {
            NightStatus::Green => "GREEN".to_string(),
            NightStatus::Missing {
                last_seen,
                unproven,
            } => format!(
                "RED — missing (last seen {}; {unproven} backend_unproven)",
                last_seen.as_deref().unwrap_or("never")
            ),
        };
        let _ = writeln!(o, "| {} | {} | {t} |", s.host, s.backend);
    }
    for s in &v.undeclared {
        let _ = writeln!(o, "| {} | {} | undeclared series |", s.host, s.backend);
    }
    o.push_str("\n## apr / llama ratio over time\n");
    for (s, pts) in &v.ratios {
        let _ = writeln!(o, "\n### {} / {}\n\n| ts | apr_version | load | ttft | prefill | decode | identity |\n|---|---|---|---|---|---|---|", s.host, s.backend);
        for p in pts {
            let id = if p.identity_changed { "changed" } else { "" };
            let r = p.ratio.map(cell);
            let _ = writeln!(
                o,
                "| {} | {} | {} | {} | {} | {} | {id} |",
                p.ts, p.apr_version, r[0], r[1], r[2], r[3]
            );
        }
    }
    o.push_str("\n## Lane ms per token (p50 / p95)\n\n| host | backend | measured | failed | prefill | decode |\n|---|---|---|---|---|---|\n");
    for (s, l) in &v.lane {
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} |",
            s.host,
            s.backend,
            l.measured,
            l.failed,
            pair(l.prefill),
            pair(l.decode)
        );
    }
    o.push_str("\n## Latest trace\n\n| host | backend | ts | level | provenance | payload | sha256 |\n|---|---|---|---|---|---|---|\n");
    for (s, t) in &v.traces {
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | [{}]({}) | `{}` |",
            s.host, s.backend, t.ts, t.level, t.provenance, t.path, t.path, t.sha256
        );
    }
    o
}

#[cfg(test)]
#[path = "obs_view_tests.rs"]
mod tests;
