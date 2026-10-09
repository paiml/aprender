//! The TTFT ratio verdict (0.71.1 train, row V1).
//!
//! TTFT ratio = reference TTFT / `apr` TTFT, read from `bands[].ratios.ttft` of
//! each replicate receipt a host's run wrote. The row is green when, on every
//! GPU host the matrix names, the lower bound of the 95% interval is at least
//! the floor in `scripts/perf-matrix.yaml` (`arms.L3.v1.ttft_floor`).
//!
//! Three outcomes, never two (`contracts/apr-ttft-ratio-verdict-v1.yaml`):
//!
//! - **GREEN** (exit 0): every host named, every receipt readable, every
//!   host's minimum lcb95 at or above the floor.
//! - **RED** (exit 1): every receipt readable and some host's minimum below it.
//! - **NOT MEASURED** (exit 2): anything missing. A missing host, too few
//!   replicates, a band that was not measured on both sides, no ratio, no
//!   interval, or a provenance field the row requires. It outranks RED: a RED
//!   computed from part of the evidence is a guess.
//!
//! A host's figure is the minimum lcb95 over its replicate receipts, never the
//! mean and never the point estimate. A band counts when its comparator status
//! is MEASURED and its own status is MEASURED or NONCONFORMANT-VALID; the
//! latter is printed with its stated reasons beside its number, because the row
//! does not name that condition and a reader must be able to see it.
//!
//! The floor, the host list and the replicate minimum come from the matrix; no
//! threshold is written here.
//!
//! Beside each receipt's line, the gap split by phase (#4954): the p50 of each
//! side's TTFT, of its server-reported prefill, and of what lies outside
//! prefill; for `apr`, that last part split again into load, first token and
//! transfer. It names the phase that holds the gap and never moves the outcome.

use std::path::Path;

use serde::Deserialize;

use super::drain::{percentile, BandStatus, Outcome, SampleRow};
use super::protocol::PERF_MATRIX_SOURCE;
use super::receipt::{Receipt, ReceiptBand};

/// The three outcomes and their exit codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtftOutcome {
    /// Every host at or above the floor, on complete evidence.
    Green,
    /// Some host below the floor, on complete evidence.
    Red,
    /// The evidence is incomplete; no verdict on the floor is given.
    NotMeasured,
}

impl TtftOutcome {
    /// 0 GREEN, 1 RED, 2 NOT MEASURED.
    #[must_use]
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Green => 0,
            Self::Red => 1,
            Self::NotMeasured => 2,
        }
    }

    /// The word printed on the verdict line.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Green => "GREEN",
            Self::Red => "RED",
            Self::NotMeasured => "NOT MEASURED",
        }
    }
}

/// What the matrix says the row is: the floor, the hosts, the replicate minimum.
#[derive(Debug, Clone, PartialEq)]
pub struct TtftPolicy {
    /// `arms.L3.v1.ttft_floor`: the least acceptable lcb95.
    pub floor: f64,
    /// `arms.L3.v1.hosts`: every host that must be measured.
    pub hosts: Vec<String>,
    /// `protocol.replicates_min`: receipts each host needs.
    pub replicates_min: usize,
}

/// The matrix spelling of `arms.L3.v1`. Like the other `Matrix*` readers it
/// accepts governance keys it does not use (`threshold_class`, `author`).
#[derive(Debug, Deserialize)]
struct MatrixTtftBlock {
    ttft_floor: f64,
    hosts: Vec<String>,
}

impl TtftPolicy {
    /// Read the policy from a matrix document, naming what is missing.
    ///
    /// # Errors
    /// The matrix does not parse, has no `arms.L3.v1` or `protocol.replicates_min`,
    /// or holds a floor that is not a positive finite number or an empty host list.
    pub fn from_matrix(source: &str) -> Result<Self, String> {
        let doc: serde_yaml_ng::Value = serde_yaml_ng::from_str(source)
            .map_err(|e| format!("perf-matrix.yaml does not parse as YAML: {e}"))?;
        let block = doc
            .get("arms")
            .and_then(|a| a.get("L3"))
            .and_then(|l| l.get("v1"))
            .ok_or("perf-matrix.yaml has no `arms.L3.v1` block (ttft_floor, hosts)")?;
        let MatrixTtftBlock { ttft_floor, hosts } = serde_yaml_ng::from_value(block.clone())
            .map_err(|e| format!("perf-matrix.yaml `arms.L3.v1`: {e}"))?;
        let replicates_min = doc
            .get("protocol")
            .and_then(|p| p.get("replicates_min"))
            .and_then(serde_yaml_ng::Value::as_u64)
            .ok_or("perf-matrix.yaml has no `protocol.replicates_min`")?;
        if !(ttft_floor.is_finite() && ttft_floor > 0.0) {
            return Err(format!(
                "arms.L3.v1.ttft_floor {ttft_floor} is not a positive number"
            ));
        }
        if hosts.is_empty() || replicates_min == 0 {
            return Err("arms.L3.v1.hosts is empty or protocol.replicates_min is 0".to_string());
        }
        Ok(Self {
            floor: ttft_floor,
            hosts,
            replicates_min: usize::try_from(replicates_min).map_err(|e| e.to_string())?,
        })
    }

    /// The policy of the matrix this crate was built from.
    ///
    /// # Errors
    /// As [`Self::from_matrix`].
    pub fn compiled() -> Result<Self, String> {
        Self::from_matrix(PERF_MATRIX_SOURCE)
    }
}

/// The provenance sidecar `scripts/v1_ttft_baseline.sh` writes into a run directory.
pub const PROVENANCE_FILE: &str = "v1-provenance.json";

/// One host's run: its provenance sidecar (`v1-provenance.json`) and its
/// replicate receipts, as text. Only [`TtftRun::from_dir`] touches the
/// filesystem; the verdict itself is a pure function of these strings.
#[derive(Debug, Clone)]
pub struct TtftRun {
    /// Where the run came from, for the output (a directory path).
    pub label: String,
    /// The sidecar's text.
    pub provenance: String,
    /// `(file name, text)` for each replicate receipt.
    pub receipts: Vec<(String, String)>,
}

/// The replicate number of a `receipt.r<N>.json` file name, and nothing else.
fn replicate_of(name: &str) -> Option<u32> {
    name.strip_prefix("receipt.r")?
        .strip_suffix(".json")?
        .parse()
        .ok()
}

impl TtftRun {
    /// Read a run directory: [`PROVENANCE_FILE`] and every `receipt.r<N>.json`,
    /// in replicate order. Other files (signatures, logs) are not receipts.
    ///
    /// # Errors
    /// The directory, its sidecar, or one of its receipts cannot be read.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let label = dir.display().to_string();
        let read = |path: &Path| {
            std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
        };
        let provenance = read(&dir.join(PROVENANCE_FILE))?;
        let mut numbered = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(|e| format!("{label}: {e}"))? {
            let entry = entry.map_err(|e| format!("{label}: {e}"))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(n) = replicate_of(&name) {
                numbered.push((n, name, entry.path()));
            }
        }
        numbered.sort_by_key(|(n, _, _)| *n);
        let mut receipts = Vec::with_capacity(numbered.len());
        for (_, name, path) in numbered {
            receipts.push((name, read(&path)?));
        }
        Ok(Self {
            label,
            provenance,
            receipts,
        })
    }
}

/// The sidecar fields the row requires. Every one is optional here so that a
/// missing field is named rather than failing the whole parse.
#[derive(Debug, Default, Deserialize)]
struct Sidecar {
    host: Option<String>,
    gguf: Option<SidecarGguf>,
    comparator: Option<SidecarComparator>,
    band: Option<SidecarBand>,
    interval: Option<SidecarInterval>,
}

#[derive(Debug, Deserialize)]
struct SidecarGguf {
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SidecarComparator {
    pin: Option<String>,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SidecarBand {
    concurrency: Option<u32>,
    replicates: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SidecarInterval {
    method: Option<String>,
}

/// The sidecar after its required fields were found.
#[derive(Debug)]
struct RunIdentity {
    host: String,
    gguf_sha256: String,
    comparator_pin: String,
    comparator_sha256: String,
    concurrency: u32,
    replicates: usize,
    method: String,
}

fn non_empty(field: Option<String>, name: &str) -> Result<String, String> {
    field
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("provenance names no {name}"))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

impl RunIdentity {
    fn read(text: &str) -> Result<Self, String> {
        let s: Sidecar =
            serde_json::from_str(text).map_err(|e| format!("provenance does not parse: {e}"))?;
        let host = non_empty(s.host, "host")?;
        let gguf_sha256 = non_empty(s.gguf.and_then(|g| g.sha256), "GGUF sha256")?;
        if !is_sha256(&gguf_sha256) {
            return Err(format!(
                "provenance GGUF sha256 `{gguf_sha256}` is not a sha256"
            ));
        }
        let (pin, sha) = s.comparator.map_or((None, None), |c| (c.pin, c.sha256));
        let comparator_pin = non_empty(pin, "comparator pin")?;
        let comparator_sha256 = non_empty(sha, "comparator sha256")?;
        if !is_sha256(&comparator_sha256) {
            return Err(format!(
                "provenance comparator sha256 `{comparator_sha256}` is not a sha256"
            ));
        }
        let band = s.band.ok_or("provenance names no band")?;
        let concurrency = band
            .concurrency
            .ok_or("provenance names no band concurrency")?;
        let replicates = band
            .replicates
            .filter(|n| *n > 0)
            .ok_or("provenance names no replicate count (n)")?;
        let method = non_empty(s.interval.and_then(|i| i.method), "interval method")?;
        Ok(Self {
            host,
            gguf_sha256,
            comparator_pin,
            comparator_sha256,
            concurrency,
            replicates,
            method,
        })
    }
}

/// One replicate's reading: its lcb95 and the line printed for it.
#[derive(Debug)]
struct Reading {
    replicate: u32,
    lcb95: f64,
    line: String,
}

fn countable(band: &ReceiptBand) -> Result<(), String> {
    let measured = BandStatus::Measured.wire_token();
    let comparator = band.comparator_status.as_deref().unwrap_or("absent");
    if comparator != measured {
        return Err(format!(
            "comparator_status {comparator} (only {measured} counts)"
        ));
    }
    let ok = [measured, BandStatus::NonconformantValid.wire_token()];
    if !ok.contains(&band.status.as_str()) {
        return Err(format!(
            "band status {} (only {} count)",
            band.status,
            ok.join(" or ")
        ));
    }
    Ok(())
}

/// The receipt's own reasons for this band's status: the `unproduced_fields`
/// entries that name the band's concurrency.
fn band_reasons(receipt: &Receipt, concurrency: u32) -> Vec<String> {
    let tag = format!(" c={concurrency}:");
    receipt
        .unproduced_fields
        .iter()
        .filter(|f| f.contains(&tag))
        .map(|f| f.chars().take(120).collect())
        .collect()
}

fn the_band<'a>(receipt: &'a Receipt, id: &RunIdentity) -> Result<&'a ReceiptBand, String> {
    let mut bands = receipt
        .bands
        .iter()
        .filter(|b| b.concurrency == id.concurrency);
    match (bands.next(), bands.next()) {
        (Some(band), None) => Ok(band),
        (None, _) => Err(format!("no band at c={}", id.concurrency)),
        (Some(_), Some(_)) => Err(format!("more than one band at c={}", id.concurrency)),
    }
}

fn check_identity(receipt: &Receipt, id: &RunIdentity) -> Result<(), String> {
    if receipt.provenance.host != id.host {
        return Err(format!(
            "receipt host {} differs from provenance host {}",
            receipt.provenance.host, id.host
        ));
    }
    // The ratio's denominator: a receipt measured against another llama.cpp
    // build is not this run's evidence.
    let Some(c) = &receipt.provenance.comparator else {
        return Err("receipt names no comparator".to_string());
    };
    if c.commit != id.comparator_pin {
        return Err(format!(
            "receipt comparator commit {} differs from provenance comparator pin {}",
            c.commit, id.comparator_pin
        ));
    }
    if c.sha256 != id.comparator_sha256 {
        return Err(format!(
            "receipt comparator sha256 {} differs from provenance comparator sha256",
            c.sha256
        ));
    }
    // The baseline script leaves `model_file` null and binds the GGUF in the
    // sidecar, so the model digest is cross-checked only where a receipt has one.
    match &receipt.provenance.model_file {
        Some(m) if m.sha256 != id.gguf_sha256 => Err(format!(
            "receipt model sha256 {} differs from provenance GGUF sha256",
            m.sha256
        )),
        _ => Ok(()),
    }
}

fn read_receipt(name: &str, text: &str, id: &RunIdentity) -> Result<Reading, String> {
    let fail = |why: String| format!("{}/{name}: {why}", id.host);
    let receipt = Receipt::parse(text).map_err(|e| fail(format!("does not parse: {e}")))?;
    receipt
        .validate()
        .map_err(|e| fail(format!("is not a valid receipt: {e}")))?;
    check_identity(&receipt, id).map_err(fail)?;
    let band = the_band(&receipt, id).map_err(fail)?;
    countable(band).map_err(fail)?;
    let ttft = band
        .ratios
        .as_ref()
        .and_then(|r| r.ttft.as_ref())
        .ok_or_else(|| fail("no ratios.ttft".to_string()))?;
    let method = ttft.method.wire_token();
    if method != id.method {
        return Err(fail(format!(
            "ratio method {method} differs from provenance {}",
            id.method
        )));
    }
    if ttft.n == 0 {
        return Err(fail("ratios.ttft names n = 0".to_string()));
    }
    let lcb95 = ttft
        .lcb95
        .filter(|l| l.is_finite())
        .ok_or_else(|| fail("ratios.ttft has no lcb95".to_string()))?;
    let mut line = format!(
        "{}/{name} r{} c={} {} ttft lcb95 {lcb95:.4} (point {:.4}, n {}, {method})",
        id.host, band.replicate, band.concurrency, band.status, ttft.point, ttft.n
    );
    for reason in band_reasons(&receipt, band.concurrency) {
        line.push_str("\n    reason: ");
        line.push_str(&reason);
    }
    if let Some(phases) = phase_line(band) {
        line.push_str("\n    ");
        line.push_str(&phases);
    }
    Ok(Reading {
        replicate: band.replicate,
        lcb95,
        line,
    })
}

/// One side's TTFT by phase (#4954): the p50, in milliseconds, of each part
/// over the side's completed rows. A part a row does not report is left out
/// of that part's p50, and a part no row reports is `None`, never 0.
#[derive(Debug, Default, PartialEq)]
struct PhaseMedians {
    ttft: Option<f64>,
    prefill: Option<f64>,
    outside_prefill: Option<f64>,
    load: Option<f64>,
    first_token: Option<f64>,
    transfer: Option<f64>,
}

fn p50(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    v.sort_by(f64::total_cmp);
    percentile(&v, 0.5)
}

impl PhaseMedians {
    fn of(rows: &[SampleRow]) -> Self {
        let done: Vec<&SampleRow> = rows
            .iter()
            .filter(|r| r.outcome == Outcome::Completed)
            .collect();
        let part = |f: fn(&SampleRow) -> Option<f64>| p50(done.iter().filter_map(|r| f(r)));
        Self {
            ttft: part(|r| r.ttft_ms),
            prefill: part(|r| r.prefill_ms),
            outside_prefill: part(|r| Some(r.ttft_ms? - r.prefill_ms?)),
            load: part(|r| r.load_ms),
            first_token: part(|r| r.first_token_ms),
            // What the client waited beyond the server's three phases: the
            // request and the first chunk in transit, and parsing them.
            transfer: part(|r| Some(r.ttft_ms? - r.load_ms? - r.prefill_ms? - r.first_token_ms?)),
        }
    }
}

fn ms(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_string(), |v| format!("{v:.1}"))
}

/// `apr` against the reference, and the gap where both sides have the part.
fn versus(apr: Option<f64>, reference: Option<f64>) -> String {
    let gap = match (apr, reference) {
        (Some(a), Some(r)) => format!("{:+.1}", a - r),
        _ => "-".to_string(),
    };
    format!("{} vs {} (gap {gap})", ms(apr), ms(reference))
}

/// The band's TTFT gap split by phase, `apr` against the reference band it
/// was measured with. `None` when the band carries no reference band.
fn phase_line(band: &ReceiptBand) -> Option<String> {
    let apr = PhaseMedians::of(&band.samples);
    let reference = PhaseMedians::of(&band.baseline.as_ref()?.samples);
    Some(format!(
        "phases p50 ms, apr vs reference: ttft {}; prefill {}; outside prefill {}; \
         apr outside prefill: load {}, first token {}, transfer {}",
        versus(apr.ttft, reference.ttft),
        versus(apr.prefill, reference.prefill),
        versus(apr.outside_prefill, reference.outside_prefill),
        ms(apr.load),
        ms(apr.first_token),
        ms(apr.transfer),
    ))
}

/// One host's result: its minimum lcb95, or why it has none.
#[derive(Debug)]
struct HostResult {
    min_lcb95: Result<f64, Vec<String>>,
    lines: Vec<String>,
}

/// A host needs at least `protocol.replicates_min` readable receipts, and
/// exactly as many as its provenance says it ran: a receipt more than the run
/// wrote came from somewhere else.
fn count_gaps(host: &str, readable: usize, stated: usize, replicates_min: usize) -> Vec<String> {
    let mut gaps = Vec::new();
    if readable < replicates_min {
        gaps.push(format!(
            "{host}: {readable} readable receipts, protocol.replicates_min is {replicates_min}"
        ));
    }
    if readable != stated {
        gaps.push(format!(
            "{host}: {readable} readable receipts, the provenance states {stated} replicates"
        ));
    }
    gaps
}

fn read_host(run: &TtftRun, id: &RunIdentity, replicates_min: usize) -> HostResult {
    let mut gaps = Vec::new();
    let mut readings: Vec<Reading> = Vec::new();
    for (name, text) in &run.receipts {
        match read_receipt(name, text, id) {
            Ok(r) if readings.iter().any(|seen| seen.replicate == r.replicate) => {
                gaps.push(format!(
                    "{}/{name}: replicate {} appears twice",
                    id.host, r.replicate
                ));
            }
            Ok(r) => readings.push(r),
            Err(why) => gaps.push(why),
        }
    }
    gaps.extend(count_gaps(
        &id.host,
        readings.len(),
        id.replicates,
        replicates_min,
    ));
    let lines = readings.iter().map(|r| r.line.clone()).collect();
    let min_lcb95 = if gaps.is_empty() {
        Ok(readings
            .iter()
            .map(|r| r.lcb95)
            .fold(f64::INFINITY, f64::min))
    } else {
        Err(gaps)
    };
    HostResult { min_lcb95, lines }
}

/// The verdict: an outcome and the lines that justify it, verdict line last.
#[derive(Debug, Clone)]
pub struct TtftVerdict {
    /// GREEN, RED or NOT MEASURED.
    pub outcome: TtftOutcome,
    /// One line per receipt, one per gap, one per host, then the verdict line.
    pub lines: Vec<String>,
}

/// Identify each run's host; a run whose provenance is incomplete, or a host
/// claimed by two runs, is a gap.
fn identify(runs: &[TtftRun]) -> (Vec<(&TtftRun, RunIdentity)>, Vec<String>) {
    let mut found: Vec<(&TtftRun, RunIdentity)> = Vec::new();
    let mut gaps = Vec::new();
    for run in runs {
        match RunIdentity::read(&run.provenance) {
            Ok(id) if found.iter().any(|(_, seen)| seen.host == id.host) => {
                gaps.push(format!(
                    "{}: host {} claimed by two runs",
                    run.label, id.host
                ));
            }
            Ok(id) => found.push((run, id)),
            Err(why) => gaps.push(format!("{}: {why}", run.label)),
        }
    }
    (found, gaps)
}

/// Decide the row from the runs, against the policy.
#[must_use]
pub fn ttft_verdict(policy: &TtftPolicy, runs: &[TtftRun]) -> TtftVerdict {
    decide(policy, runs, Vec::new())
}

/// Decide the row from run directories. A directory that cannot be read is a
/// gap, so the row is NOT MEASURED rather than decided on the other runs.
#[must_use]
pub fn ttft_verdict_dirs(policy: &TtftPolicy, dirs: &[&Path]) -> TtftVerdict {
    let mut runs = Vec::new();
    let mut unreadable = Vec::new();
    for dir in dirs {
        match TtftRun::from_dir(dir) {
            Ok(run) => runs.push(run),
            Err(why) => unreadable.push(why),
        }
    }
    decide(policy, &runs, unreadable)
}

/// The verdict over `runs`, with `gaps` already found by the caller.
fn decide(policy: &TtftPolicy, runs: &[TtftRun], mut gaps: Vec<String>) -> TtftVerdict {
    let (found, identity_gaps) = identify(runs);
    gaps.extend(identity_gaps);
    let mut lines = Vec::new();
    let mut red = false;
    for host in &policy.hosts {
        let Some((run, id)) = found.iter().find(|(_, id)| &id.host == host) else {
            gaps.push(format!("{host}: no run names this host (arms.L3.v1.hosts)"));
            continue;
        };
        let result = read_host(run, id, policy.replicates_min);
        lines.extend(result.lines);
        match result.min_lcb95 {
            Ok(min) => {
                let below = min < policy.floor;
                red |= below;
                let word = if below { "below" } else { "at or above" };
                lines.push(format!("HOST {host} min lcb95 {min:.4} {word} the floor"));
            }
            Err(host_gaps) => gaps.extend(host_gaps),
        }
    }
    let outcome = if !gaps.is_empty() {
        TtftOutcome::NotMeasured
    } else if red {
        TtftOutcome::Red
    } else {
        TtftOutcome::Green
    };
    lines.extend(gaps.into_iter().map(|g| format!("NOT MEASURED {g}")));
    lines.push(format!(
        "VERDICT {} ttft ratio floor {} (perf-matrix.yaml arms.L3.v1.ttft_floor), hosts {}, {} replicates each",
        outcome.label(),
        policy.floor,
        policy.hosts.join(", "),
        policy.replicates_min
    ));
    TtftVerdict { outcome, lines }
}

#[cfg(test)]
mod tests {
    //! FALSIFY-APR-TTFT-001..009, 011 and 012 (`contracts/apr-ttft-ratio-verdict-v1.yaml`).
    //!
    //! Every receipt is replicate 1 of the lambda V1 run (#4954), a real receipt
    //! with its per-request samples and the comparator's server props emptied;
    //! the verdict reads neither. A case edits the one field it is about and
    //! re-derives `run_id`, so the receipt still validates.

    use super::*;
    use crate::perf_gate::RunId;
    use serde_json::{json, Value};

    const LAMBDA_R1: &str = include_str!("ttft_verdict_fixture.json");
    const SHA: &str = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4";
    const LLAMA_SHA: &str = "9aabba99701a9733af68d95db5f0452324a9c84daf49c55afb306d29eb30aa5c";

    fn policy() -> TtftPolicy {
        TtftPolicy {
            floor: 0.5,
            hosts: vec!["lambda".to_string(), "gx10".to_string()],
            replicates_min: 5,
        }
    }

    fn receipt_with(
        host: &str,
        replicate: u32,
        lcb95: f64,
        edit: impl FnOnce(&mut Value),
    ) -> String {
        let mut r: Value = serde_json::from_str(LAMBDA_R1).expect("the fixture is JSON");
        r["provenance"]["host"] = json!(host);
        r["bands"][0]["replicate"] = json!(replicate);
        r["bands"][0]["ratios"]["ttft"]["lcb95"] = json!(lcb95);
        edit(&mut r);
        let p = &r["provenance"];
        let id = RunId::derive(
            p["started_utc"].as_str().unwrap_or_default(),
            p["host"].as_str().unwrap_or_default(),
            p["client"]["sha256"].as_str().unwrap_or_default(),
            p["client"]["pid"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or_default(),
        );
        r["run_id"] = json!(id.as_str());
        if let Some(b) = r["bands"][0]["baseline"].as_object_mut() {
            b.insert("run_id".to_string(), json!(id.as_str()));
        }
        r.to_string()
    }

    fn receipt(host: &str, replicate: u32, lcb95: f64) -> String {
        receipt_with(host, replicate, lcb95, |_| {})
    }

    fn sidecar_with(host: &str, edit: impl FnOnce(&mut Value)) -> String {
        let mut s = json!({
            "host": host,
            "gguf": {"sha256": SHA},
            "comparator": {"pin": "d1d3c3396", "sha256": LLAMA_SHA},
            "band": {"concurrency": 1, "replicates": 5},
            "interval": {"method": "paired_percentile_bootstrap"}
        });
        edit(&mut s);
        s.to_string()
    }

    fn run_with(provenance: String, label: &str, receipts: Vec<String>) -> TtftRun {
        TtftRun {
            label: label.to_string(),
            provenance,
            receipts: receipts
                .into_iter()
                .enumerate()
                .map(|(i, t)| (format!("receipt.r{}.json", i + 1), t))
                .collect(),
        }
    }

    fn host(name: &str, lcbs: &[f64]) -> TtftRun {
        let receipts = lcbs
            .iter()
            .zip(1..)
            .map(|(&l, r)| receipt(name, r, l))
            .collect();
        run_with(sidecar_with(name, |_| {}), name, receipts)
    }

    fn green_pair() -> Vec<TtftRun> {
        vec![host("lambda", &[1.2; 5]), host("gx10", &[1.2; 5])]
    }

    fn says(v: &TtftVerdict, needle: &str) -> bool {
        v.lines.iter().any(|l| l.contains(needle))
    }

    fn assert_not_measured(v: &TtftVerdict, needle: &str) {
        assert_eq!(v.outcome, TtftOutcome::NotMeasured, "{:#?}", v.lines);
        assert_eq!(v.outcome.exit_code(), 2);
        assert!(says(v, needle), "no line names {needle:?}: {:#?}", v.lines);
    }

    /// FALSIFY-APR-TTFT-001: the floor is inclusive and decides.
    #[test]
    fn floor_at_the_floor_is_green_and_just_below_is_red() {
        let at = ttft_verdict(
            &policy(),
            &[host("lambda", &[0.5; 5]), host("gx10", &[0.5; 5])],
        );
        assert_eq!(at.outcome, TtftOutcome::Green, "{:#?}", at.lines);
        assert_eq!(at.outcome.exit_code(), 0);
        let gx10 = host("gx10", &[0.5, 0.5, 0.49, 0.5, 0.5]);
        let below = ttft_verdict(&policy(), &[host("lambda", &[0.5; 5]), gx10]);
        assert_eq!(below.outcome, TtftOutcome::Red, "{:#?}", below.lines);
        assert_eq!(below.outcome.exit_code(), 1);
        assert!(says(&below, "HOST gx10 min lcb95 0.4900 below the floor"));
    }

    /// The unedited lambda receipt parses, validates, and its own lcb95 is read.
    #[test]
    fn floor_reads_the_real_lambda_receipt() {
        let mut lambda = host("lambda", &[2.0; 5]);
        lambda.receipts[0].1 = LAMBDA_R1.to_string();
        let only = TtftPolicy {
            hosts: vec!["lambda".to_string()],
            ..policy()
        };
        let v = ttft_verdict(&only, &[lambda]);
        assert_eq!(v.outcome, TtftOutcome::Green, "{:#?}", v.lines);
        assert!(
            says(&v, "HOST lambda min lcb95 1.0765 at or above the floor"),
            "{:#?}",
            v.lines
        );
    }

    /// FALSIFY-APR-TTFT-002: the floor comes from the matrix, not the reader.
    #[test]
    fn floor_from_matrix() {
        let shipped = TtftPolicy::compiled().expect("the shipped matrix names arms.L3.v1");
        let edited = PERF_MATRIX_SOURCE.replacen(
            &format!("ttft_floor: {}", shipped.floor),
            "ttft_floor: 0.3",
            1,
        );
        assert_ne!(
            edited, PERF_MATRIX_SOURCE,
            "the edit must land, or this proves nothing"
        );
        let lowered = TtftPolicy::from_matrix(&edited).expect("the edited matrix parses");
        let runs = [host("lambda", &[0.5; 5]), host("gx10", &[0.49; 5])];
        assert_eq!(ttft_verdict(&shipped, &runs).outcome, TtftOutcome::Red);
        assert_eq!(ttft_verdict(&lowered, &runs).outcome, TtftOutcome::Green);
    }

    #[test]
    fn floor_from_matrix_refuses_a_matrix_without_the_row() {
        let none = TtftPolicy::from_matrix("protocol:\n  replicates_min: 5\narms:\n  L3: {}\n")
            .expect_err("no arms.L3.v1");
        assert!(none.contains("arms.L3.v1"), "{none}");
        let zero = TtftPolicy::from_matrix(
            "protocol:\n  replicates_min: 5\narms:\n  L3:\n    v1: {ttft_floor: 0, hosts: [lambda]}\n",
        )
        .expect_err("floor 0");
        assert!(zero.contains("ttft_floor"), "{zero}");
    }

    /// FALSIFY-APR-TTFT-003: a mean (1.04) would pass; the minimum (0.4) does not.
    #[test]
    fn min_over_replicates() {
        let lambda = host("lambda", &[1.2, 1.2, 1.2, 1.2, 0.4]);
        let v = ttft_verdict(&policy(), &[lambda, host("gx10", &[1.2; 5])]);
        assert_eq!(v.outcome, TtftOutcome::Red, "{:#?}", v.lines);
        assert!(says(&v, "HOST lambda min lcb95 0.4000 below the floor"));
        assert!(says(
            &v,
            "lambda/receipt.r5.json r5 c=1 NONCONFORMANT-VALID ttft lcb95 0.4000"
        ));
    }

    /// FALSIFY-APR-TTFT-004
    #[test]
    fn missing_host() {
        let v = ttft_verdict(&policy(), &[host("lambda", &[1.2; 5])]);
        assert_not_measured(&v, "gx10: no run names this host");
    }

    /// FALSIFY-APR-TTFT-005: four receipts that all clear the floor are still short.
    #[test]
    fn too_few_replicates() {
        let four = (1..=4).map(|r| receipt("lambda", r, 1.2)).collect();
        let claimed_four = sidecar_with("lambda", |s| s["band"]["replicates"] = json!(4));
        let runs = [
            run_with(claimed_four, "lambda", four),
            host("gx10", &[1.2; 5]),
        ];
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "lambda: 4 readable receipts, protocol.replicates_min is 5",
        );
    }

    /// FALSIFY-APR-TTFT-006, provenance half.
    #[test]
    fn missing_field_in_provenance_is_named() {
        for (parent, key, named) in [
            ("", "host", "provenance names no host"),
            ("/gguf", "sha256", "provenance names no GGUF sha256"),
            ("/comparator", "pin", "provenance names no comparator pin"),
            (
                "/comparator",
                "sha256",
                "provenance names no comparator sha256",
            ),
            (
                "/band",
                "replicates",
                "provenance names no replicate count (n)",
            ),
            ("/interval", "method", "provenance names no interval method"),
        ] {
            let mut runs = green_pair();
            runs[0].provenance = sidecar_with("lambda", |s| {
                s.pointer_mut(parent)
                    .and_then(Value::as_object_mut)
                    .expect("the field's parent exists")
                    .remove(key);
            });
            assert_not_measured(&ttft_verdict(&policy(), &runs), named);
        }
        let mut runs = green_pair();
        runs[0].provenance = sidecar_with("lambda", |s| s["gguf"]["sha256"] = json!("abc"));
        assert_not_measured(&ttft_verdict(&policy(), &runs), "is not a sha256");
    }

    /// FALSIFY-APR-TTFT-006, ratio half.
    #[test]
    fn missing_field_in_ratio_is_named() {
        for key in ["n", "method"] {
            let mut runs = green_pair();
            runs[0].receipts[2].1 = receipt_with("lambda", 3, 1.2, |r| {
                r["bands"][0]["ratios"]["ttft"]
                    .as_object_mut()
                    .expect("ttft is an object")
                    .remove(key);
            });
            assert_not_measured(
                &ttft_verdict(&policy(), &runs),
                &format!("missing field `{key}`"),
            );
        }
        let mut runs = green_pair();
        runs[0].receipts[0].1 = receipt_with("lambda", 1, 1.2, |r| {
            r["bands"][0]["ratios"]["ttft"]["n"] = json!(0)
        });
        assert_not_measured(&ttft_verdict(&policy(), &runs), "ratios.ttft names n = 0");
        let mut runs = green_pair();
        runs[0].receipts[0].1 = receipt_with("lambda", 1, 1.2, |r| {
            r["bands"][0]["ratios"]["ttft"]["method"] = json!("replicate_t_lower");
        });
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "ratio method replicate_t_lower differs",
        );
    }

    /// FALSIFY-APR-TTFT-007
    #[test]
    fn no_interval_is_not_measured() {
        let mut runs = green_pair();
        runs[1].receipts[4].1 = receipt_with("gx10", 5, 1.2, |r| {
            r["bands"][0]["ratios"]["ttft"] = Value::Null
        });
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "gx10/receipt.r5.json: no ratios.ttft",
        );
        let mut runs = green_pair();
        runs[1].receipts[4].1 = receipt_with("gx10", 5, 1.2, |r| {
            r["bands"][0]["ratios"]["ttft"]["lcb95"] = Value::Null
        });
        assert_not_measured(&ttft_verdict(&policy(), &runs), "ratios.ttft has no lcb95");
    }

    /// FALSIFY-APR-TTFT-008: statuses that do not count.
    #[test]
    fn band_status_that_does_not_count_is_not_measured() {
        for status in [
            "INVALID-CORRECTNESS",
            "COMPARATOR_STALE",
            "UNMEASURED",
            "NA",
        ] {
            let mut runs = green_pair();
            runs[0].receipts[0].1 = receipt_with("lambda", 1, 1.2, |r| {
                r["bands"][0]["status"] = json!(status)
            });
            assert_not_measured(
                &ttft_verdict(&policy(), &runs),
                &format!("band status {status}"),
            );
        }
        let mut runs = green_pair();
        runs[0].receipts[0].1 = receipt_with("lambda", 1, 1.2, |r| {
            r["bands"][0]["comparator_status"] = json!("COMPARATOR_STALE");
        });
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "comparator_status COMPARATOR_STALE",
        );
    }

    /// FALSIFY-APR-TTFT-008: statuses that count, NONCONFORMANT-VALID with its reasons.
    #[test]
    fn band_status_that_counts_prints_its_reasons() {
        let mut runs = green_pair();
        runs[0].receipts[0].1 = receipt_with("lambda", 1, 1.2, |r| {
            r["bands"][0]["status"] = json!("MEASURED")
        });
        let v = ttft_verdict(&policy(), &runs);
        assert_eq!(v.outcome, TtftOutcome::Green, "{:#?}", v.lines);
        assert!(says(&v, "lambda/receipt.r1.json r1 c=1 MEASURED"));
        assert!(says(
            &v,
            "lambda/receipt.r2.json r2 c=1 NONCONFORMANT-VALID"
        ));
        assert!(says(&v, "reason: PP-4 c=1:"), "{:#?}", v.lines);
    }

    /// FALSIFY-APR-TTFT-009
    #[test]
    fn not_measured_outranks_red() {
        let mut gx10 = host("gx10", &[1.2; 5]);
        gx10.receipts[0].1 = "{}".to_string();
        let v = ttft_verdict(&policy(), &[host("lambda", &[0.4; 5]), gx10]);
        assert_not_measured(&v, "gx10/receipt.r1.json: does not parse");
        assert!(
            says(&v, "HOST lambda min lcb95 0.4000 below the floor"),
            "{:#?}",
            v.lines
        );
    }

    /// FALSIFY-APR-TTFT-011: evidence from elsewhere is NOT MEASURED.
    #[test]
    fn anti_copy_receipt_from_another_host_or_run() {
        let mut runs = green_pair();
        runs[1].receipts[0].1 = receipt("lambda", 1, 1.2);
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "receipt host lambda differs from provenance host gx10",
        );

        let mut runs = green_pair();
        runs[0].receipts[1].1 = receipt("lambda", 1, 1.2);
        assert_not_measured(&ttft_verdict(&policy(), &runs), "replicate 1 appears twice");

        let mut runs = green_pair();
        runs.push(host("lambda", &[1.2; 5]));
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "host lambda claimed by two runs",
        );

        let mut runs = green_pair();
        runs[0]
            .receipts
            .push(("receipt.r6.json".to_string(), receipt("lambda", 6, 1.2)));
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "the provenance states 5 replicates",
        );

        let mut runs = green_pair();
        let mut tampered: Value = serde_json::from_str(&runs[0].receipts[0].1).expect("json");
        tampered["provenance"]["client"]["pid"] = json!(1);
        runs[0].receipts[0].1 = tampered.to_string();
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "is not a valid receipt: PP-3",
        );
    }

    /// FALSIFY-APR-TTFT-011: a ratio against another llama.cpp build is not
    /// this run's ratio.
    #[test]
    fn anti_copy_receipt_against_another_comparator_build() {
        let mut runs = green_pair();
        runs[0].receipts[2].1 = receipt_with("lambda", 3, 1.2, |r| {
            r["provenance"]["comparator"]["commit"] = json!("0123456789");
        });
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "receipt comparator commit 0123456789 differs from provenance comparator pin d1d3c3396",
        );

        let mut runs = green_pair();
        runs[1].receipts[4].1 = receipt_with("gx10", 5, 1.2, |r| {
            r["provenance"]["comparator"]["sha256"] = json!(SHA);
        });
        assert_not_measured(
            &ttft_verdict(&policy(), &runs),
            "differs from provenance comparator sha256",
        );

        let mut runs = green_pair();
        runs[0].provenance = sidecar_with("lambda", |s| s["comparator"]["sha256"] = json!("abc"));
        assert_not_measured(&ttft_verdict(&policy(), &runs), "is not a sha256");
    }

    // FALSIFY-APR-TTFT-012: the same verdict from run directories, as the CLI
    // reads them.
    fn write_dirs(runs: &[TtftRun]) -> (tempfile::TempDir, Vec<std::path::PathBuf>) {
        let root = tempfile::tempdir().expect("tempdir");
        let dirs = runs
            .iter()
            .map(|run| {
                let dir = root.path().join(&run.label);
                std::fs::create_dir(&dir).expect("mkdir");
                std::fs::write(dir.join(PROVENANCE_FILE), &run.provenance).expect("sidecar");
                for (name, text) in &run.receipts {
                    std::fs::write(dir.join(name), text).expect("receipt");
                }
                dir
            })
            .collect();
        (root, dirs)
    }

    fn verdict_of_dirs(dirs: &[std::path::PathBuf]) -> TtftVerdict {
        let dirs: Vec<&Path> = dirs.iter().map(std::path::PathBuf::as_path).collect();
        ttft_verdict_dirs(&policy(), &dirs)
    }

    #[test]
    fn dirs_give_each_outcome_its_exit_code() {
        let (_root, dirs) = write_dirs(&green_pair());
        let v = verdict_of_dirs(&dirs);
        assert_eq!(
            (v.outcome, v.outcome.exit_code()),
            (TtftOutcome::Green, 0),
            "{:#?}",
            v.lines
        );

        let red = [
            host("lambda", &[1.2; 5]),
            host("gx10", &[1.2, 1.2, 0.4, 1.2, 1.2]),
        ];
        let (_root, dirs) = write_dirs(&red);
        let v = verdict_of_dirs(&dirs);
        assert_eq!(
            (v.outcome, v.outcome.exit_code()),
            (TtftOutcome::Red, 1),
            "{:#?}",
            v.lines
        );

        let (root, mut dirs) = write_dirs(&green_pair());
        dirs.push(root.path().join("never-written"));
        assert_not_measured(&verdict_of_dirs(&dirs), PROVENANCE_FILE);
    }

    #[test]
    fn dirs_read_receipts_in_replicate_order_and_nothing_else() {
        let root = tempfile::tempdir().expect("tempdir");
        let dir = root.path();
        for name in [
            PROVENANCE_FILE,
            "receipt.r10.json",
            "receipt.r2.json",
            "receipt.r1.json.sig",
            "receipt.rx.json",
            "notes.txt",
        ] {
            std::fs::write(dir.join(name), name).expect("write");
        }
        let run = TtftRun::from_dir(dir).expect("readable");
        let names: Vec<&str> = run.receipts.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["receipt.r2.json", "receipt.r10.json"]);
        assert_eq!(run.receipts[1].1, "receipt.r10.json");
        assert_eq!(run.provenance, PROVENANCE_FILE);
    }

    // FALSIFY-APR-TTFT-015: the gap split by phase (#4954).

    use crate::perf_gate::drain::RequestOutcome;

    const PLANT: f64 = 100.0;

    /// A completed row whose TTFT is its four parts: load, prefill, first
    /// token, transfer. Built through `to_row`, as the band runner builds it.
    fn split_row(
        load: Option<f64>,
        prefill: f64,
        first_token: Option<f64>,
        transfer: f64,
    ) -> SampleRow {
        let ttft = load.unwrap_or(0.0) + prefill + first_token.unwrap_or(0.0) + transfer;
        RequestOutcome::completed(0.0, ttft + 400.0, 128)
            .streamed(ttft, Vec::new())
            .server_prefill(512, prefill)
            .server_edges(load, first_token)
            .to_row(0)
    }

    /// Three `apr` rows around `parts` (load, prefill, first token, transfer);
    /// the p50 is the middle row, `parts` plus 1 ms each.
    fn apr_rows(parts: [f64; 4]) -> Vec<SampleRow> {
        (0..3)
            .map(|i| {
                let d = f64::from(i);
                split_row(
                    Some(parts[0] + d),
                    parts[1] + d,
                    Some(parts[2] + d),
                    parts[3] + d,
                )
            })
            .collect()
    }

    /// Three reference rows: llama.cpp reports prefill and no edges.
    fn reference_rows() -> Vec<SampleRow> {
        (0..3)
            .map(|i| split_row(None, 40.0 + f64::from(i), None, 60.0))
            .collect()
    }

    fn parts(m: &PhaseMedians) -> [Option<f64>; 4] {
        [m.load, m.prefill, m.first_token, m.transfer]
    }

    #[test]
    fn phases_a_delay_planted_in_one_part_moves_that_part_and_only_that_part() {
        let base_parts = [5.0, 40.0, 5.0, 10.0];
        let base = PhaseMedians::of(&apr_rows(base_parts));
        assert_eq!(
            parts(&base),
            [Some(6.0), Some(41.0), Some(6.0), Some(11.0)],
            "{base:?}"
        );
        assert_eq!(base.ttft, Some(64.0), "{base:?}");
        assert_eq!(base.outside_prefill, Some(23.0), "{base:?}");
        for planted in 0..4 {
            let mut moved_parts = base_parts;
            moved_parts[planted] += PLANT;
            let moved = PhaseMedians::of(&apr_rows(moved_parts));
            for (part, (got, was)) in parts(&moved).into_iter().zip(parts(&base)).enumerate() {
                let want = if part == planted {
                    was.map(|ms| ms + PLANT)
                } else {
                    was
                };
                assert_eq!(
                    got, want,
                    "a delay in part {planted} moved part {part}: {moved:?}"
                );
            }
            assert_eq!(moved.ttft, base.ttft.map(|ms| ms + PLANT), "{moved:?}");
            let outside = if planted == 1 {
                base.outside_prefill
            } else {
                base.outside_prefill.map(|ms| ms + PLANT)
            };
            assert_eq!(moved.outside_prefill, outside, "{moved:?}");
        }
    }

    #[test]
    fn phases_a_part_no_row_reports_is_absent_not_zero() {
        let mut rows = reference_rows();
        // A request that did not complete is no part of any phase.
        let mut timed_out = split_row(Some(9e3), 9e3, Some(9e3), 9e3);
        timed_out.outcome = Outcome::Timeout;
        rows.push(timed_out);
        let m = PhaseMedians::of(&rows);
        assert_eq!(m.prefill, Some(41.0), "{m:?}");
        assert_eq!(m.ttft, Some(101.0), "{m:?}");
        assert_eq!(m.outside_prefill, Some(60.0), "{m:?}");
        assert_eq!(
            (m.load, m.first_token, m.transfer),
            (None, None, None),
            "{m:?}"
        );

        // Rows from before the split carry no prefill: nothing outside it either.
        let mut bare = split_row(None, 40.0, None, 60.0);
        bare.prefill_ms = None;
        let m = PhaseMedians::of(&[bare]);
        assert_eq!(m.ttft, Some(100.0), "{m:?}");
        assert_eq!((m.prefill, m.outside_prefill), (None, None), "{m:?}");
        assert_eq!(PhaseMedians::of(&[]), PhaseMedians::default());
    }

    #[test]
    fn phases_are_printed_beside_the_reading_and_never_move_the_outcome() {
        let phased = |name: &str, lcb95: f64| -> TtftRun {
            let receipts = (1..=5)
                .map(|r| {
                    receipt_with(name, r, lcb95, |v| {
                        v["bands"][0]["samples"] = json!(apr_rows([5.0, 40.0, 5.0, 10.0]));
                        v["bands"][0]["baseline"]["samples"] = json!(reference_rows());
                    })
                })
                .collect();
            run_with(sidecar_with(name, |_| {}), name, receipts)
        };
        for lcb95 in [1.2, 0.4] {
            let plain = ttft_verdict(
                &policy(),
                &[host("lambda", &[lcb95; 5]), host("gx10", &[lcb95; 5])],
            );
            let with = ttft_verdict(&policy(), &[phased("lambda", lcb95), phased("gx10", lcb95)]);
            assert_eq!(with.outcome, plain.outcome, "{:#?}", with.lines);
            assert!(
                says(
                    &with,
                    "prefill 41.0 vs 41.0 (gap +0.0); outside prefill 23.0 vs 60.0 (gap -37.0)"
                ),
                "{:#?}",
                with.lines
            );
            assert!(
                says(
                    &with,
                    "apr outside prefill: load 6.0, first token 6.0, transfer 11.0"
                ),
                "{:#?}",
                with.lines
            );
        }
        // A receipt from before the split prints its parts as absent.
        let plain = ttft_verdict(&policy(), &green_pair());
        assert!(says(&plain, "prefill - vs - (gap -)"), "{:#?}", plain.lines);
    }
}
