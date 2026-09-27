//! EXT-19 (aprender#4401) reading APR-OBS rows (aprender#4551, OBS-05): the
//! speed gate's input when the rows come from `apr-perf-ledger-v1` — the one
//! ledger `apr bench --json` and the nightly fleet timer write — instead of the
//! EXT-26 row shape in [`super::speed_ledger`].
//!
//! Admission follows `contracts/apr-obs-row-identity-v1.yaml`: a row missing
//! any identity field, or carrying it as null or `"unknown"`, is inadmissible,
//! and an inadmissible row counts as ABSENT — it cannot fill a (tag, cell)
//! pair. A GPU row without `gpu_proof` is `backend_unproven` and is excluded
//! the same way. The gate names each rejected line and why.
//!
//! A cell is the declared `host/backend` series, supplied by the caller and
//! never derived from the rows present (FALSIFY-OBS-PERF-001: a host that
//! stopped writing must read as a hole). A ratio is only formed between rows
//! of one model: the judged record's `model_sha256` selects its priors, so a
//! model change resets the baseline rather than comparing across it. A
//! `provenance: backfill` row can fill its own pair but is never a prior.
//!
//! A `not_run: {reason}` row (in place of `apr`/`llama`, per the #4551 schema
//! owner) fills its pair — EXT-19's `NotRun{reason}` — but is never a record.
//! It still carries the full identity block; `gpu_proof` may be null on it,
//! since a cell that did not run claims nothing about a GPU.
//!
//! As in the EXT-26 path, the apr / llama.cpp ratio lives only inside
//! [`Ratchet`]; the printed report carries verdicts, never numbers (T28).

use super::speed_gate::GateReport;
use super::speed_ledger::{judge, Ratchet};
use serde_json::Value;

pub(crate) const PERF_SCHEMA: &str = "apr-perf-ledger-v1";

/// `apr-obs-row-identity-v1` §2.1 — every admissible row carries all of these.
pub(crate) const IDENTITY: [&str; 12] = [
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

/// `append_only_atomic_lines`: a longer line was not written atomically.
pub(crate) const MAX_LINE_BYTES: usize = 4096;

/// One admitted row, reduced to what the ratchet needs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PerfRecord {
    pub tag: String,
    pub cell: String,
    pub ts: String,
    pub model_sha256: String,
    pub backfill: bool,
    /// apr / llama.cpp decode; `None` for a `not_run` row.
    ratio: Option<f64>,
}

fn present<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    match v.get(k) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() || s == "unknown" => None,
        Some(x) => Some(x),
    }
}

fn text(v: &Value, k: &str) -> Result<String, String> {
    present(v, k)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("identity field `{k}` missing, null or unknown"))
}

fn tok_s(v: &Value, side: &str) -> Result<f64, String> {
    let x = v
        .get(side)
        .and_then(|s| s.get("tg128_tok_s"))
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("`{side}.tg128_tok_s` missing"))?;
    if x.is_finite() && x > 0.0 {
        Ok(x)
    } else {
        Err(format!("`{side}.tg128_tok_s` is not a positive number"))
    }
}

/// Admit one `apr-perf-ledger-v1` row, or say why it counts as absent.
/// `llama_pin` is the `scripts/llama_pin.toml` commit the reference arm must
/// have been built from.
pub(crate) fn admit(v: &Value, llama_pin: &str) -> Result<PerfRecord, String> {
    for k in IDENTITY {
        if present(v, k).is_none() {
            return Err(format!("identity field `{k}` missing, null or unknown"));
        }
    }
    let schema = text(v, "schema")?;
    if schema != PERF_SCHEMA {
        return Err(format!("schema `{schema}` is not {PERF_SCHEMA}"));
    }
    let backend = text(v, "backend")?;
    let not_run = v.get("not_run").is_some();
    match v.get("gpu_proof") {
        None => return Err("`gpu_proof` key absent".into()),
        Some(Value::Null) if backend != "cpu" && !not_run => {
            return Err(format!(
                "backend_unproven: {backend} row has null gpu_proof"
            ));
        }
        _ => {}
    }
    let ratio = if not_run {
        not_run_reason(v)?;
        None
    } else {
        Some(measured_ratio(v, llama_pin)?)
    };
    Ok(PerfRecord {
        tag: text(v, "apr_tag")?,
        cell: format!("{}/{backend}", text(v, "host")?),
        ts: text(v, "ts")?,
        model_sha256: text(v, "model_sha256")?,
        backfill: v.get("provenance").and_then(Value::as_str) == Some("backfill"),
        ratio,
    })
}

/// A `not_run` row names why, and carries no measurement beside it.
fn not_run_reason(v: &Value) -> Result<(), String> {
    let reason = v
        .get("not_run")
        .and_then(|n| n.get("reason"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if reason.trim().is_empty() {
        return Err("`not_run.reason` missing or empty".into());
    }
    if v.get("apr").is_some() || v.get("llama").is_some() {
        return Err("a `not_run` row also carries a measurement".into());
    }
    Ok(())
}

/// The workload floor, the llama.cpp pin, and apr / llama.cpp decode.
fn measured_ratio(v: &Value, llama_pin: &str) -> Result<f64, String> {
    if v.get("prompt_n").and_then(Value::as_u64).unwrap_or(0) < 8 {
        return Err("`prompt_n` below 8".into());
    }
    if v.get("reps").and_then(Value::as_u64) != Some(5) {
        return Err("`reps` is not 5".into());
    }
    let built = v
        .get("llama")
        .and_then(|l| l.get("build_commit"))
        .and_then(Value::as_str);
    if built != Some(llama_pin) {
        return Err("`llama.build_commit` is not the scripts/llama_pin.toml commit".into());
    }
    Ok(tok_s(v, "apr")? / tok_s(v, "llama")?)
}

/// Admitted records plus each rejected line (1-based) and its reason. A line
/// that is not JSON fails the whole ledger; an empty ledger is RED, not green.
pub(crate) fn parse_perf_ledger(
    jsonl: &str,
    llama_pin: &str,
) -> Result<(Vec<PerfRecord>, Vec<(usize, String)>), String> {
    let (mut ok, mut rejected) = (Vec::new(), Vec::new());
    let mut lines = 0;
    for (i, line) in jsonl.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        lines += 1;
        if line.len() > MAX_LINE_BYTES {
            rejected.push((i + 1, format!("line over {MAX_LINE_BYTES} bytes")));
            continue;
        }
        let v: Value =
            serde_json::from_str(line).map_err(|e| format!("ledger line {}: {e}", i + 1))?;
        match admit(&v, llama_pin) {
            Ok(r) => ok.push(r),
            Err(why) => rejected.push((i + 1, why)),
        }
    }
    if lines == 0 {
        return Err("perf ledger is empty: an empty ledger is RED".into());
    }
    Ok((ok, rejected))
}

/// The newest (by `ts`) admitted record for one pair.
fn newest<'a>(recs: &'a [PerfRecord], tag: &str, cell: &str) -> Option<&'a PerfRecord> {
    recs.iter()
        .filter(|r| r.tag == tag && r.cell == cell)
        .max_by(|a, b| a.ts.cmp(&b.ts))
}

/// One cell's verdict: its newest measured record in release order is judged
/// against the non-backfill priors of the same model. `not_run` pairs are
/// covered but are not records.
fn perf_ratchet(recs: &[PerfRecord], tags: &[String], cell: &str) -> Ratchet {
    let picked: Vec<(&PerfRecord, f64)> = tags
        .iter()
        .filter_map(|t| newest(recs, t, cell))
        .filter_map(|r| r.ratio.map(|x| (r, x)))
        .collect();
    let Some((judged, prior)) = picked.split_last() else {
        return Ratchet::Unarmed { records: 0 };
    };
    let mut s: Vec<(&str, f64)> = prior
        .iter()
        .filter(|(r, _)| !r.backfill && r.model_sha256 == judged.0.model_sha256)
        .map(|(r, x)| (r.tag.as_str(), *x))
        .collect();
    s.push((judged.0.tag.as_str(), judged.1));
    judge(&s)
}

/// The gate's verdict over an APR-OBS perf ledger, with the rejected lines.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PerfGateReport {
    pub gate: GateReport,
    pub rejected: Vec<(usize, String)>,
}

impl PerfGateReport {
    pub(crate) fn passed(&self) -> bool {
        self.gate.passed()
    }

    /// Rejected lines first (field names only), then the EXT-19 report.
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        for (n, why) in &self.rejected {
            out.push_str(&format!("ABSENT   line {n}: {why}\n"));
        }
        out.push_str(&self.gate.render());
        out
    }
}

/// Judge an `apr-perf-ledger-v1` JSONL ledger against the release's tags (in
/// release order) and its declared `host/backend` cells.
pub(crate) fn perf_gate(
    ledger_jsonl: &str,
    llama_pin: &str,
    tags: &[String],
    cells: &[String],
) -> Result<PerfGateReport, String> {
    if tags.is_empty() || cells.is_empty() {
        return Err("speed gate needs at least one tag and one cell".into());
    }
    let (recs, rejected) = parse_perf_ledger(ledger_jsonl, llama_pin)?;
    let holes = tags
        .iter()
        .flat_map(|t| cells.iter().map(move |c| (t, c)))
        .filter(|(t, c)| newest(&recs, t, c).is_none())
        .map(|(t, c)| (t.clone(), c.clone()))
        .collect();
    let cells_v = cells
        .iter()
        .map(|c| (c.clone(), perf_ratchet(&recs, tags, c)))
        .collect();
    Ok(PerfGateReport {
        gate: GateReport {
            holes,
            cells: cells_v,
            pairs: tags.len() * cells.len(),
        },
        rejected,
    })
}

#[cfg(test)]
#[path = "speed_perf_rows_tests.rs"]
mod tests;
