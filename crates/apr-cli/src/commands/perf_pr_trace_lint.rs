//! `apr perf-pr-trace-lint` — APR-OBS-001 OBS-11, contract `apr-perf-pr-trace-v1`
//! (aprender#4498).
//!
//! §5.2: "Every 0.71 performance PR carries a before/after apr-trace-v1 diff for
//! the layers it claims to change, measured on the same host at the same
//! identity. A perf PR without it does not merge."
//!
//! The evidence is one JSON document committed with the PR:
//!
//! ```json
//! { "schema": "apr-perf-pr-trace-v1", "pr": 4498,
//!   "claimed": ["attention", "ffn"],
//!   "before": { "commit": "<base sha>", "identity": {…§2.1…}, "trace": {…TraceData…} },
//!   "after":  { "commit": "<head sha>", "identity": {…§2.1…}, "trace": {…TraceData…} } }
//! ```
//!
//! `identity` is the §2.1 row identity block (`apr-obs-row-identity-v1`), with
//! `schema = "apr-trace-v1"`. `trace` is realizar's `TraceData` (`level`,
//! `operations`, `total_time_us`, `breakdown[{name, time_us}]`, `provenance`).
//!
//! A gate rejection (exit 5) on any of: a wrong schema or an empty `claimed`
//! list; an identity field absent on either side (R-2: absent is never a pass);
//! the GPU rule broken; a side on `backend_unproven`; the two sides differing on
//! host, weights, backend or build (target, rustc, features, accelerator,
//! driver); a trace whose timings are not `measured`; a breakdown summing past
//! its total; a claimed layer missing from either breakdown; the two sides on
//! the same commit; or a commit that is not the `--base` / `--head` given.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::{CliError, Result};

pub(crate) const SCHEMA: &str = "apr-perf-pr-trace-v1";

/// §2.1: every field REQUIRED on each side.
const IDENTITY_FIELDS: &[&str] = &[
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
    "gpu_proof",
    "request_id",
];

const BUILD_FIELDS: &[&str] = &[
    "rustc_vv",
    "target_triple",
    "features",
    "uname_a",
    "accelerator",
    "driver",
];

/// "The same host at the same identity": what must NOT differ between the sides.
/// The crate tarball and binary differ by construction (the PR changed the code),
/// and `uname_a` is left out so a kernel update between runs is not a rejection.
const SAME_TOP: &[&str] = &["host", "model_sha256", "backend"];
const SAME_BUILD: &[&str] = &[
    "target_triple",
    "rustc_vv",
    "features",
    "accelerator",
    "driver",
];

const BACKENDS: &[&str] = &["cpu", "cuda", "wgpu", "metal", "backend_unproven"];

/// One claimed layer's timing on each side.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerDelta {
    pub name: String,
    pub before_us: f64,
    pub after_us: f64,
}

impl LayerDelta {
    pub(crate) fn delta_pct(&self) -> Option<f64> {
        (self.before_us > 0.0).then(|| (self.after_us - self.before_us) / self.before_us * 100.0)
    }
}

#[derive(Debug, Default)]
pub(crate) struct Report {
    pub findings: Vec<String>,
    pub layers: Vec<LayerDelta>,
}

fn is_sha256(v: &Value) -> bool {
    v.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    })
}

fn check_identity(side: &str, id: &Value, out: &mut Vec<String>) {
    let Some(obj) = id.as_object() else {
        out.push(format!("{side}.identity: missing or not an object"));
        return;
    };
    for f in IDENTITY_FIELDS {
        if !obj.contains_key(*f) {
            out.push(format!(
                "{side}.identity.{f}: absent (R-2: an absent key is never a pass)"
            ));
        }
    }
    if obj.contains_key("schema") && id["schema"] != "apr-trace-v1" {
        out.push(format!(
            "{side}.identity.schema: {} is not apr-trace-v1",
            id["schema"]
        ));
    }
    for f in ["crate_tarball_sha256", "binary_sha256", "model_sha256"] {
        if obj.contains_key(f) && !is_sha256(&id[f]) {
            out.push(format!("{side}.identity.{f}: not a lowercase sha256"));
        }
    }
    check_build(side, id, out);
    check_backend(side, id, out);
}

/// The §2.1 build_identity object and its six fields.
fn check_build(side: &str, id: &Value, out: &mut Vec<String>) {
    match id.get("build_identity").and_then(Value::as_object) {
        Some(b) => {
            for f in BUILD_FIELDS {
                if !b.contains_key(*f) {
                    out.push(format!("{side}.identity.build_identity.{f}: absent"));
                }
            }
        }
        None if id.get("build_identity").is_some() => {
            out.push(format!("{side}.identity.build_identity: not an object"));
        }
        None => {}
    }
}

/// The backend value and the §2.5 GPU rule.
fn check_backend(side: &str, id: &Value, out: &mut Vec<String>) {
    let backend = id["backend"].as_str().unwrap_or_default();
    if id.get("backend").is_some() && !BACKENDS.contains(&backend) {
        out.push(format!(
            "{side}.identity.backend: {} is not one of {BACKENDS:?}",
            id["backend"]
        ));
    }
    if backend == "backend_unproven" {
        out.push(format!(
            "{side}.identity.backend: backend_unproven — a GPU claim without proof is not perf evidence (§2.5)"
        ));
    }
    let proof_null = id.get("gpu_proof").is_none_or(Value::is_null);
    if backend == "cpu" && !proof_null {
        out.push(format!(
            "{side}.identity.gpu_proof: must be null on backend cpu (§2.5)"
        ));
    }
    if ["cuda", "wgpu", "metal"].contains(&backend) && proof_null {
        out.push(format!(
            "{side}.identity.gpu_proof: null on backend {backend} (§2.5)"
        ));
    }
}

/// `breakdown` name -> time_us, or findings when the trace is not usable evidence.
fn check_trace(side: &str, t: &Value, out: &mut Vec<String>) -> Vec<(String, f64)> {
    if !t.is_object() {
        out.push(format!("{side}.trace: missing or not an object"));
        return Vec::new();
    }
    match t["provenance"].as_str() {
        Some("measured") => {}
        Some(p) => out.push(format!(
            "{side}.trace.provenance: {p} — only measured per-layer timings are evidence"
        )),
        None => out.push(format!("{side}.trace.provenance: absent")),
    }
    let Some(total) = t["total_time_us"].as_f64() else {
        out.push(format!(
            "{side}.trace.total_time_us: absent or not a number"
        ));
        return Vec::new();
    };
    let Some(rows) = t["breakdown"].as_array().filter(|a| !a.is_empty()) else {
        out.push(format!("{side}.trace.breakdown: absent or empty"));
        return Vec::new();
    };
    let mut layers = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        match (r["name"].as_str(), r["time_us"].as_f64()) {
            (Some(n), Some(us)) if us >= 0.0 => layers.push((n.to_string(), us)),
            _ => out.push(format!(
                "{side}.trace.breakdown[{i}]: needs a name and time_us >= 0"
            )),
        }
    }
    let sum: f64 = layers.iter().map(|(_, us)| us).sum();
    if sum > total {
        out.push(format!(
            "{side}.trace: breakdown sums to {sum} us, past total_time_us {total}"
        ));
    }
    layers
}

fn check_commit(
    side: &str,
    got: &Value,
    want: Option<&str>,
    out: &mut Vec<String>,
) -> Option<String> {
    let Some(c) = got
        .as_str()
        .filter(|c| c.len() >= 7 && c.bytes().all(|b| b.is_ascii_hexdigit()))
    else {
        out.push(format!("{side}.commit: absent or not a commit sha"));
        return None;
    };
    if let Some(w) = want {
        if !(w.starts_with(c) || c.starts_with(w)) {
            out.push(format!("{side}.commit: {c} is not the PR's {w}"));
        }
    }
    Some(c.to_string())
}

/// "The same host at the same identity" (§5.2).
fn check_same_identity(b: &Value, a: &Value, out: &mut Vec<String>) {
    for f in SAME_TOP {
        if b[f] != a[f] {
            out.push(format!(
                "identity.{f}: before {} != after {} — not the same host and identity",
                b[f], a[f]
            ));
        }
    }
    for f in SAME_BUILD {
        let (x, y) = (&b["build_identity"][f], &a["build_identity"][f]);
        if x != y {
            out.push(format!(
                "identity.build_identity.{f}: before {x} != after {y}"
            ));
        }
    }
}

/// The whole OBS-11 gate over one evidence document.
pub(crate) fn check(doc: &Value, base: Option<&str>, head: Option<&str>) -> Report {
    let mut r = Report::default();
    let out = &mut r.findings;
    if doc["schema"] != SCHEMA {
        out.push(format!("schema: {} is not {SCHEMA}", doc["schema"]));
    }
    let claimed: Vec<&str> = doc["claimed"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if claimed.is_empty() {
        out.push("claimed: absent or empty — name the layers the PR changes".to_string());
    }
    let (b, a) = (&doc["before"], &doc["after"]);
    let bc = check_commit("before", &b["commit"], base, out);
    let ac = check_commit("after", &a["commit"], head, out);
    if bc.is_some() && bc == ac {
        out.push("before.commit == after.commit — nothing was compared".to_string());
    }
    check_identity("before", &b["identity"], out);
    check_identity("after", &a["identity"], out);
    check_same_identity(&b["identity"], &a["identity"], out);
    let bl = check_trace("before", &b["trace"], out);
    let al = check_trace("after", &a["trace"], out);
    let find = |v: &[(String, f64)], n: &str| v.iter().find(|(k, _)| k == n).map(|(_, us)| *us);
    for name in claimed {
        match (find(&bl, name), find(&al, name)) {
            (Some(before_us), Some(after_us)) => r.layers.push(LayerDelta {
                name: name.to_string(),
                before_us,
                after_us,
            }),
            (x, y) => {
                for (side, got) in [("before", x), ("after", y)] {
                    if got.is_none() {
                        r.findings
                            .push(format!("claimed layer {name}: not in the {side} breakdown"));
                    }
                }
            }
        }
    }
    r
}

fn print(path: &Path, r: &Report, json: bool) {
    if json {
        let layers: Vec<Value> = r
            .layers
            .iter()
            .map(|l| {
                serde_json::json!({
                    "name": l.name, "before_us": l.before_us,
                    "after_us": l.after_us, "delta_pct": l.delta_pct(),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "schema": SCHEMA, "file": path, "ok": r.findings.is_empty(),
                "findings": r.findings, "layers": layers,
            })
        );
        return;
    }
    println!("perf-pr-trace-lint {}", path.display());
    for l in &r.layers {
        let d = l
            .delta_pct()
            .map_or_else(|| "n/a".to_string(), |d| format!("{d:+.1}%"));
        println!(
            "  {:<24} before {:>12.1} us  after {:>12.1} us  {d}",
            l.name, l.before_us, l.after_us
        );
    }
    for f in &r.findings {
        println!("  RED {f}");
    }
    let verdict = if r.findings.is_empty() {
        "GREEN"
    } else {
        "RED"
    };
    println!("  {verdict}: {} finding(s)", r.findings.len());
}

/// `apr perf-pr-trace-lint <FILE> [--base SHA] [--head SHA]`.
///
/// # Errors
///
/// `FileNotFound` (3) for a missing file, `InvalidInput` (4) for one that is not
/// a JSON document, `ValidationFailed` (5) when the OBS-11 gate rejects it.
pub(crate) fn run(path: &Path, base: Option<&str>, head: Option<&str>, json: bool) -> Result<()> {
    if !path.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(path)));
    }
    if !path.is_file() {
        return Err(CliError::InvalidInput(format!(
            "apr perf-pr-trace-lint: not a file: {}",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path)?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| {
        CliError::InvalidInput(format!(
            "apr perf-pr-trace-lint: failed to parse JSON from {}: {e}",
            path.display()
        ))
    })?;
    let r = check(&doc, base, head);
    print(path, &r, json);
    if r.findings.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "apr perf-pr-trace-lint: {} finding(s) against {SCHEMA} (OBS-11, §5.2)",
            r.findings.len()
        )))
    }
}

#[cfg(test)]
#[path = "perf_pr_trace_lint_tests.rs"]
mod tests;
