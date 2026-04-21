//! CRUX-B-11 — `apr fp8-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the two FP8 (E4M3) classifiers in `fp8_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "frobenius": {
//!     "original":      [f32, ...],
//!     "reconstructed": [f32, ...],
//!     "threshold":     0.01
//!   },
//!   "capability": { "sm": 90 }
//! }
//! ```
//!
//! Any missing top-level key is skipped. The CLI exits non-zero on any
//! failing gate and stamps the FALSIFY id in stderr.

use crate::commands::fp8_classifier::{
    classify_frobenius_error, classify_sm_capability, CapabilityOutcome, FrobeniusOutcome,
    FP8_MAX_FROBENIUS_REL_ERR,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Fp8LintArgs {
    pub observation_file: String,
    pub json: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct GateReport {
    gate: &'static str,
    falsify_id: &'static str,
    outcome: String,
    passed: bool,
}

pub fn run(args: Fp8LintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-B-11: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-B-11: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-B-11: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-B-11: observation is not valid JSON: {e}"))?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(frob) = obs.get("frobenius") {
        let (report, err) = run_frobenius_gate(frob);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(cap) = obs.get("capability") {
        let (report, err) = run_capability_gate(cap);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err("FALSIFY-CRUX-B-11: observation has neither frobenius nor capability".into());
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-11",
            "gates": reports,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    } else {
        for r in &reports {
            let tag = if r.passed { "PASS" } else { "FAIL" };
            println!("[{tag}] {} ({}): {}", r.gate, r.falsify_id, r.outcome);
        }
    }

    if !failures.is_empty() {
        return Err(failures.join("\n"));
    }
    Ok(())
}

fn read_f32_array(v: &Value) -> Vec<f32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default()
}

fn run_frobenius_gate(v: &Value) -> (GateReport, Option<String>) {
    let original = v.get("original").map(read_f32_array).unwrap_or_default();
    let reconstructed = v
        .get("reconstructed")
        .map(read_f32_array)
        .unwrap_or_default();
    let threshold = v
        .get("threshold")
        .and_then(|x| x.as_f64())
        .unwrap_or(FP8_MAX_FROBENIUS_REL_ERR);
    let outcome = classify_frobenius_error(&original, &reconstructed, threshold);
    let passed = matches!(outcome, FrobeniusOutcome::Ok { .. });
    let desc = format!("{outcome:?}");
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-11-001 frobenius gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "frobenius",
            falsify_id: "FALSIFY-CRUX-B-11-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_capability_gate(v: &Value) -> (GateReport, Option<String>) {
    let sm = v.get("sm").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let outcome = classify_sm_capability(sm);
    let passed = matches!(outcome, CapabilityOutcome::Capable { .. });
    let desc = format!("{outcome:?}");
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-11-002 capability gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "capability",
            falsify_id: "FALSIFY-CRUX-B-11-002",
            outcome: desc,
            passed,
        },
        err,
    )
}
