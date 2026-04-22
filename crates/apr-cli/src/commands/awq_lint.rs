//! CRUX-B-08 — `apr awq-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the three AWQ classifiers in `awq_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "quality": {
//!     "p_fp16":    0.50,
//!     "p_awq":     0.45,
//!     "threshold": 0.80
//!   },
//!   "compression": {
//!     "fp16_bytes":  1_000_000_000,
//!     "awq_bytes":     250_000_000,
//!     "max_ratio":           0.30
//!   },
//!   "flags": {
//!     "argv":             ["--method", "awq", "--bits", "4", "--group-size", "128"],
//!     "expected_outcome": "ok"    // ok | missing_method | unknown_method | invalid_bits | invalid_group_size
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-B-08
//! stderr stamp on any failing gate.

use crate::commands::awq_classifier::{
    classify_compression_ratio, classify_quality_retention, parse_awq_flags, validate_awq_flags,
    AwqFlagValidation, CompressionOutcome, QualityRetention, AWQ_MAX_COMPRESSION_RATIO,
    AWQ_MIN_QUALITY_RETENTION,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct AwqLintArgs {
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

pub fn run(args: AwqLintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-B-08: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-B-08: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-B-08: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-B-08: observation is not valid JSON: {e}"))?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(q) = obs.get("quality") {
        let (report, err) = run_quality_gate(q);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(c) = obs.get("compression") {
        let (report, err) = run_compression_gate(c);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(f) = obs.get("flags") {
        let (report, err) = run_flags_gate(f);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err("FALSIFY-CRUX-B-08: observation has none of quality/compression/flags".into());
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-08",
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

fn run_quality_gate(v: &Value) -> (GateReport, Option<String>) {
    let p_fp16 = v.get("p_fp16").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let p_awq = v.get("p_awq").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let threshold = v
        .get("threshold")
        .and_then(|x| x.as_f64())
        .unwrap_or(AWQ_MIN_QUALITY_RETENTION);
    let outcome = classify_quality_retention(p_fp16, p_awq, threshold);
    let (passed, desc) = match outcome {
        QualityRetention::Retained { ratio } => (
            true,
            format!("ratio={ratio:.4} >= {threshold} (p_fp16={p_fp16}, p_awq={p_awq})"),
        ),
        QualityRetention::Degraded { ratio, threshold } => (
            false,
            format!("ratio={ratio:.4} < {threshold} (p_fp16={p_fp16}, p_awq={p_awq})"),
        ),
    };
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-08-001 quality gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "quality",
            falsify_id: "FALSIFY-CRUX-B-08-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_compression_gate(v: &Value) -> (GateReport, Option<String>) {
    let fp16 = v.get("fp16_bytes").and_then(|x| x.as_u64()).unwrap_or(0);
    let awq = v.get("awq_bytes").and_then(|x| x.as_u64()).unwrap_or(0);
    let max_ratio = v
        .get("max_ratio")
        .and_then(|x| x.as_f64())
        .unwrap_or(AWQ_MAX_COMPRESSION_RATIO);
    let outcome = classify_compression_ratio(fp16, awq, max_ratio);
    let (passed, desc) = match outcome {
        CompressionOutcome::Compressed { ratio } => (
            true,
            format!("ratio={ratio:.4} (max={max_ratio}, fp16={fp16}, awq={awq})"),
        ),
        CompressionOutcome::Insufficient { ratio, max_ratio } => (
            false,
            format!("ratio={ratio:.4} > max={max_ratio} (fp16={fp16}, awq={awq})"),
        ),
    };
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-08-003 compression gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "compression",
            falsify_id: "FALSIFY-CRUX-B-08-003",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_flags_gate(v: &Value) -> (GateReport, Option<String>) {
    let argv_owned: Vec<String> = v
        .get("argv")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let argv: Vec<&str> = argv_owned.iter().map(|s| s.as_str()).collect();
    let flags = parse_awq_flags(&argv);
    let validation = validate_awq_flags(&flags);

    let expected = v
        .get("expected_outcome")
        .and_then(|x| x.as_str())
        .unwrap_or("ok");
    let got = match &validation {
        AwqFlagValidation::Ok { .. } => "ok",
        AwqFlagValidation::MissingMethod => "missing_method",
        AwqFlagValidation::UnknownMethod { .. } => "unknown_method",
        AwqFlagValidation::InvalidBits { .. } => "invalid_bits",
        AwqFlagValidation::InvalidGroupSize { .. } => "invalid_group_size",
    };
    let passed = got == expected;
    let desc = format!("expected={expected} got={got} ({validation:?})");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-08-002 flags gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "flags",
            falsify_id: "FALSIFY-CRUX-B-08-002",
            outcome: desc,
            passed,
        },
        err,
    )
}
