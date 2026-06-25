//! CRUX-A-25 — `apr rm-gc-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the pure `blob_gc` classifier over a captured JSON
//! observation file covering the three FALSIFY gates:
//!
//! ```jsonc
//! {
//!   "rm": {
//!     "manifests":      [{"tag": "gpt2:latest", "blobs": ["sha1", "sha2"]}],
//!     "tag_to_rm":      "gpt2:latest",
//!     "all_blobs":      ["sha1", "sha2"],
//!     "expected_freed": ["sha1", "sha2"]
//!   },
//!   "safety": {
//!     "manifests":      [
//!       {"tag": "gpt2:latest", "blobs": ["sha1"]},
//!       {"tag": "gpt2:dup",    "blobs": ["sha1"]}
//!     ],
//!     "tag_to_rm":      "gpt2:latest",
//!     "all_blobs":      ["sha1"],
//!     "expected_freed": []
//!   },
//!   "dryrun": {
//!     "manifests":           [{"tag": "x", "blobs": []}],
//!     "all_blobs":           ["sha-orphan"],
//!     "expected_idempotent": true
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-A-25
//! stderr stamp on any failing gate.

use crate::commands::blob_gc::{
    apply_plan, apply_rm, compute_refcounts, plan_gc, GcPlan, Manifest,
};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct RmGcLintArgs {
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

pub fn run(args: RmGcLintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-A-25: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-A-25: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-A-25: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-A-25: observation is not valid JSON: {e}"))?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(v) = obs.get("rm") {
        let (r, err) = run_rm_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("safety") {
        let (r, err) = run_safety_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("dryrun") {
        let (r, err) = run_dryrun_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err("FALSIFY-CRUX-A-25: observation has none of rm/safety/dryrun".into());
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-A-25",
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

fn parse_manifests(v: Option<&Value>) -> Result<Vec<Manifest>, String> {
    let arr = v
        .and_then(|x| x.as_array())
        .ok_or_else(|| "manifests must be a JSON array".to_string())?;
    arr.iter()
        .map(|m| {
            let tag = m
                .get("tag")
                .and_then(|x| x.as_str())
                .ok_or_else(|| "manifest.tag must be a string".to_string())?;
            let blobs: Vec<String> = m
                .get("blobs")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            Ok(Manifest::new(tag, blobs))
        })
        .collect()
}

fn parse_blobs(v: Option<&Value>) -> BTreeSet<String> {
    v.and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn candidate_set(plan: &GcPlan) -> BTreeSet<String> {
    plan.candidates.iter().map(|c| c.sha256.clone()).collect()
}

fn run_rm_gate(v: &Value) -> (GateReport, Option<String>) {
    let manifests = match parse_manifests(v.get("manifests")) {
        Ok(m) => m,
        Err(e) => {
            let desc = format!("parse error: {e}");
            return (
                GateReport {
                    gate: "rm",
                    falsify_id: "FALSIFY-CRUX-A-25-001",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!("FALSIFY-CRUX-A-25-001 rm gate failed: {desc}")),
            );
        }
    };
    let tag = v.get("tag_to_rm").and_then(|x| x.as_str()).unwrap_or("");
    let all_blobs = parse_blobs(v.get("all_blobs"));
    let expected: BTreeSet<String> = parse_blobs(v.get("expected_freed"));

    let reduced = apply_rm(&manifests, tag);
    let rc = compute_refcounts(&reduced);
    let plan = plan_gc(&all_blobs, &rc);
    let got = candidate_set(&plan);
    let passed = got == expected;
    let desc = format!("freed={got:?} expected={expected:?}");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-25-001 rm gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "rm",
            falsify_id: "FALSIFY-CRUX-A-25-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_safety_gate(v: &Value) -> (GateReport, Option<String>) {
    let manifests = match parse_manifests(v.get("manifests")) {
        Ok(m) => m,
        Err(e) => {
            let desc = format!("parse error: {e}");
            return (
                GateReport {
                    gate: "safety",
                    falsify_id: "FALSIFY-CRUX-A-25-002",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!("FALSIFY-CRUX-A-25-002 safety gate failed: {desc}")),
            );
        }
    };
    let tag = v.get("tag_to_rm").and_then(|x| x.as_str()).unwrap_or("");
    let all_blobs = parse_blobs(v.get("all_blobs"));
    let expected: BTreeSet<String> = parse_blobs(v.get("expected_freed"));

    let reduced = apply_rm(&manifests, tag);
    let rc = compute_refcounts(&reduced);
    let plan = plan_gc(&all_blobs, &rc);
    let got = candidate_set(&plan);

    // Additional safety invariant: no surviving-manifest blob may appear
    // in the candidate list, regardless of what the observer expected.
    let surviving_blobs: BTreeSet<String> = reduced
        .iter()
        .flat_map(|m| m.blobs.iter().cloned())
        .collect();
    let violated: BTreeSet<&String> = got.intersection(&surviving_blobs).collect();

    let passed = got == expected && violated.is_empty();
    let desc = if violated.is_empty() {
        format!("freed={got:?} expected={expected:?}")
    } else {
        format!("SAFETY VIOLATION: plan candidates {violated:?} are still referenced")
    };
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-25-002 safety gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "safety",
            falsify_id: "FALSIFY-CRUX-A-25-002",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_dryrun_gate(v: &Value) -> (GateReport, Option<String>) {
    let manifests = match parse_manifests(v.get("manifests")) {
        Ok(m) => m,
        Err(e) => {
            let desc = format!("parse error: {e}");
            return (
                GateReport {
                    gate: "dryrun",
                    falsify_id: "FALSIFY-CRUX-A-25-003",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!("FALSIFY-CRUX-A-25-003 dryrun gate failed: {desc}")),
            );
        }
    };
    let all_blobs = parse_blobs(v.get("all_blobs"));
    let expected_idempotent = v
        .get("expected_idempotent")
        .and_then(|x| x.as_bool())
        .unwrap_or(true);

    let rc = compute_refcounts(&manifests);
    let plan1 = plan_gc(&all_blobs, &rc);
    let plan2 = plan_gc(&all_blobs, &rc); // "dry-run" == same pure call
    let post = apply_plan(&all_blobs, &plan1);
    let rc_post = compute_refcounts(&manifests); // manifests unchanged
    let plan3 = plan_gc(&post, &rc_post); // second gc over post-state

    let dryrun_eq = plan1 == plan2;
    let idempotent = plan3.is_noop();
    let matches_expected = idempotent == expected_idempotent;

    let passed = dryrun_eq && matches_expected;
    let desc = format!(
        "dryrun_eq={dryrun_eq} idempotent={idempotent} expected_idempotent={expected_idempotent} first_plan={} candidates",
        plan1.candidates.len()
    );
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-25-003 dryrun gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "dryrun",
            falsify_id: "FALSIFY-CRUX-A-25-003",
            outcome: desc,
            passed,
        },
        err,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_obs(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    fn args_for(f: &NamedTempFile) -> RmGcLintArgs {
        RmGcLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = RmGcLintArgs {
            observation_file: "/no/such/gc.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-A-25"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("\t\n ");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("{nope");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("none of rm/safety/dryrun"));
    }

    #[test]
    fn rm_gate_frees_unreferenced_blobs() {
        let f = write_obs(
            r#"{"rm": {"manifests": [{"tag": "gpt2:latest", "blobs": ["sha1", "sha2"]}],
                     "tag_to_rm": "gpt2:latest",
                     "all_blobs": ["sha1", "sha2"],
                     "expected_freed": ["sha1", "sha2"]}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn rm_gate_mismatch_fails() {
        let f = write_obs(
            r#"{"rm": {"manifests": [{"tag": "gpt2:latest", "blobs": ["sha1", "sha2"]}],
                     "tag_to_rm": "gpt2:latest",
                     "all_blobs": ["sha1", "sha2"],
                     "expected_freed": []}}"#,
        );
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-A-25-001"));
    }

    #[test]
    fn rm_gate_bad_manifests_is_parse_error() {
        let f = write_obs(r#"{"rm": {"manifests": "not-an-array"}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-A-25-001"));
        assert!(err.contains("parse error"));
    }

    #[test]
    fn safety_gate_protects_still_referenced_blob() {
        // gpt2:dup still references sha1 after removing gpt2:latest, so nothing is freed.
        let f = write_obs(
            r#"{"safety": {"manifests": [
                     {"tag": "gpt2:latest", "blobs": ["sha1"]},
                     {"tag": "gpt2:dup", "blobs": ["sha1"]}],
                     "tag_to_rm": "gpt2:latest",
                     "all_blobs": ["sha1"],
                     "expected_freed": []}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn dryrun_gate_idempotent_passes() {
        let f = write_obs(
            r#"{"dryrun": {"manifests": [{"tag": "x", "blobs": []}],
                     "all_blobs": ["sha-orphan"],
                     "expected_idempotent": true}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn json_mode_ok() {
        let f = write_obs(
            r#"{"rm": {"manifests": [{"tag": "a:1", "blobs": ["b1"]}],
                     "tag_to_rm": "a:1",
                     "all_blobs": ["b1"],
                     "expected_freed": ["b1"]}}"#,
        );
        let args = RmGcLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
