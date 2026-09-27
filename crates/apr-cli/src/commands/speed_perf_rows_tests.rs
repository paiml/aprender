//! EXT-19 over APR-OBS `apr-perf-ledger-v1` rows (aprender#4551): identity
//! admission, backend proof, model-change reset, empty ledger, planted
//! slowdown, and T28 on the printed report.

use super::*;
use serde_json::{json, Value};

const H: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const M2: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
const PIN: &str = "d1d3c3396";

fn side(tg: f64) -> Value {
    json!({"load_ms": 900.0, "ttft_ms": 40.0, "pp512_tok_s": 1000.0,
           "tg128_tok_s": tg, "wall_ms": 5000.0, "median": tg, "mad": 0.5})
}

/// A full admissible row; `apr` tok/s against a 100 tok/s llama.cpp arm.
fn base(tag: &str, host: &str, backend: &str, apr: f64) -> Value {
    let mut llama = side(100.0);
    llama["build_commit"] = json!(PIN);
    json!({
        "schema": "apr-perf-ledger-v1", "ts": format!("2026-09-27T00:00:00Z-{tag}"),
        "host": host, "apr_version": "0.71.0", "apr_tag": tag,
        "crate_tarball_sha256": H, "binary_sha256": H,
        "build_identity": "0.71.0+1fc469b514", "model_id": "qwen3.5-4b-q4k",
        "model_sha256": H, "backend": backend, "request_id": format!("r-{tag}-{host}"),
        "gpu_proof": if backend == "cpu" { Value::Null } else { json!({"nvidia_smi_pid": 4242}) },
        "workload_id": "decode-tg128", "prompt_n": 8, "reps": 5,
        "apr": side(apr), "llama": llama
    })
}

fn tags(n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("v0.71.{i}")).collect()
}

fn cells() -> Vec<String> {
    vec!["gx10/cuda".into(), "lambda/cpu".into()]
}

fn ledger(apr: &[f64]) -> String {
    tags(apr.len())
        .iter()
        .zip(apr)
        .flat_map(|(t, a)| [base(t, "gx10", "cuda", *a), base(t, "lambda", "cpu", 50.0)])
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_full_admissible_ledger_passes() {
    let r = perf_gate(&ledger(&[90.0, 91.0, 90.0, 89.0]), PIN, &tags(4), &cells()).expect("gate");
    assert!(r.rejected.is_empty(), "{:?}", r.rejected);
    assert!(r.passed(), "{}", r.render());
}

/// FALSIFY-OBS-PERF: each identity field, missing, null or "unknown", makes
/// the row inadmissible, and the pair it would have filled is a hole.
#[test]
fn every_identity_field_is_required() {
    for k in IDENTITY {
        for bad in [None, Some(Value::Null), Some(json!("unknown"))] {
            let mut v = base("v0.71.1", "gx10", "cuda", 90.0);
            match bad {
                None => {
                    v.as_object_mut().expect("obj").remove(k);
                }
                Some(b) => v[k] = b,
            }
            let err = admit(&v, PIN).expect_err(k);
            assert!(err.contains(k), "{k}: {err}");
            let l = format!("{v}\n{}", base("v0.71.1", "lambda", "cpu", 50.0));
            let r = perf_gate(&l, PIN, &tags(1), &cells()).expect("gate");
            assert!(!r.passed(), "{k} missing still passed:\n{}", r.render());
            assert!(r.render().contains("ABSENT   line 1"), "{}", r.render());
        }
    }
}

/// A cuda row with null gpu_proof is backend_unproven and fills nothing; a
/// cpu row with null gpu_proof is fine; a row without the key is refused.
#[test]
fn a_gpu_row_without_proof_is_absent() {
    let mut v = base("v0.71.1", "gx10", "cuda", 90.0);
    v["gpu_proof"] = Value::Null;
    assert!(admit(&v, PIN)
        .expect_err("cuda")
        .contains("backend_unproven"));
    assert!(admit(&base("v0.71.1", "lambda", "cpu", 50.0), PIN).is_ok());
    let mut v = base("v0.71.1", "lambda", "cpu", 50.0);
    v.as_object_mut().expect("obj").remove("gpu_proof");
    assert!(admit(&v, PIN).expect_err("key").contains("gpu_proof"));
}

#[test]
fn workload_floor_and_llama_pin_are_enforced() {
    let mut v = base("v0.71.1", "gx10", "cuda", 90.0);
    v["prompt_n"] = json!(7);
    assert!(admit(&v, PIN).is_err());
    let mut v = base("v0.71.1", "gx10", "cuda", 90.0);
    v["reps"] = json!(3);
    assert!(admit(&v, PIN).is_err());
    assert!(admit(&base("v0.71.1", "gx10", "cuda", 90.0), "other").is_err());
}

/// The cell list is declared, never read off the rows: a host that stopped
/// writing is a hole (FALSIFY-OBS-PERF-001).
#[test]
fn a_silent_declared_host_is_a_hole() {
    let l = ledger(&[90.0, 90.0])
        .lines()
        .filter(|l| !l.contains("\"lambda\""))
        .collect::<Vec<_>>()
        .join("\n");
    let r = perf_gate(&l, PIN, &tags(2), &cells()).expect("gate");
    assert_eq!(r.gate.holes.len(), 2);
    assert!(!r.passed());
}

#[test]
fn a_planted_slowdown_fails_the_gate() {
    let r = perf_gate(&ledger(&[90.0, 91.0, 90.0, 72.0]), PIN, &tags(4), &cells()).expect("gate");
    assert!(!r.passed(), "{}", r.render());
    assert!(
        r.render().contains("RED      gx10/cuda @ v0.71.4"),
        "{}",
        r.render()
    );
}

/// A model change resets the baseline: a slower NEW model is not judged
/// against the old model's floor, so it is unarmed, not RED.
#[test]
fn a_model_change_resets_the_baseline() {
    let mut rows: Vec<Value> = ledger(&[90.0, 91.0, 90.0, 72.0])
        .lines()
        .map(|l| serde_json::from_str(l).expect("json"))
        .collect();
    rows[6]["model_sha256"] = json!(M2);
    let l = rows
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let r = perf_gate(&l, PIN, &tags(4), &cells()).expect("gate");
    assert!(r.passed(), "{}", r.render());
    assert!(
        r.render().contains("UNARMED  gx10/cuda: 1 measured"),
        "{}",
        r.render()
    );
}

/// A backfill row fills its pair but never lifts the floor.
#[test]
fn a_backfill_row_is_never_a_baseline() {
    let mut rows: Vec<Value> = ledger(&[90.0, 90.0, 90.0, 200.0, 200.0, 90.0])
        .lines()
        .map(|l| serde_json::from_str(l).expect("json"))
        .collect();
    rows[6]["provenance"] = json!("backfill");
    rows[8]["provenance"] = json!("backfill");
    let l = rows
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let r = perf_gate(&l, PIN, &tags(6), &cells()).expect("gate");
    assert!(r.gate.holes.is_empty());
    assert!(r.passed(), "backfill lifted the floor:\n{}", r.render());
}

/// Two rows for one pair: the newest `ts` is the one judged.
#[test]
fn the_newest_row_for_a_pair_wins() {
    let mut l = ledger(&[90.0, 91.0, 90.0, 72.0]);
    let mut redo = base("v0.71.4", "gx10", "cuda", 90.0);
    redo["ts"] = json!("2026-09-28T00:00:00Z");
    l.push('\n');
    l.push_str(&redo.to_string());
    let r = perf_gate(&l, PIN, &tags(4), &cells()).expect("gate");
    assert!(r.passed(), "{}", r.render());
}

#[test]
fn an_empty_or_malformed_ledger_is_an_error_not_a_verdict() {
    assert!(perf_gate("", PIN, &tags(1), &cells())
        .expect_err("empty")
        .contains("empty"));
    assert!(perf_gate("\n  \n", PIN, &tags(1), &cells()).is_err());
    assert!(perf_gate("{nope", PIN, &tags(1), &cells()).is_err());
    assert!(perf_gate(&ledger(&[90.0]), PIN, &[], &cells()).is_err());
    let long = format!("{}\n{}", "x".repeat(MAX_LINE_BYTES + 1), ledger(&[90.0]));
    let r = perf_gate(&long, PIN, &tags(1), &cells()).expect("gate");
    assert_eq!(r.rejected[0].0, 1);
}

#[test]
fn t28_the_report_never_prints_a_ratio_or_floor() {
    for apr in [[90.0, 91.0, 90.0, 72.0], [90.0, 91.0, 90.0, 89.0]] {
        let out = perf_gate(&ledger(&apr), PIN, &tags(4), &cells())
            .expect("gate")
            .render();
        for needle in ["0.9", "0.72", "0.89", "0.5", "ratio", "floor ="] {
            assert!(!out.contains(needle), "report leaks `{needle}`:\n{out}");
        }
    }
}

fn not_run(tag: &str, host: &str, backend: &str) -> Value {
    let mut v = base(tag, host, backend, 1.0);
    let o = v.as_object_mut().expect("obj");
    o.remove("apr");
    o.remove("llama");
    o.insert("gpu_proof".into(), Value::Null);
    o.insert("not_run".into(), json!({"reason": "runner offline"}));
    v
}

/// EXT-19 `NotRun{reason}` over #4551: a not_run row fills its pair, is not a
/// record, and needs a reason and the identity block; a GPU not_run row needs
/// no gpu_proof, since it claims nothing about a GPU.
#[test]
fn a_not_run_row_fills_its_pair_but_is_not_a_record() {
    let mut rows: Vec<Value> = ledger(&[90.0, 91.0, 90.0, 89.0])
        .lines()
        .map(|l| serde_json::from_str(l).expect("json"))
        .collect();
    rows[6] = not_run("v0.71.4", "gx10", "cuda");
    let l = rows
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let r = perf_gate(&l, PIN, &tags(4), &cells()).expect("gate");
    assert!(r.rejected.is_empty(), "{:?}", r.rejected);
    assert!(r.gate.holes.is_empty());
    assert!(r.passed(), "{}", r.render());
    assert!(
        r.render().contains("UNARMED  gx10/cuda: 3 measured"),
        "{}",
        r.render()
    );

    let mut v = not_run("v0.71.1", "gx10", "cuda");
    v["not_run"] = json!({"reason": " "});
    assert!(admit(&v, PIN).is_err());
    let mut v = not_run("v0.71.1", "gx10", "cuda");
    v["apr"] = side(90.0);
    assert!(admit(&v, PIN).is_err());
    let mut v = not_run("v0.71.1", "gx10", "cuda");
    v.as_object_mut().expect("obj").remove("model_sha256");
    assert!(admit(&v, PIN).is_err());
}
