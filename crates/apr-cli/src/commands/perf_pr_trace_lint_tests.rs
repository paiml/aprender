//! OBS-11 case table: every planted defect is RED and names itself; the valid
//! document and the benign variations stay GREEN.

use super::{check, run, SCHEMA};
use crate::error::CliError;
use serde_json::{json, Value};

fn identity(backend: &str) -> Value {
    json!({
        "schema": "apr-trace-v1",
        "ts": "2026-09-27T14:00:00Z",
        "host": "lambda",
        "apr_version": "0.71.0-dev",
        "apr_tag": "v0.71.0-rc.1",
        "crate_tarball_sha256": "a".repeat(64),
        "binary_sha256": "b".repeat(64),
        "build_identity": {
            "rustc_vv": "rustc 1.93.0",
            "target_triple": "x86_64-unknown-linux-gnu",
            "features": ["cuda", "inference"],
            "uname_a": "Linux lambda 6.8.0-90-generic",
            "accelerator": "RTX 4090",
            "driver": "570.86"
        },
        "model_id": "qwen2.5-coder-1.5b-q4k",
        "model_sha256": "c".repeat(64),
        "backend": backend,
        "gpu_proof": if backend == "cpu" { Value::Null } else { json!("trace: KernelLaunch q4k_gemv sm_89") },
        "request_id": "01927f3e-0000-7000-8000-000000000000"
    })
}

fn trace(attn: f64, ffn: f64) -> Value {
    json!({
        "level": "layer",
        "operations": 2,
        "total_time_us": attn + ffn + 10.0,
        "breakdown": [{"name": "attention", "time_us": attn}, {"name": "ffn", "time_us": ffn}],
        "provenance": "measured"
    })
}

fn valid() -> Value {
    json!({
        "schema": SCHEMA,
        "pr": 4498,
        "claimed": ["attention"],
        "before": {"commit": "aca6f2d7f6", "identity": identity("cuda"), "trace": trace(100.0, 50.0)},
        "after": {"commit": "d617620271", "identity": identity("cuda"), "trace": trace(80.0, 50.0)}
    })
}

fn red(mutate: impl Fn(&mut Value), expect: &str) {
    let mut d = valid();
    mutate(&mut d);
    let r = check(&d, None, None);
    assert!(
        r.findings.iter().any(|f| f.contains(expect)),
        "must be RED with `{expect}`, got {:?}",
        r.findings
    );
}

#[test]
fn falsify_obs11_valid_evidence_is_green_with_the_delta() {
    let r = check(&valid(), Some("aca6f2d7f6"), Some("d617620271"));
    assert!(r.findings.is_empty(), "{:?}", r.findings);
    assert_eq!(r.layers.len(), 1);
    assert_eq!(r.layers[0].delta_pct(), Some(-20.0));
}

#[test]
fn falsify_obs11_benign_variations_stay_green() {
    // A kernel update between runs, and a different binary/tarball (the PR changed
    // the code): still the same host and identity.
    let mut d = valid();
    d["after"]["identity"]["build_identity"]["uname_a"] = json!("Linux lambda 6.8.0-91-generic");
    d["after"]["identity"]["binary_sha256"] = json!("d".repeat(64));
    d["after"]["identity"]["crate_tarball_sha256"] = json!("e".repeat(64));
    // Full shas against short --base/--head.
    d["before"]["commit"] = json!("aca6f2d7f6".to_string() + &"0".repeat(30));
    assert!(check(&d, Some("aca6f2d7f6"), Some("d617620271"))
        .findings
        .is_empty());
    // A cpu pair with null proof.
    let mut c = valid();
    c["before"]["identity"] = identity("cpu");
    c["after"]["identity"] = identity("cpu");
    assert!(check(&c, None, None).findings.is_empty());
}

#[test]
fn falsify_obs11_case_table_every_planted_defect_is_red() {
    red(|d| d["schema"] = json!("apr-trace-v1"), "schema:");
    red(|d| d["claimed"] = json!([]), "claimed: absent or empty");
    red(
        |d| drop(d.as_object_mut().unwrap().remove("claimed")),
        "claimed:",
    );
    red(
        |d| {
            drop(
                d["after"]["identity"]
                    .as_object_mut()
                    .unwrap()
                    .remove("request_id"),
            )
        },
        "after.identity.request_id: absent",
    );
    red(
        |d| {
            drop(
                d["before"]["identity"]
                    .as_object_mut()
                    .unwrap()
                    .remove("gpu_proof"),
            )
        },
        "before.identity.gpu_proof: absent",
    );
    red(
        |d| {
            drop(
                d["before"]["identity"]["build_identity"]
                    .as_object_mut()
                    .unwrap()
                    .remove("driver"),
            )
        },
        "before.identity.build_identity.driver: absent",
    );
    red(
        |d| d["after"]["identity"]["host"] = json!("yoga"),
        "identity.host: before",
    );
    red(
        |d| d["after"]["identity"]["model_sha256"] = json!("f".repeat(64)),
        "identity.model_sha256",
    );
    red(
        |d| d["after"]["identity"]["backend"] = json!("wgpu"),
        "identity.backend: before",
    );
    red(
        |d| d["after"]["identity"]["build_identity"]["features"] = json!(["inference"]),
        "identity.build_identity.features",
    );
    red(
        |d| d["after"]["identity"]["build_identity"]["accelerator"] = json!("RTX 3090"),
        "identity.build_identity.accelerator",
    );
    red(
        |d| d["before"]["identity"]["gpu_proof"] = Value::Null,
        "gpu_proof: null on backend cuda",
    );
    red(
        |d| {
            d["before"]["identity"]["backend"] = json!("backend_unproven");
            d["after"]["identity"]["backend"] = json!("backend_unproven");
        },
        "backend_unproven",
    );
    red(
        |d| d["before"]["identity"]["schema"] = json!("apr-lane-row-v1"),
        "is not apr-trace-v1",
    );
    red(
        |d| d["before"]["identity"]["model_sha256"] = json!("ABC"),
        "not a lowercase sha256",
    );
    red(
        |d| d["after"]["trace"]["provenance"] = json!("wall_clock_total"),
        "wall_clock_total",
    );
    red(
        |d| d["before"]["trace"]["provenance"] = json!("estimated"),
        "estimated",
    );
    red(
        |d| d["after"]["trace"]["total_time_us"] = json!(1.0),
        "past total_time_us",
    );
    red(
        |d| d["after"]["trace"]["breakdown"] = json!([]),
        "after.trace.breakdown: absent or empty",
    );
    red(
        |d| d["claimed"] = json!(["lm_head"]),
        "claimed layer lm_head: not in the before",
    );
    red(
        |d| d["after"]["commit"] = json!("aca6f2d7f6"),
        "before.commit == after.commit",
    );
    red(
        |d| d["after"]["commit"] = json!("not-a-sha"),
        "after.commit: absent",
    );
    red(
        |d| drop(d.as_object_mut().unwrap().remove("after")),
        "after.identity: missing",
    );
}

#[test]
fn falsify_obs11_commits_must_be_the_prs_base_and_head() {
    let r = check(&valid(), Some("1111111111"), Some("d617620271"));
    assert!(
        r.findings.iter().any(|f| f.contains("before.commit")),
        "{:?}",
        r.findings
    );
    let r = check(&valid(), Some("aca6f2d7f6"), Some("2222222222"));
    assert!(
        r.findings.iter().any(|f| f.contains("after.commit")),
        "{:?}",
        r.findings
    );
}

#[test]
fn obs11_exit_codes_follow_the_lint_family() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("nope.json");
    assert!(matches!(
        run(&missing, None, None, false),
        Err(CliError::FileNotFound(_))
    ));
    let bad = dir.path().join("bad.json");
    std::fs::write(&bad, "{not json").unwrap();
    assert!(matches!(
        run(&bad, None, None, false),
        Err(CliError::InvalidInput(_))
    ));
    assert!(matches!(
        run(dir.path(), None, None, false),
        Err(CliError::InvalidInput(_))
    ));
    let good = dir.path().join("good.json");
    std::fs::write(&good, valid().to_string()).unwrap();
    assert!(run(&good, None, None, true).is_ok());
    let mut d = valid();
    d["after"]["identity"]["host"] = json!("yoga");
    let rej = dir.path().join("rej.json");
    std::fs::write(&rej, d.to_string()).unwrap();
    assert!(matches!(
        run(&rej, None, None, false),
        Err(CliError::ValidationFailed(_))
    ));
}
