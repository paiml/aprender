//! CRUX-F-15 — end-to-end falsification harness for `apr nccl-diag-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-F-15-{001,002,003}
//! gate the classifier discharges has a matching captured JSON body that
//! the binary must classify exactly as the harness expects.

use serde_json::json;
use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_json(body: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-15-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(
        serde_json::to_vec_pretty(body)
            .expect("serialize")
            .as_slice(),
    )
    .expect("write");
    f.flush().expect("flush");
    f
}

fn good_body() -> serde_json::Value {
    json!({
        "host": "node-0",
        "rank": 0,
        "peer_rank": 1,
        "nccl_version": "2.20.5",
        "cuda_devices": "0,1",
        "fabric": "ib",
        "last_op": "AllReduce",
        "code": 6,
        "suggest": "See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html"
    })
}

// ===== g2: CLI shape =====

/// A `*-lint --json` outcome field says OK — in whichever of the THREE shapes that command
/// emits, because the surface is not uniform and this test is not the place to decide it.
///
/// #3051. The assertion here was `parsed[field].as_str().expect(..).contains("Ok")`, and it
/// panicked in the `expect` on clean `main` — invisibly, because this target is not on
/// ci.yml's `--test` line and had never run in CI until BSE-17's quick tier selected it.
///
/// MEASURED on main, one outcome field per row, all from `--json`:
///
/// | shape | example | seen in |
/// |---|---|---|
/// | structured object | `{"status": "ok", "efficiency": 0.875}` | `ddp-metrics-lint` |
/// | bare string | `"Ok"` | `prometheus-lint` `required_metrics` |
/// | stringified Rust `Debug` | `"Ok { code: 134 }"` | `nccl-diag-lint` `exit_code` |
///
/// `prometheus-lint` emits two of the three within ONE response. Shape 3 is a `Debug` string
/// leaking into a JSON API. That is a real finding about the `--json` surface, filed
/// separately; normalising it inside a test would hide it, and picking one shape here would
/// assert a design decision the tree has not made.
///
/// `.contains("Ok")` was also a pass-grep in Rust clothing — `"NotOk"` contains `"Ok"`. The
/// string arms therefore match at a token boundary and the object arm reads `status`, so an
/// outcome that is not ok can no longer satisfy this assertion.
fn assert_outcome_ok(parsed: &serde_json::Value, field: &str) {
    let v = &parsed[field];
    if let Some(s) = v.as_str() {
        let ok = s == "Ok" || s == "ok" || s.starts_with("Ok {") || s.starts_with("Ok(");
        assert!(ok, "{field} outcome is not ok: {s:?}");
        return;
    }
    assert_eq!(
        v["status"].as_str(),
        Some("ok"),
        "{field} outcome is neither an Ok string nor an object with status ok: {v}"
    );
}

#[test]
fn falsify_crux_f_15_cli_help_advertises_flags() {
    let out = apr_binary()
        .args(["nccl-diag-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in ["--diag-file", "--exit-code", "--require-doc-link"] {
        assert!(
            stdout.contains(flag),
            "--help must advertise {flag}; got:\n{stdout}"
        );
    }
}

#[test]
fn falsify_crux_f_15_cli_missing_file_fails() {
    let out = apr_binary()
        .args([
            "nccl-diag-lint",
            "--diag-file",
            "/nonexistent/crux-f-15-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

#[test]
fn falsify_crux_f_15_cli_malformed_json_fails() {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-15-bad-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(b"{ not json").expect("write");
    f.flush().expect("flush");
    let out = apr_binary()
        .args(["nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "malformed JSON must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_f_15_001_schema_ok_on_well_formed() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good diag must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_15_001_schema_rejects_missing_key() {
    let mut body = good_body();
    body.as_object_mut().expect("obj").remove("suggest");
    let f = write_json(&body);
    let out = apr_binary()
        .args(["nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing suggest must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingKey") && stderr.contains("suggest"),
        "stderr must name MissingKey(suggest); got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_15_001_schema_rejects_negative_rank() {
    let mut body = good_body();
    body["rank"] = json!(-5);
    let f = write_json(&body);
    let out = apr_binary()
        .args(["nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "negative rank must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("RankNotNonNegativeInt"),
        "stderr must name RankNotNonNegativeInt; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_15_002_exit_code_ge_128_passes() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .args(["--exit-code", "134"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "exit 134 must pass (>= 128); stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_15_002_exit_code_1_fails() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .args(["--exit-code", "1"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "exit 1 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("BelowThreshold"),
        "stderr must name BelowThreshold; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_15_003_doc_link_ok_on_nvidia_url() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .arg("--require-doc-link")
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "nvidia.com URL in suggest must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_15_003_doc_link_rejects_free_text() {
    let mut body = good_body();
    body["suggest"] = json!("Try restarting and check logs");
    let f = write_json(&body);
    let out = apr_binary()
        .args(["nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .arg("--require-doc-link")
        .output()
        .expect("run");
    assert!(!out.status.success(), "free-text suggest must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("NoDocLink"),
        "stderr must name NoDocLink; got:\n{stderr}"
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_f_15_json_output_contains_outcomes() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["--json", "nccl-diag-lint", "--diag-file"])
        .arg(f.path())
        .args(["--exit-code", "134"])
        .arg("--require-doc-link")
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good body must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert_outcome_ok(&parsed, "schema");
    assert_outcome_ok(&parsed, "doc_link");
    assert_outcome_ok(&parsed, "exit_code");
}
