//! CRUX-H-13 — end-to-end falsification harness for `apr audio-inspect-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-H-13-{001,002} gate
//! the classifier discharges has a matching captured JSON body that the
//! binary must classify exactly as the harness expects.

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
        .prefix("crux-h-13-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(serde_json::to_vec_pretty(body).expect("serialize").as_slice())
        .expect("write");
    f.flush().expect("flush");
    f
}

fn good_body() -> serde_json::Value {
    json!({"min": -0.85, "max": 0.92, "sample_rate": 16000, "channels": 2, "samples": 48000})
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_h_13_cli_help_advertises_flags() {
    let out = apr_binary().args(["audio-inspect-lint", "--help"]).output().expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in ["--json-file", "--expected-sample-rate", "--expected-channels"] {
        assert!(stdout.contains(flag), "--help must advertise {flag}; got:\n{stdout}");
    }
}

#[test]
fn falsify_crux_h_13_cli_missing_file_fails() {
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file", "/nonexistent/crux-h-13-missing.json"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_h_13_001_amplitude_ok_on_good_body() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good body must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_h_13_001_amplitude_rejects_below_floor() {
    let mut body = good_body();
    body["min"] = json!(-1.5);
    let f = write_json(&body);
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "min < -1.0 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MinBelowFloor"),
        "stderr must name MinBelowFloor; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_h_13_001_amplitude_rejects_above_ceiling() {
    let mut body = good_body();
    body["max"] = json!(1.5);
    let f = write_json(&body);
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "max > 1.0 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MaxAboveCeiling"),
        "stderr must name MaxAboveCeiling; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_h_13_002_sample_rate_ok_when_matches() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .args(["--expected-sample-rate", "16000"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "16000 matches expected must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_h_13_002_sample_rate_rejects_mismatch() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .args(["--expected-sample-rate", "22050"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "16000 ≠ 22050 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ExpectedRateMismatch"),
        "stderr must name ExpectedRateMismatch; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_h_13_002_sample_rate_rejects_non_canonical() {
    let mut body = good_body();
    body["sample_rate"] = json!(12345);
    let f = write_json(&body);
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "12345 must fail as non-canonical");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("NonCanonicalRate"),
        "stderr must name NonCanonicalRate; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_h_13_channel_shape_ok_when_matches_expected() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .args(["--expected-channels", "2"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "channels=2 matches expected must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_h_13_channel_shape_rejects_mismatch() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .args(["--expected-channels", "1"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "channels=2 vs --mono=1 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ExpectedChannelsMismatch"),
        "stderr must name ExpectedChannelsMismatch; got:\n{stderr}"
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_h_13_json_output_contains_outcomes() {
    let f = write_json(&good_body());
    let out = apr_binary()
        .args(["--json", "audio-inspect-lint", "--json-file"])
        .arg(f.path())
        .args(["--expected-sample-rate", "16000", "--expected-channels", "2"])
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good body must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["amplitude_bounds"].as_str().expect("bounds").contains("Ok"));
    assert!(parsed["sample_rate"].as_str().expect("rate").contains("Ok"));
    assert!(parsed["channel_shape"].as_str().expect("shape").contains("Ok"));
}
