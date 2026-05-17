//! CRUX-D-11 — end-to-end falsification harness for `apr ddp-metrics-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-D-11-{001,002,003}
//! gate the classifier discharges has a matching captured JSON pair that
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
        .prefix("crux-d-11-")
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

fn good_t1() -> serde_json::Value {
    json!({
        "tokens_per_sec": 1000.0,
        "final_loss": 2.5,
        "ddp_metrics": {"allreduce_bandwidth_gbps": [120.0, 118.0, 121.0]}
    })
}

fn good_t4() -> serde_json::Value {
    // T_4 = 3500 → eff = 3500/4000 = 0.875 ≥ 0.85; loss within 0.4%.
    json!({
        "tokens_per_sec": 3500.0,
        "final_loss": 2.51,
        "ddp_metrics": {"allreduce_bandwidth_gbps": [80.0, 82.0, 79.0]}
    })
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_d_11_cli_help_advertises_flags() {
    let out = apr_binary()
        .args(["ddp-metrics-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--metrics-1gpu-file",
        "--metrics-ngpu-file",
        "--world-size",
        "--scaling-floor",
        "--loss-tolerance",
    ] {
        assert!(
            stdout.contains(flag),
            "--help must advertise {flag}; got:\n{stdout}"
        );
    }
}

#[test]
fn falsify_crux_d_11_cli_missing_file_fails() {
    let out = apr_binary()
        .args([
            "ddp-metrics-lint",
            "--metrics-1gpu-file",
            "/nonexistent/crux-d-11-missing.json",
            "--metrics-ngpu-file",
            "/nonexistent/other.json",
            "--world-size",
            "4",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_d_11_001_scaling_ok_at_0_875() {
    let f1 = write_json(&good_t1());
    let fn_ = write_json(&good_t4());
    let out = apr_binary()
        .args(["ddp-metrics-lint", "--metrics-1gpu-file"])
        .arg(f1.path())
        .arg("--metrics-ngpu-file")
        .arg(fn_.path())
        .args(["--world-size", "4"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "0.875 efficiency must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_d_11_001_scaling_rejects_below_floor() {
    let f1 = write_json(&good_t1());
    let bad = json!({
        "tokens_per_sec": 2000.0,
        "final_loss": 2.51,
        "ddp_metrics": {"allreduce_bandwidth_gbps": [40.0]}
    });
    let fn_ = write_json(&bad);
    let out = apr_binary()
        .args(["ddp-metrics-lint", "--metrics-1gpu-file"])
        .arg(f1.path())
        .arg("--metrics-ngpu-file")
        .arg(fn_.path())
        .args(["--world-size", "4"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "0.5 efficiency must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("BelowThreshold"),
        "stderr must name BelowThreshold; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_d_11_002_loss_parity_ok_within_tolerance() {
    let f1 = write_json(&good_t1());
    let fn_ = write_json(&good_t4());
    let out = apr_binary()
        .args(["ddp-metrics-lint", "--metrics-1gpu-file"])
        .arg(f1.path())
        .arg("--metrics-ngpu-file")
        .arg(fn_.path())
        .args(["--world-size", "4"])
        .output()
        .expect("run");
    assert!(out.status.success(), "loss within 1% must pass");
}

#[test]
fn falsify_crux_d_11_002_loss_parity_rejects_divergence() {
    let f1 = write_json(&good_t1());
    let bad = json!({
        "tokens_per_sec": 3500.0,
        "final_loss": 5.0,
        "ddp_metrics": {"allreduce_bandwidth_gbps": [80.0]}
    });
    let fn_ = write_json(&bad);
    let out = apr_binary()
        .args(["ddp-metrics-lint", "--metrics-1gpu-file"])
        .arg(f1.path())
        .arg("--metrics-ngpu-file")
        .arg(fn_.path())
        .args(["--world-size", "4"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "loss divergence must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Divergence"),
        "stderr must name Divergence; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_d_11_003_allreduce_rejects_missing_array() {
    let f1 = write_json(&good_t1());
    let bad = json!({
        "tokens_per_sec": 3500.0,
        "final_loss": 2.51
        // no ddp_metrics
    });
    let fn_ = write_json(&bad);
    let out = apr_binary()
        .args(["ddp-metrics-lint", "--metrics-1gpu-file"])
        .arg(f1.path())
        .arg("--metrics-ngpu-file")
        .arg(fn_.path())
        .args(["--world-size", "4"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing ddp_metrics must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingDdpMetrics"),
        "stderr must name MissingDdpMetrics; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_d_11_003_allreduce_rejects_zero_step() {
    let f1 = write_json(&good_t1());
    let bad = json!({
        "tokens_per_sec": 3500.0,
        "final_loss": 2.51,
        "ddp_metrics": {"allreduce_bandwidth_gbps": [120.0, 0.0, 121.0]}
    });
    let fn_ = write_json(&bad);
    let out = apr_binary()
        .args(["ddp-metrics-lint", "--metrics-1gpu-file"])
        .arg(f1.path())
        .arg("--metrics-ngpu-file")
        .arg(fn_.path())
        .args(["--world-size", "4"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "zero bandwidth must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("NonPositiveBandwidth"),
        "stderr must name NonPositiveBandwidth; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_d_11_scaling_floor_configurable() {
    // T_4/4*T_1 = 0.6; default floor 0.85 fails, --scaling-floor 0.5 passes.
    let f1 = write_json(&good_t1());
    let body = json!({
        "tokens_per_sec": 2400.0,
        "final_loss": 2.51,
        "ddp_metrics": {"allreduce_bandwidth_gbps": [50.0]}
    });
    let fn_ = write_json(&body);
    let out = apr_binary()
        .args(["ddp-metrics-lint", "--metrics-1gpu-file"])
        .arg(f1.path())
        .arg("--metrics-ngpu-file")
        .arg(fn_.path())
        .args(["--world-size", "4", "--scaling-floor", "0.5"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "0.6 efficiency with floor=0.5 must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_d_11_json_output_contains_outcomes() {
    let f1 = write_json(&good_t1());
    let fn_ = write_json(&good_t4());
    let out = apr_binary()
        .args(["--json", "ddp-metrics-lint", "--metrics-1gpu-file"])
        .arg(f1.path())
        .arg("--metrics-ngpu-file")
        .arg(fn_.path())
        .args(["--world-size", "4"])
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good bodies must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["scaling_efficiency"]
        .as_str()
        .expect("scaling")
        .contains("Ok"));
    assert!(parsed["loss_parity"]
        .as_str()
        .expect("loss_parity")
        .contains("Ok"));
    assert!(parsed["allreduce_bandwidth"]
        .as_str()
        .expect("allreduce")
        .contains("Ok"));
}
