//! CRUX-F-17 — end-to-end falsification harness for `apr attn-viz-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-F-17-{001,002,003} gate
//! the classifier discharges has a matching captured JSON/HTML body that the
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
        .prefix("crux-f-17-")
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

fn write_html(body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-17-")
        .suffix(".html")
        .tempfile()
        .expect("tempfile");
    f.write_all(body.as_bytes()).expect("write");
    f.flush().expect("flush");
    f
}

fn good_attn() -> serde_json::Value {
    // (L=2, H=2, S=3, S=3); rows sum to 1, j > i is zero.
    let row0 = json!([1.0, 0.0, 0.0]);
    let row1 = json!([0.4, 0.6, 0.0]);
    let row2 = json!([0.2, 0.3, 0.5]);
    let head = json!([row0.clone(), row1.clone(), row2.clone()]);
    let layer = json!([head.clone(), head.clone()]);
    json!([layer.clone(), layer.clone()])
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_f_17_cli_help_advertises_flags() {
    let out = apr_binary()
        .args(["attn-viz-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--attn-file",
        "--html-file",
        "--expected-heatmaps",
        "--tolerance",
        "--epsilon",
    ] {
        assert!(
            stdout.contains(flag),
            "--help must advertise {flag}; got:\n{stdout}"
        );
    }
}

#[test]
fn falsify_crux_f_17_cli_requires_at_least_one_file() {
    let out = apr_binary().args(["attn-viz-lint"]).output().expect("run");
    assert!(!out.status.success(), "bare invocation must not exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("at least one of --attn-file or --html-file"),
        "stderr must explain requirement; got:\n{stderr}"
    );
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_f_17_001_row_softmax_ok_on_good_dump() {
    let f = write_json(&good_attn());
    let out = apr_binary()
        .args(["attn-viz-lint", "--attn-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good attn must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_17_001_row_softmax_rejects_unnormalized() {
    let bad = json!([[[[0.6, 0.6, 0.0], [0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]]]);
    let f = write_json(&bad);
    let out = apr_binary()
        .args(["attn-viz-lint", "--attn-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "unnormalized row must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("RowOutOfNormalization"),
        "stderr must name RowOutOfNormalization; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_17_002_causal_mask_rejects_nonzero_future() {
    // Row 0 has weight on column 1 (a future position).
    let bad = json!([[[[0.5, 0.5, 0.0], [0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]]]);
    let f = write_json(&bad);
    let out = apr_binary()
        .args(["attn-viz-lint", "--attn-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "causal-mask leak must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("NonZeroFuturePosition") || stderr.contains("RowOutOfNormalization"),
        "stderr must name NonZeroFuturePosition (or RowOutOfNormalization first); got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_17_003_html_heatmap_count_ok_when_threshold_met() {
    let html = "<html><svg></svg><svg></svg><canvas></canvas><svg></svg></html>";
    let f = write_html(html);
    let out = apr_binary()
        .args(["attn-viz-lint", "--html-file"])
        .arg(f.path())
        .args(["--expected-heatmaps", "4"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "4 heatmaps with threshold 4 must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_17_003_html_heatmap_count_rejects_too_few() {
    let html = "<html><svg></svg></html>";
    let f = write_html(html);
    let out = apr_binary()
        .args(["attn-viz-lint", "--html-file"])
        .arg(f.path())
        .args(["--expected-heatmaps", "4"])
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "1 heatmap with threshold 4 must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("TooFewHeatmaps"),
        "stderr must name TooFewHeatmaps; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_17_tolerance_relaxes_row_softmax_gate() {
    // Causal-mask honored; row 2 sums to 1.01 — strict 1e-5 fails, relaxed 0.05 passes.
    let body = json!([[[[1.0, 0.0, 0.0], [0.4, 0.6, 0.0], [0.21, 0.30, 0.50]]]]);
    let f = write_json(&body);
    let out = apr_binary()
        .args(["attn-viz-lint", "--attn-file"])
        .arg(f.path())
        .args(["--tolerance", "0.05"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "0.01 deviation within 0.05 tolerance must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_f_17_json_output_contains_outcomes() {
    let af = write_json(&good_attn());
    let hf = write_html("<svg></svg><svg></svg><svg></svg><svg></svg>");
    let out = apr_binary()
        .args(["--json", "attn-viz-lint", "--attn-file"])
        .arg(af.path())
        .arg("--html-file")
        .arg(hf.path())
        .args(["--expected-heatmaps", "4"])
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good bodies must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["row_softmax"]
        .as_str()
        .expect("row_softmax")
        .contains("Ok"));
    assert!(parsed["causal_mask"]
        .as_str()
        .expect("causal_mask")
        .contains("Ok"));
    assert!(parsed["html_heatmaps"]
        .as_str()
        .expect("html_heatmaps")
        .contains("Ok"));
}
