//! CRUX-F-06 — end-to-end falsification harness for `apr kv-timeline-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-F-06-{001..004}
//! gate the classifier discharges has a matching captured KV-timeline JSON
//! that the binary must classify exactly as the harness expects.

use serde_json::json;
use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_body(body: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-06-")
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

fn good_timeline() -> serde_json::Value {
    json!({
        "timeline": [
            {"step": 0, "t_ms": 0.0,  "used_blocks":  10, "free_blocks":  90, "used_pct": 0.10, "active_seqs": 1, "preempted_seqs": 0},
            {"step": 1, "t_ms": 8.0,  "used_blocks":  50, "free_blocks":  50, "used_pct": 0.50, "active_seqs": 1, "preempted_seqs": 0},
            {"step": 2, "t_ms": 16.0, "used_blocks":  96, "free_blocks":   4, "used_pct": 0.96, "active_seqs": 2, "preempted_seqs": 1},
        ],
        "block_size_tokens": 16,
        "total_blocks": 100,
        "peak_used_pct": 0.96,
        "preemption_count": 1,
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
fn falsify_crux_f_06_cli_help_advertises_flags() {
    let out = apr_binary()
        .args(["kv-timeline-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--timeline-file"),
        "--help must advertise --timeline-file; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--preempt-threshold"),
        "--help must advertise --preempt-threshold; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_f_06_cli_missing_file_fails() {
    let out = apr_binary()
        .args([
            "kv-timeline-lint",
            "--timeline-file",
            "/nonexistent/crux-f-06-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_f_06_001_schema_ok_on_good_body() {
    let f = write_body(&good_timeline());
    let out = apr_binary()
        .args(["kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good timeline must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_06_001_schema_reports_missing_top_key() {
    let body =
        json!({"timeline": [], "block_size_tokens": 16, "total_blocks": 100, "peak_used_pct": 0.0});
    let f = write_body(&body);
    let out = apr_binary()
        .args(["kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing preemption_count must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingTopKey"),
        "stderr must name MissingTopKey; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_06_002_block_conservation_rejects_violation() {
    let body = json!({
        "timeline": [{"step": 0, "t_ms": 0.0, "used_blocks": 30, "free_blocks": 30, "used_pct": 0.30, "active_seqs": 1, "preempted_seqs": 0}],
        "block_size_tokens": 16,
        "total_blocks": 100,
        "peak_used_pct": 0.30,
        "preemption_count": 0,
    });
    let f = write_body(&body);
    let out = apr_binary()
        .args(["kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "block conservation violation must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Violation"),
        "stderr must name Violation; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_06_002_used_pct_arithmetic_rejects_mismatch() {
    let body = json!({
        "timeline": [{"step": 0, "t_ms": 0.0, "used_blocks": 50, "free_blocks": 50, "used_pct": 0.10, "active_seqs": 1, "preempted_seqs": 0}],
        "block_size_tokens": 16,
        "total_blocks": 100,
        "peak_used_pct": 0.10,
        "preemption_count": 0,
    });
    let f = write_body(&body);
    let out = apr_binary()
        .args(["kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "used_pct mismatch must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Mismatch"),
        "stderr must name Mismatch; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_06_004_peak_consistency_rejects_peak_mismatch() {
    let mut body = good_timeline();
    body["peak_used_pct"] = json!(0.50);
    let f = write_body(&body);
    let out = apr_binary()
        .args(["kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "peak mismatch must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("PeakMismatch"),
        "stderr must name PeakMismatch; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_06_004_peak_consistency_rejects_preempt_count_mismatch() {
    let mut body = good_timeline();
    body["preemption_count"] = json!(99);
    let f = write_body(&body);
    let out = apr_binary()
        .args(["kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "preempt count mismatch must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("PreemptionCountMismatch"),
        "stderr must name PreemptionCountMismatch; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_06_003_preemption_trigger_rejects_below_threshold() {
    let body = json!({
        "timeline": [{"step": 0, "t_ms": 0.0, "used_blocks": 50, "free_blocks": 50, "used_pct": 0.50, "active_seqs": 1, "preempted_seqs": 1}],
        "block_size_tokens": 16,
        "total_blocks": 100,
        "peak_used_pct": 0.50,
        "preemption_count": 1,
    });
    let f = write_body(&body);
    let out = apr_binary()
        .args(["kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "preempt below 0.95 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("PreemptionBelowThreshold"),
        "stderr must name PreemptionBelowThreshold; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_06_003_preemption_trigger_respects_custom_threshold() {
    // A trace that preempts at 0.80; default threshold (0.95) rejects, but
    // --preempt-threshold 0.80 accepts.
    let body = json!({
        "timeline": [{"step": 0, "t_ms": 0.0, "used_blocks": 80, "free_blocks": 20, "used_pct": 0.80, "active_seqs": 1, "preempted_seqs": 1}],
        "block_size_tokens": 16,
        "total_blocks": 100,
        "peak_used_pct": 0.80,
        "preemption_count": 1,
    });
    let f = write_body(&body);
    let out = apr_binary()
        .args(["kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .args(["--preempt-threshold", "0.80"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "preempt at exactly threshold must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_f_06_json_output_contains_outcomes() {
    let f = write_body(&good_timeline());
    let out = apr_binary()
        .args(["--json", "kv-timeline-lint", "--timeline-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good body must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert_outcome_ok(&parsed, "schema");
    assert_outcome_ok(&parsed, "block_conservation");
    assert_outcome_ok(&parsed, "used_pct_arithmetic");
    assert_outcome_ok(&parsed, "peak_consistency");
    assert_outcome_ok(&parsed, "preemption_trigger");
}
