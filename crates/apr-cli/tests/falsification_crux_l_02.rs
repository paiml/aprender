//! CRUX-L-02 — end-to-end falsification harness for `apr attn-parity-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-L-02-{002,003,004} gate
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
        .prefix("crux-l-02-")
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

fn good_parity() -> serde_json::Value {
    json!({"max_abs_diff": 0.002, "cosine_sim": 0.99999})
}

fn good_provenance() -> serde_json::Value {
    json!({
        "attn_impl": "flash2",
        "kernel_source": "hf-kernels-community:flash-attn2@abcdef0123456789abcdef0123456789abcdef01",
        "fallback": null
    })
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_l_02_cli_help_advertises_flags() {
    let out = apr_binary()
        .args(["attn-parity-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in [
        "--parity-file",
        "--provenance-file",
        "--head-dim-error-file",
        "--tol-abs",
        "--tol-cos",
    ] {
        assert!(
            stdout.contains(flag),
            "--help must advertise {flag}; got:\n{stdout}"
        );
    }
}

#[test]
fn falsify_crux_l_02_cli_requires_at_least_one_file() {
    let out = apr_binary()
        .args(["attn-parity-lint"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "bare invocation must not exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("at least one of"),
        "stderr must explain requirement; got:\n{stderr}"
    );
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_l_02_002_parity_ok_within_bounds() {
    let f = write_json(&good_parity());
    let out = apr_binary()
        .args(["attn-parity-lint", "--parity-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "in-tolerance parity must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_l_02_002_parity_rejects_max_abs_diff_above() {
    let body = json!({"max_abs_diff": 0.01, "cosine_sim": 0.99999});
    let f = write_json(&body);
    let out = apr_binary()
        .args(["attn-parity-lint", "--parity-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "max_abs_diff > 5e-3 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MaxAbsDiffExceedsTolerance"),
        "stderr must name MaxAbsDiffExceedsTolerance; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_l_02_002_parity_rejects_cosine_below_floor() {
    let body = json!({"max_abs_diff": 0.001, "cosine_sim": 0.99});
    let f = write_json(&body);
    let out = apr_binary()
        .args(["attn-parity-lint", "--parity-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "cosine < 0.9999 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("CosineSimBelowFloor"),
        "stderr must name CosineSimBelowFloor; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_l_02_003_provenance_ok_on_pinned_flash2() {
    let f = write_json(&good_provenance());
    let out = apr_binary()
        .args(["attn-parity-lint", "--provenance-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "pinned flash2 source must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_l_02_003_provenance_rejects_malformed_sha() {
    let body = json!({
        "attn_impl": "flash2",
        "kernel_source": "hf-kernels-community:flash-attn2@short"
    });
    let f = write_json(&body);
    let out = apr_binary()
        .args(["attn-parity-lint", "--provenance-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "malformed sha must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("KernelSourceMalformed"),
        "stderr must name KernelSourceMalformed; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_l_02_003_provenance_ok_on_naive_with_reason() {
    let body = json!({"attn_impl": "naive", "fallback": "no-gpu"});
    let f = write_json(&body);
    let out = apr_binary()
        .args(["attn-parity-lint", "--provenance-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "naive with fallback reason must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_l_02_004_head_dim_error_ok_on_unsupported() {
    let body = json!({"error": "unsupported-head-dim: got 96, expected 64 or 128"});
    let f = write_json(&body);
    let out = apr_binary()
        .args(["attn-parity-lint", "--head-dim-error-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "unsupported-head-dim error must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_l_02_004_head_dim_error_rejects_irrelevant_error() {
    let body = json!({"error": "out of memory"});
    let f = write_json(&body);
    let out = apr_binary()
        .args(["attn-parity-lint", "--head-dim-error-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "irrelevant error must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ErrorDoesNotMentionHeadDim"),
        "stderr must name ErrorDoesNotMentionHeadDim; got:\n{stderr}"
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_l_02_json_output_contains_outcomes() {
    let fp = write_json(&good_parity());
    let fv = write_json(&good_provenance());
    let out = apr_binary()
        .args(["--json", "attn-parity-lint", "--parity-file"])
        .arg(fp.path())
        .arg("--provenance-file")
        .arg(fv.path())
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good bodies must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["parity_numerics"]
        .as_str()
        .expect("parity")
        .contains("Ok"));
    assert!(parsed["provenance"]
        .as_str()
        .expect("provenance")
        .contains("OkFlash2"));
}
