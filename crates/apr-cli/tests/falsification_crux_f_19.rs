//! CRUX-F-19 — end-to-end falsification harness for `apr explain-token-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-F-19-{001,002,003}
//! gate the classifier discharges has a matching captured JSONL body that
//! the binary must classify exactly as the harness expects.

use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_body(body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-19-")
        .suffix(".jsonl")
        .tempfile()
        .expect("tempfile");
    f.write_all(body.as_bytes()).expect("write");
    f.flush().expect("flush");
    f
}

fn good_body() -> String {
    // Two normalized steps with sampled token present in candidates.
    let l0 = r#"{"step":0,"sampled_id":7,"candidates":[{"token_id":7,"pre_prob":0.6,"post_prob":0.7,"rank":0},{"token_id":3,"pre_prob":0.3,"post_prob":0.2,"rank":1},{"token_id":5,"pre_prob":0.1,"post_prob":0.1,"rank":2}]}"#;
    let l1 = r#"{"step":1,"sampled_id":3,"candidates":[{"token_id":3,"pre_prob":0.5,"post_prob":0.5,"rank":0},{"token_id":7,"pre_prob":0.4,"post_prob":0.5,"rank":1}]}"#;
    format!("{l0}\n{l1}\n")
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_f_19_cli_help_advertises_flags() {
    let out = apr_binary()
        .args(["explain-token-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--jsonl-file"),
        "--help must advertise --jsonl-file; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--tolerance"),
        "--help must advertise --tolerance; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--require-greedy"),
        "--help must advertise --require-greedy; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_f_19_cli_missing_file_fails() {
    let out = apr_binary()
        .args([
            "explain-token-lint",
            "--jsonl-file",
            "/nonexistent/crux-f-19-missing.jsonl",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_f_19_001_probs_normalize_ok_on_good_body() {
    let f = write_body(&good_body());
    let out = apr_binary()
        .args(["explain-token-lint", "--jsonl-file"])
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
fn falsify_crux_f_19_001_probs_normalize_rejects_unnormalized() {
    let body = r#"{"step":0,"sampled_id":1,"candidates":[{"token_id":1,"pre_prob":0.5,"post_prob":0.6,"rank":0},{"token_id":2,"pre_prob":0.5,"post_prob":0.6,"rank":1}]}
"#;
    let f = write_body(body);
    let out = apr_binary()
        .args(["explain-token-lint", "--jsonl-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "unnormalized probs must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("NotNormalized"),
        "stderr must name NotNormalized; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_19_002_sampled_in_candidates_rejects_missing() {
    let body = r#"{"step":0,"sampled_id":99,"candidates":[{"token_id":1,"pre_prob":1.0,"post_prob":1.0,"rank":0}]}
"#;
    let f = write_body(body);
    let out = apr_binary()
        .args(["explain-token-lint", "--jsonl-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "sampled_id not in candidates must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("SampledNotInCandidates"),
        "stderr must name SampledNotInCandidates; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_19_003_greedy_argmax_ok_on_good_body() {
    let f = write_body(&good_body());
    let out = apr_binary()
        .args(["explain-token-lint", "--jsonl-file"])
        .arg(f.path())
        .arg("--require-greedy")
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "greedy=argmax in good body; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_19_003_greedy_argmax_rejects_non_argmax() {
    // sampled_id is rank=1, not the argmax of pre_prob
    let body = r#"{"step":0,"sampled_id":3,"candidates":[{"token_id":7,"pre_prob":0.9,"post_prob":1.0,"rank":0},{"token_id":3,"pre_prob":0.1,"post_prob":0.0,"rank":1}]}
"#;
    let f = write_body(body);
    let out = apr_binary()
        .args(["explain-token-lint", "--jsonl-file"])
        .arg(f.path())
        .arg("--require-greedy")
        .output()
        .expect("run");
    assert!(!out.status.success(), "non-argmax under greedy must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("NotArgmax"),
        "stderr must name NotArgmax; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_19_001_schema_rejects_empty_body() {
    let f = write_body("");
    let out = apr_binary()
        .args(["explain-token-lint", "--jsonl-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "empty body must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Empty"),
        "stderr must name Empty; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_19_001_schema_rejects_malformed_line() {
    let f = write_body("not json\n");
    let out = apr_binary()
        .args(["explain-token-lint", "--jsonl-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "non-json line must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("LineNotJson"),
        "stderr must name LineNotJson; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_19_tolerance_relaxes_normalize_gate() {
    // Probs sum to 1.01 — strict tolerance fails, relaxed passes.
    let body = r#"{"step":0,"sampled_id":1,"candidates":[{"token_id":1,"pre_prob":0.5,"post_prob":0.51,"rank":0},{"token_id":2,"pre_prob":0.5,"post_prob":0.50,"rank":1}]}
"#;
    let f = write_body(body);
    let out = apr_binary()
        .args(["explain-token-lint", "--jsonl-file"])
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
fn falsify_crux_f_19_json_output_contains_outcomes() {
    let f = write_body(&good_body());
    let out = apr_binary()
        .args(["--json", "explain-token-lint", "--jsonl-file"])
        .arg(f.path())
        .arg("--require-greedy")
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good body must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["schema"].as_str().expect("schema").contains("Ok"));
    assert!(parsed["probs_normalize"]
        .as_str()
        .expect("probs")
        .contains("Ok"));
    assert!(parsed["sampled_in_candidates"]
        .as_str()
        .expect("sampled")
        .contains("Ok"));
    assert!(parsed["greedy_picks_argmax"]
        .as_str()
        .expect("greedy")
        .contains("Ok"));
}
