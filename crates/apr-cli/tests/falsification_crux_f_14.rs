//! CRUX-F-14 — end-to-end falsification harness for `apr hang-trace-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-F-14-{001,002,003} gate
//! the classifier discharges has a matching captured trace directory that the
//! binary must classify exactly as the harness expects.

use std::fs;
use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

/// Build a temp directory with the given (filename, body) pairs.
fn make_trace_dir(pairs: &[(&str, &[u8])]) -> tempfile::TempDir {
    let dir = tempfile::Builder::new()
        .prefix("crux-f-14-")
        .tempdir()
        .expect("tempdir");
    for (name, body) in pairs {
        let mut f = fs::File::create(dir.path().join(name)).expect("create file");
        f.write_all(body).expect("write");
        f.flush().expect("flush");
    }
    dir
}

fn good_world_size_2() -> Vec<(&'static str, &'static [u8])> {
    vec![
        ("rank0.py.txt", b"Traceback...\nFile \"x.py\", line 1\n"),
        ("rank1.py.txt", b"Traceback...\nFile \"x.py\", line 1\n"),
    ]
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_f_14_cli_help_advertises_flags() {
    let out = apr_binary().args(["hang-trace-lint", "--help"]).output().expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in ["--trace-dir", "--mode", "--world-size", "--exit-code", "--expected-exit-code"] {
        assert!(stdout.contains(flag), "--help must advertise {flag}; got:\n{stdout}");
    }
}

#[test]
fn falsify_crux_f_14_cli_missing_dir_fails() {
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir", "/nonexistent/crux-f-14-missing"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing dir must not exit 0");
}

#[test]
fn falsify_crux_f_14_cli_rejects_invalid_mode() {
    let dir = make_trace_dir(&good_world_size_2());
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--mode", "ambiguous"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "invalid mode must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--mode must be"),
        "stderr must explain mode requirement; got:\n{stderr}"
    );
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_f_14_001_timeout_ok_on_complete_dump() {
    let dir = make_trace_dir(&good_world_size_2());
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--world-size", "2"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "complete dump must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_14_001_timeout_rejects_missing_rank() {
    let pairs: Vec<(&str, &[u8])> = vec![("rank0.py.txt", b"traceback\n")];
    let dir = make_trace_dir(&pairs);
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--world-size", "2"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing rank must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingRank"),
        "stderr must name MissingRank; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_14_001_timeout_rejects_empty_file() {
    let pairs: Vec<(&str, &[u8])> = vec![
        ("rank0.py.txt", b"traceback\n"),
        ("rank1.py.txt", b""),
    ];
    let dir = make_trace_dir(&pairs);
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--world-size", "2"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "empty rank file must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("EmptyFile"),
        "stderr must name EmptyFile; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_14_002_success_ok_when_dir_is_empty() {
    let pairs: Vec<(&str, &[u8])> = vec![];
    let dir = make_trace_dir(&pairs);
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--mode", "success"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "empty dir under success mode must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_14_002_success_rejects_unexpected_file() {
    let pairs: Vec<(&str, &[u8])> = vec![("rank0.py.txt", b"oops\n")];
    let dir = make_trace_dir(&pairs);
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--mode", "success"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "false-trigger dump must fail success mode");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("UnexpectedFile"),
        "stderr must name UnexpectedFile; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_14_003_exit_code_124_passes_timeout_expectation() {
    let dir = make_trace_dir(&good_world_size_2());
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--world-size", "2", "--exit-code", "124", "--expected-exit-code", "124"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "exit 124 matching 124 must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_14_003_exit_code_1_fails_timeout_expectation() {
    let dir = make_trace_dir(&good_world_size_2());
    let out = apr_binary()
        .args(["hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--world-size", "2", "--exit-code", "1", "--expected-exit-code", "124"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "exit 1 != 124 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ExitCodeMismatch"),
        "stderr must name ExitCodeMismatch; got:\n{stderr}"
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_f_14_json_output_contains_outcomes() {
    let dir = make_trace_dir(&good_world_size_2());
    let out = apr_binary()
        .args(["--json", "hang-trace-lint", "--trace-dir"])
        .arg(dir.path())
        .args(["--world-size", "2", "--exit-code", "124", "--expected-exit-code", "124"])
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good dir must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["timeout_dump"].as_str().expect("timeout_dump").contains("Ok"));
    assert!(parsed["exit_code"].as_str().expect("exit_code").contains("OkTimeout"));
}
