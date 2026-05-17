//! CRUX-I-06 — end-to-end falsification harness for `apr react-trace-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-I-06-{001,002} gate
//! the classifier discharges has a matching captured trace JSON that the
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
        .prefix("crux-i-06-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(serde_json::to_vec_pretty(body).expect("serialize").as_slice())
        .expect("write");
    f.flush().expect("flush");
    f
}

fn good_final_answer() -> serde_json::Value {
    json!({
        "iterations": 1,
        "answer": "4",
        "scratchpad": "Thought: I should compute 2+2.\nFinal Answer: 4",
        "exit_code": 0
    })
}

fn good_max_iterations() -> serde_json::Value {
    json!({
        "iterations": 3,
        "reason": "max_iterations",
        "scratchpad": "Thought: try\nAction: echo\nAction Input: hi\nObservation: hi\nThought: try\nAction: echo\nAction Input: hi\nObservation: hi\nThought: still trying\nAction: echo\nAction Input: x\nObservation: x",
        "exit_code": 2
    })
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_i_06_cli_help_advertises_flags() {
    let out = apr_binary().args(["react-trace-lint", "--help"]).output().expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in ["--trace-file", "--max-iterations", "--require-grammar"] {
        assert!(stdout.contains(flag), "--help must advertise {flag}; got:\n{stdout}");
    }
}

#[test]
fn falsify_crux_i_06_cli_missing_file_fails() {
    let out = apr_binary()
        .args(["react-trace-lint", "--trace-file", "/nonexistent/crux-i-06-missing.json"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_i_06_001_termination_ok_on_final_answer() {
    let f = write_json(&good_final_answer());
    let out = apr_binary()
        .args(["react-trace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "final-answer trace must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_i_06_002_termination_ok_on_max_iterations() {
    let f = write_json(&good_max_iterations());
    let out = apr_binary()
        .args(["react-trace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "max-iterations trace must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_i_06_001_termination_rejects_exit_code_mismatch() {
    let body = json!({
        "iterations": 3,
        "reason": "max_iterations",
        "scratchpad": "Thought: x\nAction: y\nAction Input: z\nObservation: o",
        "exit_code": 1
    });
    let f = write_json(&body);
    let out = apr_binary()
        .args(["react-trace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "exit-code mismatch must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ExitCodeMismatch"),
        "stderr must name ExitCodeMismatch; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_i_06_001_termination_rejects_unknown_reason() {
    let body = json!({
        "iterations": 1,
        "reason": "weird_other",
        "scratchpad": "Thought: x\nFinal Answer: 4",
        "exit_code": 7
    });
    let f = write_json(&body);
    let out = apr_binary()
        .args(["react-trace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "unknown reason must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("UnknownReason"),
        "stderr must name UnknownReason; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_i_06_001_grammar_rejects_action_after_final_answer() {
    let body = json!({
        "iterations": 1,
        "answer": "4",
        "scratchpad": "Thought: done\nFinal Answer: 4\nAction: rogue\nAction Input: x",
        "exit_code": 0
    });
    let f = write_json(&body);
    let out = apr_binary()
        .args(["react-trace-lint", "--trace-file"])
        .arg(f.path())
        .arg("--require-grammar")
        .output()
        .expect("run");
    assert!(!out.status.success(), "action-after-Final-Answer must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ActionAfterFinalAnswer"),
        "stderr must name ActionAfterFinalAnswer; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_i_06_002_iteration_bound_ok_within_budget() {
    let f = write_json(&good_max_iterations());
    let out = apr_binary()
        .args(["react-trace-lint", "--trace-file"])
        .arg(f.path())
        .args(["--max-iterations", "5"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "iterations=3 within budget=5 must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_i_06_002_iteration_bound_rejects_over_budget() {
    let body = json!({
        "iterations": 100,
        "reason": "max_iterations",
        "scratchpad": "Thought: x\nAction: y\nAction Input: z\nObservation: o",
        "exit_code": 2
    });
    let f = write_json(&body);
    let out = apr_binary()
        .args(["react-trace-lint", "--trace-file"])
        .arg(f.path())
        .args(["--max-iterations", "5"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "iterations > budget must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("IterationsExceedBudget"),
        "stderr must name IterationsExceedBudget; got:\n{stderr}"
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_i_06_json_output_contains_outcomes() {
    let f = write_json(&good_max_iterations());
    let out = apr_binary()
        .args(["--json", "react-trace-lint", "--trace-file"])
        .arg(f.path())
        .args(["--max-iterations", "5"])
        .arg("--require-grammar")
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good body must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["termination"].as_str().expect("termination").contains("Ok"));
    assert!(parsed["iteration_bound"].as_str().expect("iteration_bound").contains("Ok"));
    assert!(parsed["scratchpad_grammar"].as_str().expect("scratchpad_grammar").contains("Ok"));
}
