//! `run_repl` loop behaviour, driven through real stdin.
//!
//! `run_repl` reads `io::stdin()` directly, so each case re-executes this test
//! binary as a child with piped stdin (`HOME` pointed at a temp dir so session
//! files never touch the real one) and asserts on the child's stdout.
//!
//! Refs: #5706

#![cfg(feature = "agents")]

use std::io::Write;
use std::process::{Command, Stdio};

use batuta::agent::driver::mock::MockDriver;
use batuta::agent::manifest::AgentManifest;
use batuta::agent::memory::InMemorySubstrate;
use batuta::agent::repl::run_repl;
use batuta::agent::tool::ToolRegistry;

const CHILD_ENV: &str = "APR_REPL_RUN_CHILD";

/// Child entry point: a no-op unless re-executed by `run_child`.
#[test]
fn child_entry() {
    let Ok(spec) = std::env::var(CHILD_ENV) else { return };
    let mut parts = spec.split(',');
    let max_turns: u32 = parts.next().unwrap().parse().unwrap();
    let budget: f64 = parts.next().unwrap().parse().unwrap();
    let driver = MockDriver::single_response("hello from mock");
    let mut manifest = AgentManifest::default();
    manifest.name = "repl-run-test".to_string();
    run_repl(
        &manifest,
        &driver,
        &ToolRegistry::new(),
        &InMemorySubstrate::new(),
        max_turns,
        budget,
        None,
    )
    .expect("run_repl returns Ok");
}

/// Run `run_repl` in a child with `stdin` as input; returns its stdout.
fn run_child(max_turns: u32, budget: f64, stdin: &str) -> String {
    let home = tempfile::tempdir().expect("tempdir");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["child_entry", "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, format!("{max_turns},{budget}"))
        .env("HOME", home.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn child");
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let out = child.wait_with_output().expect("child output");
    assert!(out.status.success(), "child failed: {}", String::from_utf8_lossy(&out.stdout));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn eof_ends_the_loop_with_no_summary_when_no_turn_ran() {
    let out = run_child(10, 100.0, "");
    assert!(!out.contains("Session Summary"), "{out}");
    assert!(!out.contains("tok]"), "{out}");
}

#[test]
fn a_prompt_runs_one_turn_then_quit_prints_the_summary() {
    let out = run_child(10, 100.0, "hi\n/quit\n");
    assert!(out.contains("10/5 tok"), "{out}");
    assert!(out.contains("Session Summary"), "{out}");
    assert!(out.contains("Turns: 1,"), "{out}");
}

#[test]
fn blank_lines_do_not_consume_a_turn() {
    let out = run_child(10, 100.0, "\n   \n/quit\n");
    assert!(!out.contains("Session Summary"), "{out}");
}

#[test]
fn max_turns_stops_the_loop_before_reading_more_input() {
    let out = run_child(1, 100.0, "one\ntwo\n");
    assert!(out.contains("Max turns (1) reached"), "{out}");
    assert!(out.contains("Turns: 1,"), "{out}");
}

#[test]
fn a_zero_budget_stops_before_the_first_turn() {
    let out = run_child(10, 0.0, "hi\n");
    assert!(out.contains("Budget ($0.00) exhausted"), "{out}");
    assert!(!out.contains("tok]"), "{out}");
}
