//! Slash-command handling in the REPL, driven through real stdin.
//!
//! `handle_slash_command` is private and prints to stdout, so each case re-executes this
//! test binary as a child running `run_repl` with piped stdin (`HOME` and `NO_COLOR` set)
//! and asserts on the child's stdout.
//!
//! Refs: #5540

#![cfg(feature = "agents")]

use std::io::Write;
use std::process::{Command, Stdio};

use batuta::agent::driver::mock::MockDriver;
use batuta::agent::manifest::AgentManifest;
use batuta::agent::memory::InMemorySubstrate;
use batuta::agent::repl::run_repl;
use batuta::agent::tool::ToolRegistry;

const CHILD_ENV: &str = "APR_REPL_SLASH_CHILD";

/// Child entry point: a no-op unless re-executed by `run_child`.
#[test]
fn child_entry() {
    if std::env::var(CHILD_ENV).is_err() {
        return;
    }
    let driver = MockDriver::single_response("ok");
    let mut manifest = AgentManifest::default();
    manifest.name = "repl-slash-test".to_string();
    run_repl(&manifest, &driver, &ToolRegistry::new(), &InMemorySubstrate::new(), 10, 100.0, None)
        .expect("run_repl returns Ok");
}

/// Feed `stdin` to a fresh REPL and return its stdout.
fn run_child(stdin: &str) -> String {
    let home = tempfile::tempdir().expect("tempdir");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["child_entry", "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env("HOME", home.path())
        .env("NO_COLOR", "1")
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
fn cost_on_a_fresh_session_is_free_with_zero_counters() {
    let out = run_child("/cost\n/quit\n");
    assert!(out.contains("(local inference)") && !out.contains("% )"), "{out}");
    assert!(out.contains("Tokens: 0 in / 0 out"), "{out}");
    assert!(out.contains("Turns: 0, Tool calls: 0"), "{out}");
}

#[test]
fn cost_and_context_count_the_turn_that_ran() {
    let out = run_child("hello\n/cost\n/context\n/quit\n");
    assert!(out.contains("Tokens: 10 in / 5 out"), "{out}");
    assert!(out.contains("Turns: 1, Tool calls: 0"), "{out}");
    assert!(out.contains("History: 2 messages (1 user, 1 assistant, 0 tool)"), "{out}");
}

#[test]
fn context_on_empty_history_reports_zero_and_no_limit_warning() {
    let out = run_child("/context\n/quit\n");
    assert!(out.contains("History: 0 messages (0 user, 0 assistant, 0 tool)"), "{out}");
    assert!(out.contains("Context: ~0 / 32768 tokens (0%)"), "{out}");
    assert!(!out.contains("Near context limit"), "{out}");
}

#[test]
fn clear_empties_the_history() {
    let out = run_child("hello\n/clear\n/context\n/quit\n");
    assert!(out.contains("Screen and conversation history cleared."), "{out}");
    let after = out.rsplit("cleared.").next().unwrap();
    assert!(after.contains("History: 0 messages"), "{after}");
}

#[test]
fn compact_reports_before_and_after_message_counts() {
    let out = run_child("/compact\n/quit\n");
    assert!(out.contains("Compacted: 0 -> 0 messages"), "{out}");
}

#[test]
fn session_reports_the_id_and_message_count() {
    let out = run_child("hello\n/session\n/quit\n");
    assert!(out.contains("Turns: 1, Messages: 2"), "{out}");
    assert!(out.matches("Session: ").count() >= 2, "banner and /session both print the id:\n{out}");
}

#[test]
fn every_placeholder_command_prints_its_own_message() {
    let cases = [
        ("/model", "Model switching not yet implemented."),
        ("/mcp", "Project-root .mcp.json loader: PMAT-CODE-MCP-JSON-LOADER-001"),
        ("/config", "User-global ladder tracked in PMAT-CODE-CONFIG-LADDER-001."),
        ("/review", "tracked by PMAT-CODE-REVIEW-001."),
        ("/memory", "/memory TUI: PMAT-CODE-MEMORY-TUI-001."),
        ("/permissions", "tracked by PMAT-CODE-PERMISSIONS-001."),
        ("/hooks", "tracked by PMAT-CODE-HOOKS-001."),
        ("/init", "tracked by PMAT-CODE-INIT-001."),
        ("/resume", "CLI `apr code --resume [id]` works today."),
        ("/add-dir", "tracked by PMAT-CODE-ADDDIR-001."),
        ("/agents", "tracked by PMAT-CODE-CUSTOM-AGENTS-001."),
        ("/debug", "tracked by PMAT-CODE-SLASH-DEBUG-001"),
        ("/rename", "tracked by PMAT-CODE-SLASH-RENAME-001"),
        ("/upgrade", "tracked by PMAT-CODE-SLASH-UPGRADE-001"),
    ];
    let script: String =
        cases.iter().map(|(c, _)| format!("{c}\n")).collect::<String>() + "/quit\n";
    let out = run_child(&script);
    for (cmd, expect) in cases {
        assert!(out.contains(expect), "{cmd}: missing {expect:?} in:\n{out}");
    }
}

#[test]
fn an_unknown_command_is_named_and_does_not_start_a_turn() {
    let out = run_child("/bogus\n/quit\n");
    assert!(out.contains("Unknown command: /bogus. Type /help for commands."), "{out}");
    assert!(!out.contains("Session Summary"), "{out}");
}

#[test]
fn quit_prints_goodbye_and_ends_the_loop() {
    let out = run_child("/quit\n/cost\n");
    assert!(out.contains("Goodbye."), "{out}");
    assert!(!out.contains("Cost:"), "input after /quit must not be read:\n{out}");
}
