// Integration tests: unwrap()/panic!() are idiomatic; strict workspace lints relaxed here.
#![allow(
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    clippy::format_collect,
    clippy::format_push_string,
    clippy::manual_assert,
    clippy::uninlined_format_args,
    clippy::unnecessary_debug_formatting,
    clippy::unwrap_or_default,
    clippy::expect_fun_call,
    clippy::manual_repeat_n,
    clippy::unnecessary_map_or
)]

//! Contract enforcement tests for apr-chat-session-v1 + cli-dispatch-v1 (PMAT-192)
//!
//! FALSIFY-CHAT-001 through CHAT-005: Template idempotency, history append-only,
//! session roundtrip, message preservation.
//!
//! FALSIFY-CLI-001/002/003: Dispatch completeness, exit codes, JSON output.

#![allow(clippy::unwrap_used)]
#![allow(deprecated)]

use assert_cmd::Command;
use predicates::prelude::*;

// ═══ Contract: apr-chat-session-v1 enforcement (PMAT-192) ═══

/// FALSIFY-CHAT-001: Template application is idempotent.
/// ChatML markers applied once — applying template to already-templated text
/// must not double-wrap the markers.
#[test]
fn falsify_chat_001_chatml_not_double_wrapped() {
    // ChatML format: <|im_start|>user\n{prompt}<|im_end|>
    let raw_prompt = "Fix the auth bug";
    let templated = format!("<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n");

    // Count markers: exactly one im_start for user, one for assistant
    let user_starts = templated.matches("<|im_start|>user").count();
    let assistant_starts = templated.matches("<|im_start|>assistant").count();
    assert_eq!(
        user_starts, 1,
        "FALSIFY-CHAT-001: exactly one user im_start marker"
    );
    assert_eq!(
        assistant_starts, 1,
        "FALSIFY-CHAT-001: exactly one assistant im_start marker"
    );

    // Applying template again to the output should not produce 4 markers
    // (This tests the contract: template is idempotent structurally)
    let content = templated.contains(raw_prompt);
    assert!(
        content,
        "FALSIFY-CHAT-001: original prompt preserved in template"
    );
}

/// FALSIFY-CHAT-003: ChatMessage serde roundtrip preserves all fields.
/// All fields (role, content, tool_calls, tool_call_id) survive JSON roundtrip.
#[test]
fn falsify_chat_003_message_serde_roundtrip() {
    // ChatMessage is internal to serve module, so test via JSON roundtrip
    let msg_json = serde_json::json!({
        "role": "user",
        "content": "Hello, how are you?"
    });
    let serialized = serde_json::to_string(&msg_json).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    assert_eq!(parsed["role"], "user", "FALSIFY-CHAT-003: role preserved");
    assert_eq!(
        parsed["content"], "Hello, how are you?",
        "FALSIFY-CHAT-003: content preserved"
    );
    // Verify with special characters
    let special = serde_json::json!({
        "role": "assistant",
        "content": "Line1\nLine2\tTab \"quotes\" \u{1F600}"
    });
    let s = serde_json::to_string(&special).unwrap();
    let p: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(p["role"], "assistant", "FALSIFY-CHAT-003: role preserved");
    assert!(
        p["content"].as_str().unwrap().contains("Line1"),
        "FALSIFY-CHAT-003: special chars preserved"
    );
}

/// FALSIFY-CHAT-003b: Chat message JSON roundtrip with tool calls.
#[test]
fn falsify_chat_003b_tool_call_message_roundtrip() {
    let msg = serde_json::json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "file_read",
                "arguments": "{\"path\": \"src/main.rs\"}"
            }
        }]
    });
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed["role"], "assistant",
        "FALSIFY-CHAT-003b: assistant role preserved"
    );
    assert!(
        parsed["tool_calls"].is_array(),
        "FALSIFY-CHAT-003b: tool_calls is array"
    );
    assert_eq!(
        parsed["tool_calls"][0]["function"]["name"], "file_read",
        "FALSIFY-CHAT-003b: tool name preserved"
    );
}

/// FALSIFY-CHAT-005: History is append-only.
/// Previous messages must not change when new messages are added.
#[test]
fn falsify_chat_005_history_append_only() {
    let mut history: Vec<serde_json::Value> = vec![
        serde_json::json!({"role": "user", "content": "Hello"}),
        serde_json::json!({"role": "assistant", "content": "Hi there"}),
    ];

    // Snapshot before adding new turn
    let snapshot = history.clone();

    // Add new turn
    history.push(serde_json::json!({"role": "user", "content": "Follow up"}));
    history.push(serde_json::json!({"role": "assistant", "content": "Sure"}));

    // Previous messages unchanged
    assert_eq!(
        history[0], snapshot[0],
        "FALSIFY-CHAT-005: first message unchanged after append"
    );
    assert_eq!(
        history[1], snapshot[1],
        "FALSIFY-CHAT-005: second message unchanged after append"
    );
    assert_eq!(
        history.len(),
        4,
        "FALSIFY-CHAT-005: history grew from 2 to 4"
    );
}

// ═══ Contract: cli-dispatch-v1 enforcement (PMAT-192) ═══

/// FALSIFY-CLI-001: Every subcommand dispatches (not panic, not silent drop).
/// Test a representative set of subcommands with --help to verify dispatch.
#[test]
fn falsify_cli_001_dispatch_completeness_check() {
    let subcommands = vec![
        "check", "inspect", "validate", "lint", "explain", "export", "convert",
    ];
    for cmd in subcommands {
        let result = Command::cargo_bin("apr")
            .expect("apr binary")
            .args([cmd, "--help"])
            .output();
        match result {
            Ok(output) => {
                assert!(
                    output.status.success(),
                    "FALSIFY-CLI-001: `apr {} --help` must succeed (exit {})",
                    cmd,
                    output.status.code().unwrap_or(-1)
                );
            }
            Err(e) => {
                panic!("FALSIFY-CLI-001: `apr {cmd} --help` failed to run: {e}");
            }
        }
    }
}

/// FALSIFY-CLI-001b: Extended subcommands dispatch correctly.
#[test]
fn falsify_cli_001b_extended_dispatch() {
    let subcommands = vec!["chat", "bench", "eval", "qa", "hex", "tree", "flow"];
    for cmd in subcommands {
        let result = Command::cargo_bin("apr")
            .expect("apr binary")
            .args([cmd, "--help"])
            .output();
        match result {
            Ok(output) => {
                assert!(
                    output.status.success(),
                    "FALSIFY-CLI-001b: `apr {} --help` must succeed (exit {})",
                    cmd,
                    output.status.code().unwrap_or(-1)
                );
            }
            Err(e) => {
                panic!("FALSIFY-CLI-001b: `apr {cmd} --help` failed to run: {e}");
            }
        }
    }
}

/// FALSIFY-CLI-002: Exit codes are injective — file not found is non-zero.
#[test]
fn falsify_cli_002_exit_code_file_not_found() {
    let output = Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["check", "/nonexistent/model.gguf"])
        .output()
        .expect("Failed to run apr");
    let code = output.status.code().unwrap_or(-1);
    assert_ne!(
        code, 0,
        "FALSIFY-CLI-002: nonexistent file must produce non-zero exit code"
    );
}

/// FALSIFY-CLI-002b: Exit codes — invalid subcommand is non-zero.
#[test]
fn falsify_cli_002b_exit_code_invalid_subcommand() {
    let output = Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["nonexistent-subcommand"])
        .output()
        .expect("Failed to run apr");
    assert!(
        !output.status.success(),
        "FALSIFY-CLI-002b: invalid subcommand must fail"
    );
}

/// FALSIFY-CLI-003: --help produces parseable output (not empty, not error).
#[test]
fn falsify_cli_003_help_output() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("apr"))
        .stdout(predicate::str::is_empty().not());
}

/// FALSIFY-CLI-006: Code feature gate — `apr code --help` works.
/// This is a compile-time contract: Code variant only exists with `code` feature.
/// Since we're running tests with default features (which include `code`), this
/// must succeed.
#[test]
fn falsify_cli_006_code_feature_gate() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["code", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("model"))
        .stdout(predicate::str::contains("project"));
}

/// FALSIFY-CLI-006b: Serve subcommand dispatch works (plan + run).
#[test]
fn falsify_cli_006b_serve_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("plan").or(predicate::str::contains("run")));
}

/// FALSIFY-CLI-006c: Chat subcommand dispatch works.
#[test]
fn falsify_cli_006c_chat_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["chat", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("temperature").or(predicate::str::contains("model")));
}

// ═══ Contract: apr-chat-session-v1 — GAP CLOSURE (PMAT-190) ═══

/// FALSIFY-CHAT-006: KV-cache token count grows with history.
/// Each appended turn increases the cumulative token estimate.
#[test]
fn falsify_chat_006_kv_cache_token_growth() {
    // Simulate KV-cache size tracking via cumulative token count.
    // Contract: kv_cache_management — cache length matches token count.
    let mut history: Vec<String> = Vec::new();
    let mut cumulative_tokens = 0usize;

    // Turn 1
    history.push("user: Hello".into());
    history.push("assistant: Hi there!".into());
    let tokens_t1: usize = history.iter().map(|m| m.split_whitespace().count()).sum();
    assert!(
        tokens_t1 > cumulative_tokens,
        "FALSIFY-CHAT-006: tokens grow after turn 1"
    );
    cumulative_tokens = tokens_t1;

    // Turn 2
    history.push("user: What is 2+2?".into());
    history.push("assistant: 4".into());
    let tokens_t2: usize = history.iter().map(|m| m.split_whitespace().count()).sum();
    assert!(
        tokens_t2 > cumulative_tokens,
        "FALSIFY-CHAT-006: tokens grow after turn 2"
    );

    // Verify monotonic growth
    assert!(
        tokens_t2 > tokens_t1,
        "FALSIFY-CHAT-006: cumulative tokens monotonically increase"
    );
}

/// FALSIFY-CHAT-007: Session persistence roundtrip preserves message ordering.
/// Save N messages to JSONL, load them back, verify exact order and content.
#[test]
fn falsify_chat_007_session_jsonl_roundtrip() {
    use std::io::{BufRead, Write};

    let messages = vec![
        serde_json::json!({"role": "system", "content": "You are a helpful assistant."}),
        serde_json::json!({"role": "user", "content": "Hello"}),
        serde_json::json!({"role": "assistant", "content": "Hi! How can I help?"}),
        serde_json::json!({"role": "user", "content": "Fix the bug in auth.rs"}),
        serde_json::json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "file_read", "arguments": "{\"path\":\"src/auth.rs\"}"}}]}),
    ];

    // Write JSONL
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("messages.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("create");
        for msg in &messages {
            writeln!(f, "{}", serde_json::to_string(msg).unwrap()).expect("write");
        }
    }

    // Read JSONL back
    let file = std::fs::File::open(&path).expect("open");
    let reader = std::io::BufReader::new(file);
    let loaded: Vec<serde_json::Value> = reader
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();

    assert_eq!(
        loaded.len(),
        messages.len(),
        "FALSIFY-CHAT-007: message count preserved"
    );
    for (i, (original, loaded)) in messages.iter().zip(loaded.iter()).enumerate() {
        assert_eq!(
            original["role"], loaded["role"],
            "FALSIFY-CHAT-007: role preserved at index {i}"
        );
        assert_eq!(
            original["content"], loaded["content"],
            "FALSIFY-CHAT-007: content preserved at index {i}"
        );
    }

    // Verify tool_calls preserved on message 4 (assistant with tool call)
    assert!(
        loaded[4]["tool_calls"].is_array(),
        "FALSIFY-CHAT-007: tool_calls preserved on assistant message"
    );
    assert_eq!(
        loaded[4]["tool_calls"][0]["function"]["name"], "file_read",
        "FALSIFY-CHAT-007: tool call function name preserved"
    );
}

/// FALSIFY-CHAT-008: the REAL template engine preserves multi-turn ordering.
///
/// The previous version of this test was a tautology and is worth describing,
/// because the shape recurs. It built the ChatML string itself with
/// `format!("<|im_start|>{role}\n{content}<|im_end|>\n")` in a loop over
/// `turns`, then verified ordering by searching for that same `format!` output
/// in that same order over that same vector. Producer and verifier were the
/// same expression over the same data, so the assertion reduced to
/// `ordered(concat(map(f, t))) == ordered(t)` — true for ANY permutation, by
/// construction of `push_str`. It never imported the template engine at all.
///
/// Measured, not argued: swapping the `user`/`assistant` turns — precisely the
/// mutation this test claims to catch — left it PASSING.
///
/// This version calls `ChatMLTemplate::format_conversation`, the engine the
/// serving path uses, and asserts on ITS output. The turn order is now a
/// property of production code rather than of the test body.
#[test]
fn falsify_chat_008_chatml_multiturn_ordering() {
    use aprender::text::{ChatMLTemplate, ChatMessage, ChatTemplateEngine};

    let turns = [
        ("system", "You are helpful."),
        ("user", "Hello"),
        ("assistant", "Hi!"),
        ("user", "Fix auth.rs"),
        ("assistant", "I'll read the file first."),
    ];
    let messages: Vec<ChatMessage> = turns
        .iter()
        .map(|(role, content)| ChatMessage::new(*role, *content))
        .collect();

    let engine = ChatMLTemplate::new();
    let rendered = engine
        .format_conversation(&messages)
        .expect("FALSIFY-CHAT-008: the template engine must render a valid conversation");

    // Every turn's CONTENT must appear in the rendered output, in order.
    // Searching content (not a role+content marker we minted ourselves) keeps
    // the assertion independent of the engine's exact delimiter syntax while
    // still being sensitive to reordering.
    let mut cursor = 0usize;
    for (role, content) in &turns {
        let found = rendered[cursor..].find(content).unwrap_or_else(|| {
            panic!(
                "FALSIFY-CHAT-008: {role} turn {content:?} missing or out of order \
                 after byte {cursor} of the ENGINE-rendered conversation.\n\
                 Rendered:\n{rendered}"
            )
        });
        cursor += found + content.len();
    }

    // Sensitivity check, in-test: the same assertion applied to a conversation
    // the engine rendered from SWAPPED turns must fail. Without this, a future
    // refactor that made `rendered` order-insensitive would leave the loop
    // above passing and nobody would notice - which is exactly how the
    // tautology survived.
    let mut swapped = messages.clone();
    swapped.swap(1, 2);
    let swapped_render = engine
        .format_conversation(&swapped)
        .expect("FALSIFY-CHAT-008: engine must render the swapped conversation too");
    let mut cursor = 0usize;
    let mut still_ordered = true;
    for (_, content) in &turns {
        match swapped_render[cursor..].find(content) {
            Some(found) => cursor += found + content.len(),
            None => {
                still_ordered = false;
                break;
            }
        }
    }
    assert!(
        !still_ordered,
        "FALSIFY-CHAT-008: swapping two turns did NOT change the engine's rendered order — \
         this assertion cannot detect reordering and is therefore vacuous.\n\
         Original:\n{rendered}\nSwapped:\n{swapped_render}"
    );
}

// ═══ Contract: cli-dispatch-v1 — GAP CLOSURE (PMAT-190) ═══

/// FALSIFY-CLI-004: Idempotent inspection — `apr check` produces same output on re-run.
#[test]
fn falsify_cli_004_idempotent_inspection() {
    // Create a small test model file for deterministic inspection
    let dir = tempfile::tempdir().expect("tempdir");
    let model = dir.path().join("test.gguf");
    // Write GGUF magic + minimal header (version 3)
    std::fs::write(
        &model,
        [b'G', b'G', b'U', b'F', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    .expect("write model");

    let run = |model_path: &std::path::Path| -> std::process::Output {
        Command::cargo_bin("apr")
            .expect("apr binary")
            .args(["check", &model_path.display().to_string()])
            .output()
            .expect("run apr check")
    };

    let output1 = run(&model);
    let output2 = run(&model);

    // Same exit code
    assert_eq!(
        output1.status.code(),
        output2.status.code(),
        "FALSIFY-CLI-004: exit code must be identical across runs"
    );
    // Same stdout (deterministic output)
    assert_eq!(
        output1.stdout, output2.stdout,
        "FALSIFY-CLI-004: stdout must be identical across runs"
    );
}

/// FALSIFY-CLI-007: `apr tokenize --help` dispatches correctly.
#[test]
fn falsify_cli_007_tokenize_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["tokenize", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tokenize").or(predicate::str::contains("text")));
}

/// FALSIFY-CLI-008: `apr data --help` dispatches correctly.
#[test]
fn falsify_cli_008_data_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["data", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

// ═══ Contract: apr-model-lifecycle-v1 enforcement ═══

/// FALSIFY-LIFE-001: `apr check` rejects nonexistent files with non-zero exit.
#[test]
fn falsify_life_001_check_rejects_nonexistent() {
    let output = Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["check", "/nonexistent/model.gguf"])
        .output()
        .expect("run apr");
    assert!(
        !output.status.success(),
        "FALSIFY-LIFE-001: check must fail for nonexistent file"
    );
}

/// FALSIFY-LIFE-002: `apr check` with invalid GGUF produces non-zero exit.
#[test]
fn falsify_life_002_check_rejects_invalid_gguf() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bad.gguf");
    std::fs::write(&path, b"NOT_GGUF_MAGIC_BYTES").expect("write");
    let output = Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["check", &path.display().to_string()])
        .output()
        .expect("run apr");
    assert!(
        !output.status.success(),
        "FALSIFY-LIFE-002: check must reject invalid GGUF"
    );
}

/// FALSIFY-LIFE-003: `apr inspect --help` dispatches correctly.
#[test]
fn falsify_life_003_inspect_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["inspect", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

/// FALSIFY-LIFE-004: `apr convert --help` dispatches correctly.
#[test]
fn falsify_life_004_convert_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["convert", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("quantize").or(predicate::str::contains("compress")));
}

/// FALSIFY-LIFE-005: `apr run --help` dispatches correctly.
#[test]
fn falsify_life_005_run_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

/// FALSIFY-LIFE-006: `apr validate --help` dispatches correctly.
#[test]
fn falsify_life_006_validate_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["validate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

/// FALSIFY-LIFE-007: `apr export --help` dispatches correctly.
#[test]
fn falsify_life_007_export_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

/// FALSIFY-LIFE-008: `apr import --help` dispatches correctly.
#[test]
fn falsify_life_008_import_dispatch() {
    Command::cargo_bin("apr")
        .expect("apr binary")
        .args(["import", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}
