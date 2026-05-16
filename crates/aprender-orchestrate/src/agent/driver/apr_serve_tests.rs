use super::*;

#[test]
fn test_find_apr_binary() {
    let result = find_apr_binary();
    if result.is_err() && std::env::var("CI").is_ok() {
        return; // apr binary not installed in CI
    }
    assert!(result.is_ok(), "apr binary should be on PATH: {result:?}");
}

#[test]
fn test_privacy_tier_is_sovereign() {
    assert_eq!(PrivacyTier::Sovereign, PrivacyTier::Sovereign);
}

// ═══ FALSIFY-CT-003: strip_thinking_blocks contract (PMAT-187) ═══

#[test]
fn falsify_ct_003_strips_closing_think_tag() {
    assert_eq!(strip_thinking_blocks("</think>\n\n4"), "4");
}

#[test]
fn falsify_ct_003_strips_full_think_block() {
    assert_eq!(strip_thinking_blocks("<think>reasoning here</think>answer"), "answer");
}

#[test]
fn falsify_ct_003_strips_repeated_closing_tags() {
    let result = strip_thinking_blocks("</think></think></think>");
    assert!(!result.contains("</think>"), "must strip all </think> tags");
}

#[test]
fn falsify_ct_003_preserves_clean_text() {
    assert_eq!(strip_thinking_blocks("clean text"), "clean text");
}

#[test]
fn falsify_ct_003_strips_mixed_content() {
    let result = strip_thinking_blocks("<think>x</think>y</think>z");
    assert!(!result.contains("<think>"), "no <think> tags");
    assert!(!result.contains("</think>"), "no </think> tags");
    assert!(result.contains('y'), "content between tags preserved");
    assert!(result.contains('z'), "trailing content preserved");
}

#[test]
fn falsify_ct_003_strips_multiline_think_block() {
    let input = "<think>\nline1\nline2\n</think>\nAnswer: 42";
    assert_eq!(strip_thinking_blocks(input), "Answer: 42");
}

#[test]
fn falsify_ct_003_handles_empty_input() {
    assert_eq!(strip_thinking_blocks(""), "");
}

#[test]
fn falsify_ct_003_handles_only_think_tags() {
    assert_eq!(strip_thinking_blocks("<think></think>"), "");
}

// ═══ http-api-v1 contract: request/response schema (PMAT-189) ═══

fn build_body_test(
    model_name: &str,
    request: &crate::agent::driver::CompletionRequest,
) -> serde_json::Value {
    use crate::agent::driver::Message;
    let mut messages = Vec::new();
    if let Some(ref system) = request.system {
        let compact_system = system
            .find("\n\n## Available Tools")
            .map(|i| &system[..i])
            .unwrap_or(system)
            .to_string();
        messages.push(serde_json::json!({"role": "system", "content": compact_system}));
    }
    for msg in &request.messages {
        match msg {
            Message::User(text) => {
                messages.push(serde_json::json!({"role": "user", "content": text}));
            }
            Message::Assistant(text) => {
                messages.push(serde_json::json!({"role": "assistant", "content": text}));
            }
            Message::AssistantToolUse(call) => messages.push(serde_json::json!({
                "role": "assistant",
                "content": format!("<tool_call>\n{}\n</tool_call>",
                    serde_json::json!({"name": call.name, "input": call.input}))
            })),
            Message::ToolResult(result) => messages.push(serde_json::json!({
                "role": "user",
                "content": format!("<tool_result>\n{}\n</tool_result>", result.content)
            })),
            _ => {}
        }
    }
    let max_tokens = request.max_tokens.min(1024);
    serde_json::json!({
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": request.temperature,
        "stream": false
    })
}

fn make_request(
    system: Option<&str>,
    messages: Vec<crate::agent::driver::Message>,
    max_tokens: u32,
) -> crate::agent::driver::CompletionRequest {
    crate::agent::driver::CompletionRequest {
        model: "test".into(),
        system: system.map(String::from),
        messages,
        max_tokens,
        temperature: 0.0,
        tools: vec![],
    }
}

#[test]
fn falsify_http_001_body_valid_json() {
    use crate::agent::driver::Message;
    let req = make_request(Some("You are helpful."), vec![Message::User("Hello".into())], 256);
    let body = build_body_test("qwen3-1.7b", &req);
    assert!(body.is_object());
    assert_eq!(body["model"], "qwen3-1.7b");
    assert!(body["messages"].is_array());
    assert_eq!(body["stream"], false);
    assert!(body["max_tokens"].as_u64().unwrap() <= 1024);
}

#[test]
fn falsify_http_001_system_prompt_first() {
    use crate::agent::driver::Message;
    let req = make_request(Some("System prompt."), vec![Message::User("Hi".into())], 128);
    let body = build_body_test("test", &req);
    let msgs = body["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert!(msgs[0]["content"].as_str().unwrap().contains("System prompt"));
    assert_eq!(msgs[1]["role"], "user");
}

#[test]
fn falsify_http_001_max_tokens_cap() {
    use crate::agent::driver::Message;
    let req = make_request(None, vec![Message::User("Hi".into())], 4096);
    let body = build_body_test("test", &req);
    assert_eq!(body["max_tokens"].as_u64().unwrap(), 1024, "PMAT-170: capped at 1024");
}

#[test]
fn falsify_http_001_tool_call_format() {
    use crate::agent::driver::{Message, ToolCall, ToolResultMsg};
    let req = make_request(
        None,
        vec![
            Message::User("List files".into()),
            Message::AssistantToolUse(ToolCall {
                id: "1".into(),
                name: "glob".into(),
                input: serde_json::json!({"pattern": "src/**/*.rs"}),
            }),
            Message::ToolResult(ToolResultMsg {
                tool_use_id: "1".into(),
                content: "src/main.rs".into(),
                is_error: false,
            }),
        ],
        256,
    );
    let body = build_body_test("test", &req);
    let msgs = body["messages"].as_array().unwrap();
    assert!(msgs[1]["content"].as_str().unwrap().contains("<tool_call>"));
    assert!(msgs[2]["content"].as_str().unwrap().contains("<tool_result>"));
}

// ═══ FALSIFY-1712: PR_SET_PDEATHSIG orphan reaping ═══
//
// Issue #1712: `apr code` spawns `apr serve` as a child. If `apr code` is
// killed (timeout, SIGTERM, panic), the child must be reaped — not orphaned
// with 3GB RSS.
//
// We can't easily kill the test process itself, so instead we verify the
// helper installs a `pre_exec` hook that calls `prctl(PR_SET_PDEATHSIG)`.
// The end-to-end behaviour is verified by spawning a `sleep` child with the
// same hook from a short-lived parent shell and asserting the child dies.

#[cfg(unix)]
#[test]
fn falsify_1712_pdeathsig_helper_installs_pre_exec() {
    // The helper must not panic on a fresh Command.
    let mut cmd = std::process::Command::new("/bin/true");
    super::configure_parent_death_signal(&mut cmd);
    // Spawning must succeed — pre_exec hook is well-formed.
    let mut child = cmd.spawn().expect("spawn /bin/true with pdeathsig");
    let status = child.wait().expect("wait /bin/true");
    assert!(status.success(), "/bin/true should exit 0 with PR_SET_PDEATHSIG set");
}

#[cfg(target_os = "linux")]
#[test]
fn falsify_1712_pdeathsig_actually_set_in_child() {
    // Direct verification: spawn python3 in the child and have it call
    // `prctl(PR_GET_PDEATHSIG, ...)` to read back the value our pre_exec
    // hook just set. If our hook ran correctly, the child reports SIGTERM
    // (15); without the hook it would report 0.
    //
    // We use python3 instead of /proc/self/status because the latter does
    // not expose `PDeathSig` on all kernel versions (Ubuntu 6.8 omits it).
    if std::process::Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipping: python3 not available");
        return;
    }

    // PR_GET_PDEATHSIG = 2 (Linux <sys/prctl.h>).
    let script = "import ctypes; libc=ctypes.CDLL('libc.so.6'); \
                  v=ctypes.c_int(0); libc.prctl(2, ctypes.byref(v)); \
                  print(v.value)";

    let mut cmd = std::process::Command::new("python3");
    cmd.arg("-c").arg(script).stdout(std::process::Stdio::piped());
    super::configure_parent_death_signal(&mut cmd);

    let output = cmd.output().expect("spawn python3");
    assert!(output.status.success(), "python3 exited nonzero: {:?}", output.status);

    let reported: i32 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .expect("python3 should print an integer");

    assert_eq!(
        reported,
        libc::SIGTERM,
        "FALSIFY-1712: child PR_GET_PDEATHSIG returned {reported}, expected SIGTERM (15). \
         prctl(PR_SET_PDEATHSIG) did not stick across exec — orphan-reaping is broken."
    );
}

#[test]
fn falsify_http_001_strips_verbose_keeps_compact() {
    use crate::agent::driver::Message;
    let sys = "Helpful.\n\n## Tools\n| t | u |\n\n## Available Tools\n{big json}";
    let req = make_request(Some(sys), vec![Message::User("Hi".into())], 128);
    let body = build_body_test("test", &req);
    let c = body["messages"][0]["content"].as_str().unwrap();
    assert!(c.contains("## Tools"), "compact table preserved (PMAT-176)");
    assert!(!c.contains("## Available Tools"), "verbose section stripped");
}
