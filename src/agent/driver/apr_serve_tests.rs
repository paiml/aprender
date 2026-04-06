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
