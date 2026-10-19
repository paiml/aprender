//! PMAT-801 follow-up: an OpenAI-compatible client (e.g. the `pmcp-agent` SDK loop)
//! omits `content` on the assistant tool-call turn it echoes back in the next
//! request, which the OpenAI API itself permits. The request side used to answer
//! that body with HTTP 422 (`missing field \`content\``).
//!
//! Scope of the tolerance, pinned here from both sides:
//! - an ASSISTANT message with `content` absent or `null` reads as `""`;
//! - every other role (user, system, tool) still REQUIRES `content`;
//! - a non-string `content` is still a type error;
//! - serialization is unchanged (the response-side bytes are a contract).
//!
//! The fixture is request #3 of the spike-029 wire log
//! (`results/wire-0.8b-plain.jsonl` line 4, the turn-2 body that returned 422).
#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

use super::*;

fn wire_fixture_request() -> serde_json::Value {
    serde_json::json!({
        "model": "Qwen3.5-0.8B-Q4_K_M",
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant. Use tools when helpful."},
            {"role": "user", "content": "What's the weather in Paris right now?"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\",\"unit\":\"celsius\"}"
                }}
            ]},
            {"role": "tool", "tool_call_id": "call_0",
             "content": "{\"city\":\"Paris\",\"temperature_c\":17,\"condition\":\"light rain\"}"}
        ],
        "max_tokens": 384
    })
}

fn parse(v: serde_json::Value) -> Result<ChatCompletionRequest, serde_json::Error> {
    serde_json::from_value::<ChatCompletionRequest>(v)
}

#[test]
fn assistant_message_without_content_parses_pmat801_followup() {
    let req = parse(wire_fixture_request()).expect("turn-2 body must parse");
    assert_eq!(req.messages.len(), 4);
    assert_eq!(req.messages[2].role, "assistant");
    assert_eq!(req.messages[2].content, "");
    let calls = req.messages[2].tool_calls.as_ref().expect("tool_calls survive");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[0].function.arguments, "{\"city\":\"Paris\",\"unit\":\"celsius\"}");
    assert_eq!(req.messages[3].role, "tool");
    assert_eq!(req.messages[3].tool_call_id.as_deref(), Some("call_0"));
}

#[test]
fn assistant_message_with_null_content_parses() {
    let mut body = wire_fixture_request();
    body["messages"][2]["content"] = serde_json::Value::Null;
    let req = parse(body).expect("null assistant content must parse");
    assert_eq!(req.messages[2].content, "");
    assert!(req.messages[2].tool_calls.is_some());
}

#[test]
fn chat_message_serialization_is_byte_identical() {
    let assistant = ChatMessage {
        role: "assistant".to_string(),
        content: String::new(),
        name: None,
        tool_calls: Some(vec![ResponseToolCall {
            id: "call_0".to_string(),
            call_type: "function".to_string(),
            function: ResponseFunctionCall {
                name: "get_weather".to_string(),
                arguments: "{\"city\":\"Paris\"}".to_string(),
            },
        }]),
        tool_call_id: None,
    };
    assert_eq!(
        serde_json::to_string(&assistant).unwrap(),
        r#"{"role":"assistant","content":"","tool_calls":[{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}"#
    );
    let user = ChatMessage {
        role: "user".to_string(),
        content: "x".to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    };
    assert_eq!(serde_json::to_string(&user).unwrap(), r#"{"role":"user","content":"x"}"#);
}

#[test]
fn numeric_content_is_still_refused() {
    for role in ["assistant", "user", "system", "tool"] {
        let body = serde_json::json!({"model": "m", "messages": [{"role": role, "content": 5}]});
        assert!(parse(body).is_err(), "numeric content must stay a type error for role {role}");
    }
}

#[test]
fn non_assistant_message_without_content_is_refused() {
    for role in ["user", "system", "tool"] {
        let absent = serde_json::json!({"model": "m", "messages": [{"role": role}]});
        let err = parse(absent).expect_err("absent content must be refused").to_string();
        assert!(err.contains("content"), "role {role}, absent: got {err}");
        let null = serde_json::json!({"model": "m", "messages": [{"role": role, "content": null}]});
        let err = parse(null).expect_err("null content must be refused").to_string();
        assert!(err.contains("content"), "role {role}, null: got {err}");
        let present = serde_json::json!({"model": "m", "messages": [{"role": role, "content": "x"}]});
        assert_eq!(parse(present).expect("present content parses").messages[0].content, "x");
    }
}
