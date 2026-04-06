//! Response parsing tests for RemoteDriver (Anthropic + OpenAI API).
//!
//! Covers: body format construction, response parsing for both API
//! providers, stop reason mapping, tool call extraction.
//! See `remote_tests_body.rs` for body building edge cases and cost.

use super::*;
use crate::agent::driver::{Message, ToolDefinition};

fn test_config(provider: ApiProvider) -> RemoteDriverConfig {
    RemoteDriverConfig {
        base_url: "https://api.example.com".into(),
        api_key: "test-key".into(),
        model: "test-model".into(),
        provider,
        context_window: 4096,
    }
}

fn test_request() -> CompletionRequest {
    CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::User("Hello".into()),
            Message::Assistant("Hi there".into()),
            Message::User("What is 2+2?".into()),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: Some("You are helpful.".into()),
    }
}

fn request_with_tools() -> CompletionRequest {
    CompletionRequest {
        model: "test".into(),
        messages: vec![Message::User("Search for X".into())],
        tools: vec![ToolDefinition {
            name: "rag".into(),
            description: "Search docs".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        }],
        max_tokens: 1024,
        temperature: 0.5,
        system: None,
    }
}

#[test]
fn test_anthropic_body_format() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let body = driver.build_anthropic_body(&test_request());

    assert_eq!(body["model"], "test-model");
    assert_eq!(body["max_tokens"], 1024);
    assert_eq!(body["system"], "You are helpful.");
    let msgs = body["messages"].as_array().expect("msgs");
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[1]["role"], "assistant");
}

#[test]
fn test_openai_body_format() {
    let driver = RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let body = driver.build_openai_body(&test_request());

    assert_eq!(body["model"], "test-model");
    let msgs = body["messages"].as_array().expect("msgs");
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "You are helpful.");
    assert_eq!(msgs.len(), 4); // system + 3 user/assistant
}

#[test]
fn test_anthropic_body_with_tools() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let body = driver.build_anthropic_body(&request_with_tools());

    let tools = body["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "rag");
}

#[test]
fn test_openai_body_with_tools() {
    let driver = RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let body = driver.build_openai_body(&request_with_tools());

    let tools = body["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "rag");
}

#[test]
fn test_parse_anthropic_text_response() {
    let body = serde_json::json!({
        "content": [
            {"type": "text", "text": "Hello!"}
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.text, "Hello!");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert!(resp.tool_calls.is_empty());
    assert_eq!(resp.usage.input_tokens, 100);
    assert_eq!(resp.usage.output_tokens, 50);
}

#[test]
fn test_parse_anthropic_tool_use() {
    let body = serde_json::json!({
        "content": [
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "rag",
                "input": {"query": "SIMD"}
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 200,
            "output_tokens": 30
        }
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "rag");
    assert_eq!(resp.tool_calls[0].id, "tool_1");
}

#[test]
fn test_parse_openai_text_response() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 20
        }
    });

    let resp = remote_stream::parse_openai_response(&body);
    assert_eq!(resp.text, "Hello!");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert!(resp.tool_calls.is_empty());
    assert_eq!(resp.usage.input_tokens, 80);
    assert_eq!(resp.usage.output_tokens, 20);
}

#[test]
fn test_parse_openai_tool_calls() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "rag",
                        "arguments": "{\"query\": \"test\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 40
        }
    });

    let resp = remote_stream::parse_openai_response(&body);
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "rag");
    assert_eq!(resp.tool_calls[0].input, serde_json::json!({"query": "test"}));
}

#[test]
fn test_parse_anthropic_stop_sequence() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "partial"}],
        "stop_reason": "stop_sequence",
        "usage": {"input_tokens": 50, "output_tokens": 20}
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.stop_reason, StopReason::StopSequence);
}

#[test]
fn test_parse_anthropic_mixed_content() {
    let body = serde_json::json!({
        "content": [
            {"type": "text", "text": "I'll search for that."},
            {
                "type": "tool_use",
                "id": "tool_2",
                "name": "rag",
                "input": {"query": "SIMD"}
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.text, "I'll search for that.");
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.stop_reason, StopReason::ToolUse);
}

#[test]
fn test_parse_anthropic_empty_content() {
    let body = serde_json::json!({
        "content": [],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 0}
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert!(resp.text.is_empty());
    assert!(resp.tool_calls.is_empty());
}

#[test]
fn test_parse_openai_null_content() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 0
        }
    });

    let resp = remote_stream::parse_openai_response(&body);
    assert_eq!(resp.text, "");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
}

#[test]
fn test_parse_openai_length_finish() {
    let body = serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "truncated output"
            },
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 4096
        }
    });

    let resp = remote_stream::parse_openai_response(&body);
    assert_eq!(resp.stop_reason, StopReason::MaxTokens);
    assert_eq!(resp.text, "truncated output");
}

#[test]
fn test_parse_anthropic_max_tokens() {
    let body = serde_json::json!({
        "content": [{"type": "text", "text": "partial"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 50, "output_tokens": 4096}
    });

    let resp = remote_stream::parse_anthropic_response(&body);
    assert_eq!(resp.stop_reason, StopReason::MaxTokens);
}
