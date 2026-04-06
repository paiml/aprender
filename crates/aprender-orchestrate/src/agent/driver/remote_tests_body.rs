//! Body building and cost estimation tests for RemoteDriver.
//!
//! Extracted from `remote_tests.rs` for QA-002 (≤500 lines).
//! Covers: tool use messages, system message handling, request URL
//! construction, cost estimation, Anthropic tool result formatting.

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

#[test]
fn test_privacy_tier_is_standard() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    assert_eq!(driver.privacy_tier(), PrivacyTier::Standard);
}

#[test]
fn test_context_window() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    assert_eq!(driver.context_window(), 4096);
}

#[test]
fn test_openai_body_tool_use_messages() {
    let driver = RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::User("search for X".into()),
            Message::AssistantToolUse(ToolCall {
                id: "call_42".into(),
                name: "rag".into(),
                input: serde_json::json!({"query": "test"}),
            }),
            Message::ToolResult(crate::agent::driver::ToolResultMsg {
                tool_use_id: "call_42".into(),
                content: "found it".into(),
                is_error: false,
            }),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: Some("You are helpful.".into()),
    };

    let body = driver.build_openai_body(&request);
    let msgs = body["messages"].as_array().expect("msgs");
    // system + user + assistant(tool_calls) + tool result = 4
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[2]["role"], "assistant");
    assert!(msgs[2]["tool_calls"].is_array());
    assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "rag");
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!(msgs[3]["tool_call_id"], "call_42");
    assert_eq!(msgs[3]["content"], "found it");
}

#[test]
fn test_openai_body_system_message_in_messages() {
    let driver = RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::System("Extra system context.".into()),
            Message::User("hello".into()),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: Some("Main system prompt.".into()),
    };

    let body = driver.build_openai_body(&request);
    let msgs = body["messages"].as_array().expect("msgs");
    // request.system (1) + System message (1) + User (1) = 3
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "Main system prompt.");
    assert_eq!(msgs[1]["role"], "system");
    assert_eq!(msgs[1]["content"], "Extra system context.");
    assert_eq!(msgs[2]["role"], "user");
}

#[test]
fn test_anthropic_body_system_message_filtered() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![Message::System("filtered out".into()), Message::User("hello".into())],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: Some("system".into()),
    };

    let body = driver.build_anthropic_body(&request);
    let msgs = body["messages"].as_array().expect("msgs");
    // System messages are filtered out for Anthropic
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
}

#[test]
fn test_anthropic_body_no_system() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![Message::User("hi".into())],
        tools: vec![],
        max_tokens: 512,
        temperature: 0.0,
        system: None,
    };

    let body = driver.build_anthropic_body(&request);
    assert!(body.get("system").is_none());
}

#[test]
fn test_build_request_anthropic_url() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let (url, body) = driver.build_request(&test_request());
    assert!(url.ends_with("/v1/messages"));
    assert_eq!(body["model"], "test-model");
}

#[test]
fn test_build_request_openai_url() {
    let driver = RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let (url, body) = driver.build_request(&test_request());
    assert!(url.ends_with("/v1/chat/completions"));
    assert_eq!(body["model"], "test-model");
}

#[test]
fn test_estimate_cost() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let usage =
        crate::agent::result::TokenUsage { input_tokens: 1_000_000, output_tokens: 100_000 };
    let cost = driver.estimate_cost(&usage);
    // input: 1M * 3/1M = $3.0
    // output: 100K * 15/1M = $1.5
    let expected = 3.0 + 1.5;
    assert!((cost - expected).abs() < 0.001, "cost {cost} != expected {expected}");
}

#[test]
fn test_estimate_cost_zero() {
    let driver = RemoteDriver::new(test_config(ApiProvider::OpenAi));
    let usage = crate::agent::result::TokenUsage { input_tokens: 0, output_tokens: 0 };
    assert_eq!(driver.estimate_cost(&usage), 0.0);
}

#[test]
fn test_anthropic_tool_result_messages() {
    let driver = RemoteDriver::new(test_config(ApiProvider::Anthropic));
    let request = CompletionRequest {
        model: "test".into(),
        messages: vec![
            Message::User("search".into()),
            Message::AssistantToolUse(ToolCall {
                id: "1".into(),
                name: "rag".into(),
                input: serde_json::json!({"query": "test"}),
            }),
            Message::ToolResult(crate::agent::driver::ToolResultMsg {
                tool_use_id: "1".into(),
                content: "found it".into(),
                is_error: false,
            }),
        ],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.5,
        system: None,
    };

    let body = driver.build_anthropic_body(&request);
    let msgs = body["messages"].as_array().expect("msgs");
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[1]["role"], "assistant");
    assert!(msgs[1]["content"][0]["type"].as_str().expect("type").contains("tool_use"));
    assert_eq!(msgs[2]["role"], "user");
}
