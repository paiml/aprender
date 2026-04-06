//! Tests for SSE streaming parsers (Anthropic + OpenAI).

use super::*;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_anthropic_text_delta() {
    let event = serde_json::json!({
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "Hello"}
    });

    let (tx, mut rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    process_anthropic_event(
        &event,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    assert_eq!(text, "Hello");
    let evt = rx.try_recv().expect("expected event");
    assert!(matches!(evt, StreamEvent::TextDelta { text } if text == "Hello"));
}

#[tokio::test]
async fn test_anthropic_tool_use_stream() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    // content_block_start with tool_use
    let start = serde_json::json!({
        "type": "content_block_start",
        "content_block": {
            "type": "tool_use",
            "id": "tool_abc",
            "name": "rag"
        }
    });
    process_anthropic_event(
        &start,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;
    assert!(current_tool.is_some());

    // partial JSON delta
    let delta = serde_json::json!({
        "type": "content_block_delta",
        "delta": {"partial_json": "{\"query\":\"test\"}"}
    });
    process_anthropic_event(
        &delta,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    // content_block_stop finalizes tool call
    let stop_evt = serde_json::json!({
        "type": "content_block_stop"
    });
    process_anthropic_event(
        &stop_evt,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "tool_abc");
    assert_eq!(tool_calls[0].name, "rag");
    assert_eq!(tool_calls[0].input, serde_json::json!({"query": "test"}));
    assert!(current_tool.is_none());
}

#[tokio::test]
async fn test_anthropic_message_start_usage() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    let event = serde_json::json!({
        "type": "message_start",
        "message": {
            "usage": {"input_tokens": 250}
        }
    });
    process_anthropic_event(
        &event,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    assert_eq!(usage.input_tokens, 250);
}

#[tokio::test]
async fn test_anthropic_message_delta_stop() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    let event = serde_json::json!({
        "type": "message_delta",
        "delta": {"stop_reason": "max_tokens"},
        "usage": {"output_tokens": 4096}
    });
    process_anthropic_event(
        &event,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    assert_eq!(stop, StopReason::MaxTokens);
    assert_eq!(usage.output_tokens, 4096);
}

#[tokio::test]
async fn test_openai_text_delta() {
    let event = serde_json::json!({
        "choices": [{
            "delta": {"content": "World"},
            "index": 0
        }]
    });

    let (tx, mut rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;

    process_openai_event(&event, &mut text, &mut tool_calls, &mut usage, &mut stop, &tx).await;

    assert_eq!(text, "World");
    let evt = rx.try_recv().expect("expected event");
    assert!(matches!(evt, StreamEvent::TextDelta { text } if text == "World"));
}

#[tokio::test]
async fn test_openai_tool_call_stream() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;

    let event = serde_json::json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": {
                        "name": "rag",
                        "arguments": "{\"query\":\"test\"}"
                    }
                }]
            },
            "index": 0
        }]
    });

    process_openai_event(&event, &mut text, &mut tool_calls, &mut usage, &mut stop, &tx).await;

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_1");
    assert_eq!(tool_calls[0].name, "rag");
}

#[tokio::test]
async fn test_openai_finish_reason() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;

    let event = serde_json::json!({
        "choices": [{
            "delta": {},
            "finish_reason": "length",
            "index": 0
        }]
    });

    process_openai_event(&event, &mut text, &mut tool_calls, &mut usage, &mut stop, &tx).await;

    assert_eq!(stop, StopReason::MaxTokens);
}

#[tokio::test]
async fn test_openai_usage_event() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;

    let event = serde_json::json!({
        "choices": [{"delta": {}, "index": 0}],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 80
        }
    });

    process_openai_event(&event, &mut text, &mut tool_calls, &mut usage, &mut stop, &tx).await;

    assert_eq!(usage.input_tokens, 150);
    assert_eq!(usage.output_tokens, 80);
}

#[tokio::test]
async fn test_anthropic_unknown_event_type_ignored() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    let event = serde_json::json!({
        "type": "ping"
    });
    process_anthropic_event(
        &event,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    assert!(text.is_empty());
    assert!(tool_calls.is_empty());
}

#[tokio::test]
async fn test_anthropic_tool_use_stop_reason() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    let event = serde_json::json!({
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"output_tokens": 100}
    });
    process_anthropic_event(
        &event,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    assert_eq!(stop, StopReason::ToolUse);
}

#[tokio::test]
async fn test_anthropic_stop_sequence_reason() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    let event = serde_json::json!({
        "type": "message_delta",
        "delta": {"stop_reason": "stop_sequence"},
        "usage": {"output_tokens": 42}
    });
    process_anthropic_event(
        &event,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    assert_eq!(stop, StopReason::StopSequence);
    assert_eq!(usage.output_tokens, 42);
}

#[tokio::test]
async fn test_anthropic_content_block_start_text() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    // content_block_start with text type (not tool_use)
    let event = serde_json::json!({
        "type": "content_block_start",
        "content_block": {
            "type": "text",
            "text": ""
        }
    });
    process_anthropic_event(
        &event,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    // Should not set current_tool for text blocks
    assert!(current_tool.is_none());
}

#[tokio::test]
async fn test_anthropic_content_block_stop_no_tool() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;
    let mut current_tool = None;

    // content_block_stop without prior tool start
    let event = serde_json::json!({
        "type": "content_block_stop"
    });
    process_anthropic_event(
        &event,
        &mut text,
        &mut tool_calls,
        &mut usage,
        &mut stop,
        &mut current_tool,
        &tx,
    )
    .await;

    // No tool calls should be created
    assert!(tool_calls.is_empty());
}

#[tokio::test]
async fn test_openai_tool_calls_stop_reason() {
    let (tx, _rx) = mpsc::channel(16);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = TokenUsage { input_tokens: 0, output_tokens: 0 };
    let mut stop = StopReason::EndTurn;

    let event = serde_json::json!({
        "choices": [{
            "delta": {},
            "finish_reason": "tool_calls",
            "index": 0
        }]
    });
    process_openai_event(&event, &mut text, &mut tool_calls, &mut usage, &mut stop, &tx).await;

    assert_eq!(stop, StopReason::ToolUse);
}
