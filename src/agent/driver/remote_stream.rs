//! SSE streaming parsers for Anthropic and OpenAI APIs.
//!
//! Extracted from `remote.rs` for QA-002 (≤500 lines).
//! Handles `content_block_start/delta/stop` (Anthropic) and
//! `choices[0].delta` (OpenAI) Server-Sent Event formats.

use crate::agent::driver::{CompletionResponse, StreamEvent, ToolCall};
use crate::agent::result::{StopReason, TokenUsage};

/// Parse Anthropic Messages API response.
pub(super) fn parse_anthropic_response(body: &serde_json::Value) -> CompletionResponse {
    let stop_reason = match body["stop_reason"].as_str().unwrap_or("end_turn") {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        _ => StopReason::EndTurn,
    };

    let mut text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(content) = body["content"].as_array() {
        for block in content {
            match block["type"].as_str() {
                Some("text") => {
                    if let Some(t) = block["text"].as_str() {
                        text.push_str(t);
                    }
                }
                Some("tool_use") => {
                    tool_calls.push(ToolCall {
                        id: block["id"].as_str().unwrap_or("unknown").to_string(),
                        name: block["name"].as_str().unwrap_or("").to_string(),
                        input: block["input"].clone(),
                    });
                }
                _ => {}
            }
        }
    }

    let usage = TokenUsage {
        input_tokens: body["usage"]["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: body["usage"]["output_tokens"].as_u64().unwrap_or(0),
    };

    CompletionResponse { text, stop_reason, tool_calls, usage }
}

/// Parse `OpenAI` Chat Completions response.
pub(super) fn parse_openai_response(body: &serde_json::Value) -> CompletionResponse {
    let choice = &body["choices"][0];
    let message = &choice["message"];

    let stop_reason = match choice["finish_reason"].as_str().unwrap_or("stop") {
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    };

    let text = message["content"].as_str().unwrap_or("").to_string();

    let mut tool_calls = Vec::new();
    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            let input: serde_json::Value = call["function"]["arguments"]
                .as_str()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(serde_json::json!({}));

            tool_calls.push(ToolCall {
                id: call["id"].as_str().unwrap_or("unknown").to_string(),
                name: call["function"]["name"].as_str().unwrap_or("").to_string(),
                input,
            });
        }
    }

    let usage = TokenUsage {
        input_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        output_tokens: body["usage"]["completion_tokens"].as_u64().unwrap_or(0),
    };

    CompletionResponse { text, stop_reason, tool_calls, usage }
}

/// Process a single Anthropic SSE event.
///
/// Accumulates text deltas, tool calls (partial JSON), usage,
/// and stop reason from the Anthropic Messages streaming API.
pub(super) async fn process_anthropic_event(
    event: &serde_json::Value,
    full_text: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    usage: &mut TokenUsage,
    stop_reason: &mut StopReason,
    current_tool: &mut Option<(String, String, String)>,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) {
    let event_type = event["type"].as_str().unwrap_or("");
    match event_type {
        "content_block_start" => {
            let block = &event["content_block"];
            if block["type"].as_str() == Some("tool_use") {
                let id = block["id"].as_str().unwrap_or("").to_string();
                let name = block["name"].as_str().unwrap_or("").to_string();
                *current_tool = Some((id, name, String::new()));
            }
        }
        "content_block_delta" => {
            let delta = &event["delta"];
            if let Some(text) = delta["text"].as_str() {
                full_text.push_str(text);
                let _ = tx.send(StreamEvent::TextDelta { text: text.to_string() }).await;
            }
            if let Some(json) = delta["partial_json"].as_str() {
                if let Some((_, _, ref mut accum)) = current_tool {
                    accum.push_str(json);
                }
            }
        }
        "content_block_stop" => {
            if let Some((id, name, json_str)) = current_tool.take() {
                let input = serde_json::from_str(&json_str).unwrap_or(serde_json::json!({}));
                tool_calls.push(ToolCall { id, name, input });
            }
        }
        "message_delta" => {
            if let Some(sr) = event["delta"]["stop_reason"].as_str() {
                *stop_reason = match sr {
                    "tool_use" => StopReason::ToolUse,
                    "max_tokens" => StopReason::MaxTokens,
                    "stop_sequence" => StopReason::StopSequence,
                    _ => StopReason::EndTurn,
                };
            }
            if let Some(out) = event["usage"]["output_tokens"].as_u64() {
                usage.output_tokens = out;
            }
        }
        "message_start" => {
            if let Some(inp) = event["message"]["usage"]["input_tokens"].as_u64() {
                usage.input_tokens = inp;
            }
        }
        _ => {}
    }
}

/// Process a single OpenAI SSE event.
///
/// Accumulates text deltas, tool calls (indexed partial JSON),
/// usage, and stop reason from the OpenAI Chat Completions
/// streaming API.
pub(super) async fn process_openai_event(
    event: &serde_json::Value,
    full_text: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    usage: &mut TokenUsage,
    stop_reason: &mut StopReason,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
) {
    let choice = &event["choices"][0];
    let delta = &choice["delta"];

    if let Some(text) = delta["content"].as_str() {
        full_text.push_str(text);
        let _ = tx.send(StreamEvent::TextDelta { text: text.to_string() }).await;
    }

    if let Some(calls) = delta["tool_calls"].as_array() {
        for call in calls {
            accumulate_openai_tool_call(call, tool_calls);
        }
    }

    if let Some(fr) = choice["finish_reason"].as_str() {
        *stop_reason = match fr {
            "tool_calls" => StopReason::ToolUse,
            "length" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };
    }

    if let Some(u) = event.get("usage") {
        if let Some(inp) = u["prompt_tokens"].as_u64() {
            usage.input_tokens = inp;
        }
        if let Some(out) = u["completion_tokens"].as_u64() {
            usage.output_tokens = out;
        }
    }
}

/// Accumulate a single OpenAI tool call delta into the tool list.
fn accumulate_openai_tool_call(call: &serde_json::Value, tool_calls: &mut Vec<ToolCall>) {
    let idx = call["index"].as_u64().unwrap_or(0) as usize;
    while tool_calls.len() <= idx {
        tool_calls.push(ToolCall {
            id: String::new(),
            name: String::new(),
            input: serde_json::json!({}),
        });
    }
    if let Some(id) = call["id"].as_str() {
        tool_calls[idx].id = id.to_string();
    }
    if let Some(name) = call["function"]["name"].as_str() {
        tool_calls[idx].name = name.to_string();
    }
    if let Some(args) = call["function"]["arguments"].as_str() {
        let existing = tool_calls[idx].input.as_str().unwrap_or("");
        let combined = format!("{existing}{args}");
        tool_calls[idx].input =
            serde_json::from_str(&combined).unwrap_or(serde_json::json!(combined));
    }
}

#[cfg(test)]
#[path = "remote_stream_tests.rs"]
mod tests;
