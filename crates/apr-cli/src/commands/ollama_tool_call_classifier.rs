//! Ollama `/api/chat` **function-calling** response-shape classifier (CRUX-I-04).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-I-04-{001,002,003,005}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! already-parsed Ollama chat responses that carry `message.tool_calls[]`:
//!
//!   * `classify_tool_call_schema` — non-streaming response: `message` is an
//!     object, `message.tool_calls` is a non-empty array, each element has
//!     `function.name: string` and `function.arguments: object` (NOT a
//!     stringified JSON blob — the most common ollama/OpenAI drift bug).
//!   * `classify_tool_name_allowlist` — every called `function.name` appears
//!     in the request's declared `tools[*].function.name` set (no model
//!     hallucinated tool names).
//!   * `classify_streaming_tool_call` — in NDJSON streams with tool calls,
//!     exactly one `done == true` frame as the last frame, `tool_calls`
//!     appears only in that terminator frame (atomicity invariant).
//!
//! Full discharge blocks on a live `/api/chat` handler with `tools[]` support
//! in aprender-serve (BLOCKER-UPSTREAM-MISSING). Until then, these classifiers
//! run on captured JSON / NDJSON fixtures.
//!
//! Spec: `contracts/crux-I-04-v1.yaml`.

use serde_json::Value;

/// Outcome of `classify_tool_call_schema`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallSchemaOutcome {
    /// `message.tool_calls` is a non-empty array of well-formed tool calls.
    Ok,
    /// Top-level response is not a JSON object.
    NotAnObject,
    /// `message` field is missing.
    MissingMessage,
    /// `message` is present but not a JSON object.
    MessageNotAnObject,
    /// `message.tool_calls` field is missing.
    MissingToolCalls,
    /// `message.tool_calls` is present but not a JSON array.
    ToolCallsNotAnArray,
    /// `message.tool_calls` is an empty array.
    ToolCallsEmpty,
    /// A tool call at `index` is not a JSON object.
    ToolCallNotAnObject { index: usize },
    /// A tool call at `index` is missing its `function` field.
    MissingFunction { index: usize },
    /// A tool call's `function` at `index` is not a JSON object.
    FunctionNotAnObject { index: usize },
    /// A tool call's `function.name` at `index` is missing or not a string.
    FunctionNameNotAString { index: usize },
    /// A tool call's `function.arguments` at `index` is missing.
    MissingFunctionArguments { index: usize },
    /// A tool call's `function.arguments` at `index` is a **string** instead
    /// of a JSON object — this is the canonical OpenAI/Ollama drift bug that
    /// breaks strict JSON-schema validation downstream.
    FunctionArgumentsIsString { index: usize },
    /// A tool call's `function.arguments` at `index` is neither a string nor
    /// a JSON object (e.g. array, number, null).
    FunctionArgumentsNotAnObject { index: usize },
}

/// Non-streaming `/api/chat` function-call response-shape gate.
/// Pure; deterministic.
pub fn classify_tool_call_schema(response: &Value) -> ToolCallSchemaOutcome {
    let obj = match response.as_object() {
        Some(o) => o,
        None => return ToolCallSchemaOutcome::NotAnObject,
    };
    let message = match obj.get("message") {
        Some(m) => m,
        None => return ToolCallSchemaOutcome::MissingMessage,
    };
    let message_obj = match message.as_object() {
        Some(o) => o,
        None => return ToolCallSchemaOutcome::MessageNotAnObject,
    };
    let tool_calls = match message_obj.get("tool_calls") {
        Some(tc) => tc,
        None => return ToolCallSchemaOutcome::MissingToolCalls,
    };
    let arr = match tool_calls.as_array() {
        Some(a) => a,
        None => return ToolCallSchemaOutcome::ToolCallsNotAnArray,
    };
    if arr.is_empty() {
        return ToolCallSchemaOutcome::ToolCallsEmpty;
    }
    for (index, call) in arr.iter().enumerate() {
        let call_obj = match call.as_object() {
            Some(o) => o,
            None => return ToolCallSchemaOutcome::ToolCallNotAnObject { index },
        };
        let function = match call_obj.get("function") {
            Some(f) => f,
            None => return ToolCallSchemaOutcome::MissingFunction { index },
        };
        let fn_obj = match function.as_object() {
            Some(o) => o,
            None => return ToolCallSchemaOutcome::FunctionNotAnObject { index },
        };
        if !fn_obj.get("name").is_some_and(Value::is_string) {
            return ToolCallSchemaOutcome::FunctionNameNotAString { index };
        }
        let args = match fn_obj.get("arguments") {
            Some(a) => a,
            None => return ToolCallSchemaOutcome::MissingFunctionArguments { index },
        };
        if args.is_string() {
            return ToolCallSchemaOutcome::FunctionArgumentsIsString { index };
        }
        if !args.is_object() {
            return ToolCallSchemaOutcome::FunctionArgumentsNotAnObject { index };
        }
    }
    ToolCallSchemaOutcome::Ok
}

/// Outcome of `classify_tool_name_allowlist`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolNameAllowlistOutcome {
    /// Every called tool name appears in the declared tool set.
    Ok,
    /// A tool call at `index` invokes a name not in the declared set.
    HallucinatedToolName { index: usize, name: String },
    /// The declared set is empty (server produced tool_calls against no
    /// declared tools at all).
    NoDeclaredTools,
}

/// Allowlist gate: every called tool's `function.name` must be a member of
/// `declared_tool_names`. Does not revalidate the schema — call
/// `classify_tool_call_schema` first.
pub fn classify_tool_name_allowlist(
    response: &Value,
    declared_tool_names: &[&str],
) -> ToolNameAllowlistOutcome {
    if declared_tool_names.is_empty() {
        // If no tools were declared, any tool_call is a hallucination.
        // We still report NoDeclaredTools as the more informative outcome
        // only when there's at least one tool call to reject.
        if response
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
        {
            return ToolNameAllowlistOutcome::NoDeclaredTools;
        }
        return ToolNameAllowlistOutcome::Ok;
    }
    let calls = response
        .get("message")
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array);
    let arr = match calls {
        Some(a) => a,
        None => return ToolNameAllowlistOutcome::Ok,
    };
    for (index, call) in arr.iter().enumerate() {
        let name = call
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str);
        if let Some(n) = name {
            if !declared_tool_names.iter().any(|d| *d == n) {
                return ToolNameAllowlistOutcome::HallucinatedToolName {
                    index,
                    name: n.to_string(),
                };
            }
        }
    }
    ToolNameAllowlistOutcome::Ok
}

/// Outcome of `classify_streaming_tool_call`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamingToolCallOutcome {
    /// Exactly one `done == true` frame as the last frame, and `tool_calls`
    /// (if any) appear only in that terminator.
    Ok,
    /// Frame list is empty.
    NoFrames,
    /// A frame is not a JSON object.
    FrameNotAnObject { frame_index: usize },
    /// A frame is missing the `done` field.
    MissingDoneField { frame_index: usize },
    /// A frame's `done` field is not a boolean.
    DoneNotBool { frame_index: usize },
    /// A non-final frame is `done == true`.
    EarlyDoneFrame { frame_index: usize },
    /// No frame had `done == true`.
    NoTerminalFrame,
    /// A non-terminator frame carried a non-empty `message.tool_calls[]`
    /// (violates atomicity).
    ToolCallsInNonTerminatorFrame { frame_index: usize },
}

/// Streaming NDJSON terminator-atomicity gate for tool calls.
/// Pure; deterministic.
pub fn classify_streaming_tool_call(frames: &[Value]) -> StreamingToolCallOutcome {
    if frames.is_empty() {
        return StreamingToolCallOutcome::NoFrames;
    }
    let last_idx = frames.len() - 1;
    let mut terminal_seen = false;
    for (i, frame) in frames.iter().enumerate() {
        let obj = match frame.as_object() {
            Some(o) => o,
            None => return StreamingToolCallOutcome::FrameNotAnObject { frame_index: i },
        };
        let done_val = match obj.get("done") {
            Some(v) => v,
            None => return StreamingToolCallOutcome::MissingDoneField { frame_index: i },
        };
        let done = match done_val.as_bool() {
            Some(b) => b,
            None => return StreamingToolCallOutcome::DoneNotBool { frame_index: i },
        };
        if done {
            if i != last_idx {
                return StreamingToolCallOutcome::EarlyDoneFrame { frame_index: i };
            }
            terminal_seen = true;
        } else {
            // Non-terminator frames must not carry tool_calls.
            if frame_has_nonempty_tool_calls(frame) {
                return StreamingToolCallOutcome::ToolCallsInNonTerminatorFrame { frame_index: i };
            }
        }
    }
    if !terminal_seen {
        return StreamingToolCallOutcome::NoTerminalFrame;
    }
    StreamingToolCallOutcome::Ok
}

fn frame_has_nonempty_tool_calls(frame: &Value) -> bool {
    frame
        .get("message")
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ═══ schema classifier ═══

    fn well_formed_tool_response() -> Value {
        json!({
            "model": "tiny",
            "created_at": "2026-04-22T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "location": "San Francisco",
                                "unit": "celsius"
                            }
                        }
                    }
                ]
            },
            "done": true
        })
    }

    #[test]
    fn schema_ok_on_well_formed_tool_call() {
        assert_eq!(
            classify_tool_call_schema(&well_formed_tool_response()),
            ToolCallSchemaOutcome::Ok
        );
    }

    #[test]
    fn schema_rejects_non_object_top_level() {
        assert_eq!(
            classify_tool_call_schema(&json!([1, 2, 3])),
            ToolCallSchemaOutcome::NotAnObject
        );
    }

    #[test]
    fn schema_rejects_missing_message() {
        let v = json!({"done": true});
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::MissingMessage
        );
    }

    #[test]
    fn schema_rejects_non_object_message() {
        let v = json!({"message": "just a string", "done": true});
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::MessageNotAnObject
        );
    }

    #[test]
    fn schema_rejects_missing_tool_calls() {
        let v = json!({"message": {"role": "assistant", "content": "no tool"}, "done": true});
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::MissingToolCalls
        );
    }

    #[test]
    fn schema_rejects_tool_calls_not_array() {
        let v = json!({"message": {"role": "assistant", "tool_calls": "oops"}, "done": true});
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::ToolCallsNotAnArray
        );
    }

    #[test]
    fn schema_rejects_empty_tool_calls() {
        let v = json!({"message": {"role": "assistant", "tool_calls": []}, "done": true});
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::ToolCallsEmpty
        );
    }

    #[test]
    fn schema_rejects_non_object_tool_call() {
        let v = json!({"message": {"role": "assistant", "tool_calls": ["bad"]}, "done": true});
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::ToolCallNotAnObject { index: 0 }
        );
    }

    #[test]
    fn schema_rejects_missing_function() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [{}]},
            "done": true
        });
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::MissingFunction { index: 0 }
        );
    }

    #[test]
    fn schema_rejects_function_not_object() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [{"function": "x"}]},
            "done": true
        });
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::FunctionNotAnObject { index: 0 }
        );
    }

    #[test]
    fn schema_rejects_missing_function_name() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [{"function": {"arguments": {}}}]},
            "done": true
        });
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::FunctionNameNotAString { index: 0 }
        );
    }

    #[test]
    fn schema_rejects_non_string_function_name() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": 42, "arguments": {}}}
            ]},
            "done": true
        });
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::FunctionNameNotAString { index: 0 }
        );
    }

    #[test]
    fn schema_rejects_missing_arguments() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather"}}
            ]},
            "done": true
        });
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::MissingFunctionArguments { index: 0 }
        );
    }

    /// The canonical drift bug: `arguments` comes back as a **string** of
    /// JSON instead of a JSON object. Must be rejected even though the
    /// "content" looks parseable.
    #[test]
    fn schema_rejects_stringified_arguments() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": "{\"location\":\"SF\"}"}}
            ]},
            "done": true
        });
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::FunctionArgumentsIsString { index: 0 }
        );
    }

    #[test]
    fn schema_rejects_non_object_non_string_arguments() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": 42}}
            ]},
            "done": true
        });
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::FunctionArgumentsNotAnObject { index: 0 }
        );
    }

    #[test]
    fn schema_reports_second_call_violations() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {}}},
                {"function": {"name": "get_time", "arguments": "stringy"}}
            ]},
            "done": true
        });
        assert_eq!(
            classify_tool_call_schema(&v),
            ToolCallSchemaOutcome::FunctionArgumentsIsString { index: 1 }
        );
    }

    #[test]
    fn schema_classifier_is_deterministic() {
        let r = well_formed_tool_response();
        let a = classify_tool_call_schema(&r);
        let b = classify_tool_call_schema(&r);
        assert_eq!(a, b);
    }

    // ═══ allowlist classifier ═══

    #[test]
    fn allowlist_ok_when_called_name_matches_declared() {
        let r = well_formed_tool_response();
        assert_eq!(
            classify_tool_name_allowlist(&r, &["get_weather", "get_time"]),
            ToolNameAllowlistOutcome::Ok
        );
    }

    #[test]
    fn allowlist_rejects_hallucinated_name() {
        let r = well_formed_tool_response();
        assert_eq!(
            classify_tool_name_allowlist(&r, &["get_time"]),
            ToolNameAllowlistOutcome::HallucinatedToolName {
                index: 0,
                name: "get_weather".to_string(),
            }
        );
    }

    #[test]
    fn allowlist_flags_no_declared_tools_when_calls_present() {
        let r = well_formed_tool_response();
        assert_eq!(
            classify_tool_name_allowlist(&r, &[]),
            ToolNameAllowlistOutcome::NoDeclaredTools
        );
    }

    #[test]
    fn allowlist_ok_when_no_declared_and_no_calls() {
        let v = json!({"message": {"role": "assistant", "content": "hi"}, "done": true});
        assert_eq!(
            classify_tool_name_allowlist(&v, &[]),
            ToolNameAllowlistOutcome::Ok
        );
    }

    #[test]
    fn allowlist_ok_when_no_tool_calls_present() {
        let v = json!({"message": {"role": "assistant", "content": "hi"}, "done": true});
        assert_eq!(
            classify_tool_name_allowlist(&v, &["get_weather"]),
            ToolNameAllowlistOutcome::Ok
        );
    }

    #[test]
    fn allowlist_reports_second_call_hallucination() {
        let v = json!({
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {}}},
                {"function": {"name": "make_coffee", "arguments": {}}}
            ]},
            "done": true
        });
        assert_eq!(
            classify_tool_name_allowlist(&v, &["get_weather"]),
            ToolNameAllowlistOutcome::HallucinatedToolName {
                index: 1,
                name: "make_coffee".to_string()
            }
        );
    }

    #[test]
    fn allowlist_classifier_is_deterministic() {
        let r = well_formed_tool_response();
        let names = ["get_weather"];
        let a = classify_tool_name_allowlist(&r, &names);
        let b = classify_tool_name_allowlist(&r, &names);
        assert_eq!(a, b);
    }

    // ═══ streaming NDJSON classifier ═══

    fn delta_frame(content: &str) -> Value {
        json!({"message": {"role": "assistant", "content": content}, "done": false})
    }
    fn terminal_tool_frame() -> Value {
        json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"location": "SF"}}}
                ]
            },
            "done": true
        })
    }
    fn terminal_plain_frame() -> Value {
        json!({"done": true, "eval_count": 3, "eval_duration": 500})
    }

    #[test]
    fn ndjson_ok_on_single_terminator_with_tool_calls() {
        let frames = vec![delta_frame(""), delta_frame(""), terminal_tool_frame()];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::Ok
        );
    }

    #[test]
    fn ndjson_ok_on_terminator_only() {
        let frames = vec![terminal_tool_frame()];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::Ok
        );
    }

    #[test]
    fn ndjson_ok_when_no_tool_calls_anywhere() {
        let frames = vec![delta_frame("hi"), terminal_plain_frame()];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::Ok
        );
    }

    #[test]
    fn ndjson_rejects_empty() {
        let frames: Vec<Value> = vec![];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::NoFrames
        );
    }

    #[test]
    fn ndjson_rejects_non_object_frame() {
        let frames = vec![json!([1, 2]), terminal_plain_frame()];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::FrameNotAnObject { frame_index: 0 }
        );
    }

    #[test]
    fn ndjson_rejects_missing_done() {
        let frames = vec![
            json!({"message": {"role": "assistant", "content": "x"}}),
            terminal_plain_frame(),
        ];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::MissingDoneField { frame_index: 0 }
        );
    }

    #[test]
    fn ndjson_rejects_non_bool_done() {
        let frames = vec![json!({"done": "false"}), terminal_plain_frame()];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::DoneNotBool { frame_index: 0 }
        );
    }

    #[test]
    fn ndjson_rejects_early_done_true() {
        let frames = vec![
            delta_frame("hi"),
            json!({"done": true}),
            delta_frame("!"),
            terminal_plain_frame(),
        ];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::EarlyDoneFrame { frame_index: 1 }
        );
    }

    #[test]
    fn ndjson_rejects_missing_terminator() {
        let frames = vec![delta_frame("a"), delta_frame("b")];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::NoTerminalFrame
        );
    }

    /// A tool_calls array appears in a non-terminator frame — violates
    /// atomicity. Canonical Ollama invariant.
    #[test]
    fn ndjson_rejects_tool_calls_in_non_terminator() {
        let frames = vec![
            json!({
                "message": {"role": "assistant", "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {}}}
                ]},
                "done": false
            }),
            terminal_plain_frame(),
        ];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::ToolCallsInNonTerminatorFrame { frame_index: 0 }
        );
    }

    #[test]
    fn ndjson_allows_empty_tool_calls_array_in_non_terminator() {
        // An empty array does not violate atomicity — nothing was actually split.
        let frames = vec![
            json!({"message": {"role": "assistant", "tool_calls": []}, "done": false}),
            terminal_tool_frame(),
        ];
        assert_eq!(
            classify_streaming_tool_call(&frames),
            StreamingToolCallOutcome::Ok
        );
    }

    #[test]
    fn ndjson_classifier_is_deterministic() {
        let frames = vec![delta_frame("a"), terminal_tool_frame()];
        let a = classify_streaming_tool_call(&frames);
        let b = classify_streaming_tool_call(&frames);
        assert_eq!(a, b);
    }
}
