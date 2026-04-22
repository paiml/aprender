//! CRUX-C-11 — OpenAI tool-use function-calling classifier (PARTIAL_ALGORITHM_LEVEL)
//!
//! Pure algorithm-level discharge of the FALSIFY gates in
//! `contracts/crux-C-11-v1.yaml`:
//!   * FALSIFY-001 — response tool_calls[] schema parity with OpenAI:
//!     `function.name` ∈ declared tool set, `function.arguments` is a
//!     JSON-parseable string, `finish_reason == "tool_calls"` when
//!     tool_calls is non-empty.
//!   * FALSIFY-002 — parsed `arguments` object satisfies the
//!     declared `parameters` JSON schema (object/required/per-property
//!     type subset sufficient for the falsification fixture).
//!   * FALSIFY-003 — when no `tools[]` was requested, no tool_calls
//!     are synthesized and `finish_reason ∈ {stop, length}`.
//!
//! The classifier operates on already-parsed JSON (`serde_json::Value`)
//! so it has zero HTTP/network dependencies.

use serde_json::Value;
use std::collections::BTreeSet;

/// ─── FALSIFY-C-11-001 response schema ───────────────────────────────────

/// Representation of a declared tool in the request, stripped down to the
/// fields this classifier cares about.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredTool {
    pub name: String,
    pub parameters: Value,
}

/// One tool_call as it appears in a response `choices[i].message.tool_calls[]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub call_type: String,
    pub name: String,
    /// Arguments as emitted by the server: an OpenAI-conformant tool_call
    /// always serializes `arguments` as a JSON *string*, not an object.
    pub arguments_json_string: String,
}

/// Outcome of the response-shape classifier.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallsShapeOutcome {
    Ok {
        n_calls: usize,
    },
    /// finish_reason does not match the tool_calls vs no-tool_calls expectation.
    FinishReasonMismatch {
        n_calls: usize,
        got: String,
        expected_any_of: Vec<&'static str>,
    },
    /// A tool call names a function not present in request.tools[].
    UnknownToolName {
        index: usize,
        got: String,
    },
    /// A tool call `type` field is not "function" (the OpenAI spec value).
    WrongCallType {
        index: usize,
        got: String,
    },
    /// `arguments` is not a JSON-parseable string.
    ArgumentsNotJson {
        index: usize,
        raw: String,
    },
}

/// Validate the tool_calls schema for a single `choices[0].message`.
///
/// * `declared_tools` — tools passed in the request.
/// * `tool_calls` — the `choices[0].message.tool_calls[]` array (may be empty).
/// * `finish_reason` — the `choices[0].finish_reason` string.
pub fn classify_tool_calls_shape(
    declared_tools: &[DeclaredTool],
    tool_calls: &[ToolCall],
    finish_reason: &str,
) -> ToolCallsShapeOutcome {
    let declared: BTreeSet<&str> = declared_tools.iter().map(|t| t.name.as_str()).collect();

    for (i, call) in tool_calls.iter().enumerate() {
        if call.call_type != "function" {
            return ToolCallsShapeOutcome::WrongCallType {
                index: i,
                got: call.call_type.clone(),
            };
        }
        if !declared.contains(call.name.as_str()) {
            return ToolCallsShapeOutcome::UnknownToolName {
                index: i,
                got: call.name.clone(),
            };
        }
        if serde_json::from_str::<Value>(&call.arguments_json_string).is_err() {
            return ToolCallsShapeOutcome::ArgumentsNotJson {
                index: i,
                raw: call.arguments_json_string.clone(),
            };
        }
    }

    // finish_reason consistency.
    if tool_calls.is_empty() {
        if finish_reason == "stop" || finish_reason == "length" {
            ToolCallsShapeOutcome::Ok { n_calls: 0 }
        } else {
            ToolCallsShapeOutcome::FinishReasonMismatch {
                n_calls: 0,
                got: finish_reason.to_string(),
                expected_any_of: vec!["stop", "length"],
            }
        }
    } else if finish_reason == "tool_calls" {
        ToolCallsShapeOutcome::Ok {
            n_calls: tool_calls.len(),
        }
    } else {
        ToolCallsShapeOutcome::FinishReasonMismatch {
            n_calls: tool_calls.len(),
            got: finish_reason.to_string(),
            expected_any_of: vec!["tool_calls"],
        }
    }
}

/// ─── FALSIFY-C-11-002 arguments validate against JSON schema ────────────
///
/// Minimal JSON-schema subset sufficient for the falsification fixture:
///   {"type":"object","properties":{ <name>: {"type": <t>} }, "required":[...]}
/// Supported property types: "string", "number", "integer", "boolean",
/// "array", "object", "null". Everything else is reported as
/// `UnsupportedSchema` so we never silently pass.

#[derive(Debug, Clone, PartialEq)]
pub enum SchemaValidationOutcome {
    Ok,
    ArgumentsNotJson {
        raw: String,
    },
    ArgumentsNotObject,
    MissingRequiredProperty {
        name: String,
    },
    WrongPropertyType {
        name: String,
        expected: String,
        got: String,
    },
    UnsupportedSchema {
        reason: String,
    },
}

fn value_type_tag(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn accepts(schema_type: &str, actual: &str) -> bool {
    // JSON Schema: an integer is also a number. Everything else is strict.
    if schema_type == actual {
        return true;
    }
    if schema_type == "number" && actual == "integer" {
        return true;
    }
    false
}

/// Validate `arguments_json_string` against the declared `parameters` schema.
///
/// The classifier is intentionally strict: only the small JSON-schema
/// subset listed above is recognised; anything else returns
/// `UnsupportedSchema` so untested paths never silently pass.
pub fn classify_arguments_against_schema(
    arguments_json_string: &str,
    parameters: &Value,
) -> SchemaValidationOutcome {
    let args: Value = match serde_json::from_str(arguments_json_string) {
        Ok(v) => v,
        Err(_) => {
            return SchemaValidationOutcome::ArgumentsNotJson {
                raw: arguments_json_string.to_string(),
            }
        }
    };

    // Top-level schema must be {"type":"object", ...} for this subset.
    let obj = match parameters.as_object() {
        Some(o) => o,
        None => {
            return SchemaValidationOutcome::UnsupportedSchema {
                reason: "top-level schema is not a JSON object".into(),
            }
        }
    };
    if obj.get("type").and_then(Value::as_str) != Some("object") {
        return SchemaValidationOutcome::UnsupportedSchema {
            reason: "only top-level type:object is supported".into(),
        };
    }

    // Arguments must be an object.
    let args_obj = match args.as_object() {
        Some(o) => o,
        None => return SchemaValidationOutcome::ArgumentsNotObject,
    };

    // Required properties.
    if let Some(required) = obj.get("required") {
        let arr = match required.as_array() {
            Some(a) => a,
            None => {
                return SchemaValidationOutcome::UnsupportedSchema {
                    reason: "`required` must be an array of strings".into(),
                }
            }
        };
        for r in arr {
            let name = match r.as_str() {
                Some(s) => s,
                None => {
                    return SchemaValidationOutcome::UnsupportedSchema {
                        reason: "`required` entries must be strings".into(),
                    }
                }
            };
            if !args_obj.contains_key(name) {
                return SchemaValidationOutcome::MissingRequiredProperty {
                    name: name.to_string(),
                };
            }
        }
    }

    // Per-property type check (only over properties present in `args`).
    if let Some(props) = obj.get("properties").and_then(Value::as_object) {
        for (name, prop_schema) in props {
            let Some(actual_val) = args_obj.get(name) else {
                continue;
            };
            let Some(schema_type) = prop_schema.get("type").and_then(Value::as_str) else {
                return SchemaValidationOutcome::UnsupportedSchema {
                    reason: format!("property '{}' has no `type` field", name),
                };
            };
            let accepted = matches!(
                schema_type,
                "string" | "number" | "integer" | "boolean" | "array" | "object" | "null"
            );
            if !accepted {
                return SchemaValidationOutcome::UnsupportedSchema {
                    reason: format!("property '{}' has unsupported type '{}'", name, schema_type),
                };
            }
            let actual = value_type_tag(actual_val);
            if !accepts(schema_type, actual) {
                return SchemaValidationOutcome::WrongPropertyType {
                    name: name.clone(),
                    expected: schema_type.to_string(),
                    got: actual.to_string(),
                };
            }
        }
    }

    SchemaValidationOutcome::Ok
}

/// ─── FALSIFY-C-11-003 no-tools passthrough ──────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum NoToolsPassthroughOutcome {
    Ok,
    UnexpectedToolCalls {
        n_calls: usize,
    },
    WrongFinishReason {
        got: String,
        expected_any_of: Vec<&'static str>,
    },
}

/// Validate that a response produced from a request *without* `tools[]`
/// does not synthesize tool_calls and emits a non-tool finish_reason.
pub fn classify_no_tools_passthrough(
    tool_calls: &[ToolCall],
    finish_reason: &str,
) -> NoToolsPassthroughOutcome {
    if !tool_calls.is_empty() {
        return NoToolsPassthroughOutcome::UnexpectedToolCalls {
            n_calls: tool_calls.len(),
        };
    }
    if finish_reason == "stop" || finish_reason == "length" {
        NoToolsPassthroughOutcome::Ok
    } else {
        NoToolsPassthroughOutcome::WrongFinishReason {
            got: finish_reason.to_string(),
            expected_any_of: vec!["stop", "length"],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> DeclaredTool {
        DeclaredTool {
            name: "get_weather".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }),
        }
    }

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            name: name.into(),
            arguments_json_string: args.into(),
        }
    }

    // ─── shape classifier ─────────────────────────────────────────────────

    #[test]
    fn shape_ok_on_well_formed_single_call() {
        let tools = vec![weather_tool()];
        let calls = vec![call("get_weather", "{\"location\":\"Paris\"}")];
        assert_eq!(
            classify_tool_calls_shape(&tools, &calls, "tool_calls"),
            ToolCallsShapeOutcome::Ok { n_calls: 1 }
        );
    }

    #[test]
    fn shape_unknown_tool_name_rejected() {
        let tools = vec![weather_tool()];
        let calls = vec![call("get_time", "{}")];
        match classify_tool_calls_shape(&tools, &calls, "tool_calls") {
            ToolCallsShapeOutcome::UnknownToolName { index, got } => {
                assert_eq!(index, 0);
                assert_eq!(got, "get_time");
            }
            other => panic!("expected UnknownToolName, got {:?}", other),
        }
    }

    #[test]
    fn shape_non_function_type_rejected() {
        let tools = vec![weather_tool()];
        let mut c = call("get_weather", "{}");
        c.call_type = "code_interpreter".into();
        match classify_tool_calls_shape(&tools, &[c], "tool_calls") {
            ToolCallsShapeOutcome::WrongCallType { index, got } => {
                assert_eq!(index, 0);
                assert_eq!(got, "code_interpreter");
            }
            other => panic!("expected WrongCallType, got {:?}", other),
        }
    }

    #[test]
    fn shape_non_json_arguments_rejected() {
        let tools = vec![weather_tool()];
        let calls = vec![call("get_weather", "not-json{")];
        match classify_tool_calls_shape(&tools, &calls, "tool_calls") {
            ToolCallsShapeOutcome::ArgumentsNotJson { raw, .. } => {
                assert_eq!(raw, "not-json{");
            }
            other => panic!("expected ArgumentsNotJson, got {:?}", other),
        }
    }

    #[test]
    fn shape_wrong_finish_reason_with_tool_calls_rejected() {
        let tools = vec![weather_tool()];
        let calls = vec![call("get_weather", "{}")];
        match classify_tool_calls_shape(&tools, &calls, "stop") {
            ToolCallsShapeOutcome::FinishReasonMismatch {
                n_calls,
                got,
                expected_any_of,
            } => {
                assert_eq!(n_calls, 1);
                assert_eq!(got, "stop");
                assert_eq!(expected_any_of, vec!["tool_calls"]);
            }
            other => panic!("expected FinishReasonMismatch, got {:?}", other),
        }
    }

    #[test]
    fn shape_empty_calls_with_stop_is_ok() {
        let tools = vec![weather_tool()];
        assert_eq!(
            classify_tool_calls_shape(&tools, &[], "stop"),
            ToolCallsShapeOutcome::Ok { n_calls: 0 }
        );
    }

    #[test]
    fn shape_empty_calls_with_tool_calls_finish_reason_rejected() {
        let tools = vec![weather_tool()];
        match classify_tool_calls_shape(&tools, &[], "tool_calls") {
            ToolCallsShapeOutcome::FinishReasonMismatch { n_calls, .. } => {
                assert_eq!(n_calls, 0);
            }
            other => panic!("expected FinishReasonMismatch, got {:?}", other),
        }
    }

    // ─── schema validator ─────────────────────────────────────────────────

    fn weather_schema() -> Value {
        json!({
            "type":"object",
            "properties":{"location":{"type":"string"}},
            "required":["location"]
        })
    }

    #[test]
    fn schema_ok_on_matching_object() {
        assert_eq!(
            classify_arguments_against_schema("{\"location\":\"Paris\"}", &weather_schema()),
            SchemaValidationOutcome::Ok
        );
    }

    #[test]
    fn schema_missing_required_property_rejected() {
        match classify_arguments_against_schema("{}", &weather_schema()) {
            SchemaValidationOutcome::MissingRequiredProperty { name } => {
                assert_eq!(name, "location");
            }
            other => panic!("expected MissingRequiredProperty, got {:?}", other),
        }
    }

    #[test]
    fn schema_wrong_property_type_rejected() {
        match classify_arguments_against_schema("{\"location\":42}", &weather_schema()) {
            SchemaValidationOutcome::WrongPropertyType {
                name,
                expected,
                got,
            } => {
                assert_eq!(name, "location");
                assert_eq!(expected, "string");
                assert_eq!(got, "integer");
            }
            other => panic!("expected WrongPropertyType, got {:?}", other),
        }
    }

    #[test]
    fn schema_integer_accepted_for_number_type() {
        let s = json!({
            "type":"object",
            "properties":{"temp":{"type":"number"}},
            "required":["temp"]
        });
        assert_eq!(
            classify_arguments_against_schema("{\"temp\":25}", &s),
            SchemaValidationOutcome::Ok
        );
    }

    #[test]
    fn schema_number_rejected_for_integer_type() {
        let s = json!({
            "type":"object",
            "properties":{"count":{"type":"integer"}},
            "required":["count"]
        });
        match classify_arguments_against_schema("{\"count\":1.5}", &s) {
            SchemaValidationOutcome::WrongPropertyType { expected, got, .. } => {
                assert_eq!(expected, "integer");
                assert_eq!(got, "number");
            }
            other => panic!("expected WrongPropertyType, got {:?}", other),
        }
    }

    #[test]
    fn schema_non_json_arguments_rejected() {
        match classify_arguments_against_schema("garbage", &weather_schema()) {
            SchemaValidationOutcome::ArgumentsNotJson { raw } => {
                assert_eq!(raw, "garbage");
            }
            other => panic!("expected ArgumentsNotJson, got {:?}", other),
        }
    }

    #[test]
    fn schema_non_object_arguments_rejected() {
        match classify_arguments_against_schema("[1,2,3]", &weather_schema()) {
            SchemaValidationOutcome::ArgumentsNotObject => {}
            other => panic!("expected ArgumentsNotObject, got {:?}", other),
        }
    }

    #[test]
    fn schema_top_level_non_object_is_unsupported() {
        let s = json!({"type":"array"});
        match classify_arguments_against_schema("[]", &s) {
            SchemaValidationOutcome::UnsupportedSchema { .. } => {}
            other => panic!("expected UnsupportedSchema, got {:?}", other),
        }
    }

    #[test]
    fn schema_unsupported_property_type_is_flagged() {
        let s = json!({
            "type":"object",
            "properties":{"x":{"type":"weirdtype"}},
            "required":[]
        });
        match classify_arguments_against_schema("{\"x\":1}", &s) {
            SchemaValidationOutcome::UnsupportedSchema { reason } => {
                assert!(reason.contains("weirdtype"));
            }
            other => panic!("expected UnsupportedSchema, got {:?}", other),
        }
    }

    #[test]
    fn schema_property_without_type_is_unsupported() {
        let s = json!({
            "type":"object",
            "properties":{"y":{"description":"noop"}},
            "required":[]
        });
        match classify_arguments_against_schema("{\"y\":1}", &s) {
            SchemaValidationOutcome::UnsupportedSchema { reason } => {
                assert!(reason.contains("y"));
            }
            other => panic!("expected UnsupportedSchema, got {:?}", other),
        }
    }

    #[test]
    fn schema_validator_is_deterministic() {
        let args = "{\"location\":\"NYC\"}";
        let s = weather_schema();
        let a = classify_arguments_against_schema(args, &s);
        let b = classify_arguments_against_schema(args, &s);
        assert_eq!(a, b);
    }

    // ─── no-tools passthrough ─────────────────────────────────────────────

    #[test]
    fn no_tools_empty_calls_with_stop_is_ok() {
        assert_eq!(
            classify_no_tools_passthrough(&[], "stop"),
            NoToolsPassthroughOutcome::Ok
        );
    }

    #[test]
    fn no_tools_empty_calls_with_length_is_ok() {
        assert_eq!(
            classify_no_tools_passthrough(&[], "length"),
            NoToolsPassthroughOutcome::Ok
        );
    }

    #[test]
    fn no_tools_synthesized_calls_rejected() {
        let c = call("get_weather", "{}");
        match classify_no_tools_passthrough(&[c], "stop") {
            NoToolsPassthroughOutcome::UnexpectedToolCalls { n_calls } => {
                assert_eq!(n_calls, 1);
            }
            other => panic!("expected UnexpectedToolCalls, got {:?}", other),
        }
    }

    #[test]
    fn no_tools_wrong_finish_reason_rejected() {
        match classify_no_tools_passthrough(&[], "tool_calls") {
            NoToolsPassthroughOutcome::WrongFinishReason {
                got,
                expected_any_of,
            } => {
                assert_eq!(got, "tool_calls");
                assert_eq!(expected_any_of, vec!["stop", "length"]);
            }
            other => panic!("expected WrongFinishReason, got {:?}", other),
        }
    }

    #[test]
    fn no_tools_classifier_is_deterministic() {
        assert_eq!(
            classify_no_tools_passthrough(&[], "stop"),
            classify_no_tools_passthrough(&[], "stop"),
        );
    }
}
