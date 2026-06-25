//! `apr tool-use-lint` — CRUX-C-11 OpenAI tool-use observation linter.
//!
//! Reads a JSON observation file that captures a single /v1/chat/completions
//! response (plus its originating tools[]) and dispatches three classifiers
//! (shape, schema, passthrough). Emits a text or `--json` report.
//!
//! Spec: `contracts/crux-C-11-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.
//!
//! Observation schema (top-level keys; all optional — missing sections skip
//! the corresponding classifier):
//!
//!   {
//!     "shape": {
//!        "declared_tools": [ { "name": "get_weather", "parameters": {...} } ],
//!        "tool_calls": [
//!           { "id": "call_1", "type": "function",
//!             "name": "get_weather", "arguments": "{\"location\":\"Paris\"}" }
//!        ],
//!        "finish_reason": "tool_calls"
//!     },
//!     "schema": {
//!        "arguments": "{\"location\":\"Paris\"}",
//!        "parameters": { "type":"object", ... }
//!     },
//!     "passthrough": {
//!        "tool_calls": [],
//!        "finish_reason": "stop"
//!     }
//!   }

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::commands::tool_use_classifier as clf;
use crate::error::{CliError, Result};

pub(crate) fn run(observation_file: &Path, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr tool-use-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    let shape = classify_shape(&obs);
    let schema = classify_schema(&obs);
    let passthrough = classify_passthrough(&obs);

    let fail_reasons: Vec<String> = [
        shape.as_ref().and_then(shape_fail_reason),
        schema.as_ref().and_then(schema_fail_reason),
        passthrough.as_ref().and_then(passthrough_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        shape.as_ref(),
        schema.as_ref(),
        passthrough.as_ref(),
        json,
    );

    if fail_reasons.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(fail_reasons.join("; ")))
    }
}

fn parse_tool_calls(v: &Value) -> Option<Vec<clf::ToolCall>> {
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for el in arr {
        let obj = el.as_object()?;
        out.push(clf::ToolCall {
            id: obj
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            call_type: obj
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("function")
                .to_string(),
            name: obj.get("name").and_then(Value::as_str)?.to_string(),
            arguments_json_string: obj
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        });
    }
    Some(out)
}

fn parse_declared_tools(v: &Value) -> Option<Vec<clf::DeclaredTool>> {
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for el in arr {
        let obj = el.as_object()?;
        out.push(clf::DeclaredTool {
            name: obj.get("name").and_then(Value::as_str)?.to_string(),
            parameters: obj.get("parameters").cloned().unwrap_or(Value::Null),
        });
    }
    Some(out)
}

fn classify_shape(obs: &Value) -> Option<clf::ToolCallsShapeOutcome> {
    let sec = obs.get("shape")?.as_object()?;
    let declared = parse_declared_tools(sec.get("declared_tools")?)?;
    let calls = parse_tool_calls(sec.get("tool_calls")?)?;
    let fr = sec.get("finish_reason")?.as_str()?;
    Some(clf::classify_tool_calls_shape(&declared, &calls, fr))
}

fn classify_schema(obs: &Value) -> Option<clf::SchemaValidationOutcome> {
    let sec = obs.get("schema")?.as_object()?;
    let args = sec.get("arguments")?.as_str()?;
    let params = sec.get("parameters")?;
    Some(clf::classify_arguments_against_schema(args, params))
}

fn classify_passthrough(obs: &Value) -> Option<clf::NoToolsPassthroughOutcome> {
    let sec = obs.get("passthrough")?.as_object()?;
    let calls = parse_tool_calls(sec.get("tool_calls")?)?;
    let fr = sec.get("finish_reason")?.as_str()?;
    Some(clf::classify_no_tools_passthrough(&calls, fr))
}

fn shape_fail_reason(o: &clf::ToolCallsShapeOutcome) -> Option<String> {
    match o {
        clf::ToolCallsShapeOutcome::Ok { .. } => None,
        clf::ToolCallsShapeOutcome::FinishReasonMismatch {
            n_calls,
            got,
            expected_any_of,
        } => Some(format!(
            "FALSIFY-CRUX-C-11-001 shape: finish_reason={got:?} for n_calls={n_calls} (expected any of {expected_any_of:?})"
        )),
        clf::ToolCallsShapeOutcome::UnknownToolName { index, got } => Some(format!(
            "FALSIFY-CRUX-C-11-001 shape: tool_calls[{index}].name={got:?} not in declared_tools"
        )),
        clf::ToolCallsShapeOutcome::WrongCallType { index, got } => Some(format!(
            "FALSIFY-CRUX-C-11-001 shape: tool_calls[{index}].type={got:?} (expected \"function\")"
        )),
        clf::ToolCallsShapeOutcome::ArgumentsNotJson { index, .. } => Some(format!(
            "FALSIFY-CRUX-C-11-001 shape: tool_calls[{index}].arguments is not a JSON-parseable string"
        )),
    }
}

fn schema_fail_reason(o: &clf::SchemaValidationOutcome) -> Option<String> {
    match o {
        clf::SchemaValidationOutcome::Ok => None,
        clf::SchemaValidationOutcome::ArgumentsNotJson { .. } => Some(
            "FALSIFY-CRUX-C-11-002 schema: arguments is not a JSON-parseable string".to_string(),
        ),
        clf::SchemaValidationOutcome::ArgumentsNotObject => {
            Some("FALSIFY-CRUX-C-11-002 schema: arguments is not a JSON object".to_string())
        }
        clf::SchemaValidationOutcome::MissingRequiredProperty { name } => Some(format!(
            "FALSIFY-CRUX-C-11-002 schema: missing required property {name:?}"
        )),
        clf::SchemaValidationOutcome::WrongPropertyType {
            name,
            expected,
            got,
        } => Some(format!(
            "FALSIFY-CRUX-C-11-002 schema: property {name:?} type mismatch (expected {expected}, got {got})"
        )),
        clf::SchemaValidationOutcome::UnsupportedSchema { reason } => Some(format!(
            "FALSIFY-CRUX-C-11-002 schema: unsupported schema fragment: {reason}"
        )),
    }
}

fn passthrough_fail_reason(o: &clf::NoToolsPassthroughOutcome) -> Option<String> {
    match o {
        clf::NoToolsPassthroughOutcome::Ok => None,
        clf::NoToolsPassthroughOutcome::UnexpectedToolCalls { n_calls } => Some(format!(
            "FALSIFY-CRUX-C-11-003 passthrough: response synthesized {n_calls} tool_calls despite empty request.tools[]"
        )),
        clf::NoToolsPassthroughOutcome::WrongFinishReason {
            got,
            expected_any_of,
        } => Some(format!(
            "FALSIFY-CRUX-C-11-003 passthrough: finish_reason={got:?} (expected any of {expected_any_of:?})"
        )),
    }
}

fn print_report(
    path: &Path,
    shape: Option<&clf::ToolCallsShapeOutcome>,
    schema: Option<&clf::SchemaValidationOutcome>,
    passthrough: Option<&clf::NoToolsPassthroughOutcome>,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "observation_path": path.display().to_string(),
            "shape":       shape.map(|o| format!("{o:?}")),
            "schema":      schema.map(|o| format!("{o:?}")),
            "passthrough": passthrough.map(|o| format!("{o:?}")),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("tool-use-lint report for {}", path.display());
        print_line("  shape:       ", shape.map(|o| format!("{o:?}")));
        print_line("  schema:      ", schema.map(|o| format!("{o:?}")));
        print_line("  passthrough: ", passthrough.map(|o| format!("{o:?}")));
    }
}

fn print_line(prefix: &str, v: Option<String>) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}(missing fields — classifier skipped)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_obs(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn missing_file_is_file_not_found() {
        let err = run(Path::new("/no/such/tool.json"), false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn invalid_json_is_invalid_format() {
        let f = write_obs("not json");
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }

    #[test]
    fn empty_object_passes_no_gates() {
        let f = write_obs("{}");
        assert!(run(f.path(), false).is_ok());
    }

    #[test]
    fn empty_object_json_mode_passes() {
        let f = write_obs("{}");
        assert!(run(f.path(), true).is_ok());
    }
}
