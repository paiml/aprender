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

use crate::commands::lint_vacuity::{assert_not_vacuous, skipped_label, SectionRun};
use crate::commands::tool_use_classifier as clf;
use crate::error::{CliError, Result};

/// Each classifier reads exactly one top-level section of the same name.
static SECTION_NAMES: [&str; 3] = ["shape", "schema", "passthrough"];

pub(crate) fn run(observation_file: &Path, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidInput(format!(
            "apr tool-use-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    let shape = classify_shape(&obs);
    let schema = classify_schema(&obs);
    let passthrough = classify_passthrough(&obs);

    let mut fail_reasons: Vec<String> = [
        shape.as_ref().and_then(shape_fail_reason),
        schema.as_ref().and_then(schema_fail_reason),
        passthrough.as_ref().and_then(passthrough_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        &obs,
        shape.as_ref(),
        schema.as_ref(),
        passthrough.as_ref(),
        json,
    );

    // CRUX-C-11 is a falsifier surface: a green run is read as a discharged
    // proof obligation, so a run that discharged nothing must be red.
    let ran = [shape.is_some(), schema.is_some(), passthrough.is_some()];
    let sections: Vec<SectionRun> = SECTION_NAMES
        .iter()
        .zip(ran)
        .map(|(name, ran)| SectionRun {
            name,
            keys: std::slice::from_ref(name),
            ran,
        })
        .collect();
    if let Err(reason) = assert_not_vacuous("FALSIFY-CRUX-C-11", &obs, &sections) {
        fail_reasons.push(reason);
    }

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
    obs: &Value,
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
        print_line(
            "  shape:       ",
            shape.map(|o| format!("{o:?}")),
            obs,
            "shape",
        );
        print_line(
            "  schema:      ",
            schema.map(|o| format!("{o:?}")),
            obs,
            "schema",
        );
        print_line(
            "  passthrough: ",
            passthrough.map(|o| format!("{o:?}")),
            obs,
            "passthrough",
        );
    }
}

fn print_line(prefix: &str, v: Option<String>, obs: &Value, key: &str) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}{}", skipped_label(obs, &[key])),
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
        assert!(matches!(err, CliError::InvalidInput(_)));
    }

    /// Both of these asserted `is_ok()` on `{}`. CRUX-C-11 is a falsifier
    /// surface, so a green run is read as a discharged proof obligation —
    /// these two tests certified that an observation asserting nothing
    /// discharged it.
    #[test]
    fn falsifier_empty_object_is_rejected_in_both_output_modes() {
        for json in [false, true] {
            let f = write_obs("{}");
            match run(f.path(), json).unwrap_err() {
                CliError::ValidationFailed(msg) => assert!(
                    msg.contains("has none of shape/schema/passthrough"),
                    "json={json}: {msg}"
                ),
                other => panic!("json={json}: expected ValidationFailed, got {other:?}"),
            }
        }
    }

    #[test]
    fn falsifier_section_name_typo_is_rejected() {
        let f = write_obs(
            r#"{"shpae":{"declared_tools":[],"tool_calls":[],"finish_reason":"tool_calls"}}"#,
        );
        match run(f.path(), false).unwrap_err() {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("no gate ran"), "{msg}");
            }
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    /// A present-but-wrong-typed section is a schema error, not a skip.
    #[test]
    fn falsifier_wrong_typed_section_is_rejected() {
        for body in [
            r#"{"shape":{"declared_tools":[],"tool_calls":"notarray","finish_reason":"tool_calls"}}"#,
            r#"{"schema":{"parameters":{"type":"object"},"arguments":123}}"#,
        ] {
            let f = write_obs(body);
            match run(f.path(), false).unwrap_err() {
                CliError::ValidationFailed(msg) => {
                    assert!(msg.contains("present but unusable"), "{body}: {msg}");
                }
                other => panic!("{body}: expected ValidationFailed, got {other:?}"),
            }
        }
    }

    /// Control: a well-formed violating observation still fails through the
    /// classifier, not through the vacuity guard.
    #[test]
    fn well_formed_violation_still_reports_the_real_reason() {
        let f = write_obs(
            r#"{"shape":{"declared_tools":[{"name":"get_weather","parameters":{"type":"object"}}],
                "tool_calls":[{"id":"c1","type":"function","name":"NOT_DECLARED","arguments":"{}"}],
                "finish_reason":"tool_calls"}}"#,
        );
        match run(f.path(), false).unwrap_err() {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("NOT_DECLARED"), "{msg}");
                assert!(!msg.contains("present but unusable"), "{msg}");
            }
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }
}
