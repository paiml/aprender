//! `apr ollama-tools-lint` — CRUX-I-04 Ollama function-calling response gate.
//!
//! Reads an already-captured `/api/chat` response that was produced against a
//! request with a `tools[]` array, and dispatches the pure classifiers in
//! `ollama_tool_call_classifier`. Non-streaming mode expects a single JSON
//! object; streaming mode expects NDJSON (one JSON object per line).
//!
//! The optional `--request-file` carries the original request so we can
//! enforce the tool-name allowlist (no model hallucinated a tool the client
//! did not declare).
//!
//! Spec: `contracts/crux-I-04-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::ollama_tool_call_classifier::{
    classify_streaming_tool_call, classify_tool_call_schema, classify_tool_name_allowlist,
    StreamingToolCallOutcome, ToolCallSchemaOutcome, ToolNameAllowlistOutcome,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    response_file: &Path,
    request_file: Option<&Path>,
    stream: bool,
    json: bool,
) -> Result<()> {
    if !response_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(response_file)));
    }
    let body = std::fs::read_to_string(response_file)?;

    if stream {
        // The streaming gate never consults the allowlist.
        return run_stream(&body, response_file, json);
    }

    // The allowlist gate runs unconditionally in non-streaming mode, so
    // without a request file it compares every call against an empty declared
    // set: a response WITH tool calls got `NoDeclaredTools` and one WITHOUT
    // got `MissingToolCalls`. No response could pass, and the help text called
    // the flag "Optional". It is required here.
    let Some(request_file) = request_file else {
        return Err(CliError::ValidationFailed(
            "apr ollama-tools-lint: --request-file is required in non-streaming mode — the \
             tool-name allowlist gate has nothing to check a response against without the \
             request that declared the tools."
                .to_string(),
        ));
    };
    let declared = load_declared_tool_names(request_file)?;
    run_non_streaming(&body, response_file, &declared, json)
}

fn run_non_streaming(body: &str, path: &Path, declared: &[String], json: bool) -> Result<()> {
    let response: Value = serde_json::from_str(body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr ollama-tools-lint: failed to parse JSON from {}: {e}",
            path.display()
        ))
    })?;

    let schema = classify_tool_call_schema(&response);
    let declared_refs: Vec<&str> = declared.iter().map(String::as_str).collect();
    let allowlist = classify_tool_name_allowlist(&response, &declared_refs);

    print_non_streaming_report(path, &schema, &allowlist, declared, json);

    match (&schema, &allowlist) {
        (ToolCallSchemaOutcome::Ok, ToolNameAllowlistOutcome::Ok) => Ok(()),
        (ToolCallSchemaOutcome::Ok, bad) => Err(CliError::ValidationFailed(format!(
            "tool-name allowlist gate rejected response: {bad:?}"
        ))),
        (bad, _) => Err(CliError::ValidationFailed(format!(
            "tool-call schema gate rejected response: {bad:?}"
        ))),
    }
}

fn run_stream(body: &str, path: &Path, json: bool) -> Result<()> {
    let mut frames: Vec<Value> = Vec::new();
    for (idx, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let frame: Value = serde_json::from_str(trimmed).map_err(|e| {
            CliError::InvalidFormat(format!(
                "apr ollama-tools-lint: invalid NDJSON at line {} of {}: {e}",
                idx + 1,
                path.display()
            ))
        })?;
        frames.push(frame);
    }
    let outcome = classify_streaming_tool_call(&frames);
    print_stream_report(path, frames.len(), &outcome, json);
    match outcome {
        StreamingToolCallOutcome::Ok => Ok(()),
        bad => Err(CliError::ValidationFailed(format!(
            "NDJSON tool-call stream gate rejected frames: {bad:?}"
        ))),
    }
}

fn load_declared_tool_names(path: &Path) -> Result<Vec<String>> {
    if !path.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(path)));
    }
    let body = std::fs::read_to_string(path)?;
    let req: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr ollama-tools-lint: failed to parse request JSON from {}: {e}",
            path.display()
        ))
    })?;
    let names = req
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .map(String::from)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(names)
}

fn print_non_streaming_report(
    path: &Path,
    schema: &ToolCallSchemaOutcome,
    allowlist: &ToolNameAllowlistOutcome,
    declared: &[String],
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "mode": "non_streaming",
            "response_path": path.display().to_string(),
            "declared_tool_names": declared,
            "schema_outcome": format!("{:?}", schema),
            "allowlist_outcome": format!("{:?}", allowlist),
            "schema_ok": matches!(schema, ToolCallSchemaOutcome::Ok),
            "allowlist_ok": matches!(allowlist, ToolNameAllowlistOutcome::Ok),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("ollama-tools-lint (non-streaming) for {}", path.display());
        println!("  declared_tools:    {declared:?}");
        println!("  schema_outcome:    {schema:?}");
        println!("  allowlist_outcome: {allowlist:?}");
    }
}

fn print_stream_report(
    path: &Path,
    frame_count: usize,
    outcome: &StreamingToolCallOutcome,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "mode": "streaming_ndjson",
            "response_path": path.display().to_string(),
            "num_frames": frame_count,
            "ndjson_outcome": format!("{:?}", outcome),
            "ndjson_ok": matches!(outcome, StreamingToolCallOutcome::Ok),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!(
            "ollama-tools-lint (streaming NDJSON) for {}",
            path.display()
        );
        println!("  num_frames:     {frame_count}");
        println!("  ndjson_outcome: {outcome:?}");
    }
}

#[cfg(test)]
mod cov_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    fn w(s: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(s.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }
    #[test]
    fn missing_response_is_file_not_found() {
        let err = run(Path::new("/no/such/resp.json"), None, false, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn missing_request_file_errors() {
        let resp = w("{}");
        let err = run(
            resp.path(),
            Some(Path::new("/no/such/req.json")),
            false,
            false,
        );
        assert!(err.is_err());
    }
    /// Was `run(f.path(), None, false, false)`, which passed for the wrong
    /// reason once the missing-request-file check landed ahead of the parse.
    #[test]
    fn malformed_response_errors() {
        let f = w("not a json response");
        let req = w(r#"{"tools":[]}"#);
        let err = run(f.path(), Some(req.path()), false, false).unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }

    const RESP_ONE_CALL: &str = r#"{"model":"q","created_at":"t","done":true,"message":
        {"role":"assistant","content":"","tool_calls":
        [{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]}}"#;

    /// No response at all could pass `apr ollama-tools-lint --response-file X`:
    /// with tool calls it was `NoDeclaredTools`, without them
    /// `MissingToolCalls`. The command now says which flag is missing.
    #[test]
    fn falsifier_non_streaming_without_request_file_names_the_missing_flag() {
        let two_calls = r#"{"model":"q","created_at":"t","done":true,"message":
            {"role":"assistant","content":"","tool_calls":
            [{"function":{"name":"a","arguments":{}}},{"function":{"name":"b","arguments":{}}}]}}"#;
        let no_calls = r#"{"model":"q","created_at":"t","done":true,"message":
            {"role":"assistant","content":"hello"}}"#;

        for body in [RESP_ONE_CALL, two_calls, no_calls] {
            let resp = w(body);
            let err = run(resp.path(), None, false, false).unwrap_err();
            match err {
                CliError::ValidationFailed(msg) => {
                    assert!(msg.contains("--request-file is required"), "{msg}");
                    assert!(!msg.contains("NoDeclaredTools"), "{msg}");
                }
                other => panic!("expected ValidationFailed, got {other:?}"),
            }
        }
    }

    /// With the flag the gate is still a real gate in both directions.
    #[test]
    fn allowlist_gate_still_passes_and_still_catches_a_hallucinated_name() {
        let resp = w(RESP_ONE_CALL);
        let matching = w(r#"{"model":"q","tools":[{"type":"function",
            "function":{"name":"get_weather","parameters":{}}}]}"#);
        assert!(run(resp.path(), Some(matching.path()), false, false).is_ok());

        let other = w(r#"{"model":"q","tools":[{"type":"function",
            "function":{"name":"get_stock","parameters":{}}}]}"#);
        let err = run(resp.path(), Some(other.path()), false, false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("Hallucinated"), "{msg}"),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    /// Streaming mode does not use the allowlist, so it must stay usable
    /// without the flag.
    #[test]
    fn stream_mode_does_not_require_request_file() {
        let ndjson = w(concat!(
            r#"{"model":"q","message":{"role":"assistant","content":"","tool_calls":"#,
            r#"[{"function":{"name":"get_weather","arguments":{}}}]},"done":false}"#,
            "\n",
            r#"{"model":"q","message":{"role":"assistant","content":""},"done":true}"#,
        ));
        let res = run(ndjson.path(), None, true, false);
        assert!(
            !matches!(&res, Err(CliError::ValidationFailed(m)) if m.contains("--request-file")),
            "stream mode must not demand --request-file: {res:?}"
        );
    }
}
