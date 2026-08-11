//! `apr ollama-chat-lint` — CRUX-C-04 Ollama /api/chat response-shape gate.
//!
//! Reads an already-captured `/api/chat` response (from any Ollama-compatible
//! server — apr, ollama, etc.) and dispatches the pure classifiers in
//! `ollama_chat_classifier`. Non-streaming mode expects a single JSON object;
//! streaming mode expects NDJSON (one JSON object per line).
//!
//! Spec: `contracts/crux-C-04-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::ollama_chat_classifier::{
    classify_eval_metrics, classify_ndjson_stream, classify_ollama_chat_schema, EvalMetricsOutcome,
    NdjsonStreamOutcome, OllamaSchemaOutcome,
};
use crate::error::{CliError, Result};

pub(crate) fn run(response_file: &Path, stream: bool, json: bool) -> Result<()> {
    if !response_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(response_file)));
    }
    let body = std::fs::read_to_string(response_file)?;

    if stream {
        run_stream(&body, response_file, json)
    } else {
        run_non_streaming(&body, response_file, json)
    }
}

fn run_non_streaming(body: &str, path: &Path, json: bool) -> Result<()> {
    let response: Value = serde_json::from_str(body).map_err(|e| {
        CliError::InvalidInput(format!(
            "apr ollama-chat-lint: failed to parse JSON from {}: {e}",
            path.display()
        ))
    })?;

    let schema = classify_ollama_chat_schema(&response);
    let (eval_count, eval_duration, message_nonempty) = extract_eval_inputs(&response);
    let metrics = classify_eval_metrics(eval_count, eval_duration, message_nonempty);

    print_non_streaming_report(path, &schema, &metrics, json);

    match (&schema, &metrics) {
        (OllamaSchemaOutcome::Ok, EvalMetricsOutcome::Ok) => Ok(()),
        (OllamaSchemaOutcome::Ok, bad) => Err(CliError::ValidationFailed(format!(
            "eval-metrics gate rejected response: {bad:?}"
        ))),
        (bad, _) => Err(CliError::ValidationFailed(format!(
            "schema gate rejected response: {bad:?}"
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
            CliError::InvalidInput(format!(
                "apr ollama-chat-lint: invalid NDJSON at line {} of {}: {e}",
                idx + 1,
                path.display()
            ))
        })?;
        frames.push(frame);
    }
    let outcome = classify_ndjson_stream(&frames);
    print_stream_report(path, frames.len(), &outcome, json);
    match outcome {
        NdjsonStreamOutcome::Ok => Ok(()),
        bad => Err(CliError::ValidationFailed(format!(
            "NDJSON stream gate rejected frames: {bad:?}"
        ))),
    }
}

fn extract_eval_inputs(response: &Value) -> (u64, u64, bool) {
    let eval_count = response
        .get("eval_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let eval_duration = response
        .get("eval_duration")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let message_nonempty = response
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .map_or(false, |s| !s.is_empty());
    (eval_count, eval_duration, message_nonempty)
}

fn print_non_streaming_report(
    path: &Path,
    schema: &OllamaSchemaOutcome,
    metrics: &EvalMetricsOutcome,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "mode": "non_streaming",
            "response_path": path.display().to_string(),
            "schema_outcome": format!("{:?}", schema),
            "eval_metrics_outcome": format!("{:?}", metrics),
            "schema_ok": matches!(schema, OllamaSchemaOutcome::Ok),
            "eval_metrics_ok": matches!(metrics, EvalMetricsOutcome::Ok),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("ollama-chat-lint (non-streaming) for {}", path.display());
        println!("  schema_outcome:       {schema:?}");
        println!("  eval_metrics_outcome: {metrics:?}");
    }
}

fn print_stream_report(path: &Path, frame_count: usize, outcome: &NdjsonStreamOutcome, json: bool) {
    if json {
        let v = serde_json::json!({
            "mode": "streaming_ndjson",
            "response_path": path.display().to_string(),
            "num_frames": frame_count,
            "ndjson_outcome": format!("{:?}", outcome),
            "ndjson_ok": matches!(outcome, NdjsonStreamOutcome::Ok),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("ollama-chat-lint (streaming NDJSON) for {}", path.display());
        println!("  num_frames:     {frame_count}");
        println!("  ndjson_outcome: {outcome:?}");
    }
}
