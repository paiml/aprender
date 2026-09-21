//! One JSON document per `apr code -p --output-format json` run (#3775).
//!
//! The error fields are the #3720 response contract as aprender-fd accepted it
//! (<https://github.com/paiml/aprender/issues/3720#issuecomment-5767761121>):
//!
//! * `status`: `"ok"` | `"refused"` | `"failed"`, on every document. Refused
//!   means a limit or policy said no; failed means the work ran and errored.
//! * `error`: `{kind, message, exit_code}`, present iff `status != "ok"`.
//!   `kind` is the snake_case name of the error variant, and `exit_code` is the
//!   process's real exit status, never re-mapped.
//!
//! The Claude-Code-parity fields (`type`, `subtype`, `is_error`, `result`, …)
//! stay beside them, with `is_error == (status != "ok")`.
//!
//! Before #3775 a driver error (the serve child failing to load, an HTTP 500)
//! printed only to stderr and left stdout EMPTY, and an empty completion
//! printed an error document but exited 0. A consumer asking for JSON could
//! not tell failure from nothing.

use super::code_prompts::exit_code;
use super::result::{AgentError, AgentLoopResult, DriverError};

/// How a non-interactive run ended, in the #3720 vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeOutcome {
    /// `"refused"` or `"failed"`; a successful run has no outcome.
    pub status: &'static str,
    /// snake_case name of the error variant.
    pub kind: &'static str,
    /// Human-readable reason, the same text stderr carries.
    pub message: String,
    /// The process exit status this run ends with.
    pub exit_code: i32,
}

impl CodeOutcome {
    /// A refusal: a limit, policy or missing input said no.
    pub fn refused(kind: &'static str, message: impl Into<String>, exit_code: i32) -> Self {
        Self { status: "refused", kind, message: message.into(), exit_code }
    }

    /// A failure: the work ran and errored.
    pub fn failed(kind: &'static str, message: impl Into<String>, exit_code: i32) -> Self {
        Self { status: "failed", kind, message: message.into(), exit_code }
    }

    /// The outcome of an agent-loop error. `exit_code` is the code the caller
    /// exits with (`map_error_to_exit_code`), so the document cannot disagree
    /// with the process status.
    pub fn from_agent_error(e: &AgentError, exit_code: i32) -> Self {
        let kind = match e {
            AgentError::Driver(d) => match d {
                DriverError::RateLimited { .. } => "rate_limited",
                DriverError::Overloaded { .. } => "overloaded",
                DriverError::ModelNotFound(_) => "model_not_found",
                DriverError::InferenceFailed(_) => "inference_failed",
                DriverError::Network(_) => "network_error",
            },
            AgentError::ToolExecution { .. } => "tool_execution_failed",
            AgentError::CapabilityDenied { .. } => "capability_denied",
            AgentError::CircuitBreak(_) => "circuit_break",
            AgentError::MaxIterationsReached => "max_iterations_reached",
            AgentError::ContextOverflow { .. } => "context_overflow",
            AgentError::ManifestError(_) => "manifest_error",
            AgentError::Memory(_) => "memory_error",
        };
        let message = e.to_string();
        if matches!(e, AgentError::CapabilityDenied { .. }) {
            Self::refused(kind, message, exit_code)
        } else {
            Self::failed(kind, message, exit_code)
        }
    }

    /// An empty completion is a failure, not an answer (#3720 done_when 3).
    /// aprender-fd's ruling: it exits with the command's own inference-failure
    /// code, which for `apr code` is `AGENT_ERROR` (1).
    pub fn empty_completion(iterations: u32, tool_calls: u32) -> Self {
        Self::failed(
            "empty_completion",
            format!("the model returned no answer text ({iterations} iterations, {tool_calls} tool calls)"),
            exit_code::AGENT_ERROR,
        )
    }
}

impl std::fmt::Display for CodeOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// A `CodeOutcome` travels as an `anyhow::Error` out of `cmd_code`'s early
/// refusals, so the one place that writes the document can recover its kind.
impl std::error::Error for CodeOutcome {}

/// A UUIDv7-shaped id derived from the wall clock (the same shape
/// `emit_ccpa_trace` uses), stable for one run.
fn session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts_micros =
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_micros()).unwrap_or(0);
    format!(
        "{:08x}-{:04x}-7000-{:04x}-{:012x}",
        (ts_micros >> 64) as u32 & 0xFFFF_FFFF,
        ((ts_micros >> 48) & 0xFFFF) as u16,
        ((ts_micros >> 32) & 0xFFFF) as u16,
        (ts_micros & 0xFFFF_FFFF_FFFF) as u64
    )
}

/// The one JSON document a `-p --output-format json` run writes to stdout.
///
/// `result` is the loop's result when the loop ran; `outcome` is `None` for a
/// successful run and names the refusal or failure otherwise.
pub fn envelope(
    result: Option<&AgentLoopResult>,
    outcome: Option<&CodeOutcome>,
    elapsed: std::time::Duration,
) -> String {
    let mut doc = serde_json::json!({
        "type": "result",
        "subtype": if outcome.is_some() { "error" } else { "success" },
        "is_error": outcome.is_some(),
        "status": outcome.map_or("ok", |o| o.status),
        "duration_ms": elapsed.as_millis() as u64,
        "result": result.map_or("", |r| r.text.as_str()),
        "session_id": session_id(),
        "num_turns": result.map_or(0, |r| r.iterations),
        "tokens_in": result.map_or(0, |r| r.usage.input_tokens),
        "tokens_out": result.map_or(0, |r| r.usage.output_tokens),
        // Local sovereign inference: cost is always zero by construction.
        "total_cost_usd": 0,
    });
    if let Some(o) = outcome {
        doc["error"] = serde_json::json!({
            "kind": o.kind,
            "message": o.message,
            "exit_code": o.exit_code,
        });
    }
    doc.to_string()
}

#[cfg(test)]
#[path = "code_envelope_tests.rs"]
mod tests;
