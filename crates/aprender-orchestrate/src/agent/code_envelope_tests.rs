//! Tests for the #3775 / #3720 `apr code` JSON document.

use super::*;
use crate::agent::capability::Capability;

fn doc(result: Option<&AgentLoopResult>, outcome: Option<&CodeOutcome>) -> serde_json::Value {
    doc_on(result, outcome, None)
}

fn doc_on(
    result: Option<&AgentLoopResult>,
    outcome: Option<&CodeOutcome>,
    backend: Option<&BackendReport>,
) -> serde_json::Value {
    serde_json::from_str(&envelope(result, outcome, backend, std::time::Duration::from_millis(7)))
        .expect("the envelope is one JSON object")
}

fn loop_result(text: &str) -> AgentLoopResult {
    AgentLoopResult {
        text: text.into(),
        usage: crate::agent::result::TokenUsage { input_tokens: 11, output_tokens: 3 },
        iterations: 2,
        tool_calls: 1,
    }
}

#[test]
fn a_successful_run_is_status_ok_with_no_error_object() {
    let d = doc(Some(&loop_result("done")), None);
    assert_eq!(d["status"], "ok");
    assert_eq!(d["is_error"], false);
    assert_eq!(d["subtype"], "success");
    assert_eq!(d["result"], "done");
    assert!(d.get("error").is_none(), "an ok document carries no error object: {d}");
}

#[test]
fn a_failed_run_carries_kind_message_and_the_real_exit_code() {
    let e = AgentError::Driver(DriverError::InferenceFailed("apr serve HTTP 500: 0 layers".into()));
    let outcome = CodeOutcome::from_agent_error(&e, 1);
    let d = doc(None, Some(&outcome));
    assert_eq!(d["status"], "failed");
    assert_eq!(d["is_error"], true, "is_error == (status != ok)");
    assert_eq!(d["subtype"], "error");
    assert_eq!(d["error"]["kind"], "inference_failed");
    assert_eq!(d["error"]["exit_code"], 1);
    assert!(d["error"]["message"].as_str().is_some_and(|m| m.contains("0 layers")));
    assert_eq!(d["result"], "");
}

#[test]
fn every_agent_error_variant_has_a_snake_case_kind() {
    let cases = [
        (AgentError::Driver(DriverError::RateLimited { retry_after_ms: 1 }), "rate_limited"),
        (AgentError::Driver(DriverError::Overloaded { retry_after_ms: 1 }), "overloaded"),
        (AgentError::Driver(DriverError::ModelNotFound("m.gguf".into())), "model_not_found"),
        (AgentError::Driver(DriverError::InferenceFailed("x".into())), "inference_failed"),
        (AgentError::Driver(DriverError::Network("x".into())), "network_error"),
        (
            AgentError::ToolExecution { tool_name: "shell".into(), message: "x".into() },
            "tool_execution_failed",
        ),
        (AgentError::CircuitBreak("x".into()), "circuit_break"),
        (AgentError::MaxIterationsReached, "max_iterations_reached"),
        (AgentError::ContextOverflow { required: 2, available: 1 }, "context_overflow"),
        (AgentError::ManifestError("x".into()), "manifest_error"),
        (AgentError::Memory("x".into()), "memory_error"),
    ];
    for (e, kind) in cases {
        let o = CodeOutcome::from_agent_error(&e, 1);
        assert_eq!(o.kind, kind, "{e}");
        assert_eq!(o.status, "failed", "{e} ran and errored");
    }
}

#[test]
fn a_denied_capability_is_a_refusal_not_a_failure() {
    let e =
        AgentError::CapabilityDenied { tool_name: "shell".into(), required: Capability::Memory };
    let o = CodeOutcome::from_agent_error(&e, 4);
    assert_eq!((o.status, o.kind, o.exit_code), ("refused", "capability_denied", 4));
}

#[test]
fn an_empty_completion_fails_with_the_inference_failure_code() {
    let o = CodeOutcome::empty_completion(6, 5);
    assert_eq!((o.status, o.kind, o.exit_code), ("failed", "empty_completion", 1));
    let d = doc(Some(&loop_result("")), Some(&o));
    assert_eq!(d["status"], "failed");
    assert_eq!(d["num_turns"], 2, "the loop's counters are kept when the loop ran");
}

/// `anyhow::bail!(CodeOutcome …)` is how cmd_code's early refusals leave it;
/// the caller's `emit_error_document` recovers the kind by downcasting. If
/// bail! stopped preserving the value, every early refusal would become
/// `agent_error`.
#[test]
fn a_bailed_outcome_survives_as_a_downcastable_error() {
    fn refuse() -> anyhow::Result<()> {
        anyhow::bail!(CodeOutcome::refused("invalid_input", "--project: not a directory: x", 1));
    }
    let err = refuse().expect_err("bail! returns Err");
    let o = err.downcast_ref::<CodeOutcome>().expect("the outcome is recoverable by downcast");
    assert_eq!((o.status, o.kind), ("refused", "invalid_input"));
    assert_eq!(err.to_string(), "--project: not a directory: x", "stderr text is unchanged");
}

/// #3719: a forced-GPU run that fell back to the CPU used to print the same
/// document as one that stayed on the GPU. The document now carries what the
/// serve child reported, and a fallback says so even when the run succeeded.
#[test]
fn a_run_reports_the_device_its_completions_ran_on() {
    let fell = BackendReport { requested: "gpu", ran: Some("cpu"), fell_back: Some(true) };
    let d = doc_on(Some(&loop_result("done")), None, Some(&fell));
    assert_eq!(d["status"], "ok");
    assert_eq!(
        d["backend"],
        serde_json::json!({"requested": "gpu", "ran": "cpu", "fell_back": true})
    );

    let stayed = BackendReport { requested: "gpu", ran: Some("gpu"), fell_back: Some(false) };
    let d = doc_on(Some(&loop_result("done")), None, Some(&stayed));
    assert_eq!(d["backend"]["ran"], "gpu");
    assert_eq!(d["backend"]["fell_back"], false);

    // A failed turn still says what the completions before it ran on.
    let e = AgentError::Driver(DriverError::Network("apr serve HTTP 500: x".into()));
    let d = doc_on(None, Some(&CodeOutcome::from_agent_error(&e, 1)), Some(&fell));
    assert_eq!(d["backend"]["fell_back"], true);
}

/// No driver ran (an early refusal), or the driver cannot say: the key is
/// present and `null`, which a consumer reads as not measured, never as a pass.
#[test]
fn a_document_with_no_backend_report_says_null() {
    let d = doc(None, Some(&CodeOutcome::refused("no_model", "no model", 3)));
    assert!(d.get("backend").is_some_and(serde_json::Value::is_null), "{d}");
    let d = doc(Some(&loop_result("done")), None);
    assert!(d.get("backend").is_some_and(serde_json::Value::is_null), "{d}");
}
