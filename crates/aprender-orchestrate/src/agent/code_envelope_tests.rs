//! Tests for the #3775 / #3720 `apr code` JSON document.

use super::*;
use crate::agent::capability::Capability;

fn doc(result: Option<&AgentLoopResult>, outcome: Option<&CodeOutcome>) -> serde_json::Value {
    serde_json::from_str(&envelope(result, outcome, std::time::Duration::from_millis(7)))
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
