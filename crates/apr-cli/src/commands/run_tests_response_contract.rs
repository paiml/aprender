// #3720 — the apr response contract on `apr run --json` / `--stream`: every exit writes
// one document with `status` (ok | refused | failed) and, unless ok, `error {kind,
// message, exit_code}`; an empty completion is a failure, never exit 0 with "".

fn contract_result(text: &str, reasoning: Option<&str>) -> RunResult {
    RunResult {
        text: text.to_string(),
        reasoning: reasoning.map(str::to_string),
        thinking: reasoning.is_some(),
        reasoning_truncated: false,
        duration_secs: 1.0,
        cached: true,
        tokens_generated: Some(3),
        tok_per_sec: Some(3.0),
        used_gpu: Some(true),
        generated_tokens: Some(vec![1, 2, 3]),
        token_texts: None,
    }
}

/// Every variant: its kind, its status, its exit code. A new variant that is not
/// classified fails to compile (the matches are exhaustive); this pins the values.
#[test]
fn every_cli_error_has_the_contract_kind_status_and_exit_code() {
    use crate::error::{CliError, RunStatus};
    let p = std::path::PathBuf::from("m.gguf");
    let s = || "x".to_string();
    let table: Vec<(CliError, &str, RunStatus, u8)> = vec![
        (CliError::FileNotFound(p.clone()), "file_not_found", RunStatus::Refused, 3),
        (CliError::NotAFile(p), "not_a_file", RunStatus::Refused, 3),
        (CliError::InvalidFormat(s()), "invalid_format", RunStatus::Refused, 4),
        (CliError::InvalidInput(s()), "invalid_input", RunStatus::Refused, 4),
        (
            CliError::InvalidModelFile { format: "GGUF", message: s() },
            "invalid_model_file",
            RunStatus::Refused,
            4,
        ),
        (CliError::Io(std::io::Error::other("x")), "io_error", RunStatus::Failed, 7),
        (CliError::ValidationFailed(s()), "validation_failed", RunStatus::Failed, 5),
        (CliError::Aprender(s()), "aprender_error", RunStatus::Failed, 1),
        (CliError::ModelLoadFailed(s()), "model_load_failed", RunStatus::Failed, 6),
        (CliError::InferenceFailed(s()), "inference_failed", RunStatus::Failed, 8),
        (CliError::FeatureDisabled(s()), "feature_disabled", RunStatus::Refused, 9),
        (CliError::NetworkError(s()), "network_error", RunStatus::Failed, 10),
        (CliError::HttpNotFound(s()), "http_not_found", RunStatus::Refused, 11),
        (CliError::NotImplemented(s()), "not_implemented", RunStatus::Refused, 12),
        (CliError::ParityFailed(s()), "parity_failed", RunStatus::Refused, 13),
        (CliError::BackendUnavailable(s()), "backend_unavailable", RunStatus::Refused, 14),
        (
            CliError::ThinkingModeUnsupported(s()),
            "thinking_mode_unsupported",
            RunStatus::Refused,
            15,
        ),
        (CliError::ThinkBlockUnclosed(s()), "think_block_unclosed", RunStatus::Failed, 8),
        (CliError::EmptyCompletion(s()), "empty_completion", RunStatus::Failed, 8),
    ];
    for (e, kind, status, code) in &table {
        assert_eq!(e.kind(), *kind, "{e:?}");
        assert_eq!(e.status(), *status, "{e:?}");
        assert_eq!(e.exit_code_value(), *code, "{e:?}");
        let env = e.envelope();
        assert_eq!(env["status"], status.as_str());
        assert_eq!(env["error"]["kind"], *kind);
        assert_eq!(env["error"]["exit_code"], u64::from(*code), "never re-mapped");
        assert_eq!(env["error"]["message"], e.to_string());
    }
}

#[test]
fn a_run_that_ended_ok_says_so_and_carries_no_error() {
    let doc = final_document(&contract_result("4", None), "m.gguf", 16, false, None);
    assert_eq!(doc["status"], "ok");
    assert!(doc.get("error").is_none(), "error iff status != ok: {doc}");
    assert_eq!(doc["text"], "4");
}

#[test]
fn an_empty_completion_is_a_failure_that_names_why_and_keeps_the_reasoning() {
    let nothing = contract_result("", None);
    let err = require_answer(&nothing).expect_err("empty is not ok");
    assert_eq!(err.kind(), "empty_completion");
    assert_eq!(err.exit_code_value(), 8);
    assert!(err.to_string().contains("generation produced no text"), "{err}");

    // An answer empty AFTER a closed think block: still empty_completion, and the
    // reasoning is still reported (#3720, accepted schema).
    let only_reasoning = contract_result("  ", Some("2 plus 2"));
    let err = require_answer(&only_reasoning).expect_err("reasoning alone is not an answer");
    assert!(err.to_string().contains("after the think block"), "{err}");
    let doc = final_document(&only_reasoning, "m.gguf", 16, false, Some(&err));
    assert_eq!(doc["status"], "failed");
    assert_eq!(doc["error"]["kind"], "empty_completion");
    assert_eq!(doc["reasoning"], "2 plus 2");

    require_answer(&contract_result("4", None)).expect("an answer is ok");
}

#[test]
fn a_run_that_never_produced_a_result_still_writes_one_document() {
    let e = crate::error::CliError::ThinkingModeUnsupported(
        "thinking on refused: this model's chat template supports thinking off only".into(),
    );
    let doc = error_document(&e, "qwen2.5.gguf", false);
    assert_eq!(doc["status"], "refused");
    assert_eq!(doc["error"]["kind"], "thinking_mode_unsupported");
    assert_eq!(doc["error"]["exit_code"], 15);
    assert_eq!(doc["model"], "qwen2.5.gguf");
    assert!(doc.get("event").is_none());
    assert_eq!(error_document(&e, "m", true)["event"], "final", "the stream's terminal event");
}

#[test]
fn the_stream_final_event_carries_the_envelope() {
    let mut buf = Vec::new();
    let err = crate::error::CliError::EmptyCompletion("empty completion: x".into());
    write_stream_output_with(&mut buf, &contract_result("", None), "m.gguf", 8, false, Some(&err))
        .expect("write");
    let last = String::from_utf8(buf).expect("utf8");
    let last = last.lines().last().expect("a final line");
    let v: serde_json::Value = serde_json::from_str(last).expect("json");
    assert_eq!(v["event"], "final");
    assert_eq!(v["status"], "failed");
    assert_eq!(v["error"]["kind"], "empty_completion");
}
