//! #3793: which loop a constrained run takes, what it reports, and when it refuses.

use super::*;
use crate::constrain::{ConstraintError, ConstraintRequest};
use crate::gguf::ConstrainedStop;

fn refusal_kind(e: RealizarError) -> ConstraintError {
    match e {
        RealizarError::Constraint(c) => c,
        other => panic!("not a constraint refusal: {other}"),
    }
}

/// apr's default Qwen3 template prefills a CLOSED think block: no reasoning comes first, so a
/// constraint may apply from the first token. An OPEN block means reasoning comes first.
#[test]
fn only_a_prompt_that_leaves_think_open_is_a_thinking_prompt() {
    let no_think = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let open = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let plain = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    assert!(!prompt_opens_thinking(no_think));
    assert!(prompt_opens_thinking(open));
    assert!(!prompt_opens_thinking(plain));
    // An earlier turn's closed block does not close a new open one
    assert!(prompt_opens_thinking(&format!(
        "{no_think}ok<|im_end|>\n{open}"
    )));
}

#[test]
fn a_constrained_finish_is_reported_as_itself() {
    use run_report::FinishReason;
    assert_eq!(
        gguf_finish_reason(Some(ConstrainedStop::Complete), &[1, 2], &[9], 64),
        FinishReason::ConstraintComplete
    );
    assert_eq!(
        gguf_finish_reason(Some(ConstrainedStop::Length), &[1, 2], &[9], 2),
        FinishReason::Length
    );
    // Unconstrained runs keep #3718's rule
    assert_eq!(
        gguf_finish_reason(None, &[1, 9], &[9], 64),
        FinishReason::Stop
    );
    assert_eq!(
        gguf_finish_reason(None, &[1, 2], &[9], 2),
        FinishReason::Length
    );
}

#[test]
fn a_format_that_applies_no_constraint_refuses_a_constrained_run_only() {
    let plain = InferenceConfig::new("m.apr");
    assert!(refuse_constraint_on(&plain, "apr", "x").is_ok());
    let constrained = InferenceConfig::new("m.apr")
        .with_constraint(ConstraintRequest::Lark(r#"start: "a""#.to_string()));
    match refusal_kind(
        refuse_constraint_on(&constrained, "apr", "not scheduled").expect_err("refused"),
    ) {
        ConstraintError::UnsupportedPath { path, removed_by } => {
            assert_eq!(path, "apr");
            assert_eq!(removed_by, "not scheduled");
        },
        other => panic!("{other}"),
    }
}

/// The mock backend refuses too: a constraint is never silently dropped on any path.
#[test]
fn the_mock_backend_refuses_a_constrained_run() {
    let mut config = InferenceConfig::new("m.gguf")
        .with_prompt("hi")
        .with_constraint(ConstraintRequest::Lark(r#"start: "a""#.to_string()));
    config.use_mock_backend = true;
    let e = run_inference_report(&config).expect_err("refused");
    assert!(
        matches!(refusal_kind(e), ConstraintError::UnsupportedPath { path, .. } if path == "mock")
    );
}

#[test]
fn the_moe_loop_refuses_and_no_gpu_takes_the_cpu_loop() {
    use crate::gguf::test_factory::build_executable_pygmy_gguf;
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp");
    f.write_all(&build_executable_pygmy_gguf()).expect("write");
    let mapped = crate::gguf::MappedGGUFModel::from_path(f.path()).expect("map");
    let model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("load");
    let cpu = InferenceConfig::new(f.path()).without_gpu();
    assert_eq!(unconstrained_gguf_path(&cpu, &model, false, "llama"), None);
    assert_eq!(unconstrained_gguf_path(&cpu, &model, true, "qwen35"), None);
    // The MoE loop refuses even with --no-gpu: it applies no constraint on any device
    assert_eq!(
        unconstrained_gguf_path(&cpu, &model, false, "qwen3_moe").map(|(path, _)| path),
        Some("qwen3-moe")
    );
}
