//! #5620: every branch of `qwen35_completion_plan` returns the plan (or the refusal) the
//! `/v1/completions` Qwen3.5 arm is specified to give. The hybrid file is the one the sibling chat
//! tests use (`APR_QWEN35_MODEL` overrides); an absent file skips on a dev box.

use super::*;
use crate::api::Qwen35Served;
use std::sync::Arc;
use crate::gguf::qwen35_session::Qwen35Session;
use crate::gguf::MappedGGUFModel;

const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

fn state_or_skip() -> Option<AppState> {
    let path = std::env::var("APR_QWEN35_MODEL")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| MODEL_PATH.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: {path} is absent");
        return None;
    }
    let mapped = Arc::new(MappedGGUFModel::from_path(&path).expect("map the GGUF"));
    let vocab = mapped.model.vocabulary().expect("vocabulary");
    let session = Qwen35Session::load(&mapped, true).expect("load the hybrid");
    Some(AppState::with_qwen35_session(session, mapped, vocab).expect("app state"))
}

fn request(prompt: &str) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({ "model": "m", "prompt": prompt }))
        .expect("a minimal completion request")
}

fn status_and_error(e: RErr) -> (StatusCode, String) {
    (e.0, e.1 .0.error)
}

#[test]
fn a_state_without_a_qwen35_session_declines_so_another_arm_owns_the_request() {
    let state = AppState::demo().expect("demo state");
    let plan = qwen35_completion_plan(&state, &request("hi"), 8, 0.0, &CancelToken::never());
    assert!(matches!(plan, Ok(None)));
}

#[test]
fn the_plan_carries_the_raw_prompt_tokens_and_a_budget_capped_by_max_tokens() {
    let Some(state) = state_or_skip() else { return };
    let mapped = state.mapped_gguf_model().expect("mapped");
    let want = mapped.model.encode("The capital of Peru is").expect("encode");
    let plan = qwen35_completion_plan(
        &state,
        &request("The capital of Peru is"),
        5,
        0.0,
        &CancelToken::never(),
    )
    .ok()
    .flatten()
    .expect("a Qwen3.5 state plans");
    assert_eq!(plan.input_ids, want, "the RAW prompt, no chat template");
    assert_eq!(plan.prompt_tokens, want.len());
    assert_eq!(plan.budget, 5);
    assert_eq!(plan.gen_config.max_tokens, 5);
    assert_eq!(plan.gen_config.temperature, 0.0);
    assert_eq!(plan.gen_config.stop_tokens, plan.stop_tokens);
    if let Some(eos) = state.model_eos_token_id() {
        assert!(plan.stop_tokens.contains(&eos));
    }
}

#[test]
fn the_budget_is_what_the_context_leaves_when_max_tokens_asks_for_more() {
    let Some(state) = state_or_skip() else { return };
    let ctx = state.qwen35_session().expect("session").context_length;
    let plan = qwen35_completion_plan(&state, &request("Hello"), usize::MAX, 0.0, &CancelToken::never())
        .ok()
        .flatten()
        .expect("planned");
    assert_eq!(plan.budget, ctx - plan.prompt_tokens);
    assert_eq!(plan.gen_config.max_tokens, plan.budget);
}

#[test]
fn an_empty_prompt_is_400() {
    let Some(state) = state_or_skip() else { return };
    let err = qwen35_completion_plan(&state, &request(""), 8, 0.0, &CancelToken::never())
        .err()
        .expect("refused");
    let (status, msg) = status_and_error(err);
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(msg.contains("Prompt cannot be empty"), "{msg}");
}

#[test]
fn a_prompt_that_fills_the_context_is_refused_whole_with_both_numbers() {
    let Some(mut state) = state_or_skip() else { return };
    let served = Arc::try_unwrap(state.qwen35_session.take().expect("session"))
        .unwrap_or_else(|_| panic!("the test holds the only handle"));
    let n = state
        .mapped_gguf_model()
        .expect("mapped")
        .model
        .encode("Hello there, how are you today?")
        .expect("encode")
        .len();
    state.qwen35_session = Some(Arc::new(Qwen35Served {
        context_length: n, // prompt_tokens >= context_length
        ..served
    }));
    let err = qwen35_completion_plan(
        &state,
        &request("Hello there, how are you today?"),
        8,
        0.0,
        &CancelToken::never(),
    )
    .err()
    .expect("refused");
    let (status, msg) = status_and_error(err);
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        msg.contains(&format!("{n} tokens")) && msg.contains("refused whole"),
        "{msg}"
    );
}

#[test]
fn stop_tokens_are_the_bare_eos_without_a_tokenizer_and_empty_without_an_eos() {
    assert_eq!(qwen35_completion_stop_tokens(None, Some(7)), vec![7]);
    assert!(qwen35_completion_stop_tokens(None, None).is_empty());
}
