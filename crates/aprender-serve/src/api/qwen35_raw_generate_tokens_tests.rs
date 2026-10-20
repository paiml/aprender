//! #5619: every branch of `qwen35_raw_tokens`. The hybrid file is the one the sibling Qwen3.5
//! tests use (`APR_QWEN35_MODEL` overrides); an absent file skips on a dev box.

use super::*;
use crate::api::Qwen35Served;
use crate::gguf::qwen35_session::Qwen35Session;
use crate::gguf::MappedGGUFModel;
use std::sync::Arc;

const MODEL_PATH: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";
const PROMPT: &str = "The capital of Peru is";

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

fn refusal(state: &AppState, prompt: &str) -> (StatusCode, String) {
    match qwen35_raw_tokens(state, prompt, 4, 0.0, &CancelToken::never()) {
        Err(e) => (e.0, e.1 .0.error),
        Ok(_) => panic!("expected a refusal for {prompt:?}"),
    }
}

#[test]
fn a_state_without_a_session_declines_directly() {
    let state = AppState::demo().expect("demo state");
    let got = qwen35_raw_tokens(&state, "hi", 4, 0.0, &CancelToken::never());
    assert!(matches!(got, Ok(None)));
}

#[test]
fn the_result_is_the_raw_prompt_ids_then_at_most_max_tokens_generated() {
    let Some(state) = state_or_skip() else { return };
    let want = state
        .mapped_gguf_model()
        .expect("mapped")
        .model
        .encode(PROMPT)
        .expect("encode");
    let (tokens, prompt_len) = qwen35_raw_tokens(&state, PROMPT, 3, 0.0, &CancelToken::never())
        .ok()
        .flatten()
        .expect("a Qwen3.5 state generates");
    assert_eq!(prompt_len, want.len());
    assert_eq!(&tokens[..prompt_len], &want[..], "RAW prompt, no chat template");
    assert!(tokens.len() <= prompt_len + 3, "max_tokens bounds the output");
}

#[test]
fn greedy_decoding_is_deterministic_and_never_ends_on_a_stop_token() {
    let Some(state) = state_or_skip() else { return };
    let run = || {
        qwen35_raw_tokens(&state, PROMPT, 6, 0.0, &CancelToken::never())
            .ok()
            .flatten()
            .expect("tokens")
    };
    let (a, prompt_len) = run();
    assert_eq!(a, run().0);
    let stops = crate::api::realize_handlers::qwen35_completion_stop_tokens(
        state.tokenizer.as_deref(),
        state.model_eos_token_id(),
    );
    if a.len() > prompt_len {
        assert!(!stops.contains(a.last().expect("non-empty")), "stop token kept");
    }
}

#[test]
fn an_empty_prompt_is_400() {
    let Some(state) = state_or_skip() else { return };
    let (status, msg) = refusal(&state, "");
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(msg.contains("Prompt cannot be empty"), "{msg}");
}

#[test]
fn a_prompt_that_fills_the_context_is_refused_whole_with_both_numbers() {
    let Some(mut state) = state_or_skip() else { return };
    let n = state
        .mapped_gguf_model()
        .expect("mapped")
        .model
        .encode(PROMPT)
        .expect("encode")
        .len();
    let served = Arc::try_unwrap(state.qwen35_session.take().expect("session"))
        .unwrap_or_else(|_| panic!("the test holds the only handle"));
    state.qwen35_session = Some(Arc::new(Qwen35Served {
        context_length: n, // prompt_len >= context_length
        ..served
    }));
    let (status, msg) = refusal(&state, PROMPT);
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        msg.contains(&format!("{n} tokens")) && msg.contains("refused whole"),
        "{msg}"
    );
}
