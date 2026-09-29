//! D5 (ruling-3715-2010): a serve turn past the device KV cache is a 400 that
//! names the limit ("context exceeds 4096"), never a 500.
//!
//! The observed failure (h4-depcheck-0d): a 27155-token turn on a dense CUDA
//! model reached `reserve`, failed with "the device KV cache holds 4096" and
//! came back as HTTP 500. The handlers now refuse it with
//! [`serve_context_refusal`] before any path takes the model.

use crate::api::openai_handlers::fit_serving_context;
use crate::api::{generation_error_status, serve_context_refusal, AppState};
use crate::error::RealizarError;
use axum::http::StatusCode;

#[test]
fn d5_turn_past_the_device_kv_is_refused_and_names_the_limit() {
    let msg = serve_context_refusal(27155, 4096).expect("27155 tokens cannot fit 4096");
    assert!(msg.starts_with("context exceeds 4096"), "{msg}");
    assert!(msg.contains("27155"), "{msg}");
    assert!(msg.contains("refused whole, not truncated"), "{msg}");
}

#[test]
fn d5_boundary_a_prompt_that_fills_the_window_leaves_no_room_to_answer() {
    assert!(serve_context_refusal(4097, 4096).is_some());
    assert!(serve_context_refusal(4096, 4096).is_some());
    assert_eq!(serve_context_refusal(4095, 4096), None);
    assert_eq!(serve_context_refusal(1, 4096), None);
}

#[test]
fn d5_no_cuda_model_means_no_serving_cap() {
    // The pre-flight reads `serving_context()`; only a GGUF CUDA state sets it,
    // so the CPU and Qwen3.5 paths keep their own context rules.
    let state = AppState::demo().expect("demo state");
    assert_eq!(state.serving_context(), None);
}

#[test]
fn d5_the_session_error_behind_it_is_a_client_error() {
    let err = RealizarError::ContextLimitExceeded {
        provided: 27155,
        maximum: 4096,
    };
    assert_eq!(generation_error_status(&err), StatusCode::BAD_REQUEST);
}

#[test]
fn d5_chat_pre_flight_passes_a_turn_through_when_there_is_no_cap() {
    let state = AppState::demo().expect("demo state");
    let ids = fit_serving_context(&state, Ok(vec![7, 8, 9])).expect("no cap, no refusal");
    assert_eq!(ids, vec![7, 8, 9]);
}

#[test]
fn d5_chat_pre_flight_refuses_a_turn_past_the_cap_with_a_400() {
    let mut state = AppState::demo().expect("demo state");
    state.cached_serving_context = Some(3);
    assert_eq!(
        fit_serving_context(&state, Ok(vec![1, 2])).expect("2 tokens fit 3"),
        vec![1, 2]
    );
    let refused = fit_serving_context(&state, Ok(vec![1, 2, 3])).expect_err("3 tokens fill 3");
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn d5_chat_pre_flight_keeps_a_tokenize_error() {
    let mut state = AppState::demo().expect("demo state");
    state.cached_serving_context = Some(3);
    let err = axum::response::IntoResponse::into_response(StatusCode::IM_A_TEAPOT);
    let kept = fit_serving_context(&state, Err(err)).expect_err("the error passes through");
    assert_eq!(kept.status(), StatusCode::IM_A_TEAPOT);
}
