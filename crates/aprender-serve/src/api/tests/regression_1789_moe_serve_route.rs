//! Regression falsifier for aprender#1789 ("Option B") — `/v1/chat/completions`
//! must route a resident qwen3_moe GGUF through `try_qwen3_moe_backend`
//! (`crates/aprender-serve/src/api/cuda_chat_backend.rs::openai_chat_completions_handler`,
//! ~L823, `try_qwen3_moe_backend` defined ~L944), which calls
//! `run_qwen3_moe_generate` — the MoE-aware CPU path that correctly indexes
//! per-expert FFN tensors from the mmap, instead of falling through to the
//! dense CPU chain (`cpu_chat_backends`) which panicked on the MoE dense-FFN
//! placeholder shape.
//!
//! Builds `AppState` through the exact same chain the CLI's GGUF server-command
//! load path uses (`crates/aprender-serve/src/cli/mod_server_commands.rs`
//! ~L130-138): `AppState::with_quantized_model_and_vocab` followed by
//! `.with_mapped_gguf_model(Arc::new(mapped))` — the second call is the one
//! `try_qwen3_moe_backend` depends on to find a retained `MappedGGUFModel`.
//!
//! MUTATION (PMAT-3429 acceptance step 2): deleting the
//! `if let Some(r) = try_qwen3_moe_backend(...) { return r; }` dispatch at
//! `cuda_chat_backend.rs`~L823 makes this test fail, because the request then
//! falls through to `cpu_chat_backends` instead of the MoE-aware path —
//! measured: HTTP 500 carrying the #1790 "matmul weight has EMPTY data buffer"
//! refusal (before #1790 that path panicked).

use std::sync::Arc;

use crate::api::AppState;
use crate::gguf::regression_fixtures::build_qwen3_moe_gguf;
use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, GGUF_TYPE_Q4_K};

use super::native_routes_2376::post;

/// Builds the `AppState` exactly as the CLI's GGUF CPU serve chain does for a
/// qwen3_moe model with a retained mapped GGUF (`with_mapped_gguf_model`).
fn qwen3_moe_state() -> (tempfile::NamedTempFile, AppState) {
    let bytes = build_qwen3_moe_gguf(GGUF_TYPE_Q4_K);
    let mut temp_file = tempfile::NamedTempFile::new().expect("create temp GGUF");
    std::io::Write::write_all(&mut temp_file, &bytes).expect("write temp GGUF");

    let mapped = MappedGGUFModel::from_path(temp_file.path()).expect("map fixture GGUF");
    let quantized_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("build quantized model from fixture");

    // BPETokenizer::new requires its unk token ("<unk>") to be present in the
    // vocabulary; slot 0 carries it instead of "<tok0>" so the 32-entry
    // vocab still matches the fixture's VOCAB=32.
    let vocab: Vec<String> = std::iter::once("<unk>".to_string())
        .chain((1..32).map(|i| format!("<tok{i}>")))
        .collect();

    let state = AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        .expect("with_quantized_model_and_vocab")
        .with_mapped_gguf_model(Arc::new(mapped));

    (temp_file, state)
}

/// #1789 Option B: a resident qwen3_moe GGUF must take the MoE-aware
/// dispatch at `POST /v1/chat/completions`, not the dense CPU fallback.
#[tokio::test]
async fn regression_1789_moe_gguf_chat_completions_takes_the_moe_route() {
    let (_temp_file, state) = qwen3_moe_state();

    let (status, body) = post(
        state,
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#,
    )
    .await;

    assert!(
        status.is_success(),
        "#1789: a resident qwen3_moe GGUF must take the MoE-aware dispatch at \
         /v1/chat/completions (try_qwen3_moe_backend), not the dense CPU fallback \
         (a 500 from the #1790 empty-buffer guard; a panic before it). status={status}, body={body}"
    );

    let parsed: serde_json::Value = serde_json::from_str(&body).expect("chat completion JSON");
    let choices = parsed
        .get("choices")
        .and_then(|c| c.as_array())
        .unwrap_or_else(|| panic!("#1789: expected a `choices` array in body: {body}"));
    assert!(
        !choices.is_empty(),
        "#1789: expected a non-empty `choices` array in body: {body}"
    );
}
