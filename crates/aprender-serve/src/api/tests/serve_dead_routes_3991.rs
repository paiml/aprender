//! Falsifiers for aprender#3991 — `apr serve` advertised generation routes that did
//! not generate.
//!
//! Measured by the #3962 CRUX serve cells on a GGUF: `/v1/batch/completions`
//! answered 503 "No GPU-capable model loaded" behind an advertised route, and
//! `/generate`, `/batch/generate` and `/realize/batch` put the whole PROMPT in
//! `text`, so a grader reading `text` found the prompt's own `<answer></answer>`
//! template before the model's answer.
//!
//! A route in `GET /` is a capability claim. Every generation route it lists must
//! answer a generation probe; a route the loaded backend cannot serve is not listed.

use axum::http::StatusCode;

use super::native_routes_2376::{get, post, quantized_state};

/// `quantized_state` with an EOS the zero-weight model never samples. Its greedy
/// pick is id 0, which is also the EOS fallback, so the shared fixture stops
/// before generating anything and cannot tell prompt text from completion text.
#[cfg(feature = "gpu")]
fn generating_quantized_state() -> crate::api::AppState {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: Some(255),
    };
    let mut model = create_test_quantized_model(&config);
    // Zero weights give uniform logits; a bias on id 42 makes the greedy
    // completion "token42" repeated, a string the prompts never contain.
    let mut bias = vec![0.0f32; 256];
    bias[42] = 1.0;
    model.lm_head_bias = Some(bias);
    crate::api::AppState::with_quantized_model(model).expect("build quantized AppState")
}

/// Every generation route the router can mount, with a minimal valid body.
const GENERATION_PROBES: &[(&str, &str)] = &[
    ("POST /generate", r#"{"prompt":"token5","max_tokens":2}"#),
    ("POST /batch/generate", r#"{"prompts":["token5"],"max_tokens":2}"#),
    ("POST /stream/generate", r#"{"prompt":"token5","max_tokens":2}"#),
    ("POST /realize/generate", r#"{"prompt":"token5","max_tokens":2}"#),
    ("POST /realize/batch", r#"{"prompts":["token5"],"max_tokens":2}"#),
    (
        "POST /v1/completions",
        r#"{"model":"m","prompt":"token5","max_tokens":2}"#,
    ),
    (
        "POST /v1/chat/completions",
        r#"{"model":"m","messages":[{"role":"user","content":"token5"}],"max_tokens":2}"#,
    ),
    (
        "POST /v1/chat/completions/stream",
        r#"{"model":"m","messages":[{"role":"user","content":"token5"}],"max_tokens":2}"#,
    ),
    (
        "POST /v1/batch/completions",
        r#"{"prompts":["token5"],"max_tokens":2}"#,
    ),
    (
        "POST /api/generate",
        r#"{"model":"m","prompt":"token5","stream":false,"options":{"num_predict":2}}"#,
    ),
    (
        "POST /api/chat",
        r#"{"model":"m","messages":[{"role":"user","content":"token5"}],"stream":false,"options":{"num_predict":2}}"#,
    ),
];

async fn listed_routes(state: crate::api::AppState) -> Vec<String> {
    let (status, body) = get(state, "/").await;
    assert_eq!(status, StatusCode::OK, "GET / body: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("GET / is JSON");
    parsed["routes"]
        .as_array()
        .expect("GET / has a routes array")
        .iter()
        .map(|r| r.as_str().unwrap_or_default().to_string())
        .collect()
}

/// The #3991 must-RED: every generation route `GET /` lists on a quantized
/// (`apr serve model.gguf`) server answers a generation probe with 2xx.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn every_listed_generation_route_generates_on_a_quantized_server() {
    let listed = listed_routes(quantized_state()).await;
    let mut probed = 0;
    let mut dead = Vec::new();
    for (route, body) in GENERATION_PROBES {
        if !listed.iter().any(|r| r == route) {
            continue;
        }
        probed += 1;
        let path = route.split_once(' ').expect("METHOD /path").1;
        let (status, resp) = post(quantized_state(), path, body).await;
        if !status.is_success() {
            dead.push(format!("\n  - {route} -> {status}: {resp}"));
        }
    }
    // Not vacuous: the core generate routes must be listed and probed.
    assert!(probed >= 5, "only {probed} generation routes listed: {listed:?}");
    assert!(
        dead.is_empty(),
        "GET / advertises generation routes this server cannot answer:{}",
        dead.concat()
    );
}

/// Positive control for the unlisting: `/v1/batch/completions` needs a
/// `cached_model`, which a quantized-only server does not have — so it must not be
/// listed there. (The test above would pass vacuously if NOTHING were listed; this
/// pins the one route #3991 measured 503.)
#[tokio::test]
#[cfg(feature = "gpu")]
async fn gpu_batch_completions_is_not_advertised_without_a_cached_model() {
    let listed = listed_routes(quantized_state()).await;
    assert!(
        !listed.iter().any(|r| r == "POST /v1/batch/completions"),
        "a server with no cached_model cannot answer /v1/batch/completions: {listed:?}"
    );
    // And it is not merely hidden: unlisted means unmounted (404, not 503).
    let (status, _) = post(
        quantized_state(),
        "/v1/batch/completions",
        r#"{"prompts":["token5"],"max_tokens":2}"#,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

fn completion_of(result: &serde_json::Value) -> (Vec<u64>, String) {
    let ids: Vec<u64> = result["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("u64 id"))
        .collect();
    let n = result["num_generated"].as_u64().expect("num_generated") as usize;
    let suffix = ids[ids.len() - n..].to_vec();
    (suffix, result["text"].as_str().expect("text").to_string())
}

/// `text` is the COMPLETION, not prompt + completion.
///
/// The test model has all-zero weights plus an LM-head bias on id 42, so its greedy
/// completion does not depend on the prompt. Two different prompts therefore produce the same generated ids;
/// completion-only `text` must then be identical. Prompt-echoing `text` differs,
/// because it starts with the (different) prompt. The ids are checked first so a
/// model change cannot make this pass or fail for the wrong reason.
async fn assert_text_is_completion_only(route: &str, batch: bool) {
    let mut outs = Vec::new();
    for prompt in ["token7 token9", "token3"] {
        let body = if batch {
            serde_json::json!({"prompts":[prompt],"max_tokens":3,"temperature":0.0})
        } else {
            serde_json::json!({"prompt":prompt,"max_tokens":3,"temperature":0.0})
        };
        let (status, resp) = post(generating_quantized_state(), route, &body.to_string()).await;
        assert_eq!(status, StatusCode::OK, "{route}: {resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).expect("json");
        let result = if batch { v["results"][0].clone() } else { v };
        outs.push(completion_of(&result));
    }
    assert!(!outs[0].0.is_empty(), "{route}: nothing generated, test is inconclusive");
    assert_eq!(
        outs[0].0, outs[1].0,
        "{route}: precondition — the zero-weight model's completion should not depend on the prompt"
    );
    assert!(
        outs[0].0.iter().all(|&id| id == 42),
        "{route}: precondition — the biased model should pick id 42, got {:?}",
        outs[0].0
    );
    // The property itself, not a proxy: text IS the decoded completion.
    assert_eq!(outs[0].1, "token42".repeat(outs[0].0.len()), "{route}");
    assert_eq!(
        outs[0].1, outs[1].1,
        "{route}: same completion ids but different text — text carries the prompt"
    );
}

#[tokio::test]
#[cfg(feature = "gpu")]
async fn generate_text_is_the_completion_not_the_prompt() {
    assert_text_is_completion_only("/generate", false).await;
}

#[tokio::test]
#[cfg(feature = "gpu")]
async fn batch_generate_text_is_the_completion_not_the_prompt() {
    assert_text_is_completion_only("/batch/generate", true).await;
    assert_text_is_completion_only("/realize/batch", true).await;
}
