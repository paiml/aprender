//! Falsifiers for aprender#2609 — routed endpoints on an `AprTransformer` server,
//! and one status code for one server-side condition.
//!
//! #2609 measured four routed endpoints answering `"Model registry error: No model
//! available"` on a server whose `/generate` returned 200 and whose `/health`
//! reported `model_loaded: true` — three of them as **404**, one as **500**. The
//! functional half was closed for the *quantized* backend by #2375/#2376 (both
//! landed after the 0.63.0 the sweep measured). What survived on `main` was the
//! same class on the THIRD resident backend, `AppState::apr_transformer` — the f32
//! APR / SafeTensors CPU serve path — plus the status-code split, which no backend
//! fixed because it lives in the dense fallback.
//!
//! Measured on `main` @ bb2bd5e73, before this module's fixes:
//!
//! | route | `apr_transformer` server | no model at all |
//! |---|---|---|
//! | `POST /generate` | **200** `{"text":"t1t230",…}` | 503 |
//! | `POST /batch/generate` | **200** | 503 |
//! | `POST /stream/generate` | 503 "No model available" | 503 |
//! | `POST /v1/chat/completions` | **404** "No model available" | **404** |
//! | `POST /v1/chat/completions/stream` | **404** "No model available" | **404** |
//! | `POST /v1/embeddings` | 503 "…: /realize/embed needs a loaded model" | 503, wrong route named |
//!
//! Every test here asserts what a client observes — a status code, a body — and
//! every assertion excludes an outcome: `assert!(resp.is_ok())` would have passed
//! against all six of those rows.

use axum::http::StatusCode;

use super::native_routes_2376::{get, post};
use crate::api::AppState;

// ---------------------------------------------------------------------------
// Fixture: the server `apr serve` builds for an f32 APR / SafeTensors model
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 64;
const VOCAB_SIZE: usize = 256;

/// An `AprTransformer`-only server: a tokenizer plus `apr_transformer`, and NO
/// dense f32 `Model` and NO quantized model — exactly what
/// `AppState::with_apr_transformer_and_vocab` builds on the CPU serve path.
fn apr_transformer_state() -> AppState {
    use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};

    let num_layers = 2usize;
    let (num_heads, num_kv_heads) = (4usize, 4usize);
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: HIDDEN_DIM,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size: VOCAB_SIZE,
        intermediate_dim: HIDDEN_DIM * 4,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: None,
        ..Default::default()
    };
    let head_dim = HIDDEN_DIM / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = HIDDEN_DIM + kv_dim + kv_dim;
    let intermediate = HIDDEN_DIM * 4;
    let transformer = AprTransformer {
        config,
        token_embedding: vec![0.1; VOCAB_SIZE * HIDDEN_DIM],
        layers: (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; HIDDEN_DIM],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; qkv_out_dim * HIDDEN_DIM],
                qkv_bias: None,
                attn_output_weight: vec![0.01; HIDDEN_DIM * HIDDEN_DIM],
                attn_output_bias: None,
                ffn_gate_weight: Some(vec![0.01; intermediate * HIDDEN_DIM]),
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; intermediate * HIDDEN_DIM],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; HIDDEN_DIM * intermediate],
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; HIDDEN_DIM]),
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight: None,
                moe_expert_gate_up: None,
                moe_expert_down: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_expert_gate_weight: None,
            })
            .collect(),
        output_norm_weight: vec![1.0; HIDDEN_DIM],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; VOCAB_SIZE * HIDDEN_DIM],
        lm_head_bias: None,
        lm_head_tied: false,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };
    let vocab: Vec<String> = (0..VOCAB_SIZE)
        .map(|i| {
            if i == 0 {
                "<unk>".to_string()
            } else {
                format!("t{i}")
            }
        })
        .collect();
    AppState::with_apr_transformer_and_vocab(transformer, vocab)
        .expect("build AprTransformer AppState")
}

/// The four routes #2609 measured, plus the two that already worked. The premise
/// of the bug report is the CONTRAST, so the working two are probed too: a fix
/// that broke `/generate` would satisfy "all the same status" trivially.
const GENERATION_ROUTES: &[(&str, &str)] = &[
    ("/generate", r#"{"prompt":"t1","max_tokens":1}"#),
    ("/batch/generate", r#"{"prompts":["t1"],"max_tokens":1}"#),
    ("/stream/generate", r#"{"prompt":"t1","max_tokens":1}"#),
    (
        "/v1/chat/completions",
        r#"{"model":"m","messages":[{"role":"user","content":"t1"}],"max_tokens":1}"#,
    ),
    (
        "/v1/chat/completions/stream",
        r#"{"model":"m","messages":[{"role":"user","content":"t1"}],"max_tokens":1}"#,
    ),
    ("/v1/embeddings", r#"{"model":"m","input":"t1"}"#),
];

// ---------------------------------------------------------------------------
// The premise: this server IS loaded and DOES generate
// ---------------------------------------------------------------------------

/// `/health` must claim a model, and `/generate` must actually produce one — the
/// two facts that make every dead route below a contradiction rather than a
/// correctly-reported empty server.
#[tokio::test]
async fn apr_transformer_server_reports_loaded_and_generates() {
    let (status, body) = get(apr_transformer_state(), "/health").await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        body.contains("\"model_loaded\":true"),
        "/health must report the resident AprTransformer; body: {body}"
    );

    let (status, body) = post(
        apr_transformer_state(),
        "/generate",
        r#"{"prompt":"t1","max_tokens":1}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        body.contains("\"num_generated\":1"),
        "/generate must decode a real token, not an empty envelope; body: {body}"
    );
}

// ---------------------------------------------------------------------------
// Half 1: every routed generation endpoint answers on this server
// ---------------------------------------------------------------------------

/// No routed generation endpoint may answer "No model available" on a server that
/// holds a model.
///
/// Before: `/stream/generate` 503, `/v1/chat/completions{,/stream}` 404,
/// `/v1/embeddings` 503 — while `/generate` and `/batch/generate` returned 200.
#[tokio::test]
async fn no_routed_endpoint_is_dead_on_an_apr_transformer_server() {
    for (uri, body) in GENERATION_ROUTES {
        let (status, resp) = post(apr_transformer_state(), uri, body).await;
        assert_eq!(status, StatusCode::OK, "POST {uri} -> {status}: {resp}");
        assert!(
            !resp.contains("No model available"),
            "POST {uri} claimed no model on a loaded server: {resp}"
        );
    }
}

/// `/stream/generate` must emit at least one `token` SSE event, not just a `done`.
///
/// A backend that resolved but generated nothing would still be 200 with a bare
/// `done` frame, so status alone does not exclude the defect.
#[tokio::test]
async fn apr_transformer_stream_generate_emits_token_events() {
    let (status, body) = post(
        apr_transformer_state(),
        "/stream/generate",
        r#"{"prompt":"t1","max_tokens":2}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        body.contains("event: token"),
        "SSE stream carried no token event: {body}"
    );
    assert!(
        !body.contains("\"num_generated\":0"),
        "SSE stream terminated with zero generated tokens: {body}"
    );
}

/// `/v1/chat/completions/stream` must produce OpenAI chat chunks.
#[tokio::test]
async fn apr_transformer_chat_stream_emits_completion_chunks() {
    let (status, body) = post(
        apr_transformer_state(),
        "/v1/chat/completions/stream",
        r#"{"model":"m","messages":[{"role":"user","content":"t1"}],"max_tokens":2}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        body.contains("chat.completion.chunk"),
        "no OpenAI stream chunk in body: {body}"
    );
    assert!(body.contains("[DONE]"), "stream never terminated: {body}");
}

/// `/v1/embeddings` must return a vector of the MODEL's hidden width, not a
/// constant-dimension placeholder and not an error.
#[tokio::test]
async fn apr_transformer_embeddings_returns_model_width_vector() {
    let (status, body) = post(
        apr_transformer_state(),
        "/v1/embeddings",
        r#"{"model":"m","input":"t1"}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let parsed: serde_json::Value = serde_json::from_str(&body).expect("embedding JSON");
    let embedding = parsed["data"][0]["embedding"]
        .as_array()
        .expect("embedding array");
    assert_eq!(
        embedding.len(),
        HIDDEN_DIM,
        "embedding width must be the model's hidden_dim; body: {body}"
    );
    let norm: f64 = embedding
        .iter()
        .map(|v| {
            let x = v.as_f64().expect("f64 component");
            x * x
        })
        .sum();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "embedding must be L2-normalized, got norm^2 = {norm}"
    );
}

// ---------------------------------------------------------------------------
// Half 2: ONE condition, ONE status code
// ---------------------------------------------------------------------------

/// A server with no usable model at all must answer every routed generation
/// endpoint with the SAME status, and that status must be 503.
///
/// #2609's second defect: three of the four routes answered 404 — "route not
/// found" — for a condition that has nothing to do with routing (a genuinely
/// unrouted path returns a different body entirely), and one answered 500. A
/// client could not distinguish "endpoint does not exist" from "endpoint exists
/// but has no model", and a retry policy keyed on status treated one outage as
/// permanent on some routes and as a server bug on others.
#[tokio::test]
async fn no_model_available_is_503_on_every_routed_endpoint() {
    let mut observed: Vec<(&str, StatusCode)> = Vec::new();
    for (uri, body) in GENERATION_ROUTES {
        let state = AppState::demo_mock().expect("model-less AppState");
        let (status, resp) = post(state, uri, body).await;
        assert!(
            resp.contains("No model available"),
            "POST {uri} must report the missing model, got: {resp}"
        );
        assert_ne!(
            status,
            StatusCode::NOT_FOUND,
            "POST {uri} answered 404 for a mounted route with no model: {resp}"
        );
        assert_ne!(
            status,
            StatusCode::INTERNAL_SERVER_ERROR,
            "POST {uri} answered 500 for a server-side availability condition: {resp}"
        );
        observed.push((uri, status));
    }
    for (uri, status) in &observed {
        assert_eq!(
            *status,
            StatusCode::SERVICE_UNAVAILABLE,
            "POST {uri} disagreed with the rest of the surface on one condition; \
             full observation: {observed:?}"
        );
    }
}

/// An unrouted path must stay 404 — the fix must not blur "no such route" into
/// "no model", which is the very distinction #2609 says a client cannot make.
#[tokio::test]
async fn an_unrouted_path_is_still_404() {
    let (status, body) = get(apr_transformer_state(), "/healthz").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body: {body}");
    assert!(
        !body.contains("No model available"),
        "the 404 fallback must not claim a model problem: {body}"
    );
}

/// `AprTransformer::forward_hidden_states` must refuse a sequence longer than the
/// model's context window BEFORE allocating a KV cache for it.
///
/// An embedding route takes caller-supplied text, and sizing a cache from a
/// caller-supplied length with no ceiling is exactly what let one HTTP request
/// abort the server process (aprender#2376 finding 9). The ceiling lives in the
/// accessor, not in the handler, so every future caller inherits it.
#[test]
fn apr_forward_hidden_states_refuses_an_over_context_sequence() {
    use crate::error::RealizarError;

    let state = apr_transformer_state();
    let transformer = state
        .apr_transformer()
        .expect("resident transformer")
        .clone();
    let context = transformer.config.context_length;

    let too_long: Vec<u32> = vec![1; context + 1];
    match transformer.forward_hidden_states(&too_long) {
        Err(RealizarError::ContextLimitExceeded { provided, maximum }) => {
            assert_eq!(provided, context + 1);
            assert_eq!(maximum, context);
        },
        other => panic!("over-context sequence must be refused, got: {other:?}"),
    }

    // And an empty one is a client error, not a zero-length success.
    match transformer.forward_hidden_states(&[]) {
        Err(RealizarError::InvalidShape { .. }) => {},
        other => panic!("empty sequence must be refused, got: {other:?}"),
    }
}

/// An unavailable-model body must name the route the client actually called.
///
/// `/v1/embeddings` delegated to `realize_embed_handler`, so it answered
/// `"No model available: /realize/embed needs a loaded model"` — pointing the
/// caller at a route they never touched.
#[tokio::test]
async fn embed_unavailable_body_names_the_route_the_client_called() {
    for route in ["/v1/embeddings", "/realize/embed"] {
        let state = AppState::demo_mock().expect("model-less AppState");
        let (_, body) = post(state, route, r#"{"model":"m","input":"t1"}"#).await;
        assert!(
            body.contains(route),
            "POST {route} error body names another route: {body}"
        );
    }
}
