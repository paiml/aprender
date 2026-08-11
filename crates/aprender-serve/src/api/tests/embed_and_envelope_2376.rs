//! Falsifiers for the aprender#2376 / aprender#2396 HTTP findings #2429 left open.
//!
//! Every test here fails against `main` at d16c608b1 (the commit after #2429).
//! They assert what a client observes — status, content-type, whether two routes
//! return the same numbers — not that a function was called.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

use super::native_routes_2376::{body_string, get, post};

#[cfg(feature = "gpu")]
use super::native_routes_2376::quantized_state;

// ---------------------------------------------------------------------------
// Finding 1, seventh route (P0): the embedding routes on a quantized server
//
// #2429 fixed six of the seven routes named in finding 1. `/realize/embed` was
// left answering 503 "needs a dense (.apr / .safetensors) model" on the standard
// `apr serve run model.gguf` path — and `/v1/embeddings` delegates to the same
// handler, so BOTH embedding routes were unusable on the path most users take.
// aprender#2396(2) is the third: `/api/embeddings` was not mounted at all.
// ---------------------------------------------------------------------------

/// A quantized-only server whose token embeddings actually differ per token.
///
/// [`quantized_state`] is built by `create_test_quantized_model`, whose
/// `token_embedding` is `vec![0.1; vocab * hidden]` — every token has the SAME
/// vector — and whose Q4_K weights are all zero. That model is fine for "does the
/// route answer", but it maps every input to one identical hidden state, so it
/// cannot tell "the embedding depends on the input" from "the handler returns a
/// constant". This fixture varies the embedding rows by token id so the
/// discrimination assertions below are real: with the attention and FFN weights
/// still zero, the residual stream at position `t` is that token's embedding, so
/// two different token sequences MUST pool to two different vectors.
#[cfg(feature = "gpu")]
fn varied_quantized_state() -> AppState {
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
        eos_token_id: None,
    };
    let mut model = create_test_quantized_model(&config);
    for token in 0..config.vocab_size {
        for i in 0..config.hidden_dim {
            // Deterministic, token-dependent, non-degenerate.
            model.token_embedding[token * config.hidden_dim + i] =
                0.1 + (token as f32) * 0.01 + (i as f32) * 0.001;
        }
    }
    AppState::with_quantized_model(model).expect("build varied quantized AppState")
}

/// Parse `{"data":[{"embedding":[...]}, ...]}` into the raw vectors.
fn embeddings_from(body: &str) -> Vec<Vec<f32>> {
    let parsed: serde_json::Value = serde_json::from_str(body).expect("json embedding body");
    parsed["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|d| {
            d["embedding"]
                .as_array()
                .expect("embedding array")
                .iter()
                .map(|v| v.as_f64().expect("float") as f32)
                .collect()
        })
        .collect()
}

/// `/realize/embed` must answer with a real vector on a quantized-only server.
///
/// Before: `503 {"error":"Model registry error: No model available: /realize/embed
/// needs a dense (.apr / .safetensors) model; this server has a quantized model
/// loaded"}` — on a server whose `/generate` works and whose `/health` reports
/// `model_loaded:true`.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_realize_embed_answers_on_quantized_server() {
    let (status, body) = post(
        quantized_state(),
        "/realize/embed",
        r#"{"input":["token5 token6"]}"#,
    )
    .await;

    assert_eq!(
        status,
        StatusCode::OK,
        "a quantized model can produce hidden states; body: {body}"
    );
    assert!(
        !body.contains("No model available"),
        "the quantized backend IS a model, body: {body}"
    );

    let vectors = embeddings_from(&body);
    assert_eq!(vectors.len(), 1, "one input, one embedding");
    // The fixture model's hidden_dim — never a hardcoded 384, never empty.
    assert_eq!(
        vectors[0].len(),
        64,
        "embedding width must be the model hidden_dim"
    );
    let norm: f32 = vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "embeddings are L2-normalized, got norm {norm}"
    );
}

/// The vectors must be model-derived, not a constant: different text must give a
/// different vector, and identical text must reproduce exactly. A handler that
/// returned zeros or a fixed vector would pass a "200 with the right length"
/// check and fail this one.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_embeddings_vary_with_input_and_repeat_exactly() {
    let (_, a1) = post(
        varied_quantized_state(),
        "/realize/embed",
        r#"{"input":["token5"]}"#,
    )
    .await;
    let (_, a2) = post(
        varied_quantized_state(),
        "/realize/embed",
        r#"{"input":["token5"]}"#,
    )
    .await;
    let (_, b) = post(
        varied_quantized_state(),
        "/realize/embed",
        r#"{"input":["token9 token9 token9"]}"#,
    )
    .await;

    let v_a1 = embeddings_from(&a1).remove(0);
    let v_a2 = embeddings_from(&a2).remove(0);
    let v_b = embeddings_from(&b).remove(0);

    assert_eq!(v_a1, v_a2, "the same text must embed identically");
    assert_ne!(
        v_a1, v_b,
        "different text must embed differently — a constant vector is not an embedding"
    );
    assert!(
        v_a1.iter().any(|x| x.abs() > 1e-6),
        "an all-zero vector is not a model-backed embedding"
    );
}

/// A batch returns one vector per input, in request order, and batching must not
/// change what an individual input embeds to.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_embed_batch_is_per_input_and_ordered() {
    let (status, body) = post(
        varied_quantized_state(),
        "/realize/embed",
        r#"{"input":["token5","token9 token9 token9"]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let vectors = embeddings_from(&body);
    assert_eq!(vectors.len(), 2, "two inputs, two embeddings");
    assert_ne!(vectors[0], vectors[1], "each input gets its own vector");

    let (_, single) = post(
        varied_quantized_state(),
        "/realize/embed",
        r#"{"input":["token5"]}"#,
    )
    .await;
    assert_eq!(
        vectors[0],
        embeddings_from(&single).remove(0),
        "batching must not change an input's embedding"
    );
}

/// `/v1/embeddings` delegates to the same handler, so it was dead for the same
/// reason. Same server, same text, same numbers.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_v1_embeddings_matches_realize_embed_on_quantized_server() {
    let (native_status, native) = post(
        quantized_state(),
        "/realize/embed",
        r#"{"input":["token5"]}"#,
    )
    .await;
    let (openai_status, openai) =
        post(quantized_state(), "/v1/embeddings", r#"{"input":["token5"]}"#).await;

    assert_eq!(native_status, StatusCode::OK, "body: {native}");
    assert_eq!(openai_status, StatusCode::OK, "body: {openai}");
    assert_eq!(
        embeddings_from(&native),
        embeddings_from(&openai),
        "two routes over one model must not disagree about the same text"
    );
}

/// aprender#2396(2): `/api/embeddings` is what every Ollama embedding client
/// calls. Before: the router's 404 fallback, because the route was not mounted.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_api_embeddings_is_mounted_and_ollama_shaped() {
    let (status, body) = post(
        quantized_state(),
        "/api/embeddings",
        r#"{"model":"default","prompt":"token5"}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        !body.contains("not_found"),
        "the route must exist; got the 404 fallback: {body}"
    );

    // Ollama's wire shape: one flat `embedding`, no OpenAI envelope.
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json body");
    let vector: Vec<f32> = parsed["embedding"]
        .as_array()
        .expect("flat `embedding` array — Ollama's shape")
        .iter()
        .map(|v| v.as_f64().expect("float") as f32)
        .collect();
    assert_eq!(vector.len(), 64, "width must be the model hidden_dim");

    // And it must be the SAME vector the other two routes give for that text.
    let (_, native) = post(quantized_state(), "/realize/embed", r#"{"input":["token5"]}"#).await;
    assert_eq!(
        vector,
        embeddings_from(&native).remove(0),
        "all three embedding routes must agree"
    );
}

/// An empty embedding input is still a client error.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_embed_rejects_empty_input_with_400() {
    let (status, body) = post(quantized_state(), "/realize/embed", r#"{"input":[""]}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
}

/// With no model at all, embedding is a server-side condition: 503, never 404
/// (the route exists and the request was valid), and never 200 with a fake vector.
#[tokio::test]
async fn test_embed_without_any_model_is_503() {
    let (status, body) = post(
        AppState::demo_mock().expect("mock state"),
        "/realize/embed",
        r#"{"input":["hello"]}"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "no model at all is a server condition, body: {body}"
    );
}

// ---------------------------------------------------------------------------
// Finding 7 (P2): one error envelope for every failure
// ---------------------------------------------------------------------------

/// Send a request and return `(status, content-type, body)`.
async fn probe(
    state: AppState,
    method: &str,
    uri: &str,
    content_type: Option<&str>,
    body: &str,
) -> (StatusCode, String, String) {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(ct) = content_type {
        builder = builder.header("content-type", ct);
    }
    let response = create_router(state)
        .oneshot(builder.body(Body::from(body.to_string())).expect("request"))
        .await
        .expect("dispatch");
    let status = response.status();
    let ct = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap_or_default().to_string())
        .unwrap_or_default();
    (status, ct, body_string(response).await)
}

/// A malformed JSON body must produce a JSON envelope, not `text/plain` quoting
/// the serde parser position.
///
/// Before: `content-type: text/plain; charset=utf-8` with the body
/// `Failed to parse the request body as JSON: key must be a string at line 1 column 2`
/// — the exact leak the 422 sanitizer was added to prevent, one status code over.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_malformed_json_is_a_sanitized_json_envelope() {
    let (status, content_type, body) = probe(
        quantized_state(),
        "POST",
        "/generate",
        Some("application/json"),
        "{not json",
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert!(
        content_type.starts_with("application/json"),
        "every error must be machine-parseable, got content-type {content_type:?}"
    );
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json error body");
    assert!(
        parsed["error"].is_string(),
        "the envelope is {{\"error\": \"...\"}}, got: {body}"
    );
    assert!(
        !body.contains("line 1 column") && !body.contains("Failed to parse the request body"),
        "the parser's internals must not reach a client: {body}"
    );
}

/// A missing `Content-Type` must produce the same envelope (was bare text/plain).
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_missing_content_type_is_a_json_envelope() {
    let (status, content_type, body) = probe(
        quantized_state(),
        "POST",
        "/generate",
        None,
        r#"{"prompt":"token5","max_tokens":2}"#,
    )
    .await;

    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE, "body: {body}");
    assert!(
        content_type.starts_with("application/json"),
        "got content-type {content_type:?}, body: {body}"
    );
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json error body");
    assert!(parsed["error"].is_string(), "body: {body}");
}

/// The 422 keeps its existing sanitized wording (GH-649 must not regress).
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_schema_mismatch_stays_sanitized_422() {
    let (status, content_type, body) = probe(
        quantized_state(),
        "POST",
        "/generate",
        Some("application/json"),
        r#"{"prompt":123}"#,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "body: {body}");
    assert!(
        content_type.starts_with("application/json"),
        "content-type {content_type:?}"
    );
    assert!(body.contains("Invalid request body"), "body: {body}");
}

/// Enveloping errors must not cost the headers a client depends on: a 405 still
/// carries `allow`, which a response rebuilt from scratch would have dropped.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_405_keeps_its_allow_header_and_gains_a_json_body() {
    let response = create_router(quantized_state())
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/generate")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("dispatch");

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    let allow = response
        .headers()
        .get("allow")
        .map(|v| v.to_str().unwrap_or_default().to_string())
        .unwrap_or_default();
    assert!(
        allow.contains("POST"),
        "the allow header must survive the envelope, got {allow:?}"
    );
    let body = body_string(response).await;
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json error body");
    assert!(parsed["error"].is_string(), "body: {body}");
}

/// A handler's own JSON error must pass through verbatim — the middleware only
/// fills in for responses that are not already JSON.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_handler_json_errors_pass_through_unchanged() {
    let (status, content_type, body) = probe(
        quantized_state(),
        "POST",
        "/generate",
        Some("application/json"),
        r#"{"prompt":""}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    assert!(
        content_type.starts_with("application/json"),
        "content-type {content_type:?}"
    );
    assert!(
        body.contains("Prompt cannot be empty"),
        "the handler's own message must survive: {body}"
    );
}

/// A successful response must pass through the middleware untouched.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_success_bodies_are_not_rewritten() {
    let (status, body) = post(quantized_state(), "/generate", r#"{"prompt":"token5"}"#).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(body.contains("token_ids"), "body: {body}");
}

/// An HTTP error body must never instruct a client to call an internal Rust
/// constructor. Before: `{"error":"No APR model loaded. Use AppState::demo() or
/// load a .apr model."}`.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_v1_predict_error_names_no_internal_rust_api() {
    let (status, body) = post(quantized_state(), "/v1/predict", r#"{"features":[1.0,2.0]}"#).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body: {body}");
    assert!(
        !body.contains("AppState::demo"),
        "an HTTP client cannot call a Rust constructor: {body}"
    );
    assert!(
        body.contains("/v1/predict") && body.contains(".apr"),
        "the message must say what the OPERATOR needs to do instead: {body}"
    );
}

// ---------------------------------------------------------------------------
// Finding 8 (P2): route-surface drift between the three routers
// ---------------------------------------------------------------------------

/// `/` and `/ready` are registered by the other two routers in this repo and
/// 404'd on the GGUF serve path, so which surface you got depended on the format
/// of the file you passed. `/` now answers with the route table this router
/// actually mounted, which is what makes the surface discoverable at all.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_root_and_ready_are_mounted() {
    let (status, body) = get(quantized_state(), "/").await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json index");
    let routes = parsed["routes"].as_array().expect("routes array");
    assert!(
        routes.iter().any(|r| r == "POST /generate"),
        "the index must list what is mounted: {body}"
    );
    assert!(
        routes.iter().any(|r| r == "POST /api/embeddings"),
        "the index must list the newly mounted Ollama embedding route: {body}"
    );

    let (ready_status, ready_body) = get(quantized_state(), "/ready").await;
    assert_ne!(
        ready_status,
        StatusCode::NOT_FOUND,
        "/ready is the conventional readiness path, body: {ready_body}"
    );
}

/// Registry mode: an unknown `model_id` is a CLIENT error and must stay a 404.
///
/// The new quantized fallback must not swallow it — silently embedding the
/// caller's text with a model they did not ask for is worse than the original
/// failure, because the response looks successful.
#[tokio::test]
async fn test_registry_unknown_model_id_is_404_not_a_silent_substitution() {
    use crate::layers::{Model, ModelConfig};
    use crate::registry::ModelRegistry;
    use crate::tokenizer::BPETokenizer;

    let config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
    };
    let vocab: Vec<String> = (0..100)
        .map(|i| {
            if i == 0 {
                "<unk>".to_string()
            } else {
                format!("t{i}")
            }
        })
        .collect();
    let registry = ModelRegistry::new(10);
    registry
        .register(
            "known",
            Model::new(config).expect("model"),
            BPETokenizer::new(vocab, vec![], "<unk>").expect("tokenizer"),
        )
        .expect("register");
    let state = AppState::with_registry(registry, "known").expect("registry state");

    let (known_status, known_body) = post(
        state.clone(),
        "/realize/embed",
        r#"{"input":["t1 t2"],"model":"known"}"#,
    )
    .await;
    assert_eq!(known_status, StatusCode::OK, "body: {known_body}");

    let (unknown_status, unknown_body) = post(
        state,
        "/realize/embed",
        r#"{"input":["t1 t2"],"model":"nope"}"#,
    )
    .await;
    assert_eq!(
        unknown_status,
        StatusCode::NOT_FOUND,
        "an unknown model id must not be answered by another model, body: {unknown_body}"
    );
}
