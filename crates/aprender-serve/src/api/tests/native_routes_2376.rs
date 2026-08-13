//! Falsifiers for aprender#2376 — native serve routes on a quantized server.
//!
//! Every test here fails against the shipped 0.63.0 behaviour. They assert what a
//! client observes (status code, body content, whether two responses differ), not
//! that a function was called.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A quantized-only server: exactly what `apr serve run model.gguf` builds — a
/// tokenizer plus `quantized_model`, and NO dense f32 `Model`.
#[cfg(feature = "gpu")]
pub(super) fn quantized_state() -> AppState {
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
    AppState::with_quantized_model(create_test_quantized_model(&config))
        .expect("build quantized AppState")
}

pub(super) async fn body_string(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    String::from_utf8_lossy(&bytes).into_owned()
}

pub(super) async fn post(state: AppState, uri: &str, json: &str) -> (StatusCode, String) {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(json.to_string()))
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    let status = response.status();
    (status, body_string(response).await)
}

pub(super) async fn get(state: AppState, uri: &str) -> (StatusCode, String) {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    let status = response.status();
    (status, body_string(response).await)
}

// ---------------------------------------------------------------------------
// Finding 9 (P0): one request must not be able to kill the process
// ---------------------------------------------------------------------------

/// A `max_tokens` larger than the context window must be CLAMPED, not allocated.
///
/// 0.63.0 sized the KV cache as `prompt + max_tokens` with no ceiling, so this
/// request asked the allocator for ~1 TB and Rust's allocation-error handler
/// aborted the process — which is why the server died with no HTTP reply at all.
/// The clamp makes the cache size a property of the model, not of the request.
#[test]
#[cfg(feature = "gpu")]
fn test_generate_with_cache_clamps_max_tokens_to_context() {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::error::RealizarError;
    use crate::gguf::{ArchConstraints, GGUFConfig, QuantizedGenerateConfig};

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
    let model = create_test_quantized_model(&config);
    // 1000 > context_length 128. Kept small deliberately: with the clamp removed
    // this assertion fails in a second (1003 tokens) instead of running for the
    // hours that `max_tokens: 999_999_999` would take — same defect, fast signal.
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 1000,
        // No stop token, so nothing but the context bound can end this loop —
        // exactly the shape that ran unbounded before the clamp.
        stop_tokens: Vec::new(),
        ..Default::default()
    };

    let tokens = model
        .generate_with_cache(&[1, 2, 3], &gen_config)
        .expect("a clamped request still generates");
    assert_eq!(
        tokens.len(),
        128,
        "prompt + generated must stop at context_length, never at max_tokens"
    );

    // A prompt that alone exceeds the context is still unsatisfiable (GH-167).
    let long_prompt: Vec<u32> = (0..200).map(|i| (i % 256) as u32).collect();
    match model
        .generate_with_cache(&long_prompt, &gen_config)
        .expect_err("a prompt longer than the context cannot be served")
    {
        RealizarError::ContextLimitExceeded { provided, maximum } => {
            assert_eq!(provided, 200);
            assert_eq!(maximum, 128);
        },
        other => panic!("expected ContextLimitExceeded, got {other:?}"),
    }
}

/// The same request over HTTP must be answered, by a server that is still alive.
/// 0.63.0 returned nothing at all: the process was gone (curl exit 52, port dead).
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_generate_survives_oversized_max_tokens() {
    let (status, body) = post(
        quantized_state(),
        "/generate",
        r#"{"prompt":"token5","max_tokens":999999999}"#,
    )
    .await;

    assert_eq!(
        status,
        StatusCode::OK,
        "the request must be answered, not fatal, body: {body}"
    );
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json body");
    let generated = parsed["token_ids"].as_array().expect("token_ids").len();
    assert!(
        generated <= 128,
        "generation must be bounded by the model context (128), got {generated}"
    );

    // The server survived: a normal request on a fresh router still works.
    let (status, _) = post(quantized_state(), "/generate", r#"{"prompt":"token5"}"#).await;
    assert_eq!(status, StatusCode::OK, "server must still serve");
}

/// An over-long PROMPT is still a client error, and it must be reported as one:
/// 0.63.0 answered 500, which tells the caller the server broke.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_oversized_prompt_is_400_not_500() {
    // 300 whitespace-separated tokens against a 128-token context window.
    let prompt = "token5 ".repeat(300);
    let json = serde_json::json!({"prompt": prompt, "max_tokens": 4}).to_string();
    let (status, body) = post(quantized_state(), "/generate", &json).await;

    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a prompt over the context window is a client error, body: {body}"
    );
    assert!(
        body.to_lowercase().contains("context"),
        "the error must name the context limit, got: {body}"
    );
}

// ---------------------------------------------------------------------------
// Findings 1 + 10 (P0/P1): the native routes must work on a quantized server
// ---------------------------------------------------------------------------

#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_tokenize_works_on_quantized_server() {
    let (status, body) = post(quantized_state(), "/tokenize", r#"{"text":"token5"}"#).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        !body.contains("No model available"),
        "tokenizing needs only the tokenizer, got: {body}"
    );
    assert!(body.contains("token_ids"), "body: {body}");
}

#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_batch_tokenize_works_on_quantized_server() {
    let (status, body) = post(
        quantized_state(),
        "/batch/tokenize",
        r#"{"texts":["token5","token6"]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(!body.contains("No model available"), "body: {body}");
}

#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_batch_generate_works_on_quantized_server() {
    let (status, body) = post(
        quantized_state(),
        "/batch/generate",
        r#"{"prompts":["token5"],"max_tokens":2}"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a banner-advertised route must work when a model is loaded, body: {body}"
    );
    assert!(!body.contains("No model available"), "body: {body}");
    assert!(body.contains("results"), "body: {body}");
}

/// `/realize/batch` is the same handler under the spec §5.2 path.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_realize_batch_works_on_quantized_server() {
    let (status, body) = post(
        quantized_state(),
        "/realize/batch",
        r#"{"prompts":["token5"],"max_tokens":2}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(!body.contains("No model available"), "body: {body}");
}

#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_stream_generate_works_on_quantized_server() {
    let response = create_router(quantized_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream/generate")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"token5","max_tokens":2}"#))
                .expect("build request"),
        )
        .await
        .expect("dispatch");

    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap_or_default().to_string())
        .unwrap_or_default();
    let body = body_string(response).await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        content_type.starts_with("text/event-stream"),
        "SSE route must answer with an event stream, got content-type {content_type:?}"
    );
}

// ---------------------------------------------------------------------------
// Finding 5 (P2): one condition, one status code
// ---------------------------------------------------------------------------

/// "This server has no usable model" is 503 on EVERY route that can report it.
/// 0.63.0 answered 404 on `/tokenize` and `/stream/generate` but 500 on
/// `/batch/tokenize` and `/batch/generate` for the identical condition.
#[tokio::test]
async fn test_no_model_available_is_503_everywhere() {
    let cases: [(&str, &str); 4] = [
        ("/tokenize", r#"{"text":"hi"}"#),
        ("/batch/tokenize", r#"{"texts":["hi"]}"#),
        ("/batch/generate", r#"{"prompts":["hi"],"max_tokens":2}"#),
        ("/stream/generate", r#"{"prompt":"hi","max_tokens":2}"#),
    ];

    for (uri, json) in cases {
        let state = AppState::demo_mock().expect("mock state");
        let (status, body) = post(state, uri, json).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{uri} must report a missing model as 503, got {status} / {body}"
        );
    }
}

// ---------------------------------------------------------------------------
// Findings 2 + 4 (P1/P2): documented sampling fields must be honoured or rejected
// ---------------------------------------------------------------------------

/// `resolve_quantized_sampling` errors carry a non-Debug `Json` body; pull the status.
fn rejected_status(
    result: Result<
        crate::api::gpu_handlers::QuantizedSampling,
        (StatusCode, axum::Json<crate::api::ErrorResponse>),
    >,
) -> StatusCode {
    match result {
        Ok(_) => panic!("expected the request to be rejected"),
        Err((status, _)) => status,
    }
}

/// Unwrap a resolved sampling config (the error side is not Debug).
fn accepted(
    result: Result<
        crate::api::gpu_handlers::QuantizedSampling,
        (StatusCode, axum::Json<crate::api::ErrorResponse>),
    >,
) -> crate::api::gpu_handlers::QuantizedSampling {
    match result {
        Ok(sampling) => sampling,
        Err((status, _)) => panic!("expected the request to be accepted, got {status}"),
    }
}

#[test]
fn test_negative_temperature_is_rejected() {
    use crate::api::gpu_handlers::resolve_quantized_sampling;

    for temperature in [-1.0_f32, -5.0, -100.0, f32::NAN] {
        assert_eq!(
            rejected_status(resolve_quantized_sampling("greedy", 50, 0.9, temperature)),
            StatusCode::BAD_REQUEST,
            "temperature {temperature} inverts the softmax and must be rejected"
        );
    }
    // Positive controls: the neighbouring values still work.
    for temperature in [0.0_f32, 0.7, 2.0] {
        let _ = accepted(resolve_quantized_sampling("greedy", 50, 0.9, temperature));
    }
}

#[test]
fn test_strategy_selects_the_sampler() {
    use crate::api::gpu_handlers::resolve_quantized_sampling;

    // greedy => argmax, regardless of a wide top_k / hot temperature.
    let greedy = accepted(resolve_quantized_sampling("greedy", 500, 0.9, 2.0));
    assert_eq!(greedy.top_k, 1, "greedy must be argmax");
    assert_eq!(greedy.top_p, 1.0, "greedy must not narrow by nucleus");

    // top_k is passed through.
    let top_k = accepted(resolve_quantized_sampling("top_k", 500, 0.9, 2.0));
    assert_eq!(top_k.top_k, 500);

    // top_p reaches the engine instead of being discarded.
    let top_p = accepted(resolve_quantized_sampling("top_p", 500, 0.01, 2.0));
    assert_eq!(top_p.top_p, 0.01, "top_p must reach the sampler");
    assert_eq!(top_p.top_k, 0, "nucleus-only sampling disables the top-k cut");

    // temperature 0 is greedy on every surface.
    let zero = accepted(resolve_quantized_sampling("top_k", 500, 0.9, 0.0));
    assert_eq!(zero.top_k, 1);

    // An unknown strategy is a client error, not silence.
    assert_eq!(
        rejected_status(resolve_quantized_sampling("bogus", 50, 0.9, 1.0)),
        StatusCode::BAD_REQUEST
    );

    // An out-of-range top_p is a client error on the branch that uses it.
    assert_eq!(
        rejected_status(resolve_quantized_sampling("top_p", 50, 9.9, 1.0)),
        StatusCode::BAD_REQUEST
    );
}

/// End to end: `strategy:"greedy"` must decode like `temperature:0`, and must NOT
/// match a sampled decode of the same request. 0.63.0 sampled in all three cases.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_greedy_strategy_is_argmax_over_http() {
    let (_, reference) = post(
        quantized_state(),
        "/generate",
        r#"{"prompt":"token5","max_tokens":4,"temperature":0.0}"#,
    )
    .await;
    let (_, greedy) = post(
        quantized_state(),
        "/generate",
        r#"{"prompt":"token5","max_tokens":4,"temperature":2.0,"top_k":500,"strategy":"greedy"}"#,
    )
    .await;
    let (_, sampled) = post(
        quantized_state(),
        "/generate",
        r#"{"prompt":"token5","max_tokens":4,"temperature":2.0,"top_k":500,"strategy":"top_k","seed":7}"#,
    )
    .await;

    assert_eq!(
        greedy, reference,
        "strategy=greedy must decode identically to temperature=0"
    );
    assert_ne!(
        sampled, greedy,
        "a sampled decode must differ from greedy, else `strategy` is being ignored"
    );
}

/// The request-level rejections are visible over HTTP, on a server that has a model.
///
/// `strategy` and `top_p` are refused by the handler (400). `temperature` is
/// refused one layer earlier, by the extractor (422), because validating it in
/// the handler only covered the QUANTIZED backend: on a dense server the same
/// value reached the sampler and came back as
/// `500 "Temperature must be a positive finite number"` (aprender#2375). Both
/// are 4xx; the codes differ because the layers differ.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_generate_rejects_bad_sampling_params_with_400() {
    let handler_rejections = [
        r#"{"prompt":"token5","max_tokens":4,"strategy":"invalid_strategy"}"#,
        r#"{"prompt":"token5","max_tokens":4,"strategy":"top_p","top_p":9.9}"#,
    ];
    for json in handler_rejections {
        let (status, body) = post(quantized_state(), "/generate", json).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{json} must be rejected, got {status} / {body}"
        );
    }

    let extractor_rejections = [
        r#"{"prompt":"token5","max_tokens":4,"temperature":-1.0}"#,
        // `1e40` is a finite JSON number that BECOMES `+inf` as the `f32` this
        // field deserializes into — the case `is_nan() || < 0.0` let through.
        r#"{"prompt":"token5","max_tokens":4,"temperature":1e40}"#,
    ];
    for json in extractor_rejections {
        let (status, body) = post(quantized_state(), "/generate", json).await;
        assert_eq!(
            status,
            StatusCode::UNPROCESSABLE_ENTITY,
            "{json} must be refused before any backend runs, got {status} / {body}"
        );
        assert!(
            body.contains("temperature"),
            "the refusal must name the field: {body}"
        );
    }
    // Positive control on the same server: a well-formed request still succeeds.
    let (status, _) = post(
        quantized_state(),
        "/generate",
        r#"{"prompt":"token5","max_tokens":4,"temperature":0.7,"strategy":"top_p","top_p":0.9}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

/// Two different `seed` values must produce two different samples.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_seed_changes_the_sample_over_http() {
    let body = |seed: u32| {
        format!(
            r#"{{"prompt":"token5","max_tokens":4,"temperature":2.0,"top_k":500,"strategy":"top_k","seed":{seed}}}"#
        )
    };
    let (_, one) = post(quantized_state(), "/generate", &body(1)).await;
    let (_, two) = post(quantized_state(), "/generate", &body(2)).await;
    let (_, one_again) = post(quantized_state(), "/generate", &body(1)).await;

    assert_eq!(one, one_again, "the same seed must be reproducible");
    assert_ne!(one, two, "seed=1 and seed=2 must not be byte-identical");
}

// ---------------------------------------------------------------------------
// Finding 6 (P2): two model-info endpoints must not contradict each other
// ---------------------------------------------------------------------------

#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_models_and_realize_model_agree_on_format() {
    let (_, models) = get(quantized_state(), "/models").await;
    let (_, realize) = get(quantized_state(), "/realize/model").await;

    let models: serde_json::Value = serde_json::from_str(&models).expect("models json");
    let realize: serde_json::Value = serde_json::from_str(&realize).expect("realize json");

    let listed = models["models"][0]["format"]
        .as_str()
        .expect("format field")
        .to_string();
    let detailed = realize["format"].as_str().expect("format field").to_string();

    assert_eq!(
        listed, detailed,
        "/models and /realize/model must report one format for one model"
    );
    assert_eq!(listed, "gguf", "a quantized GGUF model is 'gguf', not 'unknown'");
}

// ---------------------------------------------------------------------------
// Finding 12 (P2): the 404 must point at something that exists
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_404_body_lists_routes() {
    let state = AppState::demo_mock().expect("mock state");
    let (status, body) = get(state, "/nope/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json 404 body");
    let routes = parsed["routes"].as_array().expect("routes array in 404 body");
    assert!(!routes.is_empty(), "404 must list the routes it promises");
    assert!(
        routes.iter().any(|r| r == "POST /generate"),
        "route list must contain the routes this server mounts: {routes:?}"
    );
    assert!(
        !body.contains("See /health for available endpoints"),
        "/health returns no endpoint list, so the 404 must not send clients there"
    );
}

/// Every route the 404 advertises must actually be mounted. This is what stops the
/// list from becoming a second false advertisement as routes move.
#[tokio::test]
async fn test_advertised_routes_are_all_mounted() {
    let state = AppState::demo_mock().expect("mock state");
    let (_, body) = get(state, "/nope/nope").await;
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json 404 body");
    let routes: Vec<String> = parsed["routes"]
        .as_array()
        .expect("routes array")
        .iter()
        .map(|r| r.as_str().unwrap_or_default().to_string())
        .collect();

    for route in routes {
        let (method, path) = route.split_once(' ').expect("METHOD /path");
        // Axum path params are placeholders; substitute something concrete.
        let path = path.replace(":request_id", "not-a-uuid");
        let state = AppState::demo_mock().expect("mock state");
        let response = create_router(state)
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(&path)
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .expect("build request"),
            )
            .await
            .expect("dispatch");
        assert_ne!(
            response.status(),
            StatusCode::NOT_FOUND,
            "{method} {path} is advertised in the 404 body but is not mounted"
        );
    }
}
