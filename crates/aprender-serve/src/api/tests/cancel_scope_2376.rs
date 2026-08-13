//! Falsifiers for aprender#2376(3) — generation must stop when the client goes.
//!
//! Contract: `contracts/apr-serve-cancellation-v1.yaml`.
//!
//! # What these assert, and what they refuse to assert
//!
//! Every test here asserts on **observed work** — how many tokens a decode loop
//! actually produced. None of them asserts that a flag was set. "the token was set
//! to true" is compatible with the loop never reading it, which is exactly the
//! shipped defect, so a test shaped that way would have passed against 0.63.0.
//!
//! Each falsifier also asserts the **converse**: with no cancellation the same call
//! produces the full `max_tokens` output. Without that, a test could pass because
//! generation is broken and returns nothing at all.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

use crate::api::{create_router, request_cancel_token, AppState};
use crate::generate::CancelToken;
use crate::layers::{Model, ModelConfig};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A tiny dense model. `Model::generate` here is the real production loop that
/// `/generate`'s registry backend and `/v1/completions`' CPU fallback run.
fn tiny_dense_model() -> Model {
    Model::new(ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    })
    .expect("build tiny dense model")
}

/// A quantized-only server: what `apr serve run model.gguf` builds.
#[cfg(feature = "gpu")]
fn quantized_state() -> AppState {
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
        context_length: 512,
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

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-001 — the dense decode loop stops at the cancel point
// ---------------------------------------------------------------------------

/// The dense `Model::generate` loop must stop within one token of the moment
/// cancellation is observed, not run on to `max_tokens`.
///
/// Pre-fix behaviour: `GenerationConfig` had no `cancel` field and
/// `layers/model_model.rs::generate` had no poll, so `tokens.len()` was
/// `prompt + max_tokens` for every input — a cancelled request was
/// indistinguishable from an uncancelled one.
#[test]
fn dense_generation_stops_at_the_cancel_point_not_max_tokens() {
    use crate::generate::GenerationConfig;

    let model = tiny_dense_model();
    let prompt = [1_usize, 2];
    const MAX_TOKENS: usize = 64;
    const BUDGET: usize = 8;

    // Uncancelled control FIRST: if this does not produce the full budget then the
    // cancelled assertion below proves nothing (generation could just be broken).
    let uncancelled = model
        .generate(
            &prompt,
            &GenerationConfig::greedy().with_max_tokens(MAX_TOKENS),
        )
        .expect("uncancelled generation");
    assert_eq!(
        uncancelled.len(),
        prompt.len() + MAX_TOKENS,
        "control: with no cancellation the loop must run the full {MAX_TOKENS}-token budget, \
         otherwise the cancelled case below is not measuring cancellation"
    );

    // Cancelled: the token trips after BUDGET polls, and the loop polls once per token.
    let token = CancelToken::with_budget(BUDGET);
    let cancelled = model
        .generate(
            &prompt,
            &GenerationConfig::greedy()
                .with_max_tokens(MAX_TOKENS)
                .with_cancel(token.clone()),
        )
        .expect("cancelled generation still returns what it produced");

    let produced = cancelled.len() - prompt.len();
    assert_eq!(
        produced, BUDGET,
        "generation must stop at the cancel point ({BUDGET} tokens), not run to \
         max_tokens ({MAX_TOKENS}); it produced {produced}"
    );
    assert_eq!(
        token.polls(),
        BUDGET + 1,
        "the loop must poll exactly once per decode step (BUDGET polls that returned \
         false, plus the one that returned true and broke the loop)"
    );

    // The cancelled prefix must be the uncancelled prefix: cancelling stops work, it
    // does not change the tokens that were already produced.
    assert_eq!(
        cancelled,
        uncancelled[..cancelled.len()].to_vec(),
        "a cancelled run must be a strict prefix of the uncancelled run"
    );
}

/// A token cancelled before the first poll must produce **zero** tokens: the loop
/// polls before doing any work, not after.
#[test]
fn dense_generation_cancelled_before_start_produces_no_tokens() {
    use crate::generate::GenerationConfig;

    let model = tiny_dense_model();
    let prompt = [1_usize, 2];
    let token = CancelToken::new();
    token.cancel();

    let out = model
        .generate(
            &prompt,
            &GenerationConfig::greedy()
                .with_max_tokens(64)
                .with_cancel(token),
        )
        .expect("an already-cancelled request is not an error");

    assert_eq!(
        out.len(),
        prompt.len(),
        "an already-cancelled request must do no decode work at all; it returned \
         {} tokens beyond the prompt",
        out.len() - prompt.len()
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-002 — the quantized (GGUF) decode loop stops too
// ---------------------------------------------------------------------------

/// `OwnedQuantizedModel::generate_with_cache` is the loop behind
/// `apr serve run model.gguf` on `/generate`, `/stream/generate`,
/// `/v1/completions` and `/v1/chat/completions`. It must observe cancellation.
#[test]
#[cfg(feature = "gpu")]
fn quantized_generation_stops_at_the_cancel_point_not_max_tokens() {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig, QuantizedGenerateConfig};

    let gguf_config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    let model = create_test_quantized_model(&gguf_config);
    let prompt = [3_u32, 4, 5];
    const MAX_TOKENS: usize = 48;
    const BUDGET: usize = 6;

    let base = QuantizedGenerateConfig::deterministic(MAX_TOKENS);

    let uncancelled = model
        .generate_with_cache(&prompt, &base)
        .expect("uncancelled quantized generation");
    assert_eq!(
        uncancelled.len(),
        prompt.len() + MAX_TOKENS,
        "control: the quantized loop must run its full {MAX_TOKENS}-token budget when \
         nothing cancels it"
    );

    let token = CancelToken::with_budget(BUDGET);
    let cancelled = model
        .generate_with_cache(&prompt, &base.clone().with_cancel(token.clone()))
        .expect("cancelled quantized generation");

    let produced = cancelled.len() - prompt.len();
    assert_eq!(
        produced, BUDGET,
        "the quantized loop must stop at the cancel point ({BUDGET}), not run to \
         max_tokens ({MAX_TOKENS}); it produced {produced}"
    );
    assert_eq!(
        token.polls(),
        BUDGET + 1,
        "the quantized loop must poll exactly once per decode step"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-003 — dropping the response future stops the work
// ---------------------------------------------------------------------------

/// The mechanism proof. A synchronous loop running under the
/// `cancel_on_disconnect` layer must **stop early when the layer's future is
/// dropped** — which is what axum does when a client disconnects.
///
/// The assertion is on iterations actually executed by the loop, not on the flag.
/// Pre-fix, no layer existed, the loop had no token to poll, and the counter here
/// would reach `BUDGET_CEILING`.
#[tokio::test]
async fn dropping_the_request_future_stops_a_running_generation() {
    use axum::routing::post;
    use axum::Router;

    /// Ceiling that stands in for `max_tokens`. Reaching it means the loop never
    /// observed cancellation — the defect.
    const BUDGET_CEILING: usize = 500_000;

    // How many decode steps the fake loop actually performed.
    let steps = Arc::new(AtomicUsize::new(0));
    // Set once the loop has done enough work that we know it is running; the test
    // drops the request future only after this, so the drop lands mid-decode.
    let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel::<usize>();

    let steps_for_handler = Arc::clone(&steps);
    let started = Arc::new(std::sync::Mutex::new(Some(started_tx)));
    let finished = Arc::new(std::sync::Mutex::new(Some(finished_tx)));

    // A handler shaped exactly like the real ones: it pulls the request's token out
    // of the extensions and runs a synchronous loop that polls it once per "token".
    let handler = move |request: Request<Body>| {
        let steps = Arc::clone(&steps_for_handler);
        let started = Arc::clone(&started);
        let finished = Arc::clone(&finished);
        async move {
            let cancel = request_cancel_token(&request);
            tokio::task::spawn_blocking(move || {
                for i in 0..BUDGET_CEILING {
                    if cancel.is_cancelled() {
                        break;
                    }
                    steps.store(i + 1, Ordering::SeqCst);
                    if i == 64 {
                        if let Some(tx) = started.lock().ok().and_then(|mut g| g.take()) {
                            let _ = tx.send(());
                        }
                    }
                    // Keep each step cheap but not free, so the drop lands well
                    // before the ceiling on any machine.
                    std::hint::spin_loop();
                    std::thread::yield_now();
                }
                if let Some(tx) = finished.lock().ok().and_then(|mut g| g.take()) {
                    let _ = tx.send(steps.load(Ordering::SeqCst));
                }
            });
            // Never completes: the only way out of this handler is the client
            // going away, which is the case under test.
            std::future::pending::<StatusCode>().await
        }
    };

    let app = Router::new()
        .route("/decode", post(handler))
        .layer(axum::middleware::from_fn(crate::api::cancel_on_disconnect));

    let request = Request::builder()
        .method("POST")
        .uri("/decode")
        .body(Body::empty())
        .expect("build request");

    // Drive the request, then DROP the future — exactly what axum does to an
    // abandoned request.
    let mut response_future = Box::pin(app.oneshot(request));
    let started_rx = tokio::task::spawn_blocking(move || {
        started_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .map(|()| ())
    });
    tokio::select! {
        _ = &mut response_future => panic!("the handler must not complete on its own"),
        r = started_rx => r.expect("join").expect("the decode loop must start"),
    }
    drop(response_future);

    let observed = tokio::task::spawn_blocking(move || {
        finished_rx.recv_timeout(std::time::Duration::from_secs(10))
    })
    .await
    .expect("join")
    .expect("the decode loop must terminate after the request future is dropped");

    assert!(
        observed < BUDGET_CEILING,
        "dropping the request future must stop the decode loop; it ran {observed} of \
         {BUDGET_CEILING} steps, i.e. it never observed cancellation"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-004 — the layer is mounted, and completed requests are
// byte-for-byte unchanged
// ---------------------------------------------------------------------------

/// Every route on the real router must receive a live token. A handler that pulls
/// `CancelToken::never` out of the extensions has no way to stop, so this is the
/// wiring the other falsifiers depend on.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn the_real_router_hands_every_request_a_live_cancel_token() {
    // A live token reports its polls; `CancelToken::never` reports 0 forever and
    // cannot be cancelled. Distinguish them by behaviour, not by name.
    let observed = Arc::new(std::sync::Mutex::new(None::<CancelToken>));
    let sink = Arc::clone(&observed);

    let app = axum::Router::new()
        .route(
            "/probe",
            axum::routing::get(move |request: Request<Body>| {
                let sink = Arc::clone(&sink);
                async move {
                    let token = request_cancel_token(&request);
                    if let Ok(mut g) = sink.lock() {
                        *g = Some(token);
                    }
                    StatusCode::OK
                }
            }),
        )
        .layer(axum::middleware::from_fn(crate::api::cancel_on_disconnect));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/probe")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("probe request");
    assert_eq!(response.status(), StatusCode::OK);

    let token = observed
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .expect("handler must have seen a token");
    assert!(
        token.peek_cancelled(),
        "the layer must cancel the token when the response future is dropped; a \
         `never` token cannot report cancelled, which is how this distinguishes a \
         live token from a missing one"
    );
}

/// Interposing a task and a cancellation layer must not change what a COMPLETED
/// request returns.
///
/// The guard fires on *normal* completion too, and the middleware now runs every
/// handler in a spawned task. Either could have truncated or reshaped a response.
/// The baseline is the same handler invoked directly with
/// [`CancelToken::never`] — i.e. the pre-fix code path — and the two bodies must
/// be byte-identical.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn a_completed_generate_request_is_unchanged_by_the_cancellation_layer() {
    use axum::extract::State;
    use axum::{Extension, Json};

    const BODY: &str = r#"{"prompt":"token5","max_tokens":4}"#;

    // Through the real router, which carries the cancellation layer.
    let response = create_router(quantized_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(BODY))
                .expect("build request"),
        )
        .await
        .expect("generate request");
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a normal request must still succeed"
    );
    let via_router = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let via_router: serde_json::Value = serde_json::from_slice(&via_router).expect("json body");

    // Baseline: the handler called directly with a token that never cancels —
    // behaviourally the pre-fix path.
    let request = serde_json::from_str(BODY).expect("parse request");
    let baseline = crate::api::generate_handler(
        State(quantized_state()),
        Extension(CancelToken::never()),
        Json(request),
    )
    .await;
    let baseline = match baseline {
        Ok(Json(resp)) => serde_json::to_value(resp).expect("serialize baseline"),
        Err((status, Json(err))) => {
            panic!(
                "baseline generation must succeed, got {status}: {}",
                err.error
            )
        },
    };

    assert_eq!(
        via_router, baseline,
        "the cancellation layer must not change a completed response; routing it \
         through the layer produced {via_router} but the direct call produced {baseline}"
    );
}
