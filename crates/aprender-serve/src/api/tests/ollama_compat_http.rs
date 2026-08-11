//! Falsifiers for the Ollama-compat and `/realize/*` metadata surfaces, at the
//! HTTP layer where the defects were observed.
//!
//! Dogfooding `cargo install aprender` 0.63.0 from crates.io found four things
//! that a unit test on a handler function could not have caught, because each
//! is a property of the ROUTER or of the response envelope:
//!
//! - `GET /api/tags`, `POST /api/show`, `GET /api/version` answered 404 while
//!   the startup banner advertised "Ollama-Parity Endpoints" (#2396). Every
//!   Ollama client calls `/api/tags` before it will issue a chat request, so
//!   the server was unreachable to all of them.
//! - `POST /api/chat` and `/api/generate` accepted `"stream":true` and returned
//!   one buffered JSON object with a `content-length` (#2396).
//! - `GET /realize/model` returned `size_bytes: 0`, `context_length: 4096`,
//!   `format: "gguf"`, `quantization: "Q4_K_M"` and a `content_hash` of
//!   `"blake3:0".repeat(16)` for every model (#2402).
//! - `POST /realize/reload`'s 501 told the caller to "Start server with
//!   --registry flag" — a flag `apr serve run` does not have (#2402).

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::test_helpers::create_test_app_shared;
use crate::api::{create_router, AppState, ModelSourceInfo};

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).expect("body is JSON")
}

// ---------------------------------------------------------------------------
// #2396 finding 2: the discovery routes must EXIST.
// ---------------------------------------------------------------------------

/// The 404 body 0.63.0 returned for these paths. If a route is missing, the
/// axum fallback produces this — so the tests below check for its ABSENCE,
/// which is what distinguishes "route wired" from "route missing".
fn is_route_not_found(json: &serde_json::Value) -> bool {
    json.get("error").and_then(serde_json::Value::as_str) == Some("not_found")
}

#[tokio::test]
async fn api_tags_is_routed_and_lists_a_model_the_client_can_ask_for() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/tags")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "/api/tags 404'd in 0.63.0; every Ollama client calls it first"
    );
    let json = body_json(response).await;
    assert!(!is_route_not_found(&json), "got the fallback body: {json}");

    let models = json["models"].as_array().expect("models is an array");
    assert_eq!(models.len(), 1, "single-model server advertises one tag");
    let name = models[0]["name"].as_str().expect("name is a string");
    assert!(
        !name.is_empty() && name.contains(':'),
        "name must be an addressable `model:tag`, got {name:?}"
    );
    assert_eq!(models[0]["model"], models[0]["name"]);
}

#[tokio::test]
async fn api_version_is_routed_and_returns_a_parseable_version() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/version")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    let version = json["version"].as_str().expect("version is a string");
    // Clients version-gate on this; it must be a real dotted version, not a
    // placeholder like "unknown".
    assert!(
        version.split('.').count() >= 2 && version.starts_with(|c: char| c.is_ascii_digit()),
        "version {version:?} must be a semver-shaped string"
    );
}

#[tokio::test]
async fn api_show_is_routed_and_answers_a_post() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/show")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"apr"}"#))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert!(!is_route_not_found(&json), "got the fallback body: {json}");
    assert!(json.get("details").is_some(), "show must carry details");
    assert!(
        json["capabilities"]
            .as_array()
            .is_some_and(|c| c.iter().any(|v| v == "completion")),
        "a server that serves /api/chat has the completion capability"
    );
}

/// `/api/show` with no recorded model source must not invent metadata. This is
/// the same "measured or absent" rule `/realize/model` violated.
#[tokio::test]
async fn api_show_without_a_measured_source_claims_nothing() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/show")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("request"),
        )
        .await
        .expect("response");

    let json = body_json(response).await;
    let details = json["details"].as_object().expect("details is an object");
    assert!(
        details.is_empty(),
        "unmeasured details must be absent, got {details:?}"
    );
    let info = json["model_info"].as_object().expect("model_info object");
    assert!(
        info.is_empty(),
        "unmeasured model_info must be absent, got {info:?}"
    );
}

/// With a measured source attached, the SAME endpoints report the measured
/// values. This is the other half of the falsifier: absence must be caused by
/// not knowing, not by the fields being unreachable.
#[tokio::test]
async fn api_show_reports_the_measured_source() {
    let source = ModelSourceInfo::default()
        .with_architecture("qwen2")
        .with_quantization("Q4_K")
        .with_context_length(128)
        .with_model_max_context_length(32768);
    let state = AppState::demo().expect("demo state").with_model_source(source);
    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/show")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("request"),
        )
        .await
        .expect("response");

    let json = body_json(response).await;
    assert_eq!(json["details"]["family"], "qwen2");
    assert_eq!(json["details"]["quantization_level"], "Q4_K");
    // The served bound and the model's own maximum are DIFFERENT facts. 0.63.0
    // collapsed them into a constant 4096 and reported that for both.
    assert_eq!(json["model_info"]["apr.configured_context_length"], 128);
    assert_eq!(json["model_info"]["general.context_length"], 32768);
}

// ---------------------------------------------------------------------------
// #2396 finding 3: stream:true must produce NDJSON, not one buffered object.
// ---------------------------------------------------------------------------

/// Read the body as text and parse it as newline-delimited JSON.
async fn ndjson_objects(response: axum::response::Response) -> Vec<serde_json::Value> {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let text = String::from_utf8(bytes.to_vec()).expect("utf-8 body");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("each line is a JSON object"))
        .collect()
}

#[tokio::test]
async fn api_chat_with_stream_true_returns_ndjson_terminated_by_done() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"apr","messages":[{"role":"user","content":"hi"}],"stream":true,"options":{"num_predict":8}}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("application/x-ndjson"),
        "stream:true must be framed as NDJSON, got content-type {content_type:?}"
    );
    // 0.63.0's proof of buffering: a content-length on a streamed response.
    assert!(
        response
            .headers()
            .get(axum::http::header::CONTENT_LENGTH)
            .is_none(),
        "a streamed body must not be length-prefixed"
    );

    let objects = ndjson_objects(response).await;
    assert!(!objects.is_empty(), "stream must carry at least one object");
    let last = objects.last().expect("non-empty");
    assert_eq!(last["done"], true, "stream must terminate with done:true");
    for obj in &objects[..objects.len() - 1] {
        assert_eq!(obj["done"], false, "only the last object may be done:true");
    }
}

#[tokio::test]
async fn api_generate_with_stream_true_returns_ndjson_terminated_by_done() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"apr","prompt":"count","stream":true,"options":{"num_predict":8}}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    // Same two wire facts as /api/chat. Without them this test passes on the
    // 0.63.0 behaviour: one buffered object IS one parseable NDJSON line with
    // done:true, so only the FRAMING distinguishes fixed from broken.
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("application/x-ndjson"),
        "stream:true must be framed as NDJSON, got content-type {content_type:?}"
    );
    assert!(
        response
            .headers()
            .get(axum::http::header::CONTENT_LENGTH)
            .is_none(),
        "a streamed body must not be length-prefixed"
    );

    let objects = ndjson_objects(response).await;
    let last = objects.last().expect("non-empty");
    assert_eq!(last["done"], true);
    assert!(
        last.get("message").is_none(),
        "/api/generate uses a flat `response`, never a nested message"
    );
    // The terminal object carries the counts; content chunks do not.
    assert!(last.get("done_reason").is_some());
}

/// `stream:false` (and an absent `stream`) must keep the single-object shape —
/// the fix must not break the clients that were working.
#[tokio::test]
async fn api_chat_without_stream_stays_a_single_json_object() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"apr","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"num_predict":4}}"#,
                ))
                .expect("request"),
        )
        .await
        .expect("response");

    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("application/json"),
        "non-streaming must stay application/json, got {content_type:?}"
    );
    let json = body_json(response).await;
    assert_eq!(json["done"], true);
    assert!(json["message"]["role"] == "assistant");
}

// ---------------------------------------------------------------------------
// #2402 finding 1: /realize/model must never fabricate provenance.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn realize_model_never_emits_a_synthetic_content_hash() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/realize/model")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;

    // The exact 128-character string 0.63.0 shipped, and the shape of it.
    let serialized = json.to_string();
    assert!(
        !serialized.contains("blake3:0blake3:0"),
        "synthetic content hash still on the wire: {serialized}"
    );
    assert!(
        json.get("lineage").is_none(),
        "lineage must be absent until a real content hash is computed, got {json}"
    );
    // Unmeasured metadata is ABSENT, not defaulted. `size_bytes: 0` reads as a
    // zero-byte model; `format: "gguf"` misreports every other container.
    for field in ["size_bytes", "context_length", "quantization", "format"] {
        assert!(
            json.get(field).is_none(),
            "{field} must be absent when unmeasured, got {}",
            json[field]
        );
    }
    // `loaded` must report the resident model, not a literal.
    //
    // This asserted `true` against a `demo_mock()` fixture, which is documented
    // as "no model = no inference overhead" — so the server has no model and the
    // endpoint was answering `loaded: true` regardless. That is the same
    // fabricated provenance this test exists to forbid, one field over: the
    // handler hardcoded `loaded: true` while `size_bytes` and `format` were
    // being correctly reported as ABSENT-when-unmeasured.
    //
    // Now sourced from `state.model_loaded()`, so on this fixture it is false.
    assert_eq!(
        json["loaded"], false,
        "the shared test fixture is demo_mock() and has no model resident, so \
         /realize/model must not claim one is loaded: {json}"
    );
}

#[tokio::test]
async fn realize_model_reports_measured_values_when_the_loader_supplied_them() {
    let dir = std::env::temp_dir().join(format!(
        "apr-realize-model-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("fixture.gguf");
    // 4-byte magic + 60 bytes of payload = 64 bytes on disk.
    std::fs::write(&path, [b"GGUF".to_vec(), vec![0u8; 60]].concat()).expect("write");

    let source = ModelSourceInfo::from_path(&path)
        .with_architecture("qwen2")
        .with_quantization("Q4_K")
        .with_context_length(128)
        .with_model_max_context_length(32768);
    let state = AppState::demo().expect("demo state").with_model_source(source);
    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/realize/model")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let json = body_json(response).await;

    assert_eq!(json["size_bytes"], 64, "size must be the real file size");
    assert_eq!(json["format"], "gguf", "format comes from the magic bytes");
    assert_eq!(json["quantization"], "Q4_K");
    assert_eq!(json["architecture"], "qwen2");
    // The two context facts stay distinct. 0.63.0 answered 4096 for a server
    // started with --context-length 128 against a 32768-context model.
    assert_eq!(json["context_length"], 128);
    assert_eq!(json["model_max_context_length"], 32768);
    assert!(
        json.get("lineage").is_none(),
        "no hash was computed, so no lineage"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A non-GGUF file must not be reported as `gguf`. 0.63.0 hardcoded the string.
#[tokio::test]
async fn realize_model_does_not_call_an_apr_file_gguf() {
    let dir = std::env::temp_dir().join(format!(
        "apr-realize-model-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("fixture.apr");
    std::fs::write(&path, [b"APR\0".to_vec(), vec![0u8; 28]].concat()).expect("write");

    let state = AppState::demo()
        .expect("demo state")
        .with_model_source(ModelSourceInfo::from_path(&path));
    let app = create_router(state);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/realize/model")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let json = body_json(response).await;
    assert_eq!(json["format"], "apr");

    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// #2402 finding 2: the 501 must not name a flag that does not exist.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn realize_reload_501_does_not_advertise_a_nonexistent_flag() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/realize/reload")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    let json = body_json(response).await;
    let error = json["error"].as_str().expect("error is a string");

    // `apr serve run --registry <FILE>` is rejected by clap with exit 2: the
    // flag does not exist on `apr serve run` or on `apr serve`. An error that
    // tells the user how to fix the problem must name a real remedy.
    assert!(
        !error.contains("Start server with --registry flag"),
        "the 0.63.0 dead-end advice is still on the wire: {error}"
    );
    assert!(
        error.contains("does not expose registry mode"),
        "the message must state that the CLI cannot enable this: {error}"
    );
    // And it must still say what the caller CAN do.
    assert!(
        error.contains("apr serve run"),
        "message must offer the working alternative: {error}"
    );
}
