//! aprender#3856 row 3 falsifiers for `GET /v1/capability`.
//!
//! The endpoint reports what this build can and cannot run on the GPU, and why,
//! from the embedded `apr-model-capability-v1` contract. Every assertion is one
//! a client can observe over HTTP:
//!
//! 1. The route is MOUNTED on a model-less server and ADVERTISED by `GET /` and
//!    the 404 body.
//! 2. It carries a known unsupported row with its reason (LayerNorm has no GPU
//!    kernel), so an empty or stubbed body fails.
//! 3. Its `ops` / `quant_types` equal the REPO-ROOT contract's, read from disk
//!    here — the served facts are the source's, not a stale copy's.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

async fn get_json(uri: &str) -> (StatusCode, serde_json::Value) {
    let state = AppState::demo_mock().expect("model-less AppState");
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

fn advertises(list: &serde_json::Value) -> bool {
    list.as_array()
        .is_some_and(|routes| routes.iter().any(|r| r.as_str() == Some("GET /v1/capability")))
}

#[tokio::test]
async fn capability_answers_on_a_model_less_server() {
    let (status, body) = get_json("/v1/capability").await;
    assert_eq!(status, StatusCode::OK, "got {status} with {body}");
    assert_eq!(
        body["source"].as_str(),
        Some("contracts/apr-model-capability-v1.yaml")
    );
    assert!(body["op_implementation"].is_array(), "{body}");
}

#[tokio::test]
async fn capability_is_advertised_by_the_index_and_the_404_body() {
    let (_, index) = get_json("/").await;
    assert!(advertises(&index["routes"]), "index:\n{index}");
    let (status, notfound) = get_json("/v1/no-such-route").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(advertises(&notfound["routes"]), "404 body:\n{notfound}");
}

#[tokio::test]
async fn layernorm_is_reported_unsupported_with_a_reason() {
    let (_, body) = get_json("/v1/capability").await;
    let row = body["ops"]
        .as_array()
        .expect("ops sequence")
        .iter()
        .find(|r| r["op"].as_str() == Some("LayerNorm"))
        .unwrap_or_else(|| panic!("ops has no LayerNorm row:\n{body}"));
    assert_eq!(row["gpu_supported"].as_bool(), Some(false), "{row}");
    assert!(
        row["reason"].as_str().is_some_and(|r| !r.is_empty()),
        "an unsupported op must say why: {row}"
    );
}

#[tokio::test]
async fn served_sections_equal_the_repo_root_contract() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../contracts/apr-model-capability-v1.yaml"
    );
    let text = std::fs::read_to_string(path).expect("read repo-root contract");
    let source: serde_json::Value = serde_yaml_ng::from_str(&text).expect("parse contract");
    let (_, body) = get_json("/v1/capability").await;
    for key in ["ops", "quant_types", "op_implementation"] {
        assert_eq!(body[key], source[key], "`{key}` differs from the source contract");
    }
}
