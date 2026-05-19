//! V1_001 integration test for qwen3-moe-serve-dispatch-v1.yaml (aprender#1789).
//!
//! Formal cargo-test discharge of FALSIFY-QWEN3_MOE_SERVE_DISPATCH_V1_001:
//! "POST /v1/chat/completions with qwen3_moe model returns a non-error response."
//!
//! Boots an in-process axum router against a real Qwen3-MoE GGUF (mmap-backed),
//! POSTs a small chat request, asserts HTTP 200 + non-empty assistant content.
//!
//! Discharges V1_001 + V1_003 (no matmul defensive guard fire) at the cargo
//! test level. Smoke test of `apr serve` had already discharged both
//! empirically (see paiml/claude-code-parity-apr
//! `evidence/phase-6/30b-moe-empirical-2026-05-19.md`); this test pins the
//! invariant into CI.
//!
//! ## Gating
//!
//! Test is `#[ignore]` by default — needs a Qwen3-MoE GGUF on disk.
//! Run with:
//!
//! ```text
//! QWEN3_MOE_GGUF_PATH=/path/to/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
//!   cargo test --test qwen3_moe_serve_dispatch_v1 -- --ignored --nocapture
//! ```

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use realizar::api::{create_router, AppState};
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::sync::Arc;
use tower::ServiceExt;

fn gguf_path() -> Option<String> {
    std::env::var("QWEN3_MOE_GGUF_PATH").ok()
}

fn ensure_qwen3_moe_arch(arch: &str) {
    let canonical = realizar::tensor_names::normalize_architecture(arch);
    assert_eq!(
        canonical, "qwen3_moe",
        "GGUF at QWEN3_MOE_GGUF_PATH must be qwen3_moe-arch (got raw '{arch}', canonical '{canonical}')"
    );
}

/// Build AppState wired the way `apr serve run <gguf> --gpu` (CPU fallback path)
/// would: mapped Arc + quantized model + tokenizer with the real vocab + cached
/// architecture.
fn build_app_state(gguf_path: &str) -> AppState {
    let mapped = Arc::new(
        MappedGGUFModel::from_path(gguf_path)
            .unwrap_or_else(|e| panic!("Failed to mmap GGUF at {gguf_path}: {e}")),
    );
    let arch = mapped
        .model
        .architecture()
        .expect("GGUF must declare general.architecture metadata")
        .to_string();
    ensure_qwen3_moe_arch(&arch);

    let quantized = OwnedQuantizedModel::from_mapped(&mapped)
        .expect("Failed to build OwnedQuantizedModel from MoE GGUF");

    let vocab_size = quantized.config().vocab_size;
    let vocab: Vec<String> = mapped.model.vocabulary().unwrap_or_else(|| {
        let mut v: Vec<String> = (0..vocab_size).map(|i| format!("token{i}")).collect();
        if !v.is_empty() {
            v[0] = "<unk>".to_string();
        }
        v
    });

    AppState::with_quantized_model_and_vocab(quantized, vocab)
        .expect("AppState::with_quantized_model_and_vocab failed")
        .with_mapped_gguf_model(mapped)
}

#[tokio::test]
#[ignore = "requires real Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH env var"]
async fn falsify_qwen3_moe_serve_dispatch_v1_001() {
    let Some(path) = gguf_path() else {
        eprintln!(
            "SKIP: QWEN3_MOE_GGUF_PATH not set. Discharges V1_001 only when a \
             real qwen3_moe GGUF is available. See \
             contracts/qwen3-moe-serve-dispatch-v1.yaml."
        );
        return;
    };

    let state = build_app_state(&path);
    let app = create_router(state);

    let body = serde_json::json!({
        "model": "qwen3-moe-v1-001",
        "messages": [
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 4,
        "temperature": 0.0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = app
        .oneshot(request)
        .await
        .expect("router oneshot dispatch failed");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collect failed")
        .to_bytes();

    let body_str = String::from_utf8_lossy(&bytes);

    // V1_001: HTTP 200 + non-error JSON shape
    assert_eq!(
        status,
        StatusCode::OK,
        "qwen3_moe dispatch must return 200, got {status} (body: {body_str})"
    );

    // V1_003: no matmul defensive guard fire — the dense FFN path was not
    // reached (which would produce InvalidShape with the #1790 guard's signature
    // text). If the response is 200, the MoE path produced tokens.
    assert!(
        !body_str.contains("InvalidShape"),
        "V1_003 violation: matmul guard fired (InvalidShape in body): {body_str}"
    );
    assert!(
        !body_str.contains("matmul weight has EMPTY data buffer"),
        "V1_003 violation: matmul defensive guard message in body: {body_str}"
    );

    let parsed: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response body must be JSON");
    let content = parsed
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .expect("response must have choices[0].message.content");
    assert!(
        !content.is_empty(),
        "V1_001 violation: assistant content empty"
    );

    eprintln!("V1_001 + V1_003 discharged. Body excerpt: {body_str}");
}
