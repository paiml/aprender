//! CRUX-C-03 falsification tests — OpenAI /v1/chat/completions (stream=false)
//! returns a single `chat.completion` JSON object.
//!
//! Contract: `contracts/crux-C-03-v1.yaml` (competitor parity: vLLM
//! openai_compatible_server, canonical reference:
//! https://platform.openai.com/docs/api-reference/chat/create).
//!
//! In-process axum via `tower::ServiceExt::oneshot`; no TCP, no network.
//! The canonical `/v1/chat/completions` route dispatches the stream=false
//! path through `registry_fallback` for the demo `AppState` (no GPU/CUDA/
//! quantized models loaded in tests).

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use realizar::api::{create_router, AppState};
use tower::ServiceExt;

fn router() -> axum::Router {
    let state = AppState::demo().expect("demo state should build");
    create_router(state)
}

fn chat_request(payload: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .expect("request build")
}

async fn collect_bytes(resp: axum::http::Response<Body>) -> Vec<u8> {
    resp.into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes()
        .to_vec()
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-03-001: response conforms to the OpenAI chat.completion
// schema (id/object/created/model/choices[0].{index,message,finish_reason}).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn falsify_crux_c_03_001_chat_completion_schema() {
    let req = chat_request(serde_json::json!({
        "model": "default",
        "messages": [{"role":"user","content":"Say hi."}],
        "stream": false,
        "max_tokens": 16,
        "temperature": 0.01
    }));
    let resp = router().oneshot(req).await.expect("oneshot");
    let status = resp.status();
    if status != StatusCode::OK {
        let bytes = collect_bytes(resp).await;
        panic!(
            "FALSIFY-CRUX-C-03-001: non-stream must return 200, got {status}: body={:?}",
            String::from_utf8_lossy(&bytes)
        );
    }

    let bytes = collect_bytes(resp).await;
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "FALSIFY-CRUX-C-03-001: body must be JSON, got {} ({e})",
            String::from_utf8_lossy(&bytes)
        )
    });

    let id = v["id"].as_str().unwrap_or("");
    assert!(
        !id.is_empty(),
        "FALSIFY-CRUX-C-03-001: id must be a non-empty string, got {v}"
    );
    assert_eq!(
        v["object"].as_str(),
        Some("chat.completion"),
        "FALSIFY-CRUX-C-03-001: object must equal literal \"chat.completion\", got {v}"
    );
    let created = v["created"].as_u64().unwrap_or(0);
    assert!(
        created > 0,
        "FALSIFY-CRUX-C-03-001: created must be a positive integer, got {v}"
    );
    assert!(
        v["model"].as_str().is_some_and(|s| !s.is_empty()),
        "FALSIFY-CRUX-C-03-001: model must be a non-empty string, got {v}"
    );

    let choices = v["choices"]
        .as_array()
        .unwrap_or_else(|| panic!("FALSIFY-CRUX-C-03-001: choices must be an array, got {v}"));
    assert!(
        !choices.is_empty(),
        "FALSIFY-CRUX-C-03-001: choices must have length >= 1, got {v}"
    );
    let c0 = &choices[0];
    assert_eq!(
        c0["index"].as_u64(),
        Some(0),
        "FALSIFY-CRUX-C-03-001: choices[0].index must be 0, got {c0}"
    );
    assert_eq!(
        c0["message"]["role"].as_str(),
        Some("assistant"),
        "FALSIFY-CRUX-C-03-001: choices[0].message.role must be \"assistant\", got {c0}"
    );
    assert!(
        c0["message"]["content"].is_string(),
        "FALSIFY-CRUX-C-03-001: choices[0].message.content must be a string, got {c0}"
    );
    let fr = c0["finish_reason"].as_str().unwrap_or("");
    assert!(
        matches!(fr, "stop" | "length" | "content_filter" | "tool_calls"),
        "FALSIFY-CRUX-C-03-001: finish_reason {fr:?} not in \
         {{stop,length,content_filter,tool_calls}}; full choice={c0}"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-03-002: token-accounting identity
// usage.total_tokens == usage.prompt_tokens + usage.completion_tokens.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn falsify_crux_c_03_002_usage_token_accounting_identity() {
    let req = chat_request(serde_json::json!({
        "model": "default",
        "messages": [{"role":"user","content":"What is 2+2?"}],
        "stream": false,
        "max_tokens": 16,
        "temperature": 0.01
    }));
    let resp = router().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = collect_bytes(resp).await;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).expect("FALSIFY-CRUX-C-03-002: body must be JSON");

    let usage = &v["usage"];
    let prompt = usage["prompt_tokens"].as_u64().unwrap_or_else(|| {
        panic!("FALSIFY-CRUX-C-03-002: usage.prompt_tokens must be a u64, got {v}")
    });
    let completion = usage["completion_tokens"].as_u64().unwrap_or_else(|| {
        panic!("FALSIFY-CRUX-C-03-002: usage.completion_tokens must be a u64, got {v}")
    });
    let total = usage["total_tokens"].as_u64().unwrap_or_else(|| {
        panic!("FALSIFY-CRUX-C-03-002: usage.total_tokens must be a u64, got {v}")
    });

    assert!(
        prompt >= 1,
        "FALSIFY-CRUX-C-03-002: usage.prompt_tokens must be >= 1, got {prompt}"
    );
    assert!(
        total >= 1,
        "FALSIFY-CRUX-C-03-002: usage.total_tokens must be >= 1, got {total}"
    );
    assert_eq!(
        total,
        prompt + completion,
        "FALSIFY-CRUX-C-03-002: total_tokens ({total}) must equal \
         prompt_tokens ({prompt}) + completion_tokens ({completion})"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-03-003: stream=false yields Content-Type: application/json
// and a single JSON object (not SSE, not an array).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn falsify_crux_c_03_003_content_type_and_single_json_object() {
    let req = chat_request(serde_json::json!({
        "model": "default",
        "messages": [{"role":"user","content":"ok"}],
        "stream": false,
        "max_tokens": 8,
        "temperature": 0.01
    }));
    let resp = router().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);

    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.to_ascii_lowercase().starts_with("application/json"),
        "FALSIFY-CRUX-C-03-003: Content-Type must start with application/json for \
         stream=false, got {ct:?}"
    );

    let bytes = collect_bytes(resp).await;
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "FALSIFY-CRUX-C-03-003: body must be a single JSON value, got {} ({e})",
            String::from_utf8_lossy(&bytes)
        )
    });
    assert!(
        v.is_object(),
        "FALSIFY-CRUX-C-03-003: body must be a single JSON object (not SSE, not array), \
         got {v}"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-03-004: response.model echoes request.model.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn falsify_crux_c_03_004_model_field_echoes_request() {
    let req_model = "default";
    let req = chat_request(serde_json::json!({
        "model": req_model,
        "messages": [{"role":"user","content":"hi"}],
        "stream": false,
        "max_tokens": 4,
        "temperature": 0.01
    }));
    let resp = router().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = collect_bytes(resp).await;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).expect("FALSIFY-CRUX-C-03-004: body must be JSON");

    let resp_model = v["model"].as_str().unwrap_or("");
    assert_eq!(
        resp_model, req_model,
        "FALSIFY-CRUX-C-03-004: response.model ({resp_model:?}) must echo \
         request.model ({req_model:?})"
    );
}
