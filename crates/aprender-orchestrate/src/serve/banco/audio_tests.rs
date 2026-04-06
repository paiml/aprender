//! Audio transcription endpoint tests.

use super::handlers_audio::{TranscribeRequest, TranscribeResponse};

// ============================================================================
// Type tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_AUDIO_001_request_serde() {
    let json = serde_json::json!({
        "audio_data": "SGVsbG8=",
        "format": "wav",
        "language": "en"
    });
    let req: TranscribeRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.format, Some("wav".to_string()));
    assert_eq!(req.language, Some("en".to_string()));
}

#[test]
#[allow(non_snake_case)]
fn test_AUDIO_002_request_defaults() {
    let json = serde_json::json!({"audio_data": "dGVzdA=="});
    let req: TranscribeRequest = serde_json::from_value(json).expect("parse");
    assert!(req.format.is_none());
    assert!(req.language.is_none());
    assert!(req.translate.is_none());
}

#[test]
#[allow(non_snake_case)]
fn test_AUDIO_003_response_serializes() {
    let resp = TranscribeResponse {
        text: "Hello world".to_string(),
        language: "en".to_string(),
        duration_secs: 1.5,
        segments: vec![],
    };
    let json = serde_json::to_value(&resp).expect("serialize");
    assert_eq!(json["text"], "Hello world");
    assert!((json["duration_secs"].as_f64().unwrap() - 1.5).abs() < 0.01);
}

// ============================================================================
// Base64 decoder tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_AUDIO_004_base64_decode() {
    // "Hello" = "SGVsbG8="
    let decoded = super::handlers_audio::base64_decode("SGVsbG8=").expect("decode");
    assert_eq!(decoded, b"Hello");
}

#[test]
#[allow(non_snake_case)]
fn test_AUDIO_005_base64_empty() {
    let decoded = super::handlers_audio::base64_decode("").expect("decode");
    assert!(decoded.is_empty());
}

// ============================================================================
// Endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_AUDIO_HDL_001_transcribe_dry_run() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    // Send a small base64-encoded "audio" (just test bytes)
    let body = serde_json::json!({
        "audio_data": "SGVsbG8gV29ybGQ=",
        "format": "wav"
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/audio/transcriptions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(
        json["text"].as_str().expect("text").contains("dry-run")
            || !json["text"].as_str().expect("text").is_empty()
    );
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_AUDIO_HDL_002_list_formats() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/audio/formats").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["sample_rate"], 16000);
    let formats = json["formats"].as_array().expect("formats");
    assert!(formats.len() >= 2);
}
