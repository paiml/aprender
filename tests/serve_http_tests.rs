//! HTTP Integration Tests for Aprender Model Serving
//!
//! Uses the canonical Rust web testing approach: `tower::ServiceExt::oneshot()`
//!
//! ## Why tower::ServiceExt?
//!
//! This is the idiomatic way to test axum applications:
//! - No actual server startup (fast)
//! - Direct router testing (no network overhead)
//! - Full request/response cycle
//! - Same code path as production
//!
//! ## References
//!
//! - axum testing docs: https://docs.rs/axum/latest/axum/#testing
//! - tower::ServiceExt: https://docs.rs/tower/latest/tower/trait.ServiceExt.html

#![cfg(feature = "aprender-serve")]

use aprender::{classification::LogisticRegression, primitives::Matrix};
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use realizar::serve::{
    create_serve_router, BatchPredictRequest, PredictInstance, PredictOptions, PredictRequest,
    PredictResponse, ServeState,
};
use tower::ServiceExt;

// ============================================================================
// Test Utilities
// ============================================================================

/// Create a trained LogisticRegression model for testing
fn create_test_model() -> LogisticRegression {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("Valid 4x2 matrix");
    let y = vec![0, 0, 1, 1];

    let mut model = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(100);

    model.fit(&x, &y).expect("Training should succeed");
    model
}

/// Create ServeState with a loaded model
fn create_test_state() -> ServeState {
    let model = create_test_model();
    ServeState::with_logistic_regression(model, "test-v1".to_string(), 2)
}

/// Create ServeState without a model (for error testing)
fn create_empty_state() -> ServeState {
    ServeState::new("empty".to_string(), "v0".to_string())
}

/// Helper to extract body as string
async fn body_to_string(body: Body) -> String {
    let bytes = body.collect().await.expect("test").to_bytes();
    String::from_utf8(bytes.to_vec()).expect("test")
}

/// Helper to extract body as JSON
async fn body_to_json<T: serde::de::DeserializeOwned>(body: Body) -> T {
    let bytes = body.collect().await.expect("test").to_bytes();
    serde_json::from_slice(&bytes).expect("test")
}

// ============================================================================
// Health Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_health_endpoint_returns_200() {
    let app = create_serve_router(create_empty_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_health_endpoint_returns_json() {
    let app = create_serve_router(create_empty_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("healthy"));
    assert!(body.contains("version"));
}

// ============================================================================
// Ready Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_ready_endpoint_with_model() {
    let app = create_serve_router(create_test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("\"ready\":true"));
    assert!(body.contains("\"model_loaded\":true"));
}

#[tokio::test]
async fn test_ready_endpoint_without_model() {
    let app = create_serve_router(create_empty_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("\"ready\":false"));
    assert!(body.contains("\"model_loaded\":false"));
}

// ============================================================================
// Predict Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_predict_endpoint_success() {
    let app = create_serve_router(create_test_state());

    let request = PredictRequest {
        model_id: None,
        features: vec![0.9, 0.9], // Should predict class 1
        options: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let pred: PredictResponse = body_to_json(response.into_body()).await;
    assert_eq!(pred.prediction, 1.0);
    assert!(pred.latency_ms < 100.0); // Should be very fast
    assert_eq!(pred.model_version, "test-v1");
}

#[tokio::test]
async fn test_predict_endpoint_with_probabilities() {
    let app = create_serve_router(create_test_state());

    let request = PredictRequest {
        model_id: None,
        features: vec![0.9, 0.9],
        options: Some(PredictOptions {
            return_probabilities: true,
            top_k: None,
        }),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let pred: PredictResponse = body_to_json(response.into_body()).await;
    assert!(pred.probabilities.is_some());
    let probs = pred.probabilities.expect("test");
    assert_eq!(probs.len(), 2);
    // Probabilities should sum to ~1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_predict_endpoint_class_0() {
    let app = create_serve_router(create_test_state());

    let request = PredictRequest {
        model_id: None,
        features: vec![0.0, 0.0], // Should predict class 0
        options: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let pred: PredictResponse = body_to_json(response.into_body()).await;
    assert_eq!(pred.prediction, 0.0);
}

#[tokio::test]
async fn test_predict_endpoint_no_model() {
    let app = create_serve_router(create_empty_state());

    let request = PredictRequest {
        model_id: None,
        features: vec![0.5, 0.5],
        options: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("No model loaded"));
}

#[tokio::test]
async fn test_predict_endpoint_wrong_dimensions() {
    let app = create_serve_router(create_test_state());

    let request = PredictRequest {
        model_id: None,
        features: vec![0.5, 0.5, 0.5], // 3 features, expected 2
        options: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("Invalid input dimension"));
}

#[tokio::test]
async fn test_predict_endpoint_invalid_json() {
    let app = create_serve_router(create_test_state());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .expect("test"),
        )
        .await
        .expect("test");

    // axum returns 400 Bad Request for JSON parse errors
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ============================================================================
// Batch Predict Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_batch_predict_endpoint_success() {
    let app = create_serve_router(create_test_state());

    let request = BatchPredictRequest {
        model_id: None,
        instances: vec![
            PredictInstance {
                features: vec![0.0, 0.0],
            },
            PredictInstance {
                features: vec![1.0, 1.0],
            },
        ],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("predictions"));
    assert!(body.contains("total_latency_ms"));
}

#[tokio::test]
async fn test_batch_predict_endpoint_empty_batch() {
    let app = create_serve_router(create_test_state());

    let request = BatchPredictRequest {
        model_id: None,
        instances: vec![],
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("Empty batch"));
}

// ============================================================================
// Models Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_models_endpoint_with_model() {
    let app = create_serve_router(create_test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/models")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("LogisticRegression"));
    assert!(body.contains("\"loaded\":true"));
}

#[tokio::test]
async fn test_models_endpoint_without_model() {
    let app = create_serve_router(create_empty_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/models")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("\"models\":[]"));
}

// ============================================================================
// Metrics Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_metrics_endpoint_prometheus_format() {
    let app = create_serve_router(create_test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_to_string(response.into_body()).await;
    // Prometheus format validation
    assert!(body.contains("# HELP"));
    assert!(body.contains("# TYPE"));
    assert!(body.contains("requests_total"));
    assert!(body.contains("model_loaded"));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[tokio::test]
async fn test_404_for_unknown_route() {
    let app = create_serve_router(create_test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/unknown/route")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_method_not_allowed() {
    let app = create_serve_router(create_test_state());

    // GET on POST-only endpoint
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/predict")
                .body(Body::empty())
                .expect("test"),
        )
        .await
        .expect("test");

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ============================================================================
// Performance Tests (Latency Validation)
// ============================================================================

#[tokio::test]
async fn test_predict_latency_is_submillisecond() {
    let app = create_serve_router(create_test_state());

    let request = PredictRequest {
        model_id: None,
        features: vec![0.5, 0.5],
        options: None,
    };

    let start = std::time::Instant::now();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).expect("test")))
                .expect("test"),
        )
        .await
        .expect("test");

    let elapsed = start.elapsed();

    assert_eq!(response.status(), StatusCode::OK);
    // Should complete in under 10ms (typically <1ms)
    assert!(
        elapsed.as_millis() < 10,
        "Prediction took {}ms, expected <10ms",
        elapsed.as_millis()
    );
}
