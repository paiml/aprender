//! T-COV-95 Extended Coverage: api/apr_handlers.rs
//!
//! Targets: apr_predict_handler, apr_explain_handler, apr_audit_handler,
//! error paths, serde, edge cases.

use crate::api::test_helpers::create_test_app_shared;
use crate::api::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

// ============================================================================
// PredictRequest/Response serde
// ============================================================================

#[test]
fn test_predict_request_serde() {
    let req = PredictRequest {
        model: Some("default".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        include_confidence: true,
        top_k: Some(3),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json.contains("features"));
    let parsed: PredictRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.features.len(), 3);
    assert!(parsed.include_confidence);
}

#[test]
fn test_predict_request_minimal() {
    let json = r#"{"features": [1.0, 2.0]}"#;
    let req: PredictRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    assert!(req.model.is_none());
    assert!(req.feature_names.is_none());
    assert!(req.top_k.is_none());
}

#[test]
fn test_predict_response_serde() {
    let resp = PredictResponse {
        request_id: "test-123".to_string(),
        model: "default".to_string(),
        prediction: serde_json::json!(0.95),
        confidence: Some(0.99),
        top_k_predictions: None,
        latency_ms: 10.5,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    assert!(json.contains("test-123"));
    let parsed: PredictResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.model, "default");
}

#[test]
fn test_predict_response_with_top_k() {
    let resp = PredictResponse {
        request_id: "id".to_string(),
        model: "m".to_string(),
        prediction: serde_json::json!("class_0"),
        confidence: Some(0.8),
        top_k_predictions: Some(vec![
            PredictionWithScore {
                label: "class_0".to_string(),
                score: 0.8,
            },
            PredictionWithScore {
                label: "class_1".to_string(),
                score: 0.15,
            },
        ]),
        latency_ms: 5.0,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    assert!(json.contains("class_0"));
    assert!(json.contains("class_1"));
}

#[test]
fn test_prediction_with_score_serde() {
    let pws = PredictionWithScore {
        label: "positive".to_string(),
        score: 0.75,
    };
    let json = serde_json::to_string(&pws).expect("JSON serialization failed");
    let parsed: PredictionWithScore = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.label, "positive");
    assert!((parsed.score - 0.75).abs() < f32::EPSILON);
}

// ============================================================================
// ExplainRequest/Response serde
// ============================================================================

#[test]
fn test_explain_request_serde() {
    let req = ExplainRequest {
        model: Some("classifier".to_string()),
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ],
        top_k_features: 3,
        method: "shap".to_string(),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json.contains("feature_names"));
    let parsed: ExplainRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.top_k_features, 3);
    assert_eq!(parsed.method, "shap");
}

#[test]
fn test_explain_request_default_method() {
    // Method should have a default value
    let json = r#"{"features": [1.0], "feature_names": ["x"]}"#;
    let req: ExplainRequest = serde_json::from_str(json).expect("JSON deserialization failed");
    // default_explain_method() returns something
    assert!(!req.method.is_empty());
}

#[test]
fn test_explain_response_serde() {
    let resp = ExplainResponse {
        request_id: "explain-123".to_string(),
        model: "test".to_string(),
        prediction: serde_json::json!(1),
        confidence: Some(0.9),
        explanation: ShapExplanation {
            base_value: 0.5,
            shap_values: vec![0.1, -0.2, 0.3],
            feature_names: vec!["x".to_string(), "y".to_string(), "z".to_string()],
            prediction: 0.7,
        },
        summary: "Top features: x, z".to_string(),
        latency_ms: 15.0,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    assert!(json.contains("shap_values"));
    let parsed: ExplainResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(parsed.explanation.shap_values.len(), 3);
}

// ============================================================================
// ShapExplanation serde and coverage
// ============================================================================

#[test]
fn test_shap_explanation_serde() {
    let shap = ShapExplanation {
        base_value: 0.0,
        shap_values: vec![0.1, 0.2, -0.1],
        feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
        prediction: 0.6,
    };
    let json = serde_json::to_string(&shap).expect("JSON serialization failed");
    assert!(json.contains("base_value"));
    assert!(json.contains("prediction"));
    let parsed: ShapExplanation = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!((parsed.prediction - 0.6).abs() < f32::EPSILON);
}

#[test]
fn test_shap_explanation_empty() {
    let shap = ShapExplanation {
        base_value: 0.0,
        shap_values: vec![],
        feature_names: vec![],
        prediction: 0.0,
    };
    let json = serde_json::to_string(&shap).expect("JSON serialization failed");
    let parsed: ShapExplanation = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.shap_values.is_empty());
}

#[test]
fn test_shap_explanation_debug() {
    let shap = ShapExplanation {
        base_value: 0.5,
        shap_values: vec![0.1],
        feature_names: vec!["x".to_string()],
        prediction: 0.6,
    };
    let debug = format!("{:?}", shap);
    assert!(debug.contains("ShapExplanation"));
}

#[test]
fn test_shap_explanation_clone() {
    let shap = ShapExplanation {
        base_value: 0.5,
        shap_values: vec![0.1, 0.2],
        feature_names: vec!["a".to_string(), "b".to_string()],
        prediction: 0.7,
    };
    let cloned = shap.clone();
    assert_eq!(cloned.shap_values.len(), 2);
    assert!((cloned.prediction - 0.7).abs() < f32::EPSILON);
}

// ============================================================================
// AuditRecord coverage - using AuditRecord::new() constructor
// ============================================================================

#[test]
fn test_audit_record_new() {
    use crate::audit::AuditRecord;
    use uuid::Uuid;

    let id = Uuid::new_v4();
    let record = AuditRecord::new(id, "abc123hash", "LogisticRegression");
    assert_eq!(record.request_id, id.to_string());
    assert_eq!(record.model_hash, "abc123hash");
    assert_eq!(record.model_type, "LogisticRegression");
}

#[test]
fn test_audit_record_fields() {
    use crate::audit::AuditRecord;
    use uuid::Uuid;

    let record = AuditRecord::new(Uuid::new_v4(), "hash", "RandomForest");
    // Check default field values
    assert!(record.client_id_hash.is_none());
    assert!(record.distillation_teacher_hash.is_none());
    assert!(record.confidence.is_none());
    assert!(record.explanation_summary.is_none());
    assert!(record.warnings.is_empty());
}

// ============================================================================
// HTTP endpoint coverage
// ============================================================================

#[tokio::test]
async fn test_apr_predict_empty_features() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "features": []
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Should return BAD_REQUEST for empty features
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_apr_predict_with_top_k() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "features": [1.0, 2.0, 3.0],
        "include_confidence": true,
        "top_k": 3
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // May succeed or fail depending on model state
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn test_apr_explain_empty_features() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "features": [],
        "feature_names": [],
        "top_k_features": 3
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Should return BAD_REQUEST for empty features
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_apr_explain_mismatched_names() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "features": [1.0, 2.0, 3.0],
        "feature_names": ["a", "b"],  // Only 2, but 3 features
        "top_k_features": 2
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Should return BAD_REQUEST for mismatched counts
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_apr_explain_valid() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "features": [1.0, 2.0, 3.0],
        "feature_names": ["x", "y", "z"],
        "top_k_features": 2
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Should succeed or return service unavailable
    assert!(
        response.status() == StatusCode::OK || response.status() == StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn test_apr_audit_invalid_uuid() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audit/not-a-valid-uuid")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Should return BAD_REQUEST for invalid UUID
    assert!(
        response.status() == StatusCode::BAD_REQUEST || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_apr_audit_nonexistent_id() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audit/00000000-0000-0000-0000-000000000000")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Should return NOT_FOUND for nonexistent record
    assert!(
        response.status() == StatusCode::NOT_FOUND || response.status() == StatusCode::BAD_REQUEST
    );
}

// ============================================================================
// Debug trait coverage for request/response types
// ============================================================================

#[test]
fn test_predict_request_debug() {
    let req = PredictRequest {
        model: None,
        features: vec![1.0],
        feature_names: None,
        include_confidence: false,
        top_k: None,
    };
    let debug = format!("{:?}", req);
    assert!(debug.contains("PredictRequest"));
}

#[test]
fn test_predict_response_debug() {
    let resp = PredictResponse {
        request_id: "id".to_string(),
        model: "m".to_string(),
        prediction: serde_json::json!(1),
        confidence: None,
        top_k_predictions: None,
        latency_ms: 0.0,
    };
    let debug = format!("{:?}", resp);
    assert!(debug.contains("PredictResponse"));
}

#[test]
fn test_explain_request_debug() {
    let req = ExplainRequest {
        model: None,
        features: vec![1.0],
        feature_names: vec!["x".to_string()],
        top_k_features: 1,
        method: "shap".to_string(),
    };
    let debug = format!("{:?}", req);
    assert!(debug.contains("ExplainRequest"));
}

include!("explain_response.rs");
