
#[test]
fn test_explain_response_debug() {
    let resp = ExplainResponse {
        request_id: "id".to_string(),
        model: "m".to_string(),
        prediction: serde_json::json!(0),
        confidence: None,
        explanation: ShapExplanation {
            base_value: 0.0,
            shap_values: vec![],
            feature_names: vec![],
            prediction: 0.0,
        },
        summary: "summary".to_string(),
        latency_ms: 0.0,
    };
    let debug = format!("{:?}", resp);
    assert!(debug.contains("ExplainResponse"));
}

#[test]
fn test_prediction_with_score_debug() {
    let pws = PredictionWithScore {
        label: "label".to_string(),
        score: 0.5,
    };
    let debug = format!("{:?}", pws);
    assert!(debug.contains("PredictionWithScore"));
}

// ============================================================================
// Clone trait coverage
// ============================================================================

#[test]
fn test_predict_request_clone() {
    let req = PredictRequest {
        model: Some("m".to_string()),
        features: vec![1.0, 2.0],
        feature_names: Some(vec!["a".to_string(), "b".to_string()]),
        include_confidence: true,
        top_k: Some(3),
    };
    let cloned = req.clone();
    assert_eq!(cloned.features.len(), 2);
    assert!(cloned.include_confidence);
}

#[test]
fn test_predict_response_clone() {
    let resp = PredictResponse {
        request_id: "id".to_string(),
        model: "m".to_string(),
        prediction: serde_json::json!(1),
        confidence: Some(0.9),
        top_k_predictions: None,
        latency_ms: 1.0,
    };
    let cloned = resp.clone();
    assert_eq!(cloned.request_id, "id");
}

#[test]
fn test_explain_request_clone() {
    let req = ExplainRequest {
        model: None,
        features: vec![1.0],
        feature_names: vec!["x".to_string()],
        top_k_features: 2,
        method: "lime".to_string(),
    };
    let cloned = req.clone();
    assert_eq!(cloned.top_k_features, 2);
    assert_eq!(cloned.method, "lime");
}

#[test]
fn test_explain_response_clone() {
    let resp = ExplainResponse {
        request_id: "id".to_string(),
        model: "m".to_string(),
        prediction: serde_json::json!(0),
        confidence: None,
        explanation: ShapExplanation {
            base_value: 0.0,
            shap_values: vec![],
            feature_names: vec![],
            prediction: 0.0,
        },
        summary: "s".to_string(),
        latency_ms: 0.0,
    };
    let cloned = resp.clone();
    assert_eq!(cloned.summary, "s");
}

#[test]
fn test_prediction_with_score_clone() {
    let pws = PredictionWithScore {
        label: "pos".to_string(),
        score: 0.75,
    };
    let cloned = pws.clone();
    assert_eq!(cloned.label, "pos");
}

// ============================================================================
// Additional ExplainRequest edge cases
// ============================================================================

#[test]
fn test_explain_request_attention_method() {
    let req = ExplainRequest {
        model: Some("transformer".to_string()),
        features: vec![1.0, 2.0],
        feature_names: vec!["x".to_string(), "y".to_string()],
        top_k_features: 2,
        method: "attention".to_string(),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json.contains("attention"));
}

#[test]
fn test_explain_request_lime_method() {
    let req = ExplainRequest {
        model: None,
        features: vec![1.0],
        feature_names: vec!["feature".to_string()],
        top_k_features: 1,
        method: "lime".to_string(),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json.contains("lime"));
}

// ============================================================================
// PredictRequest with feature_names
// ============================================================================

#[test]
fn test_predict_request_with_feature_names() {
    let req = PredictRequest {
        model: Some("model".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: Some(vec!["x".to_string(), "y".to_string(), "z".to_string()]),
        include_confidence: true,
        top_k: None,
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    assert!(json.contains("feature_names"));
    let parsed: PredictRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.feature_names.is_some());
    assert_eq!(parsed.feature_names.expect("test value should be present").len(), 3);
}

#[test]
fn test_predict_request_without_feature_names() {
    let req = PredictRequest {
        model: None,
        features: vec![1.0, 2.0],
        feature_names: None,
        include_confidence: false,
        top_k: Some(5),
    };
    let json = serde_json::to_string(&req).expect("JSON serialization failed");
    let parsed: PredictRequest = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.feature_names.is_none());
}

// ============================================================================
// Response edge cases
// ============================================================================

#[test]
fn test_predict_response_null_confidence() {
    let resp = PredictResponse {
        request_id: "id".to_string(),
        model: "m".to_string(),
        prediction: serde_json::json!(null),
        confidence: None,
        top_k_predictions: None,
        latency_ms: 0.0,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: PredictResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.confidence.is_none());
}

#[test]
fn test_explain_response_empty_explanation() {
    let resp = ExplainResponse {
        request_id: "id".to_string(),
        model: "m".to_string(),
        prediction: serde_json::json!(0),
        confidence: Some(1.0),
        explanation: ShapExplanation {
            base_value: 0.0,
            shap_values: vec![],
            feature_names: vec![],
            prediction: 0.0,
        },
        summary: "No features".to_string(),
        latency_ms: 1.0,
    };
    let json = serde_json::to_string(&resp).expect("JSON serialization failed");
    let parsed: ExplainResponse = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert!(parsed.explanation.shap_values.is_empty());
}

// ============================================================================
// Additional HTTP endpoint tests
// ============================================================================

#[tokio::test]
async fn test_apr_predict_with_model_name() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "custom_model",
        "features": [1.0, 2.0, 3.0],
        "include_confidence": true
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
    // Model may not exist, expect error or success
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_apr_explain_with_model_name() {
    let app = create_test_app_shared();
    let body = serde_json::json!({
        "model": "custom_model",
        "features": [1.0, 2.0],
        "feature_names": ["a", "b"],
        "top_k_features": 2,
        "method": "shap"
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
    // Model may not exist
    assert!(
        response.status() == StatusCode::OK
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn test_apr_predict_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/predict")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Should return BAD_REQUEST for invalid JSON
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_apr_explain_invalid_json() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/explain")
                .header("content-type", "application/json")
                .body(Body::from("{broken json"))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    // Should return BAD_REQUEST for invalid JSON
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}
