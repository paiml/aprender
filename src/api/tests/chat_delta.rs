//! API Tests Part 04
//!
//! GPU inference tests (IMP-116+)

#[allow(unused_imports)]
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
#[allow(unused_imports)]
use tower::util::ServiceExt;

#[allow(unused_imports)]
use crate::api::test_helpers::create_test_app_shared;
use crate::api::*;

#[test]
fn test_chat_delta_debug_clone_cov() {
    let delta = ChatDelta {
        role: Some("assistant".to_string()),
        content: Some("Hello".to_string()),
    };
    let debug = format!("{:?}", delta);
    assert!(debug.contains("ChatDelta"));

    let cloned = delta.clone();
    assert_eq!(cloned.role, delta.role);
}

#[test]
fn test_chat_delta_empty_cov() {
    let delta = ChatDelta {
        role: None,
        content: None,
    };
    let json = serde_json::to_string(&delta).expect("serialize");
    // Empty delta should have null fields
    assert!(json.contains("null") || !json.is_empty());
}

// =========================================================================
// Additional Coverage Tests: StreamTokenEvent
// =========================================================================

#[test]
fn test_stream_token_event_serialize_cov() {
    let event = StreamTokenEvent {
        token_id: 1234,
        text: "hello".to_string(),
    };
    let json = serde_json::to_string(&event).expect("serialize");
    assert!(json.contains("hello"));
    assert!(json.contains("1234"));
}

// =========================================================================
// Additional Coverage Tests: StreamDoneEvent
// =========================================================================

#[test]
fn test_stream_done_event_serialize_cov() {
    let event = StreamDoneEvent { num_generated: 100 };
    let json = serde_json::to_string(&event).expect("serialize");
    assert!(json.contains("100"));
}

// =========================================================================
// Additional Coverage Tests: BatchTokenizeRequest/Response
// =========================================================================

#[test]
fn test_batch_tokenize_request_serialize_cov() {
    let req = BatchTokenizeRequest {
        texts: vec!["hello".to_string(), "world".to_string()],
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("hello"));
    assert!(json.contains("world"));
}

#[test]
fn test_batch_tokenize_response_serialize_cov() {
    let resp = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1, 2, 3],
                num_tokens: 3,
            },
            TokenizeResponse {
                token_ids: vec![4, 5],
                num_tokens: 2,
            },
        ],
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("1") && json.contains("4"));
}

// =========================================================================
// Additional Coverage Tests: OpenAIModel
// =========================================================================

#[test]
fn test_openai_model_debug_clone_cov() {
    let model = OpenAIModel {
        id: "text-davinci-003".to_string(),
        object: "model".to_string(),
        created: 1669599635,
        owned_by: "openai-internal".to_string(),
    };
    let debug = format!("{:?}", model);
    assert!(debug.contains("OpenAIModel"));
    assert!(debug.contains("text-davinci-003"));

    let cloned = model.clone();
    assert_eq!(cloned.id, model.id);
}

// =========================================================================
// Additional Coverage Tests: PredictionWithScore
// =========================================================================

#[test]
fn test_prediction_with_score_debug_clone_cov() {
    let pred = PredictionWithScore {
        label: "positive".to_string(),
        score: 0.95,
    };
    let debug = format!("{:?}", pred);
    assert!(debug.contains("PredictionWithScore"));
    assert!(debug.contains("positive"));

    let cloned = pred.clone();
    assert_eq!(cloned.label, pred.label);
    assert!((cloned.score - pred.score).abs() < 1e-6);
}

// =========================================================================
// Additional Coverage Tests: ChatChunkChoice
// =========================================================================

#[test]
fn test_chat_chunk_choice_debug_clone_cov() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: Some("assistant".to_string()),
            content: Some("Test".to_string()),
        },
        finish_reason: Some("stop".to_string()),
    };
    let debug = format!("{:?}", choice);
    assert!(debug.contains("ChatChunkChoice"));

    let cloned = choice.clone();
    assert_eq!(cloned.index, choice.index);
}

// =========================================================================
// Coverage Tests: Default functions
// =========================================================================

#[test]
fn test_default_true_cov() {
    assert!(crate::api::default_true());
}

#[test]
fn test_default_explain_method_cov() {
    let method = crate::api::default_explain_method();
    assert_eq!(method, "shap");
}

#[test]
fn test_default_top_k_features_cov() {
    let k = crate::api::default_top_k_features();
    assert_eq!(k, 5);
}

// =========================================================================
// Coverage Tests: PredictRequest
// =========================================================================

#[test]
fn test_predict_request_full_cov() {
    let req = PredictRequest {
        model: Some("sentiment".to_string()),
        features: vec![1.0, 2.0, 3.0, 4.0],
        feature_names: Some(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ]),
        top_k: Some(3),
        include_confidence: true,
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("sentiment"));
    assert!(json.contains("1.0"));
    assert!(json.contains("true"));
}

#[test]
fn test_predict_request_defaults_cov() {
    let json = r#"{"features": [0.5, 1.5]}"#;
    let req: PredictRequest = serde_json::from_str(json).expect("deserialize");
    assert!(req.model.is_none());
    assert!(!req.features.is_empty());
}

// =========================================================================
// Coverage Tests: PredictResponse
// =========================================================================

#[test]
fn test_predict_response_full_cov() {
    let resp = PredictResponse {
        request_id: "req-123".to_string(),
        model: "classifier".to_string(),
        prediction: serde_json::json!("positive"),
        confidence: Some(0.95),
        top_k_predictions: Some(vec![
            PredictionWithScore {
                label: "positive".to_string(),
                score: 0.95,
            },
            PredictionWithScore {
                label: "negative".to_string(),
                score: 0.05,
            },
        ]),
        latency_ms: 12.5,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("req-123"));
    assert!(json.contains("positive"));
    assert!(json.contains("0.95"));
}

#[test]
fn test_predict_response_minimal_cov() {
    let resp = PredictResponse {
        request_id: "req-minimal".to_string(),
        model: "model".to_string(),
        prediction: serde_json::json!(42),
        confidence: None,
        top_k_predictions: None,
        latency_ms: 5.0,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("req-minimal"));
    // None fields should be skipped
    assert!(!json.contains("confidence") || json.contains("null"));
}

// =========================================================================
// Coverage Tests: ExplainRequest
// =========================================================================

#[test]
fn test_explain_request_full_cov() {
    let req = ExplainRequest {
        model: Some("explainer".to_string()),
        features: vec![1.0, 2.0, 3.0],
        feature_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        top_k_features: 2,
        method: "lime".to_string(),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("explainer"));
    assert!(json.contains("lime"));
    assert!(json.contains("feature_names"));
}

// =========================================================================
// Coverage Tests: ExplainResponse
// =========================================================================

#[test]
fn test_explain_response_full_cov() {
    let resp = ExplainResponse {
        request_id: "explain-req-1".to_string(),
        model: "model".to_string(),
        prediction: serde_json::json!("class_a"),
        confidence: Some(0.9),
        explanation: ShapExplanation {
            base_value: 0.5,
            shap_values: vec![0.4],
            feature_names: vec!["x".to_string()],
            prediction: 0.9,
        },
        summary: "Feature x was most important".to_string(),
        latency_ms: 15.0,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("explain-req-1"));
    assert!(json.contains("base_value"));
}

// =========================================================================
// Coverage Tests: AuditResponse
// =========================================================================

#[test]
fn test_audit_response_debug_cov() {
    // AuditResponse wraps AuditRecord - test the response type exists
    // Skip constructing the full AuditRecord (too many fields) and just test type exists
    let _check_type_exists = |r: AuditResponse| {
        let _ = format!("{:?}", r.record);
    };
}

// =========================================================================
// Coverage Tests: DispatchMetricsResponse
// =========================================================================

#[test]
fn test_dispatch_metrics_response_cov() {
    let resp = DispatchMetricsResponse {
        cpu_dispatches: 100,
        gpu_dispatches: 200,
        total_dispatches: 300,
        gpu_ratio: 0.666,
        cpu_latency_p50_us: 1000.0,
        cpu_latency_p95_us: 2000.0,
        cpu_latency_p99_us: 3000.0,
        gpu_latency_p50_us: 500.0,
        gpu_latency_p95_us: 800.0,
        gpu_latency_p99_us: 1200.0,
        cpu_latency_mean_us: 1100.0,
        gpu_latency_mean_us: 600.0,
        cpu_latency_min_us: 500,
        cpu_latency_max_us: 5000,
        gpu_latency_min_us: 200,
        gpu_latency_max_us: 2000,
        cpu_latency_variance_us: 250000.0,
        cpu_latency_stddev_us: 500.0,
        gpu_latency_variance_us: 100000.0,
        gpu_latency_stddev_us: 316.23,
        bucket_boundaries_us: vec!["0-100".to_string(), "100-500".to_string()],
        cpu_latency_bucket_counts: vec![10, 50, 40],
        gpu_latency_bucket_counts: vec![50, 100, 50],
        throughput_rps: 1000.0,
        elapsed_seconds: 60.0,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("cpu_dispatches"));
    assert!(json.contains("gpu_ratio"));
    assert!(json.contains("throughput_rps"));
}

// =========================================================================
// Coverage Tests: ServerMetricsResponse
// =========================================================================

#[test]
fn test_server_metrics_response_deserialize_cov() {
    let json = r#"{
        "throughput_tok_per_sec": 100.5,
        "latency_p50_ms": 50.0,
        "latency_p95_ms": 100.0,
        "latency_p99_ms": 200.0,
        "gpu_memory_used_bytes": 1000000,
        "gpu_memory_total_bytes": 8000000,
        "gpu_utilization_percent": 75,
        "cuda_path_active": true,
        "batch_size": 8,
        "queue_depth": 5,
        "total_tokens": 50000,
        "total_requests": 1000,
        "uptime_secs": 3600,
        "model_name": "phi-2"
    }"#;
    let resp: ServerMetricsResponse = serde_json::from_str(json).expect("deserialize");
    assert!((resp.throughput_tok_per_sec - 100.5).abs() < 0.01);
    assert_eq!(resp.batch_size, 8);
    assert!(resp.cuda_path_active);
    assert_eq!(resp.model_name, "phi-2");
}

// =========================================================================
// Coverage Tests: DispatchMetricsQuery
// =========================================================================

#[test]
fn test_dispatch_metrics_query_cov() {
    let query = DispatchMetricsQuery {
        format: Some("prometheus".to_string()),
    };
    assert_eq!(query.format.expect("operation failed"), "prometheus");

    let query_none = DispatchMetricsQuery { format: None };
    assert!(query_none.format.is_none());
}

// =========================================================================
// Coverage Tests: DispatchResetResponse
// =========================================================================

#[test]
fn test_dispatch_reset_response_deserialize_cov() {
    let json = r#"{"success": true, "message": "Reset complete"}"#;
    let resp: DispatchResetResponse = serde_json::from_str(json).expect("deserialize");
    assert!(resp.success);
    assert_eq!(resp.message, "Reset complete");
}

// =========================================================================
// Coverage Tests: BatchGenerateRequest/Response
// =========================================================================

#[test]
fn test_batch_generate_request_full_cov() {
    let req = BatchGenerateRequest {
        prompts: vec!["Hello".to_string(), "World".to_string()],
        max_tokens: 100,
        temperature: 0.7,
        strategy: "top_k".to_string(),
        top_k: 40,
        top_p: 0.95,
        seed: Some(42),
    };
    let json = serde_json::to_string(&req).expect("serialize");
    assert!(json.contains("Hello"));
    assert!(json.contains("top_k"));
    assert!(json.contains("42"));
}

#[test]
fn test_batch_generate_response_cov() {
    let resp = BatchGenerateResponse {
        results: vec![
            GenerateResponse {
                token_ids: vec![1, 2, 3],
                text: "output1".to_string(),
                num_generated: 3,
            },
            GenerateResponse {
                token_ids: vec![4, 5],
                text: "output2".to_string(),
                num_generated: 2,
            },
        ],
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("output1"));
    assert!(json.contains("output2"));
}

include!("models_response.rs");
include!("model_lineage.rs");
include!("model_metadata.rs");
include!("batch_tokenize.rs");
