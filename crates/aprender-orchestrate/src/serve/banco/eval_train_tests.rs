//! Eval + training store unit tests.

use super::training::{TrainingConfig, TrainingMethod, TrainingStatus};

// ============================================================================
// EvalStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_EVAL_001_record_and_get() {
    let store = super::eval::EvalStore::new();
    let result = super::eval::EvalResult {
        eval_id: "eval-test".to_string(),
        model: "test-model".to_string(),
        metric: "perplexity".to_string(),
        value: 8.42,
        tokens_evaluated: 1000,
        duration_secs: 1.5,
        status: super::eval::EvalStatus::Complete,
    };
    store.record(result.clone());
    let retrieved = store.get("eval-test").expect("should exist");
    assert!((retrieved.value - 8.42).abs() < f64::EPSILON);
}

#[test]
#[allow(non_snake_case)]
fn test_EVAL_002_list_runs() {
    let store = super::eval::EvalStore::new();
    for i in 0..3 {
        store.record(super::eval::EvalResult {
            eval_id: format!("eval-{i}"),
            model: "m".to_string(),
            metric: "ppl".to_string(),
            value: i as f64,
            tokens_evaluated: 0,
            duration_secs: 0.0,
            status: super::eval::EvalStatus::Complete,
        });
    }
    assert_eq!(store.list().len(), 3);
}

// ============================================================================
// TrainingStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_001_start_run() {
    let store = super::training::TrainingStore::new();
    let run = store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    assert!(run.id.starts_with("run-"));
    assert_eq!(run.method, TrainingMethod::Lora);
    assert_eq!(run.status, TrainingStatus::Queued);
    assert_eq!(run.config.lora_r, 16);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_002_list_runs() {
    let store = super::training::TrainingStore::new();
    store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    store.start("ds-2", TrainingMethod::Qlora, TrainingConfig::default());
    assert_eq!(store.list().len(), 2);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_003_stop_run() {
    let store = super::training::TrainingStore::new();
    let run = store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    store.stop(&run.id).expect("stop");
    assert_eq!(store.get(&run.id).expect("get").status, TrainingStatus::Stopped);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_004_delete_run() {
    let store = super::training::TrainingStore::new();
    let run = store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    store.delete(&run.id).expect("delete");
    assert!(store.get(&run.id).is_none());
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_005_config_defaults() {
    let config = TrainingConfig::default();
    assert_eq!(config.lora_r, 16);
    assert_eq!(config.lora_alpha, 32);
    assert!((config.learning_rate - 2e-4).abs() < 1e-6);
    assert_eq!(config.epochs, 3);
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.max_seq_length, 2048);
}

// ============================================================================
// Eval endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_EVAL_HDL_001_perplexity_no_model() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({"text": "Hello, world!"});
    let response = app
        .oneshot(
            Request::post("/api/v1/eval/perplexity")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["status"], "no_model");
    assert!(json["eval_id"].as_str().expect("id").starts_with("eval-"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_EVAL_HDL_002_list_runs() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/eval/runs").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

// ============================================================================
// Training endpoint tests (basic — rest in training_engine_tests.rs)
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_001_start_training() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "dataset_id": "ds-test",
        "method": "lora",
        "config": {"lora_r": 8, "epochs": 1}
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/train/start")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("run-"));
    assert_eq!(json["method"], "lora");
    assert_eq!(json["status"], "complete");
    assert!(!json["metrics"].as_array().expect("metrics").is_empty());
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_002_list_runs() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.training.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/train/runs").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["runs"].as_array().expect("runs").len(), 1);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_003_stop_run() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let run = state.training.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(
            Request::post(&format!("/api/v1/train/runs/{}/stop", run.id))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}
