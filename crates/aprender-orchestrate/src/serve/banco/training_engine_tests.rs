//! Training presets, engine, and endpoint tests (split from eval_train_tests.rs).

use super::training::{
    ExportFormat, OptimizerType, SchedulerType, TrainingConfig, TrainingMethod, TrainingPreset,
    TrainingStatus,
};

// ============================================================================
// Training presets unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_006_preset_quick_lora() {
    let (method, config) = TrainingPreset::QuickLora.expand();
    assert_eq!(method, TrainingMethod::Lora);
    assert_eq!(config.lora_r, 8);
    assert_eq!(config.epochs, 1);
    assert_eq!(config.target_modules, vec!["q_proj", "v_proj"]);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_007_preset_standard_lora() {
    let (method, config) = TrainingPreset::StandardLora.expand();
    assert_eq!(method, TrainingMethod::Lora);
    assert_eq!(config.lora_r, 16);
    assert_eq!(config.epochs, 3);
    assert_eq!(config.target_modules.len(), 4);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_008_preset_qlora_low_vram() {
    let (method, config) = TrainingPreset::QloraLowVram.expand();
    assert_eq!(method, TrainingMethod::Qlora);
    assert_eq!(config.batch_size, 2);
    assert_eq!(config.gradient_accumulation_steps, 8);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_009_preset_deep_lora() {
    let (method, config) = TrainingPreset::DeepLora.expand();
    assert_eq!(method, TrainingMethod::Lora);
    assert_eq!(config.lora_r, 32);
    assert_eq!(config.epochs, 5);
    assert_eq!(config.target_modules, vec!["all_linear"]);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_010_preset_full_finetune() {
    let (method, config) = TrainingPreset::FullFinetune.expand();
    assert_eq!(method, TrainingMethod::FullFinetune);
    assert_eq!(config.lora_r, 0);
    assert!((config.learning_rate - 5e-5).abs() < 1e-8);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_011_all_presets() {
    let presets = TrainingPreset::all();
    assert_eq!(presets.len(), 5);
    for p in &presets {
        let (_method, _config) = p.expand();
    }
}

// ============================================================================
// Training engine unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_012_run_lora_training_produces_metrics() {
    let config = TrainingConfig { epochs: 2, batch_size: 4, ..TrainingConfig::default() };
    let data: Vec<Vec<f32>> = vec![vec![0.0; 64]; 20];
    let metrics = super::training::run_lora_training(&config, &data, 32000);
    assert!(!metrics.is_empty());
    let first_loss = metrics.first().expect("first").loss;
    let last_loss = metrics.last().expect("last").loss;
    assert!(last_loss < first_loss, "loss should decrease: {first_loss} -> {last_loss}");
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_013_metrics_have_decreasing_loss() {
    let config = TrainingConfig::default();
    let data: Vec<Vec<f32>> = vec![vec![0.0; 64]; 100];
    let metrics = super::training::run_lora_training(&config, &data, 32000);
    for w in metrics.windows(2) {
        assert!(w[1].loss <= w[0].loss, "loss should be monotonically decreasing");
    }
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_014_push_metric() {
    let store = super::training::TrainingStore::new();
    let run = store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    store.push_metric(
        &run.id,
        super::training::TrainingMetric {
            step: 0,
            loss: 2.5,
            learning_rate: 2e-4,
            grad_norm: Some(1.0),
            tokens_per_sec: Some(1200),
            eta_secs: Some(3600),
        },
    );
    let updated = store.get(&run.id).expect("get");
    assert_eq!(updated.metrics.len(), 1);
    assert!((updated.metrics[0].loss - 2.5).abs() < f32::EPSILON);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_015_set_status() {
    let store = super::training::TrainingStore::new();
    let run = store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    store.set_status(&run.id, TrainingStatus::Running);
    assert_eq!(store.get(&run.id).expect("get").status, TrainingStatus::Running);
    store.set_status(&run.id, TrainingStatus::Complete);
    assert_eq!(store.get(&run.id).expect("get").status, TrainingStatus::Complete);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_016_fail_run() {
    let store = super::training::TrainingStore::new();
    let run = store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    store.fail(&run.id, "OOM: not enough VRAM");
    let updated = store.get(&run.id).expect("get");
    assert_eq!(updated.status, TrainingStatus::Failed);
    assert_eq!(updated.error.as_deref(), Some("OOM: not enough VRAM"));
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_017_export_path() {
    let store = super::training::TrainingStore::new();
    let run = store.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    store.set_export_path(&run.id, "~/.banco/exports/adapter.safetensors");
    let updated = store.get(&run.id).expect("get");
    assert_eq!(updated.export_path.as_deref(), Some("~/.banco/exports/adapter.safetensors"));
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_018_export_format_serde() {
    let json = serde_json::json!({"format": "gguf", "merge": true});
    let req: super::training::ExportRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.format, ExportFormat::Gguf);
    assert!(req.merge);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_019_preset_serde() {
    let json = serde_json::json!("quick-lora");
    let preset: TrainingPreset = serde_json::from_value(json).expect("parse");
    assert_eq!(preset, TrainingPreset::QuickLora);

    let json = serde_json::json!("qlora-low-vram");
    let preset: TrainingPreset = serde_json::from_value(json).expect("parse");
    assert_eq!(preset, TrainingPreset::QloraLowVram);
}

#[test]
#[allow(non_snake_case)]
fn test_TRAIN_020_enhanced_config_defaults() {
    let config = TrainingConfig::default();
    assert_eq!(config.optimizer, OptimizerType::AdamW);
    assert_eq!(config.scheduler, SchedulerType::Cosine);
    assert_eq!(config.warmup_steps, 100);
    assert_eq!(config.gradient_accumulation_steps, 4);
    assert!((config.max_grad_norm - 1.0).abs() < f64::EPSILON);
    assert_eq!(config.target_modules.len(), 4);
}

// ============================================================================
// Training endpoint tests (handlers)
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_004_start_with_preset() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "dataset_id": "ds-test",
        "preset": "quick-lora"
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
    assert_eq!(json["method"], "lora");
    assert_eq!(json["config"]["lora_r"], 8);
    assert_eq!(json["config"]["epochs"], 1);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_005_list_presets() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/train/presets").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    let presets = json["presets"].as_array().expect("presets");
    assert_eq!(presets.len(), 5);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_006_metrics_sse() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let config = TrainingConfig { epochs: 1, batch_size: 4, ..TrainingConfig::default() };
    let run = state.training.start("ds-test", TrainingMethod::Lora, config.clone());
    let data: Vec<Vec<f32>> = vec![vec![0.0; 64]; 20];
    let metrics = super::training::run_lora_training(&config, &data, 32000);
    for m in &metrics {
        state.training.push_metric(&run.id, m.clone());
    }
    state.training.set_status(&run.id, TrainingStatus::Complete);

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(
            Request::get(&format!("/api/v1/train/runs/{}/metrics", run.id))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let ct = response.headers().get("content-type").expect("ct").to_str().expect("str");
    assert!(ct.contains("text/event-stream"), "expected SSE content-type, got: {ct}");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_007_export_complete_run() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let run = state.training.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());
    state.training.set_status(&run.id, TrainingStatus::Complete);

    let app = super::router::create_banco_router(state);
    let body = serde_json::json!({"format": "safetensors", "merge": false});
    let response = app
        .oneshot(
            Request::post(&format!("/api/v1/train/runs/{}/export", run.id))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["format"], "safetensors");
    assert!(!json["merged"].as_bool().expect("merged"));
    assert!(json["path"].as_str().expect("path").contains("adapter"));
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_008_export_not_complete() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    let run = state.training.start("ds-1", TrainingMethod::Lora, TrainingConfig::default());

    let app = super::router::create_banco_router(state);
    let body = serde_json::json!({"format": "gguf", "merge": true});
    let response = app
        .oneshot(
            Request::post(&format!("/api/v1/train/runs/{}/export", run.id))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_TRAIN_HDL_009_metrics_not_found() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(
            Request::get("/api/v1/train/runs/nonexistent/metrics")
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
}
