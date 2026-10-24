//! #5698: `apr_predict_handler` branch coverage, called directly (no router).

use axum::{extract::State, http::StatusCode, Json};
use std::sync::Arc;

use super::super::{apr_predict_handler, AppState, PredictRequest};
use crate::apr::{AprModel, HEADER_SIZE, MAGIC};

/// Minimal APR v2 file holding one F32 tensor called `name`.
fn apr_with_tensor(name: &str, values: &[f32]) -> AprModel {
    let metadata = r#"{"model_type":"test","name":"predict-5698"}"#;
    // binary tensor index: name_len u16, name, dtype u8 (0 = F32), ndim u8, dims u64, offset u64, size u64
    let mut index = Vec::new();
    index.extend_from_slice(&(name.len() as u16).to_le_bytes());
    index.extend_from_slice(name.as_bytes());
    index.push(0);
    index.push(1);
    index.extend_from_slice(&(values.len() as u64).to_le_bytes());
    index.extend_from_slice(&0u64.to_le_bytes());
    index.extend_from_slice(&((values.len() * 4) as u64).to_le_bytes());
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
    let meta_off = HEADER_SIZE as u64;
    let index_off = ((HEADER_SIZE + metadata.len()).div_ceil(64) * 64) as u64;
    let data_off = ((index_off as usize + index.len()).div_ceil(64) * 64) as u64;
    let mut data = vec![0u8; data_off as usize + bytes.len()];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&1u32.to_le_bytes());
    data[12..20].copy_from_slice(&meta_off.to_le_bytes());
    data[20..24].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&index_off.to_le_bytes());
    data[32..40].copy_from_slice(&data_off.to_le_bytes());
    data[HEADER_SIZE..HEADER_SIZE + metadata.len()].copy_from_slice(metadata.as_bytes());
    data[index_off as usize..index_off as usize + index.len()].copy_from_slice(&index);
    data[data_off as usize..].copy_from_slice(&bytes);
    AprModel::from_bytes(data).expect("test APR parses")
}

fn state_with(name: &str, values: &[f32]) -> AppState {
    let mut state = AppState::demo().expect("demo state");
    state.apr_model = Some(Arc::new(apr_with_tensor(name, values)));
    state
}

fn request(features: Vec<f32>, top_k: Option<usize>, confidence: bool) -> PredictRequest {
    PredictRequest {
        model: None,
        features,
        feature_names: None,
        top_k,
        include_confidence: confidence,
    }
}

#[tokio::test]
async fn predict_rejects_empty_features() {
    let state = state_with("weights", &[1.0, 2.0]);
    let err = apr_predict_handler(State(state), Json(request(vec![], None, false)))
        .await
        .expect_err("empty features");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert_eq!(err.1 .0.error, "Input features cannot be empty");
}

#[tokio::test]
async fn predict_without_a_tensor_named_weights_or_output_is_400() {
    // the demo tensor is called "weight" (singular): neither lookup name matches
    let state = state_with("weight", &[1.0, 2.0]);
    let err = apr_predict_handler(State(state), Json(request(vec![1.0, 2.0], None, false)))
        .await
        .expect_err("no weights/output tensor");
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(
        err.1 .0.error.contains("Inference failed"),
        "{}",
        err.1 .0.error
    );
}

#[tokio::test]
async fn predict_regression_is_the_dot_product_with_confidence_one() {
    let state = state_with("weights", &[2.0, 3.0]);
    let resp = match apr_predict_handler(
        State(state),
        Json(request(vec![1.0, 10.0], Some(3), true)),
    )
    .await
    {
        Ok(Json(r)) => r,
        Err((s, e)) => panic!("regression predict failed: {s} {}", e.0.error),
    };
    assert_eq!(resp.prediction, serde_json::json!(32.0));
    assert_eq!(resp.confidence, Some(1.0));
    assert_eq!(resp.model, "default");
    let top = resp.top_k_predictions.expect("top_k requested");
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].label, "32.0000");
    assert_eq!(top[0].score, 1.0);
}

#[tokio::test]
async fn predict_classification_takes_argmax_and_ranks_top_k() {
    // 3 weights vs 2 features: length mismatch, so the weights are the logits
    let state = state_with("output", &[0.1, 2.0, 0.5]);
    let Ok(Json(resp)) =
        apr_predict_handler(State(state), Json(request(vec![1.0, 1.0], Some(2), false))).await
    else {
        panic!("classification predict failed");
    };
    assert_eq!(resp.prediction, serde_json::json!("class_1"));
    assert_eq!(resp.confidence, None, "include_confidence=false hides it");
    let top = resp.top_k_predictions.expect("top_k requested");
    assert_eq!(
        top.iter().map(|p| p.label.as_str()).collect::<Vec<_>>(),
        ["class_1", "class_2"]
    );
    assert!(top[0].score > top[1].score && top[0].score < 1.0);
}

#[tokio::test]
async fn predict_with_no_apr_model_loaded_is_an_error() {
    let mut state = AppState::demo().expect("demo state");
    state.apr_model = None;
    let err = apr_predict_handler(State(state), Json(request(vec![1.0], None, false)))
        .await
        .expect_err("no model");
    assert!(err.0.is_client_error() || err.0.is_server_error());
}
