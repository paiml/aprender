//! Batch inference endpoint handlers.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::Deserialize;

use super::batch::{BatchItem, BatchItemResult, BatchJob};
use super::state::BancoState;
use super::types::ErrorResponse;

/// POST /api/v1/batch — submit a batch of prompts for processing.
pub async fn submit_batch_handler(
    State(state): State<BancoState>,
    Json(request): Json<BatchRequest>,
) -> Json<BatchJob> {
    let job = state.batches.run(request.items, |item| {
        // Try inference when model is loaded
        #[cfg(feature = "realizar")]
        if let Some(result) = try_batch_inference(&state, item) {
            return result;
        }

        // Dry-run fallback
        let formatted = state.template_engine.apply(&item.messages);
        let content = format!(
            "[batch dry-run] id={} | prompt_len={} | formatted_len={}",
            item.id,
            item.messages.len(),
            formatted.len()
        );
        let tokens = (content.len() / 4) as u32;
        BatchItemResult {
            id: item.id.clone(),
            content,
            finish_reason: "dry_run".to_string(),
            tokens,
        }
    });
    Json(job)
}

/// Try to run inference for a single batch item.
#[cfg(feature = "realizar")]
fn try_batch_inference(state: &BancoState, item: &BatchItem) -> Option<BatchItemResult> {
    let model = state.model.quantized_model()?;
    let vocab = state.model.vocabulary();
    if vocab.is_empty() {
        return None;
    }

    let formatted = state.template_engine.apply(&item.messages);
    let prompt_tokens = state.model.encode_text(&formatted);
    if prompt_tokens.is_empty() {
        return None;
    }

    let server_params = state.inference_params.read().ok()?;
    let params = super::inference::SamplingParams {
        temperature: server_params.temperature,
        top_k: server_params.top_k,
        max_tokens: item.max_tokens,
    };
    drop(server_params);

    match super::inference::generate_sync(&model, &vocab, &prompt_tokens, &params) {
        Ok(result) => Some(BatchItemResult {
            id: item.id.clone(),
            content: result.text,
            finish_reason: result.finish_reason,
            tokens: result.token_count,
        }),
        Err(_) => None,
    }
}

/// GET /api/v1/batch/:id — get batch job status and results.
pub async fn get_batch_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchJob>, (StatusCode, Json<ErrorResponse>)> {
    state.batches.get(&id).map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Batch {id} not found"), "not_found", 404)),
    ))
}

/// GET /api/v1/batch — list all batch jobs.
pub async fn list_batches_handler(State(state): State<BancoState>) -> Json<BatchListResponse> {
    Json(BatchListResponse { batches: state.batches.list() })
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub items: Vec<BatchItem>,
}

#[derive(Debug, serde::Serialize)]
pub struct BatchListResponse {
    pub batches: Vec<BatchJob>,
}
