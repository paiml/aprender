//! Eval endpoint handlers.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::Deserialize;

use super::eval::{EvalResult, EvalStatus};
use super::state::BancoState;
use super::types::ErrorResponse;

/// POST /api/v1/eval/perplexity — compute perplexity on text.
pub async fn eval_perplexity_handler(
    State(state): State<BancoState>,
    Json(request): Json<PerplexityRequest>,
) -> Result<Json<EvalResult>, (StatusCode, Json<ErrorResponse>)> {
    let eval_id = state.evals.next_id();
    let model_name = state.model.info().map(|m| m.model_id).unwrap_or_else(|| "none".to_string());

    #[cfg(feature = "realizar")]
    let ppl_result = {
        let model = state.model.quantized_model();
        match model {
            Some(m) => {
                let token_ids = state.model.encode_text(&request.text);
                if token_ids.is_empty() {
                    None
                } else {
                    let max_tokens = request.max_tokens.unwrap_or(512) as usize;
                    let start = std::time::Instant::now();
                    let result = super::eval::compute_perplexity(&m, &token_ids, max_tokens);
                    let duration = start.elapsed().as_secs_f64();
                    result.map(|(ppl, tokens)| (ppl, tokens, duration))
                }
            }
            _ => None,
        }
    };
    #[cfg(not(feature = "realizar"))]
    let ppl_result: Option<(f64, usize, f64)> = {
        let _ = &request;
        None
    };

    let result = if let Some((ppl, tokens, duration)) = ppl_result {
        EvalResult {
            eval_id,
            model: model_name,
            metric: "perplexity".to_string(),
            value: ppl,
            tokens_evaluated: tokens,
            duration_secs: duration,
            status: EvalStatus::Complete,
        }
    } else {
        EvalResult {
            eval_id,
            model: model_name,
            metric: "perplexity".to_string(),
            value: 0.0,
            tokens_evaluated: 0,
            duration_secs: 0.0,
            status: EvalStatus::NoModel,
        }
    };

    state.evals.record(result.clone());
    Ok(Json(result))
}

/// GET /api/v1/eval/runs — list eval runs.
pub async fn list_eval_runs_handler(State(state): State<BancoState>) -> Json<EvalRunsResponse> {
    Json(EvalRunsResponse { runs: state.evals.list() })
}

/// GET /api/v1/eval/runs/:id — get eval result.
pub async fn get_eval_run_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<EvalResult>, (StatusCode, Json<ErrorResponse>)> {
    state.evals.get(&id).map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Eval run {id} not found"), "not_found", 404)),
    ))
}

#[derive(Debug, Deserialize)]
pub struct PerplexityRequest {
    pub text: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, serde::Serialize)]
pub struct EvalRunsResponse {
    pub runs: Vec<EvalResult>,
}
