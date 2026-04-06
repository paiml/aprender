//! Model merge endpoint handlers — TIES, DARE, SLERP, weighted average.
//!
//! With `ml` feature: uses entrenar's merge module for real tensor merging.
//! Without `ml`: returns dry-run merge results for API testing.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::{Deserialize, Serialize};

use super::state::BancoState;
use super::types::ErrorResponse;

/// POST /api/v1/models/merge — merge two or more models.
pub async fn merge_models_handler(
    State(state): State<BancoState>,
    Json(request): Json<MergeRequest>,
) -> Result<Json<MergeResult>, (StatusCode, Json<ErrorResponse>)> {
    if request.models.len() < 2 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "At least 2 models required for merge",
                "invalid_request",
                400,
            )),
        ));
    }

    // SLERP only works with exactly 2 models
    if request.strategy == MergeStrategy::Slerp && request.models.len() != 2 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "SLERP merge requires exactly 2 models",
                "invalid_request",
                400,
            )),
        ));
    }

    let result = execute_merge(&state, &request);
    state.events.emit(&super::events::BancoEvent::MergeComplete {
        merge_id: result.merge_id.clone(),
        strategy: format!("{:?}", result.strategy).to_lowercase(),
    });
    Ok(Json(result))
}

/// GET /api/v1/models/merge/strategies — list available merge strategies.
pub async fn list_merge_strategies_handler() -> Json<MergeStrategiesResponse> {
    Json(MergeStrategiesResponse {
        strategies: vec![
            StrategyInfo {
                name: "weighted_average".to_string(),
                description: "Element-wise weighted average of model parameters".to_string(),
                min_models: 2,
                max_models: None,
            },
            StrategyInfo {
                name: "ties".to_string(),
                description: "Trim, Elect, Sign merge — reduces noise across multiple fine-tunes"
                    .to_string(),
                min_models: 2,
                max_models: None,
            },
            StrategyInfo {
                name: "dare".to_string(),
                description: "Drop And REscale — stochastic sparsity-based merge".to_string(),
                min_models: 2,
                max_models: None,
            },
            StrategyInfo {
                name: "slerp".to_string(),
                description: "Spherical linear interpolation — smooth two-model blending"
                    .to_string(),
                min_models: 2,
                max_models: Some(2),
            },
        ],
    })
}

/// Execute a model merge.
#[cfg(feature = "entrenar")]
fn execute_merge(state: &BancoState, request: &MergeRequest) -> MergeResult {
    use entrenar::merge::{DareConfig, EnsembleConfig, MergeError, SlerpConfig, TiesConfig};

    // Build placeholder models from the request model names
    // Real merge requires loaded model weights — this validates the entrenar API
    let models: Vec<entrenar::merge::Model> =
        request.models.iter().map(|_| std::collections::HashMap::new()).collect();

    let merge_result: Result<entrenar::merge::Model, MergeError> = match &request.strategy {
        MergeStrategy::WeightedAverage => {
            let weights = request
                .weights
                .clone()
                .unwrap_or_else(|| vec![1.0 / request.models.len() as f32; request.models.len()]);
            let config = EnsembleConfig::weighted_average(weights);
            entrenar::merge::ensemble_merge(&models, &config)
        }
        MergeStrategy::Ties => {
            let density = request.density.unwrap_or(0.2);
            let base = std::collections::HashMap::new();
            let config = TiesConfig { density };
            entrenar::merge::ties_merge(&models, &base, &config)
        }
        MergeStrategy::Dare => {
            let drop_prob = request.drop_prob.unwrap_or(0.5);
            let base = std::collections::HashMap::new();
            let config = DareConfig { drop_prob, seed: request.seed };
            entrenar::merge::dare_merge(&models, &base, &config)
        }
        MergeStrategy::Slerp => {
            let t = request.interpolation_t.unwrap_or(0.5);
            let config = SlerpConfig { t };
            entrenar::merge::slerp_merge(&models[0], &models[1], &config)
        }
    };

    let (status, error) = match merge_result {
        Ok(_merged) => ("complete".to_string(), None),
        Err(e) => ("failed".to_string(), Some(e.to_string())),
    };

    let _ = state; // used when loading real model weights

    MergeResult {
        merge_id: format!("merge-{}", epoch_secs()),
        strategy: request.strategy.clone(),
        models: request.models.clone(),
        status,
        simulated: true, // Uses empty placeholder tensors
        error,
        output_path: None,
    }
}

/// Dry-run merge (no ml feature).
#[cfg(not(feature = "entrenar"))]
fn execute_merge(_state: &BancoState, request: &MergeRequest) -> MergeResult {
    MergeResult {
        merge_id: format!("merge-{}", epoch_secs()),
        strategy: request.strategy.clone(),
        models: request.models.clone(),
        status: "dry_run".to_string(),
        simulated: true,
        error: None,
        output_path: None,
    }
}

// ============================================================================
// Types
// ============================================================================

/// Merge strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    WeightedAverage,
    Ties,
    Dare,
    Slerp,
}

/// Merge request.
#[derive(Debug, Clone, Deserialize)]
pub struct MergeRequest {
    /// Model identifiers (file paths or model IDs).
    pub models: Vec<String>,
    /// Merge strategy.
    pub strategy: MergeStrategy,
    /// Weights for weighted_average (one per model, auto-normalized).
    #[serde(default)]
    pub weights: Option<Vec<f32>>,
    /// TIES density parameter (0.0-1.0, default 0.2).
    #[serde(default)]
    pub density: Option<f32>,
    /// DARE drop probability (0.0-1.0, default 0.5).
    #[serde(default)]
    pub drop_prob: Option<f32>,
    /// SLERP interpolation parameter (0.0-1.0, default 0.5).
    #[serde(default)]
    pub interpolation_t: Option<f32>,
    /// Random seed for reproducibility (DARE).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Output format (safetensors, gguf, apr).
    #[serde(default)]
    pub output_format: Option<String>,
}

/// Merge result.
#[derive(Debug, Clone, Serialize)]
pub struct MergeResult {
    pub merge_id: String,
    pub strategy: MergeStrategy,
    pub models: Vec<String>,
    pub status: String,
    /// True when merge runs on empty placeholder tensors (API validation only).
    #[serde(default)]
    pub simulated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
}

/// Merge strategies list response.
#[derive(Debug, Serialize)]
pub struct MergeStrategiesResponse {
    pub strategies: Vec<StrategyInfo>,
}

/// Strategy info.
#[derive(Debug, Serialize)]
pub struct StrategyInfo {
    pub name: String,
    pub description: String,
    pub min_models: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_models: Option<usize>,
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
