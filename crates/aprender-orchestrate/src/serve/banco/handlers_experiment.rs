//! Experiment endpoint handlers.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::Deserialize;

use super::experiment::{Experiment, RunComparison};
use super::state::BancoState;
use super::types::ErrorResponse;

/// POST /api/v1/experiments — create experiment.
pub async fn create_experiment_handler(
    State(state): State<BancoState>,
    Json(request): Json<CreateExperimentRequest>,
) -> Json<Experiment> {
    let exp = state.experiments.create(&request.name, &request.description.unwrap_or_default());
    Json(exp)
}

/// GET /api/v1/experiments — list experiments.
pub async fn list_experiments_handler(
    State(state): State<BancoState>,
) -> Json<ExperimentsResponse> {
    Json(ExperimentsResponse { experiments: state.experiments.list() })
}

/// POST /api/v1/experiments/:id/runs — add a run to an experiment.
pub async fn add_run_to_experiment_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(request): Json<AddRunRequest>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.experiments.add_run(&id, &request.run_id).map(|()| StatusCode::OK).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Experiment {id} not found"), "not_found", 404)),
        )
    })
}

/// GET /api/v1/experiments/:id/compare — compare runs in experiment.
pub async fn compare_experiment_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<RunComparison>, (StatusCode, Json<ErrorResponse>)> {
    state.experiments.compare(&id, &state.training).map(Json).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Experiment {id} not found"), "not_found", 404)),
        )
    })
}

#[derive(Debug, Deserialize)]
pub struct CreateExperimentRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AddRunRequest {
    pub run_id: String,
}

#[derive(Debug, serde::Serialize)]
pub struct ExperimentsResponse {
    pub experiments: Vec<Experiment>,
}
