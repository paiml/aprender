//! HTTP request handlers
//!
//! Axum handlers for the tracking server API.

use crate::server::{
    state::{AppState, RunStatus},
    ApiResponse, CreateExperimentRequest, CreateRunRequest, ExperimentResponse, HealthResponse,
    LogMetricsRequest, LogParamsRequest, RunResponse, UpdateRunRequest,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};

/// Generate a request ID
fn request_id() -> String {
    format!("req-{:016x}", rand::random::<u64>())
}

/// Health check handler
pub async fn health_check(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let health = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: state.uptime_secs(),
        experiments_count: state.storage.experiments_count(),
        runs_count: state.storage.runs_count(),
    };

    (StatusCode::OK, Json(health))
}

/// Create a new experiment
pub async fn create_experiment(
    State(state): State<AppState>,
    Json(payload): Json<CreateExperimentRequest>,
) -> (StatusCode, Json<ApiResponse<ExperimentResponse>>) {
    let req_id = request_id();

    match state.storage.create_experiment(&payload.name, payload.description, payload.tags) {
        Ok(exp) => {
            let response: ExperimentResponse = exp.into();
            (StatusCode::CREATED, Json(ApiResponse::success(response, &req_id)))
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::error(&e.to_string(), &req_id)))
        }
    }
}

/// Get an experiment by ID
pub async fn get_experiment(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> (StatusCode, Json<ApiResponse<ExperimentResponse>>) {
    let req_id = request_id();

    match state.storage.get_experiment(&id) {
        Ok(exp) => {
            let response: ExperimentResponse = exp.into();
            (StatusCode::OK, Json(ApiResponse::success(response, &req_id)))
        }
        Err(e) => (StatusCode::NOT_FOUND, Json(ApiResponse::error(&e.to_string(), &req_id))),
    }
}

/// List all experiments
pub async fn list_experiments(
    State(state): State<AppState>,
) -> (StatusCode, Json<ApiResponse<Vec<ExperimentResponse>>>) {
    let req_id = request_id();

    match state.storage.list_experiments() {
        Ok(exps) => {
            let responses: Vec<ExperimentResponse> = exps.into_iter().map(Into::into).collect();
            (StatusCode::OK, Json(ApiResponse::success(responses, &req_id)))
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::error(&e.to_string(), &req_id)))
        }
    }
}

/// Create a new run
pub async fn create_run(
    State(state): State<AppState>,
    Json(payload): Json<CreateRunRequest>,
) -> (StatusCode, Json<ApiResponse<RunResponse>>) {
    let req_id = request_id();

    match state.storage.create_run(&payload.experiment_id, payload.name, payload.tags) {
        Ok(run) => {
            let response: RunResponse = run.into();
            (StatusCode::CREATED, Json(ApiResponse::success(response, &req_id)))
        }
        Err(e) => {
            let status = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(ApiResponse::error(&e.to_string(), &req_id)))
        }
    }
}

/// Get a run by ID
pub async fn get_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> (StatusCode, Json<ApiResponse<RunResponse>>) {
    let req_id = request_id();

    match state.storage.get_run(&id) {
        Ok(run) => {
            let response: RunResponse = run.into();
            (StatusCode::OK, Json(ApiResponse::success(response, &req_id)))
        }
        Err(e) => (StatusCode::NOT_FOUND, Json(ApiResponse::error(&e.to_string(), &req_id))),
    }
}

/// Update a run
pub async fn update_run(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(payload): Json<UpdateRunRequest>,
) -> (StatusCode, Json<ApiResponse<RunResponse>>) {
    let req_id = request_id();

    let status = payload.status.as_ref().and_then(|s| s.parse::<RunStatus>().ok());

    let end_time = payload.end_time.as_ref().and_then(|t| {
        chrono::DateTime::parse_from_rfc3339(t).ok().map(|dt| dt.with_timezone(&chrono::Utc))
    });

    match state.storage.update_run(&id, status, end_time) {
        Ok(run) => {
            let response: RunResponse = run.into();
            (StatusCode::OK, Json(ApiResponse::success(response, &req_id)))
        }
        Err(e) => {
            let status_code = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status_code, Json(ApiResponse::error(&e.to_string(), &req_id)))
        }
    }
}

/// Log parameters for a run
pub async fn log_params(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(payload): Json<LogParamsRequest>,
) -> (StatusCode, Json<ApiResponse<&'static str>>) {
    let req_id = request_id();

    match state.storage.log_params(&id, payload.params) {
        Ok(()) => (StatusCode::OK, Json(ApiResponse::success("Parameters logged", &req_id))),
        Err(e) => {
            let status = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(ApiResponse::error(&e.to_string(), &req_id)))
        }
    }
}

/// Log metrics for a run
pub async fn log_metrics(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(payload): Json<LogMetricsRequest>,
) -> (StatusCode, Json<ApiResponse<&'static str>>) {
    let req_id = request_id();

    match state.storage.log_metrics(&id, payload.metrics) {
        Ok(()) => (StatusCode::OK, Json(ApiResponse::success("Metrics logged", &req_id))),
        Err(e) => {
            let status = if e.to_string().contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(ApiResponse::error(&e.to_string(), &req_id)))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerConfig;

    fn test_state() -> AppState {
        AppState::new(ServerConfig::default())
    }

    #[tokio::test]
    async fn test_health_check() {
        let state = test_state();
        let (status, Json(body)) = health_check(State(state)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.status, "healthy");
    }

    #[tokio::test]
    async fn test_create_experiment() {
        let state = test_state();
        let req =
            CreateExperimentRequest { name: "test".to_string(), description: None, tags: None };

        let (status, _) = create_experiment(State(state), Json(req)).await;
        assert_eq!(status, StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_get_experiment() {
        let state = test_state();
        let exp =
            state.storage.create_experiment("test", None, None).expect("operation should succeed");

        let (status, _) = get_experiment(State(state), Path(exp.id)).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_experiment_not_found() {
        let state = test_state();

        let (status, _) = get_experiment(State(state), Path("nonexistent".to_string())).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_list_experiments() {
        let state = test_state();
        state.storage.create_experiment("exp1", None, None).expect("operation should succeed");
        state.storage.create_experiment("exp2", None, None).expect("operation should succeed");

        let (status, _) = list_experiments(State(state)).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_create_run() {
        let state = test_state();
        let exp =
            state.storage.create_experiment("test", None, None).expect("operation should succeed");

        let req =
            CreateRunRequest { experiment_id: exp.id, name: Some("run-1".to_string()), tags: None };

        let (status, _) = create_run(State(state), Json(req)).await;
        assert_eq!(status, StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_create_run_invalid_experiment() {
        let state = test_state();

        let req =
            CreateRunRequest { experiment_id: "nonexistent".to_string(), name: None, tags: None };

        let (status, _) = create_run(State(state), Json(req)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_run() {
        let state = test_state();
        let exp =
            state.storage.create_experiment("test", None, None).expect("operation should succeed");
        let run = state.storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let (status, _) = get_run(State(state), Path(run.id)).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_update_run() {
        let state = test_state();
        let exp =
            state.storage.create_experiment("test", None, None).expect("operation should succeed");
        let run = state.storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let req = UpdateRunRequest { status: Some("completed".to_string()), end_time: None };

        let (status, _) = update_run(State(state), Path(run.id), Json(req)).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_log_params() {
        let state = test_state();
        let exp =
            state.storage.create_experiment("test", None, None).expect("operation should succeed");
        let run = state.storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let mut params = std::collections::HashMap::new();
        params.insert("lr".to_string(), serde_json::json!(0.001));
        let req = LogParamsRequest { params };

        let (status, _) = log_params(State(state), Path(run.id), Json(req)).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_log_metrics() {
        let state = test_state();
        let exp =
            state.storage.create_experiment("test", None, None).expect("operation should succeed");
        let run = state.storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let mut metrics = std::collections::HashMap::new();
        metrics.insert("loss".to_string(), 0.5);
        let req = LogMetricsRequest { metrics, step: None };

        let (status, _) = log_metrics(State(state), Path(run.id), Json(req)).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_log_params_not_found() {
        let state = test_state();

        let req = LogParamsRequest { params: std::collections::HashMap::new() };

        let (status, _) =
            log_params(State(state), Path("nonexistent".to_string()), Json(req)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
