//! Model registry handlers — pull, list, and manage cached models via pacha.
//!
//! With `native` feature (includes pacha): real registry operations.
//! Without: dry-run responses for API testing.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::{Deserialize, Serialize};

use super::state::BancoState;
use super::types::ErrorResponse;

/// POST /api/v1/models/pull — pull a model from the registry.
pub async fn pull_model_handler(
    State(_state): State<BancoState>,
    Json(request): Json<PullRequest>,
) -> Result<Json<PullResult>, (StatusCode, Json<ErrorResponse>)> {
    pull_model(&request.model_ref)
}

/// GET /api/v1/models/registry — list cached models.
pub async fn list_registry_handler(State(_state): State<BancoState>) -> Json<RegistryListResponse> {
    Json(RegistryListResponse { models: list_cached_models() })
}

/// DELETE /api/v1/models/registry/:name — remove a model from cache.
pub async fn remove_cached_model_handler(
    State(_state): State<BancoState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    remove_cached_model(&name)
}

// ============================================================================
// pacha-powered registry (native feature)
// ============================================================================

#[cfg(feature = "native")]
fn pull_model(model_ref: &str) -> Result<Json<PullResult>, (StatusCode, Json<ErrorResponse>)> {
    let mut fetcher = pacha::fetcher::ModelFetcher::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!("Registry init failed: {e}"), "registry_error", 500)),
        )
    })?;

    // Check if already cached
    if fetcher.is_cached(model_ref) {
        let cached = fetcher.list();
        let info = cached.iter().find(|m| m.name == model_ref);
        return Ok(Json(PullResult {
            model_ref: model_ref.to_string(),
            status: "cached".to_string(),
            path: info.map(|m| m.path.display().to_string()),
            size_bytes: info.map(|m| m.size_bytes),
            cache_hit: true,
            format: info.map(|m| format!("{:?}", m.format).to_lowercase()),
        }));
    }

    // Pull from registry
    match fetcher.pull_quiet(model_ref) {
        Ok(result) => Ok(Json(PullResult {
            model_ref: model_ref.to_string(),
            status: "pulled".to_string(),
            path: Some(result.path.display().to_string()),
            size_bytes: Some(result.size_bytes),
            cache_hit: result.cache_hit,
            format: Some(format!("{:?}", result.format).to_lowercase()),
        })),
        Err(e) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Model not found: {e}"), "not_found", 404)),
        )),
    }
}

#[cfg(feature = "native")]
fn list_cached_models() -> Vec<CachedModelInfo> {
    let fetcher = match pacha::fetcher::ModelFetcher::new() {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    fetcher
        .list()
        .into_iter()
        .map(|m| CachedModelInfo {
            name: m.name.clone(),
            version: m.version.clone(),
            path: m.path.display().to_string(),
            size_bytes: m.size_bytes,
            format: format!("{:?}", m.format).to_lowercase(),
        })
        .collect()
}

#[cfg(feature = "native")]
fn remove_cached_model(name: &str) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let mut fetcher = pacha::fetcher::ModelFetcher::new().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!("Registry: {e}"), "registry_error", 500)),
        )
    })?;

    match fetcher.remove(name) {
        Ok(true) => Ok(StatusCode::NO_CONTENT),
        Ok(false) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Model {name} not in cache"), "not_found", 404)),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!("Remove failed: {e}"), "registry_error", 500)),
        )),
    }
}

// ============================================================================
// Dry-run registry (no native feature)
// ============================================================================

#[cfg(not(feature = "native"))]
fn pull_model(model_ref: &str) -> Result<Json<PullResult>, (StatusCode, Json<ErrorResponse>)> {
    Ok(Json(PullResult {
        model_ref: model_ref.to_string(),
        status: "dry_run".to_string(),
        path: None,
        size_bytes: None,
        cache_hit: false,
        format: None,
    }))
}

#[cfg(not(feature = "native"))]
fn list_cached_models() -> Vec<CachedModelInfo> {
    Vec::new()
}

#[cfg(not(feature = "native"))]
fn remove_cached_model(_name: &str) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Types
// ============================================================================

/// Model pull request.
#[derive(Debug, Deserialize)]
pub struct PullRequest {
    /// Model reference: "llama3:8b-q4", "pacha://model:version", or file path.
    pub model_ref: String,
}

/// Model pull result.
#[derive(Debug, Clone, Serialize)]
pub struct PullResult {
    pub model_ref: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    pub cache_hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// Cached model info.
#[derive(Debug, Clone, Serialize)]
pub struct CachedModelInfo {
    pub name: String,
    pub version: String,
    pub path: String,
    pub size_bytes: u64,
    pub format: String,
}

/// Registry list response.
#[derive(Debug, Serialize)]
pub struct RegistryListResponse {
    pub models: Vec<CachedModelInfo>,
}
