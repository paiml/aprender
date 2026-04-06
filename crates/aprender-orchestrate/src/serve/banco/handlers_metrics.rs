//! Prometheus metrics endpoint handler.

use axum::{extract::State, http::header, response::IntoResponse};

use super::state::BancoState;

/// GET /api/v1/metrics — Prometheus-compatible metrics.
pub async fn metrics_handler(State(state): State<BancoState>) -> impl IntoResponse {
    let body = state.metrics.render(state.model.is_loaded(), 85);
    ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body)
}
