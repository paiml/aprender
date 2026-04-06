//! Audit log query endpoint.

use axum::{extract::State, response::Json};
use serde::Deserialize;

use super::audit::AuditEntry;
use super::state::BancoState;

/// GET /api/v1/audit — query recent audit log entries.
pub async fn audit_query_handler(
    State(state): State<BancoState>,
    axum::extract::Query(params): axum::extract::Query<AuditQueryParams>,
) -> Json<AuditQueryResponse> {
    let limit = params.limit.unwrap_or(100).min(1000);
    let entries = state.audit_log.recent(limit);
    Json(AuditQueryResponse { total: state.audit_log.len(), returned: entries.len(), entries })
}

#[derive(Debug, Deserialize)]
pub struct AuditQueryParams {
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
pub struct AuditQueryResponse {
    pub total: usize,
    pub returned: usize,
    pub entries: Vec<AuditEntry>,
}
