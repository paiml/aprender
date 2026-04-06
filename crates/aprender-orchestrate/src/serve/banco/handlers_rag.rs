//! RAG endpoint handlers — index management and status.

use axum::{extract::State, http::StatusCode, response::Json};

use super::rag::RagStatus;
use super::state::BancoState;

/// POST /api/v1/rag/index — force re-index all uploaded documents.
pub async fn rag_index_handler(State(state): State<BancoState>) -> Json<RagIndexResponse> {
    let files = state.files.list();
    let mut indexed = 0;

    for file_info in &files {
        if let Some(content) = state.files.read_content(&file_info.id) {
            let text = String::from_utf8_lossy(&content);
            state.rag.index_document(&file_info.id, &file_info.name, &text);
            indexed += 1;
        }
    }

    let status = state.rag.status();
    state.events.emit(&super::events::BancoEvent::RagIndexed {
        doc_count: status.doc_count,
        chunk_count: status.chunk_count,
    });
    Json(RagIndexResponse { indexed_files: indexed, status })
}

/// GET /api/v1/rag/status — index stats.
pub async fn rag_status_handler(State(state): State<BancoState>) -> Json<RagStatus> {
    Json(state.rag.status())
}

/// DELETE /api/v1/rag/index — clear the RAG index.
pub async fn rag_clear_handler(State(state): State<BancoState>) -> StatusCode {
    state.rag.clear();
    StatusCode::NO_CONTENT
}

/// GET /api/v1/rag/search?q=query&top_k=5 — search indexed documents.
pub async fn rag_search_handler(
    State(state): State<BancoState>,
    axum::extract::Query(params): axum::extract::Query<RagSearchParams>,
) -> Json<RagSearchResponse> {
    let query = params.q.unwrap_or_default();
    let top_k = params.top_k.unwrap_or(5);
    let min_score = params.min_score.unwrap_or(0.0);
    let results = state.rag.search(&query, top_k, min_score);
    Json(RagSearchResponse { query, results })
}

#[derive(Debug, serde::Deserialize)]
pub struct RagSearchParams {
    pub q: Option<String>,
    pub top_k: Option<usize>,
    pub min_score: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
pub struct RagSearchResponse {
    pub query: String,
    pub results: Vec<super::rag::RagResult>,
}

/// Response from index rebuild.
#[derive(Debug, serde::Serialize)]
pub struct RagIndexResponse {
    pub indexed_files: usize,
    pub status: RagStatus,
}
