//! Conversation CRUD handlers.

use axum::{extract::State, http::StatusCode, response::Json};

use super::conversations::Conversation;
use super::state::BancoState;
use super::types::{
    ConversationCreatedResponse, ConversationResponse, ConversationsListResponse,
    CreateConversationRequest, ErrorResponse,
};

pub async fn create_conversation_handler(
    State(state): State<BancoState>,
    Json(request): Json<CreateConversationRequest>,
) -> Json<ConversationCreatedResponse> {
    let model = request.model.unwrap_or_else(|| "banco-echo".to_string());
    let id = state.conversations.create(&model);

    if let Some(title) = request.title {
        if let Some(mut conv) = state.conversations.get(&id) {
            conv.meta.title = title;
        }
    }

    let conv = state.conversations.get(&id);
    let title = conv.map(|c| c.meta.title).unwrap_or_else(|| "New conversation".to_string());
    Json(ConversationCreatedResponse { id, title })
}

pub async fn list_conversations_handler(
    State(state): State<BancoState>,
) -> Json<ConversationsListResponse> {
    Json(ConversationsListResponse { conversations: state.conversations.list() })
}

pub async fn get_conversation_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<ConversationResponse>, (StatusCode, Json<ErrorResponse>)> {
    state.conversations.get(&id).map(|c| Json(ConversationResponse { conversation: c })).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Conversation {id} not found"), "not_found", 404)),
    ))
}

pub async fn delete_conversation_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.conversations.delete(&id).map(|()| StatusCode::NO_CONTENT).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Conversation {id} not found"), "not_found", 404)),
        )
    })
}

/// PATCH /api/v1/conversations/:id — rename a conversation.
pub async fn rename_conversation_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(request): Json<RenameRequest>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.conversations.rename(&id, &request.title).map(|()| StatusCode::OK).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Conversation {id} not found"), "not_found", 404)),
        )
    })
}

#[derive(Debug, serde::Deserialize)]
pub struct RenameRequest {
    pub title: String,
}

/// GET /api/v1/conversations/search?q=query — search conversations by content.
pub async fn search_conversations_handler(
    State(state): State<BancoState>,
    axum::extract::Query(params): axum::extract::Query<SearchParams>,
) -> Json<ConversationsListResponse> {
    let results = state.conversations.search(&params.q.unwrap_or_default());
    Json(ConversationsListResponse { conversations: results })
}

#[derive(Debug, serde::Deserialize)]
pub struct SearchParams {
    pub q: Option<String>,
}

/// GET /api/v1/conversations/export — export all conversations as JSON.
pub async fn export_conversations_handler(
    State(state): State<BancoState>,
) -> Json<Vec<Conversation>> {
    Json(state.conversations.export_all())
}

/// POST /api/v1/conversations/import — import conversations from JSON.
pub async fn import_conversations_handler(
    State(state): State<BancoState>,
    Json(conversations): Json<Vec<Conversation>>,
) -> Json<serde_json::Value> {
    let count = state.conversations.import_all(conversations);
    Json(serde_json::json!({ "imported": count }))
}
