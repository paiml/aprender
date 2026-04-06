//! Prompt preset CRUD handlers.

use axum::{extract::State, http::StatusCode, response::Json};

use super::state::BancoState;
use super::types::{ErrorResponse, PromptsListResponse, SavePromptRequest};

pub async fn list_prompts_handler(State(state): State<BancoState>) -> Json<PromptsListResponse> {
    Json(PromptsListResponse { presets: state.prompts.list() })
}

pub async fn get_prompt_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<super::prompts::PromptPreset>, (StatusCode, Json<ErrorResponse>)> {
    state.prompts.get(&id).map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Preset {id} not found"), "not_found", 404)),
    ))
}

pub async fn save_prompt_handler(
    State(state): State<BancoState>,
    Json(request): Json<SavePromptRequest>,
) -> Json<super::prompts::PromptPreset> {
    Json(state.prompts.create(&request.name, &request.content))
}

pub async fn delete_prompt_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    if state.prompts.delete(&id) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("Preset {id} not found"), "not_found", 404)),
        ))
    }
}
