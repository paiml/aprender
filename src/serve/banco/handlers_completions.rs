//! OpenAI-compatible text completion handler (POST /v1/completions).
//!
//! This is the non-chat completion endpoint — takes a prompt string,
//! returns a completion. Used by many SDK tools that don't use chat format.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use super::state::BancoState;

/// POST /v1/completions — text completion (OpenAI-compatible).
pub async fn completions_handler(
    State(state): State<BancoState>,
    Json(request): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let model_name = request
        .model
        .clone()
        .or_else(|| state.model.info().map(|m| m.model_id))
        .unwrap_or_else(|| "banco-echo".to_string());

    // Convert text prompt to chat format and reuse inference
    let prompt = match &request.prompt {
        PromptInput::Single(s) => s.clone(),
        PromptInput::Multiple(v) => v.join("\n"),
    };

    let content = generate_completion(&state, &prompt, &request);

    let prompt_tokens = (prompt.len() / 4) as u32;
    let completion_tokens = (content.len() / 4) as u32;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

    Json(CompletionResponse {
        id: format!("cmpl-{now}"),
        object: "text_completion".to_string(),
        created: now,
        model: model_name,
        choices: vec![CompletionChoice {
            text: content,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

/// GET /v1/models/:id — model detail by ID.
pub async fn model_detail_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<ModelDetail>, axum::http::StatusCode> {
    // Check if it matches the loaded model
    if let Some(info) = state.model.info() {
        if info.model_id == id || format!("{:?}", info.format).to_lowercase() == id || id == "local"
        {
            return Ok(Json(ModelDetail {
                id: info.model_id,
                object: "model".to_string(),
                owned_by: "batuta".to_string(),
                permission: vec![],
            }));
        }
    }

    // Check backend list
    let models = state.list_models();
    if let Some(model) = models.data.iter().find(|m| m.id == id) {
        return Ok(Json(ModelDetail {
            id: model.id.clone(),
            object: "model".to_string(),
            owned_by: model.owned_by.clone(),
            permission: vec![],
        }));
    }

    Err(axum::http::StatusCode::NOT_FOUND)
}

/// Generate completion text.
#[cfg(feature = "realizar")]
fn generate_completion(state: &BancoState, prompt: &str, request: &CompletionRequest) -> String {
    use crate::serve::templates::ChatMessage;

    let messages = vec![ChatMessage::user(prompt)];
    let chat_req = super::types::BancoChatRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_tokens.unwrap_or(256),
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(1.0),
        stream: false,
        conversation_id: None,
        response_format: None,
        rag: false,
        rag_config: None,
        attachments: vec![],
        tools: None,
        tool_choice: None,
    };

    super::handlers_inference::try_inference(state, &chat_req)
        .map(|(text, _, _)| text)
        .unwrap_or_else(|| {
            if state.model.is_loaded() {
                format!("[completion] prompt={} chars, model loaded but inference unavailable", prompt.len())
            } else {
                format!(
                    "No model loaded. Load a model first:\n\
                     curl -X POST http://localhost:8090/api/v1/models/load -d '{{\"model\": \"./model.gguf\"}}'\n\n\
                     Your prompt ({} chars): {}",
                    prompt.len(),
                    if prompt.len() > 200 { &prompt[..200] } else { prompt }
                )
            }
        })
}

#[cfg(not(feature = "realizar"))]
fn generate_completion(state: &BancoState, prompt: &str, _request: &CompletionRequest) -> String {
    if state.model.is_loaded() {
        format!("[completion] prompt={} chars, inference feature not enabled", prompt.len())
    } else {
        format!(
            "No model loaded. Load a model first:\n\
             curl -X POST http://localhost:8090/api/v1/models/load -d '{{\"model\": \"./model.gguf\"}}'\n\n\
             Your prompt ({} chars): {}",
            prompt.len(),
            if prompt.len() > 200 { &prompt[..200] } else { prompt }
        )
    }
}

// ============================================================================
// Types
// ============================================================================

/// Prompt can be a single string or array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Single(String),
    Multiple(Vec<String>),
}

/// OpenAI-compatible completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: PromptInput,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub n: Option<u32>,
}

/// OpenAI-compatible completion response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

/// Completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: String,
}

/// Completion usage.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Model detail response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelDetail {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub permission: Vec<serde_json::Value>,
}
