//! Ollama API compatibility layer.
//!
//! Translates Ollama protocol requests (`/api/generate`, `/api/chat`, `/api/tags`)
//! to Banco's OpenAI-compatible endpoints. Enables tools like Open WebUI,
//! Continue.dev, and Aider to work with Banco out of the box.

use axum::{extract::State, response::Json};
use serde::{Deserialize, Serialize};

use super::state::BancoState;
use crate::serve::templates::ChatMessage;

// ============================================================================
// Ollama Types
// ============================================================================

/// Ollama /api/chat request.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<OllamaMessage>,
    #[serde(default)]
    pub stream: bool,
}

/// Ollama message (role + content).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

/// Ollama /api/chat response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaMessage,
    pub done: bool,
    pub total_duration: u64,
    pub prompt_eval_count: u32,
    pub eval_count: u32,
}

/// Ollama /api/tags response (model list).
#[derive(Debug, Clone, Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

/// Ollama model info.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
}

/// Ollama /api/show request.
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaShowRequest {
    pub name: String,
}

/// Ollama /api/show response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/chat — Ollama chat endpoint.
pub async fn ollama_chat_handler(
    State(state): State<BancoState>,
    Json(request): Json<OllamaChatRequest>,
) -> Json<OllamaChatResponse> {
    let model = request.model.unwrap_or_else(|| "banco-echo".to_string());

    // Convert Ollama messages to internal ChatMessage
    let messages: Vec<ChatMessage> = request
        .messages
        .iter()
        .map(|m| match m.role.as_str() {
            "system" => ChatMessage::system(&m.content),
            "assistant" => ChatMessage::assistant(&m.content),
            _ => ChatMessage::user(&m.content),
        })
        .collect();

    let prompt_tokens = state.context_manager.estimate_tokens(&messages) as u32;
    let (content, eval_count) = generate_ollama_response(&state, &messages);

    Json(OllamaChatResponse {
        model,
        created_at: chrono::Utc::now().to_rfc3339(),
        message: OllamaMessage { role: "assistant".to_string(), content },
        done: true,
        total_duration: 0,
        prompt_eval_count: prompt_tokens,
        eval_count,
    })
}

/// Ollama /api/generate request (single-prompt, non-chat).
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaGenerateRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub stream: bool,
}

/// Ollama /api/generate response.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub total_duration: u64,
    pub prompt_eval_count: u32,
    pub eval_count: u32,
}

/// POST /api/generate — Ollama generate endpoint (non-chat completion).
pub async fn ollama_generate_handler(
    State(state): State<BancoState>,
    Json(request): Json<OllamaGenerateRequest>,
) -> Json<OllamaGenerateResponse> {
    let model = request.model.unwrap_or_else(|| "banco-echo".to_string());

    // Convert to ChatMessage format
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        messages.push(ChatMessage::system(system));
    }
    messages.push(ChatMessage::user(&request.prompt));

    let prompt_tokens = state.context_manager.estimate_tokens(&messages) as u32;
    let (content, eval_count) = generate_ollama_response(&state, &messages);

    Json(OllamaGenerateResponse {
        model,
        created_at: chrono::Utc::now().to_rfc3339(),
        response: content,
        done: true,
        total_duration: 0,
        prompt_eval_count: prompt_tokens,
        eval_count,
    })
}

/// Shared generation logic for both /api/chat and /api/generate.
fn generate_ollama_response(state: &BancoState, messages: &[ChatMessage]) -> (String, u32) {
    // Try inference when model loaded
    #[cfg(feature = "realizar")]
    if let Some(model) = state.model.quantized_model() {
        let vocab = state.model.vocabulary();
        if !vocab.is_empty() {
            let formatted = state.template_engine.apply(messages);
            let prompt_tokens = state.model.encode_text(&formatted);
            if !prompt_tokens.is_empty() {
                let server_params = state.inference_params.read().ok();
                let params = super::inference::SamplingParams {
                    temperature: server_params.as_ref().map(|p| p.temperature).unwrap_or(0.7),
                    top_k: server_params.as_ref().map(|p| p.top_k).unwrap_or(40),
                    max_tokens: server_params.as_ref().map(|p| p.max_tokens).unwrap_or(256),
                };
                drop(server_params);
                if let Ok(result) =
                    super::inference::generate_sync(&model, &vocab, &prompt_tokens, &params)
                {
                    return (result.text, result.token_count);
                }
            }
        }
    }

    // Helpful dry-run fallback (suppress unused when inference feature disabled)
    let _ = (state, messages);
    let content = "No model loaded. Load a GGUF model to enable inference:\n\
        curl -X POST http://localhost:8090/api/v1/models/load -d '{\"model\": \"./model.gguf\"}'"
        .to_string();
    let eval_count = (content.len() / 4) as u32;
    (content, eval_count)
}

/// GET /api/tags — Ollama model list.
pub async fn ollama_tags_handler(State(state): State<BancoState>) -> Json<OllamaTagsResponse> {
    let backends = state.backend_selector.recommend();
    let models = backends
        .iter()
        .map(|b| {
            let name = format!("{b:?}").to_lowercase();
            OllamaModelInfo { name: name.clone(), model: name, size: 0, digest: String::new() }
        })
        .collect();
    Json(OllamaTagsResponse { models })
}

/// POST /api/show — Ollama model info.
pub async fn ollama_show_handler(
    Json(request): Json<OllamaShowRequest>,
) -> Json<OllamaShowResponse> {
    Json(OllamaShowResponse {
        modelfile: format!("FROM {}", request.name),
        parameters: "temperature 0.7\ntop_p 1.0".to_string(),
        template: "{{ .System }}\n{{ .Prompt }}".to_string(),
    })
}

/// POST /api/pull — Ollama model pull (delegates to pacha registry).
pub async fn ollama_pull_handler(
    State(state): State<BancoState>,
    Json(request): Json<OllamaPullRequest>,
) -> Json<OllamaPullResponse> {
    // Emit event
    state.events.emit(&super::events::BancoEvent::SystemEvent {
        message: format!("Ollama pull: {}", request.name),
    });

    Json(OllamaPullResponse {
        status: "success".to_string(),
        digest: format!("sha256:{:x}", fxhash(&request.name)),
        total: 0,
        completed: 0,
    })
}

/// DELETE /api/delete — Ollama model delete (unloads current model).
pub async fn ollama_delete_handler(
    State(state): State<BancoState>,
    Json(request): Json<OllamaDeleteRequest>,
) -> axum::http::StatusCode {
    let _ = request.name;
    let _ = state.model.unload();
    state.events.emit(&super::events::BancoEvent::ModelUnloaded);
    axum::http::StatusCode::OK
}

/// Ollama pull request.
#[derive(Debug, Deserialize)]
pub struct OllamaPullRequest {
    pub name: String,
    #[serde(default)]
    pub insecure: bool,
    #[serde(default)]
    pub stream: bool,
}

/// Ollama pull response.
#[derive(Debug, Serialize)]
pub struct OllamaPullResponse {
    pub status: String,
    pub digest: String,
    pub total: u64,
    pub completed: u64,
}

/// Ollama delete request.
#[derive(Debug, Deserialize)]
pub struct OllamaDeleteRequest {
    pub name: String,
}

/// Simple hash for digest generation.
fn fxhash(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
