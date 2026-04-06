//! Token-related handlers — embeddings, tokenize, detokenize, inference parameters.

use axum::{extract::State, response::Json};

use super::state::BancoState;
use super::types::{
    DetokenizeRequest, DetokenizeResponse, EmbeddingData, EmbeddingsRequest, EmbeddingsResponse,
    EmbeddingsUsage, InferenceParams, TokenizeRequest, TokenizeResponse,
};
use crate::serve::templates::ChatMessage;

// ============================================================================
// Embeddings
// ============================================================================

pub async fn embeddings_handler(
    State(state): State<BancoState>,
    Json(request): Json<EmbeddingsRequest>,
) -> Json<EmbeddingsResponse> {
    let model_name = request.model.unwrap_or_else(|| "banco-heuristic".to_string());
    let texts = request.input.texts();

    #[cfg(feature = "realizar")]
    let use_model = state.model.has_inference_model();

    let data: Vec<EmbeddingData> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            #[cfg(feature = "realizar")]
            let embedding = if use_model {
                model_embedding(&state, text).unwrap_or_else(|| heuristic_embedding(text))
            } else {
                heuristic_embedding(text)
            };
            #[cfg(not(feature = "realizar"))]
            let embedding = heuristic_embedding(text);
            EmbeddingData { object: "embedding".to_string(), index: i as u32, embedding }
        })
        .collect();

    let total_tokens: u32 = texts
        .iter()
        .map(|t| state.context_manager.estimate_tokens(&[ChatMessage::user(*t)]) as u32)
        .sum();

    Json(EmbeddingsResponse {
        object: "list".to_string(),
        data,
        model: model_name,
        usage: EmbeddingsUsage { prompt_tokens: total_tokens, total_tokens },
    })
}

#[cfg(feature = "realizar")]
fn model_embedding(state: &BancoState, text: &str) -> Option<Vec<f32>> {
    let model = state.model.quantized_model()?;
    let token_ids = state.model.encode_text(text);
    super::inference::embed_tokens(&model, &token_ids)
}

fn heuristic_embedding(text: &str) -> Vec<f32> {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    let mut state = hash;
    (0..128)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        })
        .collect()
}

// ============================================================================
// Tokenize / Detokenize
// ============================================================================

pub async fn tokenize_handler(
    State(state): State<BancoState>,
    Json(request): Json<TokenizeRequest>,
) -> Json<TokenizeResponse> {
    #[cfg(feature = "realizar")]
    if state.model.has_inference_model() {
        let tokens = state.model.encode_text(&request.text);
        if !tokens.is_empty() {
            let count = tokens.len() as u32;
            return Json(TokenizeResponse { tokens, count });
        }
    }

    let estimated = state.context_manager.estimate_tokens(&[ChatMessage::user(&request.text)]);
    let tokens: Vec<u32> = (0..estimated as u32).collect();
    Json(TokenizeResponse { count: estimated as u32, tokens })
}

pub async fn detokenize_handler(
    State(state): State<BancoState>,
    Json(request): Json<DetokenizeRequest>,
) -> Json<DetokenizeResponse> {
    #[cfg(feature = "realizar")]
    if state.model.has_inference_model() {
        let vocab = state.model.vocabulary();
        if !vocab.is_empty() {
            let text: String =
                request.tokens.iter().filter_map(|&id| vocab.get(id as usize)).cloned().collect();
            return Json(DetokenizeResponse { text });
        }
    }

    let _ = &state;
    let approx_chars = request.tokens.len() * 4;
    let text = format!("[{} tokens ≈ {} chars]", request.tokens.len(), approx_chars);
    Json(DetokenizeResponse { text })
}

// ============================================================================
// Inference Parameters
// ============================================================================

pub async fn get_parameters_handler(State(state): State<BancoState>) -> Json<InferenceParams> {
    let params = state.inference_params.read().unwrap_or_else(|e| e.into_inner());
    Json(params.clone())
}

pub async fn update_parameters_handler(
    State(state): State<BancoState>,
    Json(update): Json<InferenceParams>,
) -> Json<InferenceParams> {
    if let Ok(mut params) = state.inference_params.write() {
        *params = update;
    }
    let params = state.inference_params.read().unwrap_or_else(|e| e.into_inner());
    Json(params.clone())
}
