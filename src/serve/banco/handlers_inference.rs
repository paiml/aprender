//! Inference bridge — connects banco handlers to realizar inference engine.
//!
//! Gated behind `#[cfg(feature = "realizar")]`.

use super::state::BancoState;
use super::types::BancoChatRequest;

/// Try to run inference if a model is loaded and the inference feature is enabled.
/// Returns Some((content, finish_reason, completion_tokens)) on success.
pub fn try_inference(
    state: &BancoState,
    request: &BancoChatRequest,
) -> Option<(String, String, u32)> {
    let model = state.model.quantized_model()?;
    let vocab = state.model.vocabulary();
    if vocab.is_empty() {
        return None;
    }

    let formatted = state.template_engine.apply(&request.messages);
    // Use proper BPE tokenizer when available, else greedy fallback
    let prompt_tokens = state.model.encode_text(&formatted);
    if prompt_tokens.is_empty() {
        return None;
    }

    let server_params = state.inference_params.read().ok()?;
    let params = super::inference::SamplingParams {
        temperature: if (request.temperature - 0.7).abs() < f32::EPSILON {
            server_params.temperature
        } else {
            request.temperature
        },
        top_k: server_params.top_k,
        max_tokens: request.max_tokens,
    };
    drop(server_params);

    match super::inference::generate_sync(&model, &vocab, &prompt_tokens, &params) {
        Ok(result) => Some((result.text, result.finish_reason, result.token_count)),
        Err(e) => {
            eprintln!("[banco] inference error: {e}");
            None
        }
    }
}

/// Try to generate streaming tokens via inference.
/// Returns Some(vec of (text, optional_finish_reason)) on success.
pub fn try_stream_inference(
    state: &BancoState,
    request: &BancoChatRequest,
) -> Option<Vec<(String, Option<String>)>> {
    let model = state.model.quantized_model()?;
    let vocab = state.model.vocabulary();
    if vocab.is_empty() {
        return None;
    }

    let formatted = state.template_engine.apply(&request.messages);
    // Use proper BPE tokenizer when available, else greedy fallback
    let prompt_tokens = state.model.encode_text(&formatted);
    if prompt_tokens.is_empty() {
        return None;
    }

    let server_params = state.inference_params.read().ok()?;
    let params = super::inference::SamplingParams {
        temperature: if (request.temperature - 0.7).abs() < f32::EPSILON {
            server_params.temperature
        } else {
            request.temperature
        },
        top_k: server_params.top_k,
        max_tokens: request.max_tokens,
    };
    drop(server_params);

    match super::inference::generate_stream_tokens(&model, &vocab, &prompt_tokens, &params) {
        Ok(stream_tokens) => {
            let result: Vec<(String, Option<String>)> =
                stream_tokens.into_iter().map(|st| (st.text, st.finish_reason)).collect();
            Some(result)
        }
        Err(e) => {
            eprintln!("[banco] stream inference error: {e}");
            None
        }
    }
}
