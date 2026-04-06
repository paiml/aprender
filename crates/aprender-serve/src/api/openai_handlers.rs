//! OpenAI-compatible API handlers
//!
//! Extracted from api/mod.rs (PMAT-802) to reduce module size.
//! Contains chat completion, streaming, and model list handlers.
#![allow(unreachable_pub)] // Items re-exported as pub from api/mod.rs

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::stream::Stream;

use super::{
    build_trace_data, clean_chat_output, format_chat_messages, AppState, ChatChoice,
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ErrorResponse,
    OpenAIModel, OpenAIModelsResponse, Usage,
};
use crate::generate::{GenerationConfig, SamplingStrategy};
use crate::tokenizer::BPETokenizer;

// ============================================================================
// Shared helpers — eliminate duplication across backend paths
// ============================================================================

/// Record failure and return an error response.
fn fail_response(state: &AppState, status: StatusCode, msg: impl std::fmt::Display) -> Response {
    state.metrics.record_failure();
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
        .into_response()
}

/// Current Unix timestamp.
fn unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Get tokenizer from state or return 500.
#[allow(clippy::result_large_err)]
fn require_tokenizer(state: &AppState) -> Result<Arc<BPETokenizer>, Response> {
    state.tokenizer.clone().ok_or_else(|| {
        fail_response(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "No tokenizer available",
        )
    })
}

/// Format chat messages, tokenize, validate non-empty.
#[allow(clippy::result_large_err)]
fn tokenize_chat_prompt(
    tokenizer: &BPETokenizer,
    messages: &[ChatMessage],
    model_hint: Option<&str>,
    state: &AppState,
) -> Result<Vec<u32>, Response> {
    let prompt_text = format_chat_messages(messages, model_hint);
    let ids = tokenizer.encode(&prompt_text);
    if ids.is_empty() {
        return Err(fail_response(
            state,
            StatusCode::BAD_REQUEST,
            "Messages cannot be empty",
        ));
    }
    Ok(ids)
}

/// Extract common generation parameters from the request.
///
/// GH-330: EOS resolution follows Design by Contract priority:
/// 1. Model config (class invariant from GGUF/APR metadata)
/// 2. Tokenizer vocabulary lookup (runtime fallback)
///
/// No hardcoded magic numbers.
fn chat_gen_params(
    request: &ChatCompletionRequest,
    tokenizer: &BPETokenizer,
    model_eos: Option<u32>,
) -> (usize, f32, u32) {
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7);
    // GH-330: Model config EOS first, then tokenizer lookup, then 0 (disabled)
    let eos_token_id = model_eos
        .or_else(|| tokenizer.get_token_id("<|im_end|>"))
        .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
        .unwrap_or(0);
    (max_tokens, temperature, eos_token_id)
}

/// Build a non-streaming ChatCompletionResponse.
fn build_chat_response(
    request_id: String,
    model: String,
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    max_tokens: usize,
    trace_level: Option<&str>,
    latency: Duration,
) -> Response {
    let (brick_trace, step_trace, layer_trace) = build_trace_data(
        trace_level,
        latency.as_micros() as u64,
        prompt_tokens,
        completion_tokens,
        28,
    );
    Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
                name: None,
            },
            finish_reason: if completion_tokens >= max_tokens {
                "length".to_string()
            } else {
                "stop".to_string()
            },
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        brick_trace,
        step_trace,
        layer_trace,
    })
    .into_response()
}

/// Serialize a value to an SSE event, returning `None` if serialization fails.
fn sse_event(value: &impl serde::Serialize) -> Option<Result<Event, Infallible>> {
    serde_json::to_string(value)
        .ok()
        .map(|data| Ok(Event::default().data(data)))
}

/// Decode a token and optionally clean the output, returning the text if non-empty.
fn decode_token(tokenizer: &BPETokenizer, token_id: u32, clean: bool) -> Option<String> {
    let text = tokenizer.decode(&[token_id]).ok()?;
    let text = if clean {
        clean_chat_output(&text)
    } else {
        text
    };
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Build a pre-generated SSE streaming response (all tokens already generated).
fn pregenerated_sse_response(
    token_ids: Vec<u32>,
    tokenizer: Arc<BPETokenizer>,
    request_id: String,
    model_name: String,
    clean: bool,
) -> Response {
    let stream = async_stream::stream! {
        if let Some(evt) = sse_event(&ChatCompletionChunk::initial(&request_id, &model_name)) {
            yield evt;
        }

        for &token_id in &token_ids {
            if let Some(text) = decode_token(&tokenizer, token_id, clean) {
                let chunk = ChatCompletionChunk::content(&request_id, &model_name, &text);
                if let Some(evt) = sse_event(&chunk) {
                    yield evt;
                }
            }
        }

        if let Some(evt) = sse_event(&ChatCompletionChunk::done(&request_id, &model_name)) {
            yield evt;
        }
        yield Ok::<_, Infallible>(Event::default().data("[DONE]".to_string()));
    };
    Sse::new(stream).into_response()
}

/// Build a true-streaming SSE response with keep-alive (tokens arrive via channel).
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
fn true_streaming_sse_response(
    rx: tokio::sync::mpsc::Receiver<Result<u32, String>>,
    tokenizer: Arc<BPETokenizer>,
    request_id: String,
    model_name: String,
    metrics: Arc<crate::metrics::MetricsCollector>,
    start: Instant,
    clean: bool,
) -> Response {
    use tokio_stream::wrappers::ReceiverStream;
    use tokio_stream::StreamExt;

    let token_stream = ReceiverStream::new(rx);
    let mut completion_tokens = 0usize;

    let stream = async_stream::stream! {
        if let Some(evt) = sse_event(&ChatCompletionChunk::initial(&request_id, &model_name)) {
            yield evt;
        }

        tokio::pin!(token_stream);
        while let Some(result) = token_stream.next().await {
            match result {
                Ok(token_id) => {
                    completion_tokens += 1;
                    if let Some(text) = decode_token(&tokenizer, token_id, clean) {
                        let chunk = ChatCompletionChunk::content(&request_id, &model_name, &text);
                        if let Some(evt) = sse_event(&chunk) {
                            yield evt;
                        }
                    }
                }
                Err(e) => {
                    if let Some(evt) = sse_event(&serde_json::json!({ "error": e })) {
                        yield evt;
                    }
                    break;
                }
            }
        }

        if let Some(evt) = sse_event(&ChatCompletionChunk::done(&request_id, &model_name)) {
            yield evt;
        }

        metrics.record_success(completion_tokens, start.elapsed());
        yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

// ============================================================================
// Backend dispatch — each returns Some(response) if handled, None to fallthrough
// ============================================================================

/// GPU (non-batched) backend.
#[cfg(feature = "gpu")]
fn try_gpu_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
) -> Option<Response> {
    use crate::gpu::GpuGenerateConfig;

    let gpu_model_lock = state.gpu_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    // GH-319: Use actual model architecture for chat template detection
    let arch_hint = state.model_architecture();
    let prompt_ids =
        match tokenize_chat_prompt(&tokenizer, &request.messages, arch_hint.as_deref(), state) {
            Ok(ids) => ids,
            Err(r) => return Some(r),
        };
    let prompt_tokens = prompt_ids.len();
    let prompt_usize: Vec<usize> = prompt_ids.iter().map(|&x| x as usize).collect();
    let (max_tokens, temperature, eos_token_id) =
        chat_gen_params(request, &tokenizer, state.model_eos_token_id());

    let gpu_config = GpuGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        stop_tokens: vec![eos_token_id as usize],
        trace: state.should_trace(trace_level),
    };

    let mut model = match gpu_model_lock.write() {
        Ok(m) => m,
        Err(e) => {
            return Some(fail_response(
                state,
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("GPU model lock error: {e}"),
            ));
        },
    };
    let generated = match model.generate(&prompt_usize, &gpu_config) {
        Ok(g) => g,
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let token_ids: Vec<u32> = generated
        .iter()
        .skip(prompt_tokens)
        .map(|&x| x as u32)
        .collect();
    let completion_tokens = token_ids.len();

    if request.stream {
        state
            .metrics
            .record_success(completion_tokens, start.elapsed());
        return Some(pregenerated_sse_response(
            token_ids,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            false,
        ));
    }

    let text = match tokenizer.decode(&token_ids) {
        Ok(t) => clean_chat_output(&t),
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let latency = start.elapsed();
    state.metrics.record_success(completion_tokens, latency);
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        trace_level,
        latency,
    ))
}

/// Cached model (GPU batched) backend.
#[cfg(feature = "gpu")]
fn try_cached_backend(
    state: &AppState,
    request: &ChatCompletionRequest,
    request_id: &str,
    trace_level: Option<&str>,
    start: Instant,
) -> Option<Response> {
    use crate::gguf::QuantizedGenerateConfig;

    let cached_model = state.cached_model()?;
    let tokenizer = match require_tokenizer(state) {
        Ok(t) => t,
        Err(r) => return Some(r),
    };
    // GH-319: Use actual model architecture for chat template detection
    let arch_hint = state.model_architecture();
    let prompt_ids =
        match tokenize_chat_prompt(&tokenizer, &request.messages, arch_hint.as_deref(), state) {
            Ok(ids) => ids,
            Err(r) => return Some(r),
        };
    let prompt_tokens = prompt_ids.len();
    let (max_tokens, temperature, eos_token_id) =
        chat_gen_params(request, &tokenizer, state.model_eos_token_id());

    let q_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        stop_tokens: vec![eos_token_id],
        trace: state.should_trace(trace_level),
        ..Default::default()
    };

    let generated = match cached_model
        .model()
        .generate_with_cache(&prompt_ids, &q_config)
    {
        Ok(g) => g,
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let token_ids: Vec<u32> = generated.iter().skip(prompt_tokens).copied().collect();
    let completion_tokens = token_ids.len();

    if request.stream {
        state
            .metrics
            .record_success(completion_tokens, start.elapsed());
        return Some(pregenerated_sse_response(
            token_ids,
            tokenizer,
            request_id.to_string(),
            request.model.clone(),
            false,
        ));
    }

    let text = match tokenizer.decode(&token_ids) {
        Ok(t) => clean_chat_output(&t),
        Err(e) => return Some(fail_response(state, StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let latency = start.elapsed();
    state.metrics.record_success(completion_tokens, latency);
    Some(build_chat_response(
        request_id.to_string(),
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
        trace_level,
        latency,
    ))
}

include!("cuda_chat_backend.rs");
include!("chat_completions_stream.rs");
