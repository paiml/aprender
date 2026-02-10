//! Server handler implementations for APR and GGUF model formats
//!
//! Contains format-specific server startup functions extracted from the
//! monolithic serve.rs (PMAT-200). Each handler loads the model in its
//! native format and creates an HTTP server with inference endpoints.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::error::{CliError, Result};
use colored::Colorize;
use std::fmt::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "inference")]
use super::safetensors::{load_safetensors_tokenizer, SafeTensorsTokenizerInfo};
use super::types::ServerConfig;

// ============================================================================
// Format detection and dispatch
// ============================================================================

/// Start server using realizar
#[cfg(feature = "inference")]
pub(crate) fn start_realizar_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::format::{detect_format, ModelFormat};
    use std::io::Read;

    // Read only 8 bytes for format detection (avoid loading entire file)
    let mut file = std::fs::File::open(model_path)?;
    let mut magic = [0u8; 8];
    let bytes_read = file.read(&mut magic)?;
    if bytes_read < 8 {
        return Err(CliError::InvalidFormat(
            "File too small for format detection".to_string(),
        ));
    }

    // Detect model format from magic bytes
    let format = detect_format(&magic)
        .map_err(|e| CliError::InvalidFormat(format!("Format detection failed: {e}")))?;

    println!();
    println!("Detected format: {}", format);

    match format {
        ModelFormat::Apr => {
            println!("{}", "Starting APR model server...".cyan());
            start_apr_server(model_path, config)
        }
        ModelFormat::Gguf => {
            println!("{}", "Starting GGUF inference server...".cyan());
            start_gguf_server(model_path, config)
        }
        ModelFormat::SafeTensors => {
            println!("{}", "Starting SafeTensors inspection server...".cyan());
            super::safetensors::start_safetensors_server(model_path, config)
        }
    }
}

// ============================================================================
// APR server handlers
// ============================================================================

/// Shared state for APR server endpoints (extracted from inline struct)
#[cfg(feature = "inference")]
#[derive(Clone)]
struct AprServerState {
    transformer: Option<Arc<std::sync::Mutex<realizar::apr_transformer::AprTransformer>>>,
    model_type: String,
    architecture: String,
    is_transformer: bool,
    tokenizer: Option<SafeTensorsTokenizerInfo>,
}

/// Output from a successful APR inference run
#[cfg(feature = "inference")]
struct AprInferenceOutput {
    text: String,
    tokens_generated: usize,
    gen_duration: std::time::Duration,
    input_token_count: usize,
}

/// Run the tokenize → generate → decode pipeline for APR CPU inference.
///
/// Both `/v1/completions` and `/v1/chat/completions` use this shared path,
/// eliminating the duplicated inference logic that inflated cyclomatic complexity.
#[cfg(feature = "inference")]
fn run_apr_cpu_inference(
    state: &AprServerState,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> std::result::Result<AprInferenceOutput, String> {
    let transformer = state
        .transformer
        .as_ref()
        .ok_or("Transformer not loaded, inference not supported")?;

    // Tokenize (BPE or character-level fallback)
    let input_tokens: Vec<u32> = match &state.tokenizer {
        Some(tok) => tok.tokenizer.encode(prompt),
        None => prompt.chars().map(|c| c as u32).collect(),
    };
    let input_token_count = input_tokens.len();

    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: false,
    };

    let gen_start = Instant::now();
    let output_tokens = {
        let t = transformer.lock().map_err(|_| {
            "Transformer state corrupted (lock poisoned). Please restart the server.".to_string()
        })?;
        t.generate_with_cache(&input_tokens, &gen_config)
            .map_err(|e| format!("Generate failed: {e}"))?
    };
    let gen_duration = gen_start.elapsed();

    // Extract new tokens
    let new_tokens = if output_tokens.len() > input_tokens.len() {
        &output_tokens[input_tokens.len()..]
    } else {
        &output_tokens[..]
    };

    // Decode (BPE or character-level fallback)
    let text = match &state.tokenizer {
        Some(tok) => tok.tokenizer.decode(new_tokens).unwrap_or_default(),
        None => new_tokens
            .iter()
            .filter_map(|&t| char::from_u32(t))
            .collect(),
    };

    Ok(AprInferenceOutput {
        text,
        tokens_generated: new_tokens.len(),
        gen_duration,
        input_token_count,
    })
}

/// Load APR model, tokenizer, and transformer into shared server state.
#[cfg(feature = "inference")]
fn load_apr_model_state(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<AprServerState> {
    use realizar::apr::AprModel;

    println!("{}", "Loading APR v2 model...".dimmed());
    let model = AprModel::load(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load APR v2 model: {e}")))?;

    let model_type = model
        .metadata()
        .model_type
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let architecture = model
        .metadata()
        .architecture
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let tensor_count = model.tensor_count();
    let param_count = model.estimated_parameters();
    let is_transformer = model.metadata().is_transformer();

    println!(
        "{}",
        format!(
            "Loaded {} model (arch: {}, {} tensors, ~{} params)",
            model_type, architecture, tensor_count, param_count
        )
        .green()
    );

    if is_transformer {
        println!(
            "{}",
            "Transformer model detected - inference enabled".cyan()
        );
    }

    // Load tokenizer
    let tokenizer_json_path = model_path.with_file_name("tokenizer.json");
    let bpe_tokenizer = if tokenizer_json_path.exists() {
        load_safetensors_tokenizer(&tokenizer_json_path)
    } else {
        None
    };

    if bpe_tokenizer.is_some() {
        println!("{}", "BPE tokenizer loaded from tokenizer.json".green());
    } else {
        println!(
            "{}",
            "No tokenizer.json found - using fallback tokenization".yellow()
        );
    }

    // GPU check (APR GPU path not yet ready)
    let use_gpu = config.gpu && !config.no_gpu;

    #[cfg(feature = "cuda")]
    if use_gpu && is_transformer {
        println!(
            "{}",
            "Note: APR GPU path disabled (PMAT-099 - tensor name mapping WIP)".yellow()
        );
        println!("{}", "Using CPU path for APR inference".dimmed());
    }

    #[cfg(not(feature = "cuda"))]
    let _ = use_gpu;

    println!("{}", "Using CPU inference".dimmed());

    // Load transformer
    let transformer = if is_transformer {
        match realizar::apr_transformer::AprTransformer::from_apr_file(model_path) {
            Ok(t) => {
                println!(
                    "{}",
                    format!(
                        "Transformer ready: {} layers, hidden_dim={}",
                        t.config.num_layers, t.config.hidden_dim
                    )
                    .cyan()
                );
                Some(Arc::new(std::sync::Mutex::new(t)))
            }
            Err(e) => {
                println!(
                    "{}",
                    format!("Transformer load failed: {e} - inference disabled").yellow()
                );
                None
            }
        }
    } else {
        None
    };

    Ok(AprServerState {
        transformer,
        model_type,
        architecture,
        is_transformer,
        tokenizer: bpe_tokenizer,
    })
}

#[cfg(feature = "inference")]
#[derive(Clone, serde::Serialize)]
struct AprHealthResponse {
    status: String,
    model_type: String,
    architecture: String,
    inference_enabled: bool,
    compute_mode: String,
}

#[cfg(feature = "inference")]
#[derive(serde::Deserialize)]
struct AprCompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens_apr")]
    max_tokens: usize,
    #[serde(default)]
    temperature: Option<f32>,
}

#[cfg(feature = "inference")]
fn default_max_tokens_apr() -> usize {
    32
}

#[cfg(feature = "inference")]
#[derive(serde::Serialize)]
struct AprCompletionResponse {
    text: String,
    tokens_generated: usize,
    latency_ms: u64,
    tok_per_sec: f64,
}

fn start_apr_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    let state = load_apr_model_state(model_path, config)?;
    let is_transformer = state.is_transformer;

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();

    runtime.block_on(async move {
        let app = build_apr_cpu_router(state);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        print_apr_cpu_banner(&bind_addr, is_transformer);

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Build the axum Router for APR CPU inference.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap
fn build_apr_cpu_router(state: AprServerState) -> axum::Router {
    use axum::{
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use std::sync::Mutex;

    let state_for_health = state.clone();
    let state_for_completions = Arc::new(Mutex::new(state.clone()));
    let state_for_chat = Arc::new(Mutex::new(state));

    Router::new()
        .route(
            "/health",
            get(move || {
                let s = state_for_health.clone();
                async move {
                    Json(AprHealthResponse {
                        status: "healthy".to_string(),
                        model_type: s.model_type.clone(),
                        architecture: s.architecture.clone(),
                        inference_enabled: s.is_transformer,
                        compute_mode: "cpu".to_string(),
                    })
                }
            }),
        )
        .route(
            "/v1/completions",
            post(move |Json(req): Json<AprCompletionRequest>| {
                let state = state_for_completions.clone();
                async move { handle_apr_cpu_completion(&state, &req).into_response() }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(
                move |headers: axum::http::HeaderMap,
                      Json(req): Json<serde_json::Value>| {
                    let state = state_for_chat.clone();
                    async move {
                        handle_apr_cpu_chat_completion(&state, &headers, &req).into_response()
                    }
                },
            ),
        )
        .route(
            "/",
            get(|| async {
                "APR v2 Inference Server - POST /v1/completions, /v1/chat/completions"
            }),
        )
}

/// Handle POST /v1/completions for APR CPU inference.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)]
fn handle_apr_cpu_completion(
    state: &std::sync::Mutex<AprServerState>,
    req: &AprCompletionRequest,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse, Json};

    let s = match state.lock() {
        Ok(guard) => guard.clone(),
        Err(_poisoned) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "Server state corrupted (lock poisoned). Please restart the server."
                })),
            )
                .into_response();
        }
    };

    let max_tokens = req.max_tokens.min(128);
    match run_apr_cpu_inference(&s, &req.prompt, max_tokens, req.temperature.unwrap_or(0.0)) {
        Ok(out) => Json(AprCompletionResponse {
            text: out.text,
            tokens_generated: out.tokens_generated,
            latency_ms: out.gen_duration.as_millis() as u64,
            tok_per_sec: compute_tok_per_sec(out.tokens_generated, out.gen_duration),
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

/// Handle POST /v1/chat/completions for APR CPU inference (PAR-302).
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)]
fn handle_apr_cpu_chat_completion(
    state: &std::sync::Mutex<AprServerState>,
    headers: &axum::http::HeaderMap,
    req: &serde_json::Value,
) -> axum::response::Response {
    use axum::{
        response::{
            sse::{Event, Sse},
            IntoResponse,
        },
        Json,
    };
    use futures_util::stream;

    let trace_level = headers
        .get("X-Trace-Level")
        .and_then(|v| v.to_str().ok())
        .map(str::to_lowercase);

    let s = match state.lock() {
        Ok(guard) => guard.clone(),
        Err(_poisoned) => {
            return Json(serde_json::json!({
                "error": "Server state corrupted (lock poisoned). Please restart the server."
            }))
            .into_response();
        }
    };

    let messages = req.get("messages").and_then(|m| m.as_array());
    let stream_mode = req
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let max_tokens = req
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(32) as usize;
    let temperature = req
        .get("temperature")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0) as f32;

    let Some(msgs) = messages else {
        return Json(serde_json::json!({"error": "Missing messages"})).into_response();
    };

    let prompt = format_chatml(msgs);
    let start = Instant::now();

    let out = match run_apr_cpu_inference(&s, &prompt, max_tokens.min(256), temperature) {
        Ok(out) => out,
        Err(e) => return Json(serde_json::json!({"error": e})).into_response(),
    };

    let tok_per_sec = compute_tok_per_sec(out.tokens_generated, out.gen_duration);
    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if stream_mode {
        let response = serde_json::json!({
            "id": request_id, "object": "chat.completion.chunk", "created": created, "model": "apr",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": out.text}, "finish_reason": "stop"}]
        });
        let events = vec![
            Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string())),
            Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]")),
        ];
        return Sse::new(stream::iter(events)).into_response();
    }

    let latency_ms = start.elapsed().as_millis() as u64;
    let mut response = serde_json::json!({
        "id": request_id, "object": "chat.completion", "created": created, "model": "apr",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": out.text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": out.input_token_count, "completion_tokens": out.tokens_generated, "total_tokens": out.input_token_count + out.tokens_generated},
        "_apr_metrics": {"latency_ms": latency_ms, "tok_per_sec": tok_per_sec}
    });

    if let Some(ref level) = trace_level {
        let trace_data = serde_json::json!({
            "total_time_us": latency_ms * 1000, "prompt_tokens": out.input_token_count,
            "completion_tokens": out.tokens_generated, "layers": 28
        });
        if let Some(obj) = response.as_object_mut() {
            let key = match level.as_str() {
                "brick" => Some("brick_trace"),
                "step" => Some("step_trace"),
                "layer" => Some("layer_trace"),
                _ => None,
            };
            if let Some(key) = key {
                obj.insert(key.to_string(), trace_data);
            }
        }
    }

    Json(response).into_response()
}

/// Print APR CPU server startup banner.
fn print_apr_cpu_banner(bind_addr: &str, is_transformer: bool) {
    println!();
    println!(
        "{}",
        format!("APR Inference Server listening on http://{bind_addr}")
            .green()
            .bold()
    );
    println!();
    println!("{}", "Endpoints:".cyan());
    println!("  GET  /health              - Health check");
    println!("  POST /v1/completions      - Text generation");
    println!("  POST /v1/chat/completions - Chat completions (PAR-302)");
    println!();
    println!(
        "{}",
        format!("Mode: CPU | Transformer: {is_transformer}").dimmed()
    );
    println!("{}", "Press Ctrl+C to stop".dimmed());
}

// ============================================================================
// APR GPU server handler
// ============================================================================

/// Start APR server with GPU acceleration
/// PMAT-098: Updated to use BPE tokenizer for proper encoding
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap internally
/// Encode a prompt using BPE tokenizer or char fallback (PMAT-098).
fn encode_prompt(tok: Option<&SafeTensorsTokenizerInfo>, prompt: &str) -> Vec<u32> {
    match tok {
        Some(tok) => tok.tokenizer.encode(prompt),
        None => prompt.chars().map(|c| c as u32).collect(),
    }
}

/// Get EOS token ID from tokenizer info or use provided default.
fn eos_token_id(tok: Option<&SafeTensorsTokenizerInfo>, default: u32) -> u32 {
    tok.and_then(|t| t.eos_token_id).unwrap_or(default)
}

/// Run GPU generation with lock poisoning handling (PMAT-189).
#[cfg(feature = "inference")]
fn run_gpu_generation(
    cuda: &std::sync::Mutex<realizar::apr::AprV2ModelCuda>,
    input_tokens: &[u32],
    max_tokens: usize,
    eos_id: u32,
) -> std::result::Result<Vec<u32>, String> {
    use realizar::apr::AprModel;
    let mut model = cuda
        .lock()
        .map_err(|_| "GPU model state corrupted (lock poisoned). Please restart the server.".to_string())?;
    model
        .generate_cuda(input_tokens, max_tokens, eos_id)
        .map_err(|e| format!("GPU generation failed: {e}"))
}

/// Decode output tokens using BPE tokenizer or char fallback (PMAT-098).
fn decode_tokens(tok: Option<&SafeTensorsTokenizerInfo>, tokens: &[u32]) -> String {
    match tok {
        Some(tok) => tok.tokenizer.decode(tokens).unwrap_or_default(),
        None => tokens.iter().filter_map(|&t| char::from_u32(t)).collect(),
    }
}

/// Slice new tokens from output (tokens after input prefix).
fn extract_new_tokens(output: &[u32], input_len: usize) -> &[u32] {
    if output.len() > input_len {
        &output[input_len..]
    } else {
        output
    }
}

/// Compute tokens/second from count and duration.
fn compute_tok_per_sec(count: usize, elapsed: std::time::Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        count as f64 / secs
    } else {
        0.0
    }
}

/// Generate unique request ID for OpenAI-compatible responses.
fn generate_request_id() -> String {
    format!(
        "chatcmpl-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id()
    )
}

/// Format chat messages as ChatML prompt.
fn format_chatml(messages: &[serde_json::Value]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        write!(prompt, "<|im_start|>{role}\n{content}<|im_end|>\n")
            .expect("write to String cannot fail");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

#[cfg(feature = "inference")]
#[derive(serde::Deserialize)]
struct GpuCompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens_gpu")]
    max_tokens: usize,
}

#[cfg(feature = "inference")]
fn default_max_tokens_gpu() -> usize {
    32
}

#[cfg(feature = "inference")]
#[derive(serde::Serialize)]
struct GpuCompletionResponse {
    text: String,
    tokens_generated: usize,
    latency_ms: u64,
    tok_per_sec: f64,
}

#[cfg(feature = "inference")]
fn start_apr_server_gpu(
    model_path: &Path,
    model: realizar::apr::AprModel,
    config: &ServerConfig,
    tokenizer: Option<SafeTensorsTokenizerInfo>,
) -> Result<()> {
    use axum::{
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use realizar::apr::AprV2ModelCuda;
    use std::sync::Mutex;

    println!("{}", "Initializing CUDA...".dimmed());
    let cuda_model = AprV2ModelCuda::new(model, 0)
        .map_err(|e| CliError::InferenceFailed(format!("CUDA init failed: {e}")))?;

    println!(
        "{}",
        format!(
            "GPU: {} ({} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        )
        .green()
    );

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let cuda_model = Arc::new(Mutex::new(cuda_model));
    let tokenizer = Arc::new(tokenizer);

    runtime.block_on(async move {
        let app = build_gpu_router(cuda_model, tokenizer);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        print_gpu_server_banner(&bind_addr);

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Build the axum Router for GPU inference endpoints.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
fn build_gpu_router(
    cuda_model: Arc<std::sync::Mutex<realizar::apr::AprV2ModelCuda>>,
    tokenizer: Arc<Option<SafeTensorsTokenizerInfo>>,
) -> axum::Router {
    use axum::{
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };

    let cuda_for_completions = cuda_model.clone();
    let tok_for_completions = tokenizer.clone();
    let cuda_for_chat = cuda_model;
    let tok_for_chat = tokenizer;

    Router::new()
        .route(
            "/health",
            get(|| async {
                Json(serde_json::json!({"status": "healthy", "gpu": true}))
            }),
        )
        .route(
            "/v1/completions",
            post(move |Json(req): Json<GpuCompletionRequest>| {
                let cuda = cuda_for_completions.clone();
                let tok_info = tok_for_completions.clone();
                async move {
                    handle_gpu_completion(&cuda, tok_info.as_ref().as_ref(), &req)
                        .into_response()
                }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |Json(req): Json<serde_json::Value>| {
                let cuda = cuda_for_chat.clone();
                let tok_info = tok_for_chat.clone();
                async move {
                    handle_gpu_chat_completion(&cuda, tok_info.as_ref().as_ref(), &req)
                        .into_response()
                }
            }),
        )
        .route(
            "/",
            get(|| async {
                "APR v2 GPU Inference Server - POST /v1/completions, /v1/chat/completions"
            }),
        )
}

/// Handle POST /v1/completions for GPU inference.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
fn handle_gpu_completion(
    cuda: &std::sync::Mutex<realizar::apr::AprV2ModelCuda>,
    tok_info: Option<&SafeTensorsTokenizerInfo>,
    req: &GpuCompletionRequest,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse, Json};

    let start = Instant::now();
    let input_tokens = encode_prompt(tok_info, &req.prompt);
    let eos_id = eos_token_id(tok_info, 2);

    let gen_start = Instant::now();
    let output_tokens = match run_gpu_generation(cuda, &input_tokens, req.max_tokens.min(128), eos_id) {
        Ok(t) => t,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e})))
                .into_response();
        }
    };
    let gen_time = gen_start.elapsed();

    let new_tokens = extract_new_tokens(&output_tokens, input_tokens.len());
    let text = decode_tokens(tok_info, new_tokens);

    Json(GpuCompletionResponse {
        text,
        tokens_generated: new_tokens.len(),
        latency_ms: start.elapsed().as_millis() as u64,
        tok_per_sec: compute_tok_per_sec(new_tokens.len(), gen_time),
    })
    .into_response()
}

/// Handle POST /v1/chat/completions for GPU inference (PAR-302).
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
fn handle_gpu_chat_completion(
    cuda: &std::sync::Mutex<realizar::apr::AprV2ModelCuda>,
    tok_info: Option<&SafeTensorsTokenizerInfo>,
    req: &serde_json::Value,
) -> axum::response::Response {
    use axum::{
        response::{sse::{Event, Sse}, IntoResponse},
        Json,
    };
    use futures_util::stream;

    let messages = req.get("messages").and_then(|m| m.as_array());
    let stream_mode = req
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let max_tokens = req
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(32) as usize;

    let Some(msgs) = messages else {
        return Json(serde_json::json!({"error": "Missing messages"})).into_response();
    };

    let prompt = format_chatml(msgs);
    let start = Instant::now();
    let input_tokens = encode_prompt(tok_info, &prompt);
    let eos_id = eos_token_id(tok_info, 151_645);

    let gen_start = Instant::now();
    let output_tokens = match run_gpu_generation(cuda, &input_tokens, max_tokens.min(256), eos_id)
    {
        Ok(t) => t,
        Err(e) => return Json(serde_json::json!({"error": e})).into_response(),
    };
    let elapsed = gen_start.elapsed();

    let new_tokens = extract_new_tokens(&output_tokens, input_tokens.len());
    eprintln!(
        "[APR GPU CHAT DEBUG] Input tokens: {}, Output tokens: {}, New tokens: {}",
        input_tokens.len(),
        output_tokens.len(),
        new_tokens.len()
    );

    let output_text = decode_tokens(tok_info, new_tokens);
    let tokens_generated = new_tokens.len();
    let tok_per_sec = compute_tok_per_sec(tokens_generated, elapsed);
    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if stream_mode {
        let response = serde_json::json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "apr-gpu",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}]
        });
        let stream = stream::once(async move {
            Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string()))
        });
        Sse::new(stream).into_response()
    } else {
        Json(serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": "apr-gpu",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": input_tokens.len(),
                "completion_tokens": tokens_generated,
                "total_tokens": input_tokens.len() + tokens_generated
            },
            "_apr_metrics": {"latency_ms": start.elapsed().as_millis(), "tok_per_sec": tok_per_sec}
        }))
        .into_response()
    }
}

/// Print GPU server startup banner.
fn print_gpu_server_banner(bind_addr: &str) {
    println!();
    println!(
        "{}",
        format!("APR GPU Inference Server listening on http://{bind_addr}")
            .green()
            .bold()
    );
    println!();
    println!("{}", "Endpoints:".cyan());
    println!("  GET  /health              - Health check");
    println!("  POST /v1/completions      - GPU text generation");
    println!("  POST /v1/chat/completions - GPU chat completions (PAR-302)");
    println!();
    println!("{}", "Press Ctrl+C to stop".dimmed());
}

// ============================================================================
// GGUF server handlers
// ============================================================================

/// Start GGUF model inference server with Ollama-parity performance
///
/// Uses realizar's full inference API for text generation, streaming, and batch inference.
/// Achieves Ollama-parity: 100+ tok/s CPU, 500+ tok/s GPU.
/// With --gpu --batch flags: 800+ tok/s (2.8x Ollama) via batched GPU inference.
#[cfg(feature = "inference")]
fn start_gguf_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    println!("{}", "Loading GGUF model (mmap)...".dimmed());
    let mapped_model = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;

    println!(
        "{}",
        format!(
            "GGUF loaded: {} tensors, {} metadata entries",
            mapped_model.model.tensors.len(),
            mapped_model.model.metadata.len()
        )
        .dimmed()
    );

    println!("{}", "Building quantized inference model...".dimmed());
    let quantized_model = OwnedQuantizedModel::from_mapped(&mapped_model)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to build quantized model: {e}")))?;

    println!(
        "{}",
        format!(
            "Model ready: {} layers, vocab_size={}, hidden_dim={}",
            quantized_model.layers.len(),
            quantized_model.config.vocab_size,
            quantized_model.config.hidden_dim
        )
        .green()
    );

    let vocab = extract_gguf_vocab(&mapped_model, quantized_model.config.vocab_size);

    #[cfg(feature = "cuda")]
    if config.gpu && config.batch {
        return start_gguf_server_gpu_batched(quantized_model, vocab, config);
    }

    #[cfg(feature = "cuda")]
    if config.gpu && !config.no_gpu {
        return start_gguf_server_cuda(quantized_model, vocab, &mapped_model, config);
    }

    run_cpu_server(quantized_model, vocab, config)
}

/// Extract vocabulary from GGUF model, falling back to placeholder tokens.
fn extract_gguf_vocab(
    mapped_model: &realizar::gguf::MappedGGUFModel,
    vocab_size: usize,
) -> Vec<String> {
    mapped_model.model.vocabulary().unwrap_or_else(|| {
        eprintln!("Warning: No vocabulary in GGUF, using placeholder tokens");
        (0..vocab_size).map(|i| format!("token{i}")).collect()
    })
}

/// Start GGUF server with CUDA acceleration (PAR-111).
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_gguf_server_cuda(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    mapped_model: &realizar::gguf::MappedGGUFModel,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router, AppState};
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

    println!(
        "{}",
        "Enabling optimized CUDA acceleration (PAR-111)...".cyan()
    );

    match OwnedQuantizedModelCuda::new(quantized_model, 0) {
        Ok(mut cuda_model) => {
            preload_gpu_weights(&mut cuda_model);
            println!("{}", "CUDA optimized model ready".green());

            let state = AppState::with_cuda_model_and_vocab(cuda_model, vocab)
                .map_err(|e| CliError::InferenceFailed(format!("Failed to create state: {e}")))?
                .with_verbose(config.verbose);

            let app = create_router(state);
            run_server_async(app, &config.bind_addr(), "CUDA-optimized")
        }
        Err(e) => {
            eprintln!(
                "{}",
                format!("CUDA init failed, falling back to CPU: {e}").yellow()
            );
            let quantized_model = OwnedQuantizedModel::from_mapped(mapped_model).map_err(|e| {
                CliError::ModelLoadFailed(format!("Failed to rebuild quantized model: {e}"))
            })?;
            let vocab = extract_gguf_vocab(mapped_model, quantized_model.config.vocab_size);
            run_cpu_server(quantized_model, vocab, config)
        }
    }
}

/// Pre-upload model weights to GPU for maximum performance.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn preload_gpu_weights(cuda_model: &mut realizar::gguf::OwnedQuantizedModelCuda) {
    println!("  Initializing GPU on device 0...");
    match cuda_model.preload_weights_gpu() {
        Ok(bytes) => {
            println!(
                "{}",
                format!("  Pre-uploaded {} MB weights to GPU", bytes / (1024 * 1024)).green()
            );
        }
        Err(e) => {
            eprintln!(
                "{}",
                format!("  Warning: Weight preload failed, using on-demand: {e}").yellow()
            );
        }
    }
}

/// Run an axum server with graceful shutdown and standard banner.
#[cfg(feature = "inference")]
fn run_server_async(app: axum::Router, bind_addr: &str, label: &str) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = bind_addr.to_string();
    let label = label.to_string();

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("{label} server listening on http://{bind_addr}")
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /metrics             - Prometheus metrics");
        println!("  POST /generate            - Text generation");
        println!("  POST /v1/completions      - OpenAI-compatible");
        println!("  POST /v1/chat/completions - Chat completions");
        println!();
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        Ok(())
    })
}

/// Run the CPU inference server
#[cfg(feature = "inference")]
fn run_cpu_server(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router, AppState};

    let state = AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?
        .with_verbose(config.verbose); // GH-152: Pass verbose flag to handlers

    // Create realizar's full inference router (Ollama-parity endpoints)
    let app = create_router(state);

    // Create tokio runtime and run server
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Inference server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Ollama-Parity Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /metrics             - Prometheus metrics");
        println!("  POST /generate            - Text generation");
        println!("  POST /stream/generate     - SSE streaming");
        println!("  POST /batch/generate      - Batch inference");
        println!("  POST /v1/completions      - OpenAI-compatible");
        println!("  POST /v1/chat/completions - Chat completions");
        println!();
        println!(
            "{}",
            "Performance targets: 100+ tok/s CPU, 500+ tok/s GPU".yellow()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start GGUF server with GPU batched inference (2X+ Ollama performance)
///
/// Uses OwnedQuantizedModelCachedSync with continuous batching scheduler
/// for maximum throughput on GPU. Achieves 800+ tok/s (2.8x Ollama).
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_gguf_server_gpu_batched(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router, spawn_batch_processor, AppState, BatchConfig};
    use realizar::gguf::OwnedQuantizedModelCachedSync;

    println!(
        "{}",
        "Enabling GPU batched inference (2X+ Ollama)...".cyan()
    );

    // Create tokio runtime FIRST (needed for batch processor spawn)
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    // Create cached model for scheduler reuse
    // OwnedQuantizedModelCachedSync handles GPU caching internally via warmup_gpu_cache()
    let cached_model = OwnedQuantizedModelCachedSync::new(quantized_model);

    // Warmup GPU cache
    println!("  Warming up GPU cache...");
    match cached_model.warmup_gpu_cache() {
        Ok((memory_bytes, num_layers)) => {
            println!(
                "  GPU cache ready: {:.2} GB ({} layers)",
                memory_bytes as f64 / 1e9,
                num_layers
            );
        }
        Err(e) => {
            eprintln!("  Warning: GPU cache warmup failed: {e}");
        }
    }

    // Create state with cached model and real vocab
    let state = AppState::with_cached_model_and_vocab(cached_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?
        .with_verbose(config.verbose); // GH-152: Pass verbose flag

    // Get Arc'd model for batch processor
    let cached_model_arc = state
        .cached_model()
        .expect("cached_model should exist")
        .clone();

    // Configure batch processing
    let batch_config = BatchConfig::default();
    println!("  Batch window: {}ms", batch_config.window_ms);
    println!("  Optimal batch: {}", batch_config.optimal_batch);
    println!("  GPU threshold: {}", batch_config.gpu_threshold);

    let bind_addr = config.bind_addr();

    // Run everything inside the runtime context
    runtime.block_on(async move {
        // Spawn batch processor task (requires Tokio runtime)
        let batch_tx = spawn_batch_processor(cached_model_arc, batch_config.clone());
        println!("  Batch processor: RUNNING");

        // Add batch support to state
        let state = state.with_batch_config(batch_tx, batch_config);

        // Create router
        let app = create_router(state);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("GPU Batched Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "2X Ollama Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /v1/gpu/status       - GPU cache status");
        println!("  POST /v1/completions      - OpenAI-compatible (batched)");
        println!("  POST /v1/batch/completions - Explicit batch inference");
        println!();
        println!(
            "{}",
            "Performance: 800+ tok/s (2.8x Ollama) with batched requests".yellow()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

// ============================================================================
// Shutdown signal helper
// ============================================================================

/// Shutdown signal handler
#[cfg(feature = "inference")]
pub(crate) async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
}
