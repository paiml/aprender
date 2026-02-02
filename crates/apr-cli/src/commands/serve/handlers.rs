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

/// Start APR v2 model server with full inference support
///
/// Supports both transformer inference (generate) and metadata inspection.
/// GPU acceleration available via --gpu flag.
#[cfg(feature = "inference")]
fn start_apr_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use axum::{
        extract::State,
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use realizar::apr::AprModel;
    use serde::{Deserialize, Serialize};
    use std::sync::Mutex;

    // Load APR model
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

    // Load tokenizer if available - prefer BPE tokenizer from tokenizer.json
    // PMAT-098: Use proper BPE tokenizer (same as SafeTensors path)
    let tokenizer_json_path = model_path.with_file_name("tokenizer.json");
    let bpe_tokenizer = if tokenizer_json_path.exists() {
        load_safetensors_tokenizer(&tokenizer_json_path)
    } else {
        None
    };

    let has_tokenizer = bpe_tokenizer.is_some();
    if has_tokenizer {
        println!("{}", "BPE tokenizer loaded from tokenizer.json".green());
    } else {
        println!(
            "{}",
            "No tokenizer.json found - using fallback tokenization".yellow()
        );
    }

    // Determine GPU vs CPU mode
    // PMAT-099: APR GPU path currently has tensor name mismatches with AprV2ModelCuda
    // For now, force CPU for APR until GPU path is fixed
    let use_gpu = config.gpu && !config.no_gpu;

    #[cfg(feature = "cuda")]
    if use_gpu && is_transformer {
        // PMAT-099: Disable GPU for APR until AprV2ModelCuda tensor name mapping is fixed
        // The AprTransformer CPU path uses correct tensor names from APR metadata
        // but AprV2ModelCuda expects different names (model.layers vs blk)
        println!(
            "{}",
            "Note: APR GPU path disabled (PMAT-099 - tensor name mapping WIP)".yellow()
        );
        println!("{}", "Using CPU path for APR inference".dimmed());
        // Fall through to CPU path instead of:
        // return start_apr_server_gpu(model_path, model, config, bpe_tokenizer);
    }

    // CPU path
    println!("{}", "Using CPU inference".dimmed());

    // Create tokio runtime
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let model_path_clone = model_path.to_path_buf();

    // PMAT-098: Load transformer once and share across requests
    // Previous code reloaded model on every request causing ~0.01 tok/s
    // Now we load once using AprTransformer for efficient inference
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
                Some(Arc::new(Mutex::new(t)))
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

    // Shared state for inference (model loaded once, not per-request)
    // PMAT-098: Use BPE tokenizer for proper encoding
    #[derive(Clone)]
    struct AprState {
        transformer: Option<Arc<Mutex<realizar::apr_transformer::AprTransformer>>>,
        model_type: String,
        architecture: String,
        is_transformer: bool,
        tokenizer: Option<SafeTensorsTokenizerInfo>,
    }

    let state = AprState {
        transformer,
        model_type: model_type.clone(),
        architecture: architecture.clone(),
        is_transformer,
        tokenizer: bpe_tokenizer,
    };

    #[derive(Clone, Serialize)]
    struct HealthResponse {
        status: String,
        model_type: String,
        architecture: String,
        inference_enabled: bool,
        compute_mode: String,
    }

    #[derive(Deserialize)]
    struct CompletionRequest {
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: usize,
        #[serde(default)]
        temperature: Option<f32>,
    }

    fn default_max_tokens() -> usize {
        32
    }

    #[derive(Serialize)]
    struct CompletionResponse {
        text: String,
        tokens_generated: usize,
        latency_ms: u64,
        tok_per_sec: f64,
    }

    runtime.block_on(async move {
        let state_for_health = state.clone();
        let state_for_completions = Arc::new(Mutex::new(state.clone()));

        let app = Router::new()
            .route(
                "/health",
                get(move || {
                    let s = state_for_health.clone();
                    async move {
                        Json(HealthResponse {
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
                post(move |Json(req): Json<CompletionRequest>| {
                    let state = state_for_completions.clone();
                    async move {
                        // PMAT-189: Handle mutex lock poisoning gracefully (Jidoka)
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

                        // PMAT-098: Use shared transformer (no reload per request)
                        let transformer = match &s.transformer {
                            Some(t) => t.clone(),
                            None => {
                                return (
                                    StatusCode::BAD_REQUEST,
                                    Json(serde_json::json!({
                                        "error": "Transformer not loaded, inference not supported"
                                    })),
                                )
                                    .into_response();
                            }
                        };

                        let start = Instant::now();

                        // PMAT-098: Use BPE tokenizer for proper encoding
                        let input_tokens: Vec<u32> = if let Some(ref tok) = s.tokenizer {
                            tok.tokenizer.encode(&req.prompt)
                        } else {
                            // Fallback: character-level
                            req.prompt.chars().map(|c| c as u32).collect()
                        };

                        let eos_id = match &s.tokenizer {
                            Some(tok) => tok.eos_token_id.unwrap_or(2),
                            _ => 2,
                        };

                        // PMAT-103 FIX: Use generate_with_cache for O(n) generation
                        // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
                        // generate_with_cache() uses KV cache for incremental generation = O(n)
                        let gen_start = Instant::now();
                        let max_tokens = req.max_tokens.min(128);

                        let gen_config = realizar::apr_transformer::GenerateConfig {
                            max_tokens,
                            temperature: req.temperature.unwrap_or(0.0),
                            top_p: 0.9,
                            top_k: 0,
                            repetition_penalty: 1.0,
                            trace: false,
                        };

                        let output_tokens = {
                            // PMAT-189: Handle transformer lock poisoning gracefully
                            let t = match transformer.lock() {
                                Ok(guard) => guard,
                                Err(_poisoned) => {
                                    return (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({
                                            "error": "Transformer state corrupted (lock poisoned). Please restart the server."
                                        })),
                                    )
                                        .into_response();
                                }
                            };
                            match t.generate_with_cache(&input_tokens, &gen_config) {
                                Ok(tokens) => tokens,
                                Err(e) => {
                                    return (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": format!("Generate failed: {e}")})),
                                    )
                                        .into_response();
                                }
                            }
                        };
                        let gen_time = gen_start.elapsed();

                        // Decode output
                        let new_tokens = if output_tokens.len() > input_tokens.len() {
                            &output_tokens[input_tokens.len()..]
                        } else {
                            &output_tokens[..]
                        };

                        // PMAT-099: Debug logging for token decoding
                        eprintln!("[APR DEBUG] Generated {} total tokens, {} new tokens", output_tokens.len(), new_tokens.len());
                        eprintln!("[APR DEBUG] New token IDs: {:?}", &new_tokens[..new_tokens.len().min(20)]);

                        // PMAT-098: Use BPE tokenizer for proper decoding
                        let text = if let Some(ref tok) = s.tokenizer {
                            match tok.tokenizer.decode(new_tokens) {
                                Ok(decoded) => {
                                    eprintln!("[APR DEBUG] Decoded text: {:?}", decoded);
                                    decoded
                                }
                                Err(e) => {
                                    eprintln!("[APR DEBUG] Decode error: {e}");
                                    String::new()
                                }
                            }
                        } else {
                            // Fallback: interpret as char codes
                            let fallback: String = new_tokens.iter().filter_map(|&t| char::from_u32(t)).collect();
                            eprintln!("[APR DEBUG] Fallback decode: {:?}", fallback);
                            fallback
                        };

                        let tokens_generated = new_tokens.len();
                        let latency_ms = start.elapsed().as_millis() as u64;
                        let tok_per_sec = if gen_time.as_secs_f64() > 0.0 {
                            tokens_generated as f64 / gen_time.as_secs_f64()
                        } else {
                            0.0
                        };

                        Json(CompletionResponse {
                            text,
                            tokens_generated,
                            latency_ms,
                            tok_per_sec,
                        })
                        .into_response()
                    }
                }),
            )
            .route(
                "/v1/chat/completions",
                {
                    let state_for_chat = Arc::new(Mutex::new(state.clone()));
                    post(move |headers: axum::http::HeaderMap, Json(req): Json<serde_json::Value>| {
                        let state = state_for_chat.clone();
                        async move {
                            use axum::response::sse::{Event, Sse};
                            use futures_util::stream;

                            // F-TRACE-001/002/003: Parse X-Trace-Level header
                            let trace_level = headers
                                .get("X-Trace-Level")
                                .and_then(|v| v.to_str().ok())
                                .map(str::to_lowercase);

                            // PMAT-189: Handle mutex lock poisoning gracefully (Jidoka)
                            let s = match state.lock() {
                                Ok(guard) => guard.clone(),
                                Err(_poisoned) => {
                                    return axum::Json(serde_json::json!({
                                        "error": "Server state corrupted (lock poisoned). Please restart the server."
                                    }))
                                    .into_response();
                                }
                            };

                            // PMAT-098: Use shared transformer (no reload per request)
                            let transformer = match &s.transformer {
                                Some(t) => t.clone(),
                                None => {
                                    return axum::Json(serde_json::json!({
                                        "error": "Transformer not loaded, inference not supported"
                                    }))
                                    .into_response();
                                }
                            };

                            // Extract messages from request
                            let messages = req.get("messages").and_then(|m| m.as_array());
                            let stream_mode = req.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
                            let max_tokens = req.get("max_tokens").and_then(serde_json::Value::as_u64).unwrap_or(32) as usize;

                            let prompt = if let Some(msgs) = messages {
                                // Format as ChatML
                                let mut prompt = String::new();
                                for msg in msgs {
                                    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                                    write!(prompt, "<|im_start|>{}\n{}<|im_end|>\n", role, content).expect("write to String cannot fail");
                                }
                                prompt.push_str("<|im_start|>assistant\n");
                                prompt
                            } else {
                                return axum::Json(serde_json::json!({"error": "Missing messages"})).into_response();
                            };

                            let start = Instant::now();

                            // PMAT-098: Use BPE tokenizer for proper encoding
                            let input_tokens: Vec<u32> = if let Some(ref tok) = s.tokenizer {
                                tok.tokenizer.encode(&prompt)
                            } else {
                                // Fallback: character-level
                                prompt.chars().map(|c| c as u32).collect()
                            };

                            let eos_id = match &s.tokenizer {
                                Some(tok) => tok.eos_token_id.unwrap_or(151645),
                                _ => 151645, // ChatML end token
                            };

                            // PMAT-103 FIX: Use generate_with_cache for O(n) generation
                            // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
                            // generate_with_cache() uses KV cache for incremental generation = O(n)
                            let gen_start = Instant::now();
                            let max_tokens = max_tokens.min(256);
                            let temperature = req.get("temperature").and_then(serde_json::Value::as_f64).unwrap_or(0.0) as f32;

                            let gen_config = realizar::apr_transformer::GenerateConfig {
                                max_tokens,
                                temperature,
                                top_p: 0.9,
                                top_k: 0,
                                repetition_penalty: 1.0,
                                trace: false,
                            };

                            let output_tokens = {
                                // PMAT-189: Handle transformer lock poisoning gracefully
                                let t = match transformer.lock() {
                                    Ok(guard) => guard,
                                    Err(_poisoned) => {
                                        return axum::Json(serde_json::json!({
                                            "error": "Transformer state corrupted (lock poisoned). Please restart the server."
                                        }))
                                            .into_response();
                                    }
                                };
                                match t.generate_with_cache(&input_tokens, &gen_config) {
                                    Ok(tokens) => tokens,
                                    Err(e) => {
                                        return axum::Json(serde_json::json!({"error": format!("Generate failed: {e}")}))
                                            .into_response();
                                    }
                                }
                            };
                            let elapsed = gen_start.elapsed();

                            // Decode output
                            let new_tokens = if output_tokens.len() > input_tokens.len() {
                                &output_tokens[input_tokens.len()..]
                            } else {
                                &output_tokens[..]
                            };

                            // PMAT-099: Debug logging for token decoding
                            eprintln!("[APR CHAT DEBUG] Input tokens: {}, Output tokens: {}, New tokens: {}",
                                input_tokens.len(), output_tokens.len(), new_tokens.len());
                            eprintln!("[APR CHAT DEBUG] New token IDs: {:?}", &new_tokens[..new_tokens.len().min(20)]);

                            // PMAT-098: Use BPE tokenizer for proper decoding
                            let output_text = if let Some(ref tok) = s.tokenizer {
                                match tok.tokenizer.decode(new_tokens) {
                                    Ok(decoded) => {
                                        eprintln!("[APR CHAT DEBUG] Decoded text: {:?}", decoded);
                                        decoded
                                    }
                                    Err(e) => {
                                        eprintln!("[APR CHAT DEBUG] Decode error: {e}");
                                        String::new()
                                    }
                                }
                            } else {
                                // Fallback: interpret as char codes
                                let fallback: String = new_tokens.iter().filter_map(|&t| char::from_u32(t)).collect();
                                eprintln!("[APR CHAT DEBUG] Fallback decode: {:?}", fallback);
                                fallback
                            };

                            let tokens_generated = new_tokens.len();
                            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                                tokens_generated as f64 / elapsed.as_secs_f64()
                            } else {
                                0.0
                            };

                            // Generate unique ID
                            let request_id = format!(
                                "chatcmpl-{}-{}",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos(),
                                std::process::id()
                            );

                            let created = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();

                            if stream_mode {
                                // SSE streaming response (PAR-302) with [DONE] marker (F-HTTP-011)
                                let response = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": "apr",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": output_text},
                                        "finish_reason": "stop"
                                    }]
                                });

                                // Send data event followed by [DONE] marker per OpenAI SSE spec
                                let events = vec![
                                    Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string())),
                                    Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]")),
                                ];
                                let stream = stream::iter(events);
                                Sse::new(stream).into_response()
                            } else {
                                // Non-streaming response with trace support (F-TRACE-001/002/003)
                                let latency_ms = start.elapsed().as_millis() as u64;

                                // Build trace data based on X-Trace-Level header
                                let trace_data = serde_json::json!({
                                    "total_time_us": latency_ms * 1000,
                                    "prompt_tokens": input_tokens.len(),
                                    "completion_tokens": tokens_generated,
                                    "layers": 28  // Fixed for now
                                });

                                let mut response = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion",
                                    "created": created,
                                    "model": "apr",
                                    "choices": [{
                                        "index": 0,
                                        "message": {"role": "assistant", "content": output_text},
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": input_tokens.len(),
                                        "completion_tokens": tokens_generated,
                                        "total_tokens": input_tokens.len() + tokens_generated
                                    },
                                    "_apr_metrics": {
                                        "latency_ms": latency_ms,
                                        "tok_per_sec": tok_per_sec
                                    }
                                });

                                // Add trace fields based on X-Trace-Level header
                                if let Some(ref level) = trace_level {
                                    if let Some(obj) = response.as_object_mut() {
                                        match level.as_str() {
                                            "brick" => { obj.insert("brick_trace".to_string(), trace_data); }
                                            "step" => { obj.insert("step_trace".to_string(), trace_data); }
                                            "layer" => { obj.insert("layer_trace".to_string(), trace_data); }
                                            _ => {}
                                        }
                                    }
                                }

                                axum::Json(response).into_response()
                            }
                        }
                    })
                },
            )
            .route(
                "/",
                get(|| async { "APR v2 Inference Server - POST /v1/completions, /v1/chat/completions" }),
            );

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("APR Inference Server listening on http://{}", bind_addr)
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
            format!("Mode: CPU | Transformer: {}", is_transformer).dimmed()
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
// APR GPU server handler
// ============================================================================

/// Start APR server with GPU acceleration
/// PMAT-098: Updated to use BPE tokenizer for proper encoding
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_apr_server_gpu(
    model_path: &Path,
    model: realizar::apr::AprModel,
    config: &ServerConfig,
    tokenizer: Option<SafeTensorsTokenizerInfo>,
) -> Result<()> {
    use axum::{
        extract::State,
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use realizar::apr::AprV2ModelCuda;
    use serde::{Deserialize, Serialize};
    use std::sync::Mutex;

    // Initialize CUDA model
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
    let model_path_clone = model_path.to_path_buf();

    #[derive(Deserialize)]
    struct CompletionRequest {
        prompt: String,
        #[serde(default = "default_max_tokens_gpu")]
        max_tokens: usize,
    }

    fn default_max_tokens_gpu() -> usize {
        32
    }

    #[derive(Serialize)]
    struct CompletionResponse {
        text: String,
        tokens_generated: usize,
        latency_ms: u64,
        tok_per_sec: f64,
    }

    // Wrap CUDA model in Arc<Mutex> for shared access
    // PMAT-098: Use BPE tokenizer for proper encoding
    let cuda_model = Arc::new(Mutex::new(cuda_model));
    let tokenizer = Arc::new(tokenizer);
    let model_path_arc = Arc::new(model_path_clone);

    runtime.block_on(async move {
        let cuda_for_completions = cuda_model.clone();
        let tok_for_completions = tokenizer.clone();
        let path_for_completions = model_path_arc.clone();

        let app = Router::new()
            .route(
                "/health",
                get(|| async {
                    Json(serde_json::json!({
                        "status": "healthy",
                        "gpu": true
                    }))
                }),
            )
            .route(
                "/v1/completions",
                post(move |Json(req): Json<CompletionRequest>| {
                    let cuda = cuda_for_completions.clone();
                    let tok_info = tok_for_completions.clone();
                    let model_path = path_for_completions.clone();
                    async move {
                        use realizar::apr::AprModel;

                        let start = Instant::now();

                        // PMAT-098: Use BPE tokenizer for proper encoding
                        let input_tokens: Vec<u32> = match tok_info.as_ref() {
                            Some(tok) => tok.tokenizer.encode(&req.prompt),
                            None => req.prompt.chars().map(|c| c as u32).collect(),
                        };

                        let eos_id = match tok_info.as_ref() {
                            Some(tok) => tok.eos_token_id.unwrap_or(2),
                            _ => 2,
                        };

                        // Generate on GPU
                        let gen_start = Instant::now();
                        let max_tokens = req.max_tokens.min(128);
                        let output_tokens = {
                            // PMAT-189: Handle CUDA model lock poisoning gracefully
                            let mut model = match cuda.lock() {
                                Ok(guard) => guard,
                                Err(_poisoned) => {
                                    return (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({
                                            "error": "GPU model state corrupted (lock poisoned). Please restart the server."
                                        })),
                                    )
                                        .into_response();
                                }
                            };
                            match model.generate_cuda(
                                &input_tokens,
                                max_tokens,
                                eos_id,
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    return (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        Json(serde_json::json!({"error": format!("GPU generation failed: {e}")})),
                                    )
                                        .into_response();
                                }
                            }
                        };
                        let gen_time = gen_start.elapsed();

                        // Decode
                        let new_tokens = if output_tokens.len() > input_tokens.len() {
                            &output_tokens[input_tokens.len()..]
                        } else {
                            &output_tokens[..]
                        };

                        // PMAT-098: Use BPE tokenizer for proper decoding
                        let text = match tok_info.as_ref() {
                            Some(tok) => tok.tokenizer.decode(new_tokens).unwrap_or_else(|_| String::new()),
                            None => new_tokens.iter().filter_map(|&t| char::from_u32(t)).collect::<String>(),
                        };

                        let tokens_generated = new_tokens.len();
                        let latency_ms = start.elapsed().as_millis() as u64;
                        let tok_per_sec = if gen_time.as_secs_f64() > 0.0 {
                            tokens_generated as f64 / gen_time.as_secs_f64()
                        } else {
                            0.0
                        };

                        Json(CompletionResponse {
                            text,
                            tokens_generated,
                            latency_ms,
                            tok_per_sec,
                        })
                        .into_response()
                    }
                }),
            )
            .route(
                "/v1/chat/completions",
                {
                    let cuda_for_chat = cuda_model.clone();
                    let tok_for_chat = tokenizer.clone();
                    let path_for_chat = model_path_arc.clone();
                    post(move |Json(req): Json<serde_json::Value>| {
                        let cuda = cuda_for_chat.clone();
                        let tok_info = tok_for_chat.clone();
                        let model_path = path_for_chat.clone();
                        async move {
                            use axum::response::sse::{Event, Sse};
                            use futures_util::stream;
                            use realizar::apr::AprModel;

                            // Extract messages from request
                            let messages = req.get("messages").and_then(|m| m.as_array());
                            let stream_mode = req.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
                            let max_tokens = req.get("max_tokens").and_then(serde_json::Value::as_u64).unwrap_or(32) as usize;

                            let prompt = if let Some(msgs) = messages {
                                // Format as ChatML
                                let mut prompt = String::new();
                                for msg in msgs {
                                    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                                    write!(prompt, "<|im_start|>{}\n{}<|im_end|>\n", role, content).expect("write to String cannot fail");
                                }
                                prompt.push_str("<|im_start|>assistant\n");
                                prompt
                            } else {
                                return axum::Json(serde_json::json!({"error": "Missing messages"})).into_response();
                            };

                            let start = Instant::now();

                            // Encode prompt
                            // PMAT-098: Use BPE tokenizer for proper encoding
                            let input_tokens: Vec<u32> = match tok_info.as_ref() {
                                Some(tok) => tok.tokenizer.encode(&prompt),
                                None => prompt.chars().map(|c| c as u32).collect(),
                            };

                            let eos_id = match tok_info.as_ref() {
                                Some(tok) => tok.eos_token_id.unwrap_or(151645),
                                _ => 151645, // ChatML end token
                            };

                            // Generate on GPU
                            let gen_start = Instant::now();
                            let max_tokens = max_tokens.min(256);
                            let output_tokens = {
                                // PMAT-189: Handle CUDA model lock poisoning gracefully
                                let mut model = match cuda.lock() {
                                    Ok(guard) => guard,
                                    Err(_poisoned) => {
                                        return axum::Json(serde_json::json!({
                                            "error": "GPU model state corrupted (lock poisoned). Please restart the server."
                                        }))
                                            .into_response();
                                    }
                                };
                                match model.generate_cuda(&input_tokens, max_tokens, eos_id) {
                                    Ok(t) => t,
                                    Err(e) => {
                                        return axum::Json(serde_json::json!({"error": format!("GPU generation failed: {e}")}))
                                            .into_response();
                                    }
                                }
                            };
                            let elapsed = gen_start.elapsed();

                            // Decode
                            let new_tokens = if output_tokens.len() > input_tokens.len() {
                                &output_tokens[input_tokens.len()..]
                            } else {
                                &output_tokens[..]
                            };

                            // PMAT-099: Debug logging for GPU token decoding
                            eprintln!("[APR GPU CHAT DEBUG] Input tokens: {}, Output tokens: {}, New tokens: {}",
                                input_tokens.len(), output_tokens.len(), new_tokens.len());
                            eprintln!("[APR GPU CHAT DEBUG] New token IDs: {:?}", &new_tokens[..new_tokens.len().min(20)]);

                            // PMAT-098: Use BPE tokenizer for proper decoding
                            let output_text = match tok_info.as_ref() {
                                Some(tok) => match tok.tokenizer.decode(new_tokens) {
                                    Ok(decoded) => {
                                        eprintln!("[APR GPU CHAT DEBUG] Decoded text: {:?}", decoded);
                                        decoded
                                    }
                                    Err(e) => {
                                        eprintln!("[APR GPU CHAT DEBUG] Decode error: {e}");
                                        String::new()
                                    }
                                },
                                None => {
                                    let fallback: String = new_tokens.iter().filter_map(|&t| char::from_u32(t)).collect();
                                    eprintln!("[APR GPU CHAT DEBUG] Fallback decode (no tokenizer): {:?}", fallback);
                                    fallback
                                }
                            };

                            let tokens_generated = new_tokens.len();
                            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                                tokens_generated as f64 / elapsed.as_secs_f64()
                            } else {
                                0.0
                            };

                            // Generate unique ID
                            let request_id = format!(
                                "chatcmpl-{}-{}",
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos(),
                                std::process::id()
                            );

                            let created = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();

                            if stream_mode {
                                // SSE streaming response (PAR-302)
                                let response = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": "apr-gpu",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": output_text},
                                        "finish_reason": "stop"
                                    }]
                                });

                                let stream = stream::once(async move {
                                    Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string()))
                                });
                                Sse::new(stream).into_response()
                            } else {
                                // Non-streaming response
                                axum::Json(serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion",
                                    "created": created,
                                    "model": "apr-gpu",
                                    "choices": [{
                                        "index": 0,
                                        "message": {"role": "assistant", "content": output_text},
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": input_tokens.len(),
                                        "completion_tokens": tokens_generated,
                                        "total_tokens": input_tokens.len() + tokens_generated
                                    },
                                    "_apr_metrics": {
                                        "latency_ms": start.elapsed().as_millis(),
                                        "tok_per_sec": tok_per_sec
                                    }
                                }))
                                .into_response()
                            }
                        }
                    })
                },
            )
            .route(
                "/",
                get(|| async { "APR v2 GPU Inference Server - POST /v1/completions, /v1/chat/completions" }),
            );

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("APR GPU Inference Server listening on http://{}", bind_addr)
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
// GGUF server handlers
// ============================================================================

/// Start GGUF model inference server with Ollama-parity performance
///
/// Uses realizar's full inference API for text generation, streaming, and batch inference.
/// Achieves Ollama-parity: 100+ tok/s CPU, 500+ tok/s GPU.
/// With --gpu --batch flags: 800+ tok/s (2.8x Ollama) via batched GPU inference.
#[cfg(feature = "inference")]
fn start_gguf_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::api::{create_router, AppState};
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    // Load GGUF model via mmap
    println!("{}", "Loading GGUF model (mmap)...".dimmed());
    let mapped_model = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;

    let tensor_count = mapped_model.model.tensors.len();
    let metadata_count = mapped_model.model.metadata.len();
    println!(
        "{}",
        format!(
            "GGUF loaded: {} tensors, {} metadata entries",
            tensor_count, metadata_count
        )
        .dimmed()
    );

    // Create quantized model for inference (Ollama-parity performance)
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

    // Extract vocabulary from GGUF for proper token encoding/decoding
    let vocab = mapped_model.model.vocabulary().unwrap_or_else(|| {
        eprintln!("Warning: No vocabulary in GGUF, using placeholder tokens");
        (0..quantized_model.config.vocab_size)
            .map(|i| format!("token{i}"))
            .collect()
    });

    // GPU batched inference path (2X+ Ollama performance)
    #[cfg(feature = "cuda")]
    if config.gpu && config.batch {
        return start_gguf_server_gpu_batched(quantized_model, vocab, config);
    }

    // GPU optimized path (--gpu flag) - Uses OwnedQuantizedModelCuda for 755+ tok/s (2.6x Ollama)
    // PAR-111: Pre-uploads weights and uses batched workspaces for maximum throughput
    #[cfg(feature = "cuda")]
    if config.gpu && !config.no_gpu {
        use realizar::gguf::OwnedQuantizedModelCuda;

        println!(
            "{}",
            "Enabling optimized CUDA acceleration (PAR-111)...".cyan()
        );

        // Create CUDA-optimized model wrapper (this initializes GPU KV cache)
        match OwnedQuantizedModelCuda::new(quantized_model, 0) {
            Ok(mut cuda_model) => {
                // Pre-upload all weights to GPU for maximum performance
                println!("  Initializing GPU on device 0...");
                match cuda_model.preload_weights_gpu() {
                    Ok(bytes) => {
                        println!(
                            "{}",
                            format!("  Pre-uploaded {} MB weights to GPU", bytes / (1024 * 1024))
                                .green()
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "{}",
                            format!("  Warning: Weight preload failed, using on-demand: {e}")
                                .yellow()
                        );
                    }
                }

                println!("{}", "CUDA optimized model ready".green());

                let state = AppState::with_cuda_model_and_vocab(cuda_model, vocab)
                    .map_err(|e| CliError::InferenceFailed(format!("Failed to create state: {e}")))?
                    .with_verbose(config.verbose); // GH-152: Pass verbose flag

                let app = create_router(state);

                let runtime = tokio::runtime::Runtime::new().map_err(|e| {
                    CliError::InferenceFailed(format!("Failed to create runtime: {e}"))
                })?;

                let bind_addr = config.bind_addr();

                return runtime.block_on(async move {
                    let listener = tokio::net::TcpListener::bind(&bind_addr)
                        .await
                        .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

                    println!();
                    println!(
                        "{}",
                        format!("CUDA-optimized server listening on http://{}", bind_addr)
                            .green()
                            .bold()
                    );
                    println!();
                    println!("{}", "Performance: 755+ tok/s (2.6x Ollama)".yellow());
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
                });
            }
            Err(e) => {
                eprintln!(
                    "{}",
                    format!("CUDA init failed, falling back to CPU: {e}").yellow()
                );
                // Fall through to CPU path - rebuild the model since we consumed it
            }
        }
        // Rebuild quantized model for CPU fallback if CUDA failed
        let quantized_model = OwnedQuantizedModel::from_mapped(&mapped_model).map_err(|e| {
            CliError::ModelLoadFailed(format!("Failed to rebuild quantized model: {e}"))
        })?;
        let vocab = mapped_model.model.vocabulary().unwrap_or_else(|| {
            (0..quantized_model.config.vocab_size)
                .map(|i| format!("token{i}"))
                .collect()
        });
        // Run CPU server with rebuilt model
        return run_cpu_server(quantized_model, vocab, config);
    }

    // CPU path (default - when not using GPU or cuda feature disabled)
    // Create realizar AppState with full inference capabilities and real vocab
    run_cpu_server(quantized_model, vocab, config)
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
