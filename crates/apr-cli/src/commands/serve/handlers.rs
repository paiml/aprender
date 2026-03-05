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

    // GH-213: Detect sharded SafeTensors index.json BEFORE reading file bytes.
    // The index.json is a small JSON file that maps tensor names to shard files.
    // Reading it as binary triggers "header too large" DOS protection.
    let path_str = model_path.to_string_lossy();
    if path_str.ends_with(".safetensors.index.json") {
        println!();
        println!("Detected format: Sharded SafeTensors (index.json)");
        println!("{}", "Starting sharded SafeTensors server...".cyan());
        return super::safetensors::start_sharded_safetensors_server(model_path, config);
    }

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
            // GH-88: Try fused Q4K GPU path first (same kernels as GGUF/APR, 200+ tok/s)
            #[cfg(feature = "cuda")]
            {
                let use_gpu = config.gpu && !config.no_gpu;
                if use_gpu {
                    println!(
                        "{}",
                        "Starting SafeTensors GPU server (fused Q4K)...".cyan()
                    );
                    match start_safetensors_server_gpu(model_path, config) {
                        Ok(()) => return Ok(()),
                        Err(e) => {
                            println!(
                                "{}",
                                format!("GPU init failed, falling back to CPU: {e}").yellow()
                            );
                        }
                    }
                }
            }
            // GH-99: Try Q4K CPU path (same fused kernels as GGUF, eliminates F32 fallback)
            #[cfg(feature = "inference")]
            {
                match start_safetensors_server_cpu_quantized(model_path, config) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        println!(
                            "{}",
                            format!("Q4K conversion failed, falling back to F32: {e}").yellow()
                        );
                    }
                }
            }
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
    /// Embedded BPE tokenizer from APR file metadata (GGUF-imported models)
    embedded_tokenizer: Option<realizar::apr::BpeTokenizer>,
    /// GH-283: Model name for request validation (derived from filename stem)
    model_name: String,
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

    // Tokenize: embedded APR tokenizer → sibling tokenizer.json → character-level fallback
    let input_tokens: Vec<u32> = if let Some(ref tok) = state.embedded_tokenizer {
        tok.encode(prompt)
    } else if let Some(ref tok) = state.tokenizer {
        tok.tokenizer.encode(prompt)
    } else {
        prompt.chars().map(|c| c as u32).collect()
    };
    let input_token_count = input_tokens.len();

    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: false,
        stop_tokens: vec![],
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

    // Decode: embedded APR tokenizer → sibling tokenizer.json → character-level fallback
    let text = if let Some(ref tok) = state.embedded_tokenizer {
        tok.decode(new_tokens)
    } else if let Some(ref tok) = state.tokenizer {
        tok.tokenizer.decode(new_tokens).unwrap_or_default()
    } else {
        new_tokens
            .iter()
            .filter_map(|&t| char::from_u32(t))
            .collect()
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
fn load_apr_model_state(model_path: &Path, config: &ServerConfig) -> Result<AprServerState> {
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

    // Try embedded tokenizer first (GGUF-imported .apr files embed tokenizer in metadata)
    let embedded_tokenizer = model.load_embedded_bpe_tokenizer();
    if embedded_tokenizer.is_some() {
        println!(
            "{}",
            "Embedded BPE tokenizer loaded from APR metadata".green()
        );
    }

    // Fall back to sibling tokenizer.json (GAP-UX-002: hash-prefix aware)
    let bpe_tokenizer = if embedded_tokenizer.is_none() {
        if let Some(tokenizer_path) =
            realizar::safetensors::find_sibling_file(model_path, "tokenizer.json")
        {
            println!(
                "{}",
                format!("BPE tokenizer loaded from {}", tokenizer_path.display()).green()
            );
            load_safetensors_tokenizer(&tokenizer_path)
        } else {
            println!(
                "{}",
                "No tokenizer found - using character-level fallback".yellow()
            );
            None
        }
    } else {
        None
    };

    // GH-259: GPU is handled by start_apr_server before reaching here.
    // This path is CPU-only.
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

    // GH-283: Derive model name from filename for request validation
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("apr")
        .to_string();

    Ok(AprServerState {
        transformer,
        model_type,
        architecture,
        is_transformer,
        tokenizer: bpe_tokenizer,
        embedded_tokenizer,
        model_name,
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
    // GH-87: Try fused Q4K GPU path first (same kernels as GGUF, 190+ tok/s)
    #[cfg(feature = "cuda")]
    {
        let use_gpu = config.gpu && !config.no_gpu;
        if use_gpu {
            match start_apr_server_gpu(model_path, config) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    println!(
                        "{}",
                        format!("GPU init failed, falling back to CPU: {e}").yellow()
                    );
                }
            }
        }
    }

    // Contract: apr-cpu-q4k-routing-v1.yaml
    // Try fused Q4K CPU path first (same engine as GGUF/SafeTensors, 3x faster than AprTransformer).
    // APR files from GGUF/SafeTensors import already contain Q4K tensors — no conversion needed.
    #[cfg(feature = "inference")]
    {
        match try_apr_quantized_cpu(model_path, config) {
            Ok(()) => return Ok(()),
            Err(e) => {
                println!(
                    "{}",
                    format!("Q4K path unavailable ({e}), using AprTransformer fallback").yellow()
                );
            }
        }
    }

    // Fallback: AprTransformer for non-transformer or non-Q4K APR models
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

/// Route APR CPU inference through fused Q4K kernels (OwnedQuantizedModel).
///
/// Contract: apr-cpu-q4k-routing-v1.yaml
/// Loading path: MappedAprModel::from_path → OwnedQuantizedModel::from_apr → run_cpu_server.
/// APR files from GGUF/SafeTensors import already contain Q4K tensors, so this avoids
/// the slow AprTransformer F32 matmul path (9.5 → ~28 tok/s).
#[cfg(feature = "inference")]
fn try_apr_quantized_cpu(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::apr::MappedAprModel;
    use realizar::gguf::OwnedQuantizedModel;

    println!("{}", "Loading APR model (fused Q4K kernels)...".dimmed());

    let mapped = MappedAprModel::from_path(model_path)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to map APR: {e}")))?;

    println!(
        "{}",
        format!(
            "APR loaded: {} tensors, {} metadata entries",
            mapped.tensors.len(),
            mapped.metadata.extra.len()
        )
        .dimmed()
    );

    let quantized = OwnedQuantizedModel::from_apr(&mapped)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create quantized model: {e}")))?;

    println!(
        "{}",
        format!(
            "Model ready: {} layers, vocab_size={}, hidden_dim={}",
            quantized.layers().len(),
            quantized.config().vocab_size,
            quantized.config().hidden_dim
        )
        .green()
    );

    // Extract vocabulary from embedded APR metadata (GGUF-imported models embed tokenizer)
    let vocab = mapped
        .metadata
        .get_embedded_vocabulary()
        .unwrap_or_else(|| {
            let vocab_size = mapped.metadata.vocab_size.unwrap_or(32000);
            eprintln!("Warning: No embedded vocabulary in APR, using placeholder tokens");
            let mut v: Vec<String> = (0..vocab_size).map(|i| format!("token{i}")).collect();
            if !v.is_empty() {
                v[0] = "<unk>".to_string();
            }
            v
        });

    println!("{}", "Q4K CPU inference ready".green());

    run_cpu_server(quantized, vocab, config)
}

/// Build the axum Router for APR CPU inference.
///
/// GH-284: Handlers are async with `spawn_blocking` to avoid blocking the
/// tokio runtime during multi-second generation.
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
                async move { handle_apr_cpu_completion(&state, &req).await }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(
                move |headers: axum::http::HeaderMap, Json(req): Json<serde_json::Value>| {
                    let state = state_for_chat.clone();
                    async move { handle_apr_cpu_chat_completion(&state, &headers, &req).await }
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

include!("handler_apr_cpu_completion.rs");
include!("handler_gpu_completion.rs");
include!("server.rs");
