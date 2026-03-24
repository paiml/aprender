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

// PMAT-339: WGPU inference state + chat completion handler
#[cfg(feature = "wgpu")]
struct WgpuInferenceState {
    fwd: std::sync::Mutex<trueno::backends::gpu::WgslForwardPass>,
    token_embedding: Vec<f32>,
    output_norm_weight: Vec<f32>,
    lm_head_f32: Vec<f32>,
    vocab: Vec<String>,
    num_layers: usize,
    vocab_size: usize,
    hidden_dim: usize,
}

#[cfg(feature = "wgpu")]
async fn wgpu_chat_completion(
    state: Arc<WgpuInferenceState>,
    axum::Json(body): axum::Json<serde_json::Value>,
) -> axum::Json<serde_json::Value> {
    let max_tokens = body["max_tokens"].as_u64().unwrap_or(64) as usize;
    let messages = body["messages"].as_array();

    // Simple tokenization: extract last user message, split on whitespace
    // TODO: Use proper BPE tokenizer
    let prompt = messages
        .and_then(|m| m.last())
        .and_then(|m| m["content"].as_str())
        .unwrap_or("Hello");

    // Very basic character-level tokenization (placeholder)
    // Real implementation needs the embedded BPE tokenizer
    let prompt_ids: Vec<u32> = prompt.chars().map(|c| (c as u32).min(state.vocab_size as u32 - 1)).collect();

    let gen_start = std::time::Instant::now();
    let mut output_ids: Vec<u32> = Vec::new();

    // Lock the forward pass for inference
    let fwd = state.fwd.lock().unwrap();

    // Prefill: process all prompt tokens
    let mut last_logits = Vec::new();
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        match fwd.forward_model(
            token_id, pos, state.num_layers,
            &state.token_embedding, &state.output_norm_weight, &state.lm_head_f32,
            state.vocab_size, 1e-6,
        ) {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                return axum::Json(serde_json::json!({
                    "error": format!("Forward failed: {e}")
                }));
            }
        }
    }

    // Decode: sample tokens autoregressively
    for step in 0..max_tokens {
        // Argmax sampling
        let next_token = last_logits.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);

        // EOS check (Qwen: 151645)
        if next_token == 151645 || next_token == 0 {
            break;
        }

        output_ids.push(next_token);

        let position = prompt_ids.len() + step;
        match fwd.forward_model(
            next_token, position, state.num_layers,
            &state.token_embedding, &state.output_norm_weight, &state.lm_head_f32,
            state.vocab_size, 1e-6,
        ) {
            Ok(logits) => last_logits = logits,
            Err(e) => break,
        }
    }
    drop(fwd); // Release lock

    let elapsed = gen_start.elapsed();

    // Detokenize using vocab
    let output_text: String = output_ids.iter()
        .map(|&id| {
            if (id as usize) < state.vocab.len() {
                state.vocab[id as usize].clone()
            } else {
                format!("[{}]", id)
            }
        })
        .collect::<Vec<_>>()
        .join("");

    let tok_s = if elapsed.as_secs_f64() > 0.0 {
        output_ids.len() as f64 / elapsed.as_secs_f64()
    } else { 0.0 };

    axum::Json(serde_json::json!({
        "id": format!("chatcmpl-wgpu-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis()),
        "object": "chat.completion",
        "model": "qwen-wgpu",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output_text,
            },
            "finish_reason": if output_ids.len() >= max_tokens { "length" } else { "stop" },
        }],
        "usage": {
            "prompt_tokens": prompt_ids.len(),
            "completion_tokens": output_ids.len(),
            "total_tokens": prompt_ids.len() + output_ids.len(),
        },
        "x_wgpu_latency_ms": elapsed.as_secs_f64() * 1000.0,
        "x_wgpu_tok_s": tok_s,
    }))
}

// ============================================================================
// Format detection and dispatch
// ============================================================================

/// Start server using realizar
#[cfg(feature = "inference")]
pub(crate) fn start_realizar_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::format::{detect_format, ModelFormat};
    use std::io::Read;

    // PMAT-332/333: If --backend wgpu, load model + dequant + WGPU upload
    if config.backend.as_deref() == Some("wgpu") {
        println!();
        println!("{}", "Backend: WGPU (Vulkan/Metal/WebGPU)".cyan());
        println!("{}", "PMAT-333: Loading model for WGPU inference...".dimmed());

        // Step 1: Load GGUF model
        use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
        let mapped = MappedGGUFModel::from_path(model_path)
            .map_err(|e| CliError::ModelLoadFailed(format!("GGUF load: {e}")))?;
        let quantized = OwnedQuantizedModel::from_mapped(&mapped)
            .map_err(|e| CliError::ModelLoadFailed(format!("Quantized model: {e}")))?;
        let num_layers = quantized.layers().len();
        println!("{}", format!(
            "Model: {} layers loaded for WGPU dequantization",
            num_layers,
        ).green());

        // Step 2: Dequantize weights
        let dequant_start = std::time::Instant::now();
        let weights = realizar::gpu::adapters::wgpu_adapter::dequant_model_weights(&quantized)
            .map_err(|e| CliError::ModelLoadFailed(format!("Dequant: {e}")))?;
        let total_mb: f64 = weights.iter().map(|(_, d, _, _)| d.len() * 4).sum::<usize>() as f64 / 1e6;
        println!("{}", format!(
            "Dequantized {} weights ({:.0} MB) in {:.1}s",
            weights.len(), total_mb, dequant_start.elapsed().as_secs_f64(),
        ).dimmed());

        // Step 3: WGPU upload + serve
        #[cfg(feature = "wgpu")]
        {
            println!("{}", "Initializing WGPU device...".dimmed());
            let gpu_dev = trueno::backends::gpu::GpuDevice::new()
                .map_err(|e| CliError::ModelLoadFailed(format!("WGPU init: {e}")))?;
            println!("{}", "WGPU device ready (Vulkan/Metal)".green());

            // Get model dims from dequanted weights (avoid private config field)
            // The first Q proj weight has shape [q_dim, hidden_dim]
            let hidden_dim = weights.iter()
                .find(|(n, _, _, _)| n.ends_with(".q_proj"))
                .map(|(_, _, _, cols)| *cols)
                .unwrap_or(1536);
            let intermediate_dim = weights.iter()
                .find(|(n, _, _, _)| n.ends_with(".gate_proj"))
                .map(|(_, _, rows, _)| *rows)
                .unwrap_or(8960);
            // Infer heads from Q proj: q_dim = num_heads * head_dim
            let q_dim = weights.iter()
                .find(|(n, _, _, _)| n.ends_with(".q_proj"))
                .map(|(_, _, rows, _)| *rows)
                .unwrap_or(hidden_dim);
            let kv_dim = weights.iter()
                .find(|(n, _, _, _)| n.ends_with(".k_proj"))
                .map(|(_, _, rows, _)| *rows)
                .unwrap_or(256);
            let head_dim = 128; // Standard for Qwen2
            let num_heads = q_dim / head_dim;
            let num_kv_heads = kv_dim / head_dim;

            let mut fwd = trueno::backends::gpu::WgslForwardPass::new(
                gpu_dev.device.clone(),
                gpu_dev.queue.clone(),
                hidden_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_dim,
            );

            let upload_start = std::time::Instant::now();
            for (name, data, _rows, _cols) in &weights {
                fwd.upload_weight(name, data);
            }
            println!("{}", format!(
                "Uploaded {} weights to GPU ({:.1} MB VRAM) in {:.1}ms",
                weights.len(),
                fwd.total_vram_bytes() as f64 / 1e6,
                upload_start.elapsed().as_secs_f64() * 1000.0,
            ).green());

            // Step 4: Extract CPU-side data for forward_model
            let token_embedding = quantized.token_embedding().to_vec();
            let output_norm_weight = quantized.output_norm_weight().to_vec();
            let vocab_size = token_embedding.len() / hidden_dim;

            // LM head from dequanted weights
            let lm_head_f32 = weights.iter()
                .find(|(n, _, _, _)| n == "lm_head")
                .map(|(_, d, _, _)| d.clone())
                .unwrap_or_else(|| token_embedding.clone()); // tied embeddings fallback

            println!("{}", format!(
                "WGPU inference ready: {} layers, vocab={}, hidden={}",
                num_layers, vocab_size, hidden_dim,
            ).green());

            // Step 5: Quick correctness test — generate one token
            let test_token = 9707u32; // "Hello" in Qwen tokenizer
            let test_start = std::time::Instant::now();
            match fwd.forward_model(
                test_token, 0, num_layers,
                &token_embedding, &output_norm_weight, &lm_head_f32,
                vocab_size, 1e-6,
            ) {
                Ok(logits) => {
                    let argmax = logits.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let elapsed = test_start.elapsed();
                    println!("{}", format!(
                        "WGPU test: token {} → logits[{}] max at idx {} ({:.1}ms)",
                        test_token, logits.len(), argmax, elapsed.as_secs_f64() * 1000.0,
                    ).cyan());
                }
                Err(e) => {
                    println!("{}", format!("WGPU forward failed: {e}").red());
                }
            }

            // PMAT-339: WGPU HTTP serve — minimal /v1/chat/completions endpoint
            println!("{}", "Starting WGPU inference server...".cyan());

            // Extract vocab for detokenization (placeholder — proper BPE needed)
            let vocab: Vec<String> = (0..vocab_size)
                .map(|i| format!("token{i}"))
                .collect();

            // Wrap inference state in Arc for axum handler
            use std::sync::{Arc, Mutex};
            let wgpu_state = Arc::new(WgpuInferenceState {
                fwd: Mutex::new(fwd),
                token_embedding,
                output_norm_weight,
                lm_head_f32,
                vocab,
                num_layers,
                vocab_size,
                hidden_dim,
            });

            // Build minimal axum router
            use axum::{routing::{get, post}, Json, Router};

            let state_health = wgpu_state.clone();
            let state_chat = wgpu_state.clone();

            let app = Router::new()
                .route("/health", get(move || async move {
                    Json(serde_json::json!({
                        "status": "healthy",
                        "compute_mode": "wgpu",
                        "version": env!("CARGO_PKG_VERSION"),
                    }))
                }))
                .route("/v1/chat/completions", post(move |body: Json<serde_json::Value>| {
                    let state = state_chat.clone();
                    async move { wgpu_chat_completion(state, body).await }
                }));

            let bind_addr = config.bind_addr();
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| CliError::InferenceFailed(format!("Runtime: {e}")))?;

            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind(&bind_addr).await
                    .map_err(|e| CliError::InferenceFailed(format!("Bind: {e}")))?;
                println!("{}", format!(
                    "WGPU inference server listening on http://{}",
                    bind_addr
                ).green().bold());
                println!("  POST /v1/chat/completions - Chat completions (WGPU)");
                println!("  GET  /health              - Health check");
                axum::serve(listener, app).await
                    .map_err(|e| CliError::InferenceFailed(format!("Serve: {e}")))?;
                Ok::<(), CliError>(())
            })?;
            return Ok(());
        }
        #[cfg(not(feature = "wgpu"))]
        {
            println!("{}", "WGPU feature not enabled. Build with --features wgpu".yellow());
        }
    }

    // GH-213 + PMAT-314: Detect sharded SafeTensors index.json BEFORE reading file bytes.
    // The index.json is a small JSON file that maps tensor names to shard files.
    // Reading it as binary triggers "header too large" DOS protection.
    // PMAT-314: Route sharded models through the same GPU→Q4K→F32 fallback chain
    // as single-file SafeTensors. apr_import handles index.json via streaming_sharded_import.
    let path_str = model_path.to_string_lossy();
    if path_str.ends_with(".safetensors.index.json") {
        println!();
        println!("Detected format: Sharded SafeTensors (index.json)");
        return start_safetensors_server_with_fallback(model_path, config);
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
        ModelFormat::SafeTensors => start_safetensors_server_with_fallback(model_path, config),
    }
}

/// Start SafeTensors server with GPU → Q4K CPU → F32 fallback chain.
#[cfg(feature = "inference")]
fn start_safetensors_server_with_fallback(model_path: &Path, config: &ServerConfig) -> Result<()> {
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
