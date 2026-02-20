//! SafeTensors format server handlers and tokenizer support
//!
//! Extracted from serve.rs (PMAT-200) - provides SafeTensors model inspection
//! and inference server with BPE tokenizer loading, chat completions (PAR-301,
//! GH-160 tool calling), and generate endpoints.

// Allow dead code and unused during development - these are planned features
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]
#![allow(clippy::format_push_string)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::if_not_else)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::inefficient_to_string)]

use super::routes::create_router;
use super::types::{ChatCompletionRequest, ChatMessage, ServerConfig};

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Shared state for SafeTensors server
#[cfg(feature = "inference")]
#[derive(Clone)]
pub(crate) struct SafeTensorsState {
    pub transformer: Option<Arc<std::sync::Mutex<realizar::apr_transformer::AprTransformer>>>,
    pub tokenizer_info: Option<SafeTensorsTokenizerInfo>,
    pub model_path: String,
}

/// Tokenizer info for SafeTensors models with proper BPE support
#[cfg(feature = "inference")]
#[derive(Clone)]
pub(crate) struct SafeTensorsTokenizerInfo {
    /// BPE tokenizer with vocab and merge rules
    pub tokenizer: std::sync::Arc<realizar::tokenizer::BPETokenizer>,
    /// Vocab for decode fallback
    pub vocab: Vec<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

/// Start SafeTensors inspection server
#[cfg(feature = "inference")]
pub(crate) fn start_safetensors_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use axum::{
        routing::{get, post},
        Json, Router,
    };
    use realizar::apr_transformer::AprTransformer;
    use realizar::safetensors::SafetensorsModel;
    use realizar::safetensors_infer::SafetensorsToAprConverter;
    use serde::Serialize;
    use std::sync::{Arc, Mutex};

    // Load SafeTensors from file (T-QA-020)
    let bytes = std::fs::read(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to read SafeTensors file: {e}")))?;
    let st_model = SafetensorsModel::from_bytes(&bytes)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to parse SafeTensors: {e}")))?;

    let tensor_names: Vec<String> = st_model
        .tensor_names()
        .into_iter()
        .map(String::from)
        .collect();
    let tensor_count = tensor_names.len();

    println!(
        "{}",
        format!("SafeTensors loaded: {} tensors", tensor_count).green()
    );

    // Try to convert to AprTransformer for inference (PAR-301)
    let transformer = match SafetensorsToAprConverter::convert(model_path) {
        Ok(t) => {
            println!(
                "{}",
                format!(
                    "Inference enabled: {} layers, hidden_dim={}",
                    t.config.num_layers, t.config.hidden_dim
                )
                .cyan()
            );
            Some(Arc::new(Mutex::new(t.into_inner())))
        }
        Err(e) => {
            println!(
                "{}",
                format!("Inference disabled: {e} (inspection-only mode)").yellow()
            );
            None
        }
    };

    // Try to load tokenizer from sibling tokenizer.json
    let tokenizer_path = model_path.with_file_name("tokenizer.json");
    let tokenizer_info = if tokenizer_path.exists() {
        load_safetensors_tokenizer(&tokenizer_path)
    } else {
        println!(
            "{}",
            "No tokenizer.json found - using fallback tokenization".yellow()
        );
        None
    };

    if tokenizer_info.is_some() {
        println!("{}", "Tokenizer loaded from sibling file".dimmed());
    }

    // Create tokio runtime
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let model_path_str = model_path.display().to_string();
    let inference_enabled = transformer.is_some();

    runtime.block_on(async move {
        #[derive(Clone, Serialize)]
        struct TensorInfo {
            count: usize,
            model: String,
            names: Vec<String>,
            inference_enabled: bool,
        }

        let info = TensorInfo {
            count: tensor_count,
            model: model_path_str.clone(),
            names: tensor_names.clone(),
            inference_enabled,
        };

        // Use module-level SafeTensorsState for inference
        let state = SafeTensorsState {
            transformer: transformer.clone(),
            tokenizer_info: tokenizer_info.clone(),
            model_path: model_path_str.clone(),
        };

        // Build base routes (no state required)
        let base_routes = Router::new()
            .route(
                "/health",
                get({
                    let inference = inference_enabled;
                    move || async move {
                        Json(serde_json::json!({
                            "status": "healthy",
                            "inference_enabled": inference
                        }))
                    }
                }),
            )
            .route("/tensors", get(move || async move { Json(info.clone()) }));

        // Build inference routes with state (PAR-301)
        let inference_routes: Router = if inference_enabled {
            Router::new()
                .route(
                    "/v1/chat/completions",
                    post(safetensors_chat_completions_handler),
                )
                .route("/generate", post(safetensors_generate_handler))
                .with_state(state)
        } else {
            Router::new()
        };

        // Merge all routes into final app
        let app = base_routes.merge(inference_routes);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /tensors             - List tensor names");
        if inference_enabled {
            println!("  POST /generate            - Text generation");
            println!("  POST /v1/chat/completions - Chat completions (PAR-301)");
        }
        println!();
        if !inference_enabled {
            println!(
                "{}",
                "Note: Inference disabled - ensure config.json exists alongside model".yellow()
            );
        }
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(super::handlers::shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start sharded SafeTensors server (GH-213)
///
/// Handles `.safetensors.index.json` files that map tensor names across multiple
/// shard files. Uses `ShardedSafeTensorsModel::load_from_index()` +
/// `SafetensorsToAprConverter::convert_sharded()` instead of reading raw bytes.
#[cfg(feature = "inference")]
pub(crate) fn start_sharded_safetensors_server(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<()> {
    use axum::{
        routing::{get, post},
        Json, Router,
    };
    use realizar::safetensors::{SafetensorsConfig, ShardedSafeTensorsModel};
    use realizar::safetensors_infer::SafetensorsToAprConverter;
    use serde::Serialize;
    use std::sync::{Arc, Mutex};

    // Load sharded model from index.json
    let sharded = ShardedSafeTensorsModel::load_from_index(model_path).map_err(|e| {
        CliError::ModelLoadFailed(format!("Failed to load sharded SafeTensors: {e}"))
    })?;

    println!(
        "{}",
        format!(
            "Sharded SafeTensors loaded: {} shards, {} tensors",
            sharded.shard_count(),
            sharded.tensor_count()
        )
        .green()
    );

    // Load config.json from sibling
    let st_config = SafetensorsConfig::load_from_sibling(model_path).ok_or_else(|| {
        CliError::ModelLoadFailed(
            "config.json not found (required for sharded SafeTensors inference)".to_string(),
        )
    })?;

    // Convert to AprTransformer
    let transformer = match SafetensorsToAprConverter::convert_sharded(&sharded, &st_config) {
        Ok(t) => {
            println!(
                "{}",
                format!(
                    "Inference enabled: {} layers, hidden_dim={}",
                    t.config.num_layers, t.config.hidden_dim
                )
                .cyan()
            );
            Some(Arc::new(Mutex::new(t.into_inner())))
        }
        Err(e) => {
            println!(
                "{}",
                format!("Inference disabled: {e} (inspection-only mode)").yellow()
            );
            None
        }
    };

    // Try to load tokenizer from sibling tokenizer.json
    let tokenizer_path = model_path.with_file_name("tokenizer.json");
    let tokenizer_info = if tokenizer_path.exists() {
        load_safetensors_tokenizer(&tokenizer_path)
    } else {
        println!(
            "{}",
            "No tokenizer.json found - using fallback tokenization".yellow()
        );
        None
    };

    if tokenizer_info.is_some() {
        println!("{}", "Tokenizer loaded from sibling file".dimmed());
    }

    // Create tokio runtime
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let model_path_str = model_path.display().to_string();
    let inference_enabled = transformer.is_some();
    let tensor_count = sharded.tensor_count();
    let shard_count = sharded.shard_count();

    runtime.block_on(async move {
        #[derive(Clone, Serialize)]
        struct ShardedInfo {
            count: usize,
            shards: usize,
            model: String,
            inference_enabled: bool,
        }

        let info = ShardedInfo {
            count: tensor_count,
            shards: shard_count,
            model: model_path_str.clone(),
            inference_enabled,
        };

        // Use SafeTensorsState for inference (same state type as single-file)
        let state = SafeTensorsState {
            transformer: transformer.clone(),
            tokenizer_info: tokenizer_info.clone(),
            model_path: model_path_str.clone(),
        };

        // Build base routes
        let base_routes = Router::new()
            .route(
                "/health",
                get({
                    let inference = inference_enabled;
                    move || async move {
                        Json(serde_json::json!({
                            "status": "healthy",
                            "inference_enabled": inference
                        }))
                    }
                }),
            )
            .route("/tensors", get(move || async move { Json(info.clone()) }));

        // Build inference routes with state (same handlers as single-file)
        let inference_routes: Router = if inference_enabled {
            Router::new()
                .route(
                    "/v1/chat/completions",
                    post(safetensors_chat_completions_handler),
                )
                .route("/generate", post(safetensors_generate_handler))
                .with_state(state)
        } else {
            Router::new()
        };

        let app = base_routes.merge(inference_routes);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /tensors             - List tensor names");
        if inference_enabled {
            println!("  POST /generate            - Text generation");
            println!("  POST /v1/chat/completions - Chat completions");
        }
        println!();
        if !inference_enabled {
            println!(
                "{}",
                "Note: Inference disabled - ensure config.json exists alongside model".yellow()
            );
        }
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(super::handlers::shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Load tokenizer from HuggingFace tokenizer.json format with BPE merge rules
///
/// PMAT-093: Proper BPE tokenization is critical for SafeTensors inference.
/// Without merge rules, tokenization produces wrong tokens causing garbage output.
#[cfg(feature = "inference")]
/// Extract special tokens from tokenizer JSON added_tokens, merge them into vocab,
/// and detect BOS/EOS token IDs (PMAT-099)
/// Parse a single added_token JSON entry into (id, content).
fn parse_special_token(token: &serde_json::Value) -> Option<(u32, String)> {
    let content = token.get("content")?.as_str()?;
    let id = token.get("id")?.as_u64()? as u32;
    Some((id, content.to_string()))
}

/// Classify BOS/EOS from token content string.
fn classify_bos_eos(content: &str) -> (bool, bool) {
    let is_bos = content.contains("bos") || content == "<s>";
    let is_eos = content.contains("eos") || content == "</s>" || content.contains("im_end");
    (is_bos, is_eos)
}

include!("chat.rs");
include!("simple.rs");
