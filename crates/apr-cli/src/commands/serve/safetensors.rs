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

use super::types::{
    ChatCompletionRequest, ChatMessage, ServerConfig,
};
use super::routes::create_router;

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
            Some(Arc::new(Mutex::new(t)))
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

/// Load tokenizer from HuggingFace tokenizer.json format with BPE merge rules
///
/// PMAT-093: Proper BPE tokenization is critical for SafeTensors inference.
/// Without merge rules, tokenization produces wrong tokens causing garbage output.
#[cfg(feature = "inference")]
pub(crate) fn load_safetensors_tokenizer(path: &Path) -> Option<SafeTensorsTokenizerInfo> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Extract vocabulary from model.vocab (token -> id mapping)
    let vocab_obj = json.get("model")?.get("vocab")?;
    let vocab_map = vocab_obj.as_object()?;

    // Build vocab vector sorted by ID (for index-based lookup)
    let mut vocab_vec: Vec<(String, u32)> = vocab_map
        .iter()
        .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
        .collect();
    vocab_vec.sort_by_key(|(_, id)| *id);
    let vocab: Vec<String> = vocab_vec.into_iter().map(|(token, _)| token).collect();

    // Extract BPE merge rules from model.merges
    // Merges are stored as ["ab cd", "ef gh", ...] meaning "merge 'ab' + 'cd'"
    let merges: Vec<(String, String)> = json
        .get("model")?
        .get("merges")?
        .as_array()?
        .iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let parts: Vec<&str> = s.split(' ').collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    // Extract special tokens and add them to vocabulary
    // PMAT-099: added_tokens must be included in vocab for decode to work
    let added_tokens = json.get("added_tokens").and_then(|v| v.as_array());
    let mut bos_token_id = None;
    let mut eos_token_id = None;
    let mut special_tokens: Vec<(u32, String)> = Vec::new();

    if let Some(tokens) = added_tokens {
        for token in tokens {
            let content = token.get("content").and_then(|v| v.as_str());
            let id = token.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);

            if let (Some(content), Some(id)) = (content, id) {
                special_tokens.push((id, content.to_string()));

                if content.contains("bos") || content == "<s>" {
                    bos_token_id = Some(id);
                }
                if content.contains("eos") || content == "</s>" || content.contains("im_end") {
                    eos_token_id = Some(id);
                }
            }
        }
    }

    // Extend vocab to include special tokens at their proper IDs
    // PMAT-099: Special tokens often have IDs beyond base vocab (e.g., 151643+)
    let mut vocab = vocab;
    if !special_tokens.is_empty() {
        let max_special_id = special_tokens.iter().map(|(id, _)| *id).max().unwrap_or(0);
        if max_special_id as usize >= vocab.len() {
            // Resize vocab to fit all special tokens
            vocab.resize(max_special_id as usize + 1, "<unused>".to_string());
        }
        // Insert special tokens at their IDs
        for (id, content) in special_tokens {
            if (id as usize) < vocab.len() {
                vocab[id as usize] = content;
            }
        }
    }

    // Create BPE tokenizer with vocab and merge rules
    let tokenizer = realizar::tokenizer::BPETokenizer::new(vocab.clone(), merges, "<unk>").ok()?;

    Some(SafeTensorsTokenizerInfo {
        tokenizer: std::sync::Arc::new(tokenizer),
        vocab,
        bos_token_id,
        eos_token_id,
    })
}

/// SafeTensors chat completions handler (PAR-301, GH-160 Tool Calling)
#[cfg(feature = "inference")]
pub(crate) async fn safetensors_chat_completions_handler(
    axum::extract::State(state): axum::extract::State<SafeTensorsState>,
    axum::Json(request): axum::Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::{sse::Event, IntoResponse, Sse};
    use futures_util::stream;

    // Parse request - try structured first, fallback to raw JSON (GH-160)
    let parsed_request: ChatCompletionRequest = match serde_json::from_value(request.clone()) {
        Ok(req) => req,
        Err(_) => {
            // Fallback: extract from raw JSON for backwards compatibility
            let messages = request.get("messages").and_then(|m| m.as_array());
            if messages.is_none() {
                return (
                    StatusCode::BAD_REQUEST,
                    axum::Json(serde_json::json!({"error": "Missing messages field"})),
                )
                    .into_response();
            }
            let msgs: Vec<ChatMessage> = messages
                .unwrap()
                .iter()
                .filter_map(|m| {
                    Some(ChatMessage {
                        role: m.get("role")?.as_str()?.to_string(),
                        content: m.get("content").and_then(|c| c.as_str()).map(String::from),
                        tool_calls: None,
                        tool_call_id: m
                            .get("tool_call_id")
                            .and_then(|t| t.as_str())
                            .map(String::from),
                        name: m.get("name").and_then(|n| n.as_str()).map(String::from),
                    })
                })
                .collect();
            ChatCompletionRequest {
                model: request
                    .get("model")
                    .and_then(|m| m.as_str())
                    .unwrap_or("default")
                    .to_string(),
                messages: msgs,
                tools: request
                    .get("tools")
                    .and_then(|t| serde_json::from_value(t.clone()).ok()),
                tool_choice: request
                    .get("tool_choice")
                    .and_then(|t| serde_json::from_value(t.clone()).ok()),
                max_tokens: request
                    .get("max_tokens")
                    .and_then(|m| m.as_u64())
                    .map(|v| v as u32),
                stream: request
                    .get("stream")
                    .and_then(|s| s.as_bool())
                    .unwrap_or(false),
                temperature: request
                    .get("temperature")
                    .and_then(|t| t.as_f64())
                    .map(|v| v as f32),
                top_p: request
                    .get("top_p")
                    .and_then(|t| t.as_f64())
                    .map(|v| v as f32),
            }
        }
    };

    let max_tokens = parsed_request.max_tokens.unwrap_or(50) as usize;
    let stream_mode = parsed_request.stream;
    let has_tools = parsed_request.tools.as_ref().is_some_and(|t| !t.is_empty());

    // Build prompt from messages (ChatML format)
    let mut prompt = String::new();

    // Add system message with tools if present (GH-160)
    if has_tools {
        let tools_prompt = super::types::format_tools_prompt(parsed_request.tools.as_deref().unwrap_or(&[]));
        // Insert tools description in system message or create one
        let has_system = parsed_request.messages.iter().any(|m| m.role == "system");
        if !has_system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str("You are a helpful assistant.");
            prompt.push_str(&tools_prompt);
            prompt.push_str("<|im_end|>\n");
        }
    }

    for msg in &parsed_request.messages {
        prompt.push_str(&format!("<|im_start|>{}\n", msg.role));

        // Handle tool messages (responses to tool calls)
        if msg.role == "tool" {
            if let Some(ref tool_call_id) = msg.tool_call_id {
                prompt.push_str(&format!("[Tool Result for {}]\n", tool_call_id));
            }
        }

        // Add tools prompt to system message
        if msg.role == "system" && has_tools {
            if let Some(ref content) = msg.content {
                prompt.push_str(content);
            }
            prompt.push_str(&super::types::format_tools_prompt(
                parsed_request.tools.as_deref().unwrap_or(&[]),
            ));
        } else if let Some(ref content) = msg.content {
            prompt.push_str(content);
        }

        // Add tool calls made by assistant
        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                prompt.push_str(&format!(
                    "\n[Tool Call: {} with args {}]",
                    tc.function.name, tc.function.arguments
                ));
            }
        }

        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");

    // Get transformer
    let transformer = match &state.transformer {
        Some(t) => t.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(
                    serde_json::json!({"error": "Inference not available - missing config.json"}),
                ),
            )
                .into_response();
        }
    };

    // Encode prompt using BPE tokenizer (PMAT-093)
    let input_ids = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info.tokenizer.encode(&prompt)
    } else {
        // Fallback: character-level tokenization (no tokenizer.json)
        prompt.chars().map(|c| c as u32).collect()
    };

    // PMAT-103 FIX: Use generate_with_cache for O(n) generation
    // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
    // generate_with_cache() uses KV cache for incremental generation = O(n)
    let start = Instant::now();
    let temperature = request
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0) as f32;
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: false,
    };
    let output_ids = {
        // PMAT-189: Handle transformer lock poisoning gracefully
        let t = match transformer.lock() {
            Ok(guard) => guard,
            Err(_poisoned) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": "Transformer state corrupted (lock poisoned). Please restart the server."
                    })),
                )
                    .into_response();
            }
        };
        match t.generate_with_cache(&input_ids, &gen_config) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": format!("Generation failed: {e}")})),
                )
                    .into_response();
            }
        }
    };
    let elapsed = start.elapsed();

    // Decode output using BPE tokenizer (PMAT-093)
    let new_tokens = &output_ids[input_ids.len()..];
    let output_text = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info
            .tokenizer
            .decode(new_tokens)
            .unwrap_or_else(|_| simple_decode(new_tokens, &tok_info.vocab))
    } else {
        new_tokens
            .iter()
            .map(|&id| char::from_u32(id.min(127)).unwrap_or('?'))
            .collect()
    };

    // Clean output (remove any trailing special tokens)
    let output_text = output_text
        .split("<|im_end|>")
        .next()
        .unwrap_or(&output_text)
        .to_string();

    let tokens_generated = new_tokens.len();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    // Generate unique ID using timestamp and process ID
    let request_id = format!(
        "chatcmpl-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id()
    );

    // GH-160: Check for tool calls in output
    let tool_calls = if has_tools {
        super::types::parse_tool_calls(&output_text)
    } else {
        None
    };
    let has_tool_calls = tool_calls.is_some();
    let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };

    // Return OpenAI-compatible response
    if stream_mode {
        // SSE streaming response
        let response = if has_tool_calls {
            serde_json::json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                "model": "safetensors",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "tool_calls": tool_calls},
                    "finish_reason": finish_reason
                }]
            })
        } else {
            serde_json::json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                "model": "safetensors",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": output_text},
                    "finish_reason": finish_reason
                }]
            })
        };

        let stream = stream::once(async move {
            Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string()))
        });
        Sse::new(stream).into_response()
    } else {
        // Non-streaming response (GH-160: tool calls support)
        let message = if has_tool_calls {
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": tool_calls
            })
        } else {
            serde_json::json!({
                "role": "assistant",
                "content": output_text
            })
        };

        axum::Json(serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "model": "safetensors",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": input_ids.len(),
                "completion_tokens": tokens_generated,
                "total_tokens": input_ids.len() + tokens_generated
            },
            "latency_ms": elapsed.as_millis(),
            "tok_per_sec": tok_per_sec
        }))
        .into_response()
    }
}

/// SafeTensors generate handler
#[cfg(feature = "inference")]
pub(crate) async fn safetensors_generate_handler(
    axum::extract::State(state): axum::extract::State<SafeTensorsState>,
    axum::Json(request): axum::Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let prompt = request.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
    let max_tokens = request
        .get("max_tokens")
        .and_then(|m| m.as_u64())
        .unwrap_or(32) as usize;

    let transformer = match &state.transformer {
        Some(t) => t.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({"error": "Inference not available"})),
            )
                .into_response();
        }
    };

    // Encode prompt using BPE tokenizer (PMAT-093)
    let input_ids = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info.tokenizer.encode(prompt)
    } else {
        prompt.chars().map(|c| c as u32).collect()
    };

    // PMAT-103 FIX: Use generate_with_cache for O(n) generation
    // Previous code used generate() which calls forward() on ALL tokens each step = O(n²)
    // generate_with_cache() uses KV cache for incremental generation = O(n)
    let start = Instant::now();
    let temperature = request
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0) as f32;
    let gen_config = realizar::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: false,
    };
    let output_ids = {
        // PMAT-189: Handle transformer lock poisoning gracefully
        let t = match transformer.lock() {
            Ok(guard) => guard,
            Err(_poisoned) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": "Transformer state corrupted (lock poisoned). Please restart the server."
                    })),
                )
                    .into_response();
            }
        };
        match t.generate_with_cache(&input_ids, &gen_config) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({"error": format!("Generation failed: {e}")})),
                )
                    .into_response();
            }
        }
    };
    let elapsed = start.elapsed();

    // Decode using BPE tokenizer (PMAT-093)
    let new_tokens = &output_ids[input_ids.len()..];
    let output_text = if let Some(ref tok_info) = state.tokenizer_info {
        tok_info
            .tokenizer
            .decode(new_tokens)
            .unwrap_or_else(|_| simple_decode(new_tokens, &tok_info.vocab))
    } else {
        new_tokens
            .iter()
            .map(|&id| char::from_u32(id.min(127)).unwrap_or('?'))
            .collect()
    };

    let tokens_generated = new_tokens.len();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        tokens_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    axum::Json(serde_json::json!({
        "text": output_text,
        "tokens_generated": tokens_generated,
        "latency_ms": elapsed.as_millis(),
        "tok_per_sec": tok_per_sec
    }))
    .into_response()
}

/// Simple tokenization using greedy longest match
pub(crate) fn simple_encode(text: &str, vocab: &[String]) -> Vec<u32> {
    let mut tokens = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Find longest matching token
        let mut best_match = None;
        let mut best_len = 0;

        for (id, token) in vocab.iter().enumerate() {
            if remaining.starts_with(token) && token.len() > best_len {
                best_match = Some(id as u32);
                best_len = token.len();
            }
        }

        if let Some(id) = best_match {
            tokens.push(id);
            remaining = &remaining[best_len..];
        } else {
            // Skip unknown character
            let char_len = remaining.chars().next().map_or(1, char::len_utf8);
            remaining = &remaining[char_len..];
        }
    }

    tokens
}

/// Simple decode using vocab lookup
pub(crate) fn simple_decode(token_ids: &[u32], vocab: &[String]) -> String {
    token_ids
        .iter()
        .map(|&id| {
            vocab
                .get(id as usize)
                .map_or("?".to_string(), |s| s.clone())
        })
        .collect::<String>()
}
