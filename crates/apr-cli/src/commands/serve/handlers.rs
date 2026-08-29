//! Server handler implementations for APR and GGUF model formats
//!
//! Contains format-specific server startup functions extracted from the
//! monolithic serve.rs (PMAT-200). Each handler loads the model in its
//! native format and creates an HTTP server with inference endpoints.

#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

#[cfg(feature = "wgpu")]
use axum::response::IntoResponse;

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
    /// PMAT-340: Token→ID map for greedy fallback
    token_to_id: std::collections::HashMap<String, u32>,
    /// PMAT-341: BPE tokenizer with merge rules (None = greedy fallback)
    bpe_tokenizer: Option<realizar::apr::BpeTokenizer>,
    num_layers: usize,
    vocab_size: usize,
    hidden_dim: usize,
}

/// PMAT-355: Detokenize a single token ID using GPT-2 byte-level BPE.
#[cfg(feature = "wgpu")]
fn wgpu_detokenize_one(id: u32, vocab: &[String]) -> String {
    let token = match vocab.get(id as usize) {
        Some(t) => t,
        None => return String::new(),
    };
    if token.starts_with("<|") && token.ends_with("|>") {
        return String::new();
    }
    if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
        if let Ok(b) = u8::from_str_radix(&token[3..5], 16) {
            return String::from_utf8_lossy(&[b]).into_owned();
        }
    }
    let mut bytes = Vec::new();
    for c in token.chars() {
        let cp = c as u32;
        let byte = if (0x21..=0x7E).contains(&cp)
            || (0xA1..=0xAC).contains(&cp)
            || (0xAE..=0xFF).contains(&cp)
        {
            cp as u8
        } else if (0x0100..=0x0143).contains(&cp) {
            let off = cp - 0x0100;
            match off {
                0..=32 => off as u8,
                33 => 0x7F,
                34..=66 => (0x80 + (off - 34)) as u8,
                67 => 0xAD,
                _ => {
                    bytes.push(b'?');
                    continue;
                }
            }
        } else {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            bytes.extend_from_slice(s.as_bytes());
            continue;
        };
        bytes.push(byte);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// PMAT-355: WGPU chat completion with streaming SSE support.
#[cfg(feature = "wgpu")]
#[provable_contracts_macros::contract("streaming-tpot-v1", equation = "tpot_definition")]
async fn wgpu_chat_completion(
    state: Arc<WgpuInferenceState>,
    axum::Json(body): axum::Json<serde_json::Value>,
) -> axum::response::Response {
    // GH-665: Cap max_tokens to prevent hangs on large values
    let max_tokens = body["max_tokens"].as_u64().unwrap_or(64).min(4096) as usize;
    let stream = body["stream"].as_bool().unwrap_or(false);
    let messages = body["messages"].as_array();

    let prompt = messages
        .and_then(|m| m.last())
        .and_then(|m| m["content"].as_str())
        .unwrap_or("Hello");

    let chat_text = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );

    let prompt_ids: Vec<u32> = if let Some(ref tok) = state.bpe_tokenizer {
        tok.encode(&chat_text)
    } else {
        tokenize_greedy(&chat_text, &state.token_to_id, state.vocab_size)
    };

    let id = format!(
        "chatcmpl-wgpu-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    if stream {
        // PMAT-355: Streaming SSE via spawn_blocking + channel
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);
        let id_clone = id.clone();
        let vocab = state.vocab.clone();
        let prompt_len = prompt_ids.len();
        tokio::task::spawn_blocking(move || {
            let gen_start = std::time::Instant::now();
            let fwd = state.fwd.lock().expect("lock(");
            let mut kv_caches: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
            let mut last_logits = Vec::new();
            // Prefill
            for (pos, &token_id) in prompt_ids.iter().enumerate() {
                match fwd.forward_model(
                    token_id,
                    pos,
                    state.num_layers,
                    &state.token_embedding,
                    &state.output_norm_weight,
                    &state.lm_head_f32,
                    state.vocab_size,
                    1e-6,
                    &mut kv_caches,
                ) {
                    Ok(l) => last_logits = l,
                    Err(_) => return,
                }
            }
            let mut completion_tokens = 0u32;
            // Decode + send each token
            for step in 0..max_tokens {
                let next_token = last_logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
                if next_token == 151645 || next_token == 0 {
                    break;
                }
                let text = wgpu_detokenize_one(next_token, &vocab);
                let chunk = serde_json::json!({
                    "id": id_clone, "object": "chat.completion.chunk", "model": "qwen-wgpu",
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": serde_json::Value::Null}]
                });
                completion_tokens += 1;
                if tx.blocking_send(chunk.to_string()).is_err() {
                    break;
                }
                let position = prompt_ids.len() + step;
                match fwd.forward_model(
                    next_token,
                    position,
                    state.num_layers,
                    &state.token_embedding,
                    &state.output_norm_weight,
                    &state.lm_head_f32,
                    state.vocab_size,
                    1e-6,
                    &mut kv_caches,
                ) {
                    Ok(l) => last_logits = l,
                    Err(_) => break,
                }
            }
            let elapsed = gen_start.elapsed();
            let tok_s = if elapsed.as_secs_f64() > 0.0 {
                completion_tokens as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            let done = serde_json::json!({
                "id": id_clone, "object": "chat.completion.chunk", "model": "qwen-wgpu",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_len, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_len as u32 + completion_tokens},
                "x_wgpu_tok_s": tok_s,
            });
            let _ = tx.blocking_send(done.to_string());
            let _ = tx.blocking_send("[DONE]".to_string());
        });
        let stream = async_stream::stream! {
            while let Some(data) = rx.recv().await {
                yield Ok::<_, std::convert::Infallible>(
                    axum::response::sse::Event::default().data(data)
                );
            }
        };
        axum::response::sse::Sse::new(stream).into_response()
    } else {
        // Non-streaming: original path
        let gen_start = std::time::Instant::now();
        let mut output_ids: Vec<u32> = Vec::new();
        let fwd = state.fwd.lock().expect("lock(");
        let mut kv_caches: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

        let mut last_logits = Vec::new();
        for (pos, &token_id) in prompt_ids.iter().enumerate() {
            match fwd.forward_model(
                token_id,
                pos,
                state.num_layers,
                &state.token_embedding,
                &state.output_norm_weight,
                &state.lm_head_f32,
                state.vocab_size,
                1e-6,
                &mut kv_caches,
            ) {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    return axum::Json(serde_json::json!({"error": format!("{e}")})).into_response()
                }
            }
        }

        for step in 0..max_tokens {
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            if next_token == 151645 || next_token == 0 {
                break;
            }
            output_ids.push(next_token);
            let position = prompt_ids.len() + step;
            match fwd.forward_model(
                next_token,
                position,
                state.num_layers,
                &state.token_embedding,
                &state.output_norm_weight,
                &state.lm_head_f32,
                state.vocab_size,
                1e-6,
                &mut kv_caches,
            ) {
                Ok(logits) => last_logits = logits,
                Err(_) => break,
            }
        }
        drop(fwd);

        let elapsed = gen_start.elapsed();
        let output_text: String = output_ids
            .iter()
            .map(|&id| wgpu_detokenize_one(id, &state.vocab))
            .collect();
        let tok_s = if elapsed.as_secs_f64() > 0.0 {
            output_ids.len() as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        axum::Json(serde_json::json!({
            "id": id, "object": "chat.completion", "model": "qwen-wgpu",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text},
                "finish_reason": if output_ids.len() >= max_tokens { "length" } else { "stop" }}],
            "usage": {"prompt_tokens": prompt_ids.len(), "completion_tokens": output_ids.len(),
                "total_tokens": prompt_ids.len() + output_ids.len()},
            "x_wgpu_latency_ms": elapsed.as_secs_f64() * 1000.0, "x_wgpu_tok_s": tok_s,
        }))
        .into_response()
    }
}

/// PMAT-340: Greedy longest-match tokenizer using vocab lookup.
/// Not BPE — just finds the longest token at each position. Works for
/// special tokens (<|im_start|>) and single characters. Good enough for
/// inference demo; proper BPE needed for production.
#[cfg(feature = "wgpu")]
fn tokenize_greedy(
    text: &str,
    token_to_id: &std::collections::HashMap<String, u32>,
    vocab_size: usize,
) -> Vec<u32> {
    let mut ids = Vec::new();
    let bytes = text.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        let mut best_len = 0;
        let mut best_id = 0u32; // <unk>
                                // Try longest match first (up to 32 bytes)
        let max_len = (bytes.len() - pos).min(32);
        for len in (1..=max_len).rev() {
            if let Ok(s) = std::str::from_utf8(&bytes[pos..pos + len]) {
                if let Some(&id) = token_to_id.get(s) {
                    best_len = len;
                    best_id = id;
                    break;
                }
            }
        }
        if best_len == 0 {
            // Single byte fallback — use byte value as token ID (capped)
            best_id = (bytes[pos] as u32).min(vocab_size as u32 - 1);
            best_len = 1;
        }
        ids.push(best_id);
        pos += best_len;
    }
    ids
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
        println!(
            "{}",
            "PMAT-333: Loading model for WGPU inference...".dimmed()
        );

        // Step 1: Load GGUF model
        use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
        let mapped = MappedGGUFModel::from_path(model_path)
            .map_err(|e| CliError::ModelLoadFailed(format!("GGUF load: {e}")))?;
        let quantized = OwnedQuantizedModel::from_mapped(&mapped)
            .map_err(|e| CliError::ModelLoadFailed(format!("Quantized model: {e}")))?;
        let num_layers = quantized.layers().len();
        println!(
            "{}",
            format!(
                "Model: {} layers loaded for WGPU dequantization",
                num_layers,
            )
            .green()
        );

        // Step 2: Dequantize weights
        let dequant_start = std::time::Instant::now();
        let weights = realizar::gpu::adapters::wgpu_adapter::dequant_model_weights(&quantized)
            .map_err(|e| CliError::ModelLoadFailed(format!("Dequant: {e}")))?;
        let total_mb: f64 = weights
            .iter()
            .map(|(_, d, _, _)| d.len() * 4)
            .sum::<usize>() as f64
            / 1e6;
        println!(
            "{}",
            format!(
                "Dequantized {} weights ({:.0} MB) in {:.1}s",
                weights.len(),
                total_mb,
                dequant_start.elapsed().as_secs_f64(),
            )
            .dimmed()
        );

        // Step 3: WGPU upload + serve
        #[cfg(feature = "wgpu")]
        {
            println!("{}", "Initializing WGPU device...".dimmed());
            let gpu_dev = trueno::backends::gpu::GpuDevice::new()
                .map_err(|e| CliError::ModelLoadFailed(format!("WGPU init: {e}")))?;
            println!("{}", "WGPU device ready (Vulkan/Metal)".green());

            // Get model dims from dequanted weights (avoid private config field)
            // The first Q proj weight has shape [q_dim, hidden_dim]
            let hidden_dim = weights
                .iter()
                .find(|(n, _, _, _)| n.ends_with(".q_proj"))
                .map(|(_, _, _, cols)| *cols)
                .unwrap_or(1536);
            let intermediate_dim = weights
                .iter()
                .find(|(n, _, _, _)| n.ends_with(".gate_proj"))
                .map(|(_, _, rows, _)| *rows)
                .unwrap_or(8960);
            // Infer heads from Q proj: q_dim = num_heads * head_dim
            let q_dim = weights
                .iter()
                .find(|(n, _, _, _)| n.ends_with(".q_proj"))
                .map(|(_, _, rows, _)| *rows)
                .unwrap_or(hidden_dim);
            let kv_dim = weights
                .iter()
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
            // PMAT-367: Q4K mode saves 10× VRAM but ~3× slower (compute-bound nibble extraction)
            let use_q4k = std::env::var("WGPU_Q4K").is_ok();
            if use_q4k {
                let q4k_raw = realizar::gpu::adapters::wgpu_adapter::raw_q4k_weights(&quantized);
                let q4k_names: std::collections::HashSet<String> =
                    q4k_raw.iter().map(|(n, _, _, _)| n.clone()).collect();
                for (name, raw_data, _rows, _cols) in &q4k_raw {
                    fwd.upload_q4k_weight(name, raw_data);
                }
                for (name, data, _rows, _cols) in &weights {
                    if q4k_names.contains(name.as_str()) {
                        continue;
                    }
                    fwd.upload_weight(name, data);
                }
                let q4k_mb: f64 =
                    q4k_raw.iter().map(|(_, d, _, _)| d.len()).sum::<usize>() as f64 / 1e6;
                println!(
                    "{}",
                    format!(
                        "Q4K mode: {} Q4K ({:.0} MB) — 10× VRAM savings",
                        q4k_raw.len(),
                        q4k_mb
                    )
                    .cyan()
                );
            } else {
                for (name, data, _rows, _cols) in &weights {
                    fwd.upload_weight(name, data);
                }
            }
            // PMAT-361: Allocate GPU KV cache buffers
            fwd.init_kv_cache(num_layers);
            println!(
                "{}",
                format!(
                    "Uploaded {} weights to GPU ({:.1} MB VRAM) in {:.1}ms",
                    weights.len(),
                    fwd.total_vram_bytes() as f64 / 1e6,
                    upload_start.elapsed().as_secs_f64() * 1000.0,
                )
                .green()
            );

            // Step 4: Extract CPU-side data for forward_model
            let token_embedding = quantized.token_embedding().to_vec();
            let output_norm_weight = quantized.output_norm_weight().to_vec();
            let vocab_size = token_embedding.len() / hidden_dim;

            // LM head from dequanted weights
            let lm_head_f32 = weights
                .iter()
                .find(|(n, _, _, _)| n == "lm_head")
                .map(|(_, d, _, _)| d.clone())
                .unwrap_or_else(|| token_embedding.clone()); // tied embeddings fallback

            println!(
                "{}",
                format!(
                    "WGPU inference ready: {} layers, vocab={}, hidden={}",
                    num_layers, vocab_size, hidden_dim,
                )
                .green()
            );

            // Step 5: Quick correctness test — generate one token
            let test_token = 9707u32; // "Hello" in Qwen tokenizer
            let test_start = std::time::Instant::now();
            let mut test_kv: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
            match fwd.forward_model(
                test_token,
                0,
                num_layers,
                &token_embedding,
                &output_norm_weight,
                &lm_head_f32,
                vocab_size,
                1e-6,
                &mut test_kv,
            ) {
                Ok(logits) => {
                    let argmax = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).expect("1"))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let elapsed = test_start.elapsed();
                    println!(
                        "{}",
                        format!(
                            "WGPU test: token {} → logits[{}] max at idx {} ({:.1}ms)",
                            test_token,
                            logits.len(),
                            argmax,
                            elapsed.as_secs_f64() * 1000.0,
                        )
                        .cyan()
                    );
                }
                Err(e) => {
                    println!("{}", format!("WGPU forward failed: {e}").red());
                }
            }

            // PMAT-339: WGPU HTTP serve — minimal /v1/chat/completions endpoint
            println!("{}", "Starting WGPU inference server...".cyan());

            // PMAT-340: Extract real vocab from GGUF for tokenization/detokenization
            let vocab: Vec<String> = mapped.model.vocabulary().unwrap_or_else(|| {
                eprintln!("Warning: No vocabulary in GGUF, using placeholder");
                let mut v: Vec<String> = (0..vocab_size).map(|i| format!("token{i}")).collect();
                if !v.is_empty() {
                    v[0] = "<unk>".to_string();
                }
                v
            });
            // PMAT-341: Extract BPE merge rules for proper tokenization
            let merges = mapped.model.merge_rules().unwrap_or_default();
            println!(
                "{}",
                format!(
                    "Vocab: {} tokens, {} merge rules from GGUF",
                    vocab.len(),
                    merges.len()
                )
                .dimmed()
            );

            // Create BPE tokenizer with merge rules
            let bpe_tokenizer = if !merges.is_empty() {
                match {
                    let mut t2id = std::collections::HashMap::new();
                    for (i, tok) in vocab.iter().enumerate() {
                        t2id.insert(tok.clone(), i as u32);
                    }
                    let special: std::collections::HashMap<String, u32> = vocab
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.starts_with("<|") && t.ends_with("|>"))
                        .map(|(i, t)| (t.clone(), i as u32))
                        .collect();
                    Ok::<_, String>(realizar::apr::BpeTokenizer {
                        token_to_id: t2id,
                        id_to_token: vocab.clone(),
                        merge_rules: merges,
                        bos_id: None,
                        eos_id: Some(151645), // Qwen2 EOS
                        special_tokens: special,
                    })
                } {
                    Ok(tok) => {
                        println!("{}", "BPE tokenizer created with merge rules".green());
                        Some(tok)
                    }
                    Err(e) => {
                        eprintln!("BPE tokenizer failed: {e} — falling back to greedy");
                        None
                    }
                }
            } else {
                println!("{}", "No merge rules — using greedy tokenizer".yellow());
                None
            };

            // Build token→ID map for greedy fallback
            let token_to_id: std::collections::HashMap<String, u32> = vocab
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i as u32))
                .collect();

            // Wrap inference state in Arc for axum handler
            use std::sync::{Arc, Mutex};
            let wgpu_state = Arc::new(WgpuInferenceState {
                fwd: Mutex::new(fwd),
                token_embedding,
                output_norm_weight,
                lm_head_f32,
                vocab,
                token_to_id,
                bpe_tokenizer,
                num_layers,
                vocab_size,
                hidden_dim,
            });

            // Build minimal axum router
            use axum::{
                routing::{get, post},
                Json, Router,
            };

            let state_health = wgpu_state.clone();
            let state_chat = wgpu_state.clone();
            // PMAT-923: Ollama endpoints reuse the SAME wgpu chat backend.
            let state_ollama_chat = wgpu_state.clone();
            let state_ollama_generate = wgpu_state.clone();

            let app = Router::new()
                .route(
                    "/health",
                    get(move || async move {
                        Json(serde_json::json!({
                            "status": "healthy",
                            "compute_mode": "wgpu",
                            "version": env!("CARGO_PKG_VERSION"),
                        }))
                    }),
                )
                .route(
                    "/v1/chat/completions",
                    post(move |body: Json<serde_json::Value>| {
                        let state = state_chat.clone();
                        async move { wgpu_chat_completion(state, body).await }
                    }),
                )
                // PMAT-923/928: Ollama native chat endpoint (drop-in Ollama
                // client). WGPU backend is batch, so `stream:true` emits NDJSON
                // framing over the coalesced result; `stream:false` coalesces.
                .route(
                    "/api/chat",
                    post(move |Json(req): Json<super::ollama::OllamaChatRequest>| {
                        let state = state_ollama_chat.clone();
                        async move {
                            let model = super::ollama::model_label(&req.model);
                            let stream = req.stream;
                            let openai_body = super::ollama::ollama_chat_to_openai(&req);
                            let inner = wgpu_chat_completion(state, Json(openai_body)).await;
                            if stream {
                                super::ollama::reshape_openai_to_ollama_ndjson(
                                    super::ollama::OllamaStreamKind::Chat,
                                    model,
                                    inner,
                                )
                                .await
                            } else {
                                super::ollama::reshape_openai_to_ollama_chat(model, inner).await
                            }
                        }
                    }),
                )
                // PMAT-923/928: Ollama native single-prompt generate endpoint.
                .route(
                    "/api/generate",
                    post(move |Json(req): Json<super::ollama::OllamaGenerateRequest>| {
                        let state = state_ollama_generate.clone();
                        async move {
                            let model = super::ollama::model_label(&req.model);
                            let stream = req.stream;
                            let openai_body = super::ollama::ollama_generate_to_openai(&req);
                            let inner = wgpu_chat_completion(state, Json(openai_body)).await;
                            if stream {
                                super::ollama::reshape_openai_to_ollama_ndjson(
                                    super::ollama::OllamaStreamKind::Generate,
                                    model,
                                    inner,
                                )
                                .await
                            } else {
                                super::ollama::reshape_openai_to_ollama_generate(model, inner).await
                            }
                        }
                    }),
                );

            let bind_addr = config.bind_addr();
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| CliError::InferenceFailed(format!("Runtime: {e}")))?;

            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::bind(&bind_addr)
                    .await
                    .map_err(|e| CliError::InferenceFailed(format!("Bind: {e}")))?;
                println!(
                    "{}",
                    format!("WGPU inference server listening on http://{}", bind_addr)
                        .green()
                        .bold()
                );
                println!("  POST /v1/chat/completions - Chat completions (WGPU)");
                println!("  GET  /health              - Health check");
                axum::serve(listener, app)
                    .await
                    .map_err(|e| CliError::InferenceFailed(format!("Serve: {e}")))?;
                Ok::<(), CliError>(())
            })?;
            return Ok(());
        }
        #[cfg(not(feature = "wgpu"))]
        {
            println!(
                "{}",
                "WGPU feature not enabled. Build with --features wgpu".yellow()
            );
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
        let use_gpu = config.wants_accelerator();
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
    /// PMAT-928 test seam ONLY: a scripted per-token text sequence used to drive
    /// the NDJSON streaming path without a real transformer. Production state
    /// always leaves this `None`; only `build_demo_apr_cpu_router_for_test` sets
    /// it, so the streaming falsifier can observe real multi-chunk NDJSON.
    demo_scripted_tokens: Option<Vec<String>>,
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
        cancel: realizar::generate::CancelToken::never(),
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
        demo_scripted_tokens: None,
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
    // GH-471: For Q4K APR models, try the pool-allocator path first. The
    // generic OwnedQuantizedModel::from_apr() path hangs on large (17GB+ /
    // 18k tensor) MoE models because it does per-tensor cuMemAlloc; the
    // ALB-095/098 Q4K scheduler does a single pool cuMemAlloc and loads
    // the same 30B Q4K MoE in ~12s vs hanging indefinitely.
    //
    // The Q4K path's spawn_apr_q4k_inference_thread does its own metadata
    // validation (`parse_apr_q4k_config`) and weight-format check
    // (`upload_apr_q4k_weights` rejects non-Q4K tensors), so passing a
    // non-Q4K APR errors cleanly and falls through to the generic GPU path.
    //
    // Without `cuda-batch` feature this function is a stub that errors
    // immediately, so the fallback chain stays identical for non-batch builds.
    #[cfg(feature = "cuda")]
    {
        let use_gpu = config.wants_accelerator();
        if use_gpu {
            match start_apr_q4k_server_gpu(model_path, config) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    // Common case for non-Q4K APR — fall through quietly to
                    // the generic GPU path (which handles dequant + small models).
                    eprintln!(
                        "[GH-471] Q4K pool-allocator path declined ({e}), trying generic GPU path"
                    );
                }
            }
            // GH-87: fused Q4K GPU path (same kernels as GGUF, 190+ tok/s).
            // Handles dequantized + non-Q4K APR via OwnedQuantizedModel::from_apr.
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
        let app = build_apr_cpu_router(state, super::auth::AuthGate::from_env());

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

    // aprender#1789 Option B: APR format has no GGUF mmap to retain, so
    // pass None — the qwen3_moe dispatch guard returns NOT_IMPLEMENTED
    // cleanly if an APR-format MoE model is ever loaded (defensive
    // fallback path in try_qwen3_moe_backend).
    run_cpu_server(quantized, vocab, None, config)
}

/// Build the axum Router for APR CPU inference.
///
/// GH-284: Handlers are async with `spawn_blocking` to avoid blocking the
/// tokio runtime during multi-second generation.
///
/// HELIX-IDEA-009 / FALSIFY-AUTH-001: every route on the returned router
/// is gated by `auth_gate` via `super::auth::layer`. Pass
/// `AuthGate::disabled()` to bypass.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap
fn build_apr_cpu_router(state: AprServerState, auth_gate: super::auth::AuthGate) -> axum::Router {
    use axum::{
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use std::sync::Mutex;

    let state_for_health = state.clone();
    let model_name_for_tags = state.model_name.clone();
    let state_for_completions = Arc::new(Mutex::new(state.clone()));
    let state_for_chat = Arc::new(Mutex::new(state));
    // PMAT-923: Ollama `/api/chat` + `/api/generate` reuse the SAME chat backend
    // as `/v1/chat/completions` (handle_apr_cpu_chat_completion), then re-shape
    // the OpenAI response into Ollama's wire schema.
    let state_for_ollama_chat = state_for_chat.clone();
    let state_for_ollama_generate = state_for_chat.clone();

    let router = Router::new()
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
        // PMAT-923/928: Ollama native chat endpoint — drop-in for an Ollama
        // client. `stream != false` (Ollama default) ⇒ NDJSON token stream;
        // `stream:false` ⇒ coalesced single object.
        .route(
            "/api/chat",
            post(move |Json(req): Json<super::ollama::OllamaChatRequest>| {
                let state = state_for_ollama_chat.clone();
                async move { handle_apr_cpu_ollama_chat(&state, &req).await }
            }),
        )
        // PMAT-923/928: Ollama native single-prompt generate endpoint.
        .route(
            "/api/generate",
            post(move |Json(req): Json<super::ollama::OllamaGenerateRequest>| {
                let state = state_for_ollama_generate.clone();
                async move { handle_apr_cpu_ollama_generate(&state, &req).await }
            }),
        )
        // PMAT-923: Ollama model-list — clients enumerate models before chatting.
        .route(
            "/api/tags",
            get(move || {
                let model = model_name_for_tags.clone();
                async move { Json(super::ollama::ollama_tags_body(&model)) }
            }),
        )
        .route(
            "/",
            get(|| async {
                "APR v2 Inference Server - POST /v1/completions, /v1/chat/completions, /api/chat, /api/generate"
            }),
        )
        // GH-672: Return JSON error body for unmatched routes (not empty 404)
        .fallback(|| async {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": "not_found",
                    "message": "Route not found. Available: /health, /v1/completions, /v1/chat/completions, /api/chat, /api/generate, /api/tags"
                })),
            )
        });
    let router = super::ollama::add_ollama_stubs(router);
    super::auth::layer(auth_gate, router)
}

/// PMAT-923 e2e test seam: build the REAL `apr serve` APR-CPU router with a
/// no-transformer demo state, exercising the exact router `apr serve <model.apr>`
/// mounts (including the `/api/chat` + `/api/generate` Ollama routes).
///
/// With `transformer: None`, generation returns a backend error which the chat
/// path surfaces as a 200 JSON `{error:...}` body — the Ollama adapter re-shapes
/// that into a terminal (`done:true`) Ollama body. A wired Ollama route is thus
/// observably distinct from the axum 404 fallback (which has no `done` field)
/// even with no model loaded, which is exactly what the falsifier asserts.
#[doc(hidden)]
#[cfg(feature = "inference")]
pub fn build_demo_apr_cpu_router_for_test() -> axum::Router {
    let state = AprServerState {
        transformer: None,
        model_type: "demo".to_string(),
        architecture: "demo".to_string(),
        is_transformer: false,
        tokenizer: None,
        embedded_tokenizer: None,
        model_name: "apr".to_string(),
        demo_scripted_tokens: None,
    };
    build_apr_cpu_router(state, super::auth::AuthGate::disabled())
}

/// PMAT-928 e2e test seam: build the REAL `apr serve` APR-CPU router whose
/// streaming path is driven by a deterministic scripted token sequence (in
/// place of a real transformer). This lets the NDJSON-streaming falsifier
/// observe MULTIPLE per-token `done:false` chunks plus a terminal `done:true`
/// over the SAME router `apr serve <model.apr>` mounts — proving real
/// incremental (not coalesced) streaming without loading a model.
///
/// The scripted tokens flow through the IDENTICAL `spawn_cpu_token_text_stream`
/// → mpsc channel → NDJSON-reshape path the production transformer uses; only
/// the token *source* is faked, never the streaming wire format.
#[doc(hidden)]
#[cfg(feature = "inference")]
pub fn build_demo_streaming_apr_cpu_router_for_test() -> axum::Router {
    let state = AprServerState {
        transformer: None,
        model_type: "demo".to_string(),
        architecture: "demo".to_string(),
        is_transformer: false,
        tokenizer: None,
        embedded_tokenizer: None,
        model_name: "apr".to_string(),
        demo_scripted_tokens: Some(vec![
            "Hello".to_string(),
            ", ".to_string(),
            "world".to_string(),
            "!".to_string(),
        ]),
    };
    build_apr_cpu_router(state, super::auth::AuthGate::disabled())
}

include!("handler_apr_cpu_completion.rs");
include!("handler_gpu_completion.rs");
include!("server.rs");
