//! ALB-095: APR Q4K GPU inference scheduler for HTTP serving.
//!
//! Spawns a dedicated thread that owns the CudaExecutor and model weights.
//! Requests are sent via channel; responses returned via oneshot.
//! This sidesteps CudaExecutor being `!Send` (raw CUDA pointers).
//!
//! # Cancellation (aprender#2465(1) — aprender#2376(3) on the path the fix missed)
//!
//! This backend serves `POST /v1/chat/completions`, `POST /v1/completions`,
//! `POST /generate` and — because the Ollama handlers delegate to the OpenAI chat
//! handler — `/api/chat` and `/api/generate`. Every one of those reached
//! [`generate_q4k`] with **no cancellation signal at all**: [`AprQ4kRequest`] had no
//! `cancel` field, so the decode loop's only exit was EOS and an abandoned request
//! burned the GPU to `max_tokens` for nobody.
//!
//! Neither of the two mechanisms documented in
//! `crates/aprender-serve/src/api/cancel_scope.rs` covered it on its own:
//!
//! - the handler's response future being dropped cannot reach a loop running on
//!   *another thread* — moving work off-task is not the same as stopping it; and
//! - the send-failure mechanism that stops streaming loops does not apply either,
//!   because this scheduler accumulates `output_tokens` and sends **one**
//!   [`AprQ4kResponse`] at the end. There is no per-token send left to fail.
//!
//! So the request carries the token and [`q4k_decode`] polls it once per decode
//! step, exactly like `layers/model_model.rs::generate` and
//! `gguf/inference/generate_quantized.rs`.
//!
//! Contract: `contracts/apr-serve-cancellation-v1.yaml`
//! (FALSIFY-SERVE-CANCEL-009/010/011).

use crate::generate::CancelToken;

/// Request to generate tokens from a prompt.
#[cfg(feature = "cuda")]
pub struct AprQ4kRequest {
    /// Tokenized prompt IDs.
    pub prompt_ids: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// EOS token IDs — generation stops when any of these are produced.
    /// ALB-109: Qwen3 uses 151643 (<|endoftext|>), not 0 or 2.
    pub eos_ids: Vec<u32>,
    /// aprender#2465(1): the requesting HTTP handler's cancellation token.
    ///
    /// Required rather than `Option`, so a new call site cannot silently submit
    /// work that runs on after its client hangs up. Pass the request's
    /// `Extension<CancelToken>`; [`CancelToken::never`] means "run to completion".
    pub cancel: CancelToken,
    /// Channel to send the response back.
    pub response_tx: tokio::sync::oneshot::Sender<Result<AprQ4kResponse, String>>,
}

/// Response from the Q4K inference thread.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct AprQ4kResponse {
    /// All generated token IDs (excluding prompt).
    pub output_tokens: Vec<u32>,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Generation time in milliseconds.
    pub generation_time_ms: f64,
    /// Tokens per second.
    pub tokens_per_second: f64,
}

/// Spawn a dedicated Q4K GPU inference thread.
///
/// Loads the APR model, uploads Q4K weights to GPU, and processes
/// requests sequentially on the CUDA thread (no tokio, no Send needed).
///
/// Returns a sender for submitting requests. The thread runs until
/// the sender is dropped.
#[cfg(feature = "cuda")]
pub fn spawn_apr_q4k_inference_thread(
    model_path: &str,
) -> Result<tokio::sync::mpsc::Sender<AprQ4kRequest>, String> {
    use crate::apr::AprV2Model;
    use crate::cuda::CudaExecutor;
    use crate::gpu::adapters::apr_q4k::{
        parse_apr_q4k_config, upload_apr_q4k_weights, AprQ4KConfig,
    };
    use std::path::Path;

    let model_path_owned = model_path.to_string();

    // Load model and upload weights on the current thread first,
    // so we can report errors synchronously.
    let path = Path::new(&model_path_owned);
    let model = AprV2Model::load(path).map_err(|e| format!("Failed to load APR: {e}"))?;
    let config =
        parse_apr_q4k_config(&model).map_err(|e| format!("Failed to parse config: {e}"))?;

    println!(
        "  Q4K GPU: {} layers, hidden={}, heads={}/{}, vocab={}",
        config.num_layers,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.vocab_size
    );
    if let Some(ne) = config.num_experts {
        println!(
            "  MoE: {} experts, top-{}, intermediate={}",
            ne,
            config.num_experts_per_tok.unwrap_or(0),
            config.moe_intermediate_size.unwrap_or(0)
        );
    }

    let mut executor = CudaExecutor::new(0).map_err(|e| format!("CUDA init failed: {e}"))?;
    let upload_result = upload_apr_q4k_weights(&model, &mut executor)
        .map_err(|e| format!("Weight upload failed: {e}"))?;

    println!(
        "  Uploaded {} tensors ({} Q4K, {} F32) — {:.1} MB VRAM",
        upload_result.num_tensors,
        upload_result.num_q4k_tensors,
        upload_result.num_f32_tensors,
        upload_result.total_bytes as f64 / (1024.0 * 1024.0)
    );

    // Extract CPU-side weights (embedding, norms)
    // Use find_tensor_name to handle GGUF/SafeTensors/HF naming variants (#167)
    let embed_name = model
        .find_tensor_name(&[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
            "token_embd.weight",
        ])
        .map_err(|e| format!("Missing embedding: {e}"))?;
    let embedding_weight = model
        .get_tensor_f32(&embed_name)
        .map_err(|e| format!("Missing embedding: {e}"))?;

    let norm_name = model
        .find_tensor_name(&[
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight",
        ])
        .map_err(|e| format!("Missing output norm: {e}"))?;
    let output_norm_weight = model
        .get_tensor_f32(&norm_name)
        .map_err(|e| format!("Missing output norm: {e}"))?;

    let mut layer_norm_weights: Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>)> =
        Vec::with_capacity(config.num_layers);
    for layer_idx in 0..config.num_layers {
        let attn_norm_name = model
            .find_tensor_name(&[
                &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                &format!("layers.{layer_idx}.input_layernorm.weight"),
                &format!("blk.{layer_idx}.attn_norm.weight"),
            ])
            .map_err(|e| format!("Missing attn norm layer {layer_idx}: {e}"))?;
        let attn_norm = model
            .get_tensor_f32(&attn_norm_name)
            .map_err(|e| format!("Missing attn norm layer {layer_idx}: {e}"))?;

        let ffn_norm_name = model
            .find_tensor_name(&[
                &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("layers.{layer_idx}.post_attention_layernorm.weight"),
                &format!("blk.{layer_idx}.ffn_norm.weight"),
            ])
            .map_err(|e| format!("Missing FFN norm layer {layer_idx}: {e}"))?;
        let ffn_norm = model
            .get_tensor_f32(&ffn_norm_name)
            .map_err(|e| format!("Missing FFN norm layer {layer_idx}: {e}"))?;

        let q_norm = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.q_norm.weight"))
            .ok();
        let k_norm = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.k_norm.weight"))
            .ok();
        layer_norm_weights.push((attn_norm, ffn_norm, q_norm, k_norm));
    }

    // PMAT-315: Extract QKV biases (required for Qwen2, optional for LLaMA/Mistral)
    let mut layer_qkv_biases: Vec<(Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>)> =
        Vec::with_capacity(config.num_layers);
    for layer_idx in 0..config.num_layers {
        let q_bias = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.q_proj.bias"))
            .ok();
        let k_bias = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.k_proj.bias"))
            .ok();
        let v_bias = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.v_proj.bias"))
            .ok();
        layer_qkv_biases.push((q_bias, k_bias, v_bias));
    }

    // Release mmap pages — weights are on GPU now
    let _ = model.release_cpu_pages();

    // Load tokenizer for decode (used on the inference thread)
    let tokenizer = AprV2Model::load_tokenizer(path);

    println!("  Q4K GPU inference thread: ready");

    // Create async-compatible channel (tokio mpsc is Send)
    let (tx, mut rx) = tokio::sync::mpsc::channel::<AprQ4kRequest>(64);

    // Spawn dedicated thread — owns executor and all CUDA state
    std::thread::spawn(move || {
        // ALB-110: CUDA contexts are thread-local. The executor was created on
        // the calling thread (where cuCtxSetCurrent was called). On this new
        // thread, the context is NOT current. Without this call, CUDA driver
        // operations (cuMemAlloc, kernel launches, cuMemFree) silently corrupt
        // GPU state and crash after ~12-37 requests.
        executor
            .make_context_current()
            .expect("Q4K inference thread: failed to set CUDA context");

        // Create a minimal tokio runtime just for channel recv
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Q4K inference thread: failed to create tokio runtime");

        rt.block_on(async move {
            while let Some(req) = rx.recv().await {
                let result = generate_q4k(
                    &mut executor,
                    &config,
                    &embedding_weight,
                    &output_norm_weight,
                    &layer_norm_weights,
                    &layer_qkv_biases,
                    &req.prompt_ids,
                    req.max_tokens,
                    req.temperature,
                    &req.eos_ids,
                    &req.cancel,
                );
                let _ = req.response_tx.send(result);
            }
            eprintln!("[Q4K] Inference thread shutting down (channel closed)");
        });
    });

    Ok(tx)
}

/// Run a single Q4K generation request (called on the inference thread).
#[cfg(feature = "cuda")]
fn generate_q4k(
    executor: &mut crate::cuda::CudaExecutor,
    config: &crate::gpu::adapters::apr_q4k::AprQ4KConfig,
    embedding_weight: &[f32],
    output_norm_weight: &[f32],
    layer_norm_weights: &[(Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>)],
    layer_qkv_biases: &[(Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>)],
    prompt_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    eos_ids: &[u32],
    cancel: &CancelToken,
) -> Result<AprQ4kResponse, String> {
    use crate::cli::inference::{argmax, sample_with_temperature};
    use crate::gpu::adapters::apr_q4k::forward_token_apr_q4k;
    use std::time::Instant;

    // Fresh KV cache per request
    let mut kv_cache_k: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers];
    let mut kv_cache_v: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers];

    let gen_start = Instant::now();

    // Prefill: process all prompt tokens
    let mut last_logits = Vec::new();
    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        last_logits = forward_token_apr_q4k(
            executor,
            config,
            embedding_weight,
            output_norm_weight,
            layer_norm_weights,
            layer_qkv_biases,
            &mut kv_cache_k,
            &mut kv_cache_v,
            token_id,
            pos,
        )
        .map_err(|e| format!("Prefill failed at pos {pos}: {e}"))?;
    }

    // Sample first token
    let first_token = if temperature <= 0.01 {
        argmax(&last_logits)
    } else {
        sample_with_temperature(&last_logits, temperature, 40)
    };

    // Autoregressive decode. The loop itself lives in `q4k_decode` so that the
    // loop which ships is the loop the falsifiers drive (aprender#2465(1)) —
    // everything CUDA-specific stays here, inside the step closure.
    let output_tokens = q4k_decode(
        first_token,
        prompt_ids.len(),
        max_tokens,
        eos_ids,
        cancel,
        |token, position, step| {
            let logits = forward_token_apr_q4k(
                executor,
                config,
                embedding_weight,
                output_norm_weight,
                layer_norm_weights,
                layer_qkv_biases,
                &mut kv_cache_k,
                &mut kv_cache_v,
                token,
                position,
            )
            .map_err(|e| format!("Decode failed at step {step}: {e}"))?;

            Ok(if temperature <= 0.01 {
                argmax(&logits)
            } else {
                sample_with_temperature(&logits, temperature, 40)
            })
        },
    )?;

    let gen_time = gen_start.elapsed();
    let tokens_generated = output_tokens.len();
    let tokens_per_second = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(AprQ4kResponse {
        output_tokens,
        tokens_generated,
        generation_time_ms: gen_time.as_secs_f64() * 1000.0,
        tokens_per_second,
    })
}

/// The Q4K scheduler's autoregressive decode loop.
///
/// `first_token` is the token sampled from the prefill logits; it is always part
/// of the output, so an uncancelled run returns exactly `max_tokens` tokens
/// (`first_token` plus `max_tokens - 1` decode steps) unless EOS or cancellation
/// stops it earlier.
///
/// `step(token, position, step_idx)` performs one decode step and returns the next
/// sampled token. In production it closes over the `CudaExecutor` and the uploaded
/// Q4K weights; in the falsifiers it is a pure function. That is the whole point of
/// the split: it is the same loop either way, so FALSIFY-SERVE-CANCEL-009/010 can
/// assert **token counts** on the shipped control flow without a GPU. Nothing about
/// the scheduler's thread/channel/oneshot architecture changes.
///
/// # Cancellation
///
/// `cancel` is polled once at the top of each decode step, **before** that step's
/// forward pass — matching `layers/model_model.rs::generate` and
/// `gguf/inference/generate_quantized.rs`. Polling at the bottom instead would cost
/// one wasted forward pass per cancelled request, which FALSIFY-SERVE-CANCEL-010
/// detects.
///
/// # Errors
///
/// Propagates whatever `step` returns, unchanged.
pub(crate) fn q4k_decode<F>(
    first_token: u32,
    prompt_len: usize,
    max_tokens: usize,
    eos_ids: &[u32],
    cancel: &CancelToken,
    mut step: F,
) -> Result<Vec<u32>, String>
where
    F: FnMut(u32, usize, usize) -> Result<u32, String>,
{
    let mut next_token = first_token;
    let mut output_tokens = vec![next_token];

    for step_idx in 0..max_tokens.saturating_sub(1) {
        // aprender#2465(1)/#2376(3): CANCELLATION POLL. The HTTP client may be
        // gone. This loop runs on the dedicated CUDA thread, so neither the
        // handler future's drop nor a failed per-token send can reach it — the
        // poll is the only thing that stops it burning the GPU to max_tokens.
        // aprender#2465(1)/#2376(3): CANCELLATION POLL. The HTTP client may be
        // gone. This loop runs on the dedicated CUDA thread, so neither the
        // handler future's drop nor a failed per-token send can reach it — the
        // poll is the only thing that stops it burning the GPU to max_tokens.
        if cancel.is_cancelled() {
            break;
        }

        // ALB-109: Configurable EOS — Qwen3 uses 151643, not 0/2
        if eos_ids.contains(&next_token) {
            break;
        }

        next_token = step(next_token, prompt_len + step_idx, step_idx)?;
        output_tokens.push(next_token);
    }

    Ok(output_tokens)
}

#[cfg(test)]
#[path = "tests/apr_q4k_cancel_2465.rs"]
mod apr_q4k_cancel_2465;
