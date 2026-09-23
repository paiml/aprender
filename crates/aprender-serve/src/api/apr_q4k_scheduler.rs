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
    /// RNG seed for a sampled step (#3786): the same request and seed give the same tokens.
    pub seed: u64,
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

    let Q4kHostWeights {
        embedding: embedding_weight,
        output_norm: output_norm_weight,
        layer_norms: layer_norm_weights,
        qkv_biases: layer_qkv_biases,
    } = load_q4k_host_weights(&model, config.num_layers)?;

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
                    req.seed,
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

#[cfg(feature = "cuda")]
/// The weights the Q4K forward reads on the HOST: the embedding table, the final
/// norm, and per layer the two norms, the optional Q/K norms and the optional QKV
/// biases. The serve thread and the parity test load them through this one
/// function, so the test drives exactly what `apr serve` drives (#3791).
pub(crate) struct Q4kHostWeights {
    pub(crate) embedding: Vec<f32>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) layer_norms: Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>)>,
    pub(crate) qkv_biases: Vec<(Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>)>,
}

#[cfg(feature = "cuda")]
/// Load [`Q4kHostWeights`] for `num_layers` layers from `model`.
///
/// # Errors
/// A required tensor is missing, or an optional one is present and will not read.
pub(crate) fn load_q4k_host_weights(
    model: &crate::apr::AprV2Model,
    num_layers: usize,
) -> Result<Q4kHostWeights, String> {
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
        Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
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

        let q_norm = optional_f32(
            &model,
            &[
                &format!("model.layers.{layer_idx}.self_attn.q_norm.weight"),
                &format!("blk.{layer_idx}.attn_q_norm.weight"),
            ],
        )?;
        let k_norm = optional_f32(
            &model,
            &[
                &format!("model.layers.{layer_idx}.self_attn.k_norm.weight"),
                &format!("blk.{layer_idx}.attn_k_norm.weight"),
            ],
        )?;
        layer_norm_weights.push((attn_norm, ffn_norm, q_norm, k_norm));
    }

    // PMAT-315: Extract QKV biases (required for Qwen2, optional for LLaMA/Mistral)
    let mut layer_qkv_biases: Vec<(Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>)> =
        Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        let bias = |hf: &str, gguf: &str| {
            optional_f32(
                &model,
                &[
                    &format!("model.layers.{layer_idx}.self_attn.{hf}.bias"),
                    &format!("blk.{layer_idx}.{gguf}.bias"),
                ],
            )
        };
        let q_bias = bias("q_proj", "attn_q")?;
        let k_bias = bias("k_proj", "attn_k")?;
        let v_bias = bias("v_proj", "attn_v")?;
        layer_qkv_biases.push((q_bias, k_bias, v_bias));
    }
    Ok(Q4kHostWeights {
        embedding: embedding_weight,
        output_norm: output_norm_weight,
        layer_norms: layer_norm_weights,
        qkv_biases: layer_qkv_biases,
    })
}

#[cfg(feature = "cuda")]
/// #3791: an OPTIONAL per-layer tensor (QKV bias, Q/K norm) under any of its names.
///
/// `Ok(None)` means the file has none of the names — the architecture has no such
/// tensor. A name that exists but will not read is an error. This used to be
/// `get_tensor_f32(<HF name>).ok()`: a GGUF-named file (`blk.N.attn_q.bias`, which is
/// what an imported Qwen2 `.apr` carries — 84 of them) read as having NO biases, and
/// the forward decoded "helf helf helf…" without a word.
fn optional_f32(
    model: &crate::apr::AprV2Model,
    names: &[&str],
) -> Result<Option<Vec<f32>>, String> {
    match model.find_tensor_name(names) {
        Err(_) => Ok(None),
        Ok(name) => model
            .get_tensor_f32(&name)
            .map(Some)
            .map_err(|e| format!("'{name}' is in the file but would not read: {e}")),
    }
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
    seed: u64,
    eos_ids: &[u32],
    cancel: &CancelToken,
) -> Result<AprQ4kResponse, String> {
    use crate::cli::inference::argmax;
    use crate::gpu::adapters::apr_q4k::forward_token_apr_q4k;
    use rand::SeedableRng;
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

    // Sample first token. #3786: one RNG per request, seeded from the request, so the
    // same request and seed give the same tokens (the old sampler hashed the wall clock).
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let first_token = if temperature <= 0.01 {
        argmax(&last_logits)
    } else {
        q4k_sampled_token(&last_logits, temperature, &mut rng)
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
                q4k_sampled_token(&logits, temperature, &mut rng)
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
/// The top-k the APR Q4K chat path samples with.
const Q4K_TOP_K: usize = 40;

/// A sampled APR Q4K step: one seeded draw through the shared sampler (#3786).
///
/// This replaced `cli::inference::sample_with_temperature`, whose uniform draw was a hash
/// of `SystemTime::now()`: a request's `seed` never reached it, and the same request
/// gave different tokens on every call. Kept outside the `cuda` gate so the draw the
/// GPU loop makes is testable on any host.
pub(crate) fn q4k_sampled_token(
    logits: &[f32],
    temperature: f32,
    rng: &mut rand::rngs::StdRng,
) -> u32 {
    crate::sampling::draw_seeded(logits, temperature, Q4K_TOP_K, 1.0, rng)
}

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

/// #3786: the APR Q4K sampled step draws from the request's seeded RNG.
#[cfg(test)]
mod sampled_token_3786 {
    use super::q4k_sampled_token;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const LOGITS: [f32; 6] = [1.0, 0.9, 1.1, 0.95, 1.05, 0.85];

    fn run(seed: u64) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..32)
            .map(|_| q4k_sampled_token(&LOGITS, 1.0, &mut rng))
            .collect()
    }

    /// The same seed reproduces the draws byte for byte: the wall-clock sampler it
    /// replaced could not, whatever the request said.
    #[test]
    fn the_same_seed_reproduces_the_sampled_tokens() {
        assert_eq!(run(7), run(7));
    }

    /// A different seed changes them, so the seed is actually read.
    #[test]
    fn a_different_seed_changes_the_sampled_tokens() {
        assert_ne!(run(7), run(8));
    }

    /// It is a draw, not the argmax (index 2) in disguise.
    #[test]
    fn a_sampled_step_draws_off_the_argmax() {
        assert!(run(7).iter().any(|&t| t != 2));
    }
}

/// #3791: the APR Q4K pool path (ALB-095) against the CPU forward, and its
/// health. Both measured wrong at 0.69.0: NaN logits (every Q6_K weight decoded
/// as Q4_K), then degenerate text (every GGUF-named QKV bias dropped), and
/// `/health` 503 while serving.
#[cfg(all(test, feature = "cuda"))]
mod q4k_path_tests {
    use super::*;
    use crate::apr::{AprV2Model, MappedAprModel};
    use crate::cuda::CudaExecutor;
    use crate::gguf::{OwnedQuantizedKVCache, OwnedQuantizedModel};
    use crate::gpu::adapters::apr_q4k::{
        forward_token_apr_q4k, parse_apr_q4k_config, upload_apr_q4k_weights,
    };
    use std::path::Path;

    /// A Q4_K_M `.apr` with Q6_K `attn_v`/`ffn_down`/`output.weight` and 84
    /// GGUF-named QKV biases — the file #3791 was measured on.
    const MODEL: &str = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4k.apr";

    /// done_when 2: the GPU forward matches the CPU forward (`apr run --no-gpu`'s)
    /// on this file, judged by the F2 rule over 64 real positions.
    #[test]
    fn q4k_gpu_forward_matches_the_cpu_forward_under_the_f2_rule() {
        if !Path::new(MODEL).exists() || !CudaExecutor::is_available() {
            eprintln!("SKIP: {MODEL} or a CUDA device is absent");
            return;
        }
        let path = Path::new(MODEL);
        let model = AprV2Model::load(path).expect("load the .apr");
        let config = parse_apr_q4k_config(&model).expect("Q4K config");
        let mut executor = CudaExecutor::new(0).expect("CUDA");
        upload_apr_q4k_weights(&model, &mut executor).expect("upload");
        let host = load_q4k_host_weights(&model, config.num_layers).expect("host weights");

        let text = "The sea is vast and deep. Write one sentence about it, then explain \
                    why the tide rises twice a day, in plain words, for a child of ten.";
        // Two passes of the text: 64 real positions, the F2 guard's probe width.
        let mut tokens = AprV2Model::encode_text(path, &format!("{text} {text}")).expect("encode");
        tokens.truncate(64);
        assert_eq!(tokens.len(), 64, "a full-width probe");

        let mapped = MappedAprModel::from_path(MODEL).expect("map the .apr");
        let cpu = OwnedQuantizedModel::from_apr(&mapped).expect("CPU model");
        let kv_dim = config.num_kv_heads * config.head_dim;
        let mut cache = OwnedQuantizedKVCache::new(config.num_layers, kv_dim, tokens.len() + 1);
        let mut k_cache = vec![Vec::new(); config.num_layers];
        let mut v_cache = vec![Vec::new(); config.num_layers];

        let (mut gpu, mut cpu_logits) = (Vec::new(), Vec::new());
        for (pos, &token) in tokens.iter().enumerate() {
            gpu.push(
                forward_token_apr_q4k(
                    &mut executor,
                    &config,
                    &host.embedding,
                    &host.output_norm,
                    &host.layer_norms,
                    &host.qkv_biases,
                    &mut k_cache,
                    &mut v_cache,
                    token,
                    pos,
                )
                .expect("GPU forward"),
            );
            cpu_logits.push(
                cpu.forward_single_with_cache(token, &mut cache, pos)
                    .expect("CPU forward"),
            );
        }
        assert!(
            gpu.iter().flatten().all(|x| x.is_finite()),
            "the GPU logits carry NaN/inf"
        );
        let report = crate::infer::f2_multi_position_report(&cpu_logits, &gpu);
        eprintln!(
            "[#3791] {} positions: accepted={} min cosine {:.4}, first bad pos {} (cpu {} gpu {} cos {:.4})",
            tokens.len(),
            report.accepted,
            report.min_cosine_real,
            report.first_bad_pos,
            report.first_bad_cpu_argmax,
            report.first_bad_gpu_argmax,
            report.first_bad_cosine
        );
        assert!(
            report.accepted,
            "the Q4K GPU forward disagrees with the CPU forward"
        );
    }

    /// done_when 4: a serving Q4K state is loaded and on the GPU — through the one
    /// predicate `/health` reads (#3571's). No device needed: the state holds only
    /// the inference thread's channel.
    #[test]
    fn a_q4k_state_reports_loaded_and_gpu() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let state = crate::api::AppState::with_apr_q4k_and_vocab_eos(
            tx,
            vec!["<unk>".to_string(), "a".to_string()],
            None,
        )
        .expect("state");
        assert!(
            state.model_loaded(),
            "a serving Q4K state is a loaded model"
        );
    }
}
