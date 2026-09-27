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
//! So the request carries the token into the one engine: since #4269 (M2b) each
//! request decodes through a [`Session`](crate::session::Session) over
//! [`AprQ4kForward`](crate::api::apr_q4k_forward::AprQ4kForward), and the session
//! polls it once per generated token, as it does for every other architecture.
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

    let weights = load_q4k_host_weights(&model, config.num_layers)?;
    let context_length = model.metadata().max_position_embeddings;

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
                    &weights,
                    context_length,
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

#[cfg(feature = "cuda")]
/// Run one request through the one engine (#4269 M2b), on the inference thread:
/// a fresh session per request (so a fresh KV cache, as before) over the
/// executor this thread owns. The session owns the prefill, token choice, EOS
/// and the cancellation poll.
#[allow(clippy::too_many_arguments)]
fn generate_q4k(
    executor: &mut crate::cuda::CudaExecutor,
    config: &crate::gpu::adapters::apr_q4k::AprQ4KConfig,
    weights: &Q4kHostWeights,
    context_length: Option<usize>,
    prompt_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    seed: u64,
    eos_ids: &[u32],
    cancel: &CancelToken,
) -> Result<AprQ4kResponse, String> {
    use crate::api::apr_q4k_forward::{AprQ4kForward, AprQ4kSession, CudaQ4kStep};
    use std::time::Instant;

    let gen_start = Instant::now();
    let step = CudaQ4kStep::new(executor, config, weights);
    let mut session = AprQ4kSession::new(AprQ4kForward::new(step, true, context_length));
    let gen_config = q4k_generate_config(max_tokens, temperature, seed, eos_ids, cancel);
    let turn = session
        .generate(prompt_ids, &gen_config, &mut |_| true)
        .map_err(|e| format!("Q4K generate failed: {e}"))?;
    let output_tokens = turn.tokens[prompt_ids.len()..].to_vec();

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

/// The top-k the APR Q4K chat path samples with.
const Q4K_TOP_K: usize = 40;

/// The session config a Q4K request decodes with.
///
/// The Q4K path has always decoded greedily at `temperature <= 0.01`, and
/// samples otherwise with top-k 40, top-p 1.0 and one RNG seeded from the
/// request (#3786). Kept outside the `cuda` gate so the falsifiers decode with
/// the exact config the GPU thread does.
pub(crate) fn q4k_generate_config(
    max_tokens: usize,
    temperature: f32,
    seed: u64,
    eos_ids: &[u32],
    cancel: &CancelToken,
) -> crate::gguf::QuantizedGenerateConfig {
    crate::gguf::QuantizedGenerateConfig {
        max_tokens,
        temperature: if temperature <= 0.01 {
            0.0
        } else {
            temperature
        },
        top_k: Q4K_TOP_K,
        top_p: 1.0,
        seed,
        stop_tokens: eos_ids.to_vec(),
        cancel: cancel.clone(),
        ..crate::gguf::QuantizedGenerateConfig::default()
    }
}

#[cfg(test)]
#[path = "tests/apr_q4k_cancel_2465.rs"]
mod apr_q4k_cancel_2465;

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
