
/// PMAT-795: OpenAI `finish_reason` for a `/v1/completions` choice.
///
/// `"length"` only when the generation hit `max_tokens` with no stop-string match;
/// otherwise `"stop"` (a matched stop string takes precedence over `max_tokens`, and a
/// model that terminated before the budget also reports `"stop"`). This mirrors the chat
/// path's `finalize_chat_text` and the `completion_resp` helper so all completion backends
/// agree. Pure + unit-tested so the GPU backend's behavior is falsifiable without a GPU.
#[cfg(any(feature = "gpu", test))]
fn completion_finish_reason(stopped: bool, completion_tokens: usize, max_tokens: usize) -> &'static str {
    if !stopped && completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    }
}

/// GPU model backend.
#[cfg(feature = "gpu")]
fn try_gpu_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<Option<CompletionResponse>, RErr> {
    use crate::gpu::GpuGenerateConfig;

    let gpu_model_lock = match state.gpu_model() {
        Some(l) => l,
        None => return Ok(None),
    };
    let tokenizer = state.tokenizer.clone().ok_or_else(|| {
        rerr(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            "No tokenizer available",
        )
    })?;
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(
            state,
            StatusCode::BAD_REQUEST,
            "Prompt cannot be empty",
        ));
    }
    let prompt_tokens = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    let gpu_config = GpuGenerateConfig {
        max_tokens,
        temperature,
        top_k: 1,
        stop_tokens: Vec::new(),
        trace: state.is_trace_enabled(),
    };

    let mut gpu_model = gpu_model_lock.write().map_err(|e| {
        rerr(
            state,
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("GPU lock: {e}"),
        )
    })?;
    let generated = gpu_model
        .generate(&prompt, &gpu_config)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let token_ids: Vec<u32> = generated
        .iter()
        .skip(prompt_tokens)
        .filter_map(|&id| u32::try_from(id).ok())
        .collect();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // PMAT-755: apply OpenAI stop sequences (this GPU backend previously ignored them).
    let orig_text_len = text.len();
    let text = truncate_at_stop(text, request.stop.as_deref());
    let stopped = text.len() < orig_text_len;
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    // PMAT-795: compute finish_reason instead of hardcoding "stop". This GPU backend
    // passes an empty `stop_tokens` to `generate`, so generation always runs the full
    // `max_tokens` budget — every token-limited completion was mislabeled "stop" when
    // OpenAI semantics require "length". A matched stop *string* still takes precedence
    // (returns "stop"), matching the chat path's `finalize_chat_text`.
    let finish_reason = completion_finish_reason(stopped, completion_tokens, max_tokens);
    let response_id = format!("cmpl-{}", &uuid::Uuid::new_v4().to_string()[..8]);
    Ok(Some(CompletionResponse {
        id: response_id,
        object: "text_completion".to_string(),
        created: epoch_secs(),
        model: request.model.clone(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// CPU model fallback.
fn registry_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<CompletionResponse, RErr> {
    let model_id = if request.model == "default" || request.model.is_empty() {
        None
    } else {
        Some(request.model.as_str())
    };

    let (model, tokenizer) = state
        .get_model(model_id)
        .map_err(|e| rerr(state, StatusCode::NOT_FOUND, e))?;
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(
            state,
            StatusCode::BAD_REQUEST,
            "Prompt cannot be empty",
        ));
    }
    let prompt_tokens = prompt_ids.len();
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    let mut config = GenerationConfig::default()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature);
    if let Some(top_p) = request.top_p {
        config.strategy = SamplingStrategy::TopP { p: top_p as f32 };
    }

    let generated = model
        .generate(&prompt, &config)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let token_ids: Vec<u32> = generated
        .iter()
        .skip(prompt_tokens)
        .filter_map(|&id| u32::try_from(id).ok())
        .collect();
    let completion_tokens = token_ids.len();
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    state
        .metrics
        .record_success(completion_tokens, start.elapsed());

    Ok(completion_resp(
        "cmpl",
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
    ))
}

/// ALB-098: Q4K GPU completions via dedicated inference thread.
#[cfg(feature = "cuda")]
async fn try_apr_q4k_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<Option<CompletionResponse>, RErr> {
    use crate::api::apr_q4k_scheduler::AprQ4kRequest;

    let q4k_tx = match state.apr_q4k_tx() {
        Some(tx) => tx,
        None => return Ok(None),
    };
    let tokenizer = state.tokenizer.clone().ok_or_else(|| {
        rerr(state, StatusCode::INTERNAL_SERVER_ERROR, "No tokenizer available")
    })?;
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(state, StatusCode::BAD_REQUEST, "Prompt cannot be empty"));
    }
    let prompt_tokens = prompt_ids.len();

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    // ALB-109: Get EOS token IDs from model config or tokenizer.
    let eos_ids = state.model_eos_ids();

    q4k_tx
        .send(AprQ4kRequest {
            prompt_ids,
            max_tokens,
            temperature,
            eos_ids,
            response_tx,
        })
        .await
        .map_err(|_| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, "Q4K thread unavailable"))?;

    let result = response_rx
        .await
        .map_err(|_| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, "Q4K thread dropped response"))?;

    let resp = result.map_err(|e| {
        rerr(state, StatusCode::INTERNAL_SERVER_ERROR, format!("Q4K generation failed: {e}"))
    })?;

    let text = tokenizer
        .decode(&resp.output_tokens)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // PMAT-755: apply OpenAI stop sequences (this backend previously ignored them).
    let text = truncate_at_stop(text, request.stop.as_deref());
    let completion_tokens = resp.tokens_generated;
    state.metrics.record_success(completion_tokens, start.elapsed());

    Ok(Some(completion_resp(
        "cmpl",
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
    )))
}

/// realizar#184 / ALB-136: CUDA GGUF completions via batch scheduler.
///
/// Root cause: `with_cuda_model_and_vocab()` sets `model: None` so the registry
/// fallback fails with "No model available". The CUDA GGUF model is only reachable
/// via `cuda_batch_tx` (the batch scheduler channel). This function bridges the gap
/// for non-streaming /v1/completions requests.
#[cfg(feature = "cuda")]
async fn try_cuda_gguf_completions(
    state: &AppState,
    request: &CompletionRequest,
    max_tokens: usize,
    temperature: f32,
    start: std::time::Instant,
) -> Result<Option<CompletionResponse>, RErr> {
    use crate::api::cuda_batch_scheduler::CudaBatchRequest;
    use crate::gguf::QuantizedGenerateConfig;

    let batch_tx = match state.cuda_batch_tx() {
        Some(tx) => tx,
        None => return Ok(None),
    };
    let tokenizer = state.tokenizer.clone().ok_or_else(|| {
        rerr(state, StatusCode::INTERNAL_SERVER_ERROR, "No tokenizer available")
    })?;
    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(state, StatusCode::BAD_REQUEST, "Prompt cannot be empty"));
    }
    let prompt_tokens = prompt_ids.len();

    let eos = state.cached_eos_token_id.unwrap_or(151643);
    let q_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        stop_tokens: vec![eos],
        ..Default::default()
    };

    // realizr#212: non_streaming flag tells scheduler to accumulate + bulk-send.
    let (token_tx, mut token_rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(max_tokens + 1);

    let batch_req = CudaBatchRequest {
        prompt_ids,
        config: q_config,
        token_tx,
        non_streaming: true,
        enqueue_time: std::time::Instant::now(),
    };

    batch_tx
        .try_send(batch_req)
        .map_err(|_| rerr(state, StatusCode::SERVICE_UNAVAILABLE, "CUDA batch queue full"))?;

    // Collect all generated tokens
    let mut output_tokens = Vec::with_capacity(max_tokens);
    while let Some(result) = token_rx.recv().await {
        match result {
            Ok(token_id) => output_tokens.push(token_id),
            Err(e) => {
                return Err(rerr(state, StatusCode::INTERNAL_SERVER_ERROR, format!("CUDA generation: {e}")));
            }
        }
    }

    let completion_tokens = output_tokens.len();
    let text = tokenizer
        .decode(&output_tokens)
        .map_err(|e| rerr(state, StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // PMAT-761: truncate at the EARLIEST stop POSITION via the shared helper. The previous
    // inline loop cut at the first-LISTED stop that matched, not the earliest-position one —
    // e.g. stop=["world","hello"] on "hello world" wrongly kept "hello ". This makes
    // try_cuda_gguf_completions consistent with every other completion backend (PMAT-754/755).
    let text = truncate_at_stop(text, request.stop.as_deref());

    state.metrics.record_success(completion_tokens, start.elapsed());

    Ok(Some(completion_resp(
        "cmpl",
        request.model.clone(),
        text,
        prompt_tokens,
        completion_tokens,
        max_tokens,
    )))
}

pub async fn openai_completions_handler(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, RErr> {
    let start = std::time::Instant::now();
    let max_tokens = request.max_tokens.unwrap_or(256);
    let temperature = request.temperature.unwrap_or(0.7) as f32;

    #[cfg(feature = "gpu")]
    if let Some(r) =
        try_cached_completions(&state, &request, max_tokens, temperature, start).await?
    {
        return Ok(Json(r));
    }

    if let Some(r) = try_quantized_completions(&state, &request, max_tokens, temperature, start)? {
        return Ok(Json(r));
    }

    #[cfg(feature = "gpu")]
    if let Some(r) = try_gpu_completions(&state, &request, max_tokens, temperature, start)? {
        return Ok(Json(r));
    }

    #[cfg(feature = "cuda")]
    if let Some(r) = try_apr_q4k_completions(&state, &request, max_tokens, temperature, start).await? {
        return Ok(Json(r));
    }

    // realizar#184 / ALB-136: CUDA GGUF models via batch scheduler
    #[cfg(feature = "cuda")]
    if let Some(r) = try_cuda_gguf_completions(&state, &request, max_tokens, temperature, start).await? {
        return Ok(Json(r));
    }

    // GH-627/637/670: Direct CUDA model fallback for APR GPU path
    // with_cuda_model_and_vocab sets cuda_model but NOT model,
    // so registry_completions fails with "No model available".
    #[cfg(feature = "cuda")]
    if let Some(cuda_lock) = state.cuda_model() {
        use crate::gguf::QuantizedGenerateConfig;
        let tokenizer = state.tokenizer.clone().ok_or_else(|| {
            rerr(&state, StatusCode::INTERNAL_SERVER_ERROR, "No tokenizer")
        })?;
        let prompt_ids = tokenizer.encode(&request.prompt);
        if prompt_ids.is_empty() {
            return Err(rerr(&state, StatusCode::BAD_REQUEST, "Prompt cannot be empty"));
        }
        let eos = state.cached_eos_token_id.unwrap_or(151643);
        let config = QuantizedGenerateConfig {
            max_tokens: max_tokens.min(4096),
            temperature,
            top_k: if temperature == 0.0 { 1 } else { 40 },
            stop_tokens: vec![eos],
            ..Default::default()
        };
        let result = {
            let mut model = cuda_lock.write().expect("CUDA model lock");
            model.generate_gpu_resident_logprobs(
                &prompt_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                &config,
            ).map_err(|e| rerr(&state, StatusCode::INTERNAL_SERVER_ERROR, e))?
        };
        let prompt_len = prompt_ids.len();
        let gen_tokens: Vec<u32> = result.tokens[prompt_len..].to_vec();
        let text: String = gen_tokens.iter()
            .map(|&t| tokenizer.decode(&[t]).unwrap_or_else(|_| format!("<{t}>")))
            .collect();
        let elapsed = start.elapsed();
        let completion_tokens = gen_tokens.len();
        return Ok(Json(CompletionResponse {
            id: format!("cmpl-cuda-{}", elapsed.as_millis()),
            object: "text_completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: request.model.clone(),
            choices: vec![CompletionChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: if completion_tokens >= max_tokens { "length" } else { "stop" }.to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_ids.len(),
                completion_tokens,
                total_tokens: prompt_ids.len() + completion_tokens,
            },
        }));
    }

    Ok(Json(registry_completions(
        &state,
        &request,
        max_tokens,
        temperature,
        start,
    )?))
}

/// realizr#191: Logprobs endpoint for perplexity measurement (F-QUALITY-01).
///
/// Returns per-token log probabilities for the generated sequence.
/// Uses the direct CUDA path (not batch scheduler) to access logits.
///
/// POST /v1/logprobs { "prompt": "...", "max_tokens": 256 }
/// Returns { "tokens": [...], "logprobs": [...], "perplexity": ... }
#[cfg(feature = "cuda")]
pub async fn logprobs_handler(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<serde_json::Value>, RErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let cuda_model_lock = state.cuda_model().ok_or_else(|| {
        rerr(&state, StatusCode::SERVICE_UNAVAILABLE, "No CUDA model loaded")
    })?;
    let tokenizer = state.tokenizer.clone().ok_or_else(|| {
        rerr(&state, StatusCode::INTERNAL_SERVER_ERROR, "No tokenizer")
    })?;

    let prompt_ids = tokenizer.encode(&request.prompt);
    if prompt_ids.is_empty() {
        return Err(rerr(&state, StatusCode::BAD_REQUEST, "Empty prompt"));
    }

    let max_tokens = request.max_tokens.unwrap_or(256);
    let eos = state.cached_eos_token_id.unwrap_or(151643);
    let config = QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0, // greedy for perplexity
        top_k: 1,
        stop_tokens: vec![eos],
        logprobs: true,
        ..Default::default()
    };

    let result = {
        let mut model = cuda_model_lock.write().expect("CUDA model lock");
        model.generate_gpu_resident_logprobs(
            &prompt_ids.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            &config,
        ).map_err(|e| rerr(&state, StatusCode::INTERNAL_SERVER_ERROR, e))?
    };

    let prompt_len = prompt_ids.len();
    let gen_tokens: Vec<u32> = result.tokens[prompt_len..].to_vec();
    let gen_text: Vec<String> = gen_tokens.iter().map(|&t| {
        tokenizer.decode(&[t]).unwrap_or_else(|_| format!("<{t}>"))
    }).collect();

    // Compute perplexity: exp(-1/N * sum(logprobs))
    let n = result.logprobs.len() as f64;
    let sum_logprob: f64 = result.logprobs.iter().map(|lp| f64::from(lp.logprob)).sum();
    let perplexity = if n > 0.0 { (-sum_logprob / n).exp() } else { 0.0 };

    let logprobs_json: Vec<serde_json::Value> = result.logprobs.iter().zip(gen_text.iter()).map(|(lp, text)| {
        serde_json::json!({
            "token": text,
            "token_id": lp.token_id,
            "logprob": lp.logprob,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "prompt_tokens": prompt_len,
        "completion_tokens": gen_tokens.len(),
        "tokens": gen_text,
        "logprobs": logprobs_json,
        "perplexity": perplexity,
        "sum_logprob": sum_logprob,
    })))
}

/// realizr#191: Teacher-forcing perplexity endpoint (F-QUALITY-01).
///
/// Feeds ground-truth tokens through the model and measures how well
/// the model predicts each next token. Standard PPL methodology
/// matching llama-perplexity.
///
/// POST /v1/perplexity { "prompt": "<text>", "model": "default" }
/// Returns { "perplexity": 15.8, "num_tokens": 512 }
#[cfg(feature = "cuda")]
pub async fn perplexity_handler(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<serde_json::Value>, RErr> {
    let cuda_model_lock = state.cuda_model().ok_or_else(|| {
        rerr(&state, StatusCode::SERVICE_UNAVAILABLE, "No CUDA model loaded")
    })?;
    let tokenizer = state.tokenizer.clone().ok_or_else(|| {
        rerr(&state, StatusCode::INTERNAL_SERVER_ERROR, "No tokenizer")
    })?;

    let token_ids: Vec<u32> = tokenizer
        .encode(&request.prompt)
        .iter()
        .map(|&x| x as u32)
        .collect();
    if token_ids.len() < 2 {
        return Err(rerr(&state, StatusCode::BAD_REQUEST, "Need at least 2 tokens"));
    }

    let start = std::time::Instant::now();
    let mut model = cuda_model_lock.write().expect("CUDA model lock");

    // realizr#203: Run BOTH paths for comparison during development
    let ppl_sequential = model
        .perplexity_gpu_resident(&token_ids)
        .map_err(|e| rerr(&state, StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let ppl_batched = model.perplexity_gpu_batched(&token_ids).ok();

    drop(model);
    let elapsed = start.elapsed();

    Ok(Json(serde_json::json!({
        "perplexity": ppl_batched.unwrap_or(ppl_sequential),
        "ppl_sequential": ppl_sequential,
        "ppl_batched": ppl_batched,
        "num_tokens": token_ids.len(),
        "elapsed_ms": elapsed.as_millis(),
        "tokens_per_sec": token_ids.len() as f64 / elapsed.as_secs_f64(),
    })))
}

/// OpenAI-compatible embeddings handler (/v1/embeddings)
pub async fn openai_embeddings_handler(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Delegate to native handler
    realize_embed_handler(State(state), Json(request)).await
}

#[cfg(test)]
mod pmat795_finish_reason_tests {
    use super::completion_finish_reason;

    /// FALSIFIER (PMAT-795): the GPU `/v1/completions` backend passes an empty
    /// `stop_tokens` to `generate`, so a token-limited request runs to `max_tokens`.
    /// The handler previously HARDCODED `finish_reason: "stop"` for this case — wrong.
    /// OpenAI requires "length" when the token budget is exhausted with no stop match.
    #[test]
    fn max_tokens_hit_with_no_stop_is_length() {
        // stopped=false, completion_tokens == max_tokens => "length" (was wrongly "stop").
        assert_eq!(completion_finish_reason(false, 256, 256), "length");
        // Over budget (defensive) is also "length".
        assert_eq!(completion_finish_reason(false, 300, 256), "length");
    }

    #[test]
    fn natural_termination_before_budget_is_stop() {
        // Model emitted fewer than max_tokens (e.g. hit EOS) => "stop".
        assert_eq!(completion_finish_reason(false, 10, 256), "stop");
    }

    #[test]
    fn stop_string_match_beats_length() {
        // A matched stop string truncated the text: "stop" takes precedence over "length"
        // even when the token budget was also reached (OpenAI semantics, matches chat path).
        assert_eq!(completion_finish_reason(true, 256, 256), "stop");
        assert_eq!(completion_finish_reason(true, 10, 256), "stop");
    }
}
