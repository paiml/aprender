
/// Handle POST /v1/completions for GPU inference.
///
/// GH-284: Now async with `spawn_blocking` to avoid blocking the tokio runtime.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
async fn handle_gpu_completion(
    cuda: Arc<std::sync::Mutex<realizar::apr::AprV2ModelCuda>>,
    tok_info: Arc<Option<SafeTensorsTokenizerInfo>>,
    req: GpuCompletionRequest,
    cpu_state: Arc<std::sync::Mutex<AprServerState>>,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse, Json};

    let start = Instant::now();
    let tok_ref = tok_info.as_ref().as_ref();
    let input_tokens = encode_prompt(tok_ref, &req.prompt);
    let eos_id = eos_token_id(tok_ref, 2);
    let max_tokens = req.max_tokens.min(4096);
    let prompt = req.prompt.clone();

    // GH-284: Run GPU generation off the async runtime
    let cuda_clone = cuda.clone();
    let input_clone = input_tokens.clone();
    let result = tokio::task::spawn_blocking(move || {
        run_gpu_generation(&cuda_clone, &input_clone, max_tokens, eos_id)
    })
    .await;

    let gen_start = Instant::now();
    let output_tokens = match result {
        Ok(Ok(t)) => t,
        Ok(Err(gpu_err)) => {
            // GH-261: Per-request CPU fallback
            eprintln!("[GPU->CPU FALLBACK] {gpu_err}");
            let s = match cpu_state.lock() {
                Ok(guard) => guard.clone(),
                Err(_) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": format!("GPU failed: {gpu_err}; CPU state corrupted")
                        })),
                    )
                        .into_response();
                }
            };

            // CPU fallback also in spawn_blocking
            let result = tokio::task::spawn_blocking(move || {
                run_apr_cpu_inference(&s, &prompt, max_tokens, 0.0)
            })
            .await;

            match result {
                Ok(Ok(out)) => {
                    return Json(serde_json::json!({
                        "text": out.text,
                        "tokens_generated": out.tokens_generated,
                        "latency_ms": out.gen_duration.as_millis() as u64,
                        "tok_per_sec": compute_tok_per_sec(out.tokens_generated, out.gen_duration),
                        "compute_mode": "cpu-fallback"
                    }))
                    .into_response();
                }
                Ok(Err(cpu_err)) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": format!("GPU failed: {gpu_err}; CPU fallback also failed: {cpu_err}")
                        })),
                    )
                        .into_response();
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": format!("GPU failed: {gpu_err}; CPU task failed: {e}")
                        })),
                    )
                        .into_response();
                }
            }
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("GPU task failed: {e}")})),
            )
                .into_response();
        }
    };
    let gen_time = gen_start.elapsed();

    let new_tokens = extract_new_tokens(&output_tokens, input_tokens.len());
    let text = decode_tokens(tok_info.as_ref().as_ref(), new_tokens);

    Json(GpuCompletionResponse {
        text,
        tokens_generated: new_tokens.len(),
        latency_ms: start.elapsed().as_millis() as u64,
        tok_per_sec: compute_tok_per_sec(new_tokens.len(), gen_time),
    })
    .into_response()
}

/// GH-261: Handle GPU failure with CPU fallback inference.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)]
async fn gpu_cpu_fallback(
    gpu_err: String,
    cpu_state: &std::sync::Mutex<AprServerState>,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    start: Instant,
) -> axum::response::Response {
    use axum::{response::IntoResponse, Json};

    eprintln!("[GPU->CPU FALLBACK] {gpu_err}");
    let s = match cpu_state.lock() {
        Ok(guard) => guard.clone(),
        Err(_) => {
            return Json(serde_json::json!({
                "error": format!("GPU failed: {gpu_err}; CPU state corrupted")
            }))
            .into_response();
        }
    };

    let result = tokio::task::spawn_blocking(move || {
        run_apr_cpu_inference(&s, &prompt, max_tokens, temperature)
    })
    .await;

    match result {
        Ok(Ok(out)) => build_cpu_fallback_response(&out, start),
        Ok(Err(cpu_err)) => {
            Json(serde_json::json!({
                "error": format!("GPU failed: {gpu_err}; CPU fallback also failed: {cpu_err}")
            }))
            .into_response()
        }
        Err(e) => {
            Json(serde_json::json!({
                "error": format!("GPU failed: {gpu_err}; CPU task failed: {e}")
            }))
            .into_response()
        }
    }
}

/// Build a chat completion response for a successful CPU fallback.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)]
fn build_cpu_fallback_response(out: &AprInferenceOutput, start: Instant) -> axum::response::Response {
    use axum::{response::IntoResponse, Json};

    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Json(serde_json::json!({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": "apr-cpu-fallback",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": out.text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": out.input_token_count,
            "completion_tokens": out.tokens_generated,
            "total_tokens": out.input_token_count + out.tokens_generated
        },
        "_apr_metrics": {
            "latency_ms": start.elapsed().as_millis() as u64,
            "tok_per_sec": compute_tok_per_sec(out.tokens_generated, out.gen_duration),
            "compute_mode": "cpu-fallback"
        }
    }))
    .into_response()
}

/// Build an SSE stream from pre-generated GPU tokens.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap
fn build_gpu_sse_stream(
    tokens: Vec<u32>,
    tok_info: Arc<Option<SafeTensorsTokenizerInfo>>,
    model_name: String,
) -> axum::response::Response {
    use axum::response::{sse::{Event, Sse}, IntoResponse};

    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let stream = futures_util::stream::unfold(
        (Some(tokens.into_iter()), tok_info, request_id, created, model_name),
        |(maybe_iter, tok_info, request_id, created, model_name)| async move {
            let mut iter = maybe_iter?;
            match iter.next() {
                Some(token_id) => {
                    let text = decode_single_token((*tok_info).as_ref(), token_id);
                    let chunk = serde_json::json!({
                        "id": &request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": &model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    let event = Event::default().data(chunk.to_string());
                    Some((
                        Ok::<_, std::convert::Infallible>(event),
                        (Some(iter), tok_info, request_id, created, model_name),
                    ))
                }
                None => {
                    let event = Event::default().data("[DONE]");
                    Some((
                        Ok::<_, std::convert::Infallible>(event),
                        (None, tok_info, request_id, created, model_name),
                    ))
                }
            }
        },
    );

    Sse::new(stream).into_response()
}

/// Handle POST /v1/chat/completions for GPU inference (PAR-302).
///
/// GH-284: True per-token SSE streaming. GPU generates all tokens in
/// `spawn_blocking`, then streams them as individual SSE events.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
async fn handle_gpu_chat_completion(
    cuda: Arc<std::sync::Mutex<realizar::apr::AprV2ModelCuda>>,
    tok_info: Arc<Option<SafeTensorsTokenizerInfo>>,
    req: serde_json::Value,
    cpu_state: Arc<std::sync::Mutex<AprServerState>>,
) -> axum::response::Response {
    use axum::{response::IntoResponse, Json};

    // GH-283: Validate model name before processing
    if let Ok(s) = cpu_state.lock() {
        if let Some(err_response) = validate_request_model(&req, &s.model_name) {
            return err_response;
        }
    }

    let messages = req.get("messages").and_then(|m| m.as_array());
    let stream_mode = req.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let max_tokens = req.get("max_tokens").and_then(serde_json::Value::as_u64).unwrap_or(32) as usize;
    let temperature = req.get("temperature").and_then(serde_json::Value::as_f64).unwrap_or(0.0) as f32;

    let Some(msgs) = messages else {
        return Json(serde_json::json!({"error": "Missing messages"})).into_response();
    };

    let prompt = format_chatml(msgs);
    let start = Instant::now();
    let tok_ref = tok_info.as_ref().as_ref();
    let input_tokens = encode_prompt(tok_ref, &prompt);
    let eos_id = eos_token_id(tok_ref, 151_645);
    let max_tokens_clamped = max_tokens.min(4096);

    let cuda_clone = cuda.clone();
    let input_clone = input_tokens.clone();
    let gen_start = Instant::now();
    let gen_result = tokio::task::spawn_blocking(move || {
        run_gpu_generation(&cuda_clone, &input_clone, max_tokens_clamped, eos_id)
    })
    .await;

    let output_tokens = match gen_result {
        Ok(Ok(t)) => t,
        Ok(Err(gpu_err)) => {
            return gpu_cpu_fallback(gpu_err, &cpu_state, prompt, max_tokens_clamped, temperature, start).await;
        }
        Err(e) => {
            return Json(serde_json::json!({"error": format!("GPU task failed: {e}")})).into_response();
        }
    };
    let elapsed = gen_start.elapsed();

    let new_tokens = extract_new_tokens(&output_tokens, input_tokens.len());
    eprintln!(
        "[APR GPU CHAT DEBUG] Input tokens: {}, Output tokens: {}, New tokens: {}",
        input_tokens.len(), output_tokens.len(), new_tokens.len()
    );

    let tokens_generated = new_tokens.len();
    let tok_per_sec = compute_tok_per_sec(tokens_generated, elapsed);
    let response_model = cpu_state.lock().ok()
        .map_or_else(|| "apr-gpu".to_string(), |s| s.model_name.clone());

    if stream_mode {
        build_gpu_sse_stream(new_tokens.to_vec(), tok_info, response_model)
    } else {
        let request_id = generate_request_id();
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let output_text = decode_tokens(tok_info.as_ref().as_ref(), new_tokens);
        Json(serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": &response_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": input_tokens.len(),
                "completion_tokens": tokens_generated,
                "total_tokens": input_tokens.len() + tokens_generated
            },
            "_apr_metrics": {"latency_ms": start.elapsed().as_millis() as u64, "tok_per_sec": tok_per_sec}
        }))
        .into_response()
    }
}

/// Print GPU server startup banner.
fn print_gpu_server_banner(bind_addr: &str) {
    println!();
    println!(
        "{}",
        format!("APR GPU Inference Server listening on http://{bind_addr}")
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
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    println!("{}", "Loading GGUF model (mmap)...".dimmed());
    // aprender#1789 Option B: retain MappedGGUFModel in an Arc so it can be
    // threaded into AppState via `.with_mapped_gguf_model()`. The chat
    // handler's `try_qwen3_moe_backend` needs this for MoE inference; the
    // per-expert tensors in `run_qwen3_moe_generate` borrow directly from
    // the mmap, so the mapped model MUST outlive any inference call.
    let mapped_model = std::sync::Arc::new(
        MappedGGUFModel::from_path(model_path)
            .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?,
    );

    println!(
        "{}",
        format!(
            "GGUF loaded: {} tensors, {} metadata entries",
            mapped_model.model.tensors.len(),
            mapped_model.model.metadata.len()
        )
        .dimmed()
    );

    // #3571: the Qwen3.5 hybrid (Gated Delta Net) has no dense layers, so it is served from a
    // resident session — and it is routed there BEFORE `build_serve_model`, whose zero-layer
    // refusal its generic base would (rightly) fail. #3608 routed it to that base, which loaded
    // and then decoded nothing: the routing gap was one verb deep, and every gate we own is
    // single-stream `apr run` (#3555).
    if realizar::gguf::hybrid_forward_handles(mapped_model.model.architecture().unwrap_or_default()) {
        return start_qwen35_server(mapped_model, config);
    }

    println!("{}", "Building quantized inference model...".dimmed());
    let quantized_model = build_serve_model(&mapped_model)?;

    println!(
        "{}",
        format!(
            "Model ready: {} layers, vocab_size={}, hidden_dim={}",
            quantized_model.layers().len(),
            quantized_model.config().vocab_size,
            quantized_model.config().hidden_dim
        )
        .green()
    );

    let vocab = extract_gguf_vocab(&mapped_model)?;

    // PERF-021 / N4 / I-2: a request has a RESOLUTION, and it is reported.
    //
    // "A boolean accelerator flag has no observable resolution: `--gpu` can be
    // ignored and nothing in the output changes. `-ngl 999` cannot be ignored,
    // because the loader must state how many layers it placed." (v2.2, N4.)
    //
    // Printed on BOTH paths, including resolved=0. A line that appears only when
    // the accelerator engaged reports success and is silent on the failure it
    // exists to make visible — which is #2696's shape exactly.
    let total_layers = u32::try_from(quantized_model.layers().len()).unwrap_or(u32::MAX);
    let resolved_layers = config.resolve_layers(total_layers)?;
    println!(
        "gpu-layers: requested={} resolved={resolved_layers} total={total_layers} (backend={})",
        config
            .gpu_layers
            .map_or_else(|| "none".to_string(), |r| r.to_string()),
        // NOTE: a BUILD label, not a residency claim. What actually loaded is
        // reported by `/v1/effective-config`'s `backend_loaded`, which is
        // derived from the AppState and can say `cpu` on this very build.
        if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(feature = "wgpu") {
            "wgpu"
        } else {
            "cpu"
        }
    );
    // PP-14/PP-15: the same resolution, as a value the served process reports.
    let offload = super::offload_report(config, resolved_layers, total_layers);

    #[cfg(feature = "cuda")]
    if config.wants_accelerator() && config.batch {
        return start_gguf_server_gpu_batched(quantized_model, vocab, mapped_model, config);
    }

    #[cfg(feature = "cuda")]
    if config.wants_accelerator() {
        return start_gguf_server_cuda(quantized_model, vocab, mapped_model, config, offload);
    }

    run_cpu_server(quantized_model, vocab, Some(mapped_model), config, Some(offload))
}

/// Build the model `apr serve` hands its routes, or refuse it (#3571).
///
/// Every GGUF serve route — `--gpu`, `--gpu --batch`, the CPU server — takes the model this
/// returns, so the zero-layer refusal here is the one no route can walk around.
fn build_serve_model(
    mapped_model: &realizar::gguf::MappedGGUFModel,
) -> Result<realizar::gguf::OwnedQuantizedModel> {
    use realizar::gguf::OwnedQuantizedModel;
    // #3571: the Qwen3.5 hybrid (Gated Delta Net) has no dense layers, so `from_mapped` refuses
    // it by name and `apr serve` could not load the architecture the last release shipped — while
    // `apr run` loaded it fine, because `run_gguf_inference` has carried exactly this branch since
    // #3091. The routing gap was in ONE verb, and it was invisible because every gate we own is
    // single-stream `apr run`: no ladder rung, parity record or dogfood row has ever asked
    // `apr serve` to load a model (#3555).
    let is_qwen35 = mapped_model.model.architecture() == Some("qwen35");
    let quantized_model = if is_qwen35 {
        realizar::gguf::forward_qwen35::Qwen35Model::create_base_model(
            &mapped_model.model,
            mapped_model.data(),
        )
        .map_err(|e| {
            CliError::ModelLoadFailed(format!("Failed to build the Qwen3.5 base model: {e}"))
        })?
    } else {
        OwnedQuantizedModel::from_mapped(mapped_model)
            .map_err(|e| CliError::ModelLoadFailed(format!("Failed to build quantized model: {e}")))?
    };
    // #3571: before ANY route is chosen — --gpu, --gpu --batch or the CPU server — so no
    // request handler can be handed a stack with nothing in it.
    if let Some(refusal) = zero_layer_refusal(
        &quantized_model.config().architecture,
        quantized_model.layers().len(),
    ) {
        return Err(refusal);
    }
    Ok(quantized_model)
}

/// #3571: a model whose decoder stack resolved to zero layers is refused at load, whatever
/// its architecture, on every serve route.
///
/// Measured on #3571: the Qwen3.5 hybrid reached the servers as its BASE — embeddings, final
/// norm and `lm_head`, no layers, because its Gated-DeltaNet and attention layers live in
/// `Qwen35Model`, which no serve handler calls. With the `<unk>` refusal removed (#3609) the
/// CPU route answered HTTP 200 with 1024 tokens of `"\n"` in 6 s and the GPU route HTTP 500.
/// A stack with nothing in it has no answer to give, so the server does not start: an error
/// at load is the one a user can act on, and a 200 of nothing is the one they cannot see.
pub(super) fn zero_layer_refusal(architecture: &str, layers: usize) -> Option<CliError> {
    (layers == 0).then(|| {
        let hybrid = if architecture == "qwen35" {
            " The Qwen3.5 hybrid's layers are served by `apr run` and `apr chat` today."
        } else {
            ""
        };
        CliError::ModelLoadFailed(format!(
            "'{architecture}' resolved to 0 transformer layers, so no serve route can answer \
             from it — it would decode through the embeddings and lm_head alone. Refused at \
             load (#3571).{hybrid}"
        ))
    })
}

#[cfg(test)]
mod zero_layer_refusal_tests {
    use super::*;

    /// The case table: zero layers is refused naming the architecture and #3571; any layer
    /// at all is admitted.
    #[test]
    fn zero_layers_is_refused_by_name_and_any_layer_is_admitted() {
        let cases: [(&str, usize, bool); 5] = [
            ("qwen35", 0, true),
            ("llama", 0, true),
            ("qwen2", 1, false),
            ("qwen3", 28, false),
            ("qwen35", 24, false),
        ];
        for (arch, layers, refused) in cases {
            let got = zero_layer_refusal(arch, layers);
            assert_eq!(got.is_some(), refused, "({arch}, {layers})");
            if let Some(CliError::ModelLoadFailed(msg)) = got {
                assert!(msg.contains(&format!("'{arch}' resolved to 0 transformer layers")), "{msg}");
                assert!(msg.contains("#3571"), "{msg}");
                assert_eq!(msg.contains("apr chat"), arch == "qwen35", "the hint is the hybrid's: {msg}");
            }
        }
    }

    /// A synthetic Qwen3.5 header: its config declares a block, and it carries only the
    /// embeddings, final norm and `lm_head` — so its BASE is the zero-layer stack #3571 found,
    /// built from nothing but the writer, and CI (which has no model files) exercises the call
    /// site too. It has to be the hybrid: the dense loader already refuses `block_count = 0`
    /// at config validation ("num_layers must be > 0"), so the hybrid base — config says N,
    /// layers are empty — is the one zero-layer stack this tree can build.
    fn zero_block_gguf() -> tempfile::NamedTempFile {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
        let (hidden, vocab) = (8u64, 4u64);
        let f32s = |n: u64| vec![0u8; (n * 4) as usize];
        let tensors = vec![
            GgufTensor {
                name: "token_embd.weight".to_string(),
                shape: vec![hidden, vocab],
                dtype: GgmlType::F32,
                data: f32s(hidden * vocab),
            },
            GgufTensor {
                name: "output_norm.weight".to_string(),
                shape: vec![hidden],
                dtype: GgmlType::F32,
                data: f32s(hidden),
            },
            GgufTensor {
                name: "output.weight".to_string(),
                shape: vec![hidden, vocab],
                dtype: GgmlType::F32,
                data: f32s(hidden * vocab),
            },
        ];
        let u = |k: &str, v: u32| (k.to_string(), GgufValue::Uint32(v));
        let metadata = vec![
            ("general.architecture".to_string(), GgufValue::String("qwen35".to_string())),
            u("qwen35.block_count", 1),
            u("qwen35.embedding_length", 8),
            u("qwen35.feed_forward_length", 16),
            u("qwen35.attention.head_count", 2),
            u("qwen35.attention.head_count_kv", 2),
            u("qwen35.context_length", 32),
            u("qwen35.rope.dimension_count", 4),
            (
                "qwen35.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::Float32(1e-5),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::ArrayString(["<unk>", "a", "b", "c"].map(String::from).to_vec()),
            ),
        ];
        let file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp file");
        let mut writer = std::io::BufWriter::new(&file);
        export_tensors_to_gguf(&mut writer, &tensors, &metadata).expect("write GGUF");
        drop(writer);
        file
    }

    /// CI's row for the call site: needs no model file. Delete the check in
    /// `build_serve_model` and this goes RED — the zero-layer base loads.
    #[test]
    fn serve_refuses_a_synthetic_zero_layer_hybrid_base_at_load() {
        let file = zero_block_gguf();
        let mapped = realizar::gguf::MappedGGUFModel::from_path(file.path()).expect("map");
        match build_serve_model(&mapped) {
            Err(CliError::ModelLoadFailed(msg)) => {
                assert!(msg.contains("'qwen35' resolved to 0 transformer layers"), "{msg}");
            }
            Err(other) => panic!("refused for the wrong reason: {other}"),
            Ok(model) => panic!("a {}-layer stack reached the serve routes", model.layers().len()),
        }
    }

    /// The load path itself refuses, before any route is chosen. Delete the check in
    /// `build_serve_model` and this goes RED: the zero-layer base loads. Needs the real
    /// Qwen3.5 file, whose base IS a zero-layer stack.
    #[test]
    fn serve_refuses_the_qwen35_base_at_load() {
        const MODEL: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";
        if !Path::new(MODEL).exists() {
            eprintln!("SKIP: {MODEL} is absent");
            return;
        }
        let mapped = realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map the GGUF");
        match build_serve_model(&mapped) {
            Err(CliError::ModelLoadFailed(msg)) => {
                assert!(msg.contains("'qwen35' resolved to 0 transformer layers"), "{msg}");
            }
            Err(other) => panic!("refused for the wrong reason: {other}"),
            Ok(model) => panic!("a {}-layer stack reached the serve routes", model.layers().len()),
        }
    }
}

/// Extract the vocabulary from a GGUF model, or refuse to serve it (#3609).
///
/// GH-226 once filled a missing `tokenizer.ggml.tokens` with `token0..tokenN` and put
/// `<unk>` in slot 0, because `BPETokenizer::new` required one. So a model with NO
/// vocabulary loaded, while a model with a real vocabulary that lacked `<unk>` (Qwen3.5)
/// was refused: the less-specified input was the one accepted. The unknown token is
/// now optional, and a model with no vocabulary refuses by name. Placeholder tokens
/// are not a tokenizer.
fn extract_gguf_vocab(mapped_model: &realizar::gguf::MappedGGUFModel) -> Result<Vec<String>> {
    mapped_model
        .model
        .vocabulary()
        .ok_or_else(|| no_vocabulary("the GGUF (no tokenizer.ggml.tokens)"))
}

/// #3609: the refusal every serve path gives for a model with no vocabulary.
fn no_vocabulary(what: &str) -> CliError {
    CliError::ModelLoadFailed(format!(
        "{what} has no vocabulary to tokenize with, so there is nothing to serve; \
         refusing to substitute placeholder tokens (#3609)"
    ))
}

/// #2762: resolve the KV-cache context length for the GGUF + CUDA serve path.
///
/// THE DEFECT. `--context-length` writes `REALIZR_CONTEXT_LENGTH`
/// (`serve::run`, `serve/mod.rs`). This path read `REALIZR_MAX_SEQ_LEN` -- a
/// name that is READ in exactly one place in the tree and WRITTEN in none. So
/// `apr serve run --gpu` on a GGUF model always built its KV cache for 2048
/// whatever the operator asked for, and said so in its own banner:
///
/// ```text
/// $ apr serve run qwen2.5-coder-7b-instruct-q4_k_m.gguf --gpu --context-length 4096
///   Max sequence length: 2048
///   [PAR-119] ... stride=1048576 (ctx=2048)
/// ```
///
/// `1048576 = 4 x 2048 x 128` is the stride #2762 reported. A prompt long
/// enough to walk past the 2048-sized allocation then reads out of bounds:
/// `CUDA_ERROR_ILLEGAL_ADDRESS` is the GOOD case.
///
/// `REALIZR_MAX_SEQ_LEN` is kept, and kept FIRST, because GH-129 introduced it
/// as a memory-constrained-device escape hatch (Jetson, 7.4 GB unified) and an
/// explicit override must still beat the flag. The 2048 default now applies
/// only when neither is set.
///
/// NOTE ON SCOPE. This is NOT the same root as #2774 even though both land in
/// the same allocation. #2774 is a VRAM budget computed before the weights are
/// resident; this is a flag written to one name and read from another. They are
/// the same SHAPE -- a batch/context constant that agrees with the default and
/// so is invisible until a configuration diverges from it -- and they interact:
/// while `--context-length` was ignored the batched KV was half-sized, which is
/// the only reason the 7B appeared to survive c=4 on a 24 GB card at all.
#[cfg(any(all(feature = "inference", feature = "cuda"), test))]
fn resolve_serve_max_seq_len(explicit_override: Option<&str>, context_length: Option<&str>) -> usize {
    const DEFAULT_MAX_SEQ_LEN: usize = 2048;
    explicit_override
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| context_length.and_then(|v| v.parse::<usize>().ok()))
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_SEQ_LEN)
}

/// Start GGUF server with CUDA acceleration (PAR-111).
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_gguf_server_cuda(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    mapped_model: std::sync::Arc<realizar::gguf::MappedGGUFModel>,
    config: &ServerConfig,
    offload: realizar::api::OffloadReport,
) -> Result<()> {
    use realizar::api::{create_router_with_config, AppState, BatchConfig};
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

    println!(
        "{}",
        "Enabling optimized CUDA acceleration (PAR-111)...".cyan()
    );

    // GH-129 + #2762: resolve the KV-cache context length.
    let max_seq_len = resolve_serve_max_seq_len(
        std::env::var("REALIZR_MAX_SEQ_LEN").ok().as_deref(),
        std::env::var("REALIZR_CONTEXT_LENGTH").ok().as_deref(),
    );
    println!("  Max sequence length: {max_seq_len}");

    match OwnedQuantizedModelCuda::with_max_seq_len(quantized_model, 0, max_seq_len) {
        Ok(mut cuda_model) => {
            preload_gpu_weights(&mut cuda_model);

            // GH-129: Free CPU weight copies on unified memory devices (Jetson).
            // After GPU preload, the CPU Vec<u8> copies are redundant — saves ~1 GB.
            // NOTE: Disabled — OwnedQuantizedModelCuda::free_cpu_weights() removed upstream.
            // Re-enable when realizar re-exposes this method.
            // if std::env::var("REALIZR_FREE_CPU_WEIGHTS").as_deref() == Ok("1") {
            //     cuda_model.free_cpu_weights();
            // }

            println!("{}", "CUDA optimized model ready".green());

            let state = AppState::with_cuda_model_and_vocab(cuda_model, vocab)
                .map_err(|e| CliError::InferenceFailed(format!("Failed to create state: {e}")))?
                .with_mapped_gguf_model(mapped_model.clone())
                // PP-14/PP-15/§9 #8: the resolved offload AND this binary's
                // `cfg!` feature list (including `cuda-batch`) reach the served
                // process, so a receipt records what the build was rather than
                // what the operator typed at `--server-feature`.
                .with_offload_report(offload);

            // PMAT-044/088: Spawn continuous batch scheduler for concurrent request handling
            // ITERATION_SCHEDULER=1 enables decode-maximal scheduling (Orca/Sarathi-Serve)
            // PERF-001: gated on `cuda`, not `cuda-batch`. The scheduler module is
            // already `#[cfg(feature = "cuda")]` in aprender-serve, so the extra
            // feature bought nothing except a build every user makes that has NO
            // batch_tx at all -- every request then took the cuda_chat_backend
            // "direct RwLock path (serialized)" fallback. Measured on an RTX 4090,
            // 400-token generations: serialization_index went 1.00/1.99/3.98/7.96
            // (linear in N -- an exclusive lock) without it and 1.00/2.45/2.32/2.41
            // (flat from N=2) with it, for 3.28x aggregate at N=8 and no change at
            // N=1, because the single-request fast path is preserved.
            #[cfg(feature = "cuda")]
            let state = {
                let cuda_model_arc = state.cuda_model().expect("just created").clone();
                let use_iteration = std::env::var("ITERATION_SCHEDULER").as_deref() == Ok("1");
                // PP-13/PP-24: where the admission ceiling came from. The
                // `CUDA_MAX_BATCH` env transport made an operator-set ceiling
                // and a loader-computed one the same string in the same
                // variable; `MaxBatchSizing::source` is what recovers it.
                let admission_reason = realizar::api::admission_ceiling_reason(
                    cuda_model_arc
                        .read()
                        .ok()
                        .and_then(|m| m.max_batch_sizing())
                        .map(|s| s.source),
                );
                let in_flight = realizar::api::InFlightCounter::new();
                if use_iteration {
                    let iter_config =
                        realizar::api::iteration_scheduler::IterationSchedulerConfig::default();
                    println!(
                        "  ITERATION SCHEDULER: max_slots={}, prefill_chunk={} (PMAT-088)",
                        iter_config.max_slots, iter_config.prefill_chunk_size
                    );
                    let report = iter_config.report(admission_reason);
                    let batch_tx =
                        realizar::api::iteration_scheduler::spawn_iteration_scheduler(
                            cuda_model_arc,
                            iter_config,
                        );
                    state
                        .with_cuda_batch_tx(batch_tx)
                        .with_scheduler_report(report, None)
                        .with_verbose(config.verbose)
                } else {
                    let batch_config =
                        realizar::api::cuda_batch_scheduler::CudaBatchConfig::default();
                    println!(
                        "  CONTINUOUS BATCHING: max_batch={}, window={}ms (PMAT-044)",
                        batch_config.max_batch, batch_config.window_ms
                    );
                    let report = batch_config.report(admission_reason);
                    let batch_tx =
                        realizar::api::cuda_batch_scheduler::spawn_cuda_batch_scheduler(
                            cuda_model_arc,
                            batch_config,
                            in_flight.clone(),
                        );
                    state
                        .with_cuda_batch_tx(batch_tx)
                        .with_scheduler_report(report, Some(in_flight))
                        .with_verbose(config.verbose)
                }
            };
            #[cfg(not(feature = "cuda"))]
            let state = state.with_verbose(config.verbose);

            let app = create_router_with_config(state, config.router_config());
            run_server_async(app, &config.bind_addr(), "CUDA-optimized")
        }
        Err(e) => {
            eprintln!(
                "{}",
                format!("CUDA init failed, falling back to CPU: {e}").yellow()
            );
            // #3571: the rebuild goes through the one GGUF serve loader and its refusal.
            let quantized_model = build_serve_model(&mapped_model)?;
            let vocab = extract_gguf_vocab(&mapped_model)?;
            // CUDA init failed and this process fell back to CPU. The offload
            // report travels with it UNCHANGED, so `/v1/effective-config` shows
            // `gpu_layers_resolved` beside `backend_loaded: ["cpu"]` — which is
            // the fallback, visible, rather than a report that quietly agrees
            // with whatever happened.
            run_cpu_server(quantized_model, vocab, Some(mapped_model), config, Some(offload))
        }
    }
}

/// Pre-upload model weights to GPU for maximum performance.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn preload_gpu_weights(cuda_model: &mut realizar::gguf::OwnedQuantizedModelCuda) {
    println!("  Initializing GPU on device 0...");
    match cuda_model.preload_weights_gpu() {
        Ok(bytes) => {
            println!(
                "{}",
                format!("  Pre-uploaded {} MB weights to GPU", bytes / (1024 * 1024)).green()
            );
        }
        Err(e) => {
            eprintln!(
                "{}",
                format!("  Warning: Weight preload failed, using on-demand: {e}").yellow()
            );
        }
    }
}

include!("server_runtime.rs");

/// #2762 falsification. Each case FAILS on the pre-fix source, which read only
/// `REALIZR_MAX_SEQ_LEN` -- a variable nothing in the tree ever sets.
#[cfg(test)]
mod ctx_length_2762_tests {
    use super::resolve_serve_max_seq_len;

    /// RED without the fix: `--context-length 4096` writes REALIZR_CONTEXT_LENGTH
    /// and the old body ignored it, returning the 2048 default.
    #[test]
    fn context_length_flag_reaches_the_kv_cache() {
        assert_eq!(
            resolve_serve_max_seq_len(None, Some("4096")),
            4096,
            "--context-length is written to REALIZR_CONTEXT_LENGTH and must be \
             the KV cache's max_len; ignoring it sizes the batched KV stride \
             from a constant (#2762)"
        );
    }

    /// DISCRIMINATION: stays GREEN both before and after. GH-129's explicit
    /// override must still win, or a Jetson that lowered the context to fit
    /// 7.4 GB of unified memory silently gets 4096 back.
    #[test]
    fn explicit_override_still_beats_the_flag() {
        assert_eq!(resolve_serve_max_seq_len(Some("1024"), Some("4096")), 1024);
    }

    /// DISCRIMINATION: stays GREEN both before and after. With neither set the
    /// historical default is unchanged, so this fix moves no default.
    #[test]
    fn default_is_unchanged_when_nothing_is_set() {
        assert_eq!(resolve_serve_max_seq_len(None, None), 2048);
    }

    /// A garbled value must not resolve to 0 -- `num_kv_heads * 0 * head_dim`
    /// is a zero-length KV cache, which the allocator accepts and the attention
    /// kernel then reads out of.
    #[test]
    fn junk_and_zero_fall_back_to_the_default() {
        assert_eq!(resolve_serve_max_seq_len(None, Some("banana")), 2048);
        assert_eq!(resolve_serve_max_seq_len(None, Some("0")), 2048);
        assert_eq!(resolve_serve_max_seq_len(Some("0"), None), 2048);
    }
}
