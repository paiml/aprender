/// GH-87 / GH-471: APR GPU serve with Q4K detection.
///
/// Two paths:
///   1. Q4K APR (ALB-095): spawn_apr_q4k_inference_thread → pool allocator → 15 tok/s
///   2. F32 APR: MappedAprModel → OwnedQuantizedModel → OwnedQuantizedModelCuda (GGUF path)
///
/// Try Q4K path first (avoids redundant is_apr_q4k scan). Falls back to F32 on error.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_apr_server_gpu(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::create_router_with_config;

    // #170: Q4K inference thread disabled — produces garbage output.
    // All APR models route through OwnedQuantizedModel::from_apr → OwnedQuantizedModelCuda
    // (same proven path as GGUF, validated by qwen-coder-deploy at 148 tok/s).
    use realizar::api::AppState;
    use realizar::apr::MappedAprModel;
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

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

    // #3571: a stack with no layers has no answer to give — refused at load, as on every route.
    if let Some(refusal) =
        zero_layer_refusal(&quantized.config().architecture, quantized.layers().len())
    {
        return Err(refusal);
    }

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

    // Extract vocabulary from embedded APR metadata
    let vocab = mapped
        .metadata
        .get_embedded_vocabulary()
        .ok_or_else(|| no_vocabulary("the APR file (no embedded vocabulary)"))?;

    // GH-88: Extract merge rules for proper BPE tokenization (HuggingFace models)
    let merges = mapped.metadata.get_embedded_merges();

    println!("{}", "Enabling fused CUDA acceleration (GH-87)...".cyan());

    let mut cuda_model = OwnedQuantizedModelCuda::new(quantized, 0)
        .map_err(|e| CliError::InferenceFailed(format!("CUDA init failed: {e}")))?;

    preload_gpu_weights(&mut cuda_model);
    println!("{}", "CUDA fused Q4K model ready".green());

    // GH-88: Use BPE tokenizer with merge rules when available (SafeTensors/HF imports).
    let state = if let Some(merge_rules) = merges {
        AppState::with_cuda_model_and_bpe(cuda_model, vocab, merge_rules)
    } else {
        AppState::with_cuda_model_and_vocab(cuda_model, vocab)
    }
    .map_err(|e| CliError::InferenceFailed(format!("Failed to create state: {e}")))?
    .with_verbose(false) // with_batch_config deferred until realizar API stabilizes
    .with_verbose(config.verbose);

    let app = create_router_with_config(state, config.router_config());
    run_server_async(app, &config.bind_addr(), "APR GPU (fused Q4K kernels)", config.drain_timeout_secs)
}

/// GH-471: APR Q4K GPU serve via dedicated inference thread (ALB-095/098).
///
/// Uses realizar's spawn_apr_q4k_inference_thread which:
///   - Pool allocator: single cuMemAlloc for all tensors (~17 GB)
///   - Dedicated thread: CudaExecutor is !Send, owns GPU context
///   - Channel-based: HTTP handler → mpsc → inference thread → oneshot → response
///
/// PP-LLAMA-001 §9 #8: gated on `cuda`, NOT `cuda-batch`. The implication ran
/// the wrong way — `cuda-batch = ["cuda"]` means enabling `cuda-batch` enables
/// `cuda`, not the reverse, so every plain `--features cuda` build compiled a
/// stub in place of this function ("Q4K batch scheduler not available") and
/// `handlers.rs` caught that `Err` and printed "path declined, trying generic
/// GPU path". A `--features cuda` build therefore served the 30B Q4K MoE
/// through the per-tensor `cuMemAlloc` path that hangs on it, quietly. The
/// stub is deleted; the realizar scheduler this calls is itself
/// `#[cfg(feature = "cuda")]`, so the extra feature bought nothing.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_apr_q4k_server_gpu(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{apr_q4k_scheduler, create_router_with_config, AppState};
    use realizar::apr::AprV2Model;

    eprintln!("[GH-471] Entering Q4K GPU path for {}", model_path.display());
    println!("{}", "Loading APR Q4K model (ALB-095 GPU path)...".cyan());

    let model_str = model_path.to_string_lossy();

    // Load tokenizer from sibling file or embedded metadata
    // ALB-109: Capture EOS token ID — Qwen3 uses 151643, not 0/2
    eprintln!("[GH-471] Loading tokenizer...");
    let (vocab, eos_id) = AprV2Model::load_tokenizer_from_sibling(model_path)
        .map(|(v, _, eos)| (v, eos))
        .or_else(|| {
            AprV2Model::load(model_path)
                .ok()
                .and_then(|m| m.load_embedded_tokenizer())
                .map(|t| (t.id_to_token.clone(), None))
        })
        .ok_or_else(|| {
            no_vocabulary("the APR model (no sibling tokenizer.json, no embedded tokenizer)")
        })?;

    println!("  Vocab: {} tokens", vocab.len());
    if let Some(eos) = eos_id {
        println!("  EOS token ID: {eos}");
    }

    // #3571: the pool path reads its layer count on its own thread, after the upload has
    // begun. Read the count the file declares first, so a zero-layer APR is refused at load
    // like every other route. A file that declares none is left to the pool path, which
    // refuses a missing `num_layers` by name (`parse_apr_q4k_config`).
    // #3791: the declared architecture also picks the chat template — this state holds no
    // model object for the shared selector to read it from.
    let mut declared_architecture = None;
    if let Ok(apr) = AprV2Model::load(model_path) {
        let meta = apr.metadata();
        declared_architecture = meta.architecture.clone();
        if let Some(refusal) = meta.num_layers.and_then(|layers| {
            zero_layer_refusal(meta.architecture.as_deref().unwrap_or("apr"), layers)
        }) {
            return Err(refusal);
        }

        // #3885: THE LOAD-TIME REJECTION THE FALLBACK DEPENDS ON.
        //
        // `start_apr_server`'s comment states the invariant this path is chosen
        // under: "passing a non-Q4K APR errors cleanly and falls through to the
        // generic GPU path". It was false. `parse_apr_q4k_config` validates
        // METADATA ONLY — hidden_size, num_heads, num_layers, vocab_size — and
        // never looks at a tensor, so an all-f16 `.apr` passed it, the thread
        // spawned, `/health` answered, and the weight-format check in
        // `upload_apr_q4k_weights` fired PER REQUEST at prefill:
        //
        //   HTTP 500  Q4K generation failed: Prefill failed at pos 0: GPU error:
        //             Q launch: Invalid launch config: Quantized weight
        //             'model.layers.0.self_attn.q_proj.weight' not cached
        //
        // By then the caller has already returned Ok and the `match … Err(e) =>`
        // fallback can never run. Measured on gx10 0.69.1: all six serve routes
        // 500 on a model whose `run`, `chat` and three golden cases all pass, and
        // which serves correctly with `--no-gpu`.
        //
        // NAMING-INDEPENDENT ON PURPOSE. Checking one well-known tensor would make
        // the gate depend on HF-vs-GGUF naming, which `parse_apr_q4k_config` already
        // has to special-case. A model with NO quantized tensor at all is not a Q4K
        // model under any naming.
        if let Some(reason) = non_q4k_refusal(apr.tensor_index()) {
            return Err(CliError::InferenceFailed(reason));
        }
    }

    // Spawn Q4K inference thread (loads model, uploads weights to GPU via pool allocator)
    let q4k_tx = apr_q4k_scheduler::spawn_apr_q4k_inference_thread(&model_str)
        .map_err(|e| CliError::InferenceFailed(format!("Q4K inference thread failed: {e}")))?;

    let mut state = AppState::with_apr_q4k_and_vocab_eos(q4k_tx, vocab, eos_id)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create state: {e}")))?
        .with_verbose(config.verbose);
    if let Some(architecture) = declared_architecture {
        state = state.with_architecture(architecture);
    }

    println!("{}", "Q4K GPU inference ready (ALB-095)".green());

    let app = create_router_with_config(state, config.router_config());
    run_server_async(app, &config.bind_addr(), "APR GPU (Q4K CUDA — ALB-095)", config.drain_timeout_secs)
}

/// GH-88 / F-KERNEL-DISPATCH-001: SafeTensors GPU serve using fused Q4K kernels.
///
/// Loading path: SafeTensors → apr_import(Q4K) → temp APR → MappedAprModel →
/// OwnedQuantizedModel::from_apr() → OwnedQuantizedModelCuda.
/// Uses realizar's built-in AppState + create_router (same as GGUF/APR serve path)
/// for full Ollama-parity endpoints with fused Q4K/Q6K GEMV kernels.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_safetensors_server_gpu(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<()> {
    use aprender::format::{ImportOptions, QuantizationType};
    use realizar::apr::MappedAprModel;
    use realizar::api::{create_router_with_config, AppState, BatchConfig};
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

    println!("{}", "Converting SafeTensors → Q4K (one-time)...".dimmed());

    let tmp_apr = std::env::temp_dir().join("serve-safetensors-q4k.apr");
    let import_opts = ImportOptions {
        quantize: Some(QuantizationType::Q4K),
        ..ImportOptions::default()
    };
    aprender::format::apr_import(&model_path.display().to_string(), &tmp_apr, import_opts)
        .map_err(|e| CliError::InferenceFailed(format!("SafeTensors→APR Q4K conversion failed: {e}")))?;

    println!("{}", "Loading Q4K model (fused kernels)...".dimmed());

    let mapped = MappedAprModel::from_path(&tmp_apr)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to map temp APR: {e}")))?;

    let quantized = OwnedQuantizedModel::from_apr(&mapped)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create quantized model: {e}")))?;

    // #3571: a stack with no layers has no answer to give — refused at load, as on every route.
    if let Some(refusal) =
        zero_layer_refusal(&quantized.config().architecture, quantized.layers().len())
    {
        return Err(refusal);
    }

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

    // Extract vocabulary from embedded APR metadata
    let vocab = mapped
        .metadata
        .get_embedded_vocabulary()
        .ok_or_else(|| no_vocabulary("the APR file (no embedded vocabulary)"))?;

    // GH-88: Extract merge rules for proper BPE tokenization (HuggingFace models)
    let merges = mapped.metadata.get_embedded_merges();

    println!("{}", "Enabling fused CUDA acceleration (GH-88)...".cyan());

    let mut cuda_model = OwnedQuantizedModelCuda::new(quantized, 0)
        .map_err(|e| CliError::InferenceFailed(format!("CUDA init failed: {e}")))?;

    preload_gpu_weights(&mut cuda_model);
    println!("{}", "CUDA fused Q4K model ready".green());

    let _ = std::fs::remove_file(&tmp_apr);

    // GH-88: Use BPE tokenizer with merge rules when available (SafeTensors imports)
    let state = if let Some(merge_rules) = merges {
        AppState::with_cuda_model_and_bpe(cuda_model, vocab, merge_rules)
    } else {
        AppState::with_cuda_model_and_vocab(cuda_model, vocab)
    }
    .map_err(|e| CliError::InferenceFailed(format!("Failed to create state: {e}")))?
    .with_verbose(false) // with_batch_config deferred until realizar API stabilizes
    .with_verbose(config.verbose);

    let app = create_router_with_config(state, config.router_config());
    run_server_async(app, &config.bind_addr(), "SafeTensors GPU (fused Q4K kernels)", config.drain_timeout_secs)
}

/// GH-99: SafeTensors CPU serve using fused Q4K kernels.
///
/// Loading path: SafeTensors → apr_import(Q4K) → temp APR → MappedAprModel →
/// OwnedQuantizedModel::from_apr() → run_cpu_server (quantized CPU inference).
/// Eliminates 36% throughput gap vs GGUF CPU by using Q4K matmul instead of F32.
#[cfg(feature = "inference")]
fn start_safetensors_server_cpu_quantized(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<()> {
    use aprender::format::{ImportOptions, QuantizationType};
    use realizar::apr::MappedAprModel;
    use realizar::gguf::OwnedQuantizedModel;

    println!("{}", "Converting SafeTensors → Q4K (one-time)...".dimmed());

    let tmp_apr = std::env::temp_dir().join("serve-safetensors-cpu-q4k.apr");
    let import_opts = ImportOptions {
        quantize: Some(QuantizationType::Q4K),
        ..ImportOptions::default()
    };
    aprender::format::apr_import(&model_path.display().to_string(), &tmp_apr, import_opts)
        .map_err(|e| CliError::InferenceFailed(format!("SafeTensors→APR Q4K conversion failed: {e}")))?;

    println!("{}", "Loading Q4K model (fused kernels)...".dimmed());

    let mapped = MappedAprModel::from_path(&tmp_apr)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to map temp APR: {e}")))?;

    let quantized = OwnedQuantizedModel::from_apr(&mapped)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create quantized model: {e}")))?;

    // #3571: a stack with no layers has no answer to give — refused at load, as on every route.
    if let Some(refusal) =
        zero_layer_refusal(&quantized.config().architecture, quantized.layers().len())
    {
        return Err(refusal);
    }

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

    // Extract vocabulary from embedded APR metadata
    let vocab = mapped
        .metadata
        .get_embedded_vocabulary()
        .ok_or_else(|| no_vocabulary("the APR file (no embedded vocabulary)"))?;

    let _ = std::fs::remove_file(&tmp_apr);

    println!("{}", "Q4K CPU inference ready (GH-99)".green());

    // aprender#1789 Option B: APR-format CPU path has no GGUF mmap.
    // This path (APR/SafeTensors CPU) does not resolve a GGUF layer count, so
    // it reports no offload block rather than a fabricated one.
    run_cpu_server(quantized, vocab, None, config, None)
}

/// Build the axum Router for GPU inference endpoints.
///
/// GH-284: Handlers are async with `spawn_blocking` to avoid blocking the runtime.
///
/// HELIX-IDEA-009 / FALSIFY-AUTH-001: every route on the returned router is
/// gated by `auth_gate` via `super::auth::layer`. Pass `AuthGate::disabled()`
/// to bypass.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
fn build_gpu_router(
    cuda_model: Arc<std::sync::Mutex<realizar::apr::AprV2ModelCuda>>,
    tokenizer: Arc<Option<SafeTensorsTokenizerInfo>>,
    cpu_state: Arc<std::sync::Mutex<AprServerState>>,
    auth_gate: super::auth::AuthGate,
) -> axum::Router {
    use axum::{
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };

    let model_name_for_tags = cpu_state
        .lock()
        .map(|s| s.model_name.clone())
        .unwrap_or_else(|_| "apr".to_string());
    let cuda_for_completions = cuda_model.clone();
    let tok_for_completions = tokenizer.clone();
    let cpu_for_completions = cpu_state.clone();
    let cuda_for_chat = cuda_model.clone();
    let tok_for_chat = tokenizer.clone();
    let cpu_for_chat = cpu_state.clone();
    // PMAT-923: Ollama endpoints reuse the SAME GPU chat backend.
    let cuda_for_ochat = cuda_model.clone();
    let tok_for_ochat = tokenizer.clone();
    let cpu_for_ochat = cpu_state.clone();
    let cuda_for_ogen = cuda_model;
    let tok_for_ogen = tokenizer;
    let cpu_for_ogen = cpu_state;

    // #3979: mounted AND recorded; GET / and the 404 come from the record.
    let router = super::route_index::Indexed::new()
        .route(
            "GET",
            "/health",
            get(|| async {
                Json(serde_json::json!({"status": "healthy", "gpu": true, "gpu_fallback": true}))
            }),
        )
        .route(
            "POST",
            "/v1/completions",
            post(move |Json(req): Json<GpuCompletionRequest>| {
                let cuda = cuda_for_completions.clone();
                let tok_info = tok_for_completions.clone();
                let cpu = cpu_for_completions.clone();
                async move {
                    handle_gpu_completion(cuda, tok_info, req, cpu).await
                }
            }),
        )
        .route(
            "POST",
            "/v1/chat/completions",
            post(move |Json(req): Json<serde_json::Value>| {
                let cuda = cuda_for_chat.clone();
                let tok_info = tok_for_chat.clone();
                let cpu = cpu_for_chat.clone();
                async move {
                    handle_gpu_chat_completion(cuda, tok_info, req, cpu).await
                }
            }),
        )
        // PMAT-923/928: Ollama native chat endpoint (drop-in Ollama client).
        // GPU backend is batch (no per-token callback) so `stream:true` emits
        // NDJSON framing over the coalesced result (intermediate done:false +
        // terminal done:true); `stream:false` keeps a single object.
        .route(
            "POST",
            "/api/chat",
            post(move |Json(req): Json<super::ollama::OllamaChatRequest>| {
                let cuda = cuda_for_ochat.clone();
                let tok_info = tok_for_ochat.clone();
                let cpu = cpu_for_ochat.clone();
                async move {
                    let model = super::ollama::model_label(&req.model);
                    let stream = req.stream;
                    let openai_body = super::ollama::ollama_chat_to_openai(&req);
                    let inner =
                        handle_gpu_chat_completion(cuda, tok_info, openai_body, cpu).await;
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
            "POST",
            "/api/generate",
            post(move |Json(req): Json<super::ollama::OllamaGenerateRequest>| {
                let cuda = cuda_for_ogen.clone();
                let tok_info = tok_for_ogen.clone();
                let cpu = cpu_for_ogen.clone();
                async move {
                    let model = super::ollama::model_label(&req.model);
                    let stream = req.stream;
                    let openai_body = super::ollama::ollama_generate_to_openai(&req);
                    let inner =
                        handle_gpu_chat_completion(cuda, tok_info, openai_body, cpu).await;
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
        )
        // PMAT-923: Ollama model-list — clients enumerate models before chatting.
        .route(
            "GET",
            "/api/tags",
            get(move || {
                let model = model_name_for_tags.clone();
                async move { Json(super::ollama::ollama_tags_body(&model)) }
            }),
        )

        .routes(super::ollama::ollama_stub_table())
        .finish();
    super::auth::layer(auth_gate, router)
}

/// Why this `.apr` does not belong on the Q4K pool path, or `None` if it does (#3885).
///
/// PURE, and separated from the loader so it can be exercised both ways. The bug it
/// exists to prevent is not "the check is wrong" but "the check happens too late":
/// `parse_apr_q4k_config` validates METADATA only, so an all-f16 `.apr` passed it,
/// the thread spawned, `/health` answered, and the weight-format check inside
/// `upload_apr_q4k_weights` fired per request at prefill — after the caller had
/// already returned `Ok` and its documented fallback could no longer run.
///
/// NAMING-INDEPENDENT ON PURPOSE. Checking one well-known tensor would tie the gate
/// to HF-vs-GGUF naming, which `parse_apr_q4k_config` already special-cases. A model
/// with no GPU-quantized tensor at all is not a Q4K model under any naming.
fn non_q4k_refusal(tensors: &[realizar::apr::TensorEntry]) -> Option<String> {
    if tensors
        .iter()
        .any(|t| realizar::apr::is_quantized_dtype(&t.dtype))
    {
        return None;
    }
    let dtypes: std::collections::BTreeSet<&str> =
        tensors.iter().map(|t| t.dtype.as_str()).collect();
    Some(format!(
        "not a Q4K APR: none of its {} tensors carries a GPU-quantized dtype (found: {}). \
         The Q4K pool path cannot serve it, so this declines at LOAD and the generic GPU \
         path takes it (#3885).",
        tensors.len(),
        dtypes.into_iter().collect::<Vec<_>>().join(", ")
    ))
}

#[cfg(test)]
mod non_q4k_refusal_tests {
    use super::non_q4k_refusal;
    use realizar::apr::TensorEntry;

    fn t(name: &str, dtype: &str) -> TensorEntry {
        TensorEntry {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: vec![2, 2],
            offset: 0,
            size: 16,
        }
    }

    /// The measured case: gx10's `qwen2.5-coder-1.5b-instruct-fp16.apr`, 339 tensors,
    /// every one `f16`. It must decline HERE, at load, so the caller falls through.
    #[test]
    fn an_all_f16_apr_is_refused_and_the_reason_names_the_dtype() {
        let tensors = vec![t("a.weight", "f16"), t("b.weight", "f16")];
        let reason = non_q4k_refusal(&tensors).expect("an all-f16 apr is not a Q4K apr");
        assert!(reason.contains("f16"), "the reason must name what it found: {reason}");
        assert!(reason.contains('2'), "the reason must name how many it looked at: {reason}");
    }

    /// The other direction, and the one that makes the test able to fail: a model
    /// WITH a quantized tensor must NOT be refused. Without this, a predicate
    /// hardwired to `Some(...)` would pass the case above and break every Q4K model.
    #[test]
    fn an_apr_with_a_quantized_tensor_is_accepted() {
        let tensors = vec![t("a.weight", "f32"), t("b.weight", "Q4_K")];
        assert!(
            non_q4k_refusal(&tensors).is_none(),
            "one GPU-quantized tensor is enough to belong on this path"
        );
    }

    /// BF16 is not a GPU-quantized dtype either — the lambda `.apr` that CUDA and
    /// wgpu both decline (#3889). Distinct from f16 so a predicate that special-cased
    /// one string would fail here.
    #[test]
    fn an_all_bf16_apr_is_also_refused() {
        assert!(non_q4k_refusal(&[t("a.weight", "bf16")]).is_some());
    }

    /// An empty index is refused rather than accepted by vacuous `any()`: `any` over
    /// nothing is false, which happens to give the right answer, and this pins it so
    /// a future rewrite cannot flip it silently.
    #[test]
    fn an_empty_tensor_index_is_refused() {
        assert!(non_q4k_refusal(&[]).is_some());
    }
}
