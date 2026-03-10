/// GH-87 / GH-471: APR GPU serve with Q4K detection.
///
/// Two paths:
///   1. Q4K APR (ALB-095): spawn_apr_q4k_inference_thread → pool allocator → 15 tok/s
///   2. F32 APR: MappedAprModel → OwnedQuantizedModel → OwnedQuantizedModelCuda (GGUF path)
///
/// Try Q4K path first (avoids redundant is_apr_q4k scan). Falls back to F32 on error.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_apr_server_gpu(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::create_router;

    // GH-471: Try Q4K inference thread first — if model has Q4K tensors, this succeeds.
    // Avoids redundant is_apr_q4k scan that loads the entire 17 GB APR file just to check dtypes.
    match start_apr_q4k_server_gpu(model_path, config) {
        Ok(()) => return Ok(()),
        Err(e) => {
            println!(
                "{}",
                format!("Q4K GPU path unavailable ({e}), trying F32 dequant path").yellow()
            );
        }
    }

    // F32 APR: fall through to OwnedQuantizedModel path (original GH-87)
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
    let vocab = mapped.metadata.get_embedded_vocabulary().unwrap_or_else(|| {
        let vocab_size = mapped.metadata.vocab_size.unwrap_or(32000);
        eprintln!("Warning: No embedded vocabulary in APR, using placeholder tokens");
        let mut v: Vec<String> = (0..vocab_size).map(|i| format!("token{i}")).collect();
        if !v.is_empty() {
            v[0] = "<unk>".to_string();
        }
        v
    });

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

    let app = create_router(state);
    run_server_async(app, &config.bind_addr(), "APR GPU (fused Q4K kernels)")
}

/// GH-471: APR Q4K GPU serve via dedicated inference thread (ALB-095/098).
///
/// Uses realizar's spawn_apr_q4k_inference_thread which:
///   - Pool allocator: single cuMemAlloc for all tensors (~17 GB)
///   - Dedicated thread: CudaExecutor is !Send, owns GPU context
///   - Channel-based: HTTP handler → mpsc → inference thread → oneshot → response
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_apr_q4k_server_gpu(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{apr_q4k_scheduler, create_router, AppState};
    use realizar::apr::AprV2Model;

    eprintln!("[GH-471] Entering Q4K GPU path for {}", model_path.display());
    println!("{}", "Loading APR Q4K model (ALB-095 GPU path)...".cyan());

    let model_str = model_path.to_string_lossy();

    // Load tokenizer from sibling file or embedded metadata
    eprintln!("[GH-471] Loading tokenizer...");
    let vocab = AprV2Model::load_tokenizer_from_sibling(model_path)
        .map(|(v, _, _)| v)
        .or_else(|| {
            AprV2Model::load(model_path)
                .ok()
                .and_then(|m| m.load_embedded_tokenizer())
                .map(|t| t.id_to_token.clone())
        })
        .unwrap_or_else(|| {
            println!(
                "{}",
                "Warning: No vocabulary found, using placeholder tokens".yellow()
            );
            (0..151936).map(|i| format!("token{i}")).collect()
        });

    println!("  Vocab: {} tokens", vocab.len());

    // Spawn Q4K inference thread (loads model, uploads weights to GPU via pool allocator)
    let q4k_tx = apr_q4k_scheduler::spawn_apr_q4k_inference_thread(&model_str)
        .map_err(|e| CliError::InferenceFailed(format!("Q4K inference thread failed: {e}")))?;

    let state = AppState::with_apr_q4k_and_vocab(q4k_tx, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create state: {e}")))?
        .with_verbose(config.verbose);

    println!("{}", "Q4K GPU inference ready (ALB-095)".green());

    let app = create_router(state);
    run_server_async(app, &config.bind_addr(), "APR GPU (Q4K CUDA — ALB-095)")
}

/// GH-88 / F-KERNEL-DISPATCH-001: SafeTensors GPU serve using fused Q4K kernels.
///
/// Loading path: SafeTensors → apr_import(Q4K) → temp APR → MappedAprModel →
/// OwnedQuantizedModel::from_apr() → OwnedQuantizedModelCuda.
/// Uses realizar's built-in AppState + create_router (same as GGUF/APR serve path)
/// for full Ollama-parity endpoints with fused Q4K/Q6K GEMV kernels.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_safetensors_server_gpu(
    model_path: &Path,
    config: &ServerConfig,
) -> Result<()> {
    use aprender::format::{ImportOptions, QuantizationType};
    use realizar::apr::MappedAprModel;
    use realizar::api::{create_router, AppState, BatchConfig};
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
    let vocab = mapped.metadata.get_embedded_vocabulary().unwrap_or_else(|| {
        let vocab_size = mapped.metadata.vocab_size.unwrap_or(32000);
        eprintln!("Warning: No embedded vocabulary in APR, using placeholder tokens");
        let mut v: Vec<String> = (0..vocab_size).map(|i| format!("token{i}")).collect();
        if !v.is_empty() {
            v[0] = "<unk>".to_string();
        }
        v
    });

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

    let app = create_router(state);
    run_server_async(app, &config.bind_addr(), "SafeTensors GPU (fused Q4K kernels)")
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
    let vocab = mapped.metadata.get_embedded_vocabulary().unwrap_or_else(|| {
        let vocab_size = mapped.metadata.vocab_size.unwrap_or(32000);
        eprintln!("Warning: No embedded vocabulary in APR, using placeholder tokens");
        let mut v: Vec<String> = (0..vocab_size).map(|i| format!("token{i}")).collect();
        if !v.is_empty() {
            v[0] = "<unk>".to_string();
        }
        v
    });

    let _ = std::fs::remove_file(&tmp_apr);

    println!("{}", "Q4K CPU inference ready (GH-99)".green());

    run_cpu_server(quantized, vocab, config)
}

/// Build the axum Router for GPU inference endpoints.
///
/// GH-284: Handlers are async with `spawn_blocking` to avoid blocking the runtime.
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
fn build_gpu_router(
    cuda_model: Arc<std::sync::Mutex<realizar::apr::AprV2ModelCuda>>,
    tokenizer: Arc<Option<SafeTensorsTokenizerInfo>>,
    cpu_state: Arc<std::sync::Mutex<AprServerState>>,
) -> axum::Router {
    use axum::{
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };

    let cuda_for_completions = cuda_model.clone();
    let tok_for_completions = tokenizer.clone();
    let cpu_for_completions = cpu_state.clone();
    let cuda_for_chat = cuda_model;
    let tok_for_chat = tokenizer;
    let cpu_for_chat = cpu_state;

    Router::new()
        .route(
            "/health",
            get(|| async {
                Json(serde_json::json!({"status": "healthy", "gpu": true, "gpu_fallback": true}))
            }),
        )
        .route(
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
        .route(
            "/",
            get(|| async {
                "APR v2 GPU Inference Server - POST /v1/completions, /v1/chat/completions"
            }),
        )
}
