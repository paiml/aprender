#[cfg(feature = "inference")]
fn start_apr_server_gpu(
    model_path: &Path,
    model: realizar::apr::AprModel,
    config: &ServerConfig,
    tokenizer: Option<SafeTensorsTokenizerInfo>,
) -> Result<()> {
    use realizar::apr::AprV2ModelCuda;
    use std::sync::Mutex;

    println!("{}", "Initializing CUDA...".dimmed());
    let cuda_model = AprV2ModelCuda::new(model, 0)
        .map_err(|e| CliError::InferenceFailed(format!("CUDA init failed: {e}")))?;

    println!(
        "{}",
        format!(
            "GPU: {} ({} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        )
        .green()
    );

    // GH-261: Load CPU transformer as per-request fallback
    let cpu_state = load_apr_model_state(model_path, config)?;
    println!(
        "{}",
        "CPU fallback: enabled (AprTransformer ready)".cyan()
    );

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let cuda_model = Arc::new(Mutex::new(cuda_model));
    let tokenizer = Arc::new(tokenizer);
    let cpu_state = Arc::new(Mutex::new(cpu_state));

    runtime.block_on(async move {
        let app = build_gpu_router(cuda_model, tokenizer, cpu_state);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        print_gpu_server_banner(&bind_addr);

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Build the axum Router for GPU inference endpoints.
///
/// GH-284: Handlers are async with `spawn_blocking` to avoid blocking the runtime.
#[cfg(feature = "inference")]
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
