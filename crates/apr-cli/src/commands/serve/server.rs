
/// Run the CPU inference server
#[cfg(feature = "inference")]
fn run_cpu_server(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router, AppState};

    let state = AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?
        .with_verbose(config.verbose); // GH-152: Pass verbose flag to handlers

    // Create realizar's full inference router (Ollama-parity endpoints)
    let app = create_router(state);

    // Create tokio runtime and run server
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Inference server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Ollama-Parity Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /metrics             - Prometheus metrics");
        println!("  POST /generate            - Text generation");
        println!("  POST /stream/generate     - SSE streaming");
        println!("  POST /batch/generate      - Batch inference");
        println!("  POST /v1/completions      - OpenAI-compatible");
        println!("  POST /v1/chat/completions - Chat completions");
        println!();
        println!(
            "{}",
            "Performance targets: 100+ tok/s CPU, 500+ tok/s GPU".yellow()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start GGUF server with GPU batched inference (2X+ Ollama performance)
///
/// Uses OwnedQuantizedModelCachedSync with continuous batching scheduler
/// for maximum throughput on GPU. Achieves 800+ tok/s (2.8x Ollama).
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_gguf_server_gpu_batched(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router, spawn_batch_processor, AppState, BatchConfig};
    use realizar::gguf::OwnedQuantizedModelCachedSync;

    println!(
        "{}",
        "Enabling GPU batched inference (2X+ Ollama)...".cyan()
    );

    // Create tokio runtime FIRST (needed for batch processor spawn)
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    // Create cached model for scheduler reuse
    // OwnedQuantizedModelCachedSync handles GPU caching internally via warmup_gpu_cache()
    let cached_model = OwnedQuantizedModelCachedSync::new(quantized_model);

    // Warmup GPU cache
    println!("  Warming up GPU cache...");
    match cached_model.warmup_gpu_cache() {
        Ok((memory_bytes, num_layers)) => {
            println!(
                "  GPU cache ready: {:.2} GB ({} layers)",
                memory_bytes as f64 / 1e9,
                num_layers
            );
        }
        Err(e) => {
            eprintln!("  Warning: GPU cache warmup failed: {e}");
        }
    }

    // Create state with cached model and real vocab
    let state = AppState::with_cached_model_and_vocab(cached_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?
        .with_verbose(config.verbose); // GH-152: Pass verbose flag

    // Get Arc'd model for batch processor
    let cached_model_arc = state
        .cached_model()
        .expect("cached_model should exist")
        .clone();

    // Configure batch processing
    let batch_config = BatchConfig::default();
    println!("  Batch window: {}ms", batch_config.window_ms);
    println!("  Optimal batch: {}", batch_config.optimal_batch);
    println!("  GPU threshold: {}", batch_config.gpu_threshold);

    let bind_addr = config.bind_addr();

    // Run everything inside the runtime context
    runtime.block_on(async move {
        // Spawn batch processor task (requires Tokio runtime)
        let batch_tx = spawn_batch_processor(cached_model_arc, batch_config.clone());
        println!("  Batch processor: RUNNING");

        // Add batch support to state
        let state = state.with_batch_config(batch_tx, batch_config);

        // Create router
        let app = create_router(state);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("GPU Batched Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "2X Ollama Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /v1/gpu/status       - GPU cache status");
        println!("  POST /v1/completions      - OpenAI-compatible (batched)");
        println!("  POST /v1/batch/completions - Explicit batch inference");
        println!();
        println!(
            "{}",
            "Performance: 800+ tok/s (2.8x Ollama) with batched requests".yellow()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

// ============================================================================
// Shutdown signal helper
// ============================================================================

/// Shutdown signal handler
#[cfg(feature = "inference")]
pub(crate) async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
}
