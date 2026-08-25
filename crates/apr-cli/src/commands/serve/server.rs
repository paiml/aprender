
/// The quantization actually present in the loaded weights.
///
/// Derived from the qtype of the projection tensors themselves — the ground
/// truth — not from GGUF's advisory `general.file_type`, which goes stale when
/// a file is requantized. Returns the modal qtype across every layer's
/// attention-output and FFN projections, or `None` when no layer carries a
/// qtype this build knows a name for. Never guesses "Q4_K_M".
#[cfg(feature = "inference")]
fn dominant_quantization(model: &realizar::gguf::OwnedQuantizedModel) -> Option<&'static str> {
    use std::collections::HashMap;

    let mut counts: HashMap<u32, usize> = HashMap::new();
    for layer in model.layers() {
        let mut tally = |qtype: u32| *counts.entry(qtype).or_insert(0) += 1;
        tally(layer.attn_output_weight.qtype);
        tally(layer.ffn_up_weight.qtype);
        tally(layer.ffn_down_weight.qtype);
        if let Some(gate) = layer.ffn_gate_weight.as_ref() {
            tally(gate.qtype);
        }
    }
    counts
        .into_iter()
        .filter_map(|(qtype, n)| realizar::api::gguf_qtype_name(qtype).map(|name| (name, n)))
        .max_by_key(|&(_, n)| n)
        .map(|(name, _)| name)
}

/// Everything this process actually knows about the model it just loaded.
///
/// Facts come from three places, all measured: the file (size, container
/// format from magic bytes), the loaded weights (architecture, quantization,
/// the model's own advertised context length) and the operator's flags
/// (`--context-length`). No field is defaulted — an unmeasured field is
/// reported as absent by the metadata handlers.
#[cfg(feature = "inference")]
fn measured_model_source(
    model: &realizar::gguf::OwnedQuantizedModel,
    config: &ServerConfig,
) -> realizar::api::ModelSourceInfo {
    let base = config
        .model_path
        .as_deref()
        .map(realizar::api::ModelSourceInfo::from_path)
        .unwrap_or_default();

    let mut source = base
        .with_architecture(model.config().architecture.as_str())
        .with_model_max_context_length(model.config().context_length)
        .with_context_length(config.context_length);
    if let Some(quantization) = dominant_quantization(model) {
        source = source.with_quantization(quantization);
    }
    source
}

/// Run the CPU inference server
///
/// `mapped_model` is `None` for non-GGUF formats (APR / SafeTensors). For
/// GGUF this MUST be `Some(Arc<MappedGGUFModel>)` retained from the loader
/// — aprender#1789 Option B threads this into AppState so qwen3_moe chat
/// dispatch via `try_qwen3_moe_backend` can borrow per-expert tensors
/// directly from the mmap (the mapped model MUST outlive any inference
/// call). For non-MoE GGUF archs this is just an extra Arc reference.
#[cfg(feature = "inference")]
fn run_cpu_server(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    mapped_model: Option<std::sync::Arc<realizar::gguf::MappedGGUFModel>>,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router_with_config, AppState};

    // Measure the model BEFORE it is moved into AppState. Anything not
    // measurable here stays absent — `/realize/model` no longer substitutes
    // `size_bytes: 0` / `context_length: 4096` / `quantization: "Q4_K_M"`.
    let model_source = measured_model_source(&quantized_model, config);

    let mut state = AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?
        .with_model_source(model_source);
    if let Some(mapped) = mapped_model {
        state = state.with_mapped_gguf_model(mapped);
    }
    let state = state.with_verbose(config.verbose); // GH-152: Pass verbose flag to handlers

    // Create realizar's full inference router (Ollama-parity endpoints).
    // --no-cors / --no-metrics must reach the router, not stop at the banner.
    let app = create_router_with_config(state, config.router_config());

    // Create tokio runtime and run server
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    // aprender#2376(8): the banner is read from the router's own table, not restated.
    // The previous hand-written list named 11 of the 31 mounted routes and omitted
    // /tokenize, /realize/*, /models and the health probes entirely, while a
    // separate list printed before format detection named routes that 404.
    let endpoints = realizar::api::advertised_routes(&config.router_config());

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
        println!("{}", "Endpoints:".cyan());
        for endpoint in &endpoints {
            println!("  {endpoint}");
        }
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
/// for maximum throughput on GPU. Measure it with `apr test llm bench`;
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_gguf_server_gpu_batched(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    mapped_model: std::sync::Arc<realizar::gguf::MappedGGUFModel>,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router_with_config, spawn_batch_processor, AppState, BatchConfig};
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
    // aprender#1789 Option B: attach mapped GGUF so qwen3_moe chat dispatch
    // via `try_qwen3_moe_backend` can borrow per-expert tensors.
    let state = AppState::with_cached_model_and_vocab(cached_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?
        .with_mapped_gguf_model(mapped_model)
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
    let router_config = config.router_config();

    // Run everything inside the runtime context
    runtime.block_on(async move {
        // Spawn batch processor task (requires Tokio runtime)
        let batch_tx = spawn_batch_processor(cached_model_arc, batch_config.clone());
        println!("  Batch processor: RUNNING");

        // Add batch support to state
        let state = state.with_batch_config(batch_tx, batch_config);

        // Create router
        let app = create_router_with_config(state, router_config);

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
            // #2696: this printed "Performance: 800+ tok/s (2.8x Ollama)" —
            // a throughput comparison asserted by a server that had measured
            // nothing, on a path that in fact HANGS on four concurrent chat
            // requests. A claim a user reads as a result must come from a
            // measurement; there is none here, so there is no claim.
            "Batched inference enabled. Measure with `apr test llm bench`.".yellow()
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
