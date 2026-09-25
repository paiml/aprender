
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
    mapped: Option<&realizar::gguf::MappedGGUFModel>,
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
    // #4254: counted from the file's own tensor table, which is the whole model. The
    // owned model is not: the Qwen3.5 hybrid reaches the servers as its BASE (#3571).
    if let Some(mapped) = mapped {
        source = source.with_parameter_count(gguf_parameter_count(&mapped.model));
    }
    source
}

/// #4254: [`dominant_quantization`]'s rule, read off the GGUF's own tensor table: the
/// modal qtype across every layer's attention-output and FFN projections. The Qwen3.5
/// hybrid has no owned model to ask, so the tensor table is the only measurement it has.
/// The embedding is left out on both paths — on a 0.8B model its Q6_K would outweigh
/// every Q4_K projection, and it is not what "the model's quantization" names.
#[cfg(feature = "inference")]
fn gguf_dominant_quantization(model: &realizar::gguf::GGUFModel) -> Option<&'static str> {
    use std::collections::HashMap;

    const PROJECTIONS: [&str; 4] = [
        ".attn_output.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
        ".ffn_gate.weight",
    ];
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for t in model
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.") && PROJECTIONS.iter().any(|p| t.name.ends_with(p)))
    {
        *counts.entry(t.qtype).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .filter_map(|(qtype, n)| realizar::api::gguf_qtype_name(qtype).map(|name| (name, n)))
        .max_by_key(|&(_, n)| n)
        .map(|(name, _)| name)
}

/// #4254: every element of every tensor the GGUF declares.
#[cfg(feature = "inference")]
fn gguf_parameter_count(model: &realizar::gguf::GGUFModel) -> u64 {
    model.tensors.iter().map(|t| t.dims.iter().product::<u64>()).sum()
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
    offload: Option<realizar::api::OffloadReport>,
) -> Result<()> {
    use realizar::api::AppState;

    // Measure the model BEFORE it is moved into AppState. Anything not
    // measurable here stays absent — `/realize/model` no longer substitutes
    // `size_bytes: 0` / `context_length: 4096` / `quantization: "Q4_K_M"`.
    let model_source = measured_model_source(&quantized_model, mapped_model.as_deref(), config);

    let mut state = AppState::with_quantized_model_and_vocab(quantized_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?
        .with_model_source(model_source);
    if let Some(mapped) = mapped_model {
        state = state.with_mapped_gguf_model(mapped);
    }
    let mut state = state.with_verbose(config.verbose); // GH-152: Pass verbose flag to handlers
    // PP-14/PP-15: the resolution the loader printed, retained so
    // `/v1/effective-config` reports it. `None` on the paths that do not know
    // the layer count — absent, never a fabricated zero.
    if let Some(offload) = offload {
        state = state.with_offload_report(offload);
    }
    serve_router(state, config)
}

/// #3571: serve the Qwen3.5 hybrid from a resident session.
///
/// The dense servers cannot: the hybrid has no dense layers, and the base
/// `apr serve` used to hand them (#3608) was the embeddings, the final norm and
/// `lm_head` alone — HTTP 200 with 1024 tokens of `"\n"` on the CPU route, HTTP
/// 500 on the GPU one. The session is the one `apr chat` holds (#3595): built,
/// uploaded and F2-validated once, its decode state carried from one request to
/// the next.
///
/// The hybrid places every layer on one backend or none, so `--gpu-layers`
/// resolves to all of them or zero, and the line reports where they actually
/// are — a GPU that could not take the model is the printed fallback and
/// `resolved=0`, never a claim.
#[cfg(feature = "inference")]
fn start_qwen35_server(
    mapped_model: std::sync::Arc<realizar::gguf::MappedGGUFModel>,
    config: &ServerConfig,
) -> Result<()> {
    let state = build_qwen35_state(mapped_model, config)?;
    serve_router(state, config)
}

/// The serving state for the Qwen3.5 hybrid: the resident session, its resolution report
/// and its measured source — everything [`start_qwen35_server`] serves, short of the socket.
#[cfg(feature = "inference")]
fn build_qwen35_state(
    mapped_model: std::sync::Arc<realizar::gguf::MappedGGUFModel>,
    config: &ServerConfig,
) -> Result<realizar::api::AppState> {
    use realizar::api::AppState;
    use realizar::gguf::qwen35_session::Qwen35Session;

    let vocab = extract_gguf_vocab(&mapped_model)?;
    let session = Qwen35Session::load(&mapped_model, !config.wants_accelerator())
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load the Qwen3.5 hybrid: {e}")))?;
    let total_layers = u32::try_from(session.num_layers()).unwrap_or(u32::MAX);
    // Refuses a partial --gpu-layers before a request is ever served.
    config.resolve_layers(total_layers)?;
    let on_gpu = session.on_gpu();
    if config.wants_accelerator() && !on_gpu {
        let why = session.notices().join("; ");
        config.refuse_unengaged_accelerator(&why)?;
        eprintln!(
            "{}",
            format!("warning: the default GPU could not be used, serving on the CPU: {why}")
                .yellow()
        );
    }
    let resolved_layers = if on_gpu { total_layers } else { 0 };
    let context_length = session.context_length();
    println!(
        "{}",
        format!(
            "Model ready: Qwen3.5 hybrid, {total_layers} layers resident on the {}, declared context {context_length} tokens",
            if on_gpu { "GPU" } else { "CPU" }
        )
        .green()
    );
    // #3719 (apr code) reads a cell's thinking mode from this line. It reports what the chat
    // endpoint will render — the template the SHARED selection picks for this architecture
    // (`format_messages` -> `detect_format_from_name`, the path run and chat use too, never a
    // template named here: #3755 is replacing the one it picks today), and the thinking mode
    // read off that template's own rendering. It is the server's DEFAULT; #3723 lets a
    // request choose.
    let architecture = mapped_model.model.architecture().unwrap_or("qwen35");
    let template = realizar::chat_template::detect_format_from_name(architecture);
    println!(
        "chat template: {template:?} (thinking {})",
        rendered_thinking_mode(architecture)
    );
    println!(
        "gpu-layers: requested={} resolved={resolved_layers} total={total_layers} (backend={})",
        config
            .gpu_layers
            .map_or_else(|| "none".to_string(), |r| r.to_string()),
        if on_gpu { "cuda" } else { "cpu" }
    );
    let offload = super::offload_report(config, resolved_layers, total_layers);

    let model_source = config
        .model_path
        .as_deref()
        .map(realizar::api::ModelSourceInfo::from_path)
        .unwrap_or_default()
        .with_architecture("qwen35")
        .with_model_max_context_length(context_length)
        .with_context_length(config.context_length)
        .with_parameter_count(gguf_parameter_count(&mapped_model.model));
    // #4254: the v0.69.1 CPU asset served this model with `quantization: null`.
    let model_source = match gguf_dominant_quantization(&mapped_model.model) {
        Some(q) => model_source.with_quantization(q),
        None => model_source,
    };
    let state = AppState::with_qwen35_session(session, mapped_model, vocab)
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create app state: {e}")))?
        .with_model_source(model_source)
        .with_verbose(config.verbose)
        .with_offload_report(offload);
    Ok(state)
}

/// #3571 unit (2), the cop's condition (b): the hybrid's generic stack is legitimately zero
/// layers, so `apr serve` must route it to its resident session BEFORE the zero-layer refusal
/// of unit (1) — and must still refuse any other zero-layer stack.
/// The thinking mode a chat template actually renders for `architecture`: its assistant prefix
/// ending in a closed `</think>` pre-fills an empty think block (thinking off), an open
/// `<think>` forces one (on), anything else leaves it to the model. Read off the rendering, not
/// the template's name, so a replacement template reports itself (#3755).
#[cfg(feature = "inference")]
fn rendered_thinking_mode(architecture: &str) -> &'static str {
    // The render the chat endpoint does (`format_chat_messages` is this call).
    let probe = [realizar::chat_template::ChatMessage::new("user", "x")];
    let rendered =
        realizar::chat_template::format_messages(&probe, Some(architecture)).unwrap_or_default();
    let tail = rendered.trim_end();
    if tail.ends_with("</think>") {
        "off"
    } else if tail.ends_with("<think>") {
        "on"
    } else {
        "the model's choice"
    }
}

#[cfg(test)]
mod qwen35_serve_route_tests {
    use super::*;

    const MODEL: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

    #[test]
    fn qwen35_serve_passes_the_zero_layer_refusal_through_its_session() {
        if !Path::new(MODEL).exists() {
            eprintln!("SKIP: {MODEL} is absent");
            return;
        }
        let mapped = std::sync::Arc::new(
            realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map the GGUF"),
        );
        // start_gguf_server takes the hybrid branch first ...
        assert!(realizar::gguf::hybrid_forward_handles(
            mapped.model.architecture().unwrap_or_default()
        ));
        // ... because the dense loader would (rightly) refuse its zero-layer base,
        match build_serve_model(&mapped) {
            Err(CliError::ModelLoadFailed(msg)) => assert!(msg.contains("#3571"), "{msg}"),
            other => panic!("the dense loader must refuse the base: {:?}", other.map(|m| m.layers().len())),
        }
        // ... and the hybrid route builds a state that serves every layer from the session.
        let config = ServerConfig {
            no_gpu: true,
            ..ServerConfig::default()
        };
        let state = build_qwen35_state(mapped, &config).expect("qwen35 serve passes");
        let served = state.qwen35_session().expect("a resident session");
        let layers = served.session.lock().expect("lock").num_layers();
        assert!(layers > 0, "the session holds the hybrid's layers: {layers}");
    }

    /// #4254: the v0.69.1 CPU asset served Qwen3.5 with `quantization: null` and
    /// `parameter_count: null` in `/v1/effective-config` — the hybrid path built its
    /// source from the path alone. Both are now read off the GGUF's own tensor table.
    #[test]
    fn qwen35_serve_source_carries_quantization_and_parameter_count_4254() {
        if !Path::new(MODEL).exists() {
            eprintln!("SKIP: {MODEL} is absent");
            return;
        }
        let mapped = std::sync::Arc::new(
            realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map the GGUF"),
        );
        let config = ServerConfig {
            no_gpu: true,
            ..ServerConfig::default()
        };
        let state = build_qwen35_state(mapped, &config).expect("qwen35 serve passes");
        let body = realizar::api::effective_config(&state);
        // A Q4_K_M file holds most of its weight elements in Q4_K (some
        // projections are Q6_K); "Q4_K_M" is a file label, not a tensor type.
        assert_eq!(body.model.quantization.as_deref(), Some("Q4_K"));
        // Qwen3.5-0.8B: a count, not a label — anything off by 10x is a wrong sum.
        let n = body.model.parameter_count.expect("parameter_count is measured");
        assert!((500_000_000..1_500_000_000).contains(&n), "parameter_count {n}");
        assert_eq!(body.compute_class, "cpu");
    }

    /// #4254: one process, two endpoints, one answer. `/v1/effective-config` on the
    /// hybrid named no backend (`backend_loaded: []`, `compute_class: unknown`), no
    /// quantization and no parameter count while `/health` said `cpu` — and listed a
    /// `gpu` feature this binary cannot dispatch to.
    #[test]
    fn qwen35_effective_config_agrees_with_health() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        if !Path::new(MODEL).exists() {
            eprintln!("SKIP: {MODEL} is absent");
            return;
        }
        let mapped = std::sync::Arc::new(
            realizar::gguf::MappedGGUFModel::from_path(MODEL).expect("map the GGUF"),
        );
        let want_params =
            realizar::gguf::forward_qwen35::qwen35_known_issue::gguf_param_count(&mapped.model);
        let config = ServerConfig {
            no_gpu: true,
            ..ServerConfig::default()
        };
        let state = build_qwen35_state(mapped, &config).expect("qwen35 state");
        let router = realizar::api::create_router_with_config(state, config.router_config());
        let get = |uri: &'static str| {
            let router = router.clone();
            async move {
                let req = Request::builder().uri(uri).body(Body::empty()).expect("request");
                let resp = router.oneshot(req).await.expect("route");
                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                    .await
                    .expect("body");
                serde_json::from_slice::<serde_json::Value>(&bytes).expect("JSON")
            }
        };
        let (health, ec) = tokio::runtime::Runtime::new()
            .expect("runtime")
            .block_on(async { (get("/health").await, get("/v1/effective-config").await) });

        assert_eq!(health["compute_mode"], "cpu", "{health}");
        assert_eq!(ec["compute_class"], "cpu", "must agree with /health: {ec}");
        assert_eq!(ec["backend_loaded"], serde_json::json!(["cpu"]), "{ec}");
        let q = ec["model"]["quantization"].as_str().expect("quantization named");
        assert!(q.starts_with('Q'), "a GGUF qtype name, got {q:?}");
        assert_eq!(ec["model"]["parameter_count"], want_params, "{ec}");
        assert!(
            (600_000_000..1_000_000_000).contains(&want_params),
            "fixture: the 0.8B file counts {want_params}"
        );
        // The launcher's own set decides which accelerators are dispatchable.
        let features: Vec<&str> = ec["build_features"]
            .as_array()
            .expect("build_features")
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect();
        assert_eq!(features.contains(&"gpu"), cfg!(feature = "wgpu"), "{features:?}");
        assert_eq!(features.contains(&"cuda"), cfg!(feature = "cuda"), "{features:?}");
        let commit = ec["server"]["build_commit"].as_str().expect("build_commit");
        assert_eq!(commit, env!("APR_GIT_SHA"));
        assert!(!commit.ends_with("+no-git"), "built from a checkout: {commit}");
    }

    /// The startup line's thinking mode is read off what the template renders.
    #[test]
    fn the_thinking_mode_is_read_off_the_rendered_template() {
        // Today's qwen35 selection pre-fills an empty think block; whichever template #3755
        // puts in its place, the line must report what that one renders.
        let rendered = realizar::chat_template::format_messages(
            &[realizar::chat_template::ChatMessage::new("user", "x")],
            Some("qwen35"),
        )
        .expect("the qwen35 template renders");
        let want = if rendered.trim_end().ends_with("</think>") {
            "off"
        } else if rendered.trim_end().ends_with("<think>") {
            "on"
        } else {
            "the model's choice"
        };
        assert_eq!(rendered_thinking_mode("qwen35"), want, "{rendered:?}");
        // A plain ChatML model pre-fills nothing: the choice is the model's.
        assert_eq!(rendered_thinking_mode("qwen2"), "the model's choice");
    }
}

/// Serve realizar's full router over `state` until Ctrl+C.
///
/// The half of [`run_cpu_server`] that does not depend on which model the
/// state holds — shared with [`start_qwen35_server`] (#3571), so the hybrid is
/// served by the same router, banner and shutdown as every other GGUF.
#[cfg(feature = "inference")]
fn serve_router(state: realizar::api::AppState, config: &ServerConfig) -> Result<()> {
    use realizar::api::create_router_with_config;

    // aprender#2376(8): the banner is read from the router's own table, not restated.
    // The previous hand-written list named 11 of the 31 mounted routes and omitted
    // /tokenize, /realize/*, /models and the health probes entirely, while a
    // separate list printed before format detection named routes that 404.
    // #3991: read from the STATE the router is built with, before it moves.
    let endpoints = realizar::api::advertised_routes(&config.router_config(), &state);

    // Create realizar's full inference router (Ollama-parity endpoints).
    // --no-cors / --no-metrics must reach the router, not stop at the banner.
    let app = create_router_with_config(state, config.router_config());

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
