/// MoE GGUF benchmark — bridges `apr bench` to the Qwen3-MoE forward path.
///
/// Closes #1749. Pre-fix, `apr bench` against any MoE GGUF (Qwen3-Coder-30B-
/// A3B-Instruct, etc.) routes through the dense `forward_single_with_cache`
/// path which calls `matmul_fused.rs:211` on tensor names that don't exist on
/// MoE models, panicking with `index out of bounds: len=0 but index ≈ 91M`.
///
/// This module detects MoE via `gguf.expert_count().is_some()` and routes to
/// `forward_qwen3_moe` (CPU) or `forward_qwen3_moe_cuda` (GPU), running them
/// autoregressively (re-running prefill each step) to get a tok/s number.
///
/// # Why autoregressive re-prefill
///
/// The existing `forward_qwen3_moe[_cuda]` helpers don't take a KV cache —
/// they run a full forward over `token_ids` each call. For an N-token bench
/// run, the i-th iteration runs forward over `prompt + first (i-1)
/// generated tokens`. This is O(N²) in N but for `--max-tokens 32` that's
/// 32 forwards over a ≤ (prompt_len + 32) sequence — bounded.
///
/// True KV-cache MoE decoding is a separate concern (M-GPU-MOE-3 PR-4
/// throughput cascade); this bench produces a usable upper-bound tok/s
/// number without requiring that work.

#[cfg(feature = "inference")]
fn is_moe_gguf(gguf: &realizar::gguf::GGUFModel) -> bool {
    gguf.expert_count().unwrap_or(0) > 0
}

/// MoE GGUF benchmark entry point. Called from `run_gguf_benchmark` after
/// MoE detection. Loads the model + per-layer MoE tensor descriptors once,
/// then runs warmup + iterations through the appropriate forward path.
#[cfg(feature = "inference")]
fn run_gguf_moe_benchmark(
    path: &Path,
    config: &BenchConfig,
    use_cuda: bool,
    prompt_tokens: &[u32],
    _tracer: &TracerImpl,
) -> Result<BenchResult> {
    use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    if !config.quiet {
        eprintln!("{}", "Loading MoE GGUF model...".yellow());
    }
    let start = Instant::now();

    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to mmap MoE model: {e}")))?;

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create MoE model: {e}")))?;

    // Read MoE config from GGUF metadata. expert_count() must be Some
    // (caller already gated on is_moe_gguf), but use unwrap_or with a panic-
    // safe default to satisfy clippy::disallowed_methods.
    let num_experts = mapped.model.expert_count().ok_or_else(|| {
        CliError::ValidationFailed("MoE bench routed but expert_count() returned None".to_string())
    })?;
    let num_experts_per_tok = mapped.model.expert_used_count().ok_or_else(|| {
        CliError::ValidationFailed(
            "MoE bench: expert_used_count() returned None on a MoE GGUF".to_string(),
        )
    })?;
    let moe_intermediate = mapped.model.expert_feed_forward_length().ok_or_else(|| {
        CliError::ValidationFailed(
            "MoE bench: expert_feed_forward_length() returned None on a MoE GGUF".to_string(),
        )
    })?;

    let num_layers = model.layers().len();
    let mut moe_layers = Vec::with_capacity(num_layers);
    let data = mapped.data();
    for layer_idx in 0..num_layers {
        let layer = load_qwen3_moe_layer(&mapped.model, data, layer_idx).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to load MoE layer {layer_idx}: {e}"))
        })?;
        moe_layers.push(layer);
    }

    let load_time = start.elapsed();
    if !config.quiet {
        eprintln!(
            "{} in {:.2}s ({} layers, {} experts × top-{})",
            "MoE model ready".green(),
            load_time.as_secs_f32(),
            num_layers,
            num_experts,
            num_experts_per_tok
        );
        eprintln!();
    }

    #[cfg(feature = "cuda")]
    if use_cuda {
        return run_cuda_moe_benchmark(
            model,
            moe_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            prompt_tokens,
            config,
            mapped.data().to_vec(),
        );
    }
    #[cfg(not(feature = "cuda"))]
    let _ = use_cuda;

    run_cpu_moe_benchmark(
        model,
        moe_layers,
        num_experts,
        num_experts_per_tok,
        moe_intermediate,
        prompt_tokens,
        config,
        mapped.data().to_vec(),
    )
}

/// CUDA MoE benchmark path. Runs `forward_qwen3_moe_cuda` autoregressively.
#[cfg(all(feature = "inference", feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn run_cuda_moe_benchmark(
    model: realizar::gguf::OwnedQuantizedModel,
    moe_layers: Vec<realizar::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer>,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate: usize,
    prompt_tokens: &[u32],
    config: &BenchConfig,
    data: Vec<u8>,
) -> Result<BenchResult> {
    use realizar::gguf::OwnedQuantizedModelCuda;

    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
        .map_err(|e| CliError::ValidationFailed(format!("MoE CUDA init failed: {e}")))?;

    bench_log(config, &"Running warmup (CUDA MoE)...".yellow().to_string());
    for i in 0..config.warmup {
        let warmup_start = Instant::now();
        let _ = cuda_model
            .forward_qwen3_moe_cuda(
                prompt_tokens,
                &moe_layers,
                num_experts,
                num_experts_per_tok,
                moe_intermediate,
                &data,
            )
            .map_err(|e| CliError::ValidationFailed(format!("MoE warmup forward failed: {e}")))?;
        bench_log_iter(config, i, warmup_start.elapsed(), Some(1));
    }
    bench_log_done(config);

    bench_log(
        config,
        &"Running measurement (CUDA MoE, autoregressive)..."
            .yellow()
            .to_string(),
    );
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let mut tokens = prompt_tokens.to_vec();

    for i in 0..config.iterations {
        let iter_start = Instant::now();
        let logits = cuda_model
            .forward_qwen3_moe_cuda(
                &tokens,
                &moe_layers,
                num_experts,
                num_experts_per_tok,
                moe_intermediate,
                &data,
            )
            .map_err(|e| CliError::ValidationFailed(format!("MoE measure forward failed: {e}")))?;
        let elapsed = iter_start.elapsed();
        if i == 0 {
            first_token_time = elapsed;
        }
        iteration_times.push(elapsed);
        total_tokens += 1;

        // Greedy decode: append argmax(logits) to drive autoregressive step
        if let Some((argmax_idx, _)) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            tokens.push(argmax_idx as u32);
        }
        bench_log_iter(config, i, elapsed, Some(1));

        if total_tokens >= config.max_tokens {
            break;
        }
    }
    bench_log_done(config);

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// CPU MoE benchmark path. Runs `forward_qwen3_moe` autoregressively.
#[cfg(feature = "inference")]
#[allow(clippy::too_many_arguments)]
fn run_cpu_moe_benchmark(
    model: realizar::gguf::OwnedQuantizedModel,
    moe_layers: Vec<realizar::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer>,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate: usize,
    prompt_tokens: &[u32],
    config: &BenchConfig,
    data: Vec<u8>,
) -> Result<BenchResult> {
    bench_log(config, &"Running warmup (CPU MoE)...".yellow().to_string());
    for i in 0..config.warmup {
        let warmup_start = Instant::now();
        let _ = model
            .forward_qwen3_moe(
                prompt_tokens,
                &moe_layers,
                num_experts,
                num_experts_per_tok,
                moe_intermediate,
                &data,
            )
            .map_err(|e| CliError::ValidationFailed(format!("MoE CPU warmup failed: {e}")))?;
        bench_log_iter(config, i, warmup_start.elapsed(), Some(1));
    }
    bench_log_done(config);

    bench_log(
        config,
        &"Running measurement (CPU MoE, autoregressive)..."
            .yellow()
            .to_string(),
    );
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let mut tokens = prompt_tokens.to_vec();

    for i in 0..config.iterations {
        let iter_start = Instant::now();
        let logits = model
            .forward_qwen3_moe(
                &tokens,
                &moe_layers,
                num_experts,
                num_experts_per_tok,
                moe_intermediate,
                &data,
            )
            .map_err(|e| CliError::ValidationFailed(format!("MoE CPU measure failed: {e}")))?;
        let elapsed = iter_start.elapsed();
        if i == 0 {
            first_token_time = elapsed;
        }
        iteration_times.push(elapsed);
        total_tokens += 1;

        if let Some((argmax_idx, _)) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            tokens.push(argmax_idx as u32);
        }
        bench_log_iter(config, i, elapsed, Some(1));

        if total_tokens >= config.max_tokens {
            break;
        }
    }
    bench_log_done(config);

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}
