/// Which GPU model `apr profile` builds for a GGUF architecture (SRV-TIM-001 item 6).
///
/// `apr profile` on a Qwen3.5 GGUF used to exit 5: the GPU pass built the dense
/// `OwnedQuantizedModel`, which refuses `qwen35` by name, and the CPU fallback
/// refused it the same way. The hybrid (Gated DeltaNet + gated attention)
/// forward that `apr run` serves is `Qwen35CudaModel` (#3090), so the profiler
/// asks the SAME predicate the runtime dispatches on.
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuProfileLoader {
    /// `OwnedQuantizedModel` → `OwnedQuantizedModelCuda` (dense transformers).
    Dense,
    /// `Qwen35Model` → `Qwen35CudaModel` (the #3090 hybrid forward).
    Hybrid,
}

#[cfg(feature = "inference")]
fn gpu_profile_loader(architecture: &str) -> GpuProfileLoader {
    if realizar::gguf::hybrid_forward_handles(architecture) {
        GpuProfileLoader::Hybrid
    } else {
        GpuProfileLoader::Dense
    }
}

/// Profile the Qwen3.5 hybrid forward on the GPU: prefill and per-token decode
/// latency, then one eager brick-profiled decode pass for the per-op table
/// (`qwen35.gdn.*`, `qwen35.attn.*`, `qwen35.lm_head`) and the roofline.
///
/// Decode runs `forward_single_greedy`, the eager path with a device argmax —
/// the path `apr run` takes at temperature 0 without `QWEN35_CUDA_GRAPH=1`.
#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(feature = "inference", feature = "cuda"))]
fn profile_gpu_generation_qwen35(
    path: &Path,
    mapped: &realizar::gguf::MappedGGUFModel,
    architecture: String,
    tokens_per_pass: usize,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    use realizar::gguf::forward_qwen35::Qwen35Model;
    use realizar::gguf::Qwen35CudaModel;

    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to load the Qwen3.5 base model: {e}"))
    })?;
    let cpu = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).map_err(
        |e| CliError::ValidationFailed(format!("Failed to load the Qwen3.5 hybrid layers: {e}")),
    )?;
    let config = base.config();
    let (num_layers, vocab_size, hidden_dim) =
        (config.num_layers, config.vocab_size, config.hidden_dim);

    let executor = realizar::cuda::CudaExecutor::new(0)
        .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;
    // The same prompt ids as the dense path, so the two reports are comparable.
    let test_tokens: Vec<u32> = vec![791, 7438, 315, 2324, 374];
    let decode_steps = tokens_per_pass.max(PROFILE_PASS_TOKENS);
    let max_seq = test_tokens.len() + decode_steps + 1;
    let mut gpu = Qwen35CudaModel::with_max_seq_len(&cpu, executor, max_seq).map_err(|e| {
        CliError::ValidationFailed(format!("CUDA hybrid model build failed: {e}"))
    })?;
    gpu.set_decode_graph(false);
    eprintln!(
        "{}",
        "Hybrid Qwen3.5 GPU forward (#3090): Gated DeltaNet + gated attention".dimmed()
    );

    // One pass: prefill the prompt, then `steps` greedy decode tokens.
    // Returns (prefill_ms, decode_ms, tokens decoded).
    let run_pass = |gpu: &mut Qwen35CudaModel<'_>, steps: usize| -> Result<(f64, f64, usize), CliError> {
        let gpu_err = |e: realizar::RealizarError| CliError::ValidationFailed(format!("qwen35 GPU pass: {e}"));
        let mut state = gpu.new_state().map_err(gpu_err)?;
        let t0 = Instant::now();
        let logits = gpu.prefill(&test_tokens, &mut state, 0).map_err(gpu_err)?;
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);
        let t1 = Instant::now();
        for step in 0..steps {
            token = gpu
                .forward_single_greedy(token, &mut state, test_tokens.len() + step)
                .map_err(gpu_err)?;
        }
        Ok((prefill_ms, t1.elapsed().as_secs_f64() * 1000.0, steps))
    };

    eprintln!(
        "{}",
        format!("GPU warmup: {warmup_passes} passes x {tokens_per_pass} tokens...").dimmed()
    );
    for i in 0..warmup_passes {
        if let Err(e) = run_pass(&mut gpu, tokens_per_pass) {
            eprintln!("Warning: GPU warmup pass {i} failed: {e}");
        }
    }

    eprintln!(
        "{}",
        format!("GPU measurement: {measure_passes} passes x {tokens_per_pass} tokens...").dimmed()
    );
    let mut decode_times = Vec::with_capacity(measure_passes);
    let mut prefill_times = Vec::with_capacity(measure_passes);
    let mut total_times = Vec::with_capacity(measure_passes);
    let mut total_tokens_generated = 0usize;
    for _ in 0..measure_passes {
        let (prefill_ms, decode_ms, n) = run_pass(&mut gpu, tokens_per_pass)?;
        prefill_times.push(prefill_ms);
        decode_times.push(decode_ms.max(0.1));
        total_times.push(prefill_ms + decode_ms);
        total_tokens_generated += n;
    }
    let stats = compute_profile_stats(
        &decode_times,
        &prefill_times,
        &total_times,
        total_tokens_generated,
        measure_passes,
        test_tokens.len(),
    );

    // The brick pass: eager, one sync per brick. Its decode wall time is the
    // denominator of the launch-overhead figure (PERF-015: same pass, same mode).
    eprintln!("{}", "Per-operation profiling pass (eager, brick-synced)...".dimmed());
    // The profiler defaults to Deferred, where a brick timer never syncs and
    // records the host's launch time, not the kernel's (FALSIFY-GDP-003).
    gpu.executor_mut()
        .set_profiler_sync_mode(trueno::SyncMode::Immediate);
    gpu.executor_mut().enable_profiling();
    gpu.executor_mut().reset_profiler();
    let (_, profile_wall_ms, _) = run_pass(&mut gpu, PROFILE_PASS_TOKENS)?;
    let hotspots = extract_gpu_hotspots(gpu.executor_mut().profiler(), hidden_dim, vocab_size);
    gpu.executor_mut().disable_profiling();
    if hotspots.is_empty() {
        return Err(CliError::ValidationFailed(
            "qwen35 brick pass recorded no bricks — the hybrid forward's timers did not fire"
                .to_string(),
        ));
    }
    let category_summary = Some(compute_category_summary(&hotspots));
    let (launch_overhead_us, launch_overhead_pct) =
        compute_kernel_launch_overhead(&hotspots, profile_wall_ms * 1000.0);

    let mut results = RealProfileResults {
        model_path: path.display().to_string(),
        architecture,
        num_layers,
        vocab_size,
        hidden_dim,
        warmup_passes,
        measure_passes,
        total_inference_us: stats.avg_total_ms * 1000.0,
        throughput_tok_s: stats.decode_tok_s,
        tokens_per_pass: stats.tokens_per_decode,
        hotspots,
        per_layer_us: vec![],
        is_real_data: true,
        roofline: None,
        category_summary,
        backend: "cuda".to_string(),
        latency_p50_ms: stats.p50,
        latency_p95_ms: stats.p95,
        latency_p99_ms: stats.p99,
        latency_min_ms: stats.lat_min,
        latency_max_ms: stats.lat_max,
        prefill_tok_s: stats.prefill_tok_s,
        decode_tok_s: stats.decode_tok_s,
        total_tokens_generated,
        kernel_launch_overhead_pct: launch_overhead_pct,
        kernel_launch_overhead_us: launch_overhead_us,
    };
    results.roofline = Some(compute_roofline(&results));
    Ok(results)
}

#[cfg(all(test, feature = "inference"))]
mod gpu_profile_loader_tests {
    use super::*;

    /// FALSIFY-GDP-016: `apr profile` routes a `qwen35` GGUF to the hybrid GPU
    /// forward, and every other architecture to the dense one. The mutant that
    /// drops the hybrid route (the pre-fix behaviour, exit 5) turns this RED.
    #[test]
    fn qwen35_routes_to_the_hybrid_gpu_forward() {
        assert_eq!(gpu_profile_loader("qwen35"), GpuProfileLoader::Hybrid);
        for dense in ["qwen2", "qwen3", "llama", "qwen3_moe", "qwen3_5", "Qwen35"] {
            assert_eq!(gpu_profile_loader(dense), GpuProfileLoader::Dense, "{dense}");
        }
    }
}
