
/// Profile GPU token generation with full decode loop
///
/// This is the KEY profiling path — it measures what users actually care about:
/// - Full token generation (prefill + decode)
/// - Per-token decode latency with real percentiles (p50, p95, p99)
/// - Prefill vs decode throughput separated
///
/// References:
/// - Williams et al. (2009) "Roofline: An Insightful Visual Performance Model"
/// - Pope et al. (2023) "Efficiently Scaling Transformer Inference"
#[cfg(feature = "inference")]
fn profile_gpu_generation(
    path: &Path,
    tokens_per_pass: usize,
    warmup_passes: usize,
    measure_passes: usize,
) -> Result<RealProfileResults, CliError> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    let format = detect_format(path);

    // Currently GPU generation profiling only for GGUF (primary format)
    if format != "gguf" {
        return Err(CliError::ValidationFailed(format!(
            "GPU generation profiling requires GGUF format (got {format})"
        )));
    }

    println!(
        "{}",
        "Loading model for GPU generation profiling...".dimmed()
    );
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    let architecture = mapped.model.architecture().unwrap_or("unknown").to_string();
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let num_layers = model.config.num_layers;
    let vocab_size = model.config.vocab_size;
    let hidden_dim = model.config.hidden_dim;

    // Try GPU path
    let mut cuda_model = match realizar::gguf::OwnedQuantizedModelCuda::new(model, 0) {
        Ok(m) => m,
        Err(e) => {
            return Err(CliError::ValidationFailed(format!("CUDA init failed: {e}")));
        }
    };

    // Test prompt: "The meaning of life is" — enough tokens for meaningful prefill
    let test_tokens: Vec<u32> = vec![791, 7438, 315, 2324, 374]; // "The meaning of life is"

    let gen_config = QuantizedGenerateConfig {
        max_tokens: tokens_per_pass,
        temperature: 0.0, // Greedy for deterministic profiling
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };

    // Warmup passes
    println!(
        "{}",
        format!(
            "GPU warmup: {} passes x {} tokens...",
            warmup_passes, tokens_per_pass
        )
        .dimmed()
    );
    for _ in 0..warmup_passes {
        let _ = cuda_model.generate_gpu_resident(&test_tokens, &gen_config);
    }

    // Measurement passes — collect per-token timing
    println!(
        "{}",
        format!(
            "GPU measurement: {} passes x {} tokens...",
            measure_passes, tokens_per_pass
        )
        .dimmed()
    );

    let mut per_pass_decode_times: Vec<f64> = Vec::new(); // ms per pass (decode only)
    let mut per_pass_prefill_times: Vec<f64> = Vec::new(); // ms per pass (prefill only)
    let mut per_pass_total_times: Vec<f64> = Vec::new(); // ms per pass (total)
    let mut total_tokens_generated: usize = 0;

    for pass in 0..measure_passes {
        let total_start = Instant::now();

        // Time prefill separately by generating just 1 token first
        let prefill_start = Instant::now();
        let prefill_config = QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![],
            trace: false,
        };
        let _ = cuda_model.generate_gpu_resident(&test_tokens, &prefill_config);
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        per_pass_prefill_times.push(prefill_ms);

        // Now time full generation (includes prefill again — we subtract)
        let gen_start = Instant::now();
        let result = cuda_model.generate_gpu_resident(&test_tokens, &gen_config);
        let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        per_pass_total_times.push(total_ms);

        if let Ok(ref tokens) = result {
            let generated = tokens.len().saturating_sub(test_tokens.len());
            total_tokens_generated += generated;

            // Decode time = total generation time - prefill time (estimated)
            // Better: decode_ms = gen_ms - (prefill portion)
            // Since gen includes its own prefill, decode = gen_ms - prefill_ms
            let decode_ms = (gen_ms - prefill_ms).max(0.1);
            per_pass_decode_times.push(decode_ms);

            if pass == 0 {
                println!(
                    "{}",
                    format!(
                        "  Pass 0: {} tokens in {:.1}ms (prefill: {:.1}ms, decode: {:.1}ms = {:.1} tok/s)",
                        generated,
                        gen_ms,
                        prefill_ms,
                        decode_ms,
                        generated as f64 / (decode_ms / 1000.0)
                    )
                    .dimmed()
                );
            }
        }
    }

    let stats = compute_profile_stats(
        &per_pass_decode_times,
        &per_pass_prefill_times,
        &per_pass_total_times,
        total_tokens_generated,
        measure_passes,
        test_tokens.len(),
    );

    let hotspots = run_brick_profiler_pass(
        &mut cuda_model, &test_tokens, num_layers, hidden_dim, vocab_size,
    );
    let category_summary = Some(compute_category_summary(&hotspots));

    // F-PROFILE-009: Compute kernel launch overhead
    let total_decode_us = stats.avg_decode_ms * 1000.0;
    let (launch_overhead_us, launch_overhead_pct) =
        compute_kernel_launch_overhead(&hotspots, total_decode_us);

    // Compute roofline with the real results
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

    // Compute roofline analysis
    results.roofline = Some(compute_roofline(&results));

    // Restore CUDA graph env
    std::env::remove_var("SKIP_CUDA_GRAPH");

    Ok(results)
}

/// Computed statistics from GPU generation profiling passes.
#[cfg(feature = "inference")]
struct ProfileStats {
    p50: f64,
    p95: f64,
    p99: f64,
    lat_min: f64,
    lat_max: f64,
    avg_decode_ms: f64,
    tokens_per_decode: usize,
    decode_tok_s: f64,
    prefill_tok_s: f64,
    avg_total_ms: f64,
}

/// Compute percentile latencies and throughput from measurement passes.
#[cfg(feature = "inference")]
fn compute_profile_stats(
    decode_times: &[f64],
    prefill_times: &[f64],
    total_times: &[f64],
    total_tokens: usize,
    measure_passes: usize,
    prompt_len: usize,
) -> ProfileStats {
    let mut sorted_decode = decode_times.to_vec();
    sorted_decode.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |pct: f64| -> f64 {
        if sorted_decode.is_empty() {
            return 0.0;
        }
        let idx = ((pct / 100.0) * (sorted_decode.len() - 1) as f64) as usize;
        sorted_decode[idx.min(sorted_decode.len() - 1)]
    };

    let p50 = percentile(50.0);
    let p95 = percentile(95.0);
    let p99 = percentile(99.0);
    let lat_min = sorted_decode.first().copied().unwrap_or(0.0);
    let lat_max = sorted_decode.last().copied().unwrap_or(0.0);

    let avg_decode_ms = if decode_times.is_empty() {
        0.0
    } else {
        decode_times.iter().sum::<f64>() / decode_times.len() as f64
    };
    let tokens_per_decode = if measure_passes > 0 {
        total_tokens / measure_passes
    } else {
        0
    };
    let decode_tok_s = if avg_decode_ms > 0.0 {
        tokens_per_decode as f64 / (avg_decode_ms / 1000.0)
    } else {
        0.0
    };

    let avg_prefill_ms = if prefill_times.is_empty() {
        0.0
    } else {
        prefill_times.iter().sum::<f64>() / prefill_times.len() as f64
    };
    let prefill_tok_s = if avg_prefill_ms > 0.0 {
        prompt_len as f64 / (avg_prefill_ms / 1000.0)
    } else {
        0.0
    };

    let avg_total_ms = if total_times.is_empty() {
        0.0
    } else {
        total_times.iter().sum::<f64>() / total_times.len() as f64
    };

    ProfileStats {
        p50,
        p95,
        p99,
        lat_min,
        lat_max,
        avg_decode_ms,
        tokens_per_decode,
        decode_tok_s,
        prefill_tok_s,
        avg_total_ms,
    }
}

/// Run BrickProfiler pass for per-operation GPU timing breakdown.
///
/// Disables CUDA graph to get individual kernel timing via stream sync.
/// This adds overhead (~2x slower) but gives exact per-brick measurements.
#[cfg(feature = "inference")]
fn run_brick_profiler_pass(
    cuda_model: &mut realizar::gguf::OwnedQuantizedModelCuda,
    test_tokens: &[u32],
    num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> Vec<Hotspot> {
    use realizar::gguf::QuantizedGenerateConfig;

    println!(
        "{}",
        "Per-operation profiling pass (no CUDA graph)...".dimmed()
    );

    // SKIP_CUDA_GRAPH is checked per-call (not cached in OnceLock)
    std::env::set_var("SKIP_CUDA_GRAPH", "1");
    cuda_model.clear_decode_graph();
    cuda_model.enable_profiling();
    cuda_model.reset_profiler();

    let profile_config = QuantizedGenerateConfig {
        max_tokens: 16,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
    };
    let _ = cuda_model.generate_gpu_resident(test_tokens, &profile_config);

    extract_gpu_hotspots(cuda_model, num_layers, hidden_dim, vocab_size)
}

/// Estimate data bytes moved per kernel invocation based on operation name and model dims.
///
/// For memory-bandwidth-bound kernels (GEMV, RMSNorm), the data movement is dominated
/// by reading the weight matrix. We estimate conservatively: read weights + read/write activations.
#[cfg(feature = "inference")]
fn estimate_kernel_data_bytes(name: &str, hidden_dim: usize, vocab_size: usize) -> Option<u64> {
    let name_lower = name.to_lowercase();
    let op = classify_kernel_op(&name_lower);
    compute_kernel_bytes(op, hidden_dim, vocab_size)
}

/// Kernel operation types for data movement estimation.
#[cfg(feature = "inference")]
enum KernelOp {
    QkvProj,
    OutProj,
    FfnGateUp,
    FfnDown,
    LmHead,
    Norm,
    Rope,
    Attention,
    Embed,
    Unknown,
}

/// Classify kernel operation by name substring matching.
#[cfg(feature = "inference")]
fn classify_kernel_op(name: &str) -> KernelOp {
    if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj") {
        KernelOp::QkvProj
    } else if name.contains("o_proj") || name.contains("out_proj") {
        KernelOp::OutProj
    } else if name.contains("gate_proj") || name.contains("up_proj") {
        KernelOp::FfnGateUp
    } else if name.contains("down_proj") {
        KernelOp::FfnDown
    } else if name.contains("lm_head") || name.contains("output") {
        KernelOp::LmHead
    } else if name.contains("rmsnorm") || name.contains("layernorm") {
        KernelOp::Norm
    } else if name.contains("rope") || name.contains("rotary") {
        KernelOp::Rope
    } else if name.contains("softmax") || name.contains("attention") {
        KernelOp::Attention
    } else if name.contains("embed") {
        KernelOp::Embed
    } else {
        KernelOp::Unknown
    }
}

/// Compute estimated data bytes moved for a kernel operation.
#[cfg(feature = "inference")]
fn compute_kernel_bytes(op: KernelOp, hidden_dim: usize, vocab_size: usize) -> Option<u64> {
    let q4k_bpe: f64 = 144.0 / 256.0; // Q4K bytes/element
    let act_rw = (hidden_dim * 8) as u64; // activation read+write
    let h = hidden_dim as f64;

    match op {
        KernelOp::QkvProj | KernelOp::OutProj => {
            Some((h * h * q4k_bpe) as u64 + act_rw)
        }
        KernelOp::FfnGateUp => {
            Some((h * (hidden_dim * 4) as f64 * q4k_bpe) as u64 + act_rw)
        }
        KernelOp::FfnDown => {
            Some(((hidden_dim * 4) as f64 * h * q4k_bpe) as u64 + act_rw)
        }
        KernelOp::LmHead => {
            Some((h * vocab_size as f64 * q4k_bpe) as u64 + (vocab_size * 4 + hidden_dim * 4) as u64)
        }
        KernelOp::Norm => Some(act_rw + (hidden_dim * 4) as u64),
        KernelOp::Rope => Some(act_rw),
        KernelOp::Attention => Some(act_rw * 2),
        KernelOp::Embed => Some((hidden_dim * 4) as u64),
        KernelOp::Unknown => None,
    }
}

/// Extract per-operation GPU hotspots from BrickProfiler after a profiling pass.
///
/// Converts trueno `BrickStats` into our `Hotspot` format with category
/// classification, bottleneck analysis, bandwidth estimation, and time breakdown.
#[cfg(feature = "inference")]
fn extract_gpu_hotspots(
    cuda_model: &realizar::gguf::OwnedQuantizedModelCuda,
    _num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> Vec<Hotspot> {
    let profiler = cuda_model.profiler();
    let total_ns = profiler.total_ns();

    let mut hotspots: Vec<Hotspot> = profiler
        .all_brick_stats()
        .map(|stats| {
            let total_us = stats.total_ns as f64 / 1000.0;
            let pct = if total_ns > 0 {
                100.0 * stats.total_ns as f64 / total_ns as f64
            } else {
                0.0
            };
            let avg_us = if stats.count > 0 {
                total_us / stats.count as f64
            } else {
                0.0
            };

            // F-PROFILE-008: Estimate per-kernel bandwidth
            let data_bytes = estimate_kernel_data_bytes(&stats.name, hidden_dim, vocab_size);
            let bandwidth = data_bytes.and_then(|bytes| {
                if avg_us > 0.0 {
                    // GB/s = bytes / (µs * 1e-6) / 1e9 = bytes / (µs * 1e3)
                    Some(bytes as f64 / (avg_us * 1000.0))
                } else {
                    None
                }
            });

            Hotspot {
                name: stats.name.clone(),
                time_us: total_us,
                percent: pct,
                count: stats.count as usize,
                avg_us,
                min_us: stats.min_us(),
                max_us: stats.max_us(),
                bottleneck: Some(classify_operation_bottleneck(&stats.name)),
                efficiency_pct: bandwidth.map(|bw| (bw / 1008.0 * 100.0).min(100.0)), // RTX 4090 peak: 1008 GB/s
                category: Some(classify_operation_category(&stats.name)),
                bandwidth_gbs: bandwidth,
                data_bytes_per_call: data_bytes,
            }
        })
        .collect();

    // Sort by total time descending (hottest first)
    hotspots.sort_by(|a, b| {
        b.time_us
            .partial_cmp(&a.time_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    hotspots
}

/// Compute kernel launch overhead from profiler data (F-PROFILE-009).
///
/// Returns (total_launch_overhead_us, launch_overhead_percent_of_decode).
/// Launch overhead is estimated as the gap between sum of kernel times and total wall time.
#[cfg(feature = "inference")]
fn compute_kernel_launch_overhead(hotspots: &[Hotspot], total_decode_us: f64) -> (f64, f64) {
    let sum_kernel_us: f64 = hotspots.iter().map(|h| h.time_us).sum();
    // Launch overhead = total decode time - sum of kernel compute time
    // This includes: CUDA launch latency, memory allocation, synchronization
    let overhead_us = (total_decode_us - sum_kernel_us).max(0.0);
    let overhead_pct = if total_decode_us > 0.0 {
        overhead_us / total_decode_us * 100.0
    } else {
        0.0
    };
    (overhead_us, overhead_pct)
}
