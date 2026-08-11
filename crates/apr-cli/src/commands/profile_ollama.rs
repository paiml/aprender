
/// Run Ollama and collect baseline performance
fn run_ollama_comparison(path: &Path, tokens: usize) -> Option<OllamaBaseline> {
    let filename = path
        .file_stem()
        .and_then(|f| f.to_str())
        .unwrap_or("unknown");

    let ollama_model = match map_filename_to_ollama_model(filename) {
        Some(m) => m,
        None => {
            output::warn(&format!(
                "Cannot auto-detect Ollama model name for '{}'. Use known model files.",
                filename
            ));
            return None;
        }
    };

    println!(
        "{}",
        format!(
            "Running Ollama baseline: {} ({} tokens)...",
            ollama_model, tokens
        )
        .dimmed()
    );

    let result = std::process::Command::new("ollama")
        .args([
            "run",
            ollama_model,
            "--verbose",
            "Write a short essay about the history of computing in exactly 128 words.",
        ])
        .output();

    match result {
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            parse_ollama_baseline(&stderr, ollama_model)
        }
        Err(e) => {
            output::warn(&format!("Ollama not available: {e}"));
            None
        }
    }
}

/// Map a model filename to its registered Ollama model tag.
fn map_filename_to_ollama_model(filename: &str) -> Option<&'static str> {
    if filename.contains("qwen2.5-coder-7b") {
        Some("qwen2.5-coder:7b")
    } else if filename.contains("qwen2.5-coder-1.5b") {
        Some("qwen2.5-coder:1.5b")
    } else if filename.contains("TinyLlama") || filename.contains("tinyllama") {
        Some("tinyllama")
    } else {
        None
    }
}

/// Parse decode/prefill eval-rate lines from `ollama run --verbose` stderr.
///
/// IMPORTANT: "prompt eval rate:" also contains "eval rate:", so the decode
/// predicate must match "eval rate:" while rejecting "prompt eval rate:".
fn parse_ollama_baseline(stderr: &str, ollama_model: &str) -> Option<OllamaBaseline> {
    let decode_tok_s = parse_eval_rate(stderr, |l| {
        l.contains("eval rate:") && !l.contains("prompt eval rate:")
    });
    let prefill_tok_s = parse_eval_rate(stderr, |l| l.contains("prompt eval rate:"));

    if decode_tok_s > 0.0 {
        Some(OllamaBaseline {
            decode_tok_s,
            prefill_tok_s,
            model_name: ollama_model.to_string(),
        })
    } else {
        output::warn("Failed to parse Ollama output. Is Ollama running?");
        None
    }
}

/// Find the first line matching `pred` and return its first parseable f64, or 0.0.
fn parse_eval_rate(stderr: &str, pred: impl Fn(&&str) -> bool) -> f64 {
    stderr
        .lines()
        .find(pred)
        .and_then(|l| {
            l.split_whitespace()
                .find(|w| w.parse::<f64>().is_ok())
                .and_then(|w| w.parse::<f64>().ok())
        })
        .unwrap_or(0.0)
}

/// Grade an Ollama parity ratio on the A+/A/B/C/D/F scale.
///
/// C = parity (1.0x), A = 2.0x, F = <0.5x.
fn parity_grade(parity_ratio: f64) -> (&'static str, &'static str, &'static str) {
    match parity_ratio {
        r if r >= 2.0 => ("A+", "Excellent — 2x+ Ollama", "green"),
        r if r >= 1.5 => ("A", "Great — 1.5x+ Ollama", "green"),
        r if r >= 1.0 => ("B", "Good — Ollama parity achieved", "cyan"),
        r if r >= 0.75 => ("C", "Passing — within 75% of Ollama", "yellow"),
        r if r >= 0.5 => ("D", "Below parity — 50-75% of Ollama", "yellow"),
        _ => ("F", "Critical — less than 50% of Ollama", "red"),
    }
}

/// Print the decode + optional prefill throughput rows inside the parity table.
fn print_parity_table(
    results: &RealProfileResults,
    baseline: &OllamaBaseline,
    parity_ratio: f64,
) {
    println!("  ┌────────────┬──────────────┬──────────────┬───────────┐");
    println!("  │ Metric     │ apr          │ Ollama       │ Ratio     │");
    println!("  ├────────────┼──────────────┼──────────────┼───────────┤");

    let decode_ratio_str = format!("{:.2}x", parity_ratio);
    println!(
        "  │ Decode     │ {:>8.1} t/s │ {:>8.1} t/s │ {:>9} │",
        results.decode_tok_s, baseline.decode_tok_s, decode_ratio_str
    );

    if baseline.prefill_tok_s > 0.0 && results.prefill_tok_s > 0.0 {
        let prefill_ratio = results.prefill_tok_s / baseline.prefill_tok_s;
        println!(
            "  │ Prefill    │ {:>8.1} t/s │ {:>8.1} t/s │ {:>8.2}x │",
            results.prefill_tok_s, baseline.prefill_tok_s, prefill_ratio
        );
    }

    println!("  └────────────┴──────────────┴──────────────┴───────────┘");
}

/// Print the methodology citations block.
fn print_methodology_citations() {
    println!("  {}", "Methodology:".dimmed());
    println!(
        "  {}",
        "  Pope et al. (2023) 'Efficiently Scaling Transformer Inference'".dimmed()
    );
    println!(
        "  {}",
        "  Williams et al. (2009) 'Roofline: An Insightful Visual Performance Model'".dimmed()
    );
}

/// Print Ollama comparison report
fn print_ollama_comparison(results: &RealProfileResults, baseline: &OllamaBaseline) {
    println!();
    output::subheader("Ollama Parity Report");
    println!();

    let parity_ratio = if baseline.decode_tok_s > 0.0 {
        results.decode_tok_s / baseline.decode_tok_s
    } else {
        0.0
    };
    let grade = parity_grade(parity_ratio);

    println!(
        "  {} ({})",
        baseline.model_name.cyan(),
        results.backend.to_uppercase()
    );
    println!();

    print_parity_table(results, baseline, parity_ratio);
    println!();

    println!("  Grade: {} — {}", grade.0.bold(), grade.1);
    println!(
        "  Parity: {:.1}% of Ollama decode throughput",
        parity_ratio * 100.0
    );
    println!();

    print_methodology_citations();
}

// ============================================================================
// Roofline & Classification Helpers
// ============================================================================

/// Classify an operation into BrickId category (Attention, FFN, Norm, Other)
///
/// Supports both GPU brick names (QKV, RoPE, Attention, OProj) and
/// CPU brick names (QkvProjection, RopeEmbedding, etc.)
fn classify_operation_category(name: &str) -> String {
    match name {
        // GPU brick names (from indexed.rs start_brick_timer calls)
        "QKV" | "RoPE" | "RopeEmbedding" | "Attention" | "OProj" => "Attention".to_string(),
        "FFNGateUp" | "SwiGLU" | "FFNDown" => "FFN".to_string(),
        "RmsNorm1" | "RmsNorm2" | "OutputNorm" => "Norm".to_string(),
        "LmHead" => "FFN".to_string(), // LM head is a GEMV (same category as FFN projections)
        "Residual1" | "Residual2" => "Other".to_string(),
        // CPU brick names (legacy)
        "QkvProjection" | "AttentionScore" | "AttentionSoftmax" | "AttentionOutput"
        | "OutputProjection" => "Attention".to_string(),
        "GateProjection" | "UpProjection" | "Activation" | "DownProjection" => "FFN".to_string(),
        "RmsNorm" | "LayerNorm" => "Norm".to_string(),
        // Tokenization operations (GH-378)
        "Tokenize" | "TokenizeEncode" | "TokenizeDecode" => "Tokenize".to_string(),
        // Training operations
        "LoraForward" | "LoraBackward" | "OptimizerStep" | "LossCompute" | "TrainStep" => {
            "Training".to_string()
        }
        // Serving operations
        "TTFT" | "Decode" | "BatchGenerate" => "Serving".to_string(),
        _ => "Other".to_string(),
    }
}

/// Classify operation bottleneck (Memory vs Compute bound)
///
/// Q4K decode-time matmul is overwhelmingly memory-bandwidth limited:
/// AI = 2*N / (N/2 bytes_per_weight) = ~4, threshold ~82 for GPU, ~10 for CPU.
/// Only softmax and activation are compute-bound (element-wise).
fn classify_operation_bottleneck(name: &str) -> String {
    match name {
        // Element-wise ops: compute-bound (low memory traffic, high FLOP/byte)
        "SwiGLU" | "Activation" | "RoPE" | "RopeEmbedding" | "AttentionSoftmax" => {
            "COMPUTE".to_string()
        }
        // Everything else: memory-bound (weight/KV reads dominate)
        _ => "MEMORY".to_string(),
    }
}

/// Build real per-layer timing from profiler report's per_layer data
#[cfg(feature = "inference")]
fn build_per_layer_timing(report: &realizar::brick::ProfileReport, num_layers: usize) -> Vec<f64> {
    if num_layers == 0 {
        return vec![];
    }

    // Sum per-layer entries from all per-layer-aware operations
    let mut layer_times = vec![0.0_f64; num_layers];
    for stats in report.operations.values() {
        // Each operation's per_layer vec has one entry per call
        // For N layers × M passes, the entries alternate:
        //   layer0_pass0, layer0_pass1, ..., layer1_pass0, ...
        // But BrickProfiler just appends in order.
        // The most useful view: divide entries across layers
        if stats.per_layer.len() >= num_layers {
            // Distribute entries evenly across layers
            let entries_per_layer = stats.per_layer.len() / num_layers;
            if entries_per_layer > 0 {
                for (layer_idx, time) in layer_times.iter_mut().enumerate() {
                    let start = layer_idx * entries_per_layer;
                    let end = start + entries_per_layer;
                    let layer_total: f64 = stats.per_layer[start..end.min(stats.per_layer.len())]
                        .iter()
                        .sum();
                    *time += layer_total / entries_per_layer as f64; // Average across passes
                }
            }
        }
    }
    layer_times
}

/// Compute category time summary from hotspots
fn compute_category_summary(hotspots: &[Hotspot]) -> CategorySummary {
    let total: f64 = hotspots.iter().map(|h| h.time_us).sum();
    if total <= 0.0 {
        return CategorySummary::default();
    }

    let mut attn = 0.0_f64;
    let mut ffn = 0.0_f64;
    let mut norm = 0.0_f64;
    let mut tokenize = 0.0_f64;
    let mut other = 0.0_f64;

    let mut training = 0.0_f64;
    let mut serving = 0.0_f64;

    for h in hotspots {
        let cat = match h.category.as_deref() {
            Some(c) => c.to_string(),
            None => classify_operation_category(&h.name),
        };
        match cat.as_str() {
            "Attention" => attn += h.time_us,
            "FFN" => ffn += h.time_us,
            "Norm" => norm += h.time_us,
            "Tokenize" => tokenize += h.time_us,
            "Training" => training += h.time_us,
            "Serving" => serving += h.time_us,
            _ => other += h.time_us,
        }
    }

    CategorySummary {
        attention_pct: (attn / total) * 100.0,
        ffn_pct: (ffn / total) * 100.0,
        norm_pct: (norm / total) * 100.0,
        tokenize_pct: (tokenize / total) * 100.0,
        training_pct: (training / total) * 100.0,
        serving_pct: (serving / total) * 100.0,
        other_pct: (other / total) * 100.0,
    }
}

/// Resolve (peak_gflops, peak_bw_gbps, ai_threshold, hardware_model) for the backend.
#[cfg(feature = "inference")]
fn roofline_hardware_specs(is_gpu: bool) -> (f64, f64, f64, String) {
    if is_gpu {
        detect_gpu_hardware()
    } else {
        let hw = trueno::hardware::HardwareCapability::detect();
        (
            hw.cpu.peak_gflops,
            hw.cpu.memory_bw_gbps,
            hw.roofline.cpu_arithmetic_intensity,
            // GH-2395: this read `Unknown Unknown (24 cores, 512)` — vendor and
            // model unresolved, and the SIMD width printed as a bare "512" with no
            // unit, in the same parenthesis as a core count.
            cpu_hardware_label(
                &hw.cpu.vendor,
                &hw.cpu.model,
                hw.cpu.cores,
                hw.cpu.simd.bits(),
            ),
        )
    }
}

/// FLOPs and bytes for one forward pass, estimated from transformer dims.
///
/// Per-layer FLOPs: `QKV(2·h·3h) + OutProj(2·h·h) + 3×FFN(2·h·4h) = 32·h²`.
/// Bytes assume Q4K weights (≈0.5 B/element).
#[cfg(feature = "inference")]
fn roofline_flops_bytes(results: &RealProfileResults) -> (f64, f64) {
    let hidden = results.hidden_dim as f64;
    let vocab = results.vocab_size as f64;
    let layers = results.num_layers as f64;

    let flops_per_layer = 32.0 * hidden * hidden;
    let flops_lm_head = 2.0 * hidden * vocab;
    let total_flops = flops_per_layer * layers + flops_lm_head;

    let bytes_per_layer = 16.0 * hidden * hidden * 0.5;
    let bytes_lm_head = hidden * vocab * 0.5;
    let total_bytes = bytes_per_layer * layers + bytes_lm_head;

    (total_flops, total_bytes)
}

/// Effective per-forward-pass seconds. GPU path subtracts kernel launch
/// overhead (F-BRICKPARITY-01) so efficiency reflects per-kernel throughput,
/// not pipeline throughput (which would double-count idle between launches).
#[cfg(feature = "inference")]
fn roofline_inference_seconds(results: &RealProfileResults, is_gpu: bool) -> f64 {
    if !is_gpu || results.decode_tok_s <= 0.0 {
        return results.total_inference_us / 1_000_000.0;
    }
    let pipeline_sec = 1.0 / results.decode_tok_s;
    let overhead_frac = results.kernel_launch_overhead_pct / 100.0;
    let kernel_active_sec = pipeline_sec * (1.0 - overhead_frac);
    if kernel_active_sec > 0.0 {
        kernel_active_sec
    } else {
        pipeline_sec
    }
}

/// `num / denom` with 0.0 short-circuit when `denom <= 0.0`.
#[cfg(feature = "inference")]
fn safe_ratio(num: f64, denom: f64) -> f64 {
    if denom > 0.0 { num / denom } else { 0.0 }
}

/// Compute roofline analysis using trueno hardware detection.
///
/// F-BRICKPARITY-01: subtracts kernel launch overhead from
/// inference time so that efficiency percentages reflect
/// per-kernel throughput (matching ncu --set roofline), not
/// pipeline throughput (which includes idle between launches).
#[cfg(feature = "inference")]
pub(crate) fn compute_roofline(results: &RealProfileResults) -> RooflineAnalysis {
    let is_gpu = results.backend == "cuda";
    let (peak_compute, peak_bw, ai_threshold, hardware_model) = roofline_hardware_specs(is_gpu);
    let (total_flops, total_bytes) = roofline_flops_bytes(results);
    let inference_sec = roofline_inference_seconds(results, is_gpu);

    let achieved_gflops = safe_ratio(total_flops / 1e9, inference_sec);
    let achieved_bw = safe_ratio(total_bytes / 1e9, inference_sec);
    let ai = safe_ratio(total_flops, total_bytes);
    let compute_eff = safe_ratio(achieved_gflops, peak_compute) * 100.0;
    let memory_eff = safe_ratio(achieved_bw, peak_bw) * 100.0;

    let bottleneck = if ai < ai_threshold {
        "MEMORY BOUND"
    } else {
        "COMPUTE BOUND"
    };

    RooflineAnalysis {
        peak_compute,
        peak_bandwidth_gbps: peak_bw,
        achieved_gflops,
        achieved_bandwidth_gbps: achieved_bw,
        compute_efficiency_pct: compute_eff,
        memory_efficiency_pct: memory_eff,
        arithmetic_intensity: ai,
        ai_threshold,
        bottleneck: bottleneck.to_string(),
        backend: results.backend.clone(),
        hardware_model,
    }
}

/// Detect GPU hardware specs for roofline analysis
/// Returns (peak_tflops_as_gflops, peak_bw_gbps, ai_threshold, model_name)
/// Look up known GPU specs (peak GFLOPS, peak BW GB/s, AI threshold) by name.
pub(crate) fn gpu_specs_by_name(name: &str) -> (f64, f64, f64) {
    match name {
        n if n.contains("4090") => (82_580.0, 1008.0, 82.0),
        n if n.contains("4080") => (48_740.0, 716.8, 68.0),
        n if n.contains("4070") => (29_150.0, 504.2, 57.8),
        n if n.contains("3090") => (35_580.0, 936.0, 38.0),
        n if n.contains("3080") => (29_770.0, 760.0, 39.2),
        n if n.contains("A100") => (19_500.0, 2039.0, 9.6),
        n if n.contains("H100") => (51_200.0, 3350.0, 15.3),
        _ => (30_000.0, 800.0, 37.5),
    }
}

/// Parse nvidia-smi output to extract GPU name. Returns None if unavailable.
pub(crate) fn query_nvidia_smi_gpu_name() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let info = String::from_utf8_lossy(&output.stdout);
    let line = info.lines().next()?;
    let parts: Vec<&str> = line.split(", ").collect();
    if parts.len() >= 2 {
        Some(parts[0].trim().to_string())
    } else {
        None
    }
}

pub(crate) fn detect_gpu_hardware() -> (f64, f64, f64, String) {
    if let Some(gpu_name) = query_nvidia_smi_gpu_name() {
        let (peak_gflops, peak_bw, ai_thresh) = gpu_specs_by_name(&gpu_name);
        return (peak_gflops, peak_bw, ai_thresh, gpu_name);
    }
    // Fallback: generic CUDA GPU
    (30_000.0, 800.0, 37.5, "CUDA GPU (unknown)".to_string())
}

/// Query total GPU VRAM in MB via nvidia-smi.
pub(crate) fn query_gpu_vram_mb() -> Option<f64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let info = String::from_utf8_lossy(&output.stdout);
    let line = info.lines().next()?;
    line.trim().parse::<f64>().ok()
}
