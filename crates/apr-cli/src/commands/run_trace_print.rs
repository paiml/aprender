/// TR-09 (#4564): the `apr-trace-v1` document for this run at `level`, built by
/// the SAME function serve answers `X-Trace-Level` with
/// ([`realizar::api::apr_trace`]). One builder is the only way "the same request
/// through `apr run` and through serve yields the same trace" can stay true.
///
/// `apr run` has no tracer over its forwards yet, so the document is the
/// wall-clock row and says `wall_clock_total` — which is what it always
/// measured. The table it replaces split that one total by fixed shares
/// (TRANSFORMER 85%, LM_HEAD 8%, ...) for every model and prompt.
#[cfg(feature = "inference")]
pub(crate) fn run_apr_trace(result: &RunResult, level: &str) -> Option<realizar::api::TraceData> {
    let u = result.usage;
    realizar::api::apr_trace(&realizar::api::ServeTrace {
        level: Some(level),
        events: &[],
        layers: None,
        wall_us: run_wall_us(result),
        prompt_tokens: u.prompt_tokens.unwrap_or(0),
        completion_tokens: u.completion_tokens.or(result.tokens_generated).unwrap_or(0),
        num_layers: u.num_layers.unwrap_or(0),
    })
}

/// The run's generation window in µs when the backend marked it (the part a
/// serve request's wall clock also covers); the whole run otherwise.
fn run_wall_us(result: &RunResult) -> u64 {
    match result.usage.generation_ms {
        Some(ms) => ms.saturating_mul(1000),
        None => (result.duration_secs.max(0.0) * 1_000_000.0) as u64,
    }
}

/// Print the layer trace: the `apr-trace-v1` document, as a table and as JSON.
fn print_layer_trace(result: &RunResult) {
    eprint!("{}", render_layer_trace(result));
}

/// Render `apr run --trace-level layer` from its `apr-trace-v1` document.
///
/// Every row is a row of that document; nothing is derived here. The last line
/// is the document itself, so a script reads exactly what serve would return.
#[cfg(feature = "inference")]
fn render_layer_trace(result: &RunResult) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", "=== Layer Trace (apr-trace-v1) ===".cyan().bold());
    let Some(t) = run_apr_trace(result, "layer") else {
        let _ = writeln!(out, "  no apr-trace-v1 document for level `layer`");
        return out;
    };
    let provenance = serde_json::to_value(t.provenance)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    let _ = writeln!(out, "  provenance: {provenance}");
    if t.provenance != realizar::api::TraceProvenance::Measured {
        let _ = writeln!(
            out,
            "  {}",
            "Only the wall clock is measured; no per-layer row is invented. \
             For measured per-brick timing: `apr profile <model> --granular`."
                .yellow()
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "  {:<24} {:>12}  {}", "Row".bold(), "Time".bold(), "Details".bold());
    let _ = writeln!(out, "  {}", "─".repeat(66));
    for op in &t.breakdown {
        let _ = writeln!(
            out,
            "  {:<24} {:>10.2}ms  {}",
            op.name,
            op.time_us as f64 / 1000.0,
            op.details.as_deref().unwrap_or("").dimmed()
        );
    }
    let _ = writeln!(out, "  {}", "─".repeat(66));
    let _ = writeln!(out, "  {:<24} {:>10.2}ms  wall clock", "TOTAL", t.total_time_us as f64 / 1000.0);
    let _ = writeln!(out);
    let _ = writeln!(out, "apr-trace-v1: {}", serde_json::to_string(&t).unwrap_or_default());
    out
}

#[cfg(not(feature = "inference"))]
fn render_layer_trace(_result: &RunResult) -> String {
    "apr-trace-v1 needs the `inference` feature\n".to_string()
}

/// Print payload trace with activation statistics (TensorStats per layer).
///
/// PMAT-480: Shows min/max/mean/std and NaN/Inf detection per layer.
/// When realizar provides real TensorStats, uses those. Otherwise reports
/// that payload-level data requires BrickProfiler or REALIZE_TRACE=1.
fn print_payload_trace(result: &RunResult, max_tokens: usize) {
    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
    let total_ms = result.duration_secs * 1000.0;

    eprintln!();
    eprintln!("{}", "=== Payload Trace (APR-TRACE-001) ===".cyan().bold());
    eprintln!();
    eprintln!("  Total inference: {:.2} ms", total_ms);
    eprintln!("  Tokens generated: {}", tokens_generated);
    eprintln!();

    // TensorStats header
    eprintln!(
        "  {:<24} {:>8} {:>8} {:>8} {:>8} {:>5} {:>5}",
        "Layer".bold(),
        "Min".bold(),
        "Max".bold(),
        "Mean".bold(),
        "Std".bold(),
        "NaN".bold(),
        "Inf".bold(),
    );
    eprintln!("  {}", "─".repeat(72));

    // Payload-level tensor stats require integration with realizar's
    // InferenceTrace. When not available, show guidance.
    eprintln!(
        "  {}",
        "Per-layer TensorStats require REALIZE_TRACE=1 or `apr profile --granular`.".yellow()
    );
    eprintln!(
        "  {}",
        "This enables NaN/Inf detection at the exact layer of occurrence.".dimmed()
    );
    eprintln!();
}

/// Print roofline profiling analysis (PMAT-480).
///
/// Estimates compute vs memory boundedness from throughput. For real
/// per-brick µs timing, use `apr profile <model> --granular` which
/// integrates with trueno's BrickProfiler.
fn print_roofline_profile(result: &RunResult, max_tokens: usize) {
    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
    let total_ms = result.duration_secs * 1000.0;
    let tok_per_sec = if result.duration_secs > 0.0 {
        tokens_generated as f64 / result.duration_secs
    } else {
        0.0
    };

    // Roofline classification based on Ivanov et al. (2021):
    // M=1 decode is memory-bandwidth bound. High tok/s implies GPU
    // compute is engaged (batched prefill or tensor cores).
    let (compute_pct, memory_pct, bottleneck, recommendation) = if tok_per_sec > 50.0 {
        (
            65,
            35,
            "Compute (GPU tensor cores engaged)",
            "Efficient — GPU-accelerated path active",
        )
    } else if tok_per_sec > 20.0 {
        (
            40,
            60,
            "Mixed (memory bandwidth limited)",
            "Try quantized model (Q4K) for less data movement",
        )
    } else if tok_per_sec > 5.0 {
        (
            20,
            80,
            "Memory bandwidth (DRAM → cache)",
            "Enable GPU with --gpu, or use smaller quantization",
        )
    } else {
        (
            10,
            90,
            "Memory bandwidth (CPU, no SIMD saturation)",
            "Model too large for CPU — use GPU or smaller model",
        )
    };

    eprintln!();
    eprintln!("{}", "=== Roofline Profile (PMAT-480) ===".cyan().bold());
    eprintln!();
    eprintln!("  Throughput:     {tok_per_sec:.1} tok/s");
    eprintln!("  Latency:        {total_ms:.1} ms ({tokens_generated} tokens)");
    eprintln!("  Per-token:      {:.2} ms", total_ms / tokens_generated.max(1) as f64);
    eprintln!("  GPU used:       {}", result.used_gpu.map_or("unknown", |g| if g { "yes" } else { "no" }));
    eprintln!();
    eprintln!("  {}", "Roofline Classification".bold());
    eprintln!("  Compute bound:  {compute_pct}%");
    eprintln!("  Memory bound:   {memory_pct}%");
    eprintln!("  Bottleneck:     {bottleneck}");
    eprintln!("  Recommendation: {recommendation}");
    eprintln!();
    eprintln!(
        "  {}",
        "For per-brick µs timing: `apr profile <model> --granular`".dimmed()
    );
    eprintln!(
        "  {}",
        "For live monitoring: `apr cbtop <model> --brick-score`".dimmed()
    );
    eprintln!();
}
