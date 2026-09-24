/// Print layer-level trace timing breakdown.
///
/// Only reports measurable wall-clock totals. Per-layer breakdown is not
/// available without BrickProfiler instrumentation — use `apr profile --granular`
/// for real per-operation telemetry.
///
/// PMAT-480: Layer trace shows the 8-step inference state machine with
/// per-step timing. When realizar provides TensorStats (min/max/mean/std/
/// NaN/Inf counts), those are printed per layer. Otherwise falls back to
/// aggregate timing from RunResult.
fn print_layer_trace(result: &RunResult, max_tokens: usize) {
    eprint!("{}", render_layer_trace(result, max_tokens));
}

/// Fixed share of per-token wall time attributed to each step of the 8-step
/// state machine. These are ASSUMPTIONS, not measurements — see
/// [`render_layer_trace`].
fn layer_trace_share(step: &str) -> f64 {
    match step {
        "TRANSFORMER" => 0.85,
        "LM_HEAD" => 0.08,
        "SAMPLE" => 0.02,
        _ => 0.017,
    }
}

/// Render the layer-trace table.
///
/// # These numbers are ESTIMATES and the table says so
///
/// The per-step figures are `wall_ms / tokens * <fixed share>` — a constant
/// split of one measured total, identical for every model and every prompt.
/// The table used to head that column `Time` and print the values bare, so it
/// read as a measurement: it always "proved" TRANSFORMER was 85% of the run,
/// while the real `[BRICK-PROFILE]` block a few lines above the same output
/// reported FFN 42% / Qkv 21% / LmHead 4.5% for that identical run. TOKENIZE,
/// EMBED and DECODE came out equal to the hundredth of a millisecond in every
/// run, which is the tell.
///
/// Two things follow, and both are in the rendered output now:
///
/// 1. The column is `Est. Time`, each value is prefixed `~`, and the share used
///    to derive it is printed next to it. A reader cannot mistake a derived
///    number for a measured one.
/// 2. `TOTAL` is labelled wall-clock **including model load**, and its rate is
///    labelled end-to-end. An unlabelled end-to-end rate printed next to the
///    profiler's decode rate for the same run disagrees with it by more than an
///    order of magnitude, a contradiction inside one screen of output.
///
/// Returned as a `String` so the rendering is directly assertable; the caller
/// prints it to stderr.
fn render_layer_trace(result: &RunResult, max_tokens: usize) -> String {
    use std::fmt::Write as _;

    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
    let total_ms = result.duration_secs * 1000.0;
    let tok_per_sec = if result.duration_secs > 0.0 {
        tokens_generated as f64 / result.duration_secs
    } else {
        0.0
    };

    // 8-step inference state machine trace
    let steps = [
        ("TOKENIZE", "Text → Token IDs"),
        ("EMBED", "Token IDs → Vectors"),
        ("TRANSFORMER", "Vectors → Vectors (×N layers)"),
        ("LM_HEAD", "Hidden → Logits"),
        ("SAMPLE", "Logits → Token ID"),
        ("DECODE", "Token ID → Text"),
    ];

    let per_token_ms = if tokens_generated > 0 {
        total_ms / tokens_generated as f64
    } else {
        total_ms
    };

    let mut out = String::new();
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", "=== Layer Trace (APR-TRACE-001) ===".cyan().bold());
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        "ESTIMATED — per-step values are a fixed share of measured wall time,".yellow()
    );
    let _ = writeln!(
        out,
        "  {}",
        "NOT per-step measurements. Same ratios for every model and prompt.".yellow()
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {:<16} {:<12} {:<8} {}",
        "Step".bold(),
        "Est. Time".bold(),
        "Share".bold(),
        "Description".bold()
    );
    let _ = writeln!(out, "  {}", "─".repeat(66));

    for (name, desc) in &steps {
        let share = layer_trace_share(name);
        let _ = writeln!(
            out,
            "  {:<16} ~{:>8.2}ms  {:>6.1}%  {}",
            name,
            per_token_ms * share,
            share * 100.0,
            desc.dimmed()
        );
    }

    let _ = writeln!(out, "  {}", "─".repeat(66));
    let _ = writeln!(
        out,
        "  {:<16} {:>9.2}ms  measured wall clock, incl. model load",
        "TOTAL", total_ms
    );
    let _ = writeln!(
        out,
        "  {:<16} {:>9.1} tok/s  end-to-end ({} tokens / total wall clock);",
        "RATE", tok_per_sec, tokens_generated
    );
    let _ = writeln!(
        out,
        "  {:<16} {}",
        "",
        "decode-only throughput is the [BRICK-PROFILE] figure, which excludes load".dimmed()
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        "For measured per-brick µs timing: `apr profile <model> --granular`.".dimmed()
    );
    let _ = writeln!(out);
    out
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

/// Roofline classification of one run: `(compute %, memory %, bottleneck, recommendation)`.
///
/// Based on Ivanov et al. (2021): M=1 decode is memory-bandwidth bound.
///
/// #4211: the backend that RAN is an input. The tok/s thresholds alone once
/// told a CUDA run (`GPU used: yes`) that it was CPU-bound and to "Enable GPU
/// with --gpu". On a GPU run the throughput is wall clock including load and
/// one-off CUDA setup, so a low number is not evidence of a CPU bottleneck;
/// the GPU arm never names the CPU and never recommends `--gpu`. A CPU run
/// never claims tensor cores. `None` (backend not reported) keeps the
/// throughput-only tiers, but its top tier names no backend: only a run that
/// reported `GPU used: yes` is told its tensor cores are engaged.
fn classify_roofline(
    tok_per_sec: f64,
    used_gpu: Option<bool>,
) -> (u8, u8, &'static str, &'static str) {
    if used_gpu == Some(true) {
        return if tok_per_sec > 50.0 {
            (
                65,
                35,
                "Compute (GPU tensor cores engaged)",
                "Efficient — GPU-accelerated path active",
            )
        } else {
            (
                20,
                80,
                "Memory bandwidth (GPU VRAM); wall-clock tok/s includes load and one-off CUDA setup",
                "Measure decode-only throughput with `apr bench` before tuning",
            )
        };
    }
    if tok_per_sec > 50.0 {
        return if used_gpu == Some(false) {
            (
                40,
                60,
                "Mixed (CPU SIMD path, memory bandwidth limited)",
                "Efficient for CPU — enable GPU with --gpu for more",
            )
        } else {
            (
                65,
                35,
                "Compute (high throughput; backend not reported)",
                "Efficient at this throughput",
            )
        };
    }
    if tok_per_sec > 20.0 {
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
    }
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

    let (compute_pct, memory_pct, bottleneck, recommendation) =
        classify_roofline(tok_per_sec, result.used_gpu);

    eprintln!();
    eprintln!("{}", "=== Roofline Profile (PMAT-480) ===".cyan().bold());
    eprintln!();
    eprintln!("  Throughput:     {tok_per_sec:.1} tok/s (wall clock, load included)");
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
