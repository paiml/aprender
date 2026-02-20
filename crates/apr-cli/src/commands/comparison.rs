
/// Print flamegraph SVG (GH-174: supports --output for file output)
fn print_flamegraph(
    results: &RealProfileResults,
    output_path: Option<&Path>,
) -> Result<(), CliError> {
    let mut svg = String::new();
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"400\">\n");
    svg.push_str("  <style>\n");
    svg.push_str("    .frame {{ stroke: #333; }}\n");
    svg.push_str("    .label {{ font-family: monospace; font-size: 12px; }}\n");
    svg.push_str("  </style>\n");
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"#f8f8f8\"/>\n");
    svg.push_str(
        "  <text x=\"400\" y=\"30\" text-anchor=\"middle\" font-size=\"16\" font-weight=\"bold\">\n",
    );
    svg.push_str("    apr profile: Real Telemetry Flamegraph (PMAT-112)\n");
    svg.push_str("  </text>\n");

    let total_time: f64 = results.hotspots.iter().map(|h| h.time_us).sum();
    let mut y = 350.0_f64;
    let height = 25.0_f64;

    for hotspot in results.hotspots.iter().rev() {
        let percent = if total_time > 0.0 {
            (hotspot.time_us / total_time) * 100.0
        } else {
            0.0
        };
        let width = (percent / 100.0) * 760.0;
        let x = 20.0 + ((100.0 - percent) / 200.0) * 760.0;

        // Color based on percentage (hotter = more red)
        let r = (255.0 * (percent / 100.0).min(1.0)) as u8;
        let g = (200.0 * (1.0 - percent / 100.0).max(0.0)) as u8;
        let color = format!("#{:02X}{:02X}50", r, g);

        writeln!(
            svg,
            "  <rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{width:.1}\" height=\"{height:.1}\" fill=\"{color}\" class=\"frame\"/>"
        ).expect("write to String is infallible");
        writeln!(
            svg,
            "  <text x=\"{:.1}\" y=\"{:.1}\" class=\"label\">{} ({:.1}%)</text>",
            x + 5.0,
            y + 16.0,
            hotspot.name,
            percent
        )
        .expect("write to String is infallible");

        y -= height + 2.0;
    }

    svg.push_str("</svg>\n");

    // GH-174: Write to file if output path specified, otherwise print to stdout
    if let Some(path) = output_path {
        std::fs::write(path, &svg).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Failed to write flamegraph to {}: {e}",
                path.display()
            ))
        })?;
        output::success(&format!("Flamegraph written to: {}", path.display()));
    } else {
        println!("{svg}");
    }
    Ok(())
}

// ============================================================================
// Cross-format performance comparison (F-PROFILE-011)
// ============================================================================

/// Run side-by-side performance comparison between two model formats.
///
/// Profiles both models using real inference passes, then prints a table
/// comparing decode tok/s, prefill tok/s, and latency percentiles.
///
/// Usage: `apr profile model.apr --compare model.gguf`
#[cfg(feature = "inference")]
pub(crate) fn run_cross_format_comparison(
    path_a: &Path,
    path_b: &Path,
    warmup: usize,
    measure: usize,
    tokens: usize,
    no_gpu: bool,
) -> Result<(), CliError> {
    let format_a = detect_format(path_a);
    let format_b = detect_format(path_b);

    println!(
        "\n{}",
        format!(
            "Cross-Format Comparison: {} ({}) vs {} ({})",
            path_a.file_name().and_then(|f| f.to_str()).unwrap_or("?"),
            format_a.to_uppercase(),
            path_b.file_name().and_then(|f| f.to_str()).unwrap_or("?"),
            format_b.to_uppercase(),
        )
        .cyan()
        .bold()
    );
    println!("{}", "=".repeat(60));

    // Profile first model
    println!(
        "\n{}",
        format!("[1/2] Profiling {} ({})...", path_a.display(), format_a).dimmed()
    );
    let results_a = if no_gpu {
        profile_real_inference_cpu(path_a, warmup, measure)
    } else {
        profile_gpu_or_cpu(path_a, warmup, measure, tokens)
    }?;

    // Profile second model
    println!(
        "\n{}",
        format!("[2/2] Profiling {} ({})...", path_b.display(), format_b).dimmed()
    );
    let results_b = if no_gpu {
        profile_real_inference_cpu(path_b, warmup, measure)
    } else {
        profile_gpu_or_cpu(path_b, warmup, measure, tokens)
    }?;

    // Print comparison table
    println!("\n{}", "Performance Comparison".green().bold());
    println!("{}", "-".repeat(60));
    println!(
        "{:<24} {:>15} {:>15}",
        "Metric",
        format!("{} ({})", format_a.to_uppercase(), results_a.backend),
        format!("{} ({})", format_b.to_uppercase(), results_b.backend),
    );
    println!("{}", "-".repeat(60));

    print_comparison_row(
        "Decode (tok/s)",
        results_a.decode_tok_s,
        results_b.decode_tok_s,
    );
    print_comparison_row(
        "Prefill (tok/s)",
        results_a.prefill_tok_s,
        results_b.prefill_tok_s,
    );
    print_comparison_row(
        "Throughput (tok/s)",
        results_a.throughput_tok_s,
        results_b.throughput_tok_s,
    );
    print_comparison_row(
        "Latency p50 (ms)",
        results_a.latency_p50_ms,
        results_b.latency_p50_ms,
    );
    print_comparison_row(
        "Latency p99 (ms)",
        results_a.latency_p99_ms,
        results_b.latency_p99_ms,
    );
    println!("{}", "-".repeat(60));

    // Summary
    let decode_ratio = if results_b.decode_tok_s > 0.0 {
        results_a.decode_tok_s / results_b.decode_tok_s
    } else {
        0.0
    };
    let throughput_ratio = if results_b.throughput_tok_s > 0.0 {
        results_a.throughput_tok_s / results_b.throughput_tok_s
    } else {
        0.0
    };

    println!(
        "\n{} is {:.2}x decode, {:.2}x throughput vs {}",
        format_a.to_uppercase(),
        decode_ratio,
        throughput_ratio,
        format_b.to_uppercase(),
    );

    Ok(())
}

/// Try GPU profiling first, fall back to CPU if unavailable.
#[cfg(feature = "inference")]
fn profile_gpu_or_cpu(
    path: &Path,
    warmup: usize,
    measure: usize,
    tokens: usize,
) -> Result<RealProfileResults, CliError> {
    #[cfg(feature = "cuda")]
    {
        match profile_gpu_generation(path, warmup, measure, tokens) {
            Ok(r) => return Ok(r),
            Err(_) => {
                output::info("GPU profiling unavailable, falling back to CPU");
            }
        }
    }
    let _ = tokens; // Unused in CPU-only builds
    profile_real_inference_cpu(path, warmup, measure)
}

/// Print a comparison row with color-coded values.
fn print_comparison_row(label: &str, value_a: f64, value_b: f64) {
    let a_str = if value_a > 0.0 {
        format!("{value_a:.1}")
    } else {
        "N/A".to_string()
    };
    let b_str = if value_b > 0.0 {
        format!("{value_b:.1}")
    } else {
        "N/A".to_string()
    };

    println!("{:<24} {:>15} {:>15}", label, a_str, b_str);
}
