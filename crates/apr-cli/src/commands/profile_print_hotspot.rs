
/// Share of total inference time to print for each hotspot row.
///
/// GH-2395: this used to be `time_us / sum(time_us of the rows being printed)`.
/// After `--focus mlp` dropped every row but one, that denominator was the single
/// survivor, so the table reported "100.0%" for an operation the same tool's own
/// Category Summary put at 23.2% — a fabricated number of exactly the kind that
/// gets pasted into an optimisation ticket. `Hotspot::percent` is recorded against
/// TOTAL inference time when the hotspot is built, so use it; the local sum is
/// only a fallback for results that carry no recorded percentages at all.
fn hotspot_display_percents(hotspots: &[Hotspot]) -> Vec<f64> {
    let recorded_total: f64 = hotspots.iter().map(|h| h.percent).sum();
    if recorded_total > 0.0 {
        return hotspots.iter().map(|h| h.percent).collect();
    }
    let total_time: f64 = hotspots.iter().map(|h| h.time_us).sum();
    hotspots
        .iter()
        .map(|h| {
            if total_time > 0.0 {
                (h.time_us / total_time) * 100.0
            } else {
                0.0
            }
        })
        .collect()
}

fn print_hotspot_table(results: &RealProfileResults, granular: bool) {
    output::subheader("Per-Operation Hotspots (ungraphed — SKIP_CUDA_GRAPH=1)");
    println!();

    let percents = hotspot_display_percents(&results.hotspots);
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, hotspot) in results.hotspots.iter().enumerate() {
        let percent = percents[i];
        let bar = output::progress_bar(percent as usize, 100, 20);
        let bottleneck_str = hotspot.bottleneck.as_deref().unwrap_or("-");
        let mut row = vec![
            format!("#{}", i + 1),
            hotspot.name.clone(),
            format!("{:.0}µs", hotspot.time_us),
            format!("{:.1}%", percent),
            format!("{}", hotspot.count),
            bottleneck_str.to_string(),
            bar,
        ];
        if granular {
            let bw_str = hotspot
                .bandwidth_gbs
                .map(|bw| format!(", bw={:.1}GB/s", bw))
                .unwrap_or_default();
            let eff_str = hotspot
                .efficiency_pct
                .map(|e| format!(", eff={:.0}%", e))
                .unwrap_or_default();
            row.push(format!(
                "avg={:.1}µs, min={:.1}µs, max={:.1}µs{}{}",
                hotspot.avg_us, hotspot.min_us, hotspot.max_us, bw_str, eff_str
            ));
        }
        rows.push(row);
    }

    let headers: &[&str] = if granular {
        &[
            "#",
            "Operation",
            "Time",
            "%",
            "Calls",
            "Bottleneck",
            "Bar",
            "Detail",
        ]
    } else {
        &["#", "Operation", "Time", "%", "Calls", "Bottleneck", "Bar"]
    };
    println!("{}", output::table(headers, &rows));
    println!();
    output::info(
        "Per-op times include per-launch host overhead (~1-2µs/call).",
    );
    output::info(
        "Graphed replay amortizes launches to ~0.3µs/node — use these",
    );
    output::info(
        "numbers for fusion-target triage, NOT graphed-mode cost estimation.",
    );
    println!();
}

fn print_category_summary(results: &RealProfileResults) {
    let Some(ref cat) = results.category_summary else {
        return;
    };
    output::subheader("Category Summary (ungraphed)");
    println!();
    let bw = 40;
    let bar = |pct: f64| "█".repeat(((pct / 100.0) * bw as f64) as usize);
    println!(
        "  Attention: {:5.1}%  {}",
        cat.attention_pct,
        bar(cat.attention_pct).cyan()
    );
    println!(
        "  FFN:       {:5.1}%  {}",
        cat.ffn_pct,
        bar(cat.ffn_pct).green()
    );
    println!(
        "  Norm:      {:5.1}%  {}",
        cat.norm_pct,
        bar(cat.norm_pct).yellow()
    );
    if cat.tokenize_pct > 0.0 {
        println!(
            "  Tokenize:  {:5.1}%  {}",
            cat.tokenize_pct,
            bar(cat.tokenize_pct).magenta()
        );
    }
    if cat.training_pct > 0.0 {
        println!(
            "  Training:  {:5.1}%  {}",
            cat.training_pct,
            bar(cat.training_pct).red()
        );
    }
    if cat.serving_pct > 0.0 {
        println!(
            "  Serving:   {:5.1}%  {}",
            cat.serving_pct,
            bar(cat.serving_pct).blue()
        );
    }
    println!(
        "  Other:     {:5.1}%  {}",
        cat.other_pct,
        bar(cat.other_pct).dimmed()
    );
    println!();
}

fn print_kernel_launch_overhead(results: &RealProfileResults) {
    if results.kernel_launch_overhead_us <= 0.0 {
        return;
    }
    output::subheader("Non-Kernel Host Overhead (F-PROFILE-009)");
    println!();
    println!(
        "  Overhead: {:.0}µs ({:.1}% of decode time)",
        results.kernel_launch_overhead_us, results.kernel_launch_overhead_pct
    );
    // PERF-015: both terms now come from the same profiler pass — same token
    // count, same dispatch mode. The label says which path, because it is the
    // UNGRAPHED one and that is not what production runs.
    println!("  (= this pass's wall time − its own kernel sum, UNGRAPHED path)");
    println!("  Includes sync, H2D/D2H copies, dispatch, and any kernel the");
    println!("  brick profiler does not instrument. Which of those dominates is");
    println!("  NOT determined here.");

    // PERF-014: this printed "investigate sampling sync (gpu_argmax D2H)" above
    // 40%. That was a HARDCODED STRING on a threshold — nothing in the tool
    // inspects sampling, D2H, or dispatch. It was quoted back in an
    // investigation as evidence for a root cause. A profiler may report a
    // magnitude it measured; it may not name a cause it did not.
    let msg = if results.kernel_launch_overhead_pct > 40.0 {
        "Non-kernel time dominates this pass — profile the host path to find out why".red()
    } else if results.kernel_launch_overhead_pct > 20.0 {
        "Non-kernel time is significant in this pass".yellow()
    } else {
        "Kernels dominate this pass".green()
    };
    println!("  {msg}");
    println!();
}

fn print_per_layer_timing(results: &RealProfileResults, granular: bool) {
    if !granular || results.per_layer_us.is_empty() {
        return;
    }
    output::subheader("Per-Layer Timing (real)");
    println!();

    let max_t = results.per_layer_us.iter().copied().fold(0.0f64, f64::max);
    let min_t = results
        .per_layer_us
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if max_t > 0.0 && min_t > 0.0 {
        let cv = (max_t - min_t) / f64::midpoint(max_t, min_t);
        if cv < 0.01 {
            println!(
                "  {}",
                output::badge_warn(
                    "WARNING: Per-layer timing shows zero variance (may be estimated)"
                )
            );
        } else {
            println!(
                "  {} (CV={:.1}%)",
                output::badge_pass("Real per-layer timing verified"),
                cv * 100.0
            );
        }
        println!();
    }

    let rows: Vec<Vec<String>> = results
        .per_layer_us
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let bw = if max_t > 0.0 {
                ((t / max_t) * 100.0) as usize
            } else {
                0
            };
            vec![
                format!("Layer {i}"),
                format!("{t:.1}µs"),
                output::progress_bar(bw, 100, 30),
            ]
        })
        .collect();
    println!("{}", output::table(&["Layer", "Time", "Bar"], &rows));
    println!();
}

fn print_roofline_section(results: &RealProfileResults) {
    let Some(ref r) = results.roofline else {
        return;
    };
    output::subheader("Roofline Analysis");
    println!();
    println!(
        "  Hardware:       {} (peak {:.1} GFLOPS, {:.1} GB/s)",
        r.hardware_model, r.peak_compute, r.peak_bandwidth_gbps
    );
    println!(
        "  Achieved:       {:.1} GFLOPS, {:.1} GB/s",
        r.achieved_gflops, r.achieved_bandwidth_gbps
    );
    println!("  Compute eff:    {:.1}% (per-kernel, excl launch overhead)", r.compute_efficiency_pct);
    println!("  Memory eff:     {:.1}% (per-kernel, excl launch overhead)", r.memory_efficiency_pct);
    println!(
        "  Arithmetic int: {:.2} (threshold={:.1})",
        r.arithmetic_intensity, r.ai_threshold
    );
    println!(
        "  {}",
        if r.bottleneck == "MEMORY BOUND" {
            output::badge_info(&r.bottleneck)
        } else {
            output::badge_warn(&r.bottleneck)
        }
    );
    println!();
    if r.bottleneck == "MEMORY BOUND" {
        output::info("Decode is memory-bandwidth limited. Matmul operations transfer");
        output::info("more bytes than FLOPs computed. Focus on memory access patterns.");
    }
    println!();
}

fn print_perf_grade_section(results: &RealProfileResults, show: bool) {
    if !show {
        return;
    }
    let eff = results.roofline.as_ref().map_or(0.0, |r| {
        r.memory_efficiency_pct.max(r.compute_efficiency_pct)
    });
    let grade = PerfGrade::from_efficiency(eff);
    output::subheader("Performance Grade");
    println!();
    println!(
        "  Grade: {}  —  {}",
        grade.label().bold(),
        grade.description()
    );
    println!("  Efficiency: {:.1}%", eff);
    println!();
}

/// One reason the model looks naively implemented.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NaiveSuspicion {
    /// Human-readable reason, printed under `--detect-naive`.
    pub(crate) reason: String,
}

/// Decide whether a profile looks naively implemented, at a GFLOPS `threshold`.
///
/// GH-2395: `--threshold <GFLOPS>` is documented as "GFLOPS threshold for naive
/// detection" but nothing read it — `run()` did `let _ = naive_threshold;` and the
/// only heuristic was "one op takes over half of total time". `--threshold 100000`
/// on a run achieving 22.1 GFLOPS therefore still reported "No obvious naive
/// implementations detected". Both signals now feed the verdict, and the verdict is
/// what `--fail-on-naive` exits on.
///
/// Returns an empty vector when nothing is suspicious.
fn detect_naive_implementations(
    results: &RealProfileResults,
    threshold_gflops: f64,
) -> Vec<NaiveSuspicion> {
    let mut suspicions = Vec::new();

    // Signal 1: achieved compute throughput below the floor the user asked for.
    if let Some(ref r) = results.roofline {
        if threshold_gflops > 0.0 && r.achieved_gflops < threshold_gflops {
            suspicions.push(NaiveSuspicion {
                reason: format!(
                    "achieved {:.1} GFLOPS < --threshold {:.1} GFLOPS",
                    r.achieved_gflops, threshold_gflops
                ),
            });
        }
    }

    // Signal 2: a single operation dominating the forward pass, which is the
    // classic shape of a scalar fallback inside one kernel.
    for h in &results.hotspots {
        if h.count > 0 && h.avg_us > results.total_inference_us * 0.5 {
            suspicions.push(NaiveSuspicion {
                reason: format!(
                    "{} takes {:.1}% of total time ({:.0}µs avg) — check for scalar fallback",
                    h.name, h.percent, h.avg_us
                ),
            });
        }
    }

    suspicions
}

fn print_naive_detection(results: &RealProfileResults, detect: bool, threshold_gflops: f64) {
    if !detect {
        return;
    }
    output::subheader("Naive Implementation Detection");
    println!();
    let suspicions = detect_naive_implementations(results, threshold_gflops);
    if suspicions.is_empty() {
        println!(
            "  {} No obvious naive implementations detected (threshold {:.1} GFLOPS)",
            output::badge_pass("OK"),
            threshold_gflops
        );
    } else {
        for s in &suspicions {
            println!("  {} {}", output::badge_warn("NAIVE?"), s.reason);
        }
    }
    println!();
}

fn print_generation_performance(results: &RealProfileResults) {
    if results.decode_tok_s <= 0.0 && results.prefill_tok_s <= 0.0 {
        return;
    }
    output::subheader("Generation Performance");
    println!();
    println!(
        "{}",
        output::kv_table(&[
            (
                "Decode throughput",
                format!("{:.1} tok/s", results.decode_tok_s)
            ),
            (
                "Prefill throughput",
                format!("{:.1} tok/s", results.prefill_tok_s)
            ),
            (
                "Tokens generated",
                format!("{}", results.total_tokens_generated)
            ),
        ])
    );
    println!();
}

fn print_latency_percentiles(results: &RealProfileResults) {
    if results.latency_p50_ms <= 0.0 {
        return;
    }
    output::subheader("Latency Distribution (decode pass)");
    println!();
    println!(
        "{}",
        output::kv_table(&[
            ("p50 (median)", format!("{:.1} ms", results.latency_p50_ms)),
            ("p95", format!("{:.1} ms", results.latency_p95_ms)),
            ("p99", format!("{:.1} ms", results.latency_p99_ms)),
            ("min", format!("{:.1} ms", results.latency_min_ms)),
            ("max", format!("{:.1} ms", results.latency_max_ms)),
        ])
    );
    println!();
}

fn print_profile_summary(results: &RealProfileResults) {
    output::subheader("Summary");
    println!();
    output::metric(
        "Avg forward pass",
        format!(
            "{:.1}µs ({:.2}ms)",
            results.total_inference_us,
            results.total_inference_us / 1000.0
        ),
        "",
    );
    output::metric(
        "Throughput",
        format!("{:.2}", results.throughput_tok_s),
        "tok/s",
    );
    output::metric("Tokens per pass", results.tokens_per_pass, "");
    output::metric("Operations profiled", results.hotspots.len(), "");
    println!();
}

/// Print JSON output
fn print_json_results(results: &RealProfileResults) -> Result<(), CliError> {
    let mut json = String::from("{\n");
    writeln!(json, "  \"model\": \"{}\",", results.model_path)
        .expect("write to String is infallible");
    writeln!(json, "  \"architecture\": \"{}\",", results.architecture)
        .expect("write to String is infallible");
    writeln!(json, "  \"num_layers\": {},", results.num_layers)
        .expect("write to String is infallible");
    writeln!(json, "  \"vocab_size\": {},", results.vocab_size)
        .expect("write to String is infallible");
    writeln!(json, "  \"hidden_dim\": {},", results.hidden_dim)
        .expect("write to String is infallible");
    writeln!(json, "  \"is_real_data\": {},", results.is_real_data)
        .expect("write to String is infallible");
    json.push_str("  \"timing\": {\n");
    writeln!(json, "    \"warmup_passes\": {},", results.warmup_passes)
        .expect("write to String is infallible");
    writeln!(json, "    \"measure_passes\": {},", results.measure_passes)
        .expect("write to String is infallible");
    writeln!(
        json,
        "    \"avg_inference_us\": {:.2},",
        results.total_inference_us
    )
    .expect("write to String is infallible");
    writeln!(
        json,
        "    \"throughput_tok_s\": {:.2}",
        results.throughput_tok_s
    )
    .expect("write to String is infallible");
    json.push_str("  },\n");

    json.push_str("  \"hotspots\": [\n");
    for (i, hotspot) in results.hotspots.iter().enumerate() {
        json.push_str("    {\n");
        writeln!(json, "      \"name\": \"{}\",", hotspot.name)
            .expect("write to String is infallible");
        writeln!(json, "      \"total_us\": {:.2},", hotspot.time_us)
            .expect("write to String is infallible");
        writeln!(json, "      \"avg_us\": {:.2},", hotspot.avg_us)
            .expect("write to String is infallible");
        writeln!(json, "      \"min_us\": {:.2},", hotspot.min_us)
            .expect("write to String is infallible");
        writeln!(json, "      \"max_us\": {:.2},", hotspot.max_us)
            .expect("write to String is infallible");
        writeln!(json, "      \"count\": {}", hotspot.count)
            .expect("write to String is infallible");
        if i < results.hotspots.len() - 1 {
            json.push_str("    },\n");
        } else {
            json.push_str("    }\n");
        }
    }
    json.push_str("  ],\n");

    json.push_str("  \"per_layer_us\": [");
    for (i, time) in results.per_layer_us.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        write!(json, "{:.2}", time).expect("write to String is infallible");
    }
    json.push_str("]\n");

    json.push_str("}\n");

    println!("{json}");
    Ok(())
}
