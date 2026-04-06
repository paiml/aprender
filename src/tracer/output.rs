//! Output formatting and summary printing for tracer results
//!
//! Handles text, JSON, CSV, HTML, and HPU output formats,
//! as well as decision trace summaries and analysis summaries.

use super::{AnalysisConfig, Tracers};

/// Print text statistics summary
///
/// Sprint 32: Now accepts optional `OtlpExporter` for compute block tracing
pub(super) fn print_text_stats(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    stats_extended: bool,
    anomaly_threshold: f32,
    #[cfg(feature = "otlp")] otlp_exporter: Option<&crate::otlp_exporter::OtlpExporter>,
    #[cfg(not(feature = "otlp"))] _otlp_exporter: Option<&()>,
) {
    if let Some(ref tracker) = stats_tracker {
        tracker.print_summary();
        if stats_extended {
            #[cfg(feature = "otlp")]
            tracker.print_extended_summary(anomaly_threshold, otlp_exporter);
            #[cfg(not(feature = "otlp"))]
            tracker.print_extended_summary(anomaly_threshold, _otlp_exporter);
        }
    }
}

/// Print JSON output
pub(super) fn print_json_output(mut output: crate::json_output::JsonOutput, exit_code: i32) {
    output.set_exit_code(exit_code);
    match output.to_json() {
        Ok(json) => println!("{json}"),
        Err(e) => eprintln!("Failed to serialize JSON: {e}"),
    }
}

/// Print CSV statistics output
pub(super) fn print_csv_stats(
    mut csv_stats: crate::csv_output::CsvStatsOutput,
    stats_tracker: &Option<crate::stats::StatsTracker>,
    timing_mode: bool,
    stats_extended: bool,
    anomaly_threshold: f32,
) {
    if let Some(ref tracker) = stats_tracker {
        for (syscall_name, stats) in tracker.stats_map() {
            let total_time_us = if timing_mode { Some(stats.total_time_us) } else { None };
            csv_stats.add_stat(crate::csv_output::CsvStat {
                syscall: syscall_name.clone(),
                calls: stats.count,
                errors: stats.errors,
                total_time_us,
            });
        }
        if stats_extended {
            // Note: CSV output doesn't get OTLP tracing - pass None
            #[cfg(feature = "otlp")]
            tracker.print_extended_summary(anomaly_threshold, None);
            #[cfg(not(feature = "otlp"))]
            tracker.print_extended_summary(anomaly_threshold, None);
        }
    }
    print!("{}", csv_stats.to_csv(timing_mode));
}

/// Print HPU analysis report
pub(super) fn print_hpu_analysis(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    hpu_cpu_only: bool,
) {
    if let Some(ref tracker) = stats_tracker {
        let mut hpu_data = std::collections::HashMap::new();
        for (syscall_name, stats) in tracker.stats_map() {
            let total_time_ns = stats.total_time_us * 1000;
            hpu_data.insert(syscall_name.clone(), (stats.count, total_time_ns));
        }
        let profiler = crate::hpu::HPUProfiler::new(hpu_cpu_only);
        let report = profiler.analyze(&hpu_data);
        print!("{}", report.format());
    }
}

/// Print optional profiling and tracing summaries
pub(super) fn print_optional_summaries(
    profiling_ctx: Option<crate::profiling::ProfilingContext>,
    function_profiler: Option<crate::function_profiler::FunctionProfiler>,
    anomaly_detector: Option<crate::anomaly::AnomalyDetector>,
) {
    if let Some(ctx) = profiling_ctx {
        ctx.print_summary();
    }
    if let Some(profiler) = function_profiler {
        profiler.print_summary();
    }
    if let Some(detector) = anomaly_detector {
        detector.print_summary();
    }
}
/// Sprint 26: Print decision trace summary
pub(super) fn print_decision_trace_summary(
    decision_tracer: Option<crate::decision_trace::DecisionTracer>,
) {
    if let Some(tracer) = decision_tracer {
        if tracer.count() == 0 {
            return;
        }

        // Sprint 27 Phase 3: Write to memory-mapped file (.ruchy/decisions.msgpack)
        let mmap_path = std::path::Path::new(".ruchy/decisions.msgpack");
        let manifest_path = std::path::Path::new(".ruchy/decision_manifest.json");

        // Write MessagePack file
        match tracer.write_to_msgpack(mmap_path) {
            Ok(()) => {
                println!("\n\u{2705} Decision traces written to: {}", mmap_path.display());
            }
            Err(e) => {
                eprintln!(
                    "\u{26a0}\u{fe0f}  Failed to write decision traces to {}: {}",
                    mmap_path.display(),
                    e
                );
            }
        }

        // Write manifest file
        match tracer.write_manifest(
            manifest_path,
            "2.0.0",
            None, // git_commit (could add via git2 crate)
            Some(env!("CARGO_PKG_VERSION")),
        ) {
            Ok(()) => {
                println!("\u{2705} Decision manifest written to: {}", manifest_path.display());
            }
            Err(e) => {
                eprintln!(
                    "\u{26a0}\u{fe0f}  Failed to write decision manifest to {}: {}",
                    manifest_path.display(),
                    e
                );
            }
        }

        // Also print summary to stdout for convenience
        println!("\n=== Transpiler Decision Traces ===\n");

        for trace in tracer.traces() {
            // Format: category::name with input and result
            print!("[{}::{}] ", trace.category, trace.name);

            // Print input (compact JSON)
            print!("input={}", trace.input);

            // Print result if available
            if let Some(ref result) = trace.result {
                print!(" result={result}");
            }

            // Print decision_id if available (Sprint 27)
            if let Some(decision_id) = trace.decision_id {
                print!(" id=0x{decision_id:X}");
            }

            println!();
        }

        println!("\nTotal decision traces: {}", tracer.count());
        println!("Decision manifest: {}", manifest_path.display());
        println!("Binary traces: {}", mmap_path.display());
    }
}

/// Print analysis summaries (HPU, ML, Isolation Forest, Autoencoder)
pub(super) fn print_analysis_summaries(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    analysis: &AnalysisConfig,
) {
    if analysis.hpu_analysis {
        print_hpu_analysis(stats_tracker, analysis.hpu_cpu_only);
    }
    if analysis.ml_anomaly {
        super::ml_analysis::print_ml_analysis(
            stats_tracker,
            analysis.ml_clusters,
            analysis.ml_compare,
            analysis.anomaly_threshold,
        );
    }
    if analysis.ml_outliers {
        super::ml_analysis::print_isolation_forest_analysis(
            stats_tracker,
            analysis.ml_outlier_trees,
            analysis.ml_outlier_threshold,
            analysis.explain,
        );
    }
    if analysis.dl_anomaly {
        super::ml_analysis::print_autoencoder_analysis(
            stats_tracker,
            analysis.dl_hidden_size,
            analysis.dl_epochs,
            analysis.dl_threshold,
            analysis.explain,
        );
    }
}

/// Handle JSON output with ML analysis additions
pub(super) fn handle_json_output(
    mut output: crate::json_output::JsonOutput,
    stats_tracker: &Option<crate::stats::StatsTracker>,
    analysis: &AnalysisConfig,
    exit_code: i32,
) {
    if analysis.ml_anomaly {
        if let Some(report) =
            super::ml_analysis::generate_ml_analysis_for_json(stats_tracker, analysis.ml_clusters)
        {
            output.set_ml_analysis(report);
        }
    }
    if analysis.ml_outliers {
        if let Some(report) = super::ml_analysis::generate_isolation_forest_analysis_for_json(
            stats_tracker,
            analysis.ml_outlier_trees,
            analysis.ml_outlier_threshold,
            analysis.explain,
        ) {
            output.set_isolation_forest_analysis(report, analysis.explain);
        }
    }
    if analysis.dl_anomaly {
        if let Some(report) = super::ml_analysis::generate_autoencoder_analysis_for_json(
            stats_tracker,
            analysis.dl_hidden_size,
            analysis.dl_epochs,
            f64::from(analysis.dl_threshold),
            analysis.explain,
        ) {
            output.set_autoencoder_analysis(report, analysis.dl_threshold, analysis.explain);
        }
    }
    print_json_output(output, exit_code);
}

/// Print syscall result
pub(super) fn print_syscall_result(result: i64, timing_mode: bool, duration_us: u64) {
    if timing_mode && duration_us > 0 {
        println!("{} <{:.6}>", result, duration_us as f64 / 1_000_000.0);
    } else {
        println!("{result}");
    }
}

/// Check if syscall result should be printed to stdout
pub(super) fn should_print_result(
    syscall_entry: &Option<super::SyscallEntry>,
    in_stats_mode: bool,
    in_json_mode: bool,
    in_csv_mode: bool,
    in_html_mode: bool,
) -> bool {
    syscall_entry.is_some() && !in_stats_mode && !in_json_mode && !in_csv_mode && !in_html_mode
}

/// Print all summaries at end of tracing
pub(super) fn print_summaries(
    tracers: Tracers,
    timing_mode: bool,
    exit_code: i32,
    analysis: &AnalysisConfig,
) {
    let Tracers {
        stats_tracker,
        json_output,
        csv_output,
        csv_stats_output,
        html_output,
        profiling_ctx,
        function_profiler,
        anomaly_detector,
        decision_tracer,
        #[cfg(feature = "otlp")]
        mut otlp_exporter,
        visualizer_sink: _,
    } = tracers;

    // Sprint 31: Export decision traces to OTLP (before ending root span)
    #[cfg(feature = "otlp")]
    if let (Some(ref mut exporter), Some(ref tracer)) = (&mut otlp_exporter, &decision_tracer) {
        for trace in tracer.traces() {
            exporter.record_decision(
                &trace.category,
                &trace.name,
                trace.result.as_ref().and_then(|v| v.as_str()),
                trace.timestamp_us,
            );
        }
    }

    // Print statistics summary if in statistics mode (text format)
    if stats_tracker.is_some() && csv_stats_output.is_none() {
        #[cfg(feature = "otlp")]
        print_text_stats(
            &stats_tracker,
            analysis.stats_extended,
            analysis.anomaly_threshold,
            otlp_exporter.as_ref(),
        );
        #[cfg(not(feature = "otlp"))]
        print_text_stats(&stats_tracker, analysis.stats_extended, analysis.anomaly_threshold, None);
    }

    // Sprint 30: End root span and shutdown OTLP exporter
    #[cfg(feature = "otlp")]
    if let Some(ref mut exporter) = otlp_exporter {
        exporter.end_root_span(exit_code);
        exporter.shutdown();
    }

    // Handle output formats
    if let Some(output) = json_output {
        handle_json_output(output, &stats_tracker, analysis, exit_code);
    }
    if let Some(output) = csv_output {
        print!("{}", output.to_csv());
    }
    if let Some(csv_stats) = csv_stats_output {
        print_csv_stats(
            csv_stats,
            &stats_tracker,
            timing_mode,
            analysis.stats_extended,
            analysis.anomaly_threshold,
        );
    }
    if let Some(output) = html_output {
        print!("{}", output.to_html(stats_tracker.as_ref()));
    }

    // Print profiling and tracing summaries
    print_optional_summaries(profiling_ctx, function_profiler, anomaly_detector);
    print_analysis_summaries(&stats_tracker, analysis);
    print_decision_trace_summary(decision_tracer);
}

/// Print syscall entry with optional source location
#[allow(clippy::too_many_arguments)]
pub(super) fn print_syscall_entry(
    child: nix::unistd::Pid,
    name: &str,
    syscall_num: i64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    source_info: &Option<crate::dwarf::SourceLocation>,
    transpiler_map: Option<&crate::transpiler_map::TranspilerMap>,
) {
    // Print source location if available
    if let Some(src) = source_info {
        // Try to map to transpiler source first
        if let Some(transpiled_source) =
            super::syscall_handling::map_to_transpiler_source(src, transpiler_map)
        {
            // Show both Rust and original source
            print!("{transpiled_source} ");
        } else {
            // Show just Rust source from DWARF
            print!("{}:{} ", src.file, src.line);
            if let Some(func) = &src.function {
                print!("{func} ");
            }
        }
    }

    // Print syscall with arguments
    match name {
        "openat" => {
            let filename = super::syscall_handling::read_string(child, arg2 as usize)
                .unwrap_or_else(|_| format!("{arg2:#x}"));
            print!("{name}({arg1:#x}, \"{filename}\", {arg3:#x}) = ");
        }
        "unknown" => {
            print!("syscall_{syscall_num}({arg1:#x}, {arg2:#x}, {arg3:#x}) = ");
        }
        _ => {
            print!("{name}({arg1:#x}, {arg2:#x}, {arg3:#x}) = ");
        }
    }
    std::io::Write::flush(&mut std::io::stdout()).ok();
}
