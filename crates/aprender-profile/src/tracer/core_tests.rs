use super::*;

#[test]
fn test_trace_command_requires_nonempty_array() {
    let empty: Vec<String> = vec![];
    let config = TracerConfig {
        enable_source: false,
        filter: crate::filter::SyscallFilter::all(),
        statistics_mode: false,
        timing_mode: false,
        output_format: crate::cli::OutputFormat::Text,
        follow_forks: false,
        profile_self: false,
        function_time: false,
        stats_extended: false,                    // Sprint 19
        anomaly_threshold: 3.0,                   // Sprint 19
        anomaly_realtime: false,                  // Sprint 20
        anomaly_window_size: 100,                 // Sprint 20
        hpu_analysis: false,                      // Sprint 21
        hpu_cpu_only: false,                      // Sprint 21
        ml_anomaly: false,                        // Sprint 23
        ml_clusters: 3,                           // Sprint 23
        ml_compare: false,                        // Sprint 23
        ml_outliers: false,                       // Sprint 22
        ml_outlier_threshold: 0.1,                // Sprint 22
        ml_outlier_trees: 100,                    // Sprint 22
        explain: false,                           // Sprint 22/23
        dl_anomaly: false,                        // Sprint 23
        dl_threshold: 2.0,                        // Sprint 23
        dl_hidden_size: 3,                        // Sprint 23
        dl_epochs: 100,                           // Sprint 23
        trace_transpiler_decisions: false,        // Sprint 26
        transpiler_map: None,                     // Sprint 24-28
        otlp_endpoint: None,                      // Sprint 30
        otlp_service_name: "renacer".to_string(), // Sprint 30
        trace_parent: None,                       // Sprint 33
        chaos_config: None,                       // Sprint 47
        visualizer_sink: None,                    // Sprint 52-55
    };
    let result = trace_command(&empty, config);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("empty"));
}

#[test]
#[ignore = "requires ptrace permissions - run with --ignored"]
fn test_trace_command_basic() {
    // GREEN: trace_command now works - verify it succeeds for basic commands
    let cmd = vec!["echo".to_string(), "test".to_string()];
    let config = TracerConfig {
        enable_source: false,
        filter: crate::filter::SyscallFilter::all(),
        statistics_mode: false,
        timing_mode: false,
        output_format: crate::cli::OutputFormat::Text,
        follow_forks: false,
        profile_self: false,
        function_time: false,
        stats_extended: false,                    // Sprint 19
        anomaly_threshold: 3.0,                   // Sprint 19
        anomaly_realtime: false,                  // Sprint 20
        anomaly_window_size: 100,                 // Sprint 20
        hpu_analysis: false,                      // Sprint 21
        hpu_cpu_only: false,                      // Sprint 21
        ml_anomaly: false,                        // Sprint 23
        ml_clusters: 3,                           // Sprint 23
        ml_compare: false,                        // Sprint 23
        ml_outliers: false,                       // Sprint 22
        ml_outlier_threshold: 0.1,                // Sprint 22
        ml_outlier_trees: 100,                    // Sprint 22
        explain: false,                           // Sprint 22/23
        dl_anomaly: false,                        // Sprint 23
        dl_threshold: 2.0,                        // Sprint 23
        dl_hidden_size: 3,                        // Sprint 23
        dl_epochs: 100,                           // Sprint 23
        trace_transpiler_decisions: false,        // Sprint 26
        transpiler_map: None,                     // Sprint 24-28
        otlp_endpoint: None,                      // Sprint 30
        otlp_service_name: "renacer".to_string(), // Sprint 30
        trace_parent: None,                       // Sprint 33
        chaos_config: None,                       // Sprint 47
        visualizer_sink: None,                    // Sprint 52-55
    };
    let result = trace_command(&cmd, config);
    assert!(result.is_ok(), "trace_command failed: {:?}", result);
}

#[test]
fn test_syscall_entry_creation() {
    let entry = SyscallEntry {
        name: "open".to_string(),
        args: vec!["arg1".to_string(), "arg2".to_string()],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: Some(1),
        raw_arg2: Some(2),
        _raw_arg3: Some(3),
    };
    assert_eq!(entry.name, "open");
    assert_eq!(entry.args.len(), 2);
    assert!(entry.source.is_none());
    assert!(entry.function_name.is_none());
}

#[test]
fn test_syscall_entry_with_source() {
    let source = crate::json_output::JsonSourceLocation {
        file: "test.rs".to_string(),
        line: 42,
        function: Some("main".to_string()),
    };
    let entry = SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: Some(source),
        function_name: Some("main".to_string()),
        caller_name: None,
        raw_arg1: Some(0),
        raw_arg2: Some(0),
        _raw_arg3: Some(0),
    };
    assert_eq!(entry.name, "read");
    assert!(entry.source.is_some());
    let src = entry.source.expect("test");
    assert_eq!(src.file, "test.rs");
    assert_eq!(src.line, 42);
    assert_eq!(src.function, Some("main".to_string()));
    assert_eq!(entry.function_name, Some("main".to_string()));
}

#[test]
fn test_attach_to_pid_invalid_pid() {
    // Test attaching to a non-existent PID (should fail)
    let config = TracerConfig::default();
    let result = attach_to_pid(999999, config);
    assert!(result.is_err());
    // Error message should mention attach failure
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("attach") || err_msg.contains("Failed"), "Error: {}", err_msg);
}

// TracerConfig default tests
#[test]
fn test_tracer_config_default() {
    let config = TracerConfig::default();
    assert!(!config.enable_source);
    assert!(!config.statistics_mode);
    assert!(!config.timing_mode);
    assert!(!config.follow_forks);
    assert!(!config.profile_self);
    assert!(!config.function_time);
    assert!(!config.stats_extended);
    assert!((config.anomaly_threshold - 2.0).abs() < f32::EPSILON);
    assert!(!config.anomaly_realtime);
    assert_eq!(config.anomaly_window_size, 100);
    assert!(!config.hpu_analysis);
    assert!(!config.hpu_cpu_only);
    assert!(!config.ml_anomaly);
    assert_eq!(config.ml_clusters, 5);
    assert!(!config.ml_compare);
    assert!(!config.ml_outliers);
    assert!((config.ml_outlier_threshold - 0.1).abs() < f32::EPSILON);
    assert_eq!(config.ml_outlier_trees, 100);
    assert!(!config.explain);
    assert!(!config.dl_anomaly);
    assert!((config.dl_threshold - 2.0).abs() < f32::EPSILON);
    assert_eq!(config.dl_hidden_size, 8);
    assert_eq!(config.dl_epochs, 50);
    assert!(!config.trace_transpiler_decisions);
    assert!(config.transpiler_map.is_none());
    assert!(config.otlp_endpoint.is_none());
    assert_eq!(config.otlp_service_name, "renacer");
    assert!(config.trace_parent.is_none());
    assert!(config.chaos_config.is_none());
}

// ProcessState tests
#[test]
fn test_process_state_new() {
    let state = ProcessState::new();
    assert!(!state.in_syscall);
    assert!(state.current_syscall_entry.is_none());
    assert!(state.syscall_entry_time.is_none());
    assert!(state.dwarf_ctx.is_none());
    assert!(!state.dwarf_loaded);
}

// initialize_profiling_tracers tests
#[test]
fn test_initialize_profiling_tracers_all_disabled() {
    let config = TracerConfig::default();
    let (profiling_ctx, function_profiler, anomaly_detector) =
        initialize_profiling_tracers(&config);
    assert!(profiling_ctx.is_none());
    assert!(function_profiler.is_none());
    assert!(anomaly_detector.is_none());
}

#[test]
fn test_initialize_profiling_tracers_profile_self() {
    let mut config = TracerConfig::default();
    config.profile_self = true;
    let (profiling_ctx, function_profiler, anomaly_detector) =
        initialize_profiling_tracers(&config);
    assert!(profiling_ctx.is_some());
    assert!(function_profiler.is_none());
    assert!(anomaly_detector.is_none());
}

#[test]
fn test_initialize_profiling_tracers_function_time() {
    let mut config = TracerConfig::default();
    config.function_time = true;
    let (profiling_ctx, function_profiler, anomaly_detector) =
        initialize_profiling_tracers(&config);
    assert!(profiling_ctx.is_none());
    assert!(function_profiler.is_some());
    assert!(anomaly_detector.is_none());
}

#[test]
fn test_initialize_profiling_tracers_anomaly_realtime() {
    let mut config = TracerConfig::default();
    config.anomaly_realtime = true;
    let (profiling_ctx, function_profiler, anomaly_detector) =
        initialize_profiling_tracers(&config);
    assert!(profiling_ctx.is_none());
    assert!(function_profiler.is_none());
    assert!(anomaly_detector.is_some());
}

#[test]
fn test_initialize_profiling_tracers_all_enabled() {
    let mut config = TracerConfig::default();
    config.profile_self = true;
    config.function_time = true;
    config.anomaly_realtime = true;
    let (profiling_ctx, function_profiler, anomaly_detector) =
        initialize_profiling_tracers(&config);
    assert!(profiling_ctx.is_some());
    assert!(function_profiler.is_some());
    assert!(anomaly_detector.is_some());
}

// initialize_output_tracers tests
#[test]
fn test_initialize_output_tracers_text_format() {
    let config = TracerConfig::default();
    let (json, csv, csv_stats, html) = initialize_output_tracers(&config);
    assert!(json.is_none());
    assert!(csv.is_none());
    assert!(csv_stats.is_none());
    assert!(html.is_none());
}

#[test]
fn test_initialize_output_tracers_json_format() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Json;
    let (json, csv, csv_stats, html) = initialize_output_tracers(&config);
    assert!(json.is_some());
    assert!(csv.is_none());
    assert!(csv_stats.is_none());
    assert!(html.is_none());
}

#[test]
fn test_initialize_output_tracers_csv_format() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Csv;
    let (json, csv, csv_stats, html) = initialize_output_tracers(&config);
    assert!(json.is_none());
    assert!(csv.is_some());
    assert!(csv_stats.is_none());
    assert!(html.is_none());
}

#[test]
fn test_initialize_output_tracers_csv_stats_format() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Csv;
    config.statistics_mode = true;
    let (json, csv, csv_stats, html) = initialize_output_tracers(&config);
    assert!(json.is_none());
    assert!(csv.is_none());
    assert!(csv_stats.is_some());
    assert!(html.is_none());
}

#[test]
fn test_initialize_output_tracers_html_format() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Html;
    let (json, csv, csv_stats, html) = initialize_output_tracers(&config);
    assert!(json.is_none());
    assert!(csv.is_none());
    assert!(csv_stats.is_none());
    assert!(html.is_some());
}

// initialize_tracers tests
#[test]
fn test_initialize_tracers_default() {
    let config = TracerConfig::default();
    let tracers = initialize_tracers(&config);
    assert!(tracers.profiling_ctx.is_none());
    assert!(tracers.function_profiler.is_none());
    assert!(tracers.stats_tracker.is_none());
    assert!(tracers.json_output.is_none());
    assert!(tracers.csv_output.is_none());
    assert!(tracers.csv_stats_output.is_none());
    assert!(tracers.html_output.is_none());
    assert!(tracers.anomaly_detector.is_none());
    assert!(tracers.decision_tracer.is_none());
}

#[test]
fn test_initialize_tracers_with_statistics() {
    let mut config = TracerConfig::default();
    config.statistics_mode = true;
    let tracers = initialize_tracers(&config);
    assert!(tracers.stats_tracker.is_some());
}

#[test]
fn test_initialize_tracers_with_ml_anomaly() {
    let mut config = TracerConfig::default();
    config.ml_anomaly = true;
    let tracers = initialize_tracers(&config);
    assert!(tracers.stats_tracker.is_some());
}

#[test]
fn test_initialize_tracers_with_ml_outliers() {
    let mut config = TracerConfig::default();
    config.ml_outliers = true;
    let tracers = initialize_tracers(&config);
    assert!(tracers.stats_tracker.is_some());
}

#[test]
fn test_initialize_tracers_with_dl_anomaly() {
    let mut config = TracerConfig::default();
    config.dl_anomaly = true;
    let tracers = initialize_tracers(&config);
    assert!(tracers.stats_tracker.is_some());
}

#[test]
fn test_initialize_tracers_with_decision_tracer() {
    let mut config = TracerConfig::default();
    config.trace_transpiler_decisions = true;
    let tracers = initialize_tracers(&config);
    assert!(tracers.decision_tracer.is_some());
}

#[test]
fn test_initialize_tracers_json_output() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Json;
    let tracers = initialize_tracers(&config);
    assert!(tracers.json_output.is_some());
}

#[test]
fn test_initialize_tracers_html_output() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Html;
    let tracers = initialize_tracers(&config);
    assert!(tracers.html_output.is_some());
}

// SyscallEntry tests
#[test]
fn test_syscall_entry_with_raw_args() {
    let entry = SyscallEntry {
        name: "write".to_string(),
        args: vec!["1".to_string(), "buf".to_string(), "10".to_string()],
        source: None,
        function_name: None,
        caller_name: Some("caller_fn".to_string()),
        raw_arg1: Some(1),
        raw_arg2: Some(0x7fff_0000_0000),
        _raw_arg3: Some(10),
    };
    assert_eq!(entry.raw_arg1, Some(1));
    assert_eq!(entry.raw_arg2, Some(0x7fff_0000_0000));
    assert_eq!(entry._raw_arg3, Some(10));
    assert_eq!(entry.caller_name, Some("caller_fn".to_string()));
}

#[test]
fn test_syscall_entry_empty_args() {
    let entry = SyscallEntry {
        name: "getpid".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    };
    assert!(entry.args.is_empty());
    assert!(entry.raw_arg1.is_none());
}

// Test TraceChild helper functions implicitly
#[test]
fn test_tracer_config_with_custom_values() {
    let config = TracerConfig {
        enable_source: true,
        filter: crate::filter::SyscallFilter::all(),
        statistics_mode: true,
        timing_mode: true,
        output_format: crate::cli::OutputFormat::Json,
        follow_forks: true,
        profile_self: true,
        function_time: true,
        stats_extended: true,
        anomaly_threshold: 5.0,
        anomaly_realtime: true,
        anomaly_window_size: 200,
        hpu_analysis: true,
        hpu_cpu_only: true,
        ml_anomaly: true,
        ml_clusters: 10,
        ml_compare: true,
        ml_outliers: true,
        ml_outlier_threshold: 0.2,
        ml_outlier_trees: 200,
        explain: true,
        dl_anomaly: true,
        dl_threshold: 3.0,
        dl_hidden_size: 16,
        dl_epochs: 100,
        trace_transpiler_decisions: true,
        transpiler_map: None,
        otlp_endpoint: Some("http://localhost:4317".to_string()),
        otlp_service_name: "test-service".to_string(),
        trace_parent: Some("00-trace-parent-01".to_string()),
        chaos_config: None,
        visualizer_sink: None, // Sprint 52-55
    };

    assert!(config.enable_source);
    assert!(config.statistics_mode);
    assert!(config.timing_mode);
    assert!(config.follow_forks);
    assert!(config.profile_self);
    assert!(config.function_time);
    assert!(config.stats_extended);
    assert!((config.anomaly_threshold - 5.0).abs() < f32::EPSILON);
    assert!(config.anomaly_realtime);
    assert_eq!(config.anomaly_window_size, 200);
    assert!(config.hpu_analysis);
    assert!(config.hpu_cpu_only);
    assert!(config.ml_anomaly);
    assert_eq!(config.ml_clusters, 10);
    assert!(config.ml_compare);
    assert!(config.ml_outliers);
    assert!((config.ml_outlier_threshold - 0.2).abs() < f32::EPSILON);
    assert_eq!(config.ml_outlier_trees, 200);
    assert!(config.explain);
    assert!(config.dl_anomaly);
    assert!((config.dl_threshold - 3.0).abs() < f32::EPSILON);
    assert_eq!(config.dl_hidden_size, 16);
    assert_eq!(config.dl_epochs, 100);
    assert!(config.trace_transpiler_decisions);
    assert!(config.otlp_endpoint.is_some());
    assert_eq!(config.otlp_service_name, "test-service");
    assert!(config.trace_parent.is_some());
}

#[test]
fn test_initialize_tracers_full_config() {
    let mut config = TracerConfig::default();
    config.profile_self = true;
    config.function_time = true;
    config.anomaly_realtime = true;
    config.statistics_mode = true;
    config.ml_anomaly = true;
    config.ml_outliers = true;
    config.dl_anomaly = true;
    config.trace_transpiler_decisions = true;
    config.output_format = crate::cli::OutputFormat::Json;

    let tracers = initialize_tracers(&config);
    assert!(tracers.profiling_ctx.is_some());
    assert!(tracers.function_profiler.is_some());
    assert!(tracers.anomaly_detector.is_some());
    assert!(tracers.stats_tracker.is_some());
    assert!(tracers.json_output.is_some());
    assert!(tracers.decision_tracer.is_some());
}

// Tests for analysis and output functions

#[test]
fn test_print_text_stats_none() {
    let stats_tracker: Option<crate::stats::StatsTracker> = None;
    #[cfg(feature = "otlp")]
    output::print_text_stats(&stats_tracker, false, 2.0, None);
    #[cfg(not(feature = "otlp"))]
    output::print_text_stats(&stats_tracker, false, 2.0, None);
    // Should not panic with None
}

#[test]
fn test_print_text_stats_with_tracker() {
    let stats_tracker = Some(crate::stats::StatsTracker::new());
    #[cfg(feature = "otlp")]
    output::print_text_stats(&stats_tracker, false, 2.0, None);
    #[cfg(not(feature = "otlp"))]
    output::print_text_stats(&stats_tracker, false, 2.0, None);
}

#[test]
fn test_print_text_stats_extended() {
    let stats_tracker = Some(crate::stats::StatsTracker::new());
    #[cfg(feature = "otlp")]
    output::print_text_stats(&stats_tracker, true, 3.0, None);
    #[cfg(not(feature = "otlp"))]
    output::print_text_stats(&stats_tracker, true, 3.0, None);
}

#[test]
fn test_print_json_output() {
    let json_out = crate::json_output::JsonOutput::new();
    // Should not panic
    output::print_json_output(json_out, 0);
}

#[test]
fn test_print_json_output_nonzero_exit() {
    let json_out = crate::json_output::JsonOutput::new();
    output::print_json_output(json_out, 1);
}

#[test]
fn test_generate_ml_analysis_none() {
    let result = ml_analysis::generate_ml_analysis_for_json(&None, 5);
    assert!(result.is_none());
}

#[test]
fn test_generate_ml_analysis_empty_tracker() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let result = ml_analysis::generate_ml_analysis_for_json(&tracker, 3);
    assert!(result.is_some());
}

#[test]
fn test_generate_isolation_forest_none() {
    let result = ml_analysis::generate_isolation_forest_analysis_for_json(&None, 100, 0.1, false);
    assert!(result.is_none());
}

#[test]
fn test_generate_isolation_forest_empty_tracker() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let result = ml_analysis::generate_isolation_forest_analysis_for_json(&tracker, 50, 0.1, false);
    assert!(result.is_some());
}

#[test]
fn test_generate_isolation_forest_with_explain() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let result = ml_analysis::generate_isolation_forest_analysis_for_json(&tracker, 50, 0.1, true);
    assert!(result.is_some());
}

#[test]
fn test_generate_autoencoder_none() {
    let result = ml_analysis::generate_autoencoder_analysis_for_json(&None, 8, 50, 2.0, false);
    assert!(result.is_none());
}

#[test]
fn test_generate_autoencoder_empty_tracker() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let result = ml_analysis::generate_autoencoder_analysis_for_json(&tracker, 8, 50, 2.0, false);
    assert!(result.is_some());
}

#[test]
fn test_generate_autoencoder_with_explain() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let result = ml_analysis::generate_autoencoder_analysis_for_json(&tracker, 8, 50, 2.0, true);
    assert!(result.is_some());
}

#[test]
fn test_print_csv_stats_none() {
    let csv_stats = crate::csv_output::CsvStatsOutput::new();
    output::print_csv_stats(csv_stats, &None, false, false, 2.0);
}

#[test]
fn test_print_csv_stats_with_tracker() {
    let csv_stats = crate::csv_output::CsvStatsOutput::new();
    let tracker = Some(crate::stats::StatsTracker::new());
    output::print_csv_stats(csv_stats, &tracker, false, false, 2.0);
}

#[test]
fn test_print_csv_stats_with_timing() {
    let csv_stats = crate::csv_output::CsvStatsOutput::new();
    let tracker = Some(crate::stats::StatsTracker::new());
    output::print_csv_stats(csv_stats, &tracker, true, false, 2.0);
}

#[test]
fn test_print_csv_stats_extended() {
    let csv_stats = crate::csv_output::CsvStatsOutput::new();
    let tracker = Some(crate::stats::StatsTracker::new());
    output::print_csv_stats(csv_stats, &tracker, true, true, 3.0);
}

#[test]
fn test_print_hpu_analysis_none() {
    output::print_hpu_analysis(&None, false);
}

#[test]
fn test_print_hpu_analysis_cpu_only() {
    let tracker = Some(crate::stats::StatsTracker::new());
    output::print_hpu_analysis(&tracker, true);
}

#[test]
fn test_print_hpu_analysis_with_tracker() {
    let tracker = Some(crate::stats::StatsTracker::new());
    output::print_hpu_analysis(&tracker, false);
}

#[test]
fn test_print_ml_analysis_none() {
    ml_analysis::print_ml_analysis(&None, 5, false, 2.0);
}

#[test]
fn test_print_ml_analysis_with_tracker() {
    let tracker = Some(crate::stats::StatsTracker::new());
    ml_analysis::print_ml_analysis(&tracker, 5, false, 2.0);
}

#[test]
fn test_print_ml_analysis_compare() {
    let tracker = Some(crate::stats::StatsTracker::new());
    ml_analysis::print_ml_analysis(&tracker, 5, true, 2.0);
}

#[test]
fn test_print_isolation_forest_none() {
    ml_analysis::print_isolation_forest_analysis(&None, 100, 0.1, false);
}

#[test]
fn test_print_isolation_forest_with_tracker() {
    let tracker = Some(crate::stats::StatsTracker::new());
    ml_analysis::print_isolation_forest_analysis(&tracker, 100, 0.1, false);
}

#[test]
fn test_print_isolation_forest_with_explain() {
    let tracker = Some(crate::stats::StatsTracker::new());
    ml_analysis::print_isolation_forest_analysis(&tracker, 100, 0.1, true);
}

#[test]
fn test_initialize_output_tracers_csv_with_timing() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Csv;
    config.timing_mode = true;
    let (json, csv, csv_stats, html) = initialize_output_tracers(&config);
    assert!(json.is_none());
    assert!(csv.is_some());
    assert!(csv_stats.is_none());
    assert!(html.is_none());
}

#[test]
fn test_initialize_output_tracers_csv_with_source() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Csv;
    config.enable_source = true;
    let (_, csv, _, _) = initialize_output_tracers(&config);
    assert!(csv.is_some());
}

#[test]
fn test_initialize_output_tracers_html_with_timing_and_source() {
    let mut config = TracerConfig::default();
    config.output_format = crate::cli::OutputFormat::Html;
    config.timing_mode = true;
    config.enable_source = true;
    let (_, _, _, html) = initialize_output_tracers(&config);
    assert!(html.is_some());
}

// More analysis function tests

#[test]
fn test_print_autoencoder_none() {
    ml_analysis::print_autoencoder_analysis(&None, 8, 50, 2.0, false);
}

#[test]
fn test_print_autoencoder_with_tracker() {
    let tracker = Some(crate::stats::StatsTracker::new());
    ml_analysis::print_autoencoder_analysis(&tracker, 8, 50, 2.0, false);
}

#[test]
fn test_print_autoencoder_with_explain() {
    let tracker = Some(crate::stats::StatsTracker::new());
    ml_analysis::print_autoencoder_analysis(&tracker, 8, 50, 2.0, true);
}

#[test]
fn test_print_optional_summaries_all_none() {
    output::print_optional_summaries(None, None, None);
}

#[test]
fn test_print_optional_summaries_with_profiling() {
    let ctx = Some(crate::profiling::ProfilingContext::new());
    output::print_optional_summaries(ctx, None, None);
}

#[test]
fn test_print_optional_summaries_with_function_profiler() {
    let profiler = Some(crate::function_profiler::FunctionProfiler::new());
    output::print_optional_summaries(None, profiler, None);
}

#[test]
fn test_print_optional_summaries_with_anomaly_detector() {
    let detector = Some(crate::anomaly::AnomalyDetector::new(100, 2.0));
    output::print_optional_summaries(None, None, detector);
}

#[test]
fn test_print_optional_summaries_all_some() {
    let ctx = Some(crate::profiling::ProfilingContext::new());
    let profiler = Some(crate::function_profiler::FunctionProfiler::new());
    let detector = Some(crate::anomaly::AnomalyDetector::new(100, 2.0));
    output::print_optional_summaries(ctx, profiler, detector);
}

#[test]
fn test_print_decision_trace_summary_none() {
    output::print_decision_trace_summary(None);
}

#[test]
fn test_print_decision_trace_summary_empty() {
    let tracer = Some(crate::decision_trace::DecisionTracer::new());
    output::print_decision_trace_summary(tracer);
}

#[test]
fn test_analysis_config_fields() {
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 3.0,
        hpu_analysis: true,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 10,
        ml_compare: true,
        ml_outliers: true,
        ml_outlier_threshold: 0.15,
        ml_outlier_trees: 200,
        dl_anomaly: true,
        dl_threshold: 2.5,
        dl_hidden_size: 16,
        dl_epochs: 100,
        explain: true,
    };
    assert!(analysis.stats_extended);
    assert!((analysis.anomaly_threshold - 3.0).abs() < f32::EPSILON);
    assert!(analysis.hpu_analysis);
    assert!(!analysis.hpu_cpu_only);
    assert!(analysis.ml_anomaly);
    assert_eq!(analysis.ml_clusters, 10);
}

#[test]
fn test_print_analysis_summaries_none() {
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_analysis_summaries(&None, &analysis);
}

#[test]
fn test_print_analysis_summaries_hpu() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: true,
        hpu_cpu_only: true,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_analysis_summaries(&tracker, &analysis);
}

#[test]
fn test_print_analysis_summaries_ml() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_analysis_summaries(&tracker, &analysis);
}

#[test]
fn test_print_analysis_summaries_outliers() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_analysis_summaries(&tracker, &analysis);
}

#[test]
fn test_print_analysis_summaries_dl() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_analysis_summaries(&tracker, &analysis);
}

#[test]
fn test_print_analysis_summaries_all() {
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 2.0,
        hpu_analysis: true,
        hpu_cpu_only: true,
        ml_anomaly: true,
        ml_clusters: 5,
        ml_compare: true,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: true,
    };
    output::print_analysis_summaries(&tracker, &analysis);
}

#[test]
fn test_handle_json_output_basic() {
    let json_out = crate::json_output::JsonOutput::new();
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::handle_json_output(json_out, &None, &analysis, 0);
}

#[test]
fn test_handle_json_output_with_ml() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 0);
}

#[test]
fn test_handle_json_output_with_outliers() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: true,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 0);
}

#[test]
fn test_handle_json_output_with_dl() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 0);
}

#[test]
fn test_handle_json_output_all_analysis() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(crate::stats::StatsTracker::new());
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 2.0,
        hpu_analysis: true,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 5,
        ml_compare: true,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: true,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 1);
}

// should_print_result tests
#[test]
fn test_should_print_result_none_entry() {
    assert!(!output::should_print_result(&None, false, false, false, false));
}

#[test]
fn test_should_print_result_with_entry_no_modes() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    assert!(output::should_print_result(&entry, false, false, false, false));
}

#[test]
fn test_should_print_result_stats_mode() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    assert!(!output::should_print_result(&entry, true, false, false, false));
}

#[test]
fn test_should_print_result_json_mode() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    assert!(!output::should_print_result(&entry, false, true, false, false));
}

#[test]
fn test_should_print_result_csv_mode() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    assert!(!output::should_print_result(&entry, false, false, true, false));
}

#[test]
fn test_should_print_result_html_mode() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    assert!(!output::should_print_result(&entry, false, false, false, true));
}

#[test]
fn test_should_print_result_all_modes() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    assert!(!output::should_print_result(&entry, true, true, true, true));
}

// print_syscall_result tests
#[test]
fn test_print_syscall_result_success() {
    output::print_syscall_result(0, false, 0);
}

#[test]
fn test_print_syscall_result_error() {
    output::print_syscall_result(-1, false, 0);
}

#[test]
fn test_print_syscall_result_with_timing() {
    output::print_syscall_result(42, true, 1234);
}

#[test]
fn test_print_syscall_result_negative_with_timing() {
    output::print_syscall_result(-22, true, 5678);
}

// record_stats_for_syscall tests
#[test]
fn test_record_stats_for_syscall_none_entry() {
    let mut tracker = Some(crate::stats::StatsTracker::new());
    syscall_handling::record_stats_for_syscall_test(&None, tracker.as_mut(), 0, 100);
}

#[test]
fn test_record_stats_for_syscall_none_tracker() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    syscall_handling::record_stats_for_syscall_test(&entry, None, 0, 100);
}

#[test]
fn test_record_stats_for_syscall_with_data() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    let mut tracker = Some(crate::stats::StatsTracker::new());
    syscall_handling::record_stats_for_syscall_test(&entry, tracker.as_mut(), 100, 500);
}

// record_function_profiling tests
#[test]
fn test_record_function_profiling_none_entry() {
    let mut profiler = Some(crate::function_profiler::FunctionProfiler::new());
    syscall_handling::record_function_profiling_test(&None, profiler.as_mut(), 100);
}

#[test]
fn test_record_function_profiling_none_profiler() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: Some("test_fn".to_string()),
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    syscall_handling::record_function_profiling_test(&entry, None, 100);
}

#[test]
fn test_record_function_profiling_with_data() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: Some("test_fn".to_string()),
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    let mut profiler = Some(crate::function_profiler::FunctionProfiler::new());
    syscall_handling::record_function_profiling_test(&entry, profiler.as_mut(), 100);
}

#[test]
fn test_record_function_profiling_no_function_name() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    let mut profiler = Some(crate::function_profiler::FunctionProfiler::new());
    syscall_handling::record_function_profiling_test(&entry, profiler.as_mut(), 100);
}

// handle_anomaly_detection tests
#[test]
fn test_handle_anomaly_detection_none_entry() {
    let mut detector = Some(crate::anomaly::AnomalyDetector::new(100, 2.0));
    syscall_handling::handle_anomaly_detection_test(&None, detector.as_mut(), 100);
}

#[test]
fn test_handle_anomaly_detection_none_detector() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    syscall_handling::handle_anomaly_detection_test(&entry, None, 100);
}

#[test]
fn test_handle_anomaly_detection_with_data() {
    let entry = Some(SyscallEntry {
        name: "read".to_string(),
        args: vec![],
        source: None,
        function_name: None,
        caller_name: None,
        raw_arg1: None,
        raw_arg2: None,
        _raw_arg3: None,
    });
    let mut detector = Some(crate::anomaly::AnomalyDetector::new(100, 2.0));
    syscall_handling::handle_anomaly_detection_test(&entry, detector.as_mut(), 100);
}

// print_summaries tests with Tracers struct
#[test]
fn test_print_summaries_minimal() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: None,
        json_output: None,
        csv_output: None,
        csv_stats_output: None,
        html_output: None,
        anomaly_detector: None,
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_summaries(tracers, false, 0, &analysis);
}

#[test]
fn test_print_summaries_with_stats() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: Some(crate::stats::StatsTracker::new()),
        json_output: None,
        csv_output: None,
        csv_stats_output: None,
        html_output: None,
        anomaly_detector: None,
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_summaries(tracers, true, 0, &analysis);
}

#[test]
fn test_print_summaries_with_json() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: None,
        json_output: Some(crate::json_output::JsonOutput::new()),
        csv_output: None,
        csv_stats_output: None,
        html_output: None,
        anomaly_detector: None,
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_summaries(tracers, false, 0, &analysis);
}

#[test]
fn test_print_summaries_with_csv_stats() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: Some(crate::stats::StatsTracker::new()),
        json_output: None,
        csv_output: None,
        csv_stats_output: Some(crate::csv_output::CsvStatsOutput::new()),
        html_output: None,
        anomaly_detector: None,
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_summaries(tracers, true, 0, &analysis);
}

#[test]
fn test_print_summaries_with_html() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: None,
        json_output: None,
        csv_output: None,
        csv_stats_output: None,
        html_output: Some(crate::html_output::HtmlOutput::new(true, true)),
        anomaly_detector: None,
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_summaries(tracers, false, 0, &analysis);
}

#[test]
fn test_print_summaries_with_profiling() {
    let tracers = Tracers {
        profiling_ctx: Some(crate::profiling::ProfilingContext::new()),
        function_profiler: Some(crate::function_profiler::FunctionProfiler::new()),
        stats_tracker: None,
        json_output: None,
        csv_output: None,
        csv_stats_output: None,
        html_output: None,
        anomaly_detector: Some(crate::anomaly::AnomalyDetector::new(100, 2.0)),
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_summaries(tracers, false, 0, &analysis);
}

#[test]
fn test_print_summaries_with_decision_tracer() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: None,
        json_output: None,
        csv_output: None,
        csv_stats_output: None,
        html_output: None,
        anomaly_detector: None,
        decision_tracer: Some(crate::decision_trace::DecisionTracer::new()),
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_summaries(tracers, false, 0, &analysis);
}

#[test]
fn test_print_summaries_all_enabled() {
    let tracers = Tracers {
        profiling_ctx: Some(crate::profiling::ProfilingContext::new()),
        function_profiler: Some(crate::function_profiler::FunctionProfiler::new()),
        stats_tracker: Some(crate::stats::StatsTracker::new()),
        json_output: None,
        csv_output: None,
        csv_stats_output: None,
        html_output: None,
        anomaly_detector: Some(crate::anomaly::AnomalyDetector::new(100, 2.0)),
        decision_tracer: Some(crate::decision_trace::DecisionTracer::new()),
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 3.0,
        hpu_analysis: true,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 5,
        ml_compare: true,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: true,
    };
    output::print_summaries(tracers, true, 0, &analysis);
}

// Tests with populated stats tracker to cover iteration paths
fn create_populated_stats_tracker() -> crate::stats::StatsTracker {
    let mut tracker = crate::stats::StatsTracker::new();
    // Record some syscalls to populate the tracker
    tracker.record("read", 0, 1000);
    tracker.record("write", 0, 2000);
    tracker.record("open", 0, 500);
    tracker.record("close", 0, 100);
    tracker.record("read", 0, 1500);
    tracker.record("write", -1, 3000);
    tracker
}

#[test]
fn test_generate_ml_analysis_populated() {
    let tracker = Some(create_populated_stats_tracker());
    let result = ml_analysis::generate_ml_analysis_for_json(&tracker, 3);
    assert!(result.is_some());
    let report = result.expect("test");
    assert!(report.total_samples > 0);
}

#[test]
fn test_generate_isolation_forest_populated() {
    let tracker = Some(create_populated_stats_tracker());
    let result = ml_analysis::generate_isolation_forest_analysis_for_json(&tracker, 50, 0.1, false);
    assert!(result.is_some());
}

#[test]
fn test_generate_isolation_forest_populated_with_explain() {
    let tracker = Some(create_populated_stats_tracker());
    let result = ml_analysis::generate_isolation_forest_analysis_for_json(&tracker, 50, 0.1, true);
    assert!(result.is_some());
}

#[test]
fn test_generate_autoencoder_populated() {
    let tracker = Some(create_populated_stats_tracker());
    let result = ml_analysis::generate_autoencoder_analysis_for_json(&tracker, 8, 10, 2.0, false);
    assert!(result.is_some());
}

#[test]
fn test_generate_autoencoder_populated_with_explain() {
    let tracker = Some(create_populated_stats_tracker());
    let result = ml_analysis::generate_autoencoder_analysis_for_json(&tracker, 8, 10, 2.0, true);
    assert!(result.is_some());
}

#[test]
fn test_print_text_stats_populated() {
    let tracker = Some(create_populated_stats_tracker());
    #[cfg(feature = "otlp")]
    output::print_text_stats(&tracker, false, 2.0, None);
    #[cfg(not(feature = "otlp"))]
    output::print_text_stats(&tracker, false, 2.0, None);
}

#[test]
fn test_print_text_stats_populated_extended() {
    let tracker = Some(create_populated_stats_tracker());
    #[cfg(feature = "otlp")]
    output::print_text_stats(&tracker, true, 2.0, None);
    #[cfg(not(feature = "otlp"))]
    output::print_text_stats(&tracker, true, 2.0, None);
}

#[test]
fn test_print_csv_stats_populated() {
    let csv_stats = crate::csv_output::CsvStatsOutput::new();
    let tracker = Some(create_populated_stats_tracker());
    output::print_csv_stats(csv_stats, &tracker, false, false, 2.0);
}

#[test]
fn test_print_csv_stats_populated_extended() {
    let csv_stats = crate::csv_output::CsvStatsOutput::new();
    let tracker = Some(create_populated_stats_tracker());
    output::print_csv_stats(csv_stats, &tracker, true, true, 2.0);
}

#[test]
fn test_print_hpu_analysis_populated() {
    let tracker = Some(create_populated_stats_tracker());
    output::print_hpu_analysis(&tracker, false);
}

#[test]
fn test_print_hpu_analysis_populated_cpu_only() {
    let tracker = Some(create_populated_stats_tracker());
    output::print_hpu_analysis(&tracker, true);
}

#[test]
fn test_print_ml_analysis_populated() {
    let tracker = Some(create_populated_stats_tracker());
    ml_analysis::print_ml_analysis(&tracker, 3, false, 2.0);
}

#[test]
fn test_print_ml_analysis_populated_compare() {
    let tracker = Some(create_populated_stats_tracker());
    ml_analysis::print_ml_analysis(&tracker, 3, true, 2.0);
}

#[test]
fn test_print_isolation_forest_populated() {
    let tracker = Some(create_populated_stats_tracker());
    ml_analysis::print_isolation_forest_analysis(&tracker, 50, 0.1, false);
}

#[test]
fn test_print_isolation_forest_populated_explain() {
    let tracker = Some(create_populated_stats_tracker());
    ml_analysis::print_isolation_forest_analysis(&tracker, 50, 0.1, true);
}

#[test]
fn test_print_autoencoder_populated() {
    let tracker = Some(create_populated_stats_tracker());
    ml_analysis::print_autoencoder_analysis(&tracker, 8, 10, 2.0, false);
}

#[test]
fn test_print_autoencoder_populated_explain() {
    let tracker = Some(create_populated_stats_tracker());
    ml_analysis::print_autoencoder_analysis(&tracker, 8, 10, 2.0, true);
}

#[test]
fn test_handle_json_output_populated() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(create_populated_stats_tracker());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 0);
}

#[test]
fn test_handle_json_output_populated_ml() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(create_populated_stats_tracker());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 3,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 0);
}

#[test]
fn test_handle_json_output_populated_outliers() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(create_populated_stats_tracker());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 50,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: true,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 0);
}

#[test]
fn test_handle_json_output_populated_dl() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(create_populated_stats_tracker());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 10,
        explain: false,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 0);
}

#[test]
fn test_handle_json_output_populated_all() {
    let json_out = crate::json_output::JsonOutput::new();
    let tracker = Some(create_populated_stats_tracker());
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 2.0,
        hpu_analysis: true,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 3,
        ml_compare: true,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 50,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 10,
        explain: true,
    };
    output::handle_json_output(json_out, &tracker, &analysis, 1);
}

#[test]
fn test_print_analysis_summaries_populated() {
    let tracker = Some(create_populated_stats_tracker());
    let analysis = AnalysisConfig {
        stats_extended: false,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_analysis_summaries(&tracker, &analysis);
}

#[test]
fn test_print_analysis_summaries_populated_all() {
    let tracker = Some(create_populated_stats_tracker());
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 2.0,
        hpu_analysis: true,
        hpu_cpu_only: true,
        ml_anomaly: true,
        ml_clusters: 3,
        ml_compare: true,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 50,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 10,
        explain: true,
    };
    output::print_analysis_summaries(&tracker, &analysis);
}

#[test]
fn test_print_summaries_populated_stats() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: Some(create_populated_stats_tracker()),
        json_output: None,
        csv_output: None,
        csv_stats_output: None,
        html_output: None,
        anomaly_detector: None,
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 2.0,
        hpu_analysis: true,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 3,
        ml_compare: true,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 50,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 10,
        explain: true,
    };
    output::print_summaries(tracers, true, 0, &analysis);
}

#[test]
fn test_print_summaries_populated_csv_stats() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: Some(create_populated_stats_tracker()),
        json_output: None,
        csv_output: None,
        csv_stats_output: Some(crate::csv_output::CsvStatsOutput::new()),
        html_output: None,
        anomaly_detector: None,
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: false,
        ml_clusters: 5,
        ml_compare: false,
        ml_outliers: false,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 100,
        dl_anomaly: false,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 50,
        explain: false,
    };
    output::print_summaries(tracers, true, 0, &analysis);
}

#[test]
fn test_print_summaries_populated_json() {
    let tracers = Tracers {
        profiling_ctx: None,
        function_profiler: None,
        stats_tracker: Some(create_populated_stats_tracker()),
        json_output: Some(crate::json_output::JsonOutput::new()),
        csv_output: None,
        csv_stats_output: None,
        html_output: None,
        anomaly_detector: None,
        decision_tracer: None,
        #[cfg(feature = "otlp")]
        otlp_exporter: None,
        visualizer_sink: None,
    };
    let analysis = AnalysisConfig {
        stats_extended: true,
        anomaly_threshold: 2.0,
        hpu_analysis: false,
        hpu_cpu_only: false,
        ml_anomaly: true,
        ml_clusters: 3,
        ml_compare: false,
        ml_outliers: true,
        ml_outlier_threshold: 0.1,
        ml_outlier_trees: 50,
        dl_anomaly: true,
        dl_threshold: 2.0,
        dl_hidden_size: 8,
        dl_epochs: 10,
        explain: true,
    };
    output::print_summaries(tracers, true, 0, &analysis);
}
