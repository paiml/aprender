//! CLI argument parsing and execution for Renacer

use crate::{
    chaos::ChaosConfig,
    filter, tracer, transpiler_map,
    validate::{self, ValidateConfig},
    visualize::{self, VisualizeConfig},
};
use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

/// Output format for syscall traces
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum OutputFormat {
    /// Human-readable text format (default)
    #[default]
    Text,
    /// JSON format for machine parsing
    Json,
    /// CSV format for spreadsheet analysis
    Csv,
    /// HTML format for visual reports (Sprint 22)
    Html,
}

#[derive(Parser, Debug)]
#[command(name = "renacer")]
#[command(version)]
#[command(about = "Pure Rust system call tracer with source correlation", long_about = None)]
pub struct Cli {
    /// Enable source code correlation using DWARF debug info
    #[arg(short, long)]
    pub source: bool,

    /// Filter syscalls to trace (e.g., -e trace=open,read,write or -e trace=file)
    #[arg(short = 'e', long = "expr", value_name = "EXPR")]
    pub filter: Option<String>,

    /// Show statistics summary (syscall counts and timing) instead of individual calls
    #[arg(short = 'c', long = "summary")]
    pub statistics: bool,

    /// Show time spent in each syscall
    #[arg(short = 'T', long = "timing")]
    pub timing: bool,

    /// Output format (text or json)
    #[arg(long = "format", value_enum, default_value = "text")]
    pub format: OutputFormat,

    /// Attach to running process by PID (mutually exclusive with command)
    #[arg(short = 'p', long = "pid", value_name = "PID")]
    pub pid: Option<i32>,

    /// Follow forks (trace child processes)
    #[arg(short = 'f', long = "follow-forks")]
    pub follow_forks: bool,

    /// Enable self-profiling to measure Renacer's own overhead
    #[arg(long = "profile-self")]
    pub profile_self: bool,

    /// Enable function-level timing with DWARF correlation
    #[arg(long = "function-time")]
    pub function_time: bool,

    /// Enable extended statistics with percentiles and anomaly detection (requires -c)
    #[arg(long = "stats-extended")]
    pub stats_extended: bool,

    /// Anomaly detection threshold in standard deviations (default: 3.0)
    #[arg(long = "anomaly-threshold", value_name = "SIGMA", default_value = "3.0")]
    pub anomaly_threshold: f32,

    /// Enable real-time anomaly detection (Sprint 20)
    #[arg(long = "anomaly-realtime")]
    pub anomaly_realtime: bool,

    /// Sliding window size for real-time anomaly detection (default: 100)
    #[arg(long = "anomaly-window-size", value_name = "SIZE", default_value = "100")]
    pub anomaly_window_size: usize,

    /// Enable HPU-accelerated analysis (GPU if available) (Sprint 21)
    #[arg(long = "hpu-analysis")]
    pub hpu_analysis: bool,

    /// Force CPU backend (disable GPU acceleration)
    #[arg(long = "hpu-cpu-only")]
    pub hpu_cpu_only: bool,

    /// Enable ML-based anomaly detection using Aprender (Sprint 23)
    #[arg(long = "ml-anomaly")]
    pub ml_anomaly: bool,

    /// Number of clusters for ML anomaly detection (default: 3, min: 2)
    #[arg(long = "ml-clusters", value_name = "N", default_value = "3")]
    pub ml_clusters: usize,

    /// Compare ML results with z-score anomaly detection
    #[arg(long = "ml-compare")]
    pub ml_compare: bool,

    /// Enable Isolation Forest-based outlier detection (Sprint 22)
    #[arg(long = "ml-outliers")]
    pub ml_outliers: bool,

    /// Contamination threshold for Isolation Forest (default: 0.1, range: 0.0-0.5)
    #[arg(long = "ml-outlier-threshold", value_name = "THRESHOLD", default_value = "0.1")]
    pub ml_outlier_threshold: f32,

    /// Number of trees in Isolation Forest (default: 100, min: 10)
    #[arg(long = "ml-outlier-trees", value_name = "N", default_value = "100")]
    pub ml_outlier_trees: usize,

    /// Enable explainability for ML outlier detection (Sprint 22)
    #[arg(long = "explain")]
    pub explain: bool,

    /// Enable deep learning (Autoencoder) anomaly detection (Sprint 23)
    #[arg(long = "dl-anomaly")]
    pub dl_anomaly: bool,

    /// Reconstruction error threshold for Autoencoder (default: 2.0)
    #[arg(long = "dl-threshold", value_name = "THRESHOLD", default_value = "2.0")]
    pub dl_threshold: f32,

    /// Hidden layer size for Autoencoder (default: 3)
    #[arg(long = "dl-hidden-size", value_name = "SIZE", default_value = "3")]
    pub dl_hidden_size: usize,

    /// Number of training epochs for Autoencoder (default: 100)
    #[arg(long = "dl-epochs", value_name = "N", default_value = "100")]
    pub dl_epochs: usize,

    /// Path to transpiler source map JSON file (Sprint 24)
    #[arg(long = "transpiler-map", value_name = "FILE")]
    pub transpiler_map: Option<String>,

    /// Show verbose transpiler context (Python/Rust correlation) (Sprint 25)
    #[arg(long = "show-transpiler-context")]
    pub show_transpiler_context: bool,

    /// Rewrite stack traces to show original source locations (Sprint 26)
    #[arg(long = "rewrite-stacktrace")]
    pub rewrite_stacktrace: bool,

    /// Rewrite compilation errors to show original source locations (Sprint 27)
    #[arg(long = "rewrite-errors")]
    pub rewrite_errors: bool,

    /// Trace transpiler compile-time decisions for debugging (Sprint 26)
    #[arg(long = "trace-transpiler-decisions")]
    pub trace_transpiler_decisions: bool,

    /// OpenTelemetry OTLP endpoint for trace export (Sprint 30)
    #[arg(long = "otlp-endpoint", value_name = "URL")]
    pub otlp_endpoint: Option<String>,

    /// Service name for OpenTelemetry traces (Sprint 30)
    #[arg(long = "otlp-service-name", value_name = "NAME", default_value = "renacer")]
    pub otlp_service_name: String,

    /// W3C Trace Context for distributed tracing (Sprint 33)
    ///
    /// Inject trace context to create Renacer spans as children of application spans.
    /// Format: version-trace_id-parent_id-trace_flags
    /// Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
    ///
    /// If not provided, checks TRACEPARENT or `OTEL_TRACEPARENT` environment variables.
    /// If no context found, creates new root trace (existing behavior).
    #[arg(long = "trace-parent", value_name = "TRACEPARENT")]
    pub trace_parent: Option<String>,

    /// Enable compute block tracing (Trueno SIMD operations) - Sprint 32
    ///
    /// Exports statistical computation blocks (e.g., `calculate_statistics`) as
    /// OpenTelemetry spans. Uses adaptive sampling: only traces blocks with
    /// duration >= threshold (default: 100μs). Toyota Way compliant: safe by
    /// default, cannot `DoS` tracing backend.
    #[arg(long = "trace-compute")]
    pub trace_compute: bool,

    /// Trace ALL compute blocks (bypass adaptive sampling threshold) - Sprint 32
    ///
    /// Debug mode: traces even fast compute blocks (<100μs). Use for development
    /// and debugging only. Can generate high span volume (~500 spans/sec).
    /// Requires --trace-compute flag.
    #[arg(long = "trace-compute-all", requires = "trace_compute")]
    pub trace_compute_all: bool,

    /// Custom threshold for compute block tracing (microseconds) - Sprint 32
    ///
    /// Only trace compute blocks with duration >= threshold. Default: 100μs.
    /// Lower values increase span volume. Requires --trace-compute flag.
    #[arg(
        long = "trace-compute-threshold",
        value_name = "MICROS",
        default_value = "100",
        requires = "trace_compute"
    )]
    pub trace_compute_threshold: u64,

    /// Enable debug tracing output to stderr
    #[arg(long = "debug")]
    pub debug: bool,

    // Sprint 47: Chaos Engineering CLI (Issue #17)
    /// Chaos engineering preset (gentle, aggressive)
    ///
    /// Presets configure resource limits for robustness testing:
    /// - gentle: memory=512MB, cpu=80%, timeout=120s
    /// - aggressive: memory=64MB, cpu=25%, timeout=10s, signals=on
    #[arg(long = "chaos", value_name = "PRESET")]
    pub chaos_preset: Option<String>,

    /// Memory limit for chaos testing (e.g., 64M, 512M, 1G, or bytes)
    ///
    /// Limits the traced process's virtual memory using setrlimit.
    /// Supports suffixes: K (kilobytes), M (megabytes), G (gigabytes).
    #[arg(long = "chaos-memory-limit", value_name = "SIZE")]
    pub chaos_memory_limit: Option<String>,

    /// CPU limit as fraction (0.0-1.0) for chaos testing
    ///
    /// Limits CPU time relative to real time. For example, 0.5 means
    /// the process gets 50% of CPU time.
    #[arg(long = "chaos-cpu-limit", value_name = "FRACTION")]
    pub chaos_cpu_limit: Option<f64>,

    /// Execution timeout for chaos testing (e.g., 10s, 2m, 1h)
    ///
    /// Terminates the traced process after the specified duration.
    /// Supports suffixes: s (seconds), m (minutes), h (hours).
    #[arg(long = "chaos-timeout", value_name = "DURATION")]
    pub chaos_timeout: Option<String>,

    /// Enable random signal injection for chaos testing
    ///
    /// Periodically injects signals (SIGALRM, SIGUSR1) to test
    /// signal handling robustness.
    #[arg(long = "chaos-signals")]
    pub chaos_signals: bool,

    // Sprint 48: Model Persistence CLI (Toyota Way: Muda elimination)
    /// Save trained ML model to .apr file for reuse
    ///
    /// After ML anomaly detection training, saves the model to the
    /// specified path. Enables 10-50x faster startup on subsequent runs.
    /// Requires --ml-anomaly flag.
    #[arg(long = "save-model", value_name = "FILE")]
    pub save_model: Option<String>,

    /// Load pre-trained ML model from .apr file
    ///
    /// Skips model training and loads a previously saved model.
    /// Provides instant startup for anomaly detection.
    /// Requires --ml-anomaly flag.
    #[arg(long = "load-model", value_name = "FILE")]
    pub load_model: Option<String>,

    /// Compare against baseline model for regression detection
    ///
    /// Loads a baseline model and compares current trace against it.
    /// Reports deviations from the baseline as potential regressions.
    /// Requires --ml-anomaly flag.
    #[arg(long = "baseline", value_name = "FILE")]
    pub baseline_model: Option<String>,

    /// Command to trace (everything after --)
    #[arg(last = true)]
    pub command: Option<Vec<String>>,

    /// Subcommand (validate, etc.)
    #[command(subcommand)]
    pub subcommand: Option<Commands>,
}

/// Available subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Validate traces against golden baseline (Sprint 50)
    ///
    /// Generates or compares syscall traces against a golden baseline.
    /// Exit codes: 0=passed, 1=failed, 2=baseline not found, 3=invalid baseline,
    /// 4=command error, 5=config error
    Validate(ValidateArgs),

    /// Real-time TUI visualization of syscall traces (Sprint 52-55)
    ///
    /// Launches an interactive terminal UI showing syscall heatmaps, anomaly
    /// timelines, ML clustering, and OTLP span waterfalls in real-time.
    /// Uses ttop-identical architecture for 8ms frame times.
    Visualize(VisualizeArgs),
}

/// Arguments for the validate subcommand
#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// Path to existing baseline directory for comparison
    ///
    /// Compares the traced command against the baseline and reports regressions.
    #[arg(long = "baseline", value_name = "DIR")]
    pub baseline: Option<PathBuf>,

    /// Generate new baseline in the specified directory
    ///
    /// Traces the command and saves the result as a golden baseline.
    #[arg(long = "generate", value_name = "DIR")]
    pub generate: Option<PathBuf>,

    /// Timing tolerance percentage (default: 10.0)
    ///
    /// Allow timing variations within this percentage before reporting regression.
    #[arg(long = "tolerance", value_name = "PERCENT", default_value = "10.0")]
    pub tolerance: f32,

    /// Enable strict mode (zero tolerance)
    ///
    /// Any deviation from baseline is reported as failure.
    #[arg(long = "strict")]
    pub strict: bool,

    /// Ignore timing, compare behavior only
    ///
    /// Only compare syscall sequences, not their timing.
    #[arg(long = "ignore-timing")]
    pub ignore_timing: bool,

    /// Stop on first regression (fail fast)
    #[arg(long = "fail-fast")]
    pub fail_fast: bool,

    /// Output format for validation results
    #[arg(long = "output", value_enum, default_value = "text")]
    pub output: ValidationOutputFormat,

    /// Command to trace and validate (everything after --)
    #[arg(last = true, required = true)]
    pub command: Vec<String>,
}

/// Output format for validation results (Sprint 50)
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum ValidationOutputFormat {
    /// Human-readable text format (default)
    #[default]
    Text,
    /// JSON format for machine parsing
    Json,
    /// `JUnit` XML format for CI systems
    Junit,
}

/// Arguments for the visualize subcommand (Sprint 52-55)
#[derive(Args, Debug)]
pub struct VisualizeArgs {
    /// Tick rate in milliseconds (default: 50, matches ttop)
    #[arg(long = "tick-rate", value_name = "MS", default_value = "50")]
    pub tick_rate: u64,

    /// Disable anomaly detection panel
    #[arg(long = "no-anomaly")]
    pub no_anomaly: bool,

    /// Disable ML clustering panel
    #[arg(long = "no-ml")]
    pub no_ml: bool,

    /// Enable ML clustering (disabled by default)
    #[arg(long = "ml")]
    pub enable_ml: bool,

    /// Number of ML clusters (default: 3)
    #[arg(long = "ml-clusters", value_name = "N", default_value = "3")]
    pub ml_clusters: usize,

    /// Anomaly Z-score threshold (default: 3.0)
    #[arg(long = "anomaly-threshold", value_name = "SIGMA", default_value = "3.0")]
    pub anomaly_threshold: f32,

    /// History buffer size (default: 300, matches ttop)
    #[arg(long = "history-size", value_name = "N", default_value = "300")]
    pub history_size: usize,

    /// Enable deterministic mode for testing
    #[arg(long = "deterministic")]
    pub deterministic: bool,

    /// Show FPS overlay for performance monitoring
    #[arg(long = "show-fps")]
    pub show_fps: bool,

    /// Attach to running process by PID (mutually exclusive with command)
    #[arg(short = 'p', long = "pid", value_name = "PID")]
    pub pid: Option<i32>,

    /// Enable source code correlation using DWARF debug info
    #[arg(short = 's', long = "source")]
    pub source: bool,

    /// Filter syscalls to trace (e.g., -e trace=open,read,write)
    #[arg(short = 'e', long = "expr", value_name = "EXPR")]
    pub filter: Option<String>,

    /// OTLP endpoint for span export (enables trace waterfall panel)
    #[arg(long = "otlp-endpoint", value_name = "URL")]
    pub otlp_endpoint: Option<String>,

    // Sprint 56: Metrics and Alerting Panel Options
    /// Enable metrics collection panel (counters, gauges, histograms)
    #[arg(long = "metrics")]
    pub enable_metrics: bool,

    /// Enable alerting panel (threshold/rate/absence alerts)
    #[arg(long = "alerts")]
    pub enable_alerts: bool,

    /// Alert threshold for syscall latency anomaly (microseconds, default: 10000)
    #[arg(long = "alert-latency-threshold", value_name = "US", default_value = "10000")]
    pub alert_latency_threshold: u64,

    /// Alert threshold for error rate percentage (default: 5.0)
    #[arg(long = "alert-error-rate", value_name = "PERCENT", default_value = "5.0")]
    pub alert_error_rate: f32,

    /// Command to trace (everything after --)
    #[arg(last = true)]
    pub command: Option<Vec<String>>,
}

/// Initialize tracing subscriber for debug output
fn init_tracing(debug: bool) {
    if debug {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::from_default_env().add_directive(tracing::Level::TRACE.into()),
            )
            .with_writer(std::io::stderr)
            .init();
    }
}

/// Print function mappings from transpiler source map (Sprint 25)
fn print_function_mappings(map: &transpiler_map::TranspilerMap, show_context: bool) {
    if show_context {
        println!("=== Transpiler Source Map ===");
        println!("Source Language: {} -> Rust", map.source_language());
        println!("Source File: {}", map.source_file().display());
        println!();
    }

    if !map.function_map.is_empty() {
        if show_context {
            println!("Function Mappings (Rust -> {}):", map.source_language());
            println!("─────────────────────────────────────────");
        }
        for (rust_fn, python_fn) in &map.function_map {
            println!("{rust_fn} -> {python_fn}");
        }
        if show_context {
            println!("─────────────────────────────────────────");
            println!();
        }
    }
}

/// Print stack trace mappings from transpiler source map (Sprint 26)
fn print_stack_trace_mappings(map: &transpiler_map::TranspilerMap, show_context: bool) {
    if show_context {
        println!("=== Stack Trace Mapping ===");
        println!("Source: {} -> {}", map.source_file().display(), map.generated_file().display());
        println!();
    }

    if !map.mappings.is_empty() {
        if show_context {
            println!("Line Mappings (Rust -> {}):", map.source_language());
            println!("─────────────────────────────────────────");
        }
        for mapping in &map.mappings {
            println!(
                "{} ({}:{}) -> {}:{}",
                mapping.rust_function,
                map.generated_file().display(),
                mapping.rust_line,
                map.source_file().display(),
                mapping.python_line
            );
        }
        if show_context {
            println!("─────────────────────────────────────────");
            println!();
        }
    }
}

/// Print error correlation mappings from transpiler source map (Sprint 27)
fn print_error_correlation_mappings(map: &transpiler_map::TranspilerMap, show_context: bool) {
    if show_context {
        println!("=== Error Correlation Mapping ===");
        println!(
            "Errors in {} will map to {}",
            map.generated_file().display(),
            map.source_file().display()
        );
        println!();
    }

    if !map.mappings.is_empty() {
        if show_context {
            println!("Available Line Mappings ({} entries):", map.mappings.len());
            println!("─────────────────────────────────────────");
        }
        for mapping in &map.mappings {
            println!(
                "  {}:{} -> {}:{} ({})",
                map.generated_file().display(),
                mapping.rust_line,
                map.source_file().display(),
                mapping.python_line,
                mapping.python_function
            );
        }
        if show_context {
            println!("─────────────────────────────────────────");
            println!();
        }
    }
}

/// Execute the tracer based on PID or command arguments
fn run_tracer(
    pid: Option<i32>,
    command: Option<Vec<String>>,
    config: tracer::TracerConfig,
) -> Result<i32> {
    match (pid, command) {
        (Some(pid), None) => tracer::attach_to_pid(pid, config),
        (None, Some(command)) => tracer::trace_command(&command, config),
        (Some(_), Some(_)) => {
            anyhow::bail!("Cannot specify both -p PID and command. Choose one.");
        }
        (None, None) => {
            anyhow::bail!(
                "Must specify either -p PID or command. Usage: renacer -p PID or renacer -- COMMAND [ARGS...]"
            );
        }
    }
}

/// Run the validate subcommand (Sprint 50)
fn run_validate_subcommand(args: &ValidateArgs) -> i32 {
    // Convert CLI ValidationOutputFormat to validate module's format
    let output_format = match args.output {
        ValidationOutputFormat::Text => validate::ValidationOutputFormat::Text,
        ValidationOutputFormat::Json => validate::ValidationOutputFormat::Json,
        ValidationOutputFormat::Junit => validate::ValidationOutputFormat::JUnit,
    };

    // Build ValidateConfig from CLI args
    let config = ValidateConfig::default()
        .set_tolerance(args.tolerance)
        .with_strict_mode(args.strict)
        .with_ignore_timing(args.ignore_timing)
        .with_fail_fast(args.fail_fast)
        .with_output_format(output_format);

    // Apply baseline or generate directory
    let config = if let Some(baseline) = &args.baseline {
        config.with_baseline(baseline.clone())
    } else {
        config
    };

    let config = if let Some(generate) = &args.generate {
        config.with_generate(generate.clone())
    } else {
        config
    };

    // Run validation
    let exit_code = validate::run_validate(&args.command, &config);
    exit_code.code()
}

/// Run the visualize subcommand (Sprint 52-55)
fn run_visualize_subcommand(args: &VisualizeArgs) -> Result<i32> {
    // Validate: must have either -p PID or -- COMMAND (same gate as root handler)
    let has_command = args.command.as_ref().is_some_and(|c| !c.is_empty());
    if args.pid.is_none() && !has_command {
        anyhow::bail!(
            "Must specify either -p PID or command for visualization. \
             Usage: renacer visualize -p PID or renacer visualize -- COMMAND [ARGS...]"
        );
    }
    if args.pid.is_some() && has_command {
        anyhow::bail!("Cannot specify both -p PID and command. Choose one.");
    }

    // Build VisualizeConfig from CLI args
    let config = VisualizeConfig {
        tick_rate_ms: args.tick_rate,
        enable_anomaly: !args.no_anomaly,
        enable_ml: args.enable_ml && !args.no_ml,
        ml_clusters: args.ml_clusters,
        anomaly_threshold: args.anomaly_threshold,
        history_size: args.history_size,
        deterministic: args.deterministic,
        show_fps: args.show_fps,
        pid: args.pid,
        enable_source: args.source,
        filter: args.filter.clone(),
        otlp_endpoint: args.otlp_endpoint.clone(),
        // Sprint 56: Metrics and alerting
        enable_metrics: args.enable_metrics,
        enable_alerts: args.enable_alerts,
        alert_latency_threshold_us: args.alert_latency_threshold,
        alert_error_rate_percent: args.alert_error_rate,
    };

    // Run visualization
    visualize::run_visualize(args.command.as_deref(), config)
}

/// Main entry point - parses CLI and runs the appropriate command.
/// Returns the exit code.
pub fn run() -> Result<i32> {
    let args = Cli::parse();

    // Sprint 50+52: Handle subcommands first
    if let Some(subcommand) = &args.subcommand {
        match subcommand {
            Commands::Validate(validate_args) => {
                return Ok(run_validate_subcommand(validate_args));
            }
            Commands::Visualize(visualize_args) => {
                return run_visualize_subcommand(visualize_args);
            }
        }
    }

    // Validate ml_clusters range (must be >= 2)
    if args.ml_clusters < 2 {
        anyhow::bail!("Invalid value for --ml-clusters: {} (must be >= 2)", args.ml_clusters);
    }

    // Initialize tracing if --debug flag is set
    init_tracing(args.debug);

    // Load transpiler source map if provided (Sprint 24)
    let source_map = if let Some(map_path) = &args.transpiler_map {
        Some(transpiler_map::TranspilerMap::from_file(map_path)?)
    } else {
        None
    };

    // Sprint 25: Print function name correlations when using --function-time with source map
    if let (true, Some(ref map)) = (args.function_time, &source_map) {
        print_function_mappings(map, args.show_transpiler_context);
    }

    // Sprint 26: Print stack trace mapping info when using --rewrite-stacktrace with source map
    if let (true, Some(ref map)) = (args.rewrite_stacktrace, &source_map) {
        print_stack_trace_mappings(map, args.show_transpiler_context);
    }

    // Sprint 27: Print error correlation info when using --rewrite-errors with source map
    if let (true, Some(ref map)) = (args.rewrite_errors, &source_map) {
        print_error_correlation_mappings(map, args.show_transpiler_context);
    }

    // Parse filter expression if provided
    let filter = if let Some(expr) = args.filter {
        filter::SyscallFilter::from_expr(&expr)?
    } else {
        filter::SyscallFilter::all()
    };

    // Sprint 47: Parse chaos configuration (Issue #17)
    let chaos_config = ChaosConfig::from_cli(
        args.chaos_preset.as_deref(),
        args.chaos_memory_limit.as_deref(),
        args.chaos_cpu_limit,
        args.chaos_timeout.as_deref(),
        args.chaos_signals,
    )
    .map_err(|e| anyhow::anyhow!("Chaos config error: {e}"))?;

    // Display chaos mode status if enabled
    if let Some(ref chaos) = chaos_config {
        eprintln!("⚠️  Chaos mode enabled: {}", chaos.status_line());
    }

    // Create tracer configuration
    let config = tracer::TracerConfig {
        enable_source: args.source,
        filter,
        statistics_mode: args.statistics,
        timing_mode: args.timing,
        output_format: args.format,
        follow_forks: args.follow_forks,
        profile_self: args.profile_self,
        function_time: args.function_time,
        stats_extended: args.stats_extended,
        anomaly_threshold: args.anomaly_threshold,
        anomaly_realtime: args.anomaly_realtime,
        anomaly_window_size: args.anomaly_window_size,
        hpu_analysis: args.hpu_analysis,
        hpu_cpu_only: args.hpu_cpu_only,
        ml_anomaly: args.ml_anomaly,
        ml_clusters: args.ml_clusters,
        ml_compare: args.ml_compare,
        ml_outliers: args.ml_outliers,
        ml_outlier_threshold: args.ml_outlier_threshold,
        ml_outlier_trees: args.ml_outlier_trees,
        explain: args.explain,
        dl_anomaly: args.dl_anomaly,
        dl_threshold: args.dl_threshold,
        dl_hidden_size: args.dl_hidden_size,
        dl_epochs: args.dl_epochs,
        trace_transpiler_decisions: args.trace_transpiler_decisions,
        transpiler_map: source_map,
        otlp_endpoint: args.otlp_endpoint,
        otlp_service_name: args.otlp_service_name,
        trace_parent: args.trace_parent,
        chaos_config,
        visualizer_sink: None,
    };

    // Either attach to PID or trace command (mutually exclusive)
    run_tracer(args.pid, args.command, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parses_command() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "hello"]);
        assert!(cli.command.is_some());
        let cmd = cli.command.expect("test");
        assert_eq!(cmd[0], "echo");
        assert_eq!(cmd[1], "hello");
    }

    #[test]
    fn test_cli_empty_without_command() {
        let cli = Cli::parse_from(["renacer"]);
        assert!(cli.command.is_none());
    }

    #[test]
    fn test_cli_profile_self_flag() {
        let cli = Cli::parse_from(["renacer", "--profile-self", "--", "echo", "test"]);
        assert!(cli.profile_self);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_profile_self_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.profile_self);
    }

    #[test]
    fn test_cli_function_time_flag() {
        let cli = Cli::parse_from(["renacer", "--function-time", "--", "echo", "test"]);
        assert!(cli.function_time);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_function_time_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.function_time);
    }

    #[test]
    fn test_cli_stats_extended_flag() {
        let cli = Cli::parse_from(["renacer", "--stats-extended", "--", "echo", "test"]);
        assert!(cli.stats_extended);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_stats_extended_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.stats_extended);
    }

    #[test]
    fn test_cli_anomaly_threshold_default() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert_eq!(cli.anomaly_threshold, 3.0);
    }

    #[test]
    fn test_cli_anomaly_threshold_custom() {
        let cli = Cli::parse_from(["renacer", "--anomaly-threshold", "2.5", "--", "echo", "test"]);
        assert_eq!(cli.anomaly_threshold, 2.5);
    }

    #[test]
    fn test_cli_stats_extended_with_statistics() {
        let cli = Cli::parse_from(["renacer", "-c", "--stats-extended", "--", "echo", "test"]);
        assert!(cli.statistics);
        assert!(cli.stats_extended);
    }

    #[test]
    fn test_cli_hpu_analysis_flag() {
        let cli = Cli::parse_from(["renacer", "--hpu-analysis", "--", "echo", "test"]);
        assert!(cli.hpu_analysis);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_hpu_analysis_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.hpu_analysis);
    }

    #[test]
    fn test_cli_hpu_cpu_only_flag() {
        let cli = Cli::parse_from(["renacer", "--hpu-cpu-only", "--", "echo", "test"]);
        assert!(cli.hpu_cpu_only);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_hpu_cpu_only_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.hpu_cpu_only);
    }

    #[test]
    fn test_cli_hpu_with_statistics() {
        let cli = Cli::parse_from(["renacer", "-c", "--hpu-analysis", "--", "echo", "test"]);
        assert!(cli.statistics);
        assert!(cli.hpu_analysis);
    }

    #[test]
    fn test_cli_hpu_analysis_with_cpu_only() {
        let cli =
            Cli::parse_from(["renacer", "--hpu-analysis", "--hpu-cpu-only", "--", "echo", "test"]);
        assert!(cli.hpu_analysis);
        assert!(cli.hpu_cpu_only);
    }

    #[test]
    fn test_cli_ml_anomaly_flag() {
        let cli = Cli::parse_from(["renacer", "--ml-anomaly", "--", "echo", "test"]);
        assert!(cli.ml_anomaly);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_ml_anomaly_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.ml_anomaly);
    }

    #[test]
    fn test_cli_ml_clusters_default() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert_eq!(cli.ml_clusters, 3);
    }

    #[test]
    fn test_cli_ml_clusters_custom() {
        let cli = Cli::parse_from(["renacer", "--ml-clusters", "5", "--", "echo", "test"]);
        assert_eq!(cli.ml_clusters, 5);
    }

    #[test]
    fn test_cli_ml_compare_flag() {
        let cli = Cli::parse_from(["renacer", "--ml-compare", "--", "echo", "test"]);
        assert!(cli.ml_compare);
    }

    #[test]
    fn test_cli_ml_compare_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.ml_compare);
    }

    #[test]
    fn test_cli_ml_anomaly_with_statistics() {
        let cli = Cli::parse_from(["renacer", "-c", "--ml-anomaly", "--", "echo", "test"]);
        assert!(cli.statistics);
        assert!(cli.ml_anomaly);
    }

    #[test]
    fn test_cli_transpiler_map_flag() {
        let cli = Cli::parse_from([
            "renacer",
            "--transpiler-map",
            "test.sourcemap.json",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.transpiler_map.as_deref(), Some("test.sourcemap.json"));
    }

    #[test]
    fn test_cli_transpiler_map_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.transpiler_map.is_none());
    }

    #[test]
    fn test_cli_show_transpiler_context_flag() {
        let cli = Cli::parse_from(["renacer", "--show-transpiler-context", "--", "echo", "test"]);
        assert!(cli.show_transpiler_context);
    }

    #[test]
    fn test_cli_show_transpiler_context_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.show_transpiler_context);
    }

    #[test]
    fn test_cli_transpiler_map_with_function_time() {
        let cli = Cli::parse_from([
            "renacer",
            "--transpiler-map",
            "map.json",
            "--function-time",
            "--show-transpiler-context",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.transpiler_map.is_some());
        assert!(cli.function_time);
        assert!(cli.show_transpiler_context);
    }

    #[test]
    fn test_cli_rewrite_stacktrace_flag() {
        let cli = Cli::parse_from(["renacer", "--rewrite-stacktrace", "--", "echo", "test"]);
        assert!(cli.rewrite_stacktrace);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_rewrite_stacktrace_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.rewrite_stacktrace);
    }

    #[test]
    fn test_cli_rewrite_stacktrace_with_transpiler_map() {
        let cli = Cli::parse_from([
            "renacer",
            "--transpiler-map",
            "map.json",
            "--rewrite-stacktrace",
            "--show-transpiler-context",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.transpiler_map.is_some());
        assert!(cli.rewrite_stacktrace);
        assert!(cli.show_transpiler_context);
    }

    #[test]
    fn test_cli_rewrite_errors_flag() {
        let cli = Cli::parse_from(["renacer", "--rewrite-errors", "--", "echo", "test"]);
        assert!(cli.rewrite_errors);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_rewrite_errors_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.rewrite_errors);
    }

    #[test]
    fn test_cli_rewrite_errors_with_transpiler_map() {
        let cli = Cli::parse_from([
            "renacer",
            "--transpiler-map",
            "map.json",
            "--rewrite-errors",
            "--show-transpiler-context",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.transpiler_map.is_some());
        assert!(cli.rewrite_errors);
        assert!(cli.show_transpiler_context);
    }

    #[test]
    fn test_cli_ml_outliers_flag() {
        let cli = Cli::parse_from(["renacer", "--ml-outliers", "--", "echo", "test"]);
        assert!(cli.ml_outliers);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_trace_transpiler_decisions_flag() {
        let cli =
            Cli::parse_from(["renacer", "--trace-transpiler-decisions", "--", "echo", "test"]);
        assert!(cli.trace_transpiler_decisions);
        assert!(cli.command.is_some());
    }

    #[test]
    fn test_cli_ml_outliers_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.ml_outliers);
    }

    #[test]
    fn test_cli_trace_transpiler_decisions_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.trace_transpiler_decisions);
    }

    #[test]
    fn test_cli_ml_outlier_threshold_default() {
        let cli = Cli::parse_from(["renacer", "--ml-outliers", "--", "echo", "test"]);
        assert_eq!(cli.ml_outlier_threshold, 0.1);
    }

    #[test]
    fn test_cli_trace_transpiler_decisions_with_transpiler_map() {
        let cli = Cli::parse_from([
            "renacer",
            "--transpiler-map",
            "map.json",
            "--trace-transpiler-decisions",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.transpiler_map.is_some());
        assert!(cli.trace_transpiler_decisions);
    }

    #[test]
    fn test_cli_ml_outlier_threshold_custom() {
        let cli = Cli::parse_from([
            "renacer",
            "--ml-outliers",
            "--ml-outlier-threshold",
            "0.15",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.ml_outlier_threshold, 0.15);
    }

    #[test]
    fn test_cli_ml_outlier_trees_default() {
        let cli = Cli::parse_from(["renacer", "--ml-outliers", "--", "echo", "test"]);
        assert_eq!(cli.ml_outlier_trees, 100);
    }

    #[test]
    fn test_cli_ml_outlier_trees_custom() {
        let cli = Cli::parse_from([
            "renacer",
            "--ml-outliers",
            "--ml-outlier-trees",
            "150",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.ml_outlier_trees, 150);
    }

    #[test]
    fn test_cli_explain_flag() {
        let cli = Cli::parse_from(["renacer", "--ml-outliers", "--explain", "--", "echo", "test"]);
        assert!(cli.ml_outliers);
        assert!(cli.explain);
    }

    #[test]
    fn test_cli_explain_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.explain);
    }

    #[test]
    fn test_cli_ml_outliers_with_statistics() {
        let cli = Cli::parse_from(["renacer", "-c", "--ml-outliers", "--", "echo", "test"]);
        assert!(cli.statistics);
        assert!(cli.ml_outliers);
    }

    #[test]
    fn test_cli_ml_outliers_with_kmeans() {
        let cli =
            Cli::parse_from(["renacer", "--ml-outliers", "--ml-anomaly", "--", "echo", "test"]);
        assert!(cli.ml_outliers);
        assert!(cli.ml_anomaly);
    }

    // Sprint 23: Deep Learning / Autoencoder tests
    #[test]
    fn test_cli_dl_anomaly_flag() {
        let cli = Cli::parse_from(["renacer", "--dl-anomaly", "--", "echo", "test"]);
        assert!(cli.dl_anomaly);
    }

    #[test]
    fn test_cli_dl_anomaly_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.dl_anomaly);
    }

    #[test]
    fn test_cli_dl_threshold_default() {
        let cli = Cli::parse_from(["renacer", "--dl-anomaly", "--", "echo", "test"]);
        assert_eq!(cli.dl_threshold, 2.0);
    }

    #[test]
    fn test_cli_dl_threshold_custom() {
        let cli = Cli::parse_from([
            "renacer",
            "--dl-anomaly",
            "--dl-threshold",
            "3.0",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.dl_threshold, 3.0);
    }

    #[test]
    fn test_cli_dl_hidden_size_default() {
        let cli = Cli::parse_from(["renacer", "--dl-anomaly", "--", "echo", "test"]);
        assert_eq!(cli.dl_hidden_size, 3);
    }

    #[test]
    fn test_cli_dl_hidden_size_custom() {
        let cli = Cli::parse_from([
            "renacer",
            "--dl-anomaly",
            "--dl-hidden-size",
            "5",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.dl_hidden_size, 5);
    }

    #[test]
    fn test_cli_dl_epochs_default() {
        let cli = Cli::parse_from(["renacer", "--dl-anomaly", "--", "echo", "test"]);
        assert_eq!(cli.dl_epochs, 100);
    }

    #[test]
    fn test_cli_dl_epochs_custom() {
        let cli = Cli::parse_from([
            "renacer",
            "--dl-anomaly",
            "--dl-epochs",
            "200",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.dl_epochs, 200);
    }

    #[test]
    fn test_cli_dl_anomaly_with_statistics() {
        let cli = Cli::parse_from(["renacer", "-c", "--dl-anomaly", "--", "echo", "test"]);
        assert!(cli.statistics);
        assert!(cli.dl_anomaly);
    }

    #[test]
    fn test_cli_dl_anomaly_with_other_ml() {
        let cli = Cli::parse_from([
            "renacer",
            "--dl-anomaly",
            "--ml-anomaly",
            "--ml-outliers",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.dl_anomaly);
        assert!(cli.ml_anomaly);
        assert!(cli.ml_outliers);
    }

    // Sprint 30: OpenTelemetry OTLP Export tests
    #[test]
    fn test_cli_otlp_endpoint_flag() {
        let cli = Cli::parse_from([
            "renacer",
            "--otlp-endpoint",
            "http://localhost:4317",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.otlp_endpoint.is_some());
        assert_eq!(cli.otlp_endpoint.expect("test"), "http://localhost:4317");
    }

    #[test]
    fn test_cli_otlp_endpoint_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.otlp_endpoint.is_none());
    }

    #[test]
    fn test_cli_otlp_service_name_default() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert_eq!(cli.otlp_service_name, "renacer");
    }

    #[test]
    fn test_cli_otlp_service_name_custom() {
        let cli =
            Cli::parse_from(["renacer", "--otlp-service-name", "my-app", "--", "echo", "test"]);
        assert_eq!(cli.otlp_service_name, "my-app");
    }

    #[test]
    fn test_cli_otlp_with_endpoint_and_service() {
        let cli = Cli::parse_from([
            "renacer",
            "--otlp-endpoint",
            "http://jaeger:4317",
            "--otlp-service-name",
            "traced-app",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.otlp_endpoint.expect("test"), "http://jaeger:4317");
        assert_eq!(cli.otlp_service_name, "traced-app");
    }

    #[test]
    fn test_cli_otlp_with_statistics() {
        let cli = Cli::parse_from([
            "renacer",
            "-c",
            "--otlp-endpoint",
            "http://localhost:4317",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.statistics);
        assert!(cli.otlp_endpoint.is_some());
    }

    #[test]
    fn test_cli_otlp_with_timing() {
        let cli = Cli::parse_from([
            "renacer",
            "-T",
            "--otlp-endpoint",
            "http://localhost:4317",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.timing);
        assert!(cli.otlp_endpoint.is_some());
    }

    // Sprint 32: Compute Block Tracing tests
    #[test]
    fn test_cli_trace_compute_flag() {
        let cli = Cli::parse_from(["renacer", "--trace-compute", "--", "echo", "test"]);
        assert!(cli.trace_compute);
        assert!(!cli.trace_compute_all);
        assert_eq!(cli.trace_compute_threshold, 100); // default
    }

    #[test]
    fn test_cli_trace_compute_all_flag() {
        let cli = Cli::parse_from([
            "renacer",
            "--trace-compute",
            "--trace-compute-all",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.trace_compute);
        assert!(cli.trace_compute_all);
    }

    #[test]
    fn test_cli_trace_compute_threshold_custom() {
        let cli = Cli::parse_from([
            "renacer",
            "--trace-compute",
            "--trace-compute-threshold",
            "50",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.trace_compute);
        assert_eq!(cli.trace_compute_threshold, 50);
    }

    #[test]
    fn test_cli_trace_compute_with_otlp() {
        let cli = Cli::parse_from([
            "renacer",
            "--otlp-endpoint",
            "http://localhost:4317",
            "--trace-compute",
            "-c",
            "--stats-extended",
            "--",
            "cargo",
            "build",
        ]);
        assert!(cli.otlp_endpoint.is_some());
        assert!(cli.trace_compute);
        assert!(cli.statistics);
        assert!(cli.stats_extended);
    }

    #[test]
    fn test_cli_trace_compute_all_requires_trace_compute() {
        // Should fail because --trace-compute-all requires --trace-compute
        let result = Cli::try_parse_from(["renacer", "--trace-compute-all", "--", "echo", "test"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_trace_compute_threshold_requires_trace_compute() {
        // Should fail because --trace-compute-threshold requires --trace-compute
        let result = Cli::try_parse_from([
            "renacer",
            "--trace-compute-threshold",
            "50",
            "--",
            "echo",
            "test",
        ]);
        assert!(result.is_err());
    }

    // Sprint 47: Chaos Engineering CLI Tests (Issue #17)
    #[test]
    fn test_cli_chaos_preset_gentle() {
        let cli = Cli::parse_from(["renacer", "--chaos", "gentle", "--", "echo", "test"]);
        assert_eq!(cli.chaos_preset.as_deref(), Some("gentle"));
    }

    #[test]
    fn test_cli_chaos_preset_aggressive() {
        let cli = Cli::parse_from(["renacer", "--chaos", "aggressive", "--", "echo", "test"]);
        assert_eq!(cli.chaos_preset.as_deref(), Some("aggressive"));
    }

    #[test]
    fn test_cli_chaos_preset_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.chaos_preset.is_none());
    }

    #[test]
    fn test_cli_chaos_memory_limit() {
        let cli = Cli::parse_from(["renacer", "--chaos-memory-limit", "64M", "--", "echo", "test"]);
        assert_eq!(cli.chaos_memory_limit.as_deref(), Some("64M"));
    }

    #[test]
    fn test_cli_chaos_memory_limit_bytes() {
        let cli =
            Cli::parse_from(["renacer", "--chaos-memory-limit", "67108864", "--", "echo", "test"]);
        assert_eq!(cli.chaos_memory_limit.as_deref(), Some("67108864"));
    }

    #[test]
    fn test_cli_chaos_memory_limit_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.chaos_memory_limit.is_none());
    }

    #[test]
    fn test_cli_chaos_cpu_limit() {
        let cli = Cli::parse_from(["renacer", "--chaos-cpu-limit", "0.5", "--", "echo", "test"]);
        assert!((cli.chaos_cpu_limit.expect("test") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cli_chaos_cpu_limit_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.chaos_cpu_limit.is_none());
    }

    #[test]
    fn test_cli_chaos_timeout() {
        let cli = Cli::parse_from(["renacer", "--chaos-timeout", "10s", "--", "echo", "test"]);
        assert_eq!(cli.chaos_timeout.as_deref(), Some("10s"));
    }

    #[test]
    fn test_cli_chaos_timeout_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.chaos_timeout.is_none());
    }

    #[test]
    fn test_cli_chaos_signals_flag() {
        let cli = Cli::parse_from(["renacer", "--chaos-signals", "--", "echo", "test"]);
        assert!(cli.chaos_signals);
    }

    #[test]
    fn test_cli_chaos_signals_default_false() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(!cli.chaos_signals);
    }

    #[test]
    fn test_cli_chaos_combined_with_tracing() {
        let cli = Cli::parse_from(["renacer", "-c", "--chaos", "aggressive", "--", "echo", "test"]);
        assert!(cli.statistics);
        assert_eq!(cli.chaos_preset.as_deref(), Some("aggressive"));
    }

    #[test]
    fn test_cli_chaos_preset_with_custom_memory() {
        let cli = Cli::parse_from([
            "renacer",
            "--chaos",
            "gentle",
            "--chaos-memory-limit",
            "128M",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.chaos_preset.as_deref(), Some("gentle"));
        assert_eq!(cli.chaos_memory_limit.as_deref(), Some("128M"));
    }

    #[test]
    fn test_cli_chaos_all_options() {
        let cli = Cli::parse_from([
            "renacer",
            "--chaos",
            "aggressive",
            "--chaos-memory-limit",
            "64M",
            "--chaos-cpu-limit",
            "0.25",
            "--chaos-timeout",
            "10s",
            "--chaos-signals",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.chaos_preset.as_deref(), Some("aggressive"));
        assert_eq!(cli.chaos_memory_limit.as_deref(), Some("64M"));
        assert!((cli.chaos_cpu_limit.expect("test") - 0.25).abs() < f64::EPSILON);
        assert_eq!(cli.chaos_timeout.as_deref(), Some("10s"));
        assert!(cli.chaos_signals);
    }

    // Sprint 48: Model Persistence CLI Tests
    #[test]
    fn test_cli_save_model_flag() {
        let cli = Cli::parse_from([
            "renacer",
            "--ml-anomaly",
            "--save-model",
            "baseline.apr",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.ml_anomaly);
        assert_eq!(cli.save_model.as_deref(), Some("baseline.apr"));
    }

    #[test]
    fn test_cli_save_model_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.save_model.is_none());
    }

    #[test]
    fn test_cli_load_model_flag() {
        let cli = Cli::parse_from([
            "renacer",
            "--ml-anomaly",
            "--load-model",
            "baseline.apr",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.ml_anomaly);
        assert_eq!(cli.load_model.as_deref(), Some("baseline.apr"));
    }

    #[test]
    fn test_cli_load_model_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.load_model.is_none());
    }

    #[test]
    fn test_cli_baseline_model_flag() {
        let cli = Cli::parse_from([
            "renacer",
            "--ml-anomaly",
            "--baseline",
            "golden.apr",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.ml_anomaly);
        assert_eq!(cli.baseline_model.as_deref(), Some("golden.apr"));
    }

    #[test]
    fn test_cli_baseline_model_default_none() {
        let cli = Cli::parse_from(["renacer", "--", "echo", "test"]);
        assert!(cli.baseline_model.is_none());
    }

    #[test]
    fn test_cli_save_and_load_model_combined() {
        // Can specify both (save new, load as starting point)
        let cli = Cli::parse_from([
            "renacer",
            "--ml-anomaly",
            "--load-model",
            "old.apr",
            "--save-model",
            "new.apr",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.load_model.as_deref(), Some("old.apr"));
        assert_eq!(cli.save_model.as_deref(), Some("new.apr"));
    }

    #[test]
    fn test_cli_model_with_statistics() {
        let cli = Cli::parse_from([
            "renacer",
            "-c",
            "--ml-anomaly",
            "--save-model",
            "model.apr",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.statistics);
        assert!(cli.ml_anomaly);
        assert_eq!(cli.save_model.as_deref(), Some("model.apr"));
    }

    #[test]
    fn test_cli_baseline_with_ml_outliers() {
        let cli = Cli::parse_from([
            "renacer",
            "--ml-outliers",
            "--baseline",
            "baseline.apr",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.ml_outliers);
        assert_eq!(cli.baseline_model.as_deref(), Some("baseline.apr"));
    }

    #[test]
    fn test_cli_model_path_with_directory() {
        let cli = Cli::parse_from([
            "renacer",
            "--ml-anomaly",
            "--save-model",
            "/tmp/models/baseline.apr",
            "--",
            "echo",
            "test",
        ]);
        assert_eq!(cli.save_model.as_deref(), Some("/tmp/models/baseline.apr"));
    }

    #[test]
    fn test_cli_all_model_options() {
        let cli = Cli::parse_from([
            "renacer",
            "-c",
            "--ml-anomaly",
            "--ml-outliers",
            "--load-model",
            "pretrained.apr",
            "--save-model",
            "updated.apr",
            "--baseline",
            "golden.apr",
            "--",
            "echo",
            "test",
        ]);
        assert!(cli.statistics);
        assert!(cli.ml_anomaly);
        assert!(cli.ml_outliers);
        assert_eq!(cli.load_model.as_deref(), Some("pretrained.apr"));
        assert_eq!(cli.save_model.as_deref(), Some("updated.apr"));
        assert_eq!(cli.baseline_model.as_deref(), Some("golden.apr"));
    }

    // Tests for helper functions

    #[test]
    fn test_run_tracer_both_pid_and_command_errors() {
        use crate::filter::SyscallFilter;
        let config = tracer::TracerConfig { filter: SyscallFilter::all(), ..Default::default() };
        let result = run_tracer(Some(1234), Some(vec!["echo".to_string()]), config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Cannot specify both"));
    }

    #[test]
    fn test_run_tracer_neither_pid_nor_command_errors() {
        use crate::filter::SyscallFilter;
        let config = tracer::TracerConfig { filter: SyscallFilter::all(), ..Default::default() };
        let result = run_tracer(None, None, config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Must specify either"));
    }

    #[test]
    fn test_run_validate_subcommand_no_baseline_or_generate() {
        let args = ValidateArgs {
            baseline: None,
            generate: None,
            tolerance: 10.0,
            strict: false,
            ignore_timing: false,
            fail_fast: false,
            output: ValidationOutputFormat::Text,
            command: vec!["echo".to_string()],
        };
        let code = run_validate_subcommand(&args);
        assert_eq!(code, 5); // ConfigError
    }

    #[test]
    fn test_run_validate_subcommand_baseline_not_found() {
        let args = ValidateArgs {
            baseline: Some(PathBuf::from("/nonexistent/baseline")),
            generate: None,
            tolerance: 10.0,
            strict: false,
            ignore_timing: false,
            fail_fast: false,
            output: ValidationOutputFormat::Text,
            command: vec!["echo".to_string()],
        };
        let code = run_validate_subcommand(&args);
        assert_eq!(code, 2); // BaselineNotFound
    }

    #[test]
    fn test_run_validate_subcommand_json_output() {
        let args = ValidateArgs {
            baseline: None,
            generate: None,
            tolerance: 10.0,
            strict: false,
            ignore_timing: false,
            fail_fast: false,
            output: ValidationOutputFormat::Json,
            command: vec!["echo".to_string()],
        };
        // Should still return ConfigError since no baseline/generate
        let code = run_validate_subcommand(&args);
        assert_eq!(code, 5);
    }

    #[test]
    fn test_run_validate_subcommand_junit_output() {
        let args = ValidateArgs {
            baseline: None,
            generate: None,
            tolerance: 10.0,
            strict: false,
            ignore_timing: false,
            fail_fast: false,
            output: ValidationOutputFormat::Junit,
            command: vec!["echo".to_string()],
        };
        let code = run_validate_subcommand(&args);
        assert_eq!(code, 5);
    }

    #[test]
    fn test_validate_args_defaults() {
        let cli = Cli::parse_from([
            "renacer",
            "validate",
            "--baseline",
            "/tmp/baseline",
            "--",
            "echo",
            "test",
        ]);
        if let Some(Commands::Validate(args)) = cli.subcommand {
            assert!((args.tolerance - 10.0).abs() < f32::EPSILON);
            assert!(!args.strict);
            assert!(!args.ignore_timing);
            assert!(!args.fail_fast);
            assert_eq!(args.output, ValidationOutputFormat::Text);
        } else {
            panic!("Expected Validate subcommand");
        }
    }

    #[test]
    fn test_validate_args_strict() {
        let cli = Cli::parse_from([
            "renacer",
            "validate",
            "--baseline",
            "/tmp/baseline",
            "--strict",
            "--",
            "echo",
        ]);
        if let Some(Commands::Validate(args)) = cli.subcommand {
            assert!(args.strict);
        } else {
            panic!("Expected Validate subcommand");
        }
    }

    #[test]
    fn test_validate_args_ignore_timing() {
        let cli = Cli::parse_from([
            "renacer",
            "validate",
            "--baseline",
            "/tmp/baseline",
            "--ignore-timing",
            "--",
            "echo",
        ]);
        if let Some(Commands::Validate(args)) = cli.subcommand {
            assert!(args.ignore_timing);
        } else {
            panic!("Expected Validate subcommand");
        }
    }

    #[test]
    fn test_validate_args_fail_fast() {
        let cli = Cli::parse_from([
            "renacer",
            "validate",
            "--baseline",
            "/tmp/baseline",
            "--fail-fast",
            "--",
            "echo",
        ]);
        if let Some(Commands::Validate(args)) = cli.subcommand {
            assert!(args.fail_fast);
        } else {
            panic!("Expected Validate subcommand");
        }
    }

    #[test]
    fn test_validate_args_custom_tolerance() {
        let cli = Cli::parse_from([
            "renacer",
            "validate",
            "--baseline",
            "/tmp/baseline",
            "--tolerance",
            "5.0",
            "--",
            "echo",
        ]);
        if let Some(Commands::Validate(args)) = cli.subcommand {
            assert!((args.tolerance - 5.0).abs() < f32::EPSILON);
        } else {
            panic!("Expected Validate subcommand");
        }
    }

    #[test]
    fn test_validate_args_generate() {
        let cli = Cli::parse_from([
            "renacer",
            "validate",
            "--generate",
            "/tmp/new-baseline",
            "--",
            "echo",
        ]);
        if let Some(Commands::Validate(args)) = cli.subcommand {
            assert!(args.baseline.is_none());
            assert_eq!(args.generate, Some(PathBuf::from("/tmp/new-baseline")));
        } else {
            panic!("Expected Validate subcommand");
        }
    }

    #[test]
    fn test_output_format_default() {
        let format = OutputFormat::default();
        assert!(matches!(format, OutputFormat::Text));
    }

    #[test]
    fn test_validation_output_format_eq() {
        assert_eq!(ValidationOutputFormat::Text, ValidationOutputFormat::Text);
        assert_ne!(ValidationOutputFormat::Text, ValidationOutputFormat::Json);
        assert_ne!(ValidationOutputFormat::Json, ValidationOutputFormat::Junit);
    }

    // Sprint 56: Metrics and Alerting CLI Tests

    /// Parse a visualize CLI invocation and extract VisualizeArgs
    fn parse_visualize(args: &[&str]) -> VisualizeArgs {
        let mut full: Vec<&str> = vec!["renacer", "visualize"];
        full.extend_from_slice(args);
        let cli = Cli::parse_from(full);
        match cli.subcommand {
            Some(Commands::Visualize(a)) => a,
            _ => panic!("Expected Visualize subcommand"),
        }
    }

    #[test]
    fn test_visualize_metrics_flag() {
        assert!(parse_visualize(&["--metrics", "--", "echo", "test"]).enable_metrics);
    }

    #[test]
    fn test_visualize_defaults() {
        let a = parse_visualize(&["--", "echo", "test"]);
        assert!(!a.enable_metrics);
        assert!(!a.enable_alerts);
        assert_eq!(a.alert_latency_threshold, 10000);
        assert!((a.alert_error_rate - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_visualize_alerts_flag() {
        assert!(parse_visualize(&["--alerts", "--", "echo", "test"]).enable_alerts);
    }

    #[test]
    fn test_visualize_alert_latency_threshold_custom() {
        let a = parse_visualize(&["--alert-latency-threshold", "5000", "--", "echo", "test"]);
        assert_eq!(a.alert_latency_threshold, 5000);
    }

    #[test]
    fn test_visualize_alert_error_rate_custom() {
        let a = parse_visualize(&["--alert-error-rate", "2.5", "--", "echo", "test"]);
        assert!((a.alert_error_rate - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_visualize_metrics_and_alerts_combined() {
        let a = parse_visualize(&[
            "--metrics",
            "--alerts",
            "--alert-latency-threshold",
            "8000",
            "--alert-error-rate",
            "3.0",
            "--",
            "echo",
            "test",
        ]);
        assert!(a.enable_metrics);
        assert!(a.enable_alerts);
        assert_eq!(a.alert_latency_threshold, 8000);
        assert!((a.alert_error_rate - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_visualize_no_pid_no_command_errors() {
        let args = parse_visualize(&[]);
        let err = run_visualize_subcommand(&args).unwrap_err();
        assert!(err.to_string().contains("Must specify either"));
    }

    #[test]
    fn test_visualize_both_pid_and_command_errors() {
        let mut args = parse_visualize(&["--", "echo"]);
        args.pid = Some(1234); // inject conflict: both pid AND command
        let err = run_visualize_subcommand(&args).unwrap_err();
        assert!(err.to_string().contains("Cannot specify both"));
    }

    #[test]
    fn test_visualize_empty_command_errors() {
        let mut args = parse_visualize(&[]);
        args.command = Some(vec![]); // empty vec treated same as None
        let err = run_visualize_subcommand(&args).unwrap_err();
        assert!(err.to_string().contains("Must specify either"));
    }
}
