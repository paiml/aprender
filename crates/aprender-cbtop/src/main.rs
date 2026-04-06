//! cbtop - Compute Block Top
//!
//! Real-time load testing and hardware monitoring TUI.
//!
//! Run with: cargo run -p cbtop
//!
//! Headless mode for CI/CD and AI agents:
//!   cbtop --headless --format json --duration 5

mod commands;

use cbtop::config::{ComputeBackend, LoadProfile, WorkloadType};
use cbtop::headless::OutputFormat;
use cbtop::{CbtopApp, CbtopError, Config};
use clap::{Parser, Subcommand};

/// Compute Block Top - Real-time load testing and hardware monitoring TUI
#[derive(Parser, Debug)]
#[command(name = "cbtop")]
#[command(author = "Trueno Engineering")]
#[command(version)]
#[command(about = "Real-time load testing and hardware monitoring TUI", long_about = None)]
struct Cli {
    /// Subcommand (bench for headless benchmarking)
    #[command(subcommand)]
    command: Option<Commands>,

    /// Refresh rate in milliseconds
    #[arg(short, long, default_value = "100")]
    refresh: u64,

    /// GPU device index
    #[arg(short, long, default_value = "0")]
    device: u32,

    /// Compute backend: simd, wgpu, cuda, all
    #[arg(short, long, default_value = "all")]
    backend: String,

    /// Load profile: idle, light, medium, heavy, stress
    #[arg(short, long, default_value = "idle")]
    load: String,

    /// Workload type: gemm, conv, attention, bandwidth, elementwise, reduction, all
    #[arg(short, long, default_value = "gemm")]
    workload: String,

    /// Problem size in elements
    #[arg(short, long, default_value = "1048576")]
    size: usize,

    /// Thread count for SIMD
    #[arg(short, long)]
    threads: Option<usize>,

    /// Enable deterministic mode for testing
    #[arg(long)]
    deterministic: bool,

    /// Show frame timing statistics
    #[arg(long)]
    show_fps: bool,

    /// Config file path
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,

    /// Run in headless mode (no TUI, for CI/CD and AI agents)
    #[arg(long)]
    headless: bool,

    /// Output format for headless mode: json, text
    #[arg(long, default_value = "text")]
    format: String,

    /// Benchmark duration in seconds (headless mode)
    #[arg(long, default_value = "5")]
    duration: u64,

    /// Output file path (headless mode)
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run benchmark in headless mode
    Bench {
        /// Compute backend: simd, wgpu, cuda, all
        #[arg(short, long, default_value = "simd")]
        backend: String,

        /// Workload type: gemm, dot, elementwise, reduction
        #[arg(short, long, default_value = "gemm")]
        workload: String,

        /// Problem size in elements
        #[arg(short, long, default_value = "1048576")]
        size: usize,

        /// Benchmark duration in seconds
        #[arg(short, long, default_value = "5")]
        duration: u64,

        /// Output format: json, text
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,

        /// Baseline file for regression comparison
        #[arg(long)]
        baseline: Option<std::path::PathBuf>,

        /// Fail if regression exceeds this percentage
        #[arg(long, default_value = "5.0")]
        fail_on_regression: f64,

        /// Compare multiple backends (comma-separated)
        #[arg(long)]
        compare: Option<String>,
    },

    /// Optimization identification and regression detection
    Optimize {
        #[command(subcommand)]
        action: OptimizeAction,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum OptimizeAction {
    /// Collect baseline measurements for all configurations
    Baseline {
        /// Output file for baseline JSON
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        output: std::path::PathBuf,

        /// Use quick mode (fewer configurations, shorter duration)
        #[arg(long)]
        quick: bool,

        /// Duration per benchmark in seconds
        #[arg(short, long, default_value = "3")]
        duration: u64,
    },

    /// Analyze baseline for performance bottlenecks
    Analyze {
        /// Baseline file to analyze
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        baseline: std::path::PathBuf,

        /// Output format: text, json
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },

    /// Check for performance regressions against baseline
    Check {
        /// Baseline file to compare against
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        baseline: std::path::PathBuf,

        /// Regression threshold percentage
        #[arg(short, long, default_value = "5.0")]
        threshold: f64,

        /// Use quick mode for current measurements
        #[arg(long)]
        quick: bool,

        /// Output format: text, json
        #[arg(short, long, default_value = "text")]
        format: String,
    },
}

fn parse_backend(s: &str) -> ComputeBackend {
    match s.to_lowercase().as_str() {
        "simd" => ComputeBackend::Simd,
        "wgpu" => ComputeBackend::Wgpu,
        "cuda" => ComputeBackend::Cuda,
        _ => ComputeBackend::All,
    }
}

fn parse_load_profile(s: &str) -> LoadProfile {
    match s.to_lowercase().as_str() {
        "light" => LoadProfile::Light,
        "medium" => LoadProfile::Medium,
        "heavy" => LoadProfile::Heavy,
        "stress" => LoadProfile::Stress,
        _ => LoadProfile::Idle,
    }
}

fn parse_workload(s: &str) -> WorkloadType {
    match s.to_lowercase().as_str() {
        "conv" | "conv2d" => WorkloadType::Conv2d,
        "attention" => WorkloadType::Attention,
        "bandwidth" => WorkloadType::Bandwidth,
        "elementwise" => WorkloadType::Elementwise,
        "reduction" => WorkloadType::Reduction,
        "all" => WorkloadType::All,
        _ => WorkloadType::Gemm,
    }
}

fn parse_output_format(s: &str) -> OutputFormat {
    match s.to_lowercase().as_str() {
        "json" => OutputFormat::Json,
        _ => OutputFormat::Text,
    }
}

fn main() -> Result<(), CbtopError> {
    let cli = Cli::parse();

    // Handle subcommands
    match cli.command {
        Some(Commands::Bench {
            backend,
            workload,
            size,
            duration,
            format,
            output,
            baseline,
            fail_on_regression,
            compare,
        }) => {
            return commands::run_bench(
                &backend,
                &workload,
                size,
                duration,
                &format,
                output,
                baseline,
                fail_on_regression,
                compare,
            );
        }
        Some(Commands::Optimize { action }) => {
            return commands::run_optimize(action);
        }
        None => {}
    }

    // Handle headless mode
    if cli.headless {
        return commands::run_headless(
            &cli.backend,
            &cli.workload,
            cli.size,
            cli.duration,
            &cli.format,
            cli.output,
        );
    }

    // Normal TUI mode
    let config = Config {
        refresh_ms: cli.refresh,
        device_index: cli.device,
        backend: parse_backend(&cli.backend),
        load_profile: parse_load_profile(&cli.load),
        workload: parse_workload(&cli.workload),
        problem_size: cli.size,
        threads: cli.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }),
        deterministic: cli.deterministic,
        show_fps: cli.show_fps,
        config_path: cli.config,
    };

    let mut app = CbtopApp::new(config)?;
    app.run()
}
