//! Command surface for `apr compute` — load-testing, monitoring and
//! benchmarking the compute backends (SIMD, wgpu, CUDA).
//!
//! This module is the merger of the standalone `aprender-cbtop` binary's
//! `src/main.rs` (argument definitions) and its bin-local `src/commands.rs`
//! (handlers). That binary is gone (APR-MONO: one installed binary, `apr`).
//!
//! ## Why `apr compute` and not `apr cbtop`
//!
//! `apr cbtop` already exists and is a *different* tool: "ComputeBrick Top",
//! which profiles the brick pipeline of an LLM inference run against a model
//! file. This crate is "Compute Block Top", which load-tests and monitors the
//! compute *backends* themselves and shares none of that argument surface.
//! Two tools with the same acronym cannot share one subcommand without one of
//! them losing arguments, so this one takes the namespace that names what it
//! actually operates on.
//!
//! [`ComputeCommand`] is embedded directly by `apr-cli` with
//! `#[command(subcommand)]`, so there is exactly one definition of the
//! argument surface and no second copy that can drift (defect class #2418).

use clap::Subcommand;

use crate::config::{ComputeBackend, LoadProfile, WorkloadType};
use crate::headless::{BenchmarkResult, HeadlessBenchmark, OutputFormat};
use crate::optimize::{BaselineReport, OptimizationSuite, RegressionDetector};
use crate::{CbtopApp, CbtopError, Config};

/// `apr compute` subcommands.
///
/// `Clone` so apr's dispatch tree, which hands out shared references, can
/// pass an owned command to [`run`].
#[derive(Debug, Clone, Subcommand)]
pub enum ComputeCommand {
    /// Real-time load-testing and hardware-monitoring TUI (`--headless` for
    /// CI/CD and agents).
    ///
    /// This is the standalone `cbtop` binary's no-subcommand mode; every flag
    /// it accepted is accepted here with the same name, short form and default.
    Top {
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
    },

    /// Run a backend benchmark in headless mode.
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

    /// Optimization identification and regression detection.
    Optimize {
        /// Which optimization step to run.
        #[command(subcommand)]
        action: OptimizeAction,
    },
}

/// Actions for `apr compute optimize`.
#[derive(Debug, Clone, Subcommand)]
pub enum OptimizeAction {
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

/// Parse a `--backend` string. Unrecognised values fall back to `All`, which
/// is the standalone binary's behaviour.
#[must_use]
pub fn parse_backend(s: &str) -> ComputeBackend {
    match s.to_lowercase().as_str() {
        "simd" => ComputeBackend::Simd,
        "wgpu" => ComputeBackend::Wgpu,
        "cuda" => ComputeBackend::Cuda,
        _ => ComputeBackend::All,
    }
}

/// Parse a `--load` string. Unrecognised values fall back to `Idle`.
#[must_use]
pub fn parse_load_profile(s: &str) -> LoadProfile {
    match s.to_lowercase().as_str() {
        "light" => LoadProfile::Light,
        "medium" => LoadProfile::Medium,
        "heavy" => LoadProfile::Heavy,
        "stress" => LoadProfile::Stress,
        _ => LoadProfile::Idle,
    }
}

/// Parse a `--workload` string. Unrecognised values fall back to `Gemm`.
#[must_use]
pub fn parse_workload(s: &str) -> WorkloadType {
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

/// Parse a `--format` string. Anything other than `json` is text.
#[must_use]
pub fn parse_output_format(s: &str) -> OutputFormat {
    match s.to_lowercase().as_str() {
        "json" => OutputFormat::Json,
        _ => OutputFormat::Text,
    }
}

/// Run one `apr compute` command, returning the exit code it earns.
///
/// `0` means success. Non-zero codes are returned rather than passed to
/// `std::process::exit` so that the regression gates stay testable in-process;
/// the caller decides how to surface them.
///
/// # Errors
///
/// Propagates [`CbtopError`] from the benchmark, baseline, or TUI paths.
pub fn run(command: ComputeCommand) -> Result<u8, CbtopError> {
    match command {
        ComputeCommand::Bench {
            backend,
            workload,
            size,
            duration,
            format,
            output,
            baseline,
            fail_on_regression,
            compare,
        } => run_bench(
            &backend,
            &workload,
            size,
            duration,
            &format,
            output,
            baseline,
            fail_on_regression,
            compare,
        ),
        ComputeCommand::Optimize { action } => run_optimize(action),
        ComputeCommand::Top {
            refresh,
            device,
            backend,
            load,
            workload,
            size,
            threads,
            deterministic,
            show_fps,
            config,
            headless,
            format,
            duration,
            output,
        } => {
            if headless {
                return run_headless(&backend, &workload, size, duration, &format, output)
                    .map(|()| 0);
            }
            let config = Config {
                refresh_ms: refresh,
                device_index: device,
                backend: parse_backend(&backend),
                load_profile: parse_load_profile(&load),
                workload: parse_workload(&workload),
                problem_size: size,
                threads: threads.unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(std::num::NonZeroUsize::get)
                        .unwrap_or(1)
                }),
                deterministic,
                show_fps,
                config_path: config,
            };
            let mut app = CbtopApp::new(config)?;
            app.run().map(|()| 0)
        }
    }
}

/// Run a single headless benchmark and write its report.
///
/// # Errors
///
/// Propagates [`CbtopError`] from the benchmark or from writing `output`.
pub fn run_headless(
    backend: &str,
    workload: &str,
    size: usize,
    duration: u64,
    format: &str,
    output: Option<std::path::PathBuf>,
) -> Result<(), CbtopError> {
    let benchmark = create_benchmark(backend, workload, size, duration);
    let result = benchmark.run()?;
    let output_str = result.format(parse_output_format(format));
    write_output(&output_str, output.as_deref(), true)
}

/// Write output string to a file or stdout.
fn write_output(
    output_str: &str,
    path: Option<&std::path::Path>,
    log_destination: bool,
) -> Result<(), CbtopError> {
    if let Some(p) = path {
        std::fs::write(p, output_str).map_err(|e| CbtopError::Io(e.to_string()))?;
        if log_destination {
            eprintln!("Results written to: {}", p.display());
        }
    } else {
        println!("{output_str}");
    }
    Ok(())
}

/// Create a `HeadlessBenchmark` from parsed string parameters.
fn create_benchmark(
    backend: &str,
    workload: &str,
    size: usize,
    duration: u64,
) -> HeadlessBenchmark {
    HeadlessBenchmark::new(
        parse_backend(backend),
        parse_workload(workload),
        size,
        std::time::Duration::from_secs(duration),
    )
}

/// Run comparison mode: benchmark multiple backends and output comparison.
fn run_comparison_bench(
    backends_str: &str,
    workload: &str,
    size: usize,
    duration: u64,
    output_format: OutputFormat,
    output: Option<std::path::PathBuf>,
) -> Result<(), CbtopError> {
    let backends: Vec<&str> = backends_str.split(',').collect();
    let mut results = Vec::new();

    for b in backends {
        let benchmark = create_benchmark(b.trim(), workload, size, duration);
        let result = benchmark.run()?;
        results.push((b.trim().to_string(), result));
    }

    let comparison = BenchmarkResult::compare(&results);
    let output_str = comparison.format(output_format);
    write_output(&output_str, output.as_deref(), false)
}

/// Run a single benchmark and check for regression against a baseline file.
///
/// Returns `1` when a regression is detected — the standalone binary called
/// `std::process::exit(1)` here, which made the gate untestable in-process.
fn run_regression_check(
    result: &BenchmarkResult,
    baseline_path: &std::path::Path,
    fail_on_regression: f64,
    output_format: OutputFormat,
    output: Option<std::path::PathBuf>,
) -> Result<u8, CbtopError> {
    let baseline_str =
        std::fs::read_to_string(baseline_path).map_err(|e| CbtopError::Io(e.to_string()))?;
    let baseline_result: BenchmarkResult =
        serde_json::from_str(&baseline_str).map_err(|e| CbtopError::Config(e.to_string()))?;

    let regression = result.check_regression(&baseline_result, fail_on_regression);
    let output_str = regression.format(output_format);
    write_output(&output_str, output.as_deref(), false)?;

    Ok(u8::from(regression.is_regression))
}

/// `apr compute bench` implementation.
///
/// # Errors
///
/// Propagates [`CbtopError`] from the benchmark, the baseline file, or output.
#[allow(clippy::too_many_arguments)]
pub fn run_bench(
    backend: &str,
    workload: &str,
    size: usize,
    duration: u64,
    format: &str,
    output: Option<std::path::PathBuf>,
    baseline: Option<std::path::PathBuf>,
    fail_on_regression: f64,
    compare: Option<String>,
) -> Result<u8, CbtopError> {
    let output_format = parse_output_format(format);

    if let Some(backends_str) = compare {
        return run_comparison_bench(
            &backends_str,
            workload,
            size,
            duration,
            output_format,
            output,
        )
        .map(|()| 0);
    }

    let benchmark = create_benchmark(backend, workload, size, duration);
    let result = benchmark.run()?;

    if let Some(baseline_path) = baseline {
        return run_regression_check(
            &result,
            &baseline_path,
            fail_on_regression,
            output_format,
            output,
        );
    }

    let output_str = result.format(output_format);
    write_output(&output_str, output.as_deref(), true).map(|()| 0)
}

/// `apr compute optimize` implementation (OPT-005).
///
/// # Errors
///
/// Propagates [`CbtopError`] from the suite, the baseline file, or output.
pub fn run_optimize(action: OptimizeAction) -> Result<u8, CbtopError> {
    match action {
        OptimizeAction::Baseline {
            output,
            quick,
            duration,
        } => run_optimize_baseline(output, quick, duration).map(|()| 0),
        OptimizeAction::Analyze {
            baseline,
            format,
            output,
        } => run_optimize_analyze(baseline, &format, output).map(|()| 0),
        OptimizeAction::Check {
            baseline,
            threshold,
            quick,
            format,
        } => run_optimize_check(baseline, threshold, quick, &format),
    }
}

fn run_optimize_baseline(
    output: std::path::PathBuf,
    quick: bool,
    duration: u64,
) -> Result<(), CbtopError> {
    eprintln!("Collecting baseline measurements...");

    let mut suite = if quick {
        OptimizationSuite::quick()
    } else {
        OptimizationSuite::standard()
    };
    suite.duration = std::time::Duration::from_secs(duration);

    let total_configs = suite.workloads.len() * suite.sizes.len() * suite.backends.len();
    eprintln!(
        "Running {} configurations ({} workloads x {} sizes x {} backends)",
        total_configs,
        suite.workloads.len(),
        suite.sizes.len(),
        suite.backends.len()
    );

    let baseline = suite.collect_baseline()?;

    if let Some(parent) = output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CbtopError::Io(format!("Failed to create directory: {e}")))?;
        }
    }

    baseline.save(&output)?;
    eprintln!("Baseline saved to: {}", output.display());

    eprintln!("\nBaseline Summary:");
    eprintln!("  Entries: {}", baseline.entries.len());

    let avg_gflops: f64 =
        baseline.entries.iter().map(|e| e.gflops).sum::<f64>() / baseline.entries.len() as f64;
    eprintln!("  Average GFLOP/s: {avg_gflops:.2}");

    let avg_efficiency: f64 =
        baseline.entries.iter().map(|e| e.efficiency).sum::<f64>() / baseline.entries.len() as f64;
    eprintln!("  Average Efficiency: {:.1}%", avg_efficiency * 100.0);

    Ok(())
}

fn run_optimize_analyze(
    baseline_path: std::path::PathBuf,
    format: &str,
    output: Option<std::path::PathBuf>,
) -> Result<(), CbtopError> {
    let baseline = BaselineReport::load(&baseline_path)?;
    let suite = OptimizationSuite::standard();
    let analysis = suite.analyze_bottlenecks(&baseline);

    let report = if format == "json" {
        serde_json::to_string_pretty(&analysis)
            .map_err(|e| CbtopError::Config(format!("JSON serialization failed: {e}")))?
    } else {
        analysis.format_report()
    };

    if let Some(path) = output {
        std::fs::write(&path, &report).map_err(|e| CbtopError::Io(e.to_string()))?;
        eprintln!("Analysis saved to: {}", path.display());
    } else {
        println!("{report}");
    }

    eprintln!("\nAnalysis Summary:");
    eprintln!("  Critical: {}", analysis.summary.critical_count);
    eprintln!("  Severe: {}", analysis.summary.severe_count);
    eprintln!("  Moderate: {}", analysis.summary.moderate_count);
    eprintln!("  Unstable: {}", analysis.summary.unstable_count);

    Ok(())
}

/// Returns the regression report's exit code instead of calling
/// `std::process::exit`, so the gate can be asserted on in-process.
fn run_optimize_check(
    baseline_path: std::path::PathBuf,
    threshold: f64,
    quick: bool,
    format: &str,
) -> Result<u8, CbtopError> {
    eprintln!("Checking for regressions (threshold: {threshold}%)...");

    let baseline = BaselineReport::load(&baseline_path)?;

    let suite = if quick {
        OptimizationSuite::quick()
    } else {
        OptimizationSuite::standard()
    };
    let current = suite.collect_baseline()?;

    let detector = RegressionDetector::new(baseline, threshold);
    let report = detector.check(&current);

    let output = if format == "json" {
        serde_json::to_string_pretty(&report)
            .map_err(|e| CbtopError::Config(format!("JSON serialization failed: {e}")))?
    } else {
        report.format_report()
    };

    println!("{output}");

    Ok(u8::try_from(report.exit_code()).unwrap_or(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    #[derive(Debug, clap::Parser)]
    #[command(name = "compute")]
    struct Harness {
        #[command(subcommand)]
        command: ComputeCommand,
    }

    fn parse(argv: &[&str]) -> ComputeCommand {
        Harness::try_parse_from(argv)
            .unwrap_or_else(|e| panic!("argv {argv:?} must parse: {e}"))
            .command
    }

    /// `apr compute top` must accept every flag the standalone `cbtop`
    /// no-subcommand mode accepted, by the same long AND short name.
    #[test]
    fn top_accepts_every_flag_of_the_standalone_no_subcommand_mode() {
        let ComputeCommand::Top {
            refresh,
            device,
            backend,
            load,
            workload,
            size,
            threads,
            deterministic,
            show_fps,
            config,
            headless,
            format,
            duration,
            output,
        } = parse(&[
            "compute",
            "top",
            "-r",
            "250",
            "-d",
            "1",
            "-b",
            "cuda",
            "-l",
            "stress",
            "-w",
            "attention",
            "-s",
            "4096",
            "-t",
            "12",
            "--deterministic",
            "--show-fps",
            "-c",
            "cbtop.toml",
            "--headless",
            "--format",
            "json",
            "--duration",
            "9",
            "-o",
            "out.json",
        ])
        else {
            panic!("expected Top");
        };
        assert_eq!(refresh, 250);
        assert_eq!(device, 1);
        assert_eq!(backend, "cuda");
        assert_eq!(load, "stress");
        assert_eq!(workload, "attention");
        assert_eq!(size, 4096);
        assert_eq!(threads, Some(12));
        assert!(deterministic);
        assert!(show_fps);
        assert_eq!(config, Some(std::path::PathBuf::from("cbtop.toml")));
        assert!(headless);
        assert_eq!(format, "json");
        assert_eq!(duration, 9);
        assert_eq!(output, Some(std::path::PathBuf::from("out.json")));
    }

    /// Defaults must match the standalone binary's exactly.
    #[test]
    fn top_defaults_match_the_standalone_binary() {
        let ComputeCommand::Top {
            refresh,
            device,
            backend,
            load,
            workload,
            size,
            threads,
            deterministic,
            show_fps,
            config,
            headless,
            format,
            duration,
            output,
        } = parse(&["compute", "top"])
        else {
            panic!("expected Top");
        };
        assert_eq!(refresh, 100);
        assert_eq!(device, 0);
        assert_eq!(backend, "all");
        assert_eq!(load, "idle");
        assert_eq!(workload, "gemm");
        assert_eq!(size, 1_048_576);
        assert_eq!(threads, None);
        assert!(!deterministic);
        assert!(!show_fps);
        assert_eq!(config, None);
        assert!(!headless);
        assert_eq!(format, "text");
        assert_eq!(duration, 5);
        assert_eq!(output, None);
    }

    /// `bench` keeps all nine of its arguments, including the three that only
    /// matter in regression mode.
    #[test]
    fn bench_accepts_every_flag_including_the_regression_trio() {
        let ComputeCommand::Bench {
            backend,
            workload,
            size,
            duration,
            format,
            output,
            baseline,
            fail_on_regression,
            compare,
        } = parse(&[
            "compute",
            "bench",
            "-b",
            "wgpu",
            "-w",
            "reduction",
            "-s",
            "2048",
            "-d",
            "11",
            "-f",
            "text",
            "-o",
            "r.txt",
            "--baseline",
            "b.json",
            "--fail-on-regression",
            "2.5",
            "--compare",
            "simd,cuda",
        ])
        else {
            panic!("expected Bench");
        };
        assert_eq!(backend, "wgpu");
        assert_eq!(workload, "reduction");
        assert_eq!(size, 2048);
        assert_eq!(duration, 11);
        assert_eq!(format, "text");
        assert_eq!(output, Some(std::path::PathBuf::from("r.txt")));
        assert_eq!(baseline, Some(std::path::PathBuf::from("b.json")));
        assert_eq!(fail_on_regression, 2.5);
        assert_eq!(compare.as_deref(), Some("simd,cuda"));
    }

    /// `bench` defaults differ from `top`'s on purpose (`simd`, `json`); pin
    /// both so neither silently inherits the other's.
    #[test]
    fn bench_defaults_match_the_standalone_binary() {
        let ComputeCommand::Bench {
            backend,
            format,
            fail_on_regression,
            duration,
            size,
            workload,
            ..
        } = parse(&["compute", "bench"])
        else {
            panic!("expected Bench");
        };
        assert_eq!(backend, "simd");
        assert_eq!(format, "json");
        assert_eq!(fail_on_regression, 5.0);
        assert_eq!(duration, 5);
        assert_eq!(size, 1_048_576);
        assert_eq!(workload, "gemm");
    }

    /// All three `optimize` actions survive, with their defaults.
    #[test]
    fn every_optimize_action_is_still_reachable() {
        let ComputeCommand::Optimize {
            action:
                OptimizeAction::Baseline {
                    output,
                    quick,
                    duration,
                },
        } = parse(&["compute", "optimize", "baseline"])
        else {
            panic!("expected optimize baseline");
        };
        assert_eq!(output, std::path::PathBuf::from("benchmarks/baseline.json"));
        assert!(!quick);
        assert_eq!(duration, 3);

        let ComputeCommand::Optimize {
            action: OptimizeAction::Analyze {
                baseline, format, ..
            },
        } = parse(&["compute", "optimize", "analyze"])
        else {
            panic!("expected optimize analyze");
        };
        assert_eq!(
            baseline,
            std::path::PathBuf::from("benchmarks/baseline.json")
        );
        assert_eq!(format, "text");

        let ComputeCommand::Optimize {
            action:
                OptimizeAction::Check {
                    threshold,
                    quick,
                    format,
                    baseline,
                },
        } = parse(&[
            "compute", "optimize", "check", "-t", "9.5", "--quick", "-f", "json",
        ])
        else {
            panic!("expected optimize check");
        };
        assert_eq!(threshold, 9.5);
        assert!(quick);
        assert_eq!(format, "json");
        assert_eq!(
            baseline,
            std::path::PathBuf::from("benchmarks/baseline.json")
        );
    }

    /// A non-numeric `--size` must be refused, not silently defaulted.
    #[test]
    fn non_numeric_size_is_refused() {
        let err = Harness::try_parse_from(["compute", "bench", "--size", "big"])
            .expect_err("--size must reject non-numeric input");
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    /// An unknown subcommand must be refused rather than falling through to
    /// the TUI.
    #[test]
    fn unknown_subcommand_is_refused() {
        let err = Harness::try_parse_from(["compute", "no-such-thing"])
            .expect_err("an unknown subcommand must be refused");
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn backend_strings_map_to_every_backend_variant() {
        assert_eq!(parse_backend("simd"), ComputeBackend::Simd);
        assert_eq!(parse_backend("WGPU"), ComputeBackend::Wgpu);
        assert_eq!(parse_backend("cuda"), ComputeBackend::Cuda);
        assert_eq!(parse_backend("all"), ComputeBackend::All);
        // Unrecognised input falls back to All, as the standalone binary did.
        assert_eq!(parse_backend("nonsense"), ComputeBackend::All);
    }

    #[test]
    fn workload_strings_map_to_every_workload_variant() {
        assert_eq!(parse_workload("conv"), WorkloadType::Conv2d);
        assert_eq!(parse_workload("conv2d"), WorkloadType::Conv2d);
        assert_eq!(parse_workload("attention"), WorkloadType::Attention);
        assert_eq!(parse_workload("bandwidth"), WorkloadType::Bandwidth);
        assert_eq!(parse_workload("elementwise"), WorkloadType::Elementwise);
        assert_eq!(parse_workload("reduction"), WorkloadType::Reduction);
        assert_eq!(parse_workload("all"), WorkloadType::All);
        assert_eq!(parse_workload("gemm"), WorkloadType::Gemm);
        assert_eq!(parse_workload("nonsense"), WorkloadType::Gemm);
    }

    #[test]
    fn load_profile_strings_map_to_every_profile_variant() {
        assert_eq!(parse_load_profile("light"), LoadProfile::Light);
        assert_eq!(parse_load_profile("medium"), LoadProfile::Medium);
        assert_eq!(parse_load_profile("heavy"), LoadProfile::Heavy);
        assert_eq!(parse_load_profile("stress"), LoadProfile::Stress);
        assert_eq!(parse_load_profile("idle"), LoadProfile::Idle);
        assert_eq!(parse_load_profile("nonsense"), LoadProfile::Idle);
    }

    #[test]
    fn output_format_strings_map_to_both_variants() {
        assert_eq!(parse_output_format("json"), OutputFormat::Json);
        assert_eq!(parse_output_format("JSON"), OutputFormat::Json);
        assert_eq!(parse_output_format("text"), OutputFormat::Text);
        assert_eq!(parse_output_format("nonsense"), OutputFormat::Text);
    }

    /// `bench --baseline` pointing at a missing file must be REFUSED, not
    /// treated as "no baseline, therefore no regression" — that would turn the
    /// CI gate green precisely when its input went missing.
    #[test]
    fn bench_with_a_missing_baseline_file_is_refused() {
        let outcome = run_bench(
            "simd",
            "gemm",
            64,
            0,
            "json",
            None,
            Some(std::path::PathBuf::from(
                "/nonexistent/apr-compute-baseline.json",
            )),
            5.0,
            None,
        );
        assert!(
            matches!(outcome, Err(CbtopError::Io(_))),
            "a missing --baseline must be an Io error, got {outcome:?}"
        );
    }
}
