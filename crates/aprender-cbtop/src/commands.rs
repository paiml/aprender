//! Command handlers for cbtop subcommands.

use cbtop::headless::{BenchmarkResult, HeadlessBenchmark};
use cbtop::optimize::{BaselineReport, OptimizationSuite, RegressionDetector};
use cbtop::CbtopError;

use crate::{parse_backend, parse_output_format, parse_workload, OptimizeAction};

pub(crate) fn run_headless(
    backend: &str,
    workload: &str,
    size: usize,
    duration: u64,
    format: &str,
    output: Option<std::path::PathBuf>,
) -> Result<(), CbtopError> {
    let benchmark = HeadlessBenchmark::new(
        parse_backend(backend),
        parse_workload(workload),
        size,
        std::time::Duration::from_secs(duration),
    );

    let result = benchmark.run()?;
    let output_format = parse_output_format(format);
    let output_str = result.format(output_format);

    if let Some(path) = output {
        std::fs::write(&path, &output_str).map_err(|e| CbtopError::Io(e.to_string()))?;
        eprintln!("Results written to: {}", path.display());
    } else {
        println!("{}", output_str);
    }

    Ok(())
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
        println!("{}", output_str);
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
    output_format: cbtop::headless::OutputFormat,
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
fn run_regression_check(
    result: &BenchmarkResult,
    baseline_path: &std::path::Path,
    fail_on_regression: f64,
    output_format: cbtop::headless::OutputFormat,
    output: Option<std::path::PathBuf>,
) -> Result<(), CbtopError> {
    let baseline_str =
        std::fs::read_to_string(baseline_path).map_err(|e| CbtopError::Io(e.to_string()))?;
    let baseline_result: BenchmarkResult =
        serde_json::from_str(&baseline_str).map_err(|e| CbtopError::Config(e.to_string()))?;

    let regression = result.check_regression(&baseline_result, fail_on_regression);
    let output_str = regression.format(output_format);
    write_output(&output_str, output.as_deref(), false)?;

    if regression.is_regression {
        std::process::exit(1);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_bench(
    backend: &str,
    workload: &str,
    size: usize,
    duration: u64,
    format: &str,
    output: Option<std::path::PathBuf>,
    baseline: Option<std::path::PathBuf>,
    fail_on_regression: f64,
    compare: Option<String>,
) -> Result<(), CbtopError> {
    let output_format = parse_output_format(format);

    // Handle comparison mode
    if let Some(backends_str) = compare {
        return run_comparison_bench(
            &backends_str,
            workload,
            size,
            duration,
            output_format,
            output,
        );
    }

    // Single benchmark
    let benchmark = create_benchmark(backend, workload, size, duration);
    let result = benchmark.run()?;

    // Check for regression if baseline provided
    if let Some(baseline_path) = baseline {
        return run_regression_check(
            &result,
            &baseline_path,
            fail_on_regression,
            output_format,
            output,
        );
    }

    // Output result
    let output_str = result.format(output_format);
    write_output(&output_str, output.as_deref(), true)
}

/// Run optimization subcommands (OPT-005)
pub(crate) fn run_optimize(action: OptimizeAction) -> Result<(), CbtopError> {
    match action {
        OptimizeAction::Baseline {
            output,
            quick,
            duration,
        } => run_optimize_baseline(output, quick, duration),
        OptimizeAction::Analyze {
            baseline,
            format,
            output,
        } => run_optimize_analyze(baseline, &format, output),
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

    // Ensure output directory exists
    if let Some(parent) = output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CbtopError::Io(format!("Failed to create directory: {}", e)))?;
        }
    }

    baseline.save(&output)?;
    eprintln!("Baseline saved to: {}", output.display());

    // Print summary
    eprintln!("\nBaseline Summary:");
    eprintln!("  Entries: {}", baseline.entries.len());

    let avg_gflops: f64 =
        baseline.entries.iter().map(|e| e.gflops).sum::<f64>() / baseline.entries.len() as f64;
    eprintln!("  Average GFLOP/s: {:.2}", avg_gflops);

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
            .map_err(|e| CbtopError::Config(format!("JSON serialization failed: {}", e)))?
    } else {
        analysis.format_report()
    };

    if let Some(path) = output {
        std::fs::write(&path, &report).map_err(|e| CbtopError::Io(e.to_string()))?;
        eprintln!("Analysis saved to: {}", path.display());
    } else {
        println!("{}", report);
    }

    // Print summary to stderr
    eprintln!("\nAnalysis Summary:");
    eprintln!("  Critical: {}", analysis.summary.critical_count);
    eprintln!("  Severe: {}", analysis.summary.severe_count);
    eprintln!("  Moderate: {}", analysis.summary.moderate_count);
    eprintln!("  Unstable: {}", analysis.summary.unstable_count);

    Ok(())
}

fn run_optimize_check(
    baseline_path: std::path::PathBuf,
    threshold: f64,
    quick: bool,
    format: &str,
) -> Result<(), CbtopError> {
    eprintln!("Checking for regressions (threshold: {}%)...", threshold);

    // Load baseline
    let baseline = BaselineReport::load(&baseline_path)?;

    // Collect current measurements
    let suite = if quick {
        OptimizationSuite::quick()
    } else {
        OptimizationSuite::standard()
    };
    let current = suite.collect_baseline()?;

    // Check for regressions
    let detector = RegressionDetector::new(baseline, threshold);
    let report = detector.check(&current);

    let output = if format == "json" {
        serde_json::to_string_pretty(&report)
            .map_err(|e| CbtopError::Config(format!("JSON serialization failed: {}", e)))?
    } else {
        report.format_report()
    };

    println!("{}", output);

    // Exit with appropriate code
    std::process::exit(report.exit_code());
}
