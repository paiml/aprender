//! cbtop-bench: Headless benchmarking tool for cbtop widgets.
//!
//! This tool enables automated performance testing, CI/CD integration,
//! and deterministic output capture without requiring a terminal display.
//!
//! # Usage
//!
//! ```bash
//! # Basic benchmark
//! cbtop-bench --widget cpu-grid --width 80 --height 24 --frames 1000
//!
//! # Deterministic mode for CI
//! cbtop-bench --deterministic --output metrics.json
//!
//! # Full benchmark suite
//! cbtop-bench suite --all --output results/
//! ```

use clap::{Parser, Subcommand};
use presentar_core::Rect;
use presentar_terminal::tools::bench::{
    BenchmarkHarness, BenchmarkResult, DeterministicContext, PerformanceTargets, RenderMetrics,
};
use presentar_terminal::{
    BrailleGraph, CpuGrid, GraphMode, MemoryBar, MemorySegment, ProcessTable,
};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Headless benchmarking tool for cbtop widgets.
#[derive(Parser)]
#[command(
    name = "cbtop-bench",
    version,
    about = "Headless benchmarking for cbtop widgets"
)]
#[allow(clippy::struct_excessive_bools)]
struct Cli {
    /// Widget to benchmark.
    #[arg(long)]
    widget: Option<String>,

    /// Terminal width.
    #[arg(long, default_value = "80")]
    width: u16,

    /// Terminal height.
    #[arg(long, default_value = "24")]
    height: u16,

    /// Number of benchmark frames.
    #[arg(long, default_value = "1000")]
    frames: u32,

    /// Warmup frames.
    #[arg(long, default_value = "100")]
    warmup: u32,

    /// Enable deterministic mode.
    #[arg(long)]
    deterministic: bool,

    /// Output file (JSON metrics).
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Output format (json, csv, text).
    #[arg(long, default_value = "text")]
    format: String,

    /// Validate against performance targets.
    #[arg(long)]
    validate: bool,

    /// Use strict performance targets.
    #[arg(long)]
    strict: bool,

    /// Verbose output.
    #[arg(long, short)]
    verbose: bool,

    /// Subcommand.
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run full benchmark suite.
    Suite {
        /// Run all widgets.
        #[arg(long)]
        all: bool,

        /// Output directory.
        #[arg(long)]
        output: PathBuf,
    },
    /// Compare two widgets.
    Compare {
        /// First widget.
        #[arg(long)]
        widget_a: String,

        /// Second widget.
        #[arg(long)]
        widget_b: String,
    },
    /// Generate snapshot for pixel-perfect testing.
    Snapshot {
        /// Widget to snapshot.
        #[arg(long)]
        widget: String,

        /// Output file.
        #[arg(long)]
        output: PathBuf,
    },
    /// Validate results against targets.
    Validate {
        /// Results JSON file.
        #[arg(long)]
        results: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Suite { all, output }) => {
            run_suite(*all, output, &cli);
        }
        Some(Commands::Compare { widget_a, widget_b }) => {
            run_compare(widget_a, widget_b, &cli);
        }
        Some(Commands::Snapshot { widget, output }) => {
            run_snapshot(widget, output, &cli);
        }
        Some(Commands::Validate { results }) => {
            run_validate(results, &cli);
        }
        None => {
            if let Some(ref widget) = cli.widget {
                run_single_benchmark(widget, &cli);
            } else {
                // Default: run all widgets
                run_suite(true, &PathBuf::from("bench_results"), &cli);
            }
        }
    }
}

fn run_single_benchmark(widget_name: &str, cli: &Cli) {
    let start = Instant::now();

    if cli.verbose {
        eprintln!("Benchmarking widget: {widget_name}");
        let (width, height, warmup, frames) = (cli.width, cli.height, cli.warmup, cli.frames);
        eprintln!("  Dimensions: {width}x{height}, Warmup: {warmup}, Frames: {frames}");
    }

    let result = benchmark_widget(widget_name, cli);

    if cli.verbose {
        eprintln!("  Completed in {:?}", start.elapsed());
    }

    output_result(&result, cli);

    if cli.validate {
        validate_result(&result, cli);
    }
}

fn benchmark_widget(widget_name: &str, cli: &Cli) -> BenchmarkResult {
    let mut harness = BenchmarkHarness::new(cli.width, cli.height)
        .with_frames(cli.warmup, cli.frames)
        .with_deterministic(cli.deterministic);

    let bounds = Rect::new(0.0, 0.0, f32::from(cli.width), f32::from(cli.height));

    match widget_name.to_lowercase().as_str() {
        "cpu-grid" | "cpugrid" => {
            let ctx = DeterministicContext::new();
            let mut widget = CpuGrid::new(ctx.cpu_usage).with_columns(8).compact();
            harness.benchmark(&mut widget, bounds)
        }
        "braille-graph" | "braillegraph" | "graph" => {
            let data: Vec<f64> = (0..60)
                .map(|i| (f64::from(i) * 0.1).sin().mul_add(50.0, 50.0))
                .collect();
            let mut widget = BrailleGraph::new(data).with_mode(GraphMode::Braille);
            harness.benchmark(&mut widget, bounds)
        }
        "memory-bar" | "memorybar" => {
            use presentar_core::Color;
            let mut widget = MemoryBar::new(32_000_000_000);
            widget.add_segment(MemorySegment::new(
                "Used",
                18_200_000_000,
                Color::new(0.3, 0.7, 1.0, 1.0),
            ));
            widget.add_segment(MemorySegment::new(
                "Cached",
                8_000_000_000,
                Color::new(0.5, 0.5, 0.5, 1.0),
            ));
            widget.add_segment(MemorySegment::new(
                "Free",
                5_800_000_000,
                Color::new(0.2, 0.8, 0.3, 1.0),
            ));
            harness.benchmark(&mut widget, bounds)
        }
        "process-table" | "processtable" => {
            use presentar_terminal::{ProcessEntry, ProcessState};
            let mut widget = ProcessTable::new();
            // Add some test processes
            widget.set_processes(vec![
                ProcessEntry {
                    pid: 1,
                    user: "root".to_string(),
                    cpu_percent: 0.1,
                    mem_percent: 0.5,
                    command: "systemd".to_string(),
                    cmdline: Some("/sbin/init".to_string()),
                    state: ProcessState::Sleeping,
                    oom_score: Some(0),
                    cgroup: None,
                    nice: Some(0),
                    threads: None,
                    parent_pid: None,
                    tree_depth: 0,
                    is_last_child: false,
                    tree_prefix: String::new(),
                },
                ProcessEntry {
                    pid: 1000,
                    user: "noah".to_string(),
                    cpu_percent: 45.2,
                    mem_percent: 8.3,
                    command: "claude".to_string(),
                    cmdline: Some("claude-code".to_string()),
                    state: ProcessState::Running,
                    oom_score: Some(100),
                    cgroup: None,
                    nice: Some(0),
                    threads: None,
                    parent_pid: None,
                    tree_depth: 0,
                    is_last_child: false,
                    tree_prefix: String::new(),
                },
            ]);
            harness.benchmark(&mut widget, bounds)
        }
        _ => {
            eprintln!("Unknown widget: {widget_name}");
            eprintln!("Available widgets: cpu-grid, braille-graph, memory-bar, process-table");
            std::process::exit(1);
        }
    }
}

fn output_result(result: &BenchmarkResult, cli: &Cli) {
    let output = match cli.format.as_str() {
        "json" => result.to_json(),
        "csv" => {
            format!(
                "{}\n{}",
                RenderMetrics::csv_header(),
                result
                    .metrics
                    .to_csv_row(&result.widget_name, result.width, result.height)
            )
        }
        _ => format_text_output(result),
    };

    if let Some(ref path) = cli.output {
        let mut file = fs::File::create(path).expect("Failed to create output file");
        file.write_all(output.as_bytes())
            .expect("Failed to write output");
        if cli.verbose {
            eprintln!("Results written to: {}", path.display());
        }
    } else {
        println!("{output}");
    }
}

fn format_text_output(result: &BenchmarkResult) -> String {
    let metrics = &result.metrics;
    let ft = &metrics.frame_times;

    format!(
        r"
╔══════════════════════════════════════════════════════════════════╗
║                    BENCHMARK RESULTS                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Widget: {:<54} ║
║  Dimensions: {}x{:<52} ║
╠══════════════════════════════════════════════════════════════════╣
║  FRAME TIME STATISTICS                                            ║
║  ────────────────────────────────────────────────────────────────║
║  Frames:     {:<54} ║
║  Min:        {:<50} µs ║
║  Max:        {:<50} µs ║
║  Mean:       {:<48.1} µs ║
║  P50:        {:<50} µs ║
║  P95:        {:<50} µs ║
║  P99:        {:<50} µs ║
║  Std Dev:    {:<48.1} µs ║
╠══════════════════════════════════════════════════════════════════╣
║  TARGET COMPLIANCE                                                ║
║  ────────────────────────────────────────────────────────────────║
║  60fps (16.67ms): {:<48} ║
║  1ms p99:         {:<48} ║
╚══════════════════════════════════════════════════════════════════╝
",
        result.widget_name,
        result.width,
        result.height,
        metrics.frame_count,
        ft.min_us,
        ft.max_us,
        ft.mean_us,
        ft.p50_us,
        ft.p95_us,
        ft.p99_us,
        ft.stddev_us,
        if ft.max_us <= 16_667 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        },
        if ft.p99_us <= 1_000 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        },
    )
}

fn validate_result(result: &BenchmarkResult, cli: &Cli) {
    let targets = if cli.strict {
        PerformanceTargets::strict()
    } else {
        PerformanceTargets::default()
    };

    if result.meets_targets(&targets) {
        if cli.verbose {
            eprintln!("✓ Performance targets met");
        }
    } else {
        eprintln!("✗ Performance targets NOT met");
        eprintln!(
            "  Max frame: {}µs (target: {}µs)",
            result.metrics.frame_times.max_us, targets.max_frame_us
        );
        eprintln!(
            "  P99 frame: {}µs (target: {}µs)",
            result.metrics.frame_times.p99_us, targets.p99_frame_us
        );
        std::process::exit(1);
    }
}

fn run_suite(all: bool, output_dir: &PathBuf, cli: &Cli) {
    let widgets = if all {
        vec!["cpu-grid", "braille-graph", "memory-bar", "process-table"]
    } else {
        vec!["cpu-grid"]
    };

    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let mut all_results = Vec::new();

    for widget in widgets {
        if cli.verbose {
            eprintln!("Benchmarking: {widget}");
        }

        let result = benchmark_widget(widget, cli);
        all_results.push(result.clone());

        // Write individual result
        let path = output_dir.join(format!("{widget}.json"));
        let json = result.to_json();
        fs::write(&path, json).expect("Failed to write result");
    }

    // Write summary CSV
    let csv_path = output_dir.join("summary.csv");
    let mut csv = String::from(RenderMetrics::csv_header());
    csv.push('\n');
    for result in &all_results {
        csv.push_str(
            &result
                .metrics
                .to_csv_row(&result.widget_name, result.width, result.height),
        );
        csv.push('\n');
    }
    fs::write(&csv_path, csv).expect("Failed to write CSV summary");

    // Write summary JSON
    let summary_path = output_dir.join("summary.json");
    let summary_json = format!(
        r#"{{
  "total_widgets": {},
  "all_passed": {},
  "results": [{}]
}}"#,
        all_results.len(),
        all_results
            .iter()
            .all(|r| r.meets_targets(&PerformanceTargets::default())),
        all_results
            .iter()
            .map(|r| format!("\"{}\"", r.widget_name))
            .collect::<Vec<_>>()
            .join(", ")
    );
    fs::write(&summary_path, summary_json).expect("Failed to write summary JSON");

    println!("Suite complete. Results in: {}", output_dir.display());
    println!("  - Individual results: <widget>.json");
    println!("  - Summary CSV: summary.csv");
    println!("  - Summary JSON: summary.json");
}

fn run_compare(widget_a: &str, widget_b: &str, cli: &Cli) {
    let mut harness = BenchmarkHarness::new(cli.width, cli.height)
        .with_frames(cli.warmup, cli.frames)
        .with_deterministic(cli.deterministic);

    let bounds = Rect::new(0.0, 0.0, f32::from(cli.width), f32::from(cli.height));

    // Create widgets for comparison
    let _ctx = DeterministicContext::new();
    let data: Vec<f64> = (0..60)
        .map(|i| (f64::from(i) * 0.1).sin().mul_add(50.0, 50.0))
        .collect();

    let result = if (
        widget_a.to_lowercase().as_str(),
        widget_b.to_lowercase().as_str(),
    ) == ("braille", "block")
    {
        let mut braille = BrailleGraph::new(data.clone()).with_mode(GraphMode::Braille);
        let mut block = BrailleGraph::new(data).with_mode(GraphMode::Block);
        harness.compare(&mut braille, &mut block, bounds)
    } else {
        eprintln!("Comparison not implemented for: {widget_a} vs {widget_b}");
        eprintln!("Available comparisons: braille vs block");
        std::process::exit(1);
    };

    println!("Comparison: {widget_a} vs {widget_b}");
    println!("{}", result.summary());
    println!();
    println!(
        "  {} faster: {}",
        if result.a_is_faster() {
            widget_a
        } else {
            widget_b
        },
        if result.a_is_faster() {
            format!("{:.2}x", result.speedup_ratio())
        } else {
            format!("{:.2}x", 1.0 / result.speedup_ratio())
        }
    );
}

fn run_snapshot(widget_name: &str, output_path: &PathBuf, cli: &Cli) {
    let result = benchmark_widget(widget_name, cli);

    fs::write(output_path, &result.final_frame).expect("Failed to write snapshot");

    if cli.verbose {
        eprintln!("Snapshot written to: {}", output_path.display());
    }
}

fn run_validate(results_path: &PathBuf, cli: &Cli) {
    let content = fs::read_to_string(results_path).expect("Failed to read results file");

    // Simple validation - check for "meets_targets": true
    let targets = if cli.strict {
        PerformanceTargets::strict()
    } else {
        PerformanceTargets::default()
    };

    if content.contains("\"meets_targets\": true") || content.contains("all_passed\": true") {
        println!("✓ All benchmarks passed performance targets");
        if cli.verbose {
            println!("  Max frame target: {}µs", targets.max_frame_us);
            println!("  P99 frame target: {}µs", targets.p99_frame_us);
        }
    } else {
        eprintln!("✗ Some benchmarks failed performance targets");
        std::process::exit(1);
    }
}
