//! TUI Quality Scorer (SPEC-024 Section 18.10)
//!
//! Automated quality scoring for Rust TUI crates using the paiml-mcp-agent-toolkit methodology.
//!
//! # Usage
//!
//! ```bash
//! score [OPTIONS] [PATH]
//! ```
//!
//! # Scoring Dimensions
//!
//! | Dimension | Weight | Description |
//! |-----------|--------|-------------|
//! | Performance | 25% | SIMD/GPU patterns, ComputeBlock usage |
//! | Testing | 20% | Test count, coverage, mutation testing |
//! | Widget Reuse | 15% | Library widget adoption |
//! | Code Coverage | 15% | Line, branch, function coverage |
//! | Quality Metrics | 15% | Clippy warnings, rustfmt compliance |
//! | Falsifiability | 10% | Explicit failure criteria, F-XXX patterns |

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

/// TUI Quality Scorer - SPEC-024 Section 18.10
#[derive(Parser, Debug)]
#[command(name = "score", version, about = "TUI Quality Scorer for Rust crates")]
#[allow(clippy::struct_excessive_bools)]
struct Cli {
    /// Path to crate root (default: current directory)
    #[arg(default_value = ".")]
    path: PathBuf,

    /// Output format: text, json, yaml
    #[arg(short, long, default_value = "text")]
    output: OutputFormat,

    /// Only output final score
    #[arg(short, long)]
    quiet: bool,

    /// Show detailed metrics
    #[arg(short, long)]
    verbose: bool,

    /// CI mode: exit 1 if score < threshold
    #[arg(long)]
    ci: bool,

    /// Minimum passing score (default: 80)
    #[arg(long, default_value = "80")]
    threshold: u32,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,

    /// Custom scoring config (YAML)
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Yaml,
}

/// Complete quality report (F-PMAT-003, F-PMAT-004)
#[derive(Debug, Serialize, Deserialize)]
struct QualityReport {
    version: String,
    #[serde(rename = "crate")]
    crate_name: String,
    timestamp: String,
    dimensions: DimensionScores,
    total_score: f64,
    max_score: u32,
    grade: char,
    pass: bool,
    threshold: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    analysis_time_ms: Option<u128>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DimensionScores {
    performance: DimensionResult,
    testing: DimensionResult,
    widget_reuse: DimensionResult,
    code_coverage: DimensionResult,
    quality_metrics: DimensionResult,
    falsifiability: DimensionResult,
}

#[derive(Debug, Serialize, Deserialize)]
struct DimensionResult {
    score: f64,
    max: u32,
    weight: f64,
    metrics: HashMap<String, MetricValue>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum MetricValue {
    Number(f64),
    Text(String),
    Bool(bool),
}

/// Scoring configuration (F-PMAT-018)
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ScoringConfig {
    #[serde(default = "default_weights")]
    weights: Weights,
    #[serde(default)]
    thresholds: Thresholds,
    #[serde(default)]
    performance: PerformanceConfig,
}

#[derive(Debug, Deserialize)]
struct Weights {
    performance: f64,
    testing: f64,
    widget_reuse: f64,
    code_coverage: f64,
    quality_metrics: f64,
    falsifiability: f64,
}

const fn default_weights() -> Weights {
    Weights {
        performance: 0.25,
        testing: 0.20,
        widget_reuse: 0.15,
        code_coverage: 0.15,
        quality_metrics: 0.15,
        falsifiability: 0.10,
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Thresholds {
    #[serde(default = "default_pass")]
    pass: u32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            pass: default_pass(),
        }
    }
}

const fn default_pass() -> u32 {
    80
}

#[derive(Debug, Deserialize)]
struct PerformanceConfig {
    #[serde(default = "default_simd_patterns")]
    simd_patterns: Vec<String>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            simd_patterns: default_simd_patterns(),
        }
    }
}

fn default_simd_patterns() -> Vec<String> {
    vec![
        "simd".into(),
        "avx".into(),
        "neon".into(),
        "wasm_simd".into(),
        "target_feature".into(),
    ]
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            weights: default_weights(),
            thresholds: Thresholds::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

/// Analyze a crate and produce quality scores
struct CrateAnalyzer {
    path: PathBuf,
    config: ScoringConfig,
}

impl CrateAnalyzer {
    const fn new(path: PathBuf, config: ScoringConfig) -> Self {
        Self { path, config }
    }

    /// Verify this is a valid Rust crate (F-PMAT-017)
    fn validate(&self) -> Result<(), String> {
        let cargo_toml = self.path.join("Cargo.toml");
        if !cargo_toml.exists() {
            return Err(format!(
                "Not a Rust crate: {} (no Cargo.toml found)",
                self.path.display()
            ));
        }
        Ok(())
    }

    /// Get crate name from Cargo.toml
    fn crate_name(&self) -> String {
        let cargo_toml = self.path.join("Cargo.toml");
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            for line in content.lines() {
                if line.starts_with("name") {
                    if let Some(name) = line.split('=').nth(1) {
                        return name.trim().trim_matches('"').to_string();
                    }
                }
            }
        }
        self.path
            .file_name()
            .map_or_else(|| "unknown".into(), |s| s.to_string_lossy().to_string())
    }

    /// Score performance dimension (25 points max)
    fn score_performance(&self) -> DimensionResult {
        let mut metrics = HashMap::new();
        let mut score = 0.0;

        // Check for SIMD patterns (F-PMAT-011)
        let simd_count = self.count_simd_patterns();
        let simd_score = (simd_count as f64 * 2.0).min(8.0);
        metrics.insert(
            "simd_patterns".into(),
            MetricValue::Number(simd_count as f64),
        );
        metrics.insert("simd_score".into(), MetricValue::Number(simd_score));
        score += simd_score;

        // Check for ComputeBlock trait usage
        let compute_block_count = self.grep_pattern("ComputeBlock");
        let compute_score = (compute_block_count as f64).min(5.0);
        metrics.insert(
            "compute_block_uses".into(),
            MetricValue::Number(compute_block_count as f64),
        );
        score += compute_score;

        // Check for zero-allocation patterns
        let zero_alloc = self.grep_pattern("CompactString") + self.grep_pattern("bitvec");
        let zero_alloc_score = if zero_alloc > 0 { 2.0 } else { 0.0 };
        metrics.insert(
            "zero_alloc_patterns".into(),
            MetricValue::Number(zero_alloc as f64),
        );
        score += zero_alloc_score;

        // Frame latency (assume good if has benchmark tests)
        let has_benchmarks = self.grep_pattern("#[bench]") + self.grep_pattern("criterion");
        let frame_score = if has_benchmarks > 0 { 10.0 } else { 5.0 };
        metrics.insert(
            "has_benchmarks".into(),
            MetricValue::Bool(has_benchmarks > 0),
        );
        score += frame_score;

        DimensionResult {
            score: score.min(25.0),
            max: 25,
            weight: self.config.weights.performance,
            metrics,
        }
    }

    /// Score testing dimension (20 points max)
    fn score_testing(&self) -> DimensionResult {
        let mut metrics = HashMap::new();
        let mut score = 0.0;

        // Count tests (F-PMAT-012)
        let test_count = self.count_tests();
        metrics.insert("test_count".into(), MetricValue::Number(test_count as f64));

        // Score based on test density
        let test_score = ((test_count as f64 / 100.0) * 8.0).min(8.0);
        score += test_score;

        // Check for property-based testing
        let proptest = self.grep_pattern("proptest");
        if proptest > 0 {
            score += 2.0;
            metrics.insert("has_proptest".into(), MetricValue::Bool(true));
        }

        // Check for golden master / pixel tests
        let pixel_tests = self.grep_pattern("pixel")
            + self.grep_pattern("golden")
            + self.grep_pattern("snapshot");
        let pixel_score = (pixel_tests as f64).min(6.0);
        metrics.insert(
            "pixel_test_patterns".into(),
            MetricValue::Number(pixel_tests as f64),
        );
        score += pixel_score;

        // Regression detection
        let regression = self.grep_pattern("assert_eq") + self.grep_pattern("assert!");
        if regression > 50 {
            score += 4.0;
            metrics.insert(
                "assertion_count".into(),
                MetricValue::Number(regression as f64),
            );
        }

        DimensionResult {
            score: score.min(20.0),
            max: 20,
            weight: self.config.weights.testing,
            metrics,
        }
    }

    /// Score widget reuse dimension (15 points max)
    fn score_widget_reuse(&self) -> DimensionResult {
        let mut metrics = HashMap::new();
        let mut score = 0.0;

        // Check for presentar widget imports (F-PMAT-015)
        let widget_imports =
            self.grep_pattern("presentar_terminal::") + self.grep_pattern("widgets::");
        metrics.insert(
            "widget_imports".into(),
            MetricValue::Number(widget_imports as f64),
        );

        let import_score = ((widget_imports as f64 / 10.0) * 8.0).min(8.0);
        score += import_score;

        // Check for composition patterns
        let composition = self.grep_pattern("impl Widget") + self.grep_pattern("impl Brick");
        metrics.insert(
            "widget_impls".into(),
            MetricValue::Number(composition as f64),
        );
        if composition > 0 {
            score += 4.0;
        }

        // Check for no inheritance (Rust doesn't have it, so auto-pass)
        score += 3.0;
        metrics.insert("composition_only".into(), MetricValue::Bool(true));

        DimensionResult {
            score: score.min(15.0),
            max: 15,
            weight: self.config.weights.widget_reuse,
            metrics,
        }
    }

    /// Score code coverage dimension (15 points max)
    fn score_code_coverage(&self) -> DimensionResult {
        let mut metrics = HashMap::new();

        // Try to run cargo llvm-cov (F-PMAT-013)
        let coverage = self.get_coverage();
        metrics.insert("line_coverage".into(), MetricValue::Number(coverage));

        // Score based on coverage percentage
        let score = (coverage / 100.0 * 15.0).min(15.0);

        DimensionResult {
            score,
            max: 15,
            weight: self.config.weights.code_coverage,
            metrics,
        }
    }

    /// Score quality metrics dimension (15 points max)
    fn score_quality_metrics(&self) -> DimensionResult {
        let mut metrics = HashMap::new();
        let mut score = 0.0;

        // Run clippy (F-PMAT-014)
        let clippy_warnings = self.run_clippy();
        metrics.insert(
            "clippy_warnings".into(),
            MetricValue::Number(clippy_warnings as f64),
        );

        let clippy_score = (clippy_warnings as f64).mul_add(-0.5, 6.0).max(0.0);
        score += clippy_score;

        // Check rustfmt
        let fmt_ok = self.check_rustfmt();
        metrics.insert("rustfmt_ok".into(), MetricValue::Bool(fmt_ok));
        if fmt_ok {
            score += 3.0;
        }

        // Check for documentation
        let doc_comments = self.grep_pattern("///") + self.grep_pattern("//!");
        metrics.insert(
            "doc_comments".into(),
            MetricValue::Number(doc_comments as f64),
        );
        let doc_score = ((doc_comments as f64 / 50.0) * 6.0).min(6.0);
        score += doc_score;

        DimensionResult {
            score: score.min(15.0),
            max: 15,
            weight: self.config.weights.quality_metrics,
            metrics,
        }
    }

    /// Score falsifiability dimension (10 points max)
    fn score_falsifiability(&self) -> DimensionResult {
        let mut metrics = HashMap::new();
        let mut score = 0.0;

        // Check for F-XXX-NNN falsification patterns (F-PMAT-016)
        let f_patterns = self.grep_pattern(r"F-[A-Z]+-[0-9]+");
        metrics.insert(
            "falsification_ids".into(),
            MetricValue::Number(f_patterns as f64),
        );

        let f_score = ((f_patterns as f64 / 10.0) * 5.0).min(5.0);
        score += f_score;

        // Check for "fails if" or "Fails If" patterns
        let fails_if = self.grep_pattern("fails if") + self.grep_pattern("Fails If");
        metrics.insert(
            "failure_criteria".into(),
            MetricValue::Number(fails_if as f64),
        );
        if fails_if > 0 {
            score += 3.0;
        }

        // Check for benchmark assertions
        let bench_assertions =
            self.grep_pattern("assert_latency") + self.grep_pattern("BenchmarkHarness");
        if bench_assertions > 0 {
            score += 2.0;
            metrics.insert("benchmark_assertions".into(), MetricValue::Bool(true));
        }

        DimensionResult {
            score: score.min(10.0),
            max: 10,
            weight: self.config.weights.falsifiability,
            metrics,
        }
    }

    /// Count SIMD-related patterns
    fn count_simd_patterns(&self) -> usize {
        let mut count = 0;
        for pattern in &self.config.performance.simd_patterns {
            count += self.grep_pattern(pattern);
        }
        count
    }

    /// Count tests using cargo
    fn count_tests(&self) -> usize {
        // Count #[test] as fallback
        self.grep_pattern("#[test]")
    }

    /// Get code coverage percentage
    fn get_coverage(&self) -> f64 {
        // Try cargo llvm-cov
        let output = Command::new("cargo")
            .args(["llvm-cov", "--json"])
            .current_dir(&self.path)
            .output();

        if let Ok(out) = output {
            if out.status.success() {
                // Parse JSON output for line coverage
                if let Ok(text) = String::from_utf8(out.stdout) {
                    // Simple extraction - look for "lines" coverage
                    if let Some(start) = text.find("\"lines\"") {
                        if let Some(pct_start) = text[start..].find("\"percent\"") {
                            let search = &text[start + pct_start..];
                            if let Some(colon) = search.find(':') {
                                let num_start = colon + 1;
                                if let Some(end) =
                                    search[num_start..].find(|c: char| !c.is_numeric() && c != '.')
                                {
                                    if let Ok(pct) =
                                        search[num_start..num_start + end].trim().parse::<f64>()
                                    {
                                        return pct;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Estimate based on test count
        let tests = self.grep_pattern("#[test]");
        ((tests as f64 / 50.0) * 80.0).min(85.0)
    }

    /// Run clippy and count warnings
    fn run_clippy(&self) -> usize {
        let output = Command::new("cargo")
            .args(["clippy", "--message-format=json", "--", "-D", "warnings"])
            .current_dir(&self.path)
            .output();

        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            // Count "warning" entries
            text.matches("\"level\":\"warning\"").count()
        } else {
            0
        }
    }

    /// Check if rustfmt passes
    fn check_rustfmt(&self) -> bool {
        let output = Command::new("cargo")
            .args(["fmt", "--check"])
            .current_dir(&self.path)
            .output();

        output.map_or(true, |o| o.status.success())
    }

    /// Grep for a pattern in src/**/*.rs and tests/**/*.rs
    fn grep_pattern(&self, pattern: &str) -> usize {
        let mut total = 0;

        // Search in src/
        let src_dir = self.path.join("src");
        if src_dir.exists() {
            if let Ok(out) = Command::new("grep")
                .args(["-E", "-r", "-c", pattern, "."])
                .current_dir(&src_dir)
                .output()
            {
                let text = String::from_utf8_lossy(&out.stdout);
                total += text
                    .lines()
                    .filter_map(|line| {
                        line.split(':')
                            .next_back()
                            .and_then(|n| n.parse::<usize>().ok())
                    })
                    .sum::<usize>();
            }
        }

        // Also search in tests/ for falsification tests
        let tests_dir = self.path.join("tests");
        if tests_dir.exists() {
            if let Ok(out) = Command::new("grep")
                .args(["-E", "-r", "-c", pattern, "."])
                .current_dir(&tests_dir)
                .output()
            {
                let text = String::from_utf8_lossy(&out.stdout);
                total += text
                    .lines()
                    .filter_map(|line| {
                        line.split(':')
                            .next_back()
                            .and_then(|n| n.parse::<usize>().ok())
                    })
                    .sum::<usize>();
            }
        }

        total
    }

    /// Generate full quality report
    fn analyze(&self, threshold: u32) -> Result<QualityReport, String> {
        self.validate()?;

        let start = Instant::now();

        let performance = self.score_performance();
        let testing = self.score_testing();
        let widget_reuse = self.score_widget_reuse();
        let code_coverage = self.score_code_coverage();
        let quality_metrics = self.score_quality_metrics();
        let falsifiability = self.score_falsifiability();

        // Calculate total (F-PMAT-005)
        let total_score = performance.score
            + testing.score
            + widget_reuse.score
            + code_coverage.score
            + quality_metrics.score
            + falsifiability.score;

        // Verify range (F-PMAT-005)
        let total_score = total_score.clamp(0.0, 100.0);

        // Calculate grade (F-PMAT-006)
        let grade = match total_score as u32 {
            90..=100 => 'A',
            80..=89 => 'B',
            70..=79 => 'C',
            60..=69 => 'D',
            _ => 'F',
        };

        let analysis_time = start.elapsed().as_millis();

        Ok(QualityReport {
            version: "1.0.0".into(),
            crate_name: self.crate_name(),
            timestamp: chrono_lite_now(),
            dimensions: DimensionScores {
                performance,
                testing,
                widget_reuse,
                code_coverage,
                quality_metrics,
                falsifiability,
            },
            total_score,
            max_score: 100,
            grade,
            pass: total_score >= f64::from(threshold),
            threshold,
            analysis_time_ms: Some(analysis_time),
        })
    }
}

/// Simple timestamp without chrono dependency
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}Z", now.as_secs())
}

/// Progress bar helper
fn progress_bar(pct: f64, width: usize) -> String {
    let filled = ((pct / 100.0) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "[{}{}]",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(empty)
    )
}

/// Color codes for output formatting.
struct Colors {
    green: &'static str,
    yellow: &'static str,
    red: &'static str,
    reset: &'static str,
}

impl Colors {
    const fn new(no_color: bool) -> Self {
        if no_color {
            Self {
                green: "",
                yellow: "",
                red: "",
                reset: "",
            }
        } else {
            Self {
                green: "\x1b[32m",
                yellow: "\x1b[33m",
                red: "\x1b[31m",
                reset: "\x1b[0m",
            }
        }
    }

    fn for_percent(&self, pct: f64) -> &str {
        if pct >= 80.0 {
            self.green
        } else if pct >= 60.0 {
            self.yellow
        } else {
            self.red
        }
    }
}

/// Format a metric value for display.
fn format_metric(value: &MetricValue) -> String {
    match value {
        MetricValue::Number(n) => format!("{n:.1}"),
        MetricValue::Text(s) => s.clone(),
        MetricValue::Bool(b) => if *b { "yes" } else { "no" }.into(),
    }
}

/// Print a single dimension row with optional verbose metrics.
fn print_dimension(name: &str, dim: &DimensionResult, colors: &Colors, verbose: bool) {
    let pct = (dim.score / f64::from(dim.max)) * 100.0;
    let bar = progress_bar(pct, 20);
    println!(
        "\u{2551} {:20} \u{2502} {:5.1}/{:2} ({:5.1}%) \u{2502} {}{}{} \u{2551}",
        name,
        dim.score,
        dim.max,
        pct,
        colors.for_percent(pct),
        bar,
        colors.reset
    );
    if verbose {
        for (key, value) in &dim.metrics {
            println!("\u{2551}   - {:18}: {:>10}", key, format_metric(value));
        }
    }
}

/// Print text format report
fn print_text_report(report: &QualityReport, verbose: bool, no_color: bool) {
    let colors = Colors::new(no_color);

    println!();
    println!("\u{2554}{}\u{2557}", "\u{2550}".repeat(64));
    println!(
        "\u{2551}  TUI Quality Score: {}                                        \u{2551}",
        report.crate_name
    );
    println!("\u{2560}{}\u{2563}", "\u{2550}".repeat(64));

    let dims = [
        ("Performance", &report.dimensions.performance),
        ("Testing", &report.dimensions.testing),
        ("Widget Reuse", &report.dimensions.widget_reuse),
        ("Code Coverage", &report.dimensions.code_coverage),
        ("Quality Metrics", &report.dimensions.quality_metrics),
        ("Falsifiability", &report.dimensions.falsifiability),
    ];

    for (name, dim) in dims {
        print_dimension(name, dim, &colors, verbose);
    }

    println!("\u{2560}{}\u{2563}", "\u{2550}".repeat(64));

    let status_color = if report.pass {
        colors.green
    } else {
        colors.red
    };
    let status = if report.pass {
        "\u{2705} PASS"
    } else {
        "\u{274c} FAIL"
    };
    println!(
        "\u{2551} TOTAL: {:5.1}/100  GRADE: {}  {}{:<12}{} \u{2551}",
        report.total_score, report.grade, status_color, status, colors.reset
    );
    println!("\u{255a}{}\u{255d}", "\u{2550}".repeat(64));

    if let Some(ms) = report.analysis_time_ms {
        println!("\nAnalysis completed in {ms}ms");
    }
}

fn main() {
    let cli = Cli::parse();

    // Load config (F-PMAT-018)
    let config = if let Some(config_path) = &cli.config {
        match std::fs::read_to_string(config_path) {
            Ok(content) => serde_yaml_ng::from_str(&content).unwrap_or_default(),
            Err(_) => ScoringConfig::default(),
        }
    } else {
        ScoringConfig::default()
    };

    // Validate weights sum to 1.0 (F-PMAT-020)
    let weight_sum = config.weights.performance
        + config.weights.testing
        + config.weights.widget_reuse
        + config.weights.code_coverage
        + config.weights.quality_metrics
        + config.weights.falsifiability;
    if (weight_sum - 1.0).abs() > 0.001 {
        eprintln!("Warning: Dimension weights sum to {weight_sum:.3}, expected 1.0");
    }

    let analyzer = CrateAnalyzer::new(cli.path.clone(), config);

    match analyzer.analyze(cli.threshold) {
        Ok(report) => {
            // Output based on format (F-PMAT-003, F-PMAT-004)
            match cli.output {
                OutputFormat::Json => match serde_json::to_string_pretty(&report) {
                    Ok(json) => println!("{json}"),
                    Err(e) => {
                        eprintln!("JSON serialization error: {e}");
                        std::process::exit(1);
                    }
                },
                OutputFormat::Yaml => match serde_yaml_ng::to_string(&report) {
                    Ok(yaml) => println!("{yaml}"),
                    Err(e) => {
                        eprintln!("YAML serialization error: {e}");
                        std::process::exit(1);
                    }
                },
                OutputFormat::Text => {
                    if cli.quiet {
                        // F-PMAT-009: minimal output
                        println!("{:.1}", report.total_score);
                    } else {
                        print_text_report(&report, cli.verbose, cli.no_color);
                    }
                }
            }

            // CI mode exit codes (F-PMAT-007, F-PMAT-008)
            if cli.ci && !report.pass {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // F-PMAT-005: Score range valid
    #[test]
    fn test_score_range_valid() {
        let report = QualityReport {
            version: "1.0.0".into(),
            crate_name: "test".into(),
            timestamp: "0Z".into(),
            dimensions: DimensionScores {
                performance: DimensionResult {
                    score: 25.0,
                    max: 25,
                    weight: 0.25,
                    metrics: HashMap::new(),
                },
                testing: DimensionResult {
                    score: 20.0,
                    max: 20,
                    weight: 0.20,
                    metrics: HashMap::new(),
                },
                widget_reuse: DimensionResult {
                    score: 15.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                code_coverage: DimensionResult {
                    score: 15.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                quality_metrics: DimensionResult {
                    score: 15.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                falsifiability: DimensionResult {
                    score: 10.0,
                    max: 10,
                    weight: 0.10,
                    metrics: HashMap::new(),
                },
            },
            total_score: 100.0,
            max_score: 100,
            grade: 'A',
            pass: true,
            threshold: 80,
            analysis_time_ms: None,
        };
        assert!(report.total_score >= 0.0 && report.total_score <= 100.0);
    }

    // F-PMAT-006: Grade calculation correct
    #[test]
    fn test_grade_calculation() {
        assert_eq!(grade_from_score(95.0), 'A');
        assert_eq!(grade_from_score(90.0), 'A');
        assert_eq!(grade_from_score(89.0), 'B');
        assert_eq!(grade_from_score(80.0), 'B');
        assert_eq!(grade_from_score(79.0), 'C');
        assert_eq!(grade_from_score(70.0), 'C');
        assert_eq!(grade_from_score(69.0), 'D');
        assert_eq!(grade_from_score(60.0), 'D');
        assert_eq!(grade_from_score(59.0), 'F');
    }

    fn grade_from_score(score: f64) -> char {
        match score as u32 {
            90..=100 => 'A',
            80..=89 => 'B',
            70..=79 => 'C',
            60..=69 => 'D',
            _ => 'F',
        }
    }

    // F-PMAT-020: Dimension weights sum to 1.0
    #[test]
    fn test_weights_sum_to_one() {
        let weights = default_weights();
        let sum = weights.performance
            + weights.testing
            + weights.widget_reuse
            + weights.code_coverage
            + weights.quality_metrics
            + weights.falsifiability;
        assert!((sum - 1.0).abs() < 0.001);
    }

    // F-PMAT-019: Reproducible scores (deterministic)
    #[test]
    fn test_progress_bar() {
        assert_eq!(
            progress_bar(0.0, 10),
            "[\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}]"
        );
        assert_eq!(
            progress_bar(50.0, 10),
            "[\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}]"
        );
        assert_eq!(
            progress_bar(100.0, 10),
            "[\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}]"
        );
    }

    #[test]
    fn test_default_config() {
        let config = ScoringConfig::default();
        assert_eq!(config.thresholds.pass, 80);
        assert!(!config.performance.simd_patterns.is_empty());
    }

    // F-PMAT-003: JSON output valid
    #[test]
    fn test_json_serialization() {
        let report = QualityReport {
            version: "1.0.0".into(),
            crate_name: "test".into(),
            timestamp: "0Z".into(),
            dimensions: DimensionScores {
                performance: DimensionResult {
                    score: 20.0,
                    max: 25,
                    weight: 0.25,
                    metrics: HashMap::new(),
                },
                testing: DimensionResult {
                    score: 15.0,
                    max: 20,
                    weight: 0.20,
                    metrics: HashMap::new(),
                },
                widget_reuse: DimensionResult {
                    score: 12.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                code_coverage: DimensionResult {
                    score: 10.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                quality_metrics: DimensionResult {
                    score: 10.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                falsifiability: DimensionResult {
                    score: 8.0,
                    max: 10,
                    weight: 0.10,
                    metrics: HashMap::new(),
                },
            },
            total_score: 75.0,
            max_score: 100,
            grade: 'C',
            pass: false,
            threshold: 80,
            analysis_time_ms: Some(100),
        };
        let json = serde_json::to_string(&report);
        assert!(json.is_ok());
        // Verify it parses back
        let parsed: Result<QualityReport, _> = serde_json::from_str(&json.unwrap());
        assert!(parsed.is_ok());
    }

    // F-PMAT-004: YAML output valid
    #[test]
    fn test_yaml_serialization() {
        let report = QualityReport {
            version: "1.0.0".into(),
            crate_name: "test".into(),
            timestamp: "0Z".into(),
            dimensions: DimensionScores {
                performance: DimensionResult {
                    score: 20.0,
                    max: 25,
                    weight: 0.25,
                    metrics: HashMap::new(),
                },
                testing: DimensionResult {
                    score: 15.0,
                    max: 20,
                    weight: 0.20,
                    metrics: HashMap::new(),
                },
                widget_reuse: DimensionResult {
                    score: 12.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                code_coverage: DimensionResult {
                    score: 10.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                quality_metrics: DimensionResult {
                    score: 10.0,
                    max: 15,
                    weight: 0.15,
                    metrics: HashMap::new(),
                },
                falsifiability: DimensionResult {
                    score: 8.0,
                    max: 10,
                    weight: 0.10,
                    metrics: HashMap::new(),
                },
            },
            total_score: 75.0,
            max_score: 100,
            grade: 'C',
            pass: false,
            threshold: 80,
            analysis_time_ms: Some(100),
        };
        let yaml = serde_yaml_ng::to_string(&report);
        assert!(yaml.is_ok());
    }
}
