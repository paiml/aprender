#!/usr/bin/env -S deno run --allow-read --allow-write
/**
 * Markdown Report Generator for Benchmark Results
 *
 * Generates comprehensive markdown reports from JSON benchmark results with:
 * - Executive summary with key findings
 * - Benchmark results table with statistics
 * - Performance comparisons and speedup analysis
 * - Regression/improvement warnings
 * - Hardware and software environment details
 * - Statistical methodology documentation
 *
 * Usage:
 *   deno run --allow-read --allow-write generate_markdown_report.ts input.json output.md
 */

// Type definitions matching Rust BenchmarkReport structure
interface Statistics {
  mean_ms: number;
  median_ms: number;
  std_dev_ms: number;
  min_ms: number;
  max_ms: number;
  coefficient_of_variation: number;
  confidence_interval_95: [number, number];
}

interface ComparisonResult {
  baseline_mean_ms: number;
  current_mean_ms: number;
  speedup_ratio: number;
  effect_size_cohens_d: number;
  p_value: number;
  significant: boolean;
}

interface BenchmarkResult {
  benchmark_name: string;
  benchmark_type: string;
  optimization_level: string;
  warmup_iterations: number;
  measured_iterations: number;
  raw_timings_ms: number[];
  statistics: Statistics;
  comparison?: ComparisonResult;
  metadata?: Record<string, string>;
}

interface CpuInfo {
  model: string;
  cores_physical: number;
  cores_logical: number;
  base_frequency_mhz: number;
}

interface MemoryInfo {
  total_mb: number;
  available_mb: number;
}

interface HardwareInfo {
  cpu: CpuInfo;
  memory: MemoryInfo;
}

interface SoftwareInfo {
  os_name: string;
  os_version: string;
  kernel_version: string;
  rustc_version: string;
  cargo_version: string;
  llvm_version?: string;
}

interface EnvironmentConfig {
  cpu_governor: string;
  swap_enabled: boolean;
  additional_config?: Record<string, string>;
}

interface BenchmarkMetadata {
  benchmark_suite: string;
  timestamp: string;
  git_commit: string;
  git_branch: string;
  operator: string;
  hardware: HardwareInfo;
  software: SoftwareInfo;
  environment: EnvironmentConfig;
}

interface BenchmarkSummary {
  total_benchmarks: number;
  successful: number;
  failed: number;
  total_runtime_seconds: number;
  significant_improvements: number;
  significant_regressions: number;
}

interface BenchmarkReport {
  schema_version: string;
  metadata: BenchmarkMetadata;
  benchmarks: BenchmarkResult[];
  summary: BenchmarkSummary;
}

/**
 * Format number with fixed precision
 */
function fmt(value: number, precision = 2): string {
  return value.toFixed(precision);
}

/**
 * Format percentage
 */
function fmtPct(value: number, precision = 1): string {
  return `${(value * 100).toFixed(precision)}%`;
}

/**
 * Generate performance indicator emoji
 */
function performanceIndicator(comp?: ComparisonResult): string {
  if (!comp) return "";

  if (!comp.significant) return "○"; // Not significant

  if (comp.speedup_ratio > 1.0) {
    // Improvement
    if (comp.speedup_ratio > 1.2) return "🚀"; // Major improvement (>20%)
    if (comp.speedup_ratio > 1.1) return "✅"; // Good improvement (>10%)
    return "↗"; // Minor improvement
  } else {
    // Regression
    if (comp.speedup_ratio < 0.8) return "🔴"; // Major regression (>20% slower)
    if (comp.speedup_ratio < 0.9) return "⚠️"; // Significant regression (>10% slower)
    return "↘"; // Minor regression
  }
}

/**
 * Generate executive summary section
 */
function generateExecutiveSummary(report: BenchmarkReport): string {
  const lines: string[] = [];
  const meta = report.metadata;
  const summary = report.summary;

  lines.push("# Benchmark Report: " + meta.benchmark_suite);
  lines.push("");
  lines.push("## Executive Summary");
  lines.push("");

  // Key metrics
  lines.push(`- **Total Benchmarks**: ${summary.total_benchmarks}`);
  lines.push(`- **Successful**: ${summary.successful}`);
  lines.push(`- **Failed**: ${summary.failed}`);
  lines.push(
    `- **Total Runtime**: ${fmt(summary.total_runtime_seconds, 2)} seconds`,
  );
  lines.push("");

  // Performance changes (if comparisons exist)
  const hasComparisons = report.benchmarks.some((b) => b.comparison);
  if (hasComparisons) {
    lines.push("### Performance Changes");
    lines.push("");

    if (summary.significant_regressions > 0) {
      lines.push(
        `🔴 **${summary.significant_regressions} significant regression(s) detected**`,
      );
    }
    if (summary.significant_improvements > 0) {
      lines.push(
        `🚀 **${summary.significant_improvements} significant improvement(s) detected**`,
      );
    }
    if (
      summary.significant_regressions === 0 &&
      summary.significant_improvements === 0
    ) {
      lines.push("✅ No significant performance changes detected");
    }
    lines.push("");
  }

  // Timestamp and git info
  lines.push("### Run Information");
  lines.push("");
  lines.push(`- **Timestamp**: ${meta.timestamp}`);
  lines.push(`- **Git Commit**: \`${meta.git_commit}\``);
  lines.push(`- **Git Branch**: \`${meta.git_branch}\``);
  lines.push(`- **Operator**: ${meta.operator}`);
  lines.push("");

  return lines.join("\n");
}

/**
 * Generate benchmark results table
 */
function generateResultsTable(report: BenchmarkReport): string {
  const lines: string[] = [];

  lines.push("## Benchmark Results");
  lines.push("");

  // Table header
  lines.push(
    "| Benchmark | Type | Opt | Mean (ms) | Median (ms) | Std Dev | CV | CI 95% | Speedup | Status |",
  );
  lines.push(
    "|-----------|------|-----|-----------|-------------|---------|-------|--------|---------|--------|",
  );

  // Table rows
  for (const bench of report.benchmarks) {
    const stats = bench.statistics;
    const comp = bench.comparison;

    const ci = `[${fmt(stats.confidence_interval_95[0], 2)}, ${
      fmt(stats.confidence_interval_95[1], 2)
    }]`;

    const speedup = comp ? `${fmt(comp.speedup_ratio, 2)}x` : "—";
    const indicator = performanceIndicator(comp);

    lines.push(
      `| ${bench.benchmark_name} | ${bench.benchmark_type} | ${bench.optimization_level} | ${
        fmt(stats.mean_ms, 3)
      } | ${fmt(stats.median_ms, 3)} | ${fmt(stats.std_dev_ms, 3)} | ${
        fmtPct(stats.coefficient_of_variation)
      } | ${ci} | ${speedup} | ${indicator} |`,
    );
  }

  lines.push("");

  // Legend
  lines.push("**Status Legend:**");
  lines.push("- 🚀 Major improvement (>20% faster)");
  lines.push("- ✅ Good improvement (>10% faster)");
  lines.push("- ↗ Minor improvement");
  lines.push("- ○ No significant change");
  lines.push("- ↘ Minor regression");
  lines.push("- ⚠️ Significant regression (>10% slower)");
  lines.push("- 🔴 Major regression (>20% slower)");
  lines.push("");

  return lines.join("\n");
}

/**
 * Generate detailed comparison section (if comparisons exist)
 */
function generateComparisonDetails(report: BenchmarkReport): string {
  const benchmarksWithComparison = report.benchmarks.filter((b) =>
    b.comparison
  );

  if (benchmarksWithComparison.length === 0) {
    return "";
  }

  const lines: string[] = [];

  lines.push("## Performance Comparison Details");
  lines.push("");

  for (const bench of benchmarksWithComparison) {
    const comp = bench.comparison!;

    lines.push(`### ${bench.benchmark_name}`);
    lines.push("");

    // Performance change summary
    const changePercent = ((comp.speedup_ratio - 1) * 100);
    const direction = changePercent > 0 ? "faster" : "slower";
    const absChange = Math.abs(changePercent);

    lines.push(`**Performance Change**: ${fmt(absChange, 1)}% ${direction}`);
    lines.push("");

    // Detailed metrics
    lines.push("| Metric | Value |");
    lines.push("|--------|-------|");
    lines.push(`| Baseline Mean | ${fmt(comp.baseline_mean_ms, 3)} ms |`);
    lines.push(`| Current Mean | ${fmt(comp.current_mean_ms, 3)} ms |`);
    lines.push(`| Speedup Ratio | ${fmt(comp.speedup_ratio, 3)}x |`);
    lines.push(
      `| Effect Size (Cohen's d) | ${fmt(comp.effect_size_cohens_d, 3)} |`,
    );
    lines.push(`| P-value | ${fmt(comp.p_value, 6)} |`);
    lines.push(
      `| Statistically Significant | ${comp.significant ? "Yes" : "No"} |`,
    );
    lines.push("");

    // Interpretation
    if (comp.significant) {
      const effectMagnitude = Math.abs(comp.effect_size_cohens_d);
      let interpretation = "";

      if (effectMagnitude < 0.2) interpretation = "negligible";
      else if (effectMagnitude < 0.5) interpretation = "small";
      else if (effectMagnitude < 0.8) interpretation = "medium";
      else interpretation = "large";

      lines.push(
        `**Interpretation**: The performance change is statistically significant (p < 0.05) with a ${interpretation} effect size.`,
      );
    } else {
      lines.push(
        "**Interpretation**: The performance difference is not statistically significant (p ≥ 0.05).",
      );
    }

    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Generate regressions/improvements section
 */
function generatePerformanceChanges(report: BenchmarkReport): string {
  const regressions = report.benchmarks.filter(
    (b) => b.comparison && b.comparison.significant && b.comparison.speedup_ratio < 1.0,
  );
  const improvements = report.benchmarks.filter(
    (b) => b.comparison && b.comparison.significant && b.comparison.speedup_ratio > 1.0,
  );

  if (regressions.length === 0 && improvements.length === 0) {
    return "";
  }

  const lines: string[] = [];

  lines.push("## Performance Changes Summary");
  lines.push("");

  // Regressions
  if (regressions.length > 0) {
    lines.push("### 🔴 Regressions");
    lines.push("");

    for (const bench of regressions) {
      const comp = bench.comparison!;
      const slowdown = ((1 - comp.speedup_ratio) * 100);

      lines.push(
        `- **${bench.benchmark_name}**: ${fmt(slowdown, 1)}% slower (${
          fmt(comp.current_mean_ms, 3)
        } ms vs ${fmt(comp.baseline_mean_ms, 3)} ms)`,
      );
    }

    lines.push("");
  }

  // Improvements
  if (improvements.length > 0) {
    lines.push("### 🚀 Improvements");
    lines.push("");

    for (const bench of improvements) {
      const comp = bench.comparison!;
      const speedup = ((comp.speedup_ratio - 1) * 100);

      lines.push(
        `- **${bench.benchmark_name}**: ${fmt(speedup, 1)}% faster (${
          fmt(comp.current_mean_ms, 3)
        } ms vs ${fmt(comp.baseline_mean_ms, 3)} ms)`,
      );
    }

    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Generate environment details section
 */
function generateEnvironmentDetails(report: BenchmarkReport): string {
  const lines: string[] = [];
  const meta = report.metadata;

  lines.push("## Test Environment");
  lines.push("");

  // Hardware
  lines.push("### Hardware");
  lines.push("");
  lines.push("| Component | Specification |");
  lines.push("|-----------|---------------|");
  lines.push(`| CPU | ${meta.hardware.cpu.model} |`);
  lines.push(
    `| Physical Cores | ${meta.hardware.cpu.cores_physical} |`,
  );
  lines.push(
    `| Logical Cores | ${meta.hardware.cpu.cores_logical} |`,
  );
  lines.push(
    `| Base Frequency | ${meta.hardware.cpu.base_frequency_mhz} MHz |`,
  );
  lines.push(
    `| Total Memory | ${fmt(meta.hardware.memory.total_mb / 1024, 2)} GB |`,
  );
  lines.push(
    `| Available Memory | ${
      fmt(meta.hardware.memory.available_mb / 1024, 2)
    } GB |`,
  );
  lines.push("");

  // Software
  lines.push("### Software");
  lines.push("");
  lines.push("| Component | Version |");
  lines.push("|-----------|---------|");
  lines.push(`| Operating System | ${meta.software.os_name} ${meta.software.os_version} |`);
  lines.push(`| Kernel | ${meta.software.kernel_version} |`);
  lines.push(`| Rust Compiler | ${meta.software.rustc_version} |`);
  lines.push(`| Cargo | ${meta.software.cargo_version} |`);
  if (meta.software.llvm_version) {
    lines.push(`| LLVM | ${meta.software.llvm_version} |`);
  }
  lines.push("");

  // Environment configuration
  lines.push("### Environment Configuration");
  lines.push("");
  lines.push("| Setting | Value |");
  lines.push("|---------|-------|");
  lines.push(`| CPU Governor | ${meta.environment.cpu_governor} |`);
  lines.push(`| Swap Enabled | ${meta.environment.swap_enabled ? "Yes" : "No"} |`);

  if (meta.environment.additional_config) {
    for (const [key, value] of Object.entries(meta.environment.additional_config)) {
      lines.push(`| ${key} | ${value} |`);
    }
  }

  lines.push("");

  return lines.join("\n");
}

/**
 * Generate methodology section
 */
function generateMethodology(report: BenchmarkReport): string {
  const lines: string[] = [];

  lines.push("## Statistical Methodology");
  lines.push("");

  lines.push("### Measurement Protocol");
  lines.push("");

  // Get sample benchmark for warmup/iteration info
  const sampleBench = report.benchmarks[0];
  if (sampleBench) {
    lines.push(`- **Warmup Iterations**: ${sampleBench.warmup_iterations}`);
    lines.push(`- **Measured Iterations**: ${sampleBench.measured_iterations}`);
  }

  lines.push("");

  lines.push("### Statistical Analysis");
  lines.push("");
  lines.push("**Descriptive Statistics:**");
  lines.push("- Mean, median, standard deviation, min, max");
  lines.push("- Coefficient of Variation (CV) for measurement stability");
  lines.push("- 95% confidence intervals using bootstrap method (1000 samples)");
  lines.push("");

  lines.push("**Comparative Analysis (when baseline available):**");
  lines.push("- Speedup ratio: baseline_mean / current_mean");
  lines.push("- Effect size: Cohen's d (standardized mean difference)");
  lines.push("- Statistical significance: Welch's t-test (α = 0.05)");
  lines.push("");

  lines.push("**Interpretation Guidelines:**");
  lines.push("");
  lines.push("| Cohen's d | Interpretation |");
  lines.push("|-----------|----------------|");
  lines.push("| < 0.2 | Negligible effect |");
  lines.push("| 0.2 - 0.5 | Small effect |");
  lines.push("| 0.5 - 0.8 | Medium effect |");
  lines.push("| > 0.8 | Large effect |");
  lines.push("");

  lines.push("**Significance Threshold:**");
  lines.push("- P-value < 0.05 considered statistically significant");
  lines.push("- Changes marked significant only if p < 0.05 AND |Cohen's d| > 0.2");
  lines.push("");

  return lines.join("\n");
}

/**
 * Generate reproducibility section
 */
function generateReproducibility(report: BenchmarkReport): string {
  const lines: string[] = [];

  lines.push("## Reproducibility");
  lines.push("");
  lines.push(
    "This benchmark report includes complete environment metadata for reproducibility.",
  );
  lines.push("");
  lines.push("**To reproduce these results:**");
  lines.push("");
  lines.push("1. Check out git commit: `" + report.metadata.git_commit + "`");
  lines.push("2. Install Rust toolchain: `" + report.metadata.software.rustc_version + "`");
  lines.push("3. Build with: `cargo build --release --locked`");
  lines.push("4. Run benchmarks with the same configuration");
  lines.push("");
  lines.push(
    "**Note**: Minor variations in absolute timings are expected due to hardware differences and system load.",
  );
  lines.push(
    "Statistical significance testing accounts for measurement variability.",
  );
  lines.push("");

  return lines.join("\n");
}

/**
 * Generate complete markdown report
 */
function generateMarkdownReport(report: BenchmarkReport): string {
  const sections: string[] = [];

  sections.push(generateExecutiveSummary(report));
  sections.push(generateResultsTable(report));
  sections.push(generatePerformanceChanges(report));
  sections.push(generateComparisonDetails(report));
  sections.push(generateEnvironmentDetails(report));
  sections.push(generateMethodology(report));
  sections.push(generateReproducibility(report));

  // Footer
  sections.push("---");
  sections.push("");
  sections.push(`*Report generated: ${new Date().toISOString()}*`);
  sections.push(
    `*Schema version: ${report.schema_version}*`,
  );

  return sections.filter((s) => s.length > 0).join("\n");
}

/**
 * Load benchmark report from JSON file
 */
async function loadReport(inputPath: string): Promise<BenchmarkReport> {
  const content = await Deno.readTextFile(inputPath);
  return JSON.parse(content) as BenchmarkReport;
}

/**
 * Write markdown report to file
 */
async function writeMarkdown(
  outputPath: string,
  content: string,
): Promise<void> {
  await Deno.writeTextFile(outputPath, content);
}

/**
 * Main CLI
 */
async function main() {
  const args = Deno.args;

  if (args.length < 2) {
    console.error(
      "Usage: generate_markdown_report.ts <input.json> <output.md>",
    );
    console.error("");
    console.error("Example:");
    console.error(
      "  deno run --allow-read --allow-write generate_markdown_report.ts results.json report.md",
    );
    Deno.exit(1);
  }

  const inputPath = args[0];
  const outputPath = args[1];

  console.log("=== Markdown Report Generator ===");
  console.log(`Input: ${inputPath}`);
  console.log(`Output: ${outputPath}`);
  console.log("");

  // Load benchmark report
  console.log("Loading benchmark report...");
  const report = await loadReport(inputPath);
  console.log(`  ✓ Loaded ${report.benchmarks.length} benchmarks`);
  console.log("");

  // Generate markdown report
  console.log("Generating markdown report...");
  const markdown = generateMarkdownReport(report);

  // Write to file
  await writeMarkdown(outputPath, markdown);
  console.log(`  ✓ ${outputPath}`);
  console.log("");

  console.log("Markdown report generation complete!");
}

// Run CLI if this is the main module
if (import.meta.main) {
  main();
}

// Export functions for use as library
export { generateMarkdownReport, loadReport };
