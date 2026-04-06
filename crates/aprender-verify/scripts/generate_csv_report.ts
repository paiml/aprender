#!/usr/bin/env -S deno run --allow-read --allow-write
/**
 * CSV Report Generator for Benchmark Results
 *
 * Generates CSV reports from JSON benchmark results with the following tables:
 * - Benchmark results summary (one row per benchmark)
 * - Detailed statistics table
 * - Metadata summary
 * - Hardware configuration
 * - Software environment
 *
 * Usage:
 *   deno run --allow-read --allow-write generate_csv_report.ts input.json output.csv
 *   deno run --allow-read --allow-write generate_csv_report.ts input.json output/ --multi
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
 * Escape CSV field (handle commas, quotes, newlines)
 */
function escapeCSV(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }

  const str = String(value);

  // If contains comma, quote, or newline, wrap in quotes and escape internal quotes
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }

  return str;
}

/**
 * Generate CSV row from array of values
 */
function csvRow(values: unknown[]): string {
  return values.map(escapeCSV).join(",");
}

/**
 * Generate benchmarks summary CSV
 */
function generateBenchmarkSummaryCSV(report: BenchmarkReport): string {
  const lines: string[] = [];

  // Header
  lines.push(
    csvRow([
      "Benchmark Name",
      "Type",
      "Opt Level",
      "Warmup",
      "Iterations",
      "Mean (ms)",
      "Median (ms)",
      "Std Dev (ms)",
      "CV (%)",
      "Min (ms)",
      "Max (ms)",
      "CI95 Lower (ms)",
      "CI95 Upper (ms)",
      "Speedup",
      "Cohen's d",
      "P-value",
      "Significant",
    ]),
  );

  // Data rows
  for (const bench of report.benchmarks) {
    const stats = bench.statistics;
    const comp = bench.comparison;

    lines.push(
      csvRow([
        bench.benchmark_name,
        bench.benchmark_type,
        bench.optimization_level,
        bench.warmup_iterations,
        bench.measured_iterations,
        stats.mean_ms.toFixed(3),
        stats.median_ms.toFixed(3),
        stats.std_dev_ms.toFixed(3),
        (stats.coefficient_of_variation * 100).toFixed(2),
        stats.min_ms.toFixed(3),
        stats.max_ms.toFixed(3),
        stats.confidence_interval_95[0].toFixed(3),
        stats.confidence_interval_95[1].toFixed(3),
        comp ? comp.speedup_ratio.toFixed(3) : "",
        comp ? comp.effect_size_cohens_d.toFixed(3) : "",
        comp ? comp.p_value.toFixed(6) : "",
        comp ? (comp.significant ? "Yes" : "No") : "",
      ]),
    );
  }

  return lines.join("\n");
}

/**
 * Generate metadata summary CSV
 */
function generateMetadataCSV(report: BenchmarkReport): string {
  const lines: string[] = [];
  const meta = report.metadata;

  // Header
  lines.push(csvRow(["Property", "Value"]));

  // Benchmark info
  lines.push(csvRow(["Schema Version", report.schema_version]));
  lines.push(csvRow(["Benchmark Suite", meta.benchmark_suite]));
  lines.push(csvRow(["Timestamp", meta.timestamp]));
  lines.push(csvRow(["Git Commit", meta.git_commit]));
  lines.push(csvRow(["Git Branch", meta.git_branch]));
  lines.push(csvRow(["Operator", meta.operator]));

  // Hardware
  lines.push(csvRow(["CPU Model", meta.hardware.cpu.model]));
  lines.push(
    csvRow(["CPU Cores (Physical)", meta.hardware.cpu.cores_physical]),
  );
  lines.push(csvRow(["CPU Cores (Logical)", meta.hardware.cpu.cores_logical]));
  lines.push(
    csvRow([
      "CPU Base Frequency (MHz)",
      meta.hardware.cpu.base_frequency_mhz,
    ]),
  );
  lines.push(csvRow(["Memory Total (MB)", meta.hardware.memory.total_mb]));
  lines.push(
    csvRow(["Memory Available (MB)", meta.hardware.memory.available_mb]),
  );

  // Software
  lines.push(csvRow(["OS Name", meta.software.os_name]));
  lines.push(csvRow(["OS Version", meta.software.os_version]));
  lines.push(csvRow(["Kernel Version", meta.software.kernel_version]));
  lines.push(csvRow(["Rustc Version", meta.software.rustc_version]));
  lines.push(csvRow(["Cargo Version", meta.software.cargo_version]));
  if (meta.software.llvm_version) {
    lines.push(csvRow(["LLVM Version", meta.software.llvm_version]));
  }

  // Environment
  lines.push(csvRow(["CPU Governor", meta.environment.cpu_governor]));
  lines.push(csvRow(["Swap Enabled", meta.environment.swap_enabled]));

  // Summary
  lines.push(csvRow(["Total Benchmarks", report.summary.total_benchmarks]));
  lines.push(csvRow(["Successful", report.summary.successful]));
  lines.push(csvRow(["Failed", report.summary.failed]));
  lines.push(
    csvRow([
      "Total Runtime (seconds)",
      report.summary.total_runtime_seconds.toFixed(2),
    ]),
  );
  lines.push(
    csvRow([
      "Significant Improvements",
      report.summary.significant_improvements,
    ]),
  );
  lines.push(
    csvRow(["Significant Regressions", report.summary.significant_regressions]),
  );

  return lines.join("\n");
}

/**
 * Generate raw timings CSV (one row per iteration)
 */
function generateRawTimingsCSV(report: BenchmarkReport): string {
  const lines: string[] = [];

  // Header - list all benchmark names
  const headers = ["Iteration", ...report.benchmarks.map((b) => b.benchmark_name)];
  lines.push(csvRow(headers));

  // Find max iterations across all benchmarks
  const maxIterations = Math.max(
    ...report.benchmarks.map((b) => b.raw_timings_ms.length),
  );

  // Data rows - one row per iteration
  for (let i = 0; i < maxIterations; i++) {
    const row = [i + 1];
    for (const bench of report.benchmarks) {
      if (i < bench.raw_timings_ms.length) {
        row.push(bench.raw_timings_ms[i].toFixed(3));
      } else {
        row.push("");
      }
    }
    lines.push(csvRow(row));
  }

  return lines.join("\n");
}

/**
 * Load benchmark report from JSON file
 */
async function loadReport(inputPath: string): Promise<BenchmarkReport> {
  const content = await Deno.readTextFile(inputPath);
  return JSON.parse(content) as BenchmarkReport;
}

/**
 * Write CSV to file
 */
async function writeCSV(outputPath: string, content: string): Promise<void> {
  await Deno.writeTextFile(outputPath, content);
  console.log(`  ✓ ${outputPath}`);
}

/**
 * Main CLI
 */
async function main() {
  const args = Deno.args;

  if (args.length < 2) {
    console.error("Usage: generate_csv_report.ts <input.json> <output.csv|output_dir/> [--multi]");
    console.error("");
    console.error("Options:");
    console.error("  --multi    Generate multiple CSV files (summary, metadata, raw_timings)");
    console.error("");
    console.error("Examples:");
    console.error("  deno run --allow-read --allow-write generate_csv_report.ts results.json report.csv");
    console.error("  deno run --allow-read --allow-write generate_csv_report.ts results.json reports/ --multi");
    Deno.exit(1);
  }

  const inputPath = args[0];
  const outputPath = args[1];
  const multiFile = args.includes("--multi");

  console.log("=== CSV Report Generator ===");
  console.log(`Input: ${inputPath}`);
  console.log(`Output: ${outputPath}`);
  console.log(`Mode: ${multiFile ? "multi-file" : "single-file"}`);
  console.log("");

  // Load benchmark report
  console.log("Loading benchmark report...");
  const report = await loadReport(inputPath);
  console.log(`  ✓ Loaded ${report.benchmarks.length} benchmarks`);
  console.log("");

  if (multiFile) {
    // Multi-file mode: generate separate CSV files
    console.log("Generating CSV files...");

    // Ensure output directory exists
    try {
      await Deno.mkdir(outputPath, { recursive: true });
    } catch {
      // Directory might already exist
    }

    // Generate each CSV file
    await writeCSV(
      `${outputPath}/benchmarks_summary.csv`,
      generateBenchmarkSummaryCSV(report),
    );
    await writeCSV(
      `${outputPath}/metadata.csv`,
      generateMetadataCSV(report),
    );
    await writeCSV(
      `${outputPath}/raw_timings.csv`,
      generateRawTimingsCSV(report),
    );
  } else {
    // Single-file mode: just the benchmark summary
    console.log("Generating CSV file...");
    await writeCSV(outputPath, generateBenchmarkSummaryCSV(report));
  }

  console.log("");
  console.log("CSV generation complete!");
}

// Run CLI if this is the main module
if (import.meta.main) {
  main();
}

// Export functions for use as library
export {
  generateBenchmarkSummaryCSV,
  generateMetadataCSV,
  generateRawTimingsCSV,
  loadReport,
};
