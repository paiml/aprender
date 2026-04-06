#!/usr/bin/env -S deno run --allow-read --allow-write
/**
 * Baseline Manager for Benchmark Results
 *
 * Manages baseline benchmark results for regression tracking:
 * - Save benchmark results as named baselines
 * - List available baselines with metadata
 * - Compare current results against baselines
 * - Archive and delete baselines
 * - Generate comparison reports
 *
 * Usage:
 *   baseline_manager.ts save --input results.json --name baseline-v1.0.0
 *   baseline_manager.ts list
 *   baseline_manager.ts compare --baseline baseline-v1.0.0 --current results.json
 *   baseline_manager.ts info --name baseline-v1.0.0
 *   baseline_manager.ts delete --name baseline-v1.0.0
 */

import {
  calculateCohenD,
  calculateSpeedup,
  welchTTest,
} from "./statistical_analysis.ts";

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

interface BenchmarkResult {
  benchmark_name: string;
  benchmark_type: string;
  optimization_level: string;
  warmup_iterations: number;
  measured_iterations: number;
  raw_timings_ms: number[];
  statistics: Statistics;
}

interface BenchmarkMetadata {
  benchmark_suite: string;
  timestamp: string;
  git_commit: string;
  git_branch: string;
  operator: string;
}

interface BenchmarkReport {
  schema_version: string;
  metadata: BenchmarkMetadata;
  benchmarks: BenchmarkResult[];
  summary: {
    total_benchmarks: number;
    successful: number;
    failed: number;
  };
}

interface BaselineMetadata {
  name: string;
  created: string;
  git_commit: string;
  git_branch: string;
  total_benchmarks: number;
  description?: string;
}

const BASELINE_DIR = "benchmarks/baselines";
const METADATA_FILE = "benchmarks/baselines/index.json";

/**
 * Ensure baseline directory exists
 */
async function ensureBaselineDir(): Promise<void> {
  try {
    await Deno.mkdir(BASELINE_DIR, { recursive: true });
  } catch (error) {
    if (!(error instanceof Deno.errors.AlreadyExists)) {
      throw error;
    }
  }
}

/**
 * Load baseline index (list of all baselines)
 */
async function loadIndex(): Promise<BaselineMetadata[]> {
  try {
    const content = await Deno.readTextFile(METADATA_FILE);
    return JSON.parse(content) as BaselineMetadata[];
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) {
      return [];
    }
    throw error;
  }
}

/**
 * Save baseline index
 */
async function saveIndex(index: BaselineMetadata[]): Promise<void> {
  await ensureBaselineDir();
  await Deno.writeTextFile(METADATA_FILE, JSON.stringify(index, null, 2));
}

/**
 * Load benchmark report from file
 */
async function loadReport(path: string): Promise<BenchmarkReport> {
  const content = await Deno.readTextFile(path);
  return JSON.parse(content) as BenchmarkReport;
}

/**
 * Get baseline file path from name
 */
function getBaselinePath(name: string): string {
  return `${BASELINE_DIR}/${name}.json`;
}

/**
 * Save a baseline
 */
async function saveBaseline(
  inputPath: string,
  name: string,
  description?: string,
): Promise<void> {
  await ensureBaselineDir();

  // Load the report
  const report = await loadReport(inputPath);

  // Check if baseline already exists
  const index = await loadIndex();
  const existing = index.find((b) => b.name === name);

  if (existing) {
    throw new Error(
      `Baseline "${name}" already exists. Use a different name or delete the existing baseline first.`,
    );
  }

  // Save the baseline file
  const baselinePath = getBaselinePath(name);
  await Deno.writeTextFile(baselinePath, JSON.stringify(report, null, 2));

  // Update index
  const metadata: BaselineMetadata = {
    name,
    created: new Date().toISOString(),
    git_commit: report.metadata.git_commit,
    git_branch: report.metadata.git_branch,
    total_benchmarks: report.benchmarks.length,
    description,
  };

  index.push(metadata);
  await saveIndex(index);

  console.log(`✅ Baseline "${name}" saved successfully`);
  console.log(`   File: ${baselinePath}`);
  console.log(`   Commit: ${metadata.git_commit}`);
  console.log(`   Benchmarks: ${metadata.total_benchmarks}`);
}

/**
 * List all baselines
 */
async function listBaselines(): Promise<void> {
  const index = await loadIndex();

  if (index.length === 0) {
    console.log("No baselines found.");
    console.log(`Create a baseline with: baseline_manager.ts save --input results.json --name my-baseline`);
    return;
  }

  console.log("=== Available Baselines ===");
  console.log("");

  for (const baseline of index) {
    console.log(`📊 ${baseline.name}`);
    console.log(`   Created: ${baseline.created}`);
    console.log(`   Commit: ${baseline.git_commit}`);
    console.log(`   Branch: ${baseline.git_branch}`);
    console.log(`   Benchmarks: ${baseline.total_benchmarks}`);
    if (baseline.description) {
      console.log(`   Description: ${baseline.description}`);
    }
    console.log("");
  }

  console.log(`Total baselines: ${index.length}`);
}

/**
 * Show detailed info about a baseline
 */
async function showBaselineInfo(name: string): Promise<void> {
  const index = await loadIndex();
  const metadata = index.find((b) => b.name === name);

  if (!metadata) {
    console.error(`Error: Baseline "${name}" not found`);
    Deno.exit(1);
  }

  const baselinePath = getBaselinePath(name);
  const report = await loadReport(baselinePath);

  console.log("=== Baseline Information ===");
  console.log("");
  console.log(`Name: ${metadata.name}`);
  console.log(`Created: ${metadata.created}`);
  console.log(`File: ${baselinePath}`);
  console.log("");

  console.log("Git Information:");
  console.log(`  Commit: ${report.metadata.git_commit}`);
  console.log(`  Branch: ${report.metadata.git_branch}`);
  console.log(`  Timestamp: ${report.metadata.timestamp}`);
  console.log("");

  console.log("Benchmarks:");
  console.log(`  Total: ${report.benchmarks.length}`);
  console.log(`  Successful: ${report.summary.successful}`);
  console.log(`  Failed: ${report.summary.failed}`);
  console.log("");

  console.log("Benchmark List:");
  for (const bench of report.benchmarks) {
    console.log(`  - ${bench.benchmark_name} (${bench.benchmark_type}, ${bench.optimization_level})`);
    console.log(`    Mean: ${bench.statistics.mean_ms.toFixed(3)} ms`);
  }
}

/**
 * Delete a baseline
 */
async function deleteBaseline(name: string, force = false): Promise<void> {
  const index = await loadIndex();
  const metadata = index.find((b) => b.name === name);

  if (!metadata) {
    console.error(`Error: Baseline "${name}" not found`);
    Deno.exit(1);
  }

  // Confirm deletion (unless --force)
  if (!force) {
    console.log(`Are you sure you want to delete baseline "${name}"?`);
    console.log(`  Created: ${metadata.created}`);
    console.log(`  Commit: ${metadata.git_commit}`);
    console.log("");
    console.log("Use --force to skip this confirmation.");
    console.log("Delete operation cancelled.");
    return;
  }

  // Delete the baseline file
  const baselinePath = getBaselinePath(name);
  try {
    await Deno.remove(baselinePath);
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) {
      throw error;
    }
  }

  // Update index
  const newIndex = index.filter((b) => b.name !== name);
  await saveIndex(newIndex);

  console.log(`✅ Baseline "${name}" deleted successfully`);
}

/**
 * Compare current results against a baseline
 */
async function compareWithBaseline(
  baselineName: string,
  currentPath: string,
  outputFormat: "console" | "json" | "markdown" = "console",
  outputPath?: string,
): Promise<void> {
  // Load baseline
  const index = await loadIndex();
  const metadata = index.find((b) => b.name === baselineName);

  if (!metadata) {
    console.error(`Error: Baseline "${baselineName}" not found`);
    Deno.exit(1);
  }

  const baselinePath = getBaselinePath(baselineName);
  const baseline = await loadReport(baselinePath);
  const current = await loadReport(currentPath);

  // Create comparison
  const comparison = generateComparison(baseline, current);

  // Output based on format
  if (outputFormat === "json") {
    const json = JSON.stringify(comparison, null, 2);
    if (outputPath) {
      await Deno.writeTextFile(outputPath, json);
      console.log(`JSON comparison written to: ${outputPath}`);
    } else {
      console.log(json);
    }
  } else if (outputFormat === "markdown") {
    const markdown = formatComparisonMarkdown(comparison, baselineName);
    if (outputPath) {
      await Deno.writeTextFile(outputPath, markdown);
      console.log(`Markdown comparison written to: ${outputPath}`);
    } else {
      console.log(markdown);
    }
  } else {
    // Console output (default)
    const report = formatComparisonConsole(comparison, baselineName);
    console.log(report);
  }
}

/**
 * Generate comparison between baseline and current
 */
interface ComparisonResult {
  benchmark_name: string;
  baseline_mean: number;
  current_mean: number;
  speedup_ratio: number;
  percent_change: number;
  cohen_d: number;
  p_value: number;
  significant: boolean;
  performance_direction: "faster" | "slower" | "unchanged";
}

interface Comparison {
  baseline_name?: string;
  baseline_commit: string;
  current_commit: string;
  comparisons: ComparisonResult[];
  summary: {
    total: number;
    faster: number;
    slower: number;
    unchanged: number;
    significant_changes: number;
  };
}

function generateComparison(
  baseline: BenchmarkReport,
  current: BenchmarkReport,
): Comparison {
  const comparisons: ComparisonResult[] = [];

  // Create map of baseline benchmarks
  const baselineMap = new Map<string, BenchmarkResult>();
  for (const bench of baseline.benchmarks) {
    baselineMap.set(bench.benchmark_name, bench);
  }

  // Compare each current benchmark
  let faster = 0;
  let slower = 0;
  let unchanged = 0;
  let significantChanges = 0;

  for (const currentBench of current.benchmarks) {
    const baselineBench = baselineMap.get(currentBench.benchmark_name);

    if (!baselineBench) {
      continue;
    }

    const speedup = calculateSpeedup(
      baselineBench.raw_timings_ms,
      currentBench.raw_timings_ms,
    );

    const cohenD = calculateCohenD(
      baselineBench.raw_timings_ms,
      currentBench.raw_timings_ms,
    );

    const pValue = welchTTest(
      baselineBench.raw_timings_ms,
      currentBench.raw_timings_ms,
    );

    const significant = pValue < 0.05 && Math.abs(cohenD) >= 0.2;
    const percentChange = (speedup.speedupRatio - 1) * 100;

    let direction: "faster" | "slower" | "unchanged" = "unchanged";
    if (significant) {
      if (speedup.speedupRatio > 1.0) {
        direction = "faster";
        faster++;
      } else if (speedup.speedupRatio < 1.0) {
        direction = "slower";
        slower++;
      }
      significantChanges++;
    } else {
      unchanged++;
    }

    comparisons.push({
      benchmark_name: currentBench.benchmark_name,
      baseline_mean: baselineBench.statistics.mean_ms,
      current_mean: currentBench.statistics.mean_ms,
      speedup_ratio: speedup.speedupRatio,
      percent_change: percentChange,
      cohen_d: cohenD,
      p_value: pValue,
      significant,
      performance_direction: direction,
    });
  }

  return {
    baseline_commit: baseline.metadata.git_commit,
    current_commit: current.metadata.git_commit,
    comparisons,
    summary: {
      total: comparisons.length,
      faster,
      slower,
      unchanged,
      significant_changes: significantChanges,
    },
  };
}

/**
 * Format comparison for console output
 */
function formatComparisonConsole(
  comparison: Comparison,
  baselineName: string,
): string {
  const lines: string[] = [];

  lines.push("=== Baseline Comparison ===");
  lines.push("");
  lines.push(`Baseline: ${baselineName} (${comparison.baseline_commit})`);
  lines.push(`Current:  ${comparison.current_commit}`);
  lines.push("");

  lines.push("Summary:");
  lines.push(`  Total benchmarks: ${comparison.summary.total}`);
  lines.push(`  Faster: ${comparison.summary.faster}`);
  lines.push(`  Slower: ${comparison.summary.slower}`);
  lines.push(`  Unchanged: ${comparison.summary.unchanged}`);
  lines.push(`  Significant changes: ${comparison.summary.significant_changes}`);
  lines.push("");

  lines.push("Benchmark Results:");
  lines.push("");

  for (const comp of comparison.comparisons) {
    const indicator =
      comp.performance_direction === "faster"
        ? "🚀"
        : comp.performance_direction === "slower"
        ? "🔴"
        : "○";

    lines.push(`${indicator} ${comp.benchmark_name}`);
    lines.push(`  Baseline: ${comp.baseline_mean.toFixed(3)} ms`);
    lines.push(`  Current:  ${comp.current_mean.toFixed(3)} ms`);
    lines.push(`  Change:   ${comp.percent_change.toFixed(1)}%`);
    if (comp.significant) {
      lines.push(`  Significant: Yes (p=${comp.p_value.toFixed(6)}, d=${comp.cohen_d.toFixed(3)})`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Format comparison for markdown output
 */
function formatComparisonMarkdown(
  comparison: Comparison,
  baselineName: string,
): string {
  const lines: string[] = [];

  lines.push(`# Baseline Comparison: ${baselineName}`);
  lines.push("");
  lines.push(`**Baseline Commit:** \`${comparison.baseline_commit}\``);
  lines.push(`**Current Commit:** \`${comparison.current_commit}\``);
  lines.push("");

  lines.push("## Summary");
  lines.push("");
  lines.push(`- Total benchmarks: ${comparison.summary.total}`);
  lines.push(`- Faster: ${comparison.summary.faster}`);
  lines.push(`- Slower: ${comparison.summary.slower}`);
  lines.push(`- Unchanged: ${comparison.summary.unchanged}`);
  lines.push(`- Significant changes: ${comparison.summary.significant_changes}`);
  lines.push("");

  lines.push("## Benchmark Results");
  lines.push("");
  lines.push("| Benchmark | Baseline (ms) | Current (ms) | Change | Speedup | Status |");
  lines.push("|-----------|---------------|--------------|--------|---------|--------|");

  for (const comp of comparison.comparisons) {
    const indicator =
      comp.performance_direction === "faster"
        ? "🚀"
        : comp.performance_direction === "slower"
        ? "🔴"
        : "○";

    lines.push(
      `| ${comp.benchmark_name} | ${comp.baseline_mean.toFixed(3)} | ${comp.current_mean.toFixed(3)} | ${comp.percent_change.toFixed(1)}% | ${comp.speedup_ratio.toFixed(2)}x | ${indicator} |`,
    );
  }

  lines.push("");

  return lines.join("\n");
}

/**
 * Parse command line arguments
 */
function parseArgs(args: string[]): {
  command: string;
  options: Record<string, string | boolean>;
} {
  if (args.length === 0) {
    return { command: "help", options: {} };
  }

  const command = args[0];
  const options: Record<string, string | boolean> = {};

  for (let i = 1; i < args.length; i++) {
    const arg = args[i];

    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
        options[key] = args[++i];
      } else {
        options[key] = true;
      }
    }
  }

  return { command, options };
}

/**
 * Print help
 */
function printHelp() {
  console.log("Baseline Manager - Manage benchmark baselines");
  console.log("");
  console.log("Commands:");
  console.log("  save       Save benchmark results as a baseline");
  console.log("  list       List all available baselines");
  console.log("  info       Show detailed information about a baseline");
  console.log("  compare    Compare current results against a baseline");
  console.log("  delete     Delete a baseline");
  console.log("  help       Show this help message");
  console.log("");
  console.log("Usage:");
  console.log("  baseline_manager.ts save --input <file> --name <name> [--description <desc>]");
  console.log("  baseline_manager.ts list");
  console.log("  baseline_manager.ts info --name <name>");
  console.log("  baseline_manager.ts compare --baseline <name> --current <file> [--format <format>] [--output <file>]");
  console.log("  baseline_manager.ts delete --name <name> [--force]");
  console.log("");
  console.log("Examples:");
  console.log("  # Save a baseline");
  console.log("  baseline_manager.ts save --input results.json --name v1.0.0");
  console.log("");
  console.log("  # List all baselines");
  console.log("  baseline_manager.ts list");
  console.log("");
  console.log("  # Compare against baseline");
  console.log("  baseline_manager.ts compare --baseline v1.0.0 --current results.json");
  console.log("");
  console.log("  # Generate markdown comparison");
  console.log("  baseline_manager.ts compare --baseline v1.0.0 --current results.json \\");
  console.log("    --format markdown --output comparison.md");
}

/**
 * Main entry point
 */
async function main() {
  const { command, options } = parseArgs(Deno.args);

  try {
    switch (command) {
      case "save":
        if (!options.input || !options.name) {
          console.error("Error: --input and --name are required");
          printHelp();
          Deno.exit(1);
        }
        await saveBaseline(
          options.input as string,
          options.name as string,
          options.description as string | undefined,
        );
        break;

      case "list":
        await listBaselines();
        break;

      case "info":
        if (!options.name) {
          console.error("Error: --name is required");
          Deno.exit(1);
        }
        await showBaselineInfo(options.name as string);
        break;

      case "compare":
        if (!options.baseline || !options.current) {
          console.error("Error: --baseline and --current are required");
          printHelp();
          Deno.exit(1);
        }
        await compareWithBaseline(
          options.baseline as string,
          options.current as string,
          (options.format as "console" | "json" | "markdown") ?? "console",
          options.output as string | undefined,
        );
        break;

      case "delete":
        if (!options.name) {
          console.error("Error: --name is required");
          Deno.exit(1);
        }
        await deleteBaseline(options.name as string, options.force as boolean);
        break;

      case "help":
      default:
        printHelp();
        break;
    }
  } catch (error) {
    console.error("Error:", error.message);
    Deno.exit(1);
  }
}

// Run if this is the main module
if (import.meta.main) {
  main();
}
